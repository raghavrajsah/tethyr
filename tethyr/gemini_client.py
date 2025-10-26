import asyncio
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from google.genai.live import AsyncSession
from loguru import logger

# Assuming grounding.py exists in the same directory
from .grounding import GROUNDING_TOOL_DECLARATION, GroundingDetector

DEFAULT_SYSTEM_INST = """You are an AI assistant for AR smart glasses helping users with construction tasks

Your role:
1. Ask the user to point the camera at the object(s) that they want to work on, as well as describe
what they want to construct/repair
2. Use change_detection_target tool to change what objects YOLO should detect
3. Once you identify the object, create a step-by-step plan
4. Guide the user through each step with clear, concise instructions
5. Wait for user voice confirmation before moving to the next step
6. Provide safety warnings when relevant (electrical, construction, etc.)

Important:
- Keep instructions brief and actionable (displayed on AR overlay)
- Use change_detection_target tool to focus detection on relevant repair objects
- Respond to voice commands like "next", "repeat", "help". Feel free to interrupt the user if the user never stopped talking.
- Be proactive about safety"""


class GeminiLiveSession:
    """Manages a Gemini Live API session for a single client with robust error handling"""

    def __init__(
        self,
        client_id: str,
        grounding_detector: GroundingDetector,
        system_instruction: str | None = None,
    ):
        self.client_id = client_id
        self.grounding_detector = grounding_detector
        self.model = "gemini-live-2.5-flash-preview"
        self._system_instruction = system_instruction
        self._callback: Callable[[dict[str, Any]], Any] | None = None
        self._resumption_token: str | None = None

        self._send_queue = asyncio.Queue()
        self._main_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._started_event = asyncio.Event()
        self._start_exception: Exception | None = None
        self._is_running = False

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        logger.debug(f"Initialized GeminiLiveSession for client {client_id}")

    async def _run(self, resume_token: str | None = None):
        """Internal task to manage the session lifecycle with an async context."""
        if not self._callback:
            logger.error(f"Callback not set for {self.client_id}. Stopping run.")
            self._start_exception = ValueError("Callback not provided")
            self._started_event.set()
            return

        try:
            system_instruction = self._system_instruction or DEFAULT_SYSTEM_INST

            config: types.LiveConnectConfigOrDict = {
                "response_modalities": [types.Modality.TEXT],
                "system_instruction": system_instruction,
                "tools": [{"function_declarations": [GROUNDING_TOOL_DECLARATION]}],
                **({"session_session_resumption": {"handle": resume_token}} if resume_token else {}),
            }

            async with self.client.aio.live.connect(
                model=self.model,
                config=config,
            ) as session:
                self._is_running = True
                self._started_event.set()
                logger.info(f"Gemini Live session successfully started for client {self.client_id}")

                # Start concurrent send and receive loops within the session context
                receive_task = asyncio.create_task(
                    self._receive_loop(session),
                    name=f"recv_{self.client_id}",
                )
                send_task = asyncio.create_task(
                    self._send_loop(session),
                    name=f"send_{self.client_id}",
                )
                stop_wait_task = asyncio.create_task(
                    self._stop_event.wait(),
                    name=f"stop_{self.client_id}",
                )

                tasks = [receive_task, send_task, stop_wait_task]

                # Wait for the first task to complete for any reason
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.ALL_COMPLETED,
                )

                # Check for exceptions in the task(s) that completed
                logger.info("Task finished!")
                for task in done:
                    if not task.cancelled():
                        exc = task.exception()
                        if exc:
                            logger.error(
                                f"Task {task.get_name()} failed for {self.client_id}.",
                                exception=exc,
                            )

                if stop_wait_task not in done:
                    # A worker task failed or finished. Trigger a full stop.
                    logger.warning(f"A worker task for {self.client_id} finished unexpectedly. Stopping session.")
                    self._stop_event.set()

                # Cancel all pending tasks to ensure a clean shutdown
                for task in pending:
                    task.cancel()

                # Await the cancellation of pending tasks
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

                logger.debug(f"Session task cleanup complete for {self.client_id}.")

        except Exception as e:
            self._start_exception = e
            logger.opt(exception=e).error(f"Failed to start/run Gemini session for client {self.client_id}")
        finally:
            self._is_running = False
            self._main_task = None
            self._callback = None
            self._started_event.set()  # Ensure start() unblocks on failure
            self._stop_event.set()  # Ensure any other waiters are unblocked
            logger.info(f"Gemini session fully stopped for {self.client_id}")

    async def start(
        self,
        callback: Callable[[dict[str, Any]], Any],
        resume_token: str | None = None,
    ):
        """Starts the long-running session task."""
        if self._is_running:
            raise ValueError(f"Session already running for client {self.client_id}")

        self._start_exception = None
        self._started_event.clear()
        self._stop_event.clear()
        self._callback = callback
        self._resumption_token = resume_token  # Store for potential re-use

        self._main_task = asyncio.create_task(self._run(resume_token=resume_token))

        try:
            # Wait for the _run method to successfully enter the context
            await asyncio.wait_for(self._started_event.wait(), timeout=10.0)
        except TimeoutError:
            await self.stop()  # Clean up the failed task

        if self._start_exception:
            raise RuntimeError(f"Failed to start Gemini session for {self.client_id}") from self._start_exception

        if not self._is_running:
            raise RuntimeError(f"Session for {self.client_id} failed to start and is not running.")

    async def stop(self):
        """Signals the session to stop and cleans up resources."""
        if not self._stop_event:
            logger.debug(f"Session already stopped for client {self.client_id}")
            return

        try:
            self._stop_event.set()
            if self._main_task:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            logger.info(f"Gemini session stopped for client {self.client_id}")
        except (asyncio.CancelledError, TimeoutError):
            logger.warning(f"Timeout or cancellation waiting for session cleanup for client {self.client_id}")
        finally:
            self._is_running = False
            self._main_task = None
            # Clear queue
            while not self._send_queue.empty():
                self._send_queue.get_nowait()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_resumption_token(self) -> str | None:
        return self._resumption_token

    async def send_video_frame(self, image_base64: str):
        """Queues a video frame to be sent to the session."""
        if not self._is_running:
            logger.warning(f"Cannot send video frame - session not running for client {self.client_id}")
            return

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        logger.debug("Actually put something in the video queue")
        await self._send_queue.put(("video", image_base64))

    async def send_audio_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ):
        """Queues an audio chunk to be sent to the session."""
        if not self._is_running:
            logger.warning(f"Cannot send audio chunk - session not running for client {self.client_id}")
            return

        try:
            if audio_data.dtype != np.float32:
                logger.warning(f"Audio data is not float32 ({audio_data.dtype})," " conversion to PCM16 might be incorrect")

            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            await self._send_queue.put(("audio", (audio_bytes, sample_rate)))
        except Exception as e:
            logger.opt(exception=e).error(f"Error processing audio chunk for client {self.client_id}")

    async def send_text(self, text: str, turn_complete: bool = True):
        """Queues a text message to be sent to the session."""
        if not self._is_running:
            logger.warning(f"Cannot send text - session not running for client {self.client_id}")
            return

        await self._send_queue.put(("text", (text, turn_complete)))

    async def _send_loop(self, session: AsyncSession):
        """Consumes the send queue and sends data to the active session."""
        try:
            logger.info("Send worker is up!")
            while True:
                item_type, data = await self._send_queue.get()

                if not self._is_running:
                    self._send_queue.task_done()
                    break

                try:
                    if item_type == "video":
                        blob = types.Blob(data=data, mime_type="image/jpeg")
                        logger.info("Sent videoy!")
                        await session.send_realtime_input(media=blob)

                    elif item_type == "audio":
                        audio_bytes, sample_rate = data
                        blob = types.Blob(
                            data=audio_bytes,
                            mime_type=f"audio/pcm;rate={sample_rate}",
                        )
                        logger.info("Sent audioy!")
                        await session.send_realtime_input(media=blob)

                    elif item_type == "text":
                        text, turn_complete = data
                        await session.send_client_content(
                            turns=types.Content(role="user", parts=[types.Part(text=text)]),
                            turn_complete=turn_complete,
                        )
                        logger.debug(f"Sent text to Gemini for client {self.client_id}: {text[:50]}")

                except Exception as e:
                    logger.opt(exception=e).error(f"Error sending {item_type} data for client {self.client_id}")
                    self._stop_event.set()  # Trigger a session stop

                self._send_queue.task_done()

        except asyncio.CancelledError:
            logger.debug(f"Send loop cancelled for {self.client_id}")
            raise
        except Exception as e:
            logger.opt(exception=e).error(f"Fatal error in send loop for client {self.client_id}")
            self._stop_event.set()

    async def _receive_loop(self, session: AsyncSession):
        """Receives responses from the active session."""
        try:
            while True:
                async for response in session.receive():
                    if not self._is_running:
                        logger.debug(f"Stopping receive loop for client {self.client_id}")
                        break

                    logger.info(f"Received response from Gemini for client {self.client_id}: {response}")

                    try:
                        await self._handle_response(response, session)
                    except Exception as e:
                        logger.opt(exception=e).error(f"Error handling response for client {self.client_id}")
        except asyncio.CancelledError:
            logger.info(f"Receive loop cancelled for client {self.client_id}")
            raise
        except Exception as e:
            logger.opt(exception=e).error(f"Fatal error in receive loop for client {self.client_id}")

    async def _handle_response(self, response, session: AsyncSession):
        """Handles text and tool calls from a session response."""
        if response.server_content:
            if response.text is not None:
                if self._callback:
                    await self._callback({"type": "text", "text": response.text})

        if response.tool_call:
            await self._handle_tool_call(response.tool_call, session)

    async def _handle_tool_call(self, tool_call, session: AsyncSession):
        """Executes tool calls and sends responses back to the active session."""
        if not tool_call.function_calls:
            return

        for fc in tool_call.function_calls:
            logger.info(f"Function call from Gemini for client {self.client_id}: {fc.name} with args {fc.args}")

            try:
                if fc.name == "change_detection_target":
                    result = await self._handle_change_detection_target(fc.args)

                    await session.send_tool_response(
                        function_responses=types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response=result,
                        )
                    )

                    if self._callback:
                        await self._callback(
                            {
                                "type": "prompt_changed",
                                "prompt": result.get("prompt", ""),
                            }
                        )
                else:
                    logger.error(f"Unknown function call from Gemini for client {self.client_id}: {fc.name}")
                    await session.send_tool_response(
                        function_responses=types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response={"error": f"Unknown function: {fc.name}"},
                        )
                    )

            except Exception as e:
                logger.opt(exception=e).error(f"Error executing function call {fc.name} for client {self.client_id}")
                try:
                    await session.send_tool_response(
                        function_responses=types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response={"error": str(e)},
                        )
                    )
                except Exception as se:
                    logger.opt(exception=se).error(f"Failed to send tool error response for {self.client_id}")
                    self._stop_event.set()  # Fatal error

    async def _handle_change_detection_target(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handles the 'change_detection_target' tool call logic."""
        try:
            prompt = args.get("prompt", "person, object")
            logger.info(f"Gemini requested detection target change for client {self.client_id}: new prompt='{prompt}'")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.grounding_detector.update_prompt,
                prompt,
            )

            return {
                "status": "success",
                "prompt": prompt,
                "message": f"Now detecting: {prompt}",
            }

        except Exception as e:
            logger.opt(exception=e).error(f"Error changing detection target for client {self.client_id}")
            return {
                "status": "error",
                "prompt": self.grounding_detector.get_current_prompt(),
                "message": str(e),
            }


class GeminiSessionManager:
    """Manages multiple Gemini Live sessions for different clients"""

    def __init__(self, grounding_detector: GroundingDetector | None = None):
        self.sessions: dict[str, GeminiLiveSession] = {}
        self.grounding_detector = grounding_detector or GroundingDetector()
        self._lock = asyncio.Lock()
        logger.info("Initialized GeminiSessionManager with shared grounding detector")

    async def create_session(
        self,
        client_id: str,
        callback: Callable[[dict[str, Any]], Any],
        system_instruction: str | None = None,
        resume_token: str | None = None,
    ) -> GeminiLiveSession:
        """Creates, starts, and stores a new GeminiLiveSession."""
        async with self._lock:
            if client_id in self.sessions:
                logger.warning(f"Session already exists for client {client_id}, returning existing session")
                return self.sessions[client_id]

            session = GeminiLiveSession(
                client_id=client_id,
                grounding_detector=self.grounding_detector,
                system_instruction=system_instruction,
            )

            try:
                await session.start(callback=callback, resume_token=resume_token)
                self.sessions[client_id] = session
                logger.info(f"Created Gemini session for client {client_id}. Total active sessions: {len(self.sessions)}")
                return session
            except Exception as e:
                logger.opt(exception=e).error(f"Failed to create session for client {client_id}")
                # Ensure session is cleaned up if start fails
                await session.stop()
                raise

    async def get_session(self, client_id: str) -> GeminiLiveSession | None:
        result = self.sessions.get(client_id)
        if not result:
            logger.warning(f"Client {client_id} has no associated session!")
        return result

    async def has_session(self, client_id: str) -> bool:
        return client_id in self.sessions

    async def close_session(self, client_id: str):
        """Stops and removes a session by client ID."""
        async with self._lock:
            if client_id not in self.sessions:
                logger.debug(f"No session to close for client {client_id}")
                return

            session = self.sessions[client_id]
            try:
                await session.stop()
            finally:
                del self.sessions[client_id]
                logger.info(f"Closed Gemini session for client {client_id}. Remaining sessions: {len(self.sessions)}")

    async def close_all_sessions(self):
        logger.info(f"Closing all {len(self.sessions)} active sessions...")
        client_ids = list(self.sessions.keys())

        for client_id in client_ids:
            try:
                await self.close_session(client_id)
            except Exception as e:
                logger.opt(exception=e).error(f"Error closing session for client {client_id}")

        logger.info("All Gemini sessions closed")

    @property
    def active_session_count(self) -> int:
        return len(self.sessions)
