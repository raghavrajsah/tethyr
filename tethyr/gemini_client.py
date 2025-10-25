import asyncio
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from google.genai.live import AsyncSession
from loguru import logger

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
        self.session = None
        self.model = "gemini-2.5-flash-live"
        self._system_instruction = system_instruction
        self._receive_task: asyncio.Task | None = None
        self._is_running = False
        self._callback: Callable[[dict[str, Any]], Any] | None = None
        self._resumption_token: str | None = None

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        logger.debug(f"Initialized GeminiLiveSession for client {client_id}")

    async def start(self, resume_token: str | None = None):
        if self._is_running:
            raise ValueError(f"Session already running for client {self.client_id}")

        try:
            system_instruction = self._system_instruction or DEFAULT_SYSTEM_INST

            config: types.LiveConnectConfigOrDict = {
                "response_modalities": ["TEXT"],
                "system_instruction": system_instruction,
                "tools": [{"function_declarations": [GROUNDING_TOOL_DECLARATION]}],
                **({"session_session_resumption": {"handle": resume_token}} if resume_token else {}),
            }

            self.session: AsyncSession = await self.client.aio.live.connect(
                model=self.model,
                config=config,
            ).__aenter__()
            self._is_running = True

            logger.info(f"Gemini Live session successfully started for client {self.client_id}")

        except Exception as e:
            self._is_running = False
            self.session = None
            logger.opt(exception=e).error(f"Failed to start Gemini session for client {self.client_id}")
            raise RuntimeError(f"Could not connect to Gemini Live API for client {self.client_id}") from e

    async def stop(self):
        if not self._is_running and self.session is None:
            logger.debug(f"Session already stopped for client {self.client_id}")
            return

        try:
            self._is_running = False

            if self._receive_task and not self._receive_task.done():
                logger.debug(f"Cancelling receive task for client {self.client_id}")
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=2.0)
                except (asyncio.CancelledError, TimeoutError):
                    pass

            if self.session:
                await self.session.close()
                self.session = None
                logger.info(f"Gemini session stopped for client {self.client_id}")

            self._receive_task = None
            self._callback = None

        except Exception as e:
            logger.opt(exception=e).error(f"Error stopping Gemini session for client {self.client_id}: {e}")

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_resumption_token(self) -> str | None:
        return self._resumption_token

    async def send_video_frame(self, image_base64: str):
        if not self._is_running or not self.session:
            logger.warning(f"Cannot send video frame - session not running for client {self.client_id}")
            return

        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]

            await self.session.send_realtime_input(media=types.Blob(data=image_base64, mime_type="image/jpeg"))
        except Exception as e:
            logger.opt(exception=e).error(f"Error sending video frame to Gemini for client {self.client_id}")
            self._is_running = False

    async def send_audio_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ):
        if not self._is_running or not self.session:
            logger.warning(f"Cannot send audio chunk - session not running for client {self.client_id}")
            return

        try:
            if audio_data.dtype != np.float32:
                logger.warning(f"Audio data is not float32 ({audio_data.dtype})," " conversion to PCM16 might be incorrect")

            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            await self.session.send_realtime_input(
                media=types.Blob(
                    data=audio_bytes,
                    mime_type=f"audio/pcm;rate={sample_rate}",
                )
            )
        except Exception as e:
            logger.opt(exception=e).error(f"Error sending audio chunk for client {self.client_id}")
            self._is_running = False

    # TODO: This seems to be an unused method
    async def send_text(self, text: str, turn_complete: bool = True):
        if not self.session or not self._is_running:
            logger.warning(f"Cannot send text - session not running for client {self.client_id}")
            return

        try:
            await self.session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]),
                turn_complete=turn_complete,
            )
            logger.debug(f"Sent text to Gemini for client {self.client_id}: {text[:50]}")
        except Exception as e:
            logger.opt(exception=e).error(f"Error sending text for client {self.client_id}")

    async def start_receiving(self, callback: Callable[[dict[str, Any]], Any]):
        if not self._is_running or self.session is None:
            raise ValueError(f"Cannot start receiving - session not running for client {self.client_id}")

        if self._receive_task is not None and not self._receive_task.done():
            raise ValueError(f"Receive task already running for client {self.client_id}")

        self._callback = callback
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info(f"Started receive task for client {self.client_id}")

    async def _receive_loop(self):
        try:
            async for response in self.session.receive():
                if not self._is_running:
                    logger.debug(f"Stopping receive loop for client {self.client_id}")
                    break

                logger.info(f"Received response from Gemini for client {self.client_id}: {response}")

                try:
                    await self._handle_response(response)
                except Exception as e:
                    logger.opt(exception=e).error(f"Error handling response for client {self.client_id}")

        except asyncio.CancelledError:
            logger.info(f"Receive loop cancelled for client {self.client_id}")
            raise
        except Exception as e:
            logger.opt(exception=e).error(f"Fatal error in receive loop for client {self.client_id}")
            self._is_running = False

    async def _handle_response(self, response):
        if response.server_content:
            if response.text is not None:
                if self._callback:
                    await self._callback({"type": "text", "text": response.text})

        if response.tool_call:
            await self._handle_tool_call(response.tool_call)

    async def _handle_tool_call(self, tool_call):
        if not tool_call.function_calls:
            return

        for fc in tool_call.function_calls:
            logger.info(f"Function call from Gemini for client {self.client_id}: {fc.name} with args {fc.args}")

            try:
                if fc.name == "change_detection_target":
                    result = await self._handle_change_detection_target(fc.args)

                    await self.session.send_tool_response(
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
                    await self.session.send_tool_response(
                        function_responses=types.FunctionResponse(
                            id=fc.id,
                            name=fc.name,
                            response={"error": f"Unknown function: {fc.name}"},
                        )
                    )

            except Exception as e:
                logger.opt(exception=e).error(f"Error executing function call {fc.name} for client {self.client_id}")
                await self.session.send_tool_response(
                    function_responses=types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"error": str(e)},
                    )
                )

    async def _handle_change_detection_target(self, args: dict[str, Any]) -> dict[str, Any]:
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
        system_instruction: str | None = None,
        resume_token: str | None = None,
    ) -> GeminiLiveSession:
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
                await session.start(resume_token=resume_token)
                self.sessions[client_id] = session
                logger.info(f"Created Gemini session for client {client_id}. Total active sessions: {len(self.sessions)}")
                return session

            except Exception as e:
                logger.opt(exception=e).error(f"Failed to create session for client {client_id}")
                raise

    async def get_session(self, client_id: str) -> GeminiLiveSession | None:
        return self.sessions.get(client_id)

    async def has_session(self, client_id: str) -> bool:
        return client_id in self.sessions

    async def close_session(self, client_id: str):
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
