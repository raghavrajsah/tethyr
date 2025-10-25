"""
Gemini Live API client for streaming multimodal interactions
Manages per-client sessions with audio/video streaming and tool use
"""

import asyncio
import os
from collections.abc import Callable
from typing import Any

import google.generativeai as genai
from google.generativeai import types
from loguru import logger

from .grounding import GROUNDING_TOOL_DECLARATION, GroundingDetector


class GeminiLiveSession:
    """Manages a Gemini Live API session for a single client with robust error handling"""

    def __init__(
        self,
        client_id: str,
        grounding_detector: GroundingDetector,
        system_instruction: str | None = None,
    ):
        """Initialize a Gemini Live session

        Args:
            client_id: Unique identifier for this client
            grounding_detector: Shared grounding detector instance
            system_instruction: Optional custom system instruction (uses default if None)
        """
        self.client_id = client_id
        self.grounding_detector = grounding_detector
        self.session = None
        self.model = "models/gemini-2.0-flash-exp"
        self._system_instruction = system_instruction

        # Task management
        self._receive_task: asyncio.Task | None = None
        self._is_running = False
        self._callback: Callable[[dict[str, Any]], Any] | None = None

        # Session resumption for reconnection
        self._resumption_token: str | None = None

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

        logger.debug(f"Initialized GeminiLiveSession for client {client_id}")

    def _get_default_system_instruction(self) -> str:
        """Get default system instruction for AR repair assistant

        Returns:
            System instruction string
        """
        return """You are an AI assistant for AR smart glasses helping users with home repairs.

Your role:
1. First, ask the user to point the camera at the object they want to repair
2. Use the detect_objects tool to identify what they're looking at
3. Once you identify the repair object, create a step-by-step repair plan
4. Guide the user through each step with clear, concise instructions
5. Wait for user voice confirmation before moving to the next step
6. Provide safety warnings when relevant (electrical, water, etc.)

Important:
- Keep instructions brief and actionable (displayed on AR overlay)
- Use the detect_objects tool when you need to see what the user is looking at
- Respond to voice commands like "next", "repeat", "help"
- Be proactive about safety"""

    async def start(self, resume_token: str | None = None):
        """Start the Gemini Live session with support for resumption and long sessions

        Args:
            resume_token: Optional token to resume a previous session

        Raises:
            ValueError: If session is already running
            RuntimeError: If connection to Gemini fails
        """
        if self._is_running:
            raise ValueError(
                f"Session already running for client {self.client_id}. " "Call stop() before starting a new session."
            )

        try:
            # Use custom or default system instruction
            system_instruction = self._system_instruction or self._get_default_system_instruction()

            # Configure session with long-session support
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],  # Using text for AR overlay instructions
                system_instruction=system_instruction,
            )

            # Enable context window compression for indefinite sessions
            # This prevents the 15-minute audio session limit
            config.context_window_compression = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            )

            # Enable session resumption for connection reliability
            # Resumption tokens are valid for 2 hours after session termination
            if resume_token:
                config.session_resumption = types.SessionResumptionConfig(handle=resume_token)
                logger.info(f"Resuming Gemini session for client {self.client_id} " f"with token {resume_token[:20]}...")
            else:
                config.session_resumption = types.SessionResumptionConfig()
                logger.info(f"Starting new Gemini session for client {self.client_id} " "with resumption enabled")

            # Add grounding detection tool
            config.tools = [{"function_declarations": [GROUNDING_TOOL_DECLARATION]}]

            # Connect to Gemini Live
            self.session = self.client.aio.live.connect(model=self.model, config=config)
            await self.session.__aenter__()

            self._is_running = True

            # Only send welcome message for new sessions (not resumed ones)
            if not resume_token:
                await self.session.send(
                    input={
                        "text": (
                            "Hello! I'm ready to help you with repairs. "
                            "Please point your camera at the object you'd like to repair."
                        )
                    },
                    end_of_turn=True,
                )

            logger.info(f"Gemini Live session successfully started for client {self.client_id}")

        except Exception as e:
            self._is_running = False
            self.session = None
            logger.error(f"Failed to start Gemini session for client {self.client_id}: {e}")
            raise RuntimeError(f"Could not connect to Gemini Live API for client {self.client_id}") from e

    async def stop(self):
        """Stop the Gemini Live session and cleanup resources

        This method is idempotent - safe to call multiple times
        """
        if not self._is_running and self.session is None:
            logger.debug(f"Session already stopped for client {self.client_id}")
            return

        try:
            self._is_running = False

            # Cancel receive task if running
            if self._receive_task and not self._receive_task.done():
                logger.debug(f"Cancelling receive task for client {self.client_id}")
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=2.0)
                except asyncio.CancelledError:
                    logger.debug(f"Receive task cancelled for client {self.client_id}")
                except TimeoutError:
                    logger.warning(f"Receive task did not stop cleanly for client {self.client_id}")

            # Close Gemini session
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None

                if self._resumption_token:
                    logger.info(
                        f"Gemini session stopped for client {self.client_id}. "
                        f"Resumption token available (valid for 2 hours): "
                        f"{self._resumption_token[:20]}..."
                    )
                else:
                    logger.info(f"Gemini session stopped for client {self.client_id}")

            # Cleanup state
            self._receive_task = None
            self._callback = None

        except Exception as e:
            logger.error(
                f"Error stopping Gemini session for client {self.client_id}: {e}",
                exc_info=True,
            )

    @property
    def is_running(self) -> bool:
        """Check if session is currently running

        Returns:
            True if session is active, False otherwise
        """
        return self._is_running

    def get_resumption_token(self) -> str | None:
        """Get the current resumption token for this session

        Returns:
            Resumption token string or None if not available
        """
        return self._resumption_token

    async def send_video_frame(self, image_base64: str):
        if not self._is_running or not self.session:
            logger.warning(f"Cannot send frame - session not running for client {self.client_id}")
            return

        try:
            await self.session.send(
                input={"mime_type": "image/jpeg", "data": image_base64},
                end_of_turn=False,
            )
        except Exception as e:
            logger.error(
                f"Error sending video frame to Gemini for client {self.client_id}: {e}",
                exc_info=True,
            )

    async def send_audio_chunk(self, audio_data: bytes, sample_rate: int = 16000):
        """Send audio chunk to Gemini Live API

        Args:
            audio_data: Raw audio bytes (PCM 16-bit, mono)
            sample_rate: Sample rate in Hz (typically 16000)

        Note:
            Audio should be 16-bit PCM format, mono channel
        """
        if not self.session or not self._is_running:
            logger.warning(f"Cannot send audio - session not running for client {self.client_id}")
            return

        try:
            # Create audio blob in format expected by Gemini (16-bit PCM, mono)
            audio_blob = types.Blob(mime_type="audio/pcm", data=audio_data)

            # Send to Gemini without ending turn (streaming audio)
            await self.session.send(input=audio_blob, end_of_turn=False)

        except Exception as e:
            logger.error(
                f"Error sending audio chunk for client {self.client_id}: {e}",
                exc_info=True,
            )

    async def send_text(self, text: str, end_of_turn: bool = True):
        """Send text message to Gemini

        Args:
            text: Text message (e.g., transcribed voice command or user instruction)
            end_of_turn: Whether to end the turn (default True)
        """
        if not self.session or not self._is_running:
            logger.warning(f"Cannot send text - session not running for client {self.client_id}")
            return

        try:
            await self.session.send(input={"text": text}, end_of_turn=end_of_turn)
            logger.debug(f"Sent text to Gemini for client {self.client_id}: {text[:50]}")

        except Exception as e:
            logger.error(f"Error sending text for client {self.client_id}: {e}", exc_info=True)

    async def start_receiving(self, callback: Callable[[dict[str, Any]], Any]):
        """Start receiving responses from Gemini in a background task

        Args:
            callback: Async function to call with response data

        Raises:
            ValueError: If session is not running or receive task already started
        """
        if not self._is_running or self.session is None:
            raise ValueError(f"Cannot start receiving - session not running for client {self.client_id}")

        if self._receive_task is not None and not self._receive_task.done():
            raise ValueError(f"Receive task already running for client {self.client_id}")

        self._callback = callback
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info(f"Started receive task for client {self.client_id}")

    async def _receive_loop(self):
        """Internal loop to receive responses from Gemini

        This runs in a background task and processes all incoming messages
        """
        try:
            async for response in self.session.receive():
                if not self._is_running:
                    logger.debug(f"Stopping receive loop for client {self.client_id}")
                    break

                try:
                    await self._handle_response(response)
                except Exception as e:
                    logger.error(
                        f"Error handling response for client {self.client_id}: {e}",
                        exc_info=True,
                    )
                    # Continue processing other messages

        except asyncio.CancelledError:
            logger.info(f"Receive loop cancelled for client {self.client_id}")
            raise
        except Exception as e:
            logger.error(
                f"Fatal error in receive loop for client {self.client_id}: {e}",
                exc_info=True,
            )
            # Mark session as not running on fatal error
            self._is_running = False

    async def _handle_response(self, response):
        """Handle a single response from Gemini

        Args:
            response: Response object from Gemini Live API
        """
        # Handle session resumption updates
        if hasattr(response, "session_resumption_update") and response.session_resumption_update:
            resumption_update = response.session_resumption_update
            if resumption_update.resumable and hasattr(resumption_update, "new_handle"):
                self._resumption_token = resumption_update.new_handle
                logger.info(
                    f"Received resumption token for client {self.client_id}: "
                    f"{self._resumption_token[:20]}... (valid for 2 hours)"
                )

        # Handle server content (text responses and function calls)
        if response.server_content:
            if response.server_content.model_turn:
                parts = response.server_content.model_turn.parts

                for part in parts:
                    # Handle text response
                    if hasattr(part, "text") and part.text:
                        if self._callback:
                            await self._callback({"type": "text", "text": part.text})

                    # Handle function call (tool use)
                    elif hasattr(part, "function_call") and part.function_call:
                        await self._handle_function_call(part.function_call)

                # Check if turn is complete
                if response.server_content.turn_complete:
                    logger.debug(f"Turn complete for client {self.client_id}")

    async def _handle_function_call(self, function_call):
        """Handle a function call from Gemini

        Args:
            function_call: FunctionCall object from Gemini
        """
        fc = function_call
        logger.info(f"Function call from Gemini for client {self.client_id}: " f"{fc.name} with args {fc.args}")

        try:
            # Execute tool
            if fc.name == "change_detection_target":
                result = await self._handle_change_detection_target(fc.args)

                # Send tool response back to Gemini
                function_response = types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response=result,
                )

                await self.session.send(
                    tool_response=function_response,
                    end_of_turn=True,
                )

                # Notify callback about prompt change
                if self._callback:
                    await self._callback({"type": "prompt_changed", "prompt": result["prompt"]})
            else:
                logger.error(f"Unknown function call from Gemini for client {self.client_id}: {fc.name}")
                # Send error response back to Gemini
                function_response = types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"error": f"Unknown function: {fc.name}"},
                )
                await self.session.send(
                    tool_response=function_response,
                    end_of_turn=True,
                )

        except Exception as e:
            logger.error(
                f"Error executing function call {fc.name} for client {self.client_id}: {e}",
                exc_info=True,
            )
            # Send error response back to Gemini
            function_response = types.FunctionResponse(
                id=fc.id,
                name=fc.name,
                response={"error": str(e)},
            )
            await self.session.send(
                tool_response=function_response,
                end_of_turn=True,
            )

    async def _handle_change_detection_target(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle change_detection_target tool call

        This ONLY changes what YOLO is looking for. Detection runs continuously.

        Args:
            args: Tool arguments from Gemini containing the new prompt

        Returns:
            Dict with status and prompt
        """
        try:
            prompt = args.get("prompt", "person, object")

            logger.info(f"Gemini requested detection target change for client {self.client_id}: " f"new prompt='{prompt}'")

            # Update prompt (runs in thread pool to avoid blocking)
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
            logger.error(
                f"Error changing detection target for client {self.client_id}: {e}",
                exc_info=True,
            )
            return {
                "status": "error",
                "prompt": self.grounding_detector.get_current_prompt(),
                "message": str(e),
            }


class GeminiSessionManager:
    """Manages multiple Gemini Live sessions for different clients

    This manager handles:
    - Creating and tracking sessions per client
    - Sharing a single grounding detector across all sessions
    - Clean shutdown of all sessions
    """

    def __init__(self, grounding_detector: GroundingDetector | None = None):
        """Initialize the session manager

        Args:
            grounding_detector: Optional shared grounding detector instance.
                              If None, creates a new one.
        """
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
        """Create a new Gemini Live session for a client

        Args:
            client_id: Unique client identifier
            system_instruction: Optional custom system instruction
            resume_token: Optional token to resume a previous session

        Returns:
            GeminiLiveSession instance

        Raises:
            ValueError: If session already exists for this client
        """
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

                logger.info(f"Created Gemini session for client {client_id}. " f"Total active sessions: {len(self.sessions)}")
                return session

            except Exception as e:
                logger.error(f"Failed to create session for client {client_id}: {e}")
                raise

    async def get_session(self, client_id: str) -> GeminiLiveSession | None:
        """Get existing session for a client

        Args:
            client_id: Unique client identifier

        Returns:
            GeminiLiveSession instance or None if not found
        """
        return self.sessions.get(client_id)

    async def has_session(self, client_id: str) -> bool:
        """Check if a session exists for a client

        Args:
            client_id: Unique client identifier

        Returns:
            True if session exists, False otherwise
        """
        return client_id in self.sessions

    async def close_session(self, client_id: str):
        """Close a Gemini Live session

        Args:
            client_id: Unique client identifier

        Note:
            This method is idempotent - safe to call even if session doesn't exist
        """
        async with self._lock:
            if client_id not in self.sessions:
                logger.debug(f"No session to close for client {client_id}")
                return

            session = self.sessions[client_id]

            try:
                await session.stop()
            finally:
                # Always remove from tracking even if stop failed
                del self.sessions[client_id]
                logger.info(f"Closed Gemini session for client {client_id}. " f"Remaining sessions: {len(self.sessions)}")

    async def close_all_sessions(self):
        """Close all active sessions

        Useful for graceful server shutdown
        """
        logger.info(f"Closing all {len(self.sessions)} active sessions...")

        # Create a list to avoid modifying dict during iteration
        client_ids = list(self.sessions.keys())

        for client_id in client_ids:
            try:
                await self.close_session(client_id)
            except Exception as e:
                logger.error(f"Error closing session for client {client_id}: {e}")

        logger.info("All Gemini sessions closed")

    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions

        Returns:
            Number of active sessions
        """
        return len(self.sessions)
