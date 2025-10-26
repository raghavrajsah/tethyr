"""
Tethyr WebSocket server for AR smart glasses
Handles video/audio streaming and connects to Gemini Live API
"""

import asyncio
import json
import os

from loguru import logger
from websockets import ConnectionClosed
from websockets.asyncio.server import ServerConnection, serve

from .gemini_client import GeminiSessionManager
from .handlers import handle_audio_chunk, handle_handshake, handle_video_frame
from .storage import StorageMiddleware
from .types import AudioChunkMessage, ClientState, HandshakeMessage, VideoFrameMessage
from .utils import clean_json_message, parse_client_message

# Configure logging
logger.add(
    "logs/server.log",
    rotation="100 MB",
    retention="10 days",
    level="TRACE",
    encoding="utf-8",
)


# Global state
clients: dict[str, ClientState] = {}


async def handle_client(
    websocket: ServerConnection,
    gemini_manager: GeminiSessionManager,
    storage: StorageMiddleware | None = None,
):
    """
    Main handler for WebSocket client connection

    Args:
        websocket: WebSocket connection
        gemini_manager: Gemini session manager
        storage: Optional storage middleware for debug/recording
    """
    client_id = str(id(websocket))
    client_state = ClientState(websocket=websocket, client_id=client_id)
    clients[client_id] = client_state

    logger.info(f"Client {client_id} connected. Total clients: {len(clients)}")

    try:
        async for raw_message in websocket:
            logger.trace(f"Raw message from client {client_id}: {raw_message}")

            try:
                cleaned_message = clean_json_message(raw_message)

                if cleaned_message != raw_message:
                    logger.trace(
                        f"Cleaned message from client {client_id}: "
                        f"removed {len(raw_message) - len(cleaned_message)} trailing characters"
                    )

                while cleaned_message:
                    try:
                        message_dict = json.loads(cleaned_message)
                        break
                    except json.JSONDecodeError:
                        cleaned_message = clean_json_message(cleaned_message[:-1])
                else:
                    raise json.JSONDecodeError()

                message = parse_client_message(message_dict)

                if message is None:
                    continue

                # Route to appropriate handler
                match message:
                    case HandshakeMessage():
                        logger.debug(f"Handshake from client {client_id}")
                        await handle_handshake(
                            message,
                            client_state,
                            gemini_manager,
                            storage,
                        )
                    case VideoFrameMessage():
                        # logger.debug(f"Video frame from client {client_id}")
                        await handle_video_frame(
                            message,
                            client_state,
                            gemini_manager,
                            storage,
                        )
                    case AudioChunkMessage():
                        # logger.debug(f"Audio chunk from client {client_id}")
                        await handle_audio_chunk(
                            message,
                            client_state,
                            gemini_manager,
                            storage,
                        )

            except json.JSONDecodeError as e:
                logger.opt(exception=e).error(f"Invalid JSON received from client {client_id}")
            except Exception as e:
                logger.opt(exception=e).error(f"Error handling message from client {client_id}")

    except ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")

    finally:
        if client_id in clients:
            if storage and storage.enabled:
                storage.save_session(client_state)

            await gemini_manager.close_session(client_id)

            logger.info(
                f"Session complete for client {client_id}: "
                f"{client_state.frame_count} video frames, "
                f"{client_state.audio_chunk_count} audio chunks"
            )

            del clients[client_id]

        logger.info(f"Client {client_id} removed. Total clients: {len(clients)}")


async def forever():
    await asyncio.Future()


async def main(enable_storage: bool = False):
    """
    Start the Tethyr WebSocket server

    Args:
        enable_storage: Whether to enable frame/audio storage for debugging
    """

    # Initialize Gemini session manager
    gemini_manager = GeminiSessionManager()

    # Initialize storage middleware
    storage = StorageMiddleware(enabled=enable_storage)

    async with serve(
        lambda ws: handle_client(ws, gemini_manager, storage),
        "0.0.0.0",
        5001,
        max_size=10_000_000,
        ping_interval=20,
        ping_timeout=10,
    ):
        logger.info(
            "============================================================\n"
            "Tethyr AR Server with Gemini Live\n"
            "============================================================\n"
            "WebSocket Server: ws://0.0.0.0:5001\n"
            "Waiting for Spectacles to connect...\n"
            f"Storage middleware: {'ENABLED' if enable_storage else 'DISABLED'}\n"
            "============================================================"
        )
        await forever()


def run_server():
    """
    Entry point for the tethyr server command
    Reads TETHYR_ENABLE_STORAGE environment variable to enable debug storage
    """
    enable_storage = os.getenv("TETHYR_ENABLE_STORAGE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    asyncio.run(main(enable_storage=enable_storage))


if __name__ == "__main__":
    run_server()
