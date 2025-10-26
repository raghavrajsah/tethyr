"""
Message handlers for WebSocket communication
Processes incoming client messages and coordinates responses
"""

import base64
import io
from datetime import datetime, timedelta

import numpy as np
from loguru import logger
from PIL import Image

from .gemini_client import GeminiSessionManager
from .storage import StorageMiddleware
from .types import (
    AudioChunkMessage,
    BBoxMessage,
    BoundingBox,
    ClientState,
    GeminiCallback,
    HandshakeAckMessage,
    HandshakeMessage,
    OverlayMessage,
    PromptChanged,
    Text,
    VideoFrameMessage,
)
from .utils import serialize_server_message


async def handle_handshake(
    message: HandshakeMessage,
    client_state: ClientState,
    gemini_manager: GeminiSessionManager,
    storage: StorageMiddleware | None = None,
):
    """
    Handle initial handshake from client

    Args:
        message: Handshake message from client
        client_state: Current client state
        gemini_manager: Gemini session manager
        storage: Optional storage middleware
    """
    logger.info(f"Handshake from {client_state.client_id}: {message.device}")

    # Setup storage if middleware is enabled
    if storage and storage.enabled:
        client_state.output_dir = storage.setup_client_storage(client_state)

    text_buffer = []

    async def handle_gemini_response(response_data: GeminiCallback):
        nonlocal text_buffer

        try:
            match response_data:
                case Text(type="text", text=text, timestamp=timestamp):
                    now = datetime.now()
                    text_buffer.append({"text": text, "timestamp": now})

                    cutoff_time = now - timedelta(seconds=10)
                    text_buffer = [
                        entry
                        for entry in text_buffer
                        if entry["timestamp"] > cutoff_time
                    ]

                    combined_text = " ".join(entry["text"] for entry in text_buffer)

                    overlay_msg = OverlayMessage(
                        type="overlay",
                        text=combined_text,
                        timestamp=timestamp,
                    )
                    await client_state.websocket.send(
                        serialize_server_message(overlay_msg)
                    )
                case PromptChanged(type="prompt_changed", prompt=prompt):
                    logger.info(
                        f"Detection target changed for client {client_state.client_id}: "
                        f"{prompt}"
                    )

        except Exception as e:
            logger.opt(exception=e).error(
                f"Error handling Gemini response for client {client_state.client_id}"
            )

    # Create Gemini Live session for this client
    await gemini_manager.create_session(
        client_state.client_id,
        handle_gemini_response,
    )

    # Send handshake acknowledgment
    response = HandshakeAckMessage(
        type="handshake_ack",
        server="Tethyr AR Server",
        timestamp=datetime.now().isoformat(),
    )

    await client_state.websocket.send(serialize_server_message(response))


async def handle_video_frame(
    message: VideoFrameMessage,
    client_state: ClientState,
    gemini_manager: GeminiSessionManager,
    storage: StorageMiddleware | None = None,
):
    """
    Handle incoming video frame - dual pipeline:
    1. Send to Gemini for context awareness and guidance
    2. Run YOLO detection for continuous object tracking

    Args:
        message: Video frame message from client
        client_state: Current client state
        gemini_manager: Gemini session manager
        storage: Optional storage middleware
    """
    try:
        image_data = message.data
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        if storage and storage.enabled:
            storage.save_frame(client_state, image, message.frame_number)

        # === PIPELINE 1: Send to Gemini for context awareness ===
        gemini_session = await gemini_manager.get_session(client_state.client_id)
        if gemini_session:
            await gemini_session.send_video_frame(image_data)

        # === PIPELINE 2: Run YOLO detection continuously ===
        # Convert PIL image to numpy array
        frame = np.array(image)

        # Get grounding detector from session manager
        grounding_detector = gemini_manager.grounding_detector

        # Run detection (uses current prompt set by Gemini via change_detection_target)
        detections = grounding_detector.detect_in_frame(frame, conf=0.1, iou=0.5)

        # Send bounding boxes to client
        for detection in detections:
            bbox_msg = BBoxMessage(
                type="bbox",
                bbox=BoundingBox(
                    **detection["bbox"],
                    label=detection["label"],
                    confidence=detection["confidence"],
                ),
                timestamp=datetime.now().isoformat(),
            )
            await client_state.websocket.send(serialize_server_message(bbox_msg))

        # Update client state
        client_state.frame_count += 1

        # Log periodically
        if client_state.frame_count % 100 == 0:
            logger.debug(
                f"Processed {client_state.frame_count} frames for client {client_state.client_id}, "
                f"Gemini context: active, YOLO detecting: {grounding_detector.get_current_prompt()}"
            )

    except Exception as e:
        logger.opt(exception=e).error(
            f"Error handling video frame from client {client_state.client_id}"
        )


async def handle_audio_chunk(
    message: AudioChunkMessage,
    client_state: ClientState,
    gemini_manager: GeminiSessionManager,
    storage: StorageMiddleware | None = None,
):
    """
    Handle incoming audio chunk

    Args:
        message: Audio chunk message from client
        client_state: Current client state
        gemini_manager: Gemini session manager
        storage: Optional storage middleware
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(message.data)

        # Convert bytes to float32 array
        num_samples = len(audio_bytes) // 4
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32, count=num_samples)

        # Store audio metadata from first chunk
        if client_state.audio_chunk_count == 0:
            client_state.audio_sample_rate = message.sample_rate
            client_state.audio_channels = message.channels
            logger.info(
                f"Audio configuration for client {client_state.client_id}: "
                f"{message.sample_rate}Hz, {message.channels} channels"
            )

        # Buffer audio if storage is enabled
        if storage and storage.enabled:
            storage.buffer_audio(client_state, audio_data)

        # Send audio to Gemini Live API
        gemini_session = await gemini_manager.get_session(client_state.client_id)
        if gemini_session:
            await gemini_session.send_audio_chunk(
                audio_data,
                message.sample_rate,
            )

        # Update client state
        client_state.audio_chunk_count += 1

    except Exception as e:
        logger.opt(exception=e).error(
            f"Error handling audio chunk from client {client_state.client_id}"
        )
