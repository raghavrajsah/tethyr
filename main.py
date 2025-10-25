import asyncio
import base64
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from loguru import logger
from PIL import Image
from websockets import ConnectionClosed
from websockets.asyncio.server import ServerConnection, serve

logger.add(
    "logs/server.log",
    rotation="100 MB",
    retention="10 days",
    level="TRACE",
    encoding="utf-8",
)


@dataclass
class Resolution:
    width: int
    height: int


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float


@dataclass
class Position:
    x: float
    y: float


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    label: str
    confidence: float | None = None


@dataclass
class HandshakeMessage:
    type: Literal["handshake"]
    timestamp: int
    device: str
    capabilities: dict[str, bool] | None = None


@dataclass
class VideoFrameMessage:
    type: Literal["video_frame"]
    data: str  # base64 encoded image
    timestamp: int
    frame_number: int
    resolution: Resolution


@dataclass
class AudioChunkMessage:
    type: Literal["audio_chunk"]
    data: str  # base64 encoded audio
    timestamp: int
    frame_number: int
    sample_rate: int
    samples: int
    channels: int


ClientMessage = HandshakeMessage | VideoFrameMessage | AudioChunkMessage


@dataclass
class HandshakeAckMessage:
    type: Literal["handshake_ack"]
    server: str
    timestamp: str


@dataclass
class OverlayMessage:
    type: Literal["overlay"]
    text: str
    timestamp: str
    color: Color | None = None
    position: Position | None = None


@dataclass
class BBoxMessage:
    type: Literal["bbox"]
    bbox: BoundingBox
    timestamp: str
    color: Color | None = None


@dataclass
class ClearMessage:
    type: Literal["clear"]
    timestamp: str


ServerMessage = HandshakeAckMessage | OverlayMessage | BBoxMessage | ClearMessage


@dataclass
class ClientState:
    websocket: ServerConnection
    client_id: str
    frame_count: int = 0
    audio_buffer: list[np.ndarray] = None
    last_detection: dict | None = None

    def __post_init__(self):
        if self.audio_buffer is None:
            self.audio_buffer = []


clients: dict[str, ClientState] = {}


def process_video_frame(image: Image.Image, frame_info: dict) -> ServerMessage | None:
    """
    PLACEHOLDER: Process video frame and return instruction

    Args:
        image: PIL Image object
        frame_info: Dictionary with frame metadata (frame_number, resolution, timestamp)

    Returns:
        ServerMessage or None

    TODO: Replace with your actual video processing logic:
    - Object detection (YOLO etc.)
    - Scene understanding
    - Visual question answering
    - etc.
    """
    # Example: Simple red object detection
    width, height = image.size
    img_array = np.array(image.convert("RGB"))

    red_mask = (img_array[:, :, 0] > 150) & (img_array[:, :, 1] < 100) & (img_array[:, :, 2] < 100)
    red_coords = np.argwhere(red_mask)

    if len(red_coords) > 50:
        y_coords, x_coords = red_coords[:, 0], red_coords[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        return BBoxMessage(
            type="bbox",
            bbox=BoundingBox(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
                label="Red Object",
                confidence=0.95,
            ),
            color=Color(r=1.0, g=0.0, b=0.0, a=0.8),
            timestamp=datetime.now().isoformat(),
        )

    return None


def process_audio_chunk(
    audio_data: np.ndarray,
    audio_info: dict,
) -> ServerMessage | None:
    """
    PLACEHOLDER: Process audio chunk and return instruction

    Args:
        audio_data: numpy array of audio samples (float32)
        audio_info: Dictionary with audio metadata (sample_rate, samples, channels, timestamp)

    Returns:
        ServerMessage or None

    TODO: Replace with your actual audio processing logic:
    - Speech-to-text (Whisper, Google Speech API, etc.)
    - Voice command detection
    - Audio event detection
    - Speaker identification
    - etc.
    """
    # Example: Simple volume-based detection
    volume = np.abs(audio_data).mean()

    if volume > 0.1:  # Arbitrary threshold
        return OverlayMessage(
            type="overlay",
            text=f"🎤 Audio detected (vol: {volume:.2f})",
            color=Color(r=0.0, g=1.0, b=1.0, a=1.0),
            timestamp=datetime.now().isoformat(),
        )

    return None


def process_multimodal(
    video_frame: Image.Image | None,
    audio_chunk: np.ndarray | None,
    client_state: ClientState,
) -> list[ServerMessage]:
    """
    PLACEHOLDER: Process both video and audio together for contextual understanding

    Args:
        video_frame: PIL Image or None
        audio_chunk: numpy array or None
        client_state: Current state of the client

    Returns:
        List of ServerMessage instructions

    TODO: Replace with your actual multimodal AI logic:
    - Vision + Language models
    - Context-aware assistance
    - Conversational AI with visual grounding
    - etc.
    """
    instructions = []

    # Example: Periodic status update
    if client_state.frame_count % 100 == 0:
        instructions.append(
            OverlayMessage(
                type="overlay",
                text=f"Frame {client_state.frame_count}",
                color=Color(r=1.0, g=1.0, b=1.0, a=0.7),
                timestamp=datetime.now().isoformat(),
            )
        )

    return instructions


# ============================================================================
# MESSAGE PARSING
# ============================================================================


def clean_json_message(raw_message: str) -> str:
    """
    Clean JSON message by stripping any trailing garbage after the last '}'.
    This method is necessary since, for UNKNOWN reason, Spectacle's websocket
    always send small bits of garbage after the last '}'.

    Args:
        raw_message: Raw message string that may have trailing characters

    Returns:
        Cleaned message string ending at the last '}'
    """
    if not raw_message:
        return raw_message
    last_brace_idx = raw_message.rfind("}")
    if last_brace_idx == -1:
        return raw_message
    return raw_message[: last_brace_idx + 1]


def parse_client_message(message_dict: dict) -> ClientMessage | None:
    try:
        match message_dict.get("type"):
            case "handshake":
                return HandshakeMessage(**message_dict)
            case "video_frame":
                message_dict["resolution"] = Resolution(**message_dict["resolution"])
                return VideoFrameMessage(**message_dict)
            case "audio_chunk":
                return AudioChunkMessage(**message_dict)
            case _:
                print(f"Unknown message type: {message_dict.get('type')}")
                return None
    except Exception as e:
        print(f"Error parsing message: {e}")
        return None


def serialize_server_message(message: ServerMessage) -> str:
    """Convert server message dataclass to JSON string"""
    message_dict = asdict(message)
    return json.dumps(message_dict)


# ============================================================================
# MESSAGE HANDLERS
# ============================================================================


async def handle_handshake(message: HandshakeMessage, client_state: ClientState):
    """Handle initial handshake from client"""
    print(f"Handshake from {client_state.client_id}: {message.device}")

    response = HandshakeAckMessage(
        type="handshake_ack",
        server="AR Processing Server",
        timestamp=datetime.now().isoformat(),
    )

    await client_state.websocket.send(serialize_server_message(response))


async def handle_video_frame(message: VideoFrameMessage, client_state: ClientState):
    """Handle incoming video frame"""
    try:
        # Decode base64 image
        image_data = message.data
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        # Process frame
        frame_info = {
            "frame_number": message.frame_number,
            "resolution": message.resolution,
            "timestamp": message.timestamp,
        }

        # PLACEHOLDER: Call your video processing function
        instruction = process_video_frame(image, frame_info)

        if instruction:
            await client_state.websocket.send(serialize_server_message(instruction))

        # Update client state
        client_state.frame_count += 1

    except Exception as e:
        print(f"Error handling video frame: {e}")


async def handle_audio_chunk(message: AudioChunkMessage, client_state: ClientState):
    """Handle incoming audio chunk"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(message.data)

        # Convert bytes to float32 array
        num_samples = len(audio_bytes) // 4
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32, count=num_samples)

        # Process audio
        audio_info = {
            "sample_rate": message.sample_rate,
            "samples": message.samples,
            "channels": message.channels,
            "timestamp": message.timestamp,
        }

        # PLACEHOLDER: Call your audio processing function
        instruction = process_audio_chunk(audio_data, audio_info)

        if instruction:
            await client_state.websocket.send(serialize_server_message(instruction))

        # Optionally buffer audio for longer-context processing
        client_state.audio_buffer.append(audio_data)
        if len(client_state.audio_buffer) > 10:  # Keep last 10 chunks
            client_state.audio_buffer.pop(0)

    except Exception as e:
        print(f"Error handling audio chunk: {e}")


# ============================================================================
# WEBSOCKET CONNECTION HANDLER
# ============================================================================


async def handle_client(websocket: ServerConnection):
    """Main handler for WebSocket client connection"""
    client_id = id(websocket)
    client_state = ClientState(websocket=websocket, client_id=str(client_id))
    clients[str(client_id)] = client_state

    logger.info(f"Client {client_id} connected. Total clients: {len(clients)}")

    try:
        async for raw_message in websocket:
            logger.trace(
                "Raw message: {raw_message} from client {client_id}",
                raw_message=raw_message,
                client_id=client_id,
            )
            try:
                # Clean the message to remove any trailing garbage
                cleaned_message = clean_json_message(raw_message)

                # Log if we had to clean the message
                if cleaned_message != raw_message:
                    logger.trace(
                        f"Cleaned message from client {client_id}: "
                        f"removed {len(raw_message) - len(cleaned_message)} trailing characters"
                    )

                # Parse JSON
                message_dict = json.loads(cleaned_message)
                message = parse_client_message(message_dict)

                if message is None:
                    continue

                # Route to appropriate handler
                match message:
                    case HandshakeMessage():
                        logger.debug(f"Handshake from client {client_id}")
                        await handle_handshake(message, client_state)
                    case VideoFrameMessage():
                        logger.debug(f"Video frame from client {client_id}")
                        await handle_video_frame(message, client_state)
                    case AudioChunkMessage():
                        logger.debug(f"Audio chunk from client {client_id}")
                        await handle_audio_chunk(message, client_state)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id} with message {raw_message}")
            except Exception as e:
                logger.error(f"Error handling message from client {client_id} with message {raw_message}: {e}")
    except ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    finally:
        if str(client_id) in clients:
            del clients[str(client_id)]
        logger.info(f"Client {client_id} removed. Total clients: {len(clients)}")


# ============================================================================
# MAIN SERVER
# ============================================================================


async def forever():
    await asyncio.Future()


async def main():
    print("=" * 60)
    print("AR Processing Server")
    print("=" * 60)
    print("WebSocket Server: ws://0.0.0.0:5000")
    print("Waiting for Spectacles to connect...")
    print("=" * 60)

    async with serve(
        handle_client,
        "0.0.0.0",
        5000,
        max_size=10_000_000,
        ping_interval=20,
        ping_timeout=10,
    ):
        await forever()


if __name__ == "__main__":
    asyncio.run(main())
