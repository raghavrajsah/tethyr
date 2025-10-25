import asyncio
import base64
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io.wavfile as wavfile
from loguru import logger
from PIL import Image
from websockets import ConnectionClosed
from websockets.asyncio.server import ServerConnection, serve

import load_env  # noqa: E402, F401
from agent_context import context_manager
from ai_client import stream_to_gemini_live_sync

logger.add(
    "logs/server.log",
    rotation="100 MB",
    retention="10 days",
    level="TRACE",
    encoding="utf-8",
)


# ============================================================================
# REPAIR PLANS AND SAFETY WARNINGS
# ============================================================================

# Predefined repair plans for different objects
REPAIR_PLANS = {
    "light fixture": [
        "Turn off power at the circuit breaker",
        "Remove the old light fixture cover",
        "Disconnect the wiring from the old fixture",
        "Connect wiring to the new fixture (match wire colors)",
        "Secure the new fixture to the ceiling box",
        "Attach the fixture cover and restore power",
    ],
    "faucet": [
        "Turn off water supply valves under the sink",
        "Remove faucet handle by unscrewing the set screw",
        "Unscrew and remove the old cartridge or valve",
        "Install the new cartridge (ensure proper alignment)",
        "Reattach the faucet handle",
        "Turn on water supply and test for leaks",
    ],
    "door hinge": [
        "Open the door and support it with a wedge",
        "Remove the hinge pin using a hammer and nail punch",
        "Unscrew the old hinge from the door",
        "Align and screw the new hinge to the door",
        "Reattach the door and insert the hinge pin",
        "Test door swing and adjust if needed",
    ],
    "outlet": [
        "Turn off power at the circuit breaker",
        "Remove the outlet cover plate",
        "Unscrew the outlet from the electrical box",
        "Disconnect wires from the old outlet (note positions)",
        "Connect wires to the new outlet (match positions)",
        "Screw outlet into box and replace cover plate",
    ],
}

# Safety warnings for different repair types
SAFETY_WARNINGS = {
    "light fixture": "⚠️ SAFETY: Ensure power is OFF at circuit breaker before starting!",
    "faucet": "⚠️ SAFETY: Turn off water supply before starting to avoid flooding!",
    "door hinge": "⚠️ SAFETY: Support the door to prevent it from falling!",
    "outlet": "⚠️ SAFETY: Ensure power is OFF at circuit breaker before starting!",
}


# ============================================================================
# DATA CLASSES
# ============================================================================


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
    output_dir: Path | None = None
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    def __post_init__(self):
        if self.audio_buffer is None:
            self.audio_buffer = []


clients: dict[str, ClientState] = {}


def setup_output_directory(client_id: str) -> Path:
    """
    Create output directory for this client session

    Args:
        client_id: Unique client identifier

    Returns:
        Path to the output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"session_{client_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for frames
    (output_dir / "frames").mkdir(exist_ok=True)

    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_audio_buffer(client_state: ClientState) -> None:
    """
    Save accumulated audio buffer as a .wav file

    Args:
        client_state: Client state containing audio buffer and metadata
    """
    if not client_state.audio_buffer or not client_state.output_dir:
        logger.warning(f"No audio to save for client {client_state.client_id}")
        return

    try:
        # Concatenate all audio chunks
        audio_data = np.concatenate(client_state.audio_buffer)

        # Reshape for multi-channel audio if needed
        if client_state.audio_channels > 1:
            # Interleaved format: [L, R, L, R, ...]
            audio_data = audio_data.reshape(-1, client_state.audio_channels)

        # Save as WAV file
        output_path = client_state.output_dir / "audio.wav"
        wavfile.write(output_path, client_state.audio_sample_rate, audio_data)

        duration_sec = len(audio_data) / client_state.audio_sample_rate / client_state.audio_channels
        logger.info(
            f"Saved audio for client {client_state.client_id}: "
            f"{output_path} ({duration_sec:.2f} seconds, "
            f"{client_state.audio_sample_rate}Hz, {client_state.audio_channels} channels)"
        )

        # Save session metadata
        metadata_path = client_state.output_dir / "session_metadata.json"
        metadata = {
            "client_id": client_state.client_id,
            "session_timestamp": datetime.now().isoformat(),
            "video_frames": client_state.frame_count,
            "audio_chunks": len(client_state.audio_buffer),
            "audio_duration_seconds": duration_sec,
            "audio_sample_rate": client_state.audio_sample_rate,
            "audio_channels": client_state.audio_channels,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved session metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"Error saving audio for client {client_state.client_id}: {e}")


def identify_object_from_frame(image_base64: str) -> str | None:
    """
    Use Gemini Live to identify what object is in the frame.

    Args:
        image_base64: Base64 encoded image

    Returns:
        Object type string (e.g., "light fixture") or None if not recognized
    """
    try:
        prompt = """Identify the main object in this image that might need repair.
        Respond with ONLY ONE of these options:
        - light fixture
        - faucet
        - door hinge
        - outlet
        - unknown

        Just respond with the object name, nothing else."""

        # Call Gemini Live with vision model
        responses = stream_to_gemini_live_sync(
            image_frame=image_base64,
            text=prompt,
            model="gemini-2.0-flash-exp",
        )

        # Combine streaming response chunks
        response = "".join([r.get("text", "") for r in responses if r.get("type") == "text"])

        # Clean up response
        detected_object = response.strip().lower()

        # Check if it's a known repair object
        if detected_object in REPAIR_PLANS:
            return detected_object

        return None

    except Exception as e:
        logger.error(f"Error identifying object: {e}")
        return None


def process_video_frame(image: Image.Image, frame_info: dict, client_id: str) -> ServerMessage | None:
    """
    Process video frame with AI repair assistant logic

    Args:
        image: PIL Image object
        frame_info: Dictionary with frame metadata (frame_number, resolution, timestamp)
        client_id: Client session identifier

    Returns:
        ServerMessage or None
    """
    try:
        # Get context for this session
        context = context_manager.get_context(client_id)

        # Convert image to base64 for AI analysis
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Check if this is the first frame (no repair started yet)
        if not context.is_repair_started():
            # Only try to detect object every 30 frames to avoid too many AI calls
            if frame_info.get("frame_number", 0) % 30 == 0:
                # Use AI to identify what object needs repair
                detected_object = identify_object_from_frame(image_base64)

                if detected_object and detected_object in REPAIR_PLANS:
                    # Start a new repair session
                    repair_steps = REPAIR_PLANS[detected_object]
                    context_manager.start_repair(
                        session_id=client_id,
                        object_type=detected_object,
                        repair_steps=repair_steps,
                    )

                    # Show safety warning
                    safety_warning = SAFETY_WARNINGS.get(detected_object, "")
                    if safety_warning:
                        context_manager.add_safety_warning(client_id, safety_warning)

                    logger.info(f"Detected {detected_object} for client {client_id}")

                    return OverlayMessage(
                        type="overlay",
                        text=f"Detected: {detected_object.title()}\n{safety_warning}",
                        color=Color(r=1.0, g=1.0, b=0.0, a=1.0),
                        position=Position(x=0.5, y=0.2),
                        timestamp=datetime.now().isoformat(),
                    )
                else:
                    # Object not recognized - show prompt
                    return OverlayMessage(
                        type="overlay",
                        text="Point camera at repair object",
                        color=Color(r=1.0, g=1.0, b=1.0, a=0.7),
                        position=Position(x=0.5, y=0.5),
                        timestamp=datetime.now().isoformat(),
                    )

        # Repair is in progress - show current step
        else:
            current_instruction = context.get_current_step_instruction()

            # Check if repair is complete
            if context.is_repair_complete():
                return OverlayMessage(
                    type="overlay",
                    text=f"{context.object_type.title()} repair complete! ✓",
                    color=Color(r=0.0, g=1.0, b=0.0, a=1.0),
                    position=Position(x=0.5, y=0.5),
                    timestamp=datetime.now().isoformat(),
                )

            # Show current step instruction
            return OverlayMessage(
                type="overlay",
                text=f"Step {context.current_step}/{context.total_steps}:\n{current_instruction}",
                color=Color(r=0.0, g=1.0, b=1.0, a=1.0),
                position=Position(x=0.5, y=0.3),
                timestamp=datetime.now().isoformat(),
            )

    except Exception as e:
        logger.error(f"Error in process_video_frame: {e}")
        return None


def process_audio_chunk(
    audio_data: np.ndarray,
    audio_info: dict,
    client_id: str,
) -> ServerMessage | None:
    """
    Process audio chunk for voice commands

    Args:
        audio_data: numpy array of audio samples (float32)
        audio_info: Dictionary with audio metadata (sample_rate, samples, channels, timestamp)
        client_id: Client session identifier

    Returns:
        ServerMessage or None

    TODO: Add speech-to-text for voice commands like "next step", "repeat", etc.
    For now, volume detection can trigger step advancement
    """
    # Simple volume-based detection (placeholder for actual speech recognition)
    volume = np.abs(audio_data).mean()

    # High volume could indicate voice command (placeholder logic)
    if volume > 0.3:  # Louder threshold for intentional speech
        # Get context to check if we're in a repair
        context = context_manager.get_context(client_id)

        if context.is_repair_started() and not context.is_repair_complete():
            # TODO: Replace with actual speech-to-text to detect "next" command
            # For now, loud audio advances the step
            logger.info(f"Voice command detected for client {client_id} - advancing step")
            context_manager.mark_step_complete(client_id)

            return OverlayMessage(
                type="overlay",
                text="✓ Step complete! Moving to next...",
                color=Color(r=0.0, g=1.0, b=0.0, a=1.0),
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
    logger.info(f"Handshake from {client_state.client_id}: {message.device}")

    # Set up output directory for this session
    client_state.output_dir = setup_output_directory(client_state.client_id)

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

        # Save frame to disk
        if client_state.output_dir:
            frame_filename = f"frame_{message.frame_number:06d}.jpg"
            frame_path = client_state.output_dir / "frames" / frame_filename
            image.save(frame_path, quality=95)

            if message.frame_number % 30 == 0:  # Log every 30 frames
                logger.trace(f"Saved frame {message.frame_number} to {frame_path}")

        # Process frame
        frame_info = {
            "frame_number": message.frame_number,
            "resolution": message.resolution,
            "timestamp": message.timestamp,
        }

        # Call AI repair assistant processing
        instruction = process_video_frame(image, frame_info, client_state.client_id)

        if instruction:
            await client_state.websocket.send(serialize_server_message(instruction))

        # Update client state
        client_state.frame_count += 1

    except Exception as e:
        logger.error(f"Error handling video frame from client {client_state.client_id}: {e}")


async def handle_audio_chunk(message: AudioChunkMessage, client_state: ClientState):
    """Handle incoming audio chunk"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(message.data)

        # Convert bytes to float32 array
        num_samples = len(audio_bytes) // 4
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32, count=num_samples)

        # Calculate audio chunk duration
        duration_ms = (message.samples / message.sample_rate) * 1000

        # Store audio metadata (sample rate and channels) from first chunk
        if len(client_state.audio_buffer) == 0:
            client_state.audio_sample_rate = message.sample_rate
            client_state.audio_channels = message.channels
            logger.info(
                f"Audio configuration for client {client_state.client_id}: "
                f"{message.sample_rate}Hz, {message.channels} channels"
            )

        # Process audio
        audio_info = {
            "sample_rate": message.sample_rate,
            "samples": message.samples,
            "channels": message.channels,
            "timestamp": message.timestamp,
            "duration_ms": duration_ms,
        }

        # Log audio info periodically (every 10 chunks)
        if len(client_state.audio_buffer) % 10 == 0:
            logger.debug(
                f"Audio chunk from client {client_state.client_id}: "
                f"{message.samples} samples, {message.channels} channels, "
                f"{duration_ms:.1f}ms duration, {message.sample_rate}Hz "
                f"(total buffered: {len(client_state.audio_buffer)} chunks)"
            )

        # Call voice command processing
        instruction = process_audio_chunk(audio_data, audio_info, client_state.client_id)

        if instruction:
            await client_state.websocket.send(serialize_server_message(instruction))

        # Buffer ALL audio chunks for saving at the end of the session
        client_state.audio_buffer.append(audio_data)

    except Exception as e:
        logger.error(f"Error handling audio chunk from client {client_state.client_id}: {e}")


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
                        logger.trace(f"Handshake from client {client_id}")
                        await handle_handshake(message, client_state)
                    case VideoFrameMessage():
                        logger.trace(f"Video frame from client {client_id}")
                        await handle_video_frame(message, client_state)
                    case AudioChunkMessage():
                        logger.trace(f"Audio chunk from client {client_id}")
                        await handle_audio_chunk(message, client_state)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id} with message {raw_message}")
            except Exception as e:
                logger.error(f"Error handling message from client {client_id} with message {raw_message}: {e}")
    except ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    finally:
        if str(client_id) in clients:
            # Save audio buffer to .wav file before cleanup
            save_audio_buffer(client_state)

            # Log session statistics
            logger.info(
                f"Session complete for client {client_id}: "
                f"{client_state.frame_count} video frames, "
                f"{len(client_state.audio_buffer)} audio chunks"
            )

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
