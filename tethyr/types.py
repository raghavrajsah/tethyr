"""
Data classes and type definitions for the Tethyr AR server
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from websockets.asyncio.server import ServerConnection

# ============================================================================
# MESSAGE DATA CLASSES (Client -> Server)
# ============================================================================


@dataclass
class Resolution:
    width: int
    height: int


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
    data: str  # base64 encoded audio (float32)
    timestamp: int
    frame_number: int
    sample_rate: int
    samples: int
    channels: int


ClientMessage = HandshakeMessage | VideoFrameMessage | AudioChunkMessage


# ============================================================================
# MESSAGE DATA CLASSES (Server -> Client)
# ============================================================================


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

# ============================================================================
# Gemini Callback
# ============================================================================


@dataclass
class PromptChanged:
    type: Literal["prompt_changed"]
    prompt: str = ""


@dataclass
class Text:
    type: Literal["text"]
    text: str = ""
    timestamp: str = ""
    """Timestamp in ISO format"""


GeminiCallback = PromptChanged | Text

# ============================================================================
# CLIENT STATE
# ============================================================================


@dataclass
class ClientState:
    """State for a connected client"""

    websocket: ServerConnection
    client_id: str
    frame_count: int = 0
    audio_chunk_count: int = 0
    connected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Audio metadata (captured from first audio chunk)
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # Optional: storage directory for debug/recording (set by middleware)
    output_dir: Path | None = None

    # Optional: audio buffer for debug/recording (managed by middleware)
    audio_buffer: list[np.ndarray] = field(default_factory=list)
