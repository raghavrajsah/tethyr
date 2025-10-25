"""
Utility functions for message parsing and serialization
"""

import json
from dataclasses import asdict

from loguru import logger

from .types import (
    AudioChunkMessage,
    ClientMessage,
    HandshakeMessage,
    Resolution,
    ServerMessage,
    VideoFrameMessage,
)


def clean_json_message(raw_message: str) -> str:
    """
    Clean JSON message by stripping any trailing garbage after the last '}'.
    This is necessary because Spectacle's WebSocket almost always sends
    small bits of garbage after the last '}'.

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
    """
    Parse incoming client message into typed dataclass

    Args:
        message_dict: Dictionary from parsed JSON

    Returns:
        Typed ClientMessage or None if parsing fails
    """
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
                logger.warning(f"Unknown message type: {message_dict.get('type')}")
                return None
    except Exception as e:
        logger.opt(exception=e).error("Error parsing message")
        return None


def serialize_server_message(message: ServerMessage) -> str:
    """
    Convert server message dataclass to JSON string

    Args:
        message: Server message dataclass

    Returns:
        JSON string representation
    """
    message_dict = asdict(message)
    return json.dumps(message_dict)
