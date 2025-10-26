"""Tethyr - AR Processing Server with Gemini Live integration for smart glasses"""

from .email_supervisor import EmailSupervisor
from .gemini_client import GeminiLiveSession, GeminiSessionManager
from .grounding import GroundingDetector

__version__ = "0.1.0"

__all__ = [
    "GeminiLiveSession",
    "GeminiSessionManager",
    "GroundingDetector",
    "EmailSupervisor",
]
