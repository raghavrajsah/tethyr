"""
Optional storage middleware for debugging/recording
Saves video frames and audio to disk for later analysis
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from loguru import logger
from PIL import Image

from .types import ClientState


# FIXME: The storage middleware/or maybe the transport layer is broken, that
# the audio usually sound more bass than it should be.
class StorageMiddleware:
    """
    Optional middleware to save video frames and audio to disk
    Enable this for debugging or recording sessions
    """

    def __init__(self, enabled: bool = False, base_output_dir: Path = Path("output")):
        """Initialize storage middleware

        Args:
            enabled: Whether to enable frame/audio storage
            base_output_dir: Base directory for storing session data
        """
        self.enabled = enabled
        self.base_output_dir = base_output_dir

        if self.enabled:
            logger.info("Storage middleware enabled - frames and audio will be saved")
        else:
            logger.info("Storage middleware disabled")

    def setup_client_storage(self, client_state: ClientState) -> Path | None:
        """
        Create output directory for a client session

        Args:
            client_state: Client state to setup storage for

        Returns:
            Path to output directory or None if disabled
        """
        if not self.enabled:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_output_dir / f"session_{client_state.client_id}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "frames").mkdir(exist_ok=True)

        logger.info(f"Created storage directory for client {client_state.client_id}: {output_dir}")
        return output_dir

    def save_frame(
        self,
        client_state: ClientState,
        image: Image.Image,
        frame_number: int,
    ):
        """
        Save a video frame to disk

        Args:
            client_state: Client state
            image: PIL Image object
            frame_number: Frame number
        """
        if not self.enabled or not client_state.output_dir:
            return

        try:
            frame_filename = f"frame_{frame_number:06d}.jpg"
            frame_path = client_state.output_dir / "frames" / frame_filename
            image.save(frame_path, quality=95)

            if frame_number % 30 == 0:  # Log every 30 frames
                logger.trace(f"Saved frame {frame_number} for client {client_state.client_id}")

        except Exception as e:
            logger.opt(exception=e).error(f"Error saving frame for client {client_state.client_id}")

    def buffer_audio(self, client_state: ClientState, audio_data: np.ndarray):
        """
        Buffer audio chunk for later saving

        Args:
            client_state: Client state
            audio_data: Audio data as numpy array (float32)
        """
        if not self.enabled:
            return

        client_state.audio_buffer.append(audio_data)

    def save_session(self, client_state: ClientState):
        """
        Save all buffered audio and session metadata to disk

        Args:
            client_state: Client state with buffered audio
        """
        if not self.enabled or not client_state.output_dir:
            return

        try:
            # Save audio buffer as WAV file
            if client_state.audio_buffer:
                self._save_audio_buffer(client_state)

            # Save session metadata
            self._save_session_metadata(client_state)

            logger.info(f"Saved session data for client {client_state.client_id}")

        except Exception as e:
            logger.opt(exception=e).error(f"Error saving session for client {client_state.client_id}")

    def _save_audio_buffer(self, client_state: ClientState):
        """Save accumulated audio buffer as a WAV file"""
        if not client_state.audio_buffer:
            logger.warning(f"No audio to save for client {client_state.client_id}")
            return

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

    def _save_session_metadata(self, client_state: ClientState):
        """Save session metadata to JSON file"""
        metadata_path = client_state.output_dir / "session_metadata.json"

        duration_sec = 0
        if client_state.audio_buffer:
            total_samples = sum(len(chunk) for chunk in client_state.audio_buffer)
            duration_sec = total_samples / client_state.audio_sample_rate / client_state.audio_channels

        metadata = {
            "client_id": client_state.client_id,
            "connected_at": client_state.connected_at,
            "session_end": datetime.now().isoformat(),
            "video_frames": client_state.frame_count,
            "audio_chunks": client_state.audio_chunk_count,
            "audio_duration_seconds": duration_sec,
            "audio_sample_rate": client_state.audio_sample_rate,
            "audio_channels": client_state.audio_channels,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved session metadata to {metadata_path}")
