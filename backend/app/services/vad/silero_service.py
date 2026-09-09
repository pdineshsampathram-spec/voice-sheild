"""
Silero VAD (Voice Activity Detection) service.

Uses Silero VAD to detect speech segments and filter silence.
Supports both file-based and streaming modes.
"""

import logging
import time
from typing import Optional

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class SileroVADService:
    """Voice Activity Detection using Silero VAD.

    Loaded once at startup. Supports:
    - File-based: extract speech timestamps from a complete waveform
    - Streaming: stateful chunk-by-chunk processing for live audio
    """

    def __init__(self):
        self.model = None
        self._utils = None
        self._loaded = False
        self._load_time_ms: Optional[float] = None
        self._error_message: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> str:
        if self._loaded:
            return "ONLINE"
        if self._error_message:
            return "ERROR"
        return "UNAVAILABLE"

    @property
    def status_message(self) -> Optional[str]:
        return self._error_message

    @property
    def load_time_ms(self) -> Optional[float]:
        return self._load_time_ms

    def load_model(self) -> bool:
        """Load Silero VAD from torch.hub."""
        try:
            start = time.time()
            logger.info("Loading Silero VAD...")

            # Silero VAD is optimized for CPU
            torch.set_num_threads(1)

            self.model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )

            elapsed = (time.time() - start) * 1000
            self._load_time_ms = elapsed
            self._loaded = True
            self._error_message = None
            logger.info(f"Silero VAD loaded in {elapsed:.0f}ms")
            return True

        except Exception as e:
            self._error_message = str(e)
            self._loaded = False
            logger.error(f"Failed to load Silero VAD: {e}", exc_info=True)
            return False

    def get_speech_timestamps(
        self,
        waveform: torch.Tensor,
        sampling_rate: int = None,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> list[dict]:
        """Detect speech segments in a waveform.

        Args:
            waveform: 1D tensor of audio samples at target sample rate.
            sampling_rate: Sample rate of the audio (default: from settings).
            threshold: VAD threshold (0.0-1.0).
            min_speech_duration_ms: Minimum speech segment duration.
            min_silence_duration_ms: Minimum silence between segments.

        Returns:
            List of dicts with 'start' and 'end' keys (sample indices).
        """
        if not self._loaded:
            raise RuntimeError("Silero VAD is not loaded")

        sr = sampling_rate or settings.sample_rate

        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        get_speech_timestamps = self._utils[0]

        timestamps = get_speech_timestamps(
            waveform,
            self.model,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        return timestamps

    def extract_speech(
        self,
        waveform: torch.Tensor,
        timestamps: list[dict] = None,
        sampling_rate: int = None,
    ) -> list[tuple[torch.Tensor, float, float]]:
        """Extract speech segments from a waveform.

        Args:
            waveform: 1D tensor of audio samples.
            timestamps: Pre-computed speech timestamps. If None, will compute them.
            sampling_rate: Sample rate of the audio.

        Returns:
            List of (segment_tensor, start_time_seconds, end_time_seconds) tuples.
        """
        sr = sampling_rate or settings.sample_rate

        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        if timestamps is None:
            timestamps = self.get_speech_timestamps(waveform, sr)

        segments = []
        for ts in timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            segment = waveform[start_sample:end_sample]
            start_time = start_sample / sr
            end_time = end_sample / sr
            segments.append((segment, start_time, end_time))

        return segments

    def has_speech(
        self,
        waveform: torch.Tensor,
        sampling_rate: int = None,
        threshold: float = 0.5,
    ) -> bool:
        """Quick check: does this waveform contain speech?"""
        if not self._loaded:
            return True  # Assume speech if VAD unavailable

        timestamps = self.get_speech_timestamps(
            waveform, sampling_rate, threshold,
            min_speech_duration_ms=200,
        )
        return len(timestamps) > 0

    def reset_states(self):
        """Reset internal states for streaming mode."""
        if self.model is not None:
            self.model.reset_states()


# Singleton instance
vad_service = SileroVADService()
