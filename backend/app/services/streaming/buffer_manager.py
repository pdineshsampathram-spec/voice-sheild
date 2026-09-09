"""
Rolling audio buffer for live WebSocket streaming.

Accumulates incoming PCM chunks and produces fixed-size analysis windows
with configurable overlap (sliding window). This avoids waiting for a
completely new 4-second recording before each inference.
"""

import logging
import threading
from typing import Optional, Callable

import torch

from app.core.config import settings
from app.services.audio.preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


class StreamingBuffer:
    """Thread-safe rolling audio buffer for live analysis.

    Accumulates PCM audio chunks from WebSocket and produces
    analysis-ready windows with sliding overlap.

    Example flow:
        incoming chunks → buffer → [64600 samples window] → callback
                                    [50% overlap]
                                    [next 64600 window] → callback
    """

    def __init__(
        self,
        window_size: int = None,
        overlap: float = None,
        on_window_ready: Optional[Callable[[torch.Tensor, int], None]] = None,
        input_sample_rate: int = 16000,
        input_sample_width: int = 2,
        input_channels: int = 1,
    ):
        self.window_size = window_size or settings.spectra_window_samples
        self.overlap = overlap or settings.sliding_window_overlap
        self.step_size = int(self.window_size * (1 - self.overlap))
        self.on_window_ready = on_window_ready

        self.input_sample_rate = input_sample_rate
        self.input_sample_width = input_sample_width
        self.input_channels = input_channels

        self._buffer = torch.zeros(0)
        self._lock = threading.Lock()
        self._window_count = 0
        self._preprocessor = AudioPreprocessor()

        # Total samples consumed (for position tracking)
        self._total_samples = 0

    def add_chunk(self, pcm_data: bytes):
        """Add a PCM audio chunk to the buffer.

        If enough data is accumulated for a window, triggers the callback.
        """
        # Convert PCM bytes to tensor
        tensor = self._preprocessor.pcm_bytes_to_tensor(
            pcm_data,
            sample_rate=self.input_sample_rate,
            sample_width=self.input_sample_width,
            channels=self.input_channels,
        )

        with self._lock:
            self._buffer = torch.cat([self._buffer, tensor])
            self._total_samples += len(tensor)

            # Check if we have enough for a window
            while len(self._buffer) >= self.window_size:
                window = self._buffer[:self.window_size].clone()
                self._window_count += 1

                # Slide the buffer forward by step_size (not window_size)
                self._buffer = self._buffer[self.step_size:]

                # Trigger callback
                if self.on_window_ready:
                    self.on_window_ready(window, self._window_count)

    def add_tensor(self, audio_tensor: torch.Tensor):
        """Add a pre-converted audio tensor to the buffer."""
        with self._lock:
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()
            self._buffer = torch.cat([self._buffer, audio_tensor])
            self._total_samples += len(audio_tensor)

            while len(self._buffer) >= self.window_size:
                window = self._buffer[:self.window_size].clone()
                self._window_count += 1
                self._buffer = self._buffer[self.step_size:]

                if self.on_window_ready:
                    self.on_window_ready(window, self._window_count)

    def get_current_buffer_seconds(self) -> float:
        """Get the current buffer duration in seconds."""
        return len(self._buffer) / settings.sample_rate

    def get_total_duration(self) -> float:
        """Get total audio duration processed in seconds."""
        return self._total_samples / settings.sample_rate

    @property
    def window_count(self) -> int:
        return self._window_count

    def reset(self):
        """Reset the buffer for a new session."""
        with self._lock:
            self._buffer = torch.zeros(0)
            self._window_count = 0
            self._total_samples = 0
