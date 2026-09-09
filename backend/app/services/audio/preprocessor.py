"""
Audio preprocessing service.

Handles loading, resampling, normalization, and windowing of audio
for the Spectra-AASIST and ECAPA-TDNN models.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocesses audio for ML model inference."""

    def __init__(
        self,
        target_sr: int = None,
        window_samples: int = None,
    ):
        self.target_sr = target_sr or settings.sample_rate
        self.window_samples = window_samples or settings.spectra_window_samples

    def load_file(self, file_path: Union[str, Path]) -> tuple[torch.Tensor, int]:
        """Load an audio file and return (waveform, sample_rate).

        Supports WAV, MP3, M4A, WebM, OGG, FLAC.
        """
        file_path = str(file_path)
        try:
            import soundfile as sf
            data, sr = sf.read(file_path, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T  # (channels, samples)
            logger.info(f"Loaded audio: {file_path}, sr={sr}, shape={waveform.shape}")
            return waveform, sr
        except Exception as e:
            # Fallback to torchaudio
            try:
                waveform, sr = torchaudio.load(file_path)
                return waveform, sr
            except Exception as e2:
                logger.error(f"Failed to load audio: soundfile ({e}), torchaudio ({e2})")
                raise

    def load_bytes(self, audio_bytes: bytes, format_hint: str = "wav") -> tuple[torch.Tensor, int]:
        """Load audio from bytes buffer."""
        buf = io.BytesIO(audio_bytes)
        try:
            import soundfile as sf
            data, sr = sf.read(buf, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            return waveform, sr
        except Exception:
            buf.seek(0)
            waveform, sr = torchaudio.load(buf, format=format_hint)
            return waveform, sr

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono."""
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        return waveform

    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform amplitude to [-1, 1]."""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Full preprocessing pipeline: mono → resample → normalize.

        Returns a 1D tensor at target sample rate.
        """
        waveform = self.to_mono(waveform)
        waveform = self.resample(waveform, sr)
        waveform = self.normalize(waveform)
        return waveform.squeeze(0)  # Return 1D: (samples,)

    def create_windows(
        self,
        waveform: torch.Tensor,
        overlap: float = None,
    ) -> list[torch.Tensor]:
        """Split waveform into fixed-size windows for Spectra-AASIST.

        Uses sliding window with configurable overlap. Each window is
        exactly window_samples long (padded if necessary).

        Args:
            waveform: 1D tensor of audio samples
            overlap: Fraction of overlap between windows (0.0 to 0.9)

        Returns:
            List of tensors, each of shape (window_samples,)
        """
        if overlap is None:
            overlap = settings.sliding_window_overlap

        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        total_samples = waveform.shape[0]
        window_size = self.window_samples
        step_size = int(window_size * (1 - overlap))

        if step_size < 1:
            step_size = 1

        windows = []

        if total_samples <= window_size:
            # Pad short audio to window size
            padded = torch.zeros(window_size)
            padded[:total_samples] = waveform
            windows.append(padded)
        else:
            start = 0
            while start < total_samples:
                end = start + window_size
                if end <= total_samples:
                    windows.append(waveform[start:end])
                else:
                    # Last window: pad if not enough samples
                    padded = torch.zeros(window_size)
                    remaining = total_samples - start
                    padded[:remaining] = waveform[start:]
                    windows.append(padded)
                    break
                start += step_size

        return windows

    def pcm_bytes_to_tensor(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
    ) -> torch.Tensor:
        """Convert raw PCM bytes (from WebSocket) to a float tensor.

        Assumes 16-bit signed integer PCM by default.
        """
        dtype = np.int16 if sample_width == 2 else np.float32
        audio_array = np.frombuffer(pcm_data, dtype=dtype)

        if dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0

        if channels > 1:
            audio_array = audio_array.reshape(-1, channels).mean(axis=1)

        tensor = torch.from_numpy(audio_array)

        if sample_rate != self.target_sr:
            tensor = tensor.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            tensor = resampler(tensor).squeeze(0)

        return tensor

    def get_duration(self, waveform: torch.Tensor, sr: int = None) -> float:
        """Get duration of waveform in seconds."""
        sr = sr or self.target_sr
        if waveform.ndim > 1:
            return waveform.shape[-1] / sr
        return waveform.shape[0] / sr
