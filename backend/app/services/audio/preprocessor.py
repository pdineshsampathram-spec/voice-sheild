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

    def _load_with_av(self, source: Union[str, Path, io.BytesIO]) -> tuple[torch.Tensor, int]:
        """Decode audio container (WebM, OGG, MP4, AAC, etc.) using PyAV.
        
        Resamples directly to target_sr (16 kHz) mono float32.
        """
        import av
        container = av.open(source)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if not audio_stream:
            container.close()
            raise ValueError("No audio stream found in container")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.target_sr)
        arrays = []
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            for rf in resampled_frames:
                arrays.append(rf.to_ndarray())
        # Flush resampler buffer
        flush_frames = resampler.resample(None)
        for rf in flush_frames:
            arrays.append(rf.to_ndarray())
        container.close()

        if arrays:
            data = np.concatenate(arrays, axis=1)  # shape: (1, samples)
        else:
            data = np.zeros((1, 0), dtype=np.float32)

        return torch.from_numpy(data), self.target_sr

    def load_file(self, file_path: Union[str, Path]) -> tuple[torch.Tensor, int]:
        """Load an audio file and return (waveform, sample_rate).

        Supports WAV, MP3, M4A, WebM, OGG, FLAC.
        Uses soundfile first; falls back to PyAV for WebM/Opus and container formats.
        """
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        # WebM, OGG, MP4, M4A containers typically cannot be read by libsndfile
        if ext in (".webm", ".ogg", ".opus", ".m4a", ".mp4", ".aac"):
            try:
                waveform, sr = self._load_with_av(file_path)
                logger.info(f"Loaded audio via PyAV: {file_path}, sr={sr}, shape={waveform.shape}")
                return waveform, sr
            except Exception as e_av:
                logger.debug(f"PyAV failed for {file_path}: {e_av}, trying soundfile/torchaudio fallback")

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
            # Fallback to PyAV if not tried yet
            try:
                waveform, sr = self._load_with_av(file_path)
                logger.info(f"Loaded audio via PyAV fallback: {file_path}, sr={sr}, shape={waveform.shape}")
                return waveform, sr
            except Exception as e_av:
                # Last resort fallback to torchaudio
                try:
                    waveform, sr = torchaudio.load(file_path)
                    return waveform, sr
                except Exception as e2:
                    logger.error(f"Failed to load audio: soundfile ({e}), PyAV ({e_av}), torchaudio ({e2})")
                    raise

    def load_bytes(self, audio_bytes: bytes, format_hint: str = "wav") -> tuple[torch.Tensor, int]:
        """Load audio from bytes buffer."""
        # Check if header suggests WebM (EBML 0x1A 0x45 0xDF 0xA3)
        is_webm = len(audio_bytes) >= 4 and audio_bytes[:4] == b"\x1a\x45\xdf\xa3"
        if is_webm or format_hint in ("webm", "ogg", "opus"):
            try:
                buf = io.BytesIO(audio_bytes)
                return self._load_with_av(buf)
            except Exception as e_av:
                logger.debug(f"PyAV bytes load failed: {e_av}")

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
            try:
                buf.seek(0)
                return self._load_with_av(buf)
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

    def compute_rms(self, waveform: torch.Tensor) -> float:
        """Calculate Root Mean Square (RMS) acoustic energy of waveform."""
        if waveform is None or waveform.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.mean(waveform.float() ** 2)))

