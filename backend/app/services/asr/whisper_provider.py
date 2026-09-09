"""
faster-whisper ASR provider implementation.

Uses CTranslate2-optimized Whisper for CPU-friendly transcription.
Model variant (tiny/base/small) should be selected based on measured
benchmarks on the user's actual hardware, not assumed.

This provider does NOT do naive fixed-interval chunking.
It expects pre-segmented speech (from VAD) and transcribes complete
speech segments.
"""

import logging
import time
import threading
from typing import Optional

import torch
import numpy as np

from app.services.asr.asr_provider import ASRProvider, ASRResult, ASRModelInfo

logger = logging.getLogger(__name__)


class FasterWhisperProvider(ASRProvider):
    """ASR provider using faster-whisper (CTranslate2 backend).

    Accepts complete speech segments from VAD and transcribes them.
    Does not attempt token-by-token streaming — Whisper is a
    segment-level transcription model.
    """

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Args:
            model_size: Whisper variant — "tiny", "base", "small", etc.
                        Selection should be based on measured benchmark,
                        not assumed performance.
            device: "auto", "cpu", or "cuda".
        """
        self._model_size = model_size
        self._device = device
        self._model = None
        self._loaded = False
        self._load_time_ms: float = 0.0
        self._inference_times: list[float] = []
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_size

    @property
    def backend(self) -> str:
        return "ctranslate2"

    def load_model(self) -> bool:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            start = time.time()

            # Resolve device
            compute_type = "int8"  # CPU-friendly quantization
            device = "cpu"
            if self._device == "auto":
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        device = "cuda"
                        compute_type = "float16"
                except ImportError:
                    pass
            elif self._device == "cuda":
                device = "cuda"
                compute_type = "float16"

            self._model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute_type,
            )

            self._load_time_ms = (time.time() - start) * 1000
            self._loaded = True
            logger.info(
                f"faster-whisper ({self._model_size}) loaded in "
                f"{self._load_time_ms:.0f}ms on {device} ({compute_type})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}", exc_info=True)
            self._loaded = False
            return False

    def transcribe_segment(
        self,
        audio: torch.Tensor,
        sample_offset: int = 0,
    ) -> ASRResult:
        """Transcribe a complete speech segment.

        Args:
            audio: 1D float32 tensor at 16 kHz. Should be a complete
                   speech segment from VAD, not a fixed-interval chunk.
            sample_offset: Absolute sample position for timeline alignment.

        Returns:
            ASRResult with final transcription.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("faster-whisper model not loaded")

        start = time.time()

        # Convert to numpy (faster-whisper expects numpy float32)
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio

        # Transcribe with anti-hallucination guards (locked for thread safety)
        with self._lock:
            segments, info = self._model.transcribe(
                audio_np,
                beam_size=1,
                language="en",
                vad_filter=False,  # We already do VAD externally via Silero
                word_timestamps=False,
                condition_on_previous_text=False,  # Prevent loop hallucinations ("I'm sorry", "1, 2, 3")
                no_speech_threshold=0.6,          # Drop silence / noise hallucinations
                log_prob_threshold=-1.0,          # Drop low-confidence garbled tokens
                compression_ratio_threshold=2.4,  # Drop repetitive loop text
            )

            # Collect text from segments
            text_parts = []
            for seg in segments:
                if seg.text and seg.text.strip():
                    text_parts.append(seg.text.strip())

        text = " ".join(text_parts).strip()
        latency_ms = (time.time() - start) * 1000
        self._inference_times.append(latency_ms)

        return ASRResult(
            text=text,
            is_partial=False,  # Segment-level = always final
            language=info.language if info.language else "en",
            confidence=0.0,  # Whisper doesn't expose per-segment confidence
            latency_ms=latency_ms,
            sample_start=sample_offset,
            sample_end=sample_offset + len(audio),
            model_name=f"faster-whisper-{self._model_size}",
        )

    def get_model_info(self) -> ASRModelInfo:
        """Return model info with measured (not assumed) latency."""
        p50 = 0.0
        p95 = 0.0
        if self._inference_times:
            sorted_times = sorted(self._inference_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]

        return ASRModelInfo(
            name="faster-whisper",
            variant=self._model_size,
            parameters={
                "tiny": "39M", "base": "74M", "small": "244M",
                "medium": "769M", "large-v3": "1.5B",
            }.get(self._model_size, "unknown"),
            measured_rtf=0.0,  # Calculated during benchmark
            measured_wer=0.0,  # Calculated during benchmark
            measured_latency_p50=p50,
            measured_latency_p95=p95,
        )

    def is_loaded(self) -> bool:
        return self._loaded
