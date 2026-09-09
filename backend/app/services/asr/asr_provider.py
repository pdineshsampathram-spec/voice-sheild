"""
Pluggable ASR provider abstraction.

Defines the interface for speech-to-text engines.
Implementations (FasterWhisper, etc.) plug in via this ABC.
Model selection should happen based on measured benchmarks,
not assumptions about performance.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ASRResult:
    """Result of ASR transcription on a speech segment."""
    text: str
    is_partial: bool                # True = hypothesis may change; False = final
    language: str = "en"
    confidence: float = 0.0         # 0-1, model-dependent
    latency_ms: float = 0.0        # Measured inference time
    sample_start: int = 0          # Absolute sample position
    sample_end: int = 0            # Absolute sample position
    model_name: str = "unknown"

    @property
    def duration_seconds(self) -> float:
        return (self.sample_end - self.sample_start) / 16000.0


@dataclass
class ASRModelInfo:
    """Metadata about the loaded ASR model."""
    name: str
    variant: str                   # e.g., "tiny", "base", "small"
    parameters: str = "unknown"    # e.g., "39M"
    measured_rtf: float = 0.0      # Real-time factor (measured, not claimed)
    measured_wer: float = 0.0      # Measured WER (from benchmark, not claimed)
    measured_latency_p50: float = 0.0
    measured_latency_p95: float = 0.0


class ASRProvider(ABC):
    """Abstract base class for ASR engines.

    Implementations must:
    1. Accept 16 kHz mono float32 audio tensors
    2. Return ASRResult with measured latency
    3. Support segment-level transcription (not forced streaming)
    4. Report actual model info (not assumed performance)
    """

    @abstractmethod
    def load_model(self) -> bool:
        """Load the ASR model. Returns True on success."""
        ...

    @abstractmethod
    def transcribe_segment(
        self,
        audio: torch.Tensor,
        sample_offset: int = 0,
    ) -> ASRResult:
        """Transcribe a speech segment.

        Args:
            audio: 1D float32 tensor at 16 kHz.
            sample_offset: Absolute sample position for timeline alignment.

        Returns:
            ASRResult with text and timing.
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ASRModelInfo:
        """Return model metadata with measured (not assumed) metrics."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is ready for inference."""
        ...
