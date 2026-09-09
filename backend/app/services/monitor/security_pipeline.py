"""
Security pipeline — AASIST anti-spoof detection + ECAPA speaker verification.

Consumes audio from the AudioBus, buffers to AASIST windows (64,600 samples
with 32,300-sample hop), and runs detection only when a complete window exists.

CRITICAL INVARIANTS:
1. AASIST window = 64,600 samples (4.0375s at 16kHz) — IMMUTABLE
2. Hop = 32,300 samples (2.01875s) — IMMUTABLE
3. Detection event ONLY when a valid complete window exists
4. ECAPA runs on validated speech segments using 512-sample VAD buffer
5. Strict contract distinguishing RAW SCORES, CALIBRATED PROBABILITY, and DECISION
6. No uncalibrated threshold (e.g. 0.5) displayed as validated operating point
7. Affected windows during audio discontinuity are marked INVALID and excluded from decisions
8. ECAPA speaker evidence and AASIST spoof evidence are kept strictly separate
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, Set

import torch
import numpy as np

from app.services.monitor.audio_bus import ConsumerQueue, AudioChunk, DiscontinuityEvent
from app.services.antispoof.calibrator import score_calibrator, CalibratedOutput

logger = logging.getLogger(__name__)

# AASIST window contract — IMMUTABLE, from existing validated implementation
AASIST_WINDOW_SAMPLES = 64600       # ~4.0375s at 16kHz
AASIST_HOP_SAMPLES = 32300          # 50% overlap = ~2.01875s
VAD_FRAME_SAMPLES = 512             # 32ms at 16kHz for Silero VAD


@dataclass
class DetectionEvent:
    """Anti-spoof detection event from a single AASIST window."""
    type: str = "detection"
    window_index: int = 0
    sample_start: int = 0
    sample_end: int = 0
    latency_ms: float = 0.0

    # Raw AASIST model outputs
    raw_logit_spoof: float = 0.0
    raw_logit_bonafide: float = 0.0
    spoof_score: float = 0.0        # Softmax of spoof logit
    bonafide_score: float = 0.0     # Softmax of bonafide logit
    model: str = "Spectra-AASIST"

    # Calibrated probability and decision (Strict distinction)
    calibrated_probability: Optional[float] = None  # None when uncalibrated
    decision: str = "UNCALIBRATED"                   # "UNCALIBRATED", "GENUINE", "SUSPICIOUS", "INVALID"
    calibration_status: str = "CALIBRATION_REQUIRED" # "CALIBRATION_REQUIRED", "CALIBRATED", "INVALID_WINDOW"
    calibration_version: str = "none"
    calibration_method: str = "none"
    operating_threshold: Optional[float] = None
    target_domain: str = "webrtc_opus"

    # Window validity — set to False on audio discontinuity
    window_valid: bool = True
    invalidation_reason: str = ""


@dataclass
class SpeakerEvent:
    """Speaker verification event from ECAPA (Kept strictly separate from spoofing)."""
    type: str = "speaker"
    sample_start: int = 0
    sample_end: int = 0
    embedding_dim: int = 192
    speaker_similarity: Optional[float] = None
    enrolled_speaker_id: Optional[str] = None
    match: Optional[bool] = None
    calibration: str = "REQUIRED"
    latency_ms: float = 0.0


@dataclass
class WindowProgress:
    """Progress toward the next AASIST window."""
    type: str = "window_progress"
    samples_collected: int = 0
    samples_needed: int = AASIST_WINDOW_SAMPLES
    progress_pct: float = 0.0
    next_window_index: int = 0


class SecurityPipeline:
    """Anti-spoof detection pipeline with independent VAD and AASIST windowing.

    Owns its own VAD instance (VAD-B) — completely independent from
    the caption pipeline's VAD-A. No shared mutable state.
    """

    def __init__(
        self,
        audio_queue: ConsumerQueue,
        on_event: Callable,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self._queue = audio_queue
        self._on_event = on_event
        self._executor = executor or ThreadPoolExecutor(max_workers=1)

        # VAD state — independent instance for ECAPA speech gating
        self._vad_model = None
        self._vad_loaded = False
        self._vad_accumulator = torch.tensor([], dtype=torch.float32)

        # AASIST windowing buffer
        self._buffer = torch.tensor([], dtype=torch.float32)
        self._buffer_start_offset = 0
        self._window_index = 0
        self._samples_since_last_window = 0

        # Discontinuity tracking
        self._has_discontinuity = False
        self._discontinuity_reason = ""

        # Telemetry counters
        self._windows_processed = 0
        self._windows_abstained = 0
        self._ecapa_inferences = 0
        self._discontinuity_count = 0
        self._pipeline_errors = 0

        # ECAPA embedding policy
        self._speech_accumulator: list[torch.Tensor] = []
        self._speech_sample_start = 0
        self._ecapa_min_speech_samples = 16000  # 1.0s speech minimum for reliable embedding
        self._in_speech = False

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def on_discontinuity(self, event: DiscontinuityEvent):
        """Called by AudioBus when samples are lost in consumer queue."""
        self._has_discontinuity = True
        self._discontinuity_reason = event.reason
        self._discontinuity_count += 1
        logger.warning(
            f"SecurityPipeline: audio discontinuity at "
            f"samples {event.sample_start}-{event.sample_end}: {event.reason}"
        )

    async def start(self):
        """Start the security pipeline."""
        self._load_vad()
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("SecurityPipeline started")

    async def stop(self):
        """Stop the security pipeline."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"SecurityPipeline stopped. "
            f"Windows analyzed: {self._window_index}"
        )

    def _load_vad(self):
        """Load independent VAD model instance (VAD-B)."""
        try:
            self._vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self._vad_loaded = True
            logger.info("SecurityPipeline: VAD-B loaded (independent instance)")
        except Exception as e:
            logger.error(f"SecurityPipeline: VAD-B load failed: {e}")
            self._vad_loaded = False

    def _process_ecapa_vad(self, chunk_samples: torch.Tensor, current_sample_offset: int) -> bool:
        """Evaluate VAD for ECAPA speech gating using 512-sample buffer."""
        self._vad_accumulator = torch.cat([self._vad_accumulator, chunk_samples])
        speech_detected = False

        while len(self._vad_accumulator) >= VAD_FRAME_SAMPLES:
            frame = self._vad_accumulator[:VAD_FRAME_SAMPLES]
            self._vad_accumulator = self._vad_accumulator[VAD_FRAME_SAMPLES:]

            if self._vad_loaded:
                try:
                    prob = self._vad_model(frame, 16000).item()
                    if prob > 0.5:
                        speech_detected = True
                except Exception:
                    speech_detected = True
            else:
                speech_detected = True

        return speech_detected

    async def _run(self):
        """Main processing loop."""
        while self._running:
            chunk: Optional[AudioChunk] = await self._queue.get(timeout=0.05)
            if chunk is None:
                continue

            # Append to rolling AASIST buffer
            self._buffer = torch.cat([self._buffer, chunk.samples])
            if len(self._buffer) == len(chunk.samples):
                self._buffer_start_offset = chunk.sample_offset

            # Track speech for ECAPA
            speech_detected = self._process_ecapa_vad(chunk.samples, chunk.sample_offset)
            if speech_detected:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_sample_start = chunk.sample_offset
                    self._speech_accumulator = []
                self._speech_accumulator.append(chunk.samples)
            else:
                if self._in_speech:
                    self._in_speech = False
                    await self._maybe_run_ecapa()

            # Emit progress toward next AASIST window
            progress = len(self._buffer) / AASIST_WINDOW_SAMPLES * 100
            self._on_event(WindowProgress(
                samples_collected=len(self._buffer),
                samples_needed=AASIST_WINDOW_SAMPLES,
                progress_pct=min(progress, 100.0),
                next_window_index=self._window_index,
            ))

            # Check if we have complete 64,600-sample AASIST windows
            while len(self._buffer) >= AASIST_WINDOW_SAMPLES:
                window = self._buffer[:AASIST_WINDOW_SAMPLES]
                window_start = self._buffer_start_offset

                # Check if this window suffered audio discontinuity
                window_valid = not self._has_discontinuity
                discontinuity_reason = self._discontinuity_reason if self._has_discontinuity else ""
                self._has_discontinuity = False  # Reset for next window
                self._discontinuity_reason = ""

                # Run AASIST detection with calibration
                await self._analyze_window(
                    window=window,
                    window_start=window_start,
                    window_valid=window_valid,
                    invalidation_reason=discontinuity_reason,
                )

                # Advance by hop (32,300 samples)
                self._buffer = self._buffer[AASIST_HOP_SAMPLES:]
                self._buffer_start_offset += AASIST_HOP_SAMPLES

    async def _analyze_window(
        self,
        window: torch.Tensor,
        window_start: int,
        window_valid: bool,
        invalidation_reason: str = "",
    ):
        """Run AASIST inference and score calibration on a complete 64,600-sample window."""
        self._window_index += 1
        loop = asyncio.get_event_loop()

        # Check strict sample count invariant (Rule 16)
        if len(window) != AASIST_WINDOW_SAMPLES:
            window_valid = False
            invalidation_reason = f"WINDOW_SIZE_MISMATCH ({len(window)} != {AASIST_WINDOW_SAMPLES})"

        try:
            from app.services.antispoof.spectra_detector import spectra_detector

            if not spectra_detector.is_loaded:
                self._on_event(DetectionEvent(
                    window_index=self._window_index,
                    sample_start=window_start,
                    sample_end=window_start + AASIST_WINDOW_SAMPLES,
                    model="Spectra-AASIST",
                    calibration_status="MODEL_NOT_LOADED",
                    decision="UNAVAILABLE",
                    window_valid=False,
                    invalidation_reason="AASIST model not loaded",
                ))
                return

            # Compute window acoustic energy
            window_rms = float(torch.sqrt(torch.mean(window ** 2)))

            # Run raw model inference in thread pool
            start_time = time.time()
            result = await loop.run_in_executor(
                self._executor,
                spectra_detector.predict,
                window,
            )
            latency_ms = (time.time() - start_time) * 1000

            # Pass through calibrated decision contract with quality-aware abstention
            calibrated: CalibratedOutput = score_calibrator.calibrate(
                raw_spoof_score=result.spoof_score,
                raw_logit_spoof=result.raw_logit_spoof,
                raw_logit_bonafide=result.raw_logit_bonafide,
                window_valid=window_valid,
                invalidation_reason=invalidation_reason,
                rms=window_rms,
            )

            self._windows_processed += 1
            if calibrated.decision == "ABSTAIN":
                self._windows_abstained += 1

            logger.info(
                f"[LIVE TRACE: WIN-{self._window_index}] AUDIO: samples=[{window_start}..{window_start + AASIST_WINDOW_SAMPLES}] "
                f"({AASIST_WINDOW_SAMPLES/16000:.2f}s) RMS={window_rms:.5f} | "
                f"AASIST: model={result.model} logit_spoof={calibrated.raw_logit_spoof:.4f} "
                f"logit_bonafide={calibrated.raw_logit_bonafide:.4f} spoof_score={calibrated.raw_spoof_score:.4f} "
                f"valid={calibrated.is_valid} | DECISION: {calibrated.decision} ({calibrated.calibration_status}) "
                f"latency={latency_ms:.1f}ms"
            )

            self._on_event(DetectionEvent(
                window_index=self._window_index,
                sample_start=window_start,
                sample_end=window_start + AASIST_WINDOW_SAMPLES,
                latency_ms=latency_ms,
                raw_logit_spoof=calibrated.raw_logit_spoof,
                raw_logit_bonafide=calibrated.raw_logit_bonafide,
                spoof_score=calibrated.raw_spoof_score,
                bonafide_score=result.bonafide_score,
                model=result.model,
                calibrated_probability=calibrated.calibrated_probability,
                decision=calibrated.decision,
                calibration_status=calibrated.calibration_status,
                calibration_version=calibrated.calibration_version,
                calibration_method=calibrated.calibration_method,
                operating_threshold=calibrated.operating_threshold,
                target_domain=calibrated.target_domain,
                window_valid=calibrated.is_valid,
                invalidation_reason=calibrated.invalidation_reason,
            ))

        except Exception as e:
            self._pipeline_errors += 1
            logger.error(f"AASIST inference error: {e}", exc_info=True)
            self._on_event(DetectionEvent(
                window_index=self._window_index,
                sample_start=window_start,
                sample_end=window_start + AASIST_WINDOW_SAMPLES,
                calibration_status="INFERENCE_ERROR",
                decision="ERROR",
                window_valid=False,
                invalidation_reason=str(e),
            ))

    async def _maybe_run_ecapa(self):
        """Run ECAPA if accumulated speech meets minimum duration."""
        if not self._speech_accumulator:
            return

        segment = torch.cat(self._speech_accumulator)
        self._speech_accumulator = []

        if len(segment) < self._ecapa_min_speech_samples:
            return

        loop = asyncio.get_event_loop()

        try:
            from app.services.speaker.ecapa_service import ecapa_service

            if not ecapa_service.is_loaded:
                self._on_event(SpeakerEvent(
                    sample_start=self._speech_sample_start,
                    sample_end=self._speech_sample_start + len(segment),
                    calibration="MODEL_NOT_LOADED",
                ))
                return

            start_time = time.time()
            embedding = await loop.run_in_executor(
                self._executor,
                ecapa_service.get_embedding,
                segment,
            )
            latency_ms = (time.time() - start_time) * 1000
            self._ecapa_inferences += 1

            self._on_event(SpeakerEvent(
                sample_start=self._speech_sample_start,
                sample_end=self._speech_sample_start + len(segment),
                embedding_dim=len(embedding) if embedding is not None else 0,
                speaker_similarity=None,
                calibration="REQUIRED",
                latency_ms=latency_ms,
            ))

        except Exception as e:
            self._pipeline_errors += 1
            logger.error(f"ECAPA inference error: {e}", exc_info=True)

    def get_telemetry(self) -> dict:
        """Get security pipeline telemetry."""
        return {
            "vad_loaded": self._vad_loaded,
            "aasist_windows_generated": self._window_index,
            "aasist_windows_processed": self._windows_processed,
            "aasist_abstentions": self._windows_abstained,
            "ecapa_inferences": self._ecapa_inferences,
            "discontinuities_count": self._discontinuity_count,
            "pipeline_errors": self._pipeline_errors,
        }
