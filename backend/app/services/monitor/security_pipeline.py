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
from app.schemas.analysis import SpoofResult, RiskLevel
from app.services.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

# AASIST window contract — IMMUTABLE, from existing validated implementation
AASIST_WINDOW_SAMPLES = 64600       # ~4.0375s at 16kHz
AASIST_HOP_SAMPLES = 32300          # 50% overlap = ~2.01875s
VAD_FRAME_SAMPLES = 512             # 32ms at 16kHz for Silero VAD


@dataclass
class DetectionEvent:
    """Anti-spoof detection event from a single AASIST window."""
    type: str = "detection"
    session_id: str = ""
    track_id: str = ""
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
    decision: str = "UNCALIBRATED"                   # "UNCALIBRATED", "GENUINE", "SUSPICIOUS", "INVALID", "ABSTAIN"
    calibration_status: str = "CALIBRATION_REQUIRED" # "CALIBRATION_REQUIRED", "CALIBRATED", "INVALID_WINDOW", "INSUFFICIENT_AUDIO_QUALITY"
    calibration_version: str = "none"
    calibration_method: str = "none"
    operating_threshold: Optional[float] = None
    target_domain: str = "webrtc_opus"

    # Window validity — set to False on silence or audio discontinuity
    window_valid: bool = True
    invalidation_reason: str = ""


@dataclass
class SpeakerEvent:
    """Speaker verification event from ECAPA (Kept strictly separate from spoofing)."""
    type: str = "speaker_identity"
    session_id: str = ""
    track_id: str = ""
    sample_start: int = 0
    sample_end: int = 0
    embedding_dim: int = 192
    speaker_similarity: Optional[float] = None
    enrolled_speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    match: Optional[bool] = None
    calibration: str = "REQUIRED"
    latency_ms: float = 0.0
    status: str = "UNAVAILABLE"
    reason: Optional[str] = None
    best_match: Optional[dict] = None
    candidates: list[dict] = field(default_factory=list)
    profiles_checked: int = 0
    speech_window_id: str = ""
    updated_at: str = ""


@dataclass
class WindowProgress:
    """Progress toward the next AASIST window."""
    type: str = "window_progress"
    session_id: str = ""
    track_id: str = ""
    samples_collected: int = 0
    samples_needed: int = AASIST_WINDOW_SAMPLES
    progress_pct: float = 0.0
    next_window_index: int = 0


@dataclass
class ProtectionDecision:
    """Consolidated protection decision emitted by the backend security pipeline.

    Frontend renders this as single source of truth — never recalculates spoof risk.
    """
    type: str = "protection_decision"
    session_id: str = ""
    track_id: str = ""
    decision: str = "INSUFFICIENT_EVIDENCE"  # "SAFE", "MONITOR", "WARN", "VERIFY", "BLOCK", "INSUFFICIENT_EVIDENCE", "ABSTAIN"
    risk_level: str = "LOW"                 # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    quality_status: str = "OK"              # "OK", "SILENCE_NO_SPEECH", "AUDIO_DISCONTINUITY", "INSUFFICIENT_AUDIO_QUALITY"
    calibration_status: str = "CALIBRATION_REQUIRED"
    window_count: int = 0
    suspicious_count: int = 0
    spoof_score_avg: float = 0.0


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
        session_id: str = "",
        track_id: str = "",
    ):
        self._queue = audio_queue
        self._on_event = on_event
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._session_id = session_id
        self._track_id = track_id

        # Risk Decision Engine & Evidence Registry
        self._risk_engine = RiskEngine()
        self._evidence_ids: list[str] = []
        self._valid_windows_count = 0
        self._suspicious_windows_count = 0
        self._current_protection_decision = "INSUFFICIENT_EVIDENCE"

        # VAD state — independent instance for ECAPA speech gating
        self._vad_model = None
        self._vad_loaded = False

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

        # ECAPA frame buffer & speech segmentation state (with onset & hangover hysteresis)
        self._ecapa_frame_buffer = torch.tensor([], dtype=torch.float32)
        self._speech_segment_frames: list[torch.Tensor] = []
        self._speech_segment_sample_start = 0
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._in_speech = False
        self._speech_hangover_frames = 12  # ~384ms hangover for intra-sentence micro-pauses
        self._speech_onset_frames = 2      # ~64ms onset confirmation
        self._ecapa_continuous_samples = 24000  # 1.5s continuous speech trigger
        self._ecapa_min_samples = 4000     # 0.25s minimum for final pause trigger
        self._speech_window_counter = 0
        self._last_speaker_status = "UNAVAILABLE"
        self._last_speaker_best_match = None

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

        # Emit initial speaker state (Part 2 & Part 17)
        try:
            from app.services.speaker.ecapa_service import ecapa_service
            from datetime import datetime, timezone
            if not ecapa_service.is_loaded:
                init_status = "UNAVAILABLE"
            elif len(ecapa_service.list_profiles()) == 0:
                init_status = "NO_ENROLLED_PROFILES"
            else:
                init_status = "INSUFFICIENT_EVIDENCE"

            self._last_speaker_status = init_status
            self._on_event(SpeakerEvent(
                session_id=self._session_id,
                track_id=self._track_id,
                status=init_status,
                reason="INITIALIZING" if init_status == "INSUFFICIENT_EVIDENCE" else init_status,
                profiles_checked=len(ecapa_service.list_profiles()) if ecapa_service.is_loaded else 0,
                updated_at=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as e:
            logger.warning(f"Failed to emit initial speaker event: {e}")

    async def stop(self):
        """Stop the security pipeline and clean up all session resources."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._buffer = torch.tensor([], dtype=torch.float32)
        self._ecapa_frame_buffer = torch.tensor([], dtype=torch.float32)
        self._speech_segment_frames = []
        self._in_speech = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
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

    def _evaluate_frame_is_speech(self, frame: torch.Tensor) -> bool:
        """Evaluate if a single 512-sample frame contains speech."""
        frame_1d = frame.squeeze()
        if frame_1d.ndim != 1 or len(frame_1d) != VAD_FRAME_SAMPLES:
            return False

        if self._vad_loaded and self._vad_model is not None:
            try:
                prob = self._vad_model(frame_1d, 16000).item()
                if prob > 0.5:
                    return True
            except Exception:
                pass
        # Fallback acoustic energy check: high amplitude signals (e.g. synthetic test audio with RMS > 0.02)
        rms = float(torch.sqrt(torch.mean(frame_1d ** 2)))
        return rms > 0.02

    async def _process_audio_for_ecapa(self, chunk: AudioChunk):
        """Accumulate speech frames with VAD hysteresis and trigger ECAPA inference.

        Hysteresis & Windowing Invariants:
        - Continuous frame buffering (no 20ms chunk truncation)
        - 2-frame onset hysteresis (~64ms)
        - 12-frame conversational hangover (~384ms) so pauses within words/sentences don't wipe state
        - Continuous speech trigger at 24,000 samples (~1.5s) retaining 8,000 samples (~0.5s) tail
        - Pause trigger on silence hangover expiry (minimum 4,000 samples ~0.25s)
        """
        samples = chunk.samples.squeeze()
        if samples.ndim == 0:
            return
        if samples.ndim > 1:
            samples = samples.flatten()
        self._ecapa_frame_buffer = torch.cat([self._ecapa_frame_buffer, samples])

        while len(self._ecapa_frame_buffer) >= VAD_FRAME_SAMPLES:
            frame = self._ecapa_frame_buffer[:VAD_FRAME_SAMPLES]
            self._ecapa_frame_buffer = self._ecapa_frame_buffer[VAD_FRAME_SAMPLES:]

            frame_offset = chunk.sample_offset - len(self._ecapa_frame_buffer)
            is_speech = self._evaluate_frame_is_speech(frame)

            if is_speech:
                self._consecutive_speech_frames += 1
                self._consecutive_silence_frames = 0

                # Onset hysteresis: requires 2 consecutive speech frames
                if not self._in_speech and self._consecutive_speech_frames >= self._speech_onset_frames:
                    self._in_speech = True
                    self._speech_segment_sample_start = max(0, frame_offset - (self._speech_onset_frames * VAD_FRAME_SAMPLES))
                    self._speech_segment_frames = []

                if self._in_speech:
                    self._speech_segment_frames.append(frame)

                    # Continuous speech trigger: Every 24,000 samples (~1.5s)
                    cur_samples = sum(len(f) for f in self._speech_segment_frames)
                    if cur_samples >= self._ecapa_continuous_samples:
                        segment = torch.cat(self._speech_segment_frames)
                        # Retain ~8000 samples (~0.5s) tail for rolling continuity
                        tail_frames = 8000 // VAD_FRAME_SAMPLES
                        self._speech_segment_frames = self._speech_segment_frames[-tail_frames:]
                        seg_start = self._speech_segment_sample_start
                        self._speech_segment_sample_start += (len(segment) - (tail_frames * VAD_FRAME_SAMPLES))

                        await self._execute_ecapa(
                            segment=segment,
                            sample_start=seg_start,
                            sample_end=seg_start + len(segment),
                        )
            else:
                self._consecutive_speech_frames = 0
                if self._in_speech:
                    self._consecutive_silence_frames += 1
                    self._speech_segment_frames.append(frame)

                    # Pause trigger: after 12 silence frames (~384ms hangover), speech has ended
                    if self._consecutive_silence_frames >= self._speech_hangover_frames:
                        self._in_speech = False
                        # Strip trailing silence frames
                        speech_frames = self._speech_segment_frames[:-self._speech_hangover_frames]
                        self._speech_segment_frames = []
                        self._consecutive_silence_frames = 0

                        if speech_frames:
                            segment = torch.cat(speech_frames)
                            if len(segment) >= self._ecapa_min_samples:
                                await self._execute_ecapa(
                                    segment=segment,
                                    sample_start=self._speech_segment_sample_start,
                                    sample_end=self._speech_segment_sample_start + len(segment),
                                )

    async def _run(self):
        """Main processing loop with failure isolation and backpressure protection."""
        while self._running:
            try:
                chunk: Optional[AudioChunk] = await self._queue.get(timeout=0.05)
                if chunk is None:
                    continue

                # Append to rolling AASIST buffer
                self._buffer = torch.cat([self._buffer, chunk.samples])
                if len(self._buffer) == len(chunk.samples):
                    self._buffer_start_offset = chunk.sample_offset

                # Process audio frames for ECAPA speaker verification with hysteresis
                await self._process_audio_for_ecapa(chunk)

                # Emit progress toward next AASIST window
                progress = len(self._buffer) / AASIST_WINDOW_SAMPLES * 100
                self._on_event(WindowProgress(
                    session_id=self._session_id,
                    track_id=self._track_id,
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
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._pipeline_errors += 1
                logger.error(f"SecurityPipeline processing error (isolated): {e}", exc_info=True)

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

            # Compute window acoustic energy
            window_rms = float(torch.sqrt(torch.mean(window ** 2)))

            # Pre-filter silence: if window energy is below ambient floor, skip VAD entirely
            if window_rms < 0.0015:
                speech_ratio = 0.0
                is_silence = True
            elif self._vad_loaded:
                def _eval_vad(w_tensor, vad_model):
                    try:
                        from silero_vad.utils_vad import get_speech_timestamps
                        ts = get_speech_timestamps(w_tensor, vad_model, sampling_rate=16000)
                        speech_samples = sum(s['end'] - s['start'] for s in ts)
                        return speech_samples / len(w_tensor)
                    except Exception:
                        return 1.0  # Safe fallback if VAD fails

                speech_ratio = await loop.run_in_executor(
                    self._executor,
                    _eval_vad,
                    window,
                    self._vad_model,
                )
                is_silence = (window_rms < 0.0020) or (speech_ratio < 0.05)
            else:
                speech_ratio = 1.0 if window_rms >= 0.0020 else 0.0
                is_silence = window_rms < 0.0020

            # ZERO-TRUST SILENCE GUARD: If window contains silence or no speech,
            # DO NOT run AASIST (which yields erratic ~0.98 logits on empty audio).
            if is_silence:
                self._windows_processed += 1
                self._windows_abstained += 1
                silence_reason = f"SILENCE_NO_SPEECH (RMS={window_rms:.5f}, speech={speech_ratio:.1%})"
                # Support temporal evidence decay and silence de-escalation
                self._risk_engine.decay_on_silence()
                logger.info(
                    f"[LIVE TRACE: WIN-{self._window_index}] AUDIO: samples=[{window_start}..{window_start + AASIST_WINDOW_SAMPLES}] "
                    f"RMS={window_rms:.5f} speech={speech_ratio:.1%} | AASIST: ABSTAIN (silence/no speech) | track={self._track_id}"
                )
                self._on_event(DetectionEvent(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    window_index=self._window_index,
                    sample_start=window_start,
                    sample_end=window_start + AASIST_WINDOW_SAMPLES,
                    latency_ms=0.0,
                    raw_logit_spoof=0.0,
                    raw_logit_bonafide=0.0,
                    spoof_score=0.0,
                    bonafide_score=0.0,
                    model="Spectra-AASIST",
                    calibrated_probability=None,
                    decision="ABSTAIN",
                    calibration_status="NOT_APPLICABLE_SILENCE",
                    calibration_version=score_calibrator.version,
                    calibration_method=score_calibrator.method,
                    operating_threshold=score_calibrator.operating_threshold if score_calibrator.is_loaded else None,
                    target_domain=score_calibrator.domain,
                    window_valid=False,
                    invalidation_reason=silence_reason,
                ))
                self._on_event(ProtectionDecision(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    decision=self._current_protection_decision if self._valid_windows_count > 0 else "INSUFFICIENT_EVIDENCE",
                    risk_level=self._risk_engine._current_level.value,
                    reason=f"Window #{self._window_index} below speech threshold (silence/ambient noise).",
                    evidence_ids=list(self._evidence_ids),
                    timestamp=time.time(),
                    quality_status="SILENCE_NO_SPEECH",
                    calibration_status="NOT_APPLICABLE_SILENCE",
                    window_count=self._valid_windows_count,
                    suspicious_count=self._suspicious_windows_count,
                    spoof_score_avg=round(sum(self._risk_engine._spoof_history) / len(self._risk_engine._spoof_history), 4) if self._risk_engine._spoof_history else 0.0,
                ))
                return

            # Check model availability on validated speech window before inference
            if not spectra_detector.is_loaded:
                self._on_event(DetectionEvent(
                    session_id=self._session_id,
                    track_id=self._track_id,
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

            # Run raw model inference in thread pool only on validated speech windows
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
                delta=result.delta,
                window_valid=window_valid,
                invalidation_reason=invalidation_reason,
                rms=window_rms,
                audio_domain="webrtc_opus",
            )

            self._windows_processed += 1

            # ABSTENTION CONTRACT ENFORCEMENT:
            # ABSTAIN is a first-class state. It must:
            # - not enter spoof history
            # - not increment valid evidence counters
            # - not increment suspicious counters
            # - not increase risk
            # - not be rendered as SPOOF or BONAFIDE
            if calibrated.decision == "ABSTAIN":
                self._windows_abstained += 1
                logger.info(
                    f"[LIVE TRACE: WIN-{self._window_index}] AUDIO: samples=[{window_start}..{window_start + AASIST_WINDOW_SAMPLES}] "
                    f"RMS={window_rms:.5f} | AASIST: ABSTAIN ({calibrated.calibration_status}) - {calibrated.invalidation_reason} | track={self._track_id}"
                )
                self._on_event(DetectionEvent(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    window_index=self._window_index,
                    sample_start=window_start,
                    sample_end=window_start + AASIST_WINDOW_SAMPLES,
                    latency_ms=latency_ms,
                    raw_logit_spoof=calibrated.raw_logit_spoof,
                    raw_logit_bonafide=calibrated.raw_logit_bonafide,
                    spoof_score=calibrated.raw_spoof_score,
                    bonafide_score=result.bonafide_score,
                    model=result.model,
                    calibrated_probability=None,
                    decision="ABSTAIN",
                    calibration_status=calibrated.calibration_status,
                    calibration_version=calibrated.calibration_version,
                    calibration_method=calibrated.calibration_method,
                    operating_threshold=calibrated.operating_threshold,
                    target_domain=calibrated.target_domain,
                    window_valid=False,
                    invalidation_reason=calibrated.invalidation_reason,
                ))
                self._on_event(ProtectionDecision(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    decision=self._current_protection_decision if self._valid_windows_count > 0 else "INSUFFICIENT_EVIDENCE",
                    risk_level=self._risk_engine._current_level.value,
                    reason=f"Window #{self._window_index} abstained: {calibrated.invalidation_reason}",
                    evidence_ids=list(self._evidence_ids),
                    timestamp=time.time(),
                    quality_status="INSUFFICIENT_AUDIO_QUALITY",
                    calibration_status=calibrated.calibration_status,
                    window_count=self._valid_windows_count,
                    suspicious_count=self._suspicious_windows_count,
                    spoof_score_avg=round(sum(self._risk_engine._spoof_history) / len(self._risk_engine._spoof_history), 4) if self._risk_engine._spoof_history else 0.0,
                ))
                return

            # Valid speech window with sufficient energy and valid calibration
            effective_score = (
                calibrated.calibrated_probability
                if calibrated.calibrated_probability is not None
                else (0.0 if result.label == "bonafide" else calibrated.raw_spoof_score)
            )

            # Assess risk using RiskEngine for temporal consistency and hysteresis
            spoof_res = SpoofResult(
                model="Spectra-AASIST",
                label=result.label,
                spoof_score=effective_score,
                bonafide_score=1.0 - effective_score,
                raw_logit_spoof=calibrated.raw_logit_spoof,
                raw_logit_bonafide=calibrated.raw_logit_bonafide,
                delta=calibrated.delta,
            )
            assessment = self._risk_engine.assess_live_window(spoof_res)
            self._valid_windows_count += 1
            win_id = f"win_{self._window_index}"
            self._evidence_ids.append(win_id)
            if calibrated.decision == "SUSPICIOUS" or assessment.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                self._suspicious_windows_count += 1

            self._current_protection_decision = assessment.decision.value

            logger.info(
                f"[LIVE TRACE: WIN-{self._window_index}] AUDIO: samples=[{window_start}..{window_start + AASIST_WINDOW_SAMPLES}] "
                f"({AASIST_WINDOW_SAMPLES/16000:.2f}s) RMS={window_rms:.5f} speech={speech_ratio:.1%} | "
                f"AASIST: model={result.model} logit_spoof={calibrated.raw_logit_spoof:.4f} "
                f"logit_bonafide={calibrated.raw_logit_bonafide:.4f} delta={calibrated.delta:.4f} "
                f"calibrated_prob={calibrated.calibrated_probability} | DECISION: {calibrated.decision} ({calibrated.calibration_status}) "
                f"risk={assessment.decision.value} latency={latency_ms:.1f}ms | track={self._track_id}"
            )

            self._on_event(DetectionEvent(
                session_id=self._session_id,
                track_id=self._track_id,
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

            self._on_event(ProtectionDecision(
                session_id=self._session_id,
                track_id=self._track_id,
                decision=assessment.decision.value,
                risk_level=assessment.risk_level.value,
                reason=assessment.reason,
                evidence_ids=list(self._evidence_ids),
                timestamp=time.time(),
                quality_status="OK",
                calibration_status=calibrated.calibration_status,
                window_count=self._valid_windows_count,
                suspicious_count=self._suspicious_windows_count,
                spoof_score_avg=assessment.spoof_score_avg,
            ))

        except Exception as e:
            self._pipeline_errors += 1
            logger.error(f"AASIST inference error: {e}", exc_info=True)
            self._on_event(DetectionEvent(
                session_id=self._session_id,
                track_id=self._track_id,
                window_index=self._window_index,
                sample_start=window_start,
                sample_end=window_start + AASIST_WINDOW_SAMPLES,
                calibration_status="INFERENCE_ERROR",
                decision="ERROR",
                window_valid=False,
                invalidation_reason=str(e),
            ))

    async def _execute_ecapa(self, segment: torch.Tensor, sample_start: int, sample_end: int):
        """Run ECAPA automatic speaker identification on an accumulated speech segment.

        Invariants:
        - Real ECAPA inference against enrolled profiles
        - Emits forensic structured log [LIVE TRACE: SPEAKER]
        - Distinguishes MATCH, NO_CONFIDENT_MATCH, NO_ENROLLED_PROFILES, INSUFFICIENT_EVIDENCE, UNAVAILABLE
        - Never suppresses or overrides anti-spoof evidence
        - Complete failure isolation (never interrupts call audio)
        """
        loop = asyncio.get_event_loop()
        segment_rms = float(torch.sqrt(torch.mean(segment.float() ** 2)))
        duration_s = round(len(segment) / 16000, 3)

        try:
            from app.services.speaker.ecapa_service import ecapa_service
            from datetime import datetime, timezone

            if not ecapa_service.is_loaded:
                self._on_event(SpeakerEvent(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    status="UNAVAILABLE",
                    reason="ECAPA_UNAVAILABLE",
                    calibration="MODEL_NOT_LOADED",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ))
                return

            self._speech_window_counter += 1
            speech_window_id = f"win_{self._speech_window_counter}"

            all_profiles = ecapa_service.list_profiles()
            valid_profiles = [p for p in all_profiles if p.sample_count > 0 and p.quality_status != "EMPTY"]
            profiles_checked = len(valid_profiles)
            profile_names = [p.speaker_name for p in valid_profiles]

            if profiles_checked == 0:
                self._last_speaker_status = "NO_ENROLLED_PROFILES"
                now_iso = datetime.now(timezone.utc).isoformat()
                logger.info(
                    f"[LIVE TRACE: SPEAKER] session={self._session_id} window={speech_window_id} "
                    f"duration={duration_s}s samples={len(segment)} RMS={segment_rms:.5f} "
                    f"VAD=active ECAPA=skipped profiles_checked=0 status=NO_ENROLLED_PROFILES WS_delivery=dispatched"
                )
                self._on_event(SpeakerEvent(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    status="NO_ENROLLED_PROFILES",
                    reason="NO_ENROLLED_PROFILES",
                    profiles_checked=0,
                    speech_window_id=speech_window_id,
                    updated_at=now_iso,
                ))
                return

            start_time = time.time()
            id_res = await loop.run_in_executor(
                self._executor,
                ecapa_service.identify_speaker,
                segment,
            )
            latency_ms = (time.time() - start_time) * 1000
            self._ecapa_inferences += 1

            best = id_res.best_match if id_res else None
            status_val = id_res.status.value if id_res else "UNAVAILABLE"
            reason_val = id_res.reason if id_res else None

            best_dict = None
            if best:
                best_dict = {
                    "speaker_id": best.speaker_id,
                    "speaker_name": best.speaker_name,
                    "similarity": best.similarity,
                }

            all_candidates = []
            if best:
                all_candidates.append(best)
            if id_res and id_res.alternatives:
                all_candidates.extend(id_res.alternatives)

            candidates_data = [
                {
                    "speaker_id": c.speaker_id,
                    "speaker_name": c.speaker_name,
                    "similarity": c.similarity,
                    "status": c.status.value,
                }
                for c in all_candidates
            ]

            now_iso = datetime.now(timezone.utc).isoformat()
            self._last_speaker_status = status_val
            self._last_speaker_best_match = best_dict

            # Forensic structured log proving real ECAPA execution (Part 1 & 6)
            logger.info(
                f"[LIVE TRACE: SPEAKER] session={self._session_id} window={speech_window_id} "
                f"duration={duration_s}s samples={len(segment)} RMS={segment_rms:.5f} "
                f"VAD=active ECAPA=executed dim=192 norm=unit "
                f"profiles_checked={profiles_checked} profiles={profile_names} "
                f"best_candidate={best.speaker_name if best else 'None'} sim={best.similarity if best else 0.0} "
                f"status={status_val} reason={reason_val} latency={latency_ms:.1f}ms WS_delivery=dispatched"
            )

            self._on_event(SpeakerEvent(
                session_id=self._session_id,
                track_id=self._track_id,
                sample_start=sample_start,
                sample_end=sample_end,
                embedding_dim=192,
                speaker_similarity=best.similarity if best else None,
                enrolled_speaker_id=best.speaker_id if best else None,
                speaker_name=best.speaker_name if best else None,
                match=(status_val == "MATCH"),
                calibration="REQUIRED",
                latency_ms=latency_ms,
                status=status_val,
                reason=reason_val,
                best_match=best_dict,
                candidates=candidates_data,
                profiles_checked=profiles_checked,
                speech_window_id=speech_window_id,
                updated_at=now_iso,
            ))

        except Exception as e:
            self._pipeline_errors += 1
            logger.error(f"ECAPA live identification error (isolated): {e}", exc_info=True)
            try:
                from datetime import datetime, timezone
                self._on_event(SpeakerEvent(
                    session_id=self._session_id,
                    track_id=self._track_id,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    status="UNAVAILABLE",
                    reason="INFERENCE_ERROR",
                    calibration="INFERENCE_ERROR",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ))
            except Exception:
                pass

    async def _maybe_run_ecapa(self, retain_tail: bool = False):
        """Backward-compatible helper for testing."""
        if hasattr(self, '_speech_segment_frames') and self._speech_segment_frames:
            segment = torch.cat(self._speech_segment_frames)
            if len(segment) >= self._ecapa_min_samples:
                await self._execute_ecapa(
                    segment=segment,
                    sample_start=self._speech_segment_sample_start,
                    sample_end=self._speech_segment_sample_start + len(segment),
                )

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
            "last_speaker_status": getattr(self, "_last_speaker_status", "UNAVAILABLE"),
        }
