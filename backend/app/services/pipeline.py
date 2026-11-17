"""
Unified analysis pipeline for VoiceShield.

Used by BOTH recorded and live analysis modes. Orchestrates:
1. Audio preprocessing
2. VAD (speech segment extraction)
3. Spectra-AASIST spoof detection
4. ECAPA-TDNN speaker verification (optional)
5. Risk engine assessment
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import torch

from app.core.config import settings
from app.schemas.analysis import (
    AnalysisResult, AnalysisSource, SegmentResult, SecurityEvent,
    SpoofResult, SpeakerResult, RiskLevel, WindowResult,
    VoiceAuthenticityResult, SpeakerIdentificationResult, SignalQuality,
    SecurityVerdict, MatchStatus
)
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.antispoof.spectra_detector import spectra_detector
from app.services.vad.silero_service import vad_service
from app.services.speaker.ecapa_service import ecapa_service
from app.services.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full analysis pipeline."""

    def __init__(self):
        self.preprocessor = AudioPreprocessor()

    def analyze_file(
        self,
        file_path: str = None,
        audio_bytes: bytes = None,
        format_hint: str = "wav",
        speaker_id: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze a recorded audio file with independent anti-spoof and speaker identification.

        Execution Order:
        1. Decode audio (PyAV / soundfile)
        2. Normalize & silence guard
        3. VAD speech segmentation
        4. Spectra-AASIST spoof detection (extract raw logits & margin Delta)
        5. Calibration check
        6. ECAPA speaker embedding extraction & automatic enrolled-speaker ranking
        7. Independent risk evaluation
        8. Return structured contract

        Args:
            file_path: Path to audio file (WAV, MP3, M4A, WebM, OGG, FLAC).
            audio_bytes: Raw audio bytes (alternative to file_path).
            format_hint: Format hint for bytes loading.
            speaker_id: Optional enrolled speaker ID for targeted verification.

        Returns:
            Complete AnalysisResult with structured spoof, speaker, quality, and verdict objects.
        """
        start_time = time.time()

        # 1. Load and preprocess audio
        if file_path:
            waveform, sr = self.preprocessor.load_file(file_path)
        elif audio_bytes:
            waveform, sr = self.preprocessor.load_bytes(audio_bytes, format_hint)
        else:
            raise ValueError("Either file_path or audio_bytes must be provided")

        waveform = self.preprocessor.preprocess(waveform, sr)
        duration = self.preprocessor.get_duration(waveform)
        rms = self.preprocessor.compute_rms(waveform)

        # Acoustic energy silence guard
        if rms < 0.0015:
            logger.info("Audio file below acoustic energy floor (silence). AASIST abstained.")
            from app.schemas.analysis import RiskAssessment, Decision
            elapsed_ms = (time.time() - start_time) * 1000
            reason_text = "Audio file below acoustic speech energy floor (silence). AASIST detection abstained."
            risk = RiskAssessment(
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
                decision=Decision.SAFE,
                reason=reason_text,
                spoof_score_avg=0.0,
                speaker_similarity_avg=None,
                window_count=0,
                suspicious_window_count=0,
                temporal_consistency=0.0,
            )
            return AnalysisResult(
                timestamp=datetime.now(timezone.utc),
                source=AnalysisSource.RECORDING,
                duration_seconds=round(duration, 2),
                segments=[],
                risk=risk,
                processing_time_ms=round(elapsed_ms, 1),
                spoof=VoiceAuthenticityResult(
                    raw_margin=None,
                    calibrated_probability=None,
                    status="ABSTAIN",
                    evidence="INSUFFICIENT",
                    decision="SAFE",
                ),
                speaker=SpeakerIdentificationResult(
                    best_match=None,
                    alternatives=[],
                    status=MatchStatus.UNAVAILABLE,
                ),
                quality=SignalQuality(
                    status="INSUFFICIENT",
                    reason=reason_text,
                    rms=round(rms, 5),
                    duration_seconds=round(duration, 2),
                ),
                verdict=SecurityVerdict(
                    risk=RiskLevel.LOW,
                    reason=reason_text,
                    risk_score=0.0,
                    decision=Decision.SAFE,
                ),
            )

        # 2. VAD — extract speech segments
        if vad_service.is_loaded:
            speech_segments = vad_service.extract_speech(waveform)
            if not speech_segments:
                logger.info("No speech detected in audio file (silence / non-speech). AASIST abstained.")
                from app.schemas.analysis import RiskAssessment, Decision
                elapsed_ms = (time.time() - start_time) * 1000
                reason_text = "No speech detected in audio (silence / ambient noise). AASIST detection abstained."
                risk = RiskAssessment(
                    risk_score=0.0,
                    risk_level=RiskLevel.LOW,
                    decision=Decision.SAFE,
                    reason=reason_text,
                    spoof_score_avg=0.0,
                    speaker_similarity_avg=None,
                    window_count=0,
                    suspicious_window_count=0,
                    temporal_consistency=0.0,
                )
                return AnalysisResult(
                    timestamp=datetime.now(timezone.utc),
                    source=AnalysisSource.RECORDING,
                    duration_seconds=round(duration, 2),
                    segments=[],
                    risk=risk,
                    processing_time_ms=round(elapsed_ms, 1),
                    spoof=VoiceAuthenticityResult(
                        raw_margin=None,
                        calibrated_probability=None,
                        status="ABSTAIN",
                        evidence="INSUFFICIENT",
                        decision="SAFE",
                    ),
                    speaker=SpeakerIdentificationResult(
                        best_match=None,
                        alternatives=[],
                        status=MatchStatus.UNAVAILABLE,
                    ),
                    quality=SignalQuality(
                        status="INSUFFICIENT",
                        reason=reason_text,
                        rms=round(rms, 5),
                        duration_seconds=round(duration, 2),
                    ),
                    verdict=SecurityVerdict(
                        risk=RiskLevel.LOW,
                        reason=reason_text,
                        risk_score=0.0,
                        decision=Decision.SAFE,
                    ),
                )
        else:
            logger.warning("VAD unavailable, treating entire audio as speech")
            speech_segments = [(waveform, 0.0, duration)]

        # 3. Automatic speaker identification across ALL enrolled profiles
        speaker_id_result: Optional[SpeakerIdentificationResult] = None
        if ecapa_service.is_loaded:
            try:
                speaker_id_result = ecapa_service.identify_speaker(waveform)
            except Exception as e:
                logger.warning(f"Automatic speaker identification failed: {e}")

        # If user explicitly requested a speaker_id, verify that specific speaker
        targeted_speaker_result: Optional[SpeakerResult] = None
        if speaker_id and ecapa_service.is_loaded:
            try:
                targeted_speaker_result = ecapa_service.verify_speaker(waveform, speaker_id)
            except Exception as e:
                logger.warning(f"Targeted speaker verification failed for {speaker_id}: {e}")

        matched_speaker_name = None
        if speaker_id_result and speaker_id_result.best_match:
            matched_speaker_name = speaker_id_result.best_match.speaker_name

        # 4. Create windows from speech segments and run Spectra-AASIST inference
        segment_results = []
        spoof_results = []
        speaker_results = []

        for seg_idx, (segment, start_sec, end_sec) in enumerate(speech_segments):
            windows = self.preprocessor.create_windows(segment, overlap=0.5)

            seg_spoof_scores = []
            for window in windows:
                if spectra_detector.is_loaded:
                    spoof = spectra_detector.predict(window)
                else:
                    spoof = SpoofResult(
                        model="Spectra-AASIST",
                        label="unavailable",
                        spoof_score=0.0,
                        bonafide_score=0.0,
                        raw_logit_spoof=0.0,
                        raw_logit_bonafide=0.0,
                    )
                seg_spoof_scores.append(spoof)

            best_spoof = max(seg_spoof_scores, key=lambda s: s.spoof_score)

            # Per-segment speaker similarity
            seg_speaker_res = None
            if targeted_speaker_result:
                seg_speaker_res = targeted_speaker_result
            elif speaker_id_result and speaker_id_result.best_match:
                bm = speaker_id_result.best_match
                seg_speaker_res = SpeakerResult(
                    model="ECAPA-TDNN",
                    speaker_similarity=bm.similarity,
                    match=(bm.status == MatchStatus.MATCH),
                    threshold_used=settings.speaker_threshold_match,
                    enrolled_speaker_id=bm.speaker_id,
                )

            segment_results.append(SegmentResult(
                segment_index=seg_idx,
                start_time=round(start_sec, 3),
                end_time=round(end_sec, 3),
                spoof=best_spoof,
                speaker=seg_speaker_res,
            ))
            spoof_results.append(best_spoof)
            if seg_speaker_res:
                speaker_results.append(seg_speaker_res)

        # 5. Aggregate risk assessment with strict zero-trust separation
        risk_engine = RiskEngine()
        risk = risk_engine.assess_aggregate(
            spoof_results=spoof_results,
            speaker_results=speaker_results,
            matched_speaker_name=matched_speaker_name,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Extract delta margins for structured spoof contract
        deltas = [s.delta for s in spoof_results if s.delta is not None]
        mean_delta = float(sum(deltas) / len(deltas)) if deltas else None

        calibrator = getattr(spectra_detector, "_calibrator", None)
        is_calibrated = getattr(calibrator, "is_calibrated", False) if calibrator else False
        calibrated_prob = risk.spoof_score_avg if is_calibrated else None
        cal_status = "CALIBRATED" if is_calibrated else "CALIBRATION_UNAVAILABLE"

        if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            evidence = "SPOOF"
        elif risk.risk_level == RiskLevel.MEDIUM:
            evidence = "SUSPICIOUS"
        else:
            evidence = "BONAFIDE"

        spoof_contract = VoiceAuthenticityResult(
            raw_margin=round(mean_delta, 4) if mean_delta is not None else None,
            calibrated_probability=round(calibrated_prob, 4) if calibrated_prob is not None else None,
            status=cal_status,
            evidence=evidence,
            decision=str(risk.decision.value),
        )

        quality_contract = SignalQuality(
            status="VALID",
            reason="Audio signal within nominal acoustic parameters",
            rms=round(rms, 5),
            duration_seconds=round(duration, 2),
        )

        verdict_contract = SecurityVerdict(
            risk=risk.risk_level,
            reason=risk.reason,
            risk_score=risk.risk_score,
            decision=risk.decision,
        )

        result = AnalysisResult(
            timestamp=datetime.now(timezone.utc),
            source=AnalysisSource.RECORDING,
            duration_seconds=round(duration, 2),
            segments=segment_results,
            risk=risk,
            processing_time_ms=round(elapsed_ms, 1),
            spoof=spoof_contract,
            speaker=speaker_id_result,
            quality=quality_contract,
            verdict=verdict_contract,
        )

        # Log security event for history tracking
        try:
            from app.api.sessions import log_security_event
            log_security_event(SecurityEvent(
                session_id=result.session_id,
                spoof_score=risk.spoof_score_avg,
                speaker_similarity=risk.speaker_similarity_avg,
                risk_score=risk.risk_score,
                risk_level=risk.risk_level,
                decision=risk.decision,
                reason=risk.reason,
                source=AnalysisSource.RECORDING,
                metadata={
                    "duration": round(duration, 2),
                    "segments": len(segment_results),
                    "matched_speaker": matched_speaker_name,
                },
            ))
        except Exception as e:
            logger.warning(f"Failed to log security event: {e}")

        return result

    def analyze_window(
        self,
        audio_tensor: torch.Tensor,
        risk_engine: RiskEngine,
        window_index: int,
        speaker_id: Optional[str] = None,
    ) -> WindowResult:
        """Analyze a single live audio window.

        Uses the provided RiskEngine instance which maintains
        temporal state across windows in a live session.

        Args:
            audio_tensor: 1D tensor of preprocessed audio (64600 samples).
            risk_engine: Stateful risk engine for this session.
            window_index: Sequential window number.
            speaker_id: Optional enrolled speaker for verification.

        Returns:
            WindowResult with scores and risk assessment.
        """
        # Acoustic energy silence guard
        rms = float(torch.sqrt(torch.mean(audio_tensor ** 2))) if len(audio_tensor) > 0 else 0.0
        if rms < 0.0015:
            from app.schemas.analysis import Decision
            return WindowResult(
                window_index=window_index,
                timestamp=datetime.now(timezone.utc),
                spoof=SpoofResult(
                    model="Spectra-AASIST",
                    label="silence",
                    spoof_score=0.0,
                    bonafide_score=0.0,
                    raw_logit_spoof=0.0,
                    raw_logit_bonafide=0.0,
                ),
                speaker=None,
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
                decision=Decision.SAFE,
                reason="Window below acoustic speech energy floor (silence). AASIST abstained.",
            )

        # Spoof detection
        if spectra_detector.is_loaded:
            spoof = spectra_detector.predict(audio_tensor)
        else:
            spoof = SpoofResult(
                model="Spectra-AASIST",
                label="unavailable",
                spoof_score=0.0,
                bonafide_score=0.0,
                raw_logit_spoof=0.0,
                raw_logit_bonafide=0.0,
            )

        # Speaker verification (targeted or automatic)
        speaker_result = None
        if speaker_id and ecapa_service.is_loaded:
            try:
                speaker_result = ecapa_service.verify_speaker(audio_tensor, speaker_id)
            except Exception as e:
                logger.warning(f"Speaker verification failed for window {window_index}: {e}")
        elif ecapa_service.is_loaded:
            try:
                id_res = ecapa_service.identify_speaker(audio_tensor)
                if id_res and id_res.best_match:
                    bm = id_res.best_match
                    speaker_result = SpeakerResult(
                        model="ECAPA-TDNN",
                        speaker_similarity=bm.similarity,
                        match=(bm.status == MatchStatus.MATCH),
                        threshold_used=settings.speaker_threshold_match,
                        enrolled_speaker_id=bm.speaker_id,
                    )
            except Exception as e:
                logger.debug(f"Live automatic speaker identification skipped: {e}")

        # Risk assessment with temporal state
        risk = risk_engine.assess_live_window(spoof, speaker_result)

        return WindowResult(
            window_index=window_index,
            timestamp=datetime.now(timezone.utc),
            spoof=spoof,
            speaker=speaker_result,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            decision=risk.decision,
            reason=risk.reason,
        )


# Singleton
analysis_pipeline = AnalysisPipeline()
