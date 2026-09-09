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
from datetime import datetime
from typing import Optional

import torch

from app.core.config import settings
from app.schemas.analysis import (
    AnalysisResult, AnalysisSource, SegmentResult, SecurityEvent,
    SpoofResult, SpeakerResult, RiskLevel, WindowResult
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
        """Analyze a recorded audio file.

        Args:
            file_path: Path to audio file (WAV, MP3, M4A, WebM).
            audio_bytes: Raw audio bytes (alternative to file_path).
            format_hint: Format hint for bytes loading.
            speaker_id: Optional enrolled speaker ID for verification.

        Returns:
            Complete AnalysisResult with segments and risk assessment.
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

        # 2. VAD — extract speech segments
        if vad_service.is_loaded:
            speech_segments = vad_service.extract_speech(waveform)
            if not speech_segments:
                logger.warning("No speech detected in audio")
                # Fall back to treating entire audio as one segment
                speech_segments = [(waveform, 0.0, duration)]
        else:
            logger.warning("VAD unavailable, treating entire audio as speech")
            speech_segments = [(waveform, 0.0, duration)]

        # 3. Create windows from speech segments and run inference
        segment_results = []
        spoof_results = []
        speaker_results = []

        for seg_idx, (segment, start_sec, end_sec) in enumerate(speech_segments):
            # Create analysis windows from this segment
            windows = self.preprocessor.create_windows(segment, overlap=0.5)

            # Spoof detection — use first window (or average across windows)
            seg_spoof_scores = []
            for window in windows:
                if spectra_detector.is_loaded:
                    spoof = spectra_detector.predict(window)
                else:
                    # Model unavailable — return placeholder with clear status
                    spoof = SpoofResult(
                        model="Spectra-AASIST",
                        label="unavailable",
                        spoof_score=0.0,
                        bonafide_score=0.0,
                        raw_logit_spoof=0.0,
                        raw_logit_bonafide=0.0,
                    )
                seg_spoof_scores.append(spoof)

            # Use the highest spoof score window for this segment
            best_spoof = max(seg_spoof_scores, key=lambda s: s.spoof_score)

            # Speaker verification (optional)
            speaker_result = None
            if speaker_id and ecapa_service.is_loaded:
                try:
                    speaker_result = ecapa_service.verify_speaker(segment, speaker_id)
                except Exception as e:
                    logger.warning(f"Speaker verification failed: {e}")

            segment_results.append(SegmentResult(
                segment_index=seg_idx,
                start_time=round(start_sec, 3),
                end_time=round(end_sec, 3),
                spoof=best_spoof,
                speaker=speaker_result,
            ))
            spoof_results.append(best_spoof)
            speaker_results.append(speaker_result)

        # 4. Aggregate risk assessment
        risk_engine = RiskEngine()  # Fresh instance for recorded analysis
        risk = risk_engine.assess_aggregate(spoof_results, speaker_results)

        elapsed_ms = (time.time() - start_time) * 1000

        result = AnalysisResult(
            timestamp=datetime.utcnow(),
            source=AnalysisSource.RECORDING,
            duration_seconds=round(duration, 2),
            segments=segment_results,
            risk=risk,
            processing_time_ms=round(elapsed_ms, 1),
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
                metadata={"duration": round(duration, 2), "segments": len(segment_results)},
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

        # Speaker verification (optional)
        speaker_result = None
        if speaker_id and ecapa_service.is_loaded:
            try:
                speaker_result = ecapa_service.verify_speaker(audio_tensor, speaker_id)
            except Exception as e:
                logger.warning(f"Speaker verification failed for window {window_index}: {e}")

        # Risk assessment with temporal state
        risk = risk_engine.assess_live_window(spoof, speaker_result)

        return WindowResult(
            window_index=window_index,
            timestamp=datetime.utcnow(),
            spoof=spoof,
            speaker=speaker_result,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            decision=risk.decision,
            reason=risk.reason,
        )


# Singleton
analysis_pipeline = AnalysisPipeline()
