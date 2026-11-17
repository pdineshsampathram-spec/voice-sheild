"""
Custom risk engine for VoiceShield.

Combines spoof detection scores, speaker verification, temporal consistency,
and debounce logic into a unified risk assessment.

ALL thresholds are configurable via settings. The defaults are starting points
that require calibration with your test audio data.
"""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.schemas.analysis import (
    RiskLevel, Decision, RiskAssessment, SpoofResult, SpeakerResult
)

logger = logging.getLogger(__name__)


# Explicit integer risk level ranking to prevent string comparison bugs ("CRITICAL" > "LOW" is False in Python)
RISK_LEVEL_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class RiskEngine:
    """Stateful risk scoring with temporal consistency and hysteresis.

    For live mode: maintains a rolling window of recent scores and
    applies debounce logic to avoid false alarms from noise.

    For recorded mode: evaluates all segments and produces a
    single aggregate assessment.
    """

    def __init__(self):
        # Rolling window of recent spoof scores (for live mode)
        self._spoof_history: deque[float] = deque(maxlen=settings.risk_window_count)
        self._speaker_history: deque[float] = deque(maxlen=settings.risk_window_count)
        self._last_escalation_time: float = 0
        self._current_level: RiskLevel = RiskLevel.LOW
        self._consecutive_elevated: int = 0

    def reset(self):
        """Reset state for a new session."""
        self._spoof_history.clear()
        self._speaker_history.clear()
        self._last_escalation_time = 0
        self._current_level = RiskLevel.LOW
        self._consecutive_elevated = 0

    def decay_on_silence(self, step: int = 1):
        """Apply evidence decay and silence de-escalation when speech is absent."""
        if self._consecutive_elevated > 0:
            self._consecutive_elevated = max(0, self._consecutive_elevated - step)
        # If no elevated windows remain, de-escalate towards LOW
        if self._consecutive_elevated == 0 and self._current_level != RiskLevel.LOW:
            self._current_level = RiskLevel.LOW

    def assess_single(
        self,
        spoof: SpoofResult,
        speaker: Optional[SpeakerResult] = None,
    ) -> RiskAssessment:
        """Assess risk for a single audio segment (recorded mode).

        No temporal state is used — just the raw scores.
        """
        risk_score = self._compute_risk_score(spoof.spoof_score,
                                               speaker.speaker_similarity if speaker else None)
        risk_level = self._score_to_level(risk_score)
        decision = self._level_to_decision(risk_level)
        reason = self._generate_reason(spoof, speaker, risk_level, decision)

        return RiskAssessment(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            decision=decision,
            reason=reason,
            spoof_score_avg=spoof.spoof_score,
            speaker_similarity_avg=speaker.speaker_similarity if speaker else None,
            window_count=1,
            suspicious_window_count=1 if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0,
            temporal_consistency=0.0,
        )

    def assess_aggregate(
        self,
        spoof_results: list[SpoofResult],
        speaker_results: list[Optional[SpeakerResult]] = None,
        matched_speaker_name: Optional[str] = None,
    ) -> RiskAssessment:
        """Assess risk for a complete recording (multiple segments)."""
        if not spoof_results:
            return RiskAssessment(
                risk_score=0.0,
                risk_level=RiskLevel.LOW,
                decision=Decision.SAFE,
                reason="No audio segments to analyze.",
                spoof_score_avg=0.0,
                window_count=0,
                suspicious_window_count=0,
                temporal_consistency=0.0,
            )

        available_spoofs = [r for r in spoof_results if r.label != "unavailable"]
        if not available_spoofs:
            return RiskAssessment(
                risk_score=0.5,
                risk_level=RiskLevel.MEDIUM,
                decision=Decision.MONITOR,
                reason="Antispoof detector unavailable. Cannot verify audio authenticity.",
                spoof_score_avg=0.0,
                window_count=0,
                suspicious_window_count=0,
                temporal_consistency=0.0,
            )

        spoof_scores = [r.spoof_score for r in available_spoofs]
        avg_spoof = sum(spoof_scores) / len(spoof_scores)
        max_spoof = max(spoof_scores)

        speaker_sims = []
        if speaker_results:
            for s in speaker_results:
                if s is not None:
                    speaker_sims.append(s.speaker_similarity)
        avg_speaker = sum(speaker_sims) / len(speaker_sims) if speaker_sims else None

        # Risk score considers both average and peak
        blended_spoof = avg_spoof * 0.6 + max_spoof * 0.4
        risk_score = self._compute_risk_score(blended_spoof, avg_speaker)
        risk_level = self._score_to_level(risk_score)
        decision = self._level_to_decision(risk_level)

        suspicious_count = sum(1 for s in spoof_scores if s >= settings.spoof_threshold_medium)

        # Variance of spoof scores across segments
        if len(spoof_scores) > 1:
            mean = sum(spoof_scores) / len(spoof_scores)
            variance = sum((s - mean) ** 2 for s in spoof_scores) / len(spoof_scores)
        else:
            variance = 0.0

        reason = self._generate_aggregate_reason(
            avg_spoof, max_spoof, avg_speaker, risk_level,
            len(spoof_results), suspicious_count,
            matched_speaker_name=matched_speaker_name,
        )

        return RiskAssessment(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            decision=decision,
            reason=reason,
            spoof_score_avg=round(avg_spoof, 4),
            speaker_similarity_avg=round(avg_speaker, 4) if avg_speaker is not None else None,
            window_count=len(spoof_results),
            suspicious_window_count=suspicious_count,
            temporal_consistency=round(variance, 4),
        )

    def assess_live_window(
        self,
        spoof: SpoofResult,
        speaker: Optional[SpeakerResult] = None,
    ) -> RiskAssessment:
        """Assess risk for a live streaming window.

        Uses temporal state: rolling window average, consecutive elevated
        counts, and debounce logic to stabilize decisions.
        """
        # Quality Gate / Silence Abstention: do not pollute rolling history with non-speech
        if (
            getattr(spoof, "label", "") in ("abstain", "unavailable", "invalid")
            or (spoof.spoof_score == 0.0 and getattr(spoof, "raw_logit_spoof", 0.0) == 0.0)
        ):
            avg_spoof = (
                sum(self._spoof_history) / len(self._spoof_history)
                if self._spoof_history
                else 0.0
            )
            return RiskAssessment(
                risk_score=round(avg_spoof, 4),
                risk_level=self._current_level,
                decision=self._level_to_decision(self._current_level),
                reason="Audio energy below speech threshold (silence/abstain); maintaining baseline.",
                spoof_score_avg=round(avg_spoof, 4),
                speaker_similarity_avg=(
                    sum(self._speaker_history) / len(self._speaker_history)
                    if self._speaker_history
                    else None
                ),
                window_count=len(self._spoof_history),
                suspicious_window_count=sum(
                    1 for s in self._spoof_history if s >= settings.spoof_threshold_medium
                ),
                temporal_consistency=0.0,
            )

        # Add to history
        self._spoof_history.append(spoof.spoof_score)
        if speaker is not None:
            self._speaker_history.append(speaker.speaker_similarity)

        # Compute rolling averages
        avg_spoof = sum(self._spoof_history) / len(self._spoof_history)
        avg_speaker = (sum(self._speaker_history) / len(self._speaker_history)
                       if self._speaker_history else None)

        # Risk score from rolling average
        risk_score = self._compute_risk_score(avg_spoof, avg_speaker)
        candidate_level = self._score_to_level(risk_score)

        # Temporal consistency check
        if len(self._spoof_history) > 1:
            scores = list(self._spoof_history)
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        else:
            variance = 0.0

        # Hysteresis: track consecutive elevated windows
        is_elevated = candidate_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        if is_elevated:
            self._consecutive_elevated += 1
        else:
            self._consecutive_elevated = max(0, self._consecutive_elevated - 1)

        # Debounce: only escalate if enough consecutive elevated windows
        now = time.time()
        final_level = self._current_level

        candidate_rank = RISK_LEVEL_ORDER[candidate_level]
        current_rank = RISK_LEVEL_ORDER[self._current_level]

        if candidate_rank > current_rank:
            # Escalation: require consecutive elevated windows
            if self._consecutive_elevated >= settings.risk_escalation_count:
                if (now - self._last_escalation_time) >= settings.risk_debounce_seconds:
                    final_level = candidate_level
                    self._last_escalation_time = now
        elif candidate_rank < current_rank:
            # De-escalation: allow faster when no elevated windows remain
            if self._consecutive_elevated == 0:
                final_level = candidate_level

        self._current_level = final_level
        decision = self._level_to_decision(final_level)

        # STRICT INVARIANT: If candidate level is elevated (HIGH or CRITICAL) or avg_spoof >= medium,
        # NEVER declare SAFE during debounce! Transition through MONITOR.
        if decision == Decision.SAFE and (is_elevated or avg_spoof >= settings.spoof_threshold_medium):
            decision = Decision.MONITOR

        suspicious_count = sum(1 for s in self._spoof_history
                               if s >= settings.spoof_threshold_medium)

        reason = self._generate_reason(spoof, speaker, final_level, decision)

        return RiskAssessment(
            risk_score=round(risk_score, 4),
            risk_level=final_level,
            decision=decision,
            reason=reason,
            spoof_score_avg=round(avg_spoof, 4),
            speaker_similarity_avg=round(avg_speaker, 4) if avg_speaker is not None else None,
            window_count=len(self._spoof_history),
            suspicious_window_count=suspicious_count,
            temporal_consistency=round(variance, 4),
        )

    def _compute_risk_score(
        self,
        spoof_score: float,
        speaker_similarity: Optional[float] = None,
    ) -> float:
        """Compute a 0-1 risk score strictly from spoof detection evidence.

        SECURITY INVARIANT:
        'Whose voice does this resemble?' != 'Is this voice authentic?'
        Speaker resemblance must NEVER suppress or downgrade spoof risk.
        An AI-cloned voice of an enrolled speaker will have high speaker similarity
        and high spoof risk -> The final risk score must remain elevated.
        """
        risk = spoof_score
        return min(max(risk, 0.0), 1.0)

    def _score_to_level(self, risk_score: float) -> RiskLevel:
        """Map risk score to level using configurable thresholds."""
        if risk_score >= settings.spoof_threshold_critical:
            return RiskLevel.CRITICAL
        elif risk_score >= settings.spoof_threshold_high:
            return RiskLevel.HIGH
        elif risk_score >= settings.spoof_threshold_medium:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _level_to_decision(self, level: RiskLevel) -> Decision:
        """Map risk level to recommended decision."""
        return {
            RiskLevel.LOW: Decision.SAFE,
            RiskLevel.MEDIUM: Decision.MONITOR,
            RiskLevel.HIGH: Decision.WARN,
            RiskLevel.CRITICAL: Decision.VERIFY,
        }[level]

    def _generate_reason(
        self,
        spoof: SpoofResult,
        speaker: Optional[SpeakerResult],
        level: RiskLevel,
        decision: Decision = Decision.SAFE,
        speaker_name: Optional[str] = None,
    ) -> str:
        """Generate a human-readable reason for the risk assessment.

        Explicitly articulates the orthogonal signals of speaker resemblance
        and voice authenticity.
        """
        parts = []
        name_str = speaker_name or (f"ID {speaker.enrolled_speaker_id}" if speaker and speaker.enrolled_speaker_id else "enrolled speaker")

        is_synthetic = level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        has_speaker_match = speaker is not None and (speaker.match or speaker.speaker_similarity >= settings.speaker_threshold_match)

        if is_synthetic and has_speaker_match:
            parts.append(f"Audio resembles {name_str}, but synthetic-voice indicators are detected.")
        elif is_synthetic:
            parts.append("Elevated synthetic voice indicators detected; no confident enrolled match.")
        elif level == RiskLevel.MEDIUM:
            if has_speaker_match:
                parts.append(f"Audio resembles {name_str}, with moderate synthetic voice indicators under observation.")
            else:
                parts.append("Moderate synthetic voice indicators present.")
        elif decision == Decision.MONITOR:
            parts.append("Potential synthetic voice anomaly under observation.")
        else:
            if has_speaker_match:
                parts.append(f"Voice matches {name_str} with authentic voice characteristics.")
            else:
                parts.append("Voice appears authentic.")

        parts.append(f"Raw spoof score: {spoof.spoof_score:.1%}")

        if speaker is not None:
            status_text = "matches enrolled profile" if speaker.match else "does NOT match enrolled profile"
            parts.append(f"Speaker similarity: {speaker.speaker_similarity:.2f} ({status_text})")

        # Sustained alert when level is genuinely elevated
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and self._consecutive_elevated > 1:
            parts.append(
                f"Sustained alert: {self._consecutive_elevated} consecutive elevated windows."
            )
        elif self._consecutive_elevated > 0 and level == RiskLevel.LOW and decision == Decision.MONITOR:
            parts.append(
                f"Observing: {self._consecutive_elevated}/{settings.risk_escalation_count} elevated windows."
            )

        return " ".join(parts)

    def _generate_aggregate_reason(
        self,
        avg_spoof: float,
        max_spoof: float,
        avg_speaker: Optional[float],
        level: RiskLevel,
        total_segments: int,
        suspicious_segments: int,
        matched_speaker_name: Optional[str] = None,
    ) -> str:
        """Generate reason for aggregate (recorded) assessment."""
        parts = []
        is_synthetic = level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        has_speaker_match = avg_speaker is not None and avg_speaker >= settings.speaker_threshold_match
        name_str = matched_speaker_name or "enrolled speaker"

        if is_synthetic and has_speaker_match:
            parts.append(f"Audio resembles enrolled speaker {name_str}, but synthetic-voice indicators are detected.")
        elif is_synthetic:
            parts.append("Recording shows elevated synthetic voice indicators.")
        elif level == RiskLevel.MEDIUM:
            if has_speaker_match:
                parts.append(f"Recording resembles {name_str}, with moderate synthetic indicators in some segments.")
            else:
                parts.append("Recording has moderate synthetic indicators in some segments.")
        else:
            if has_speaker_match:
                parts.append(f"Recording matches enrolled speaker {name_str} with authentic speech characteristics.")
            else:
                parts.append("Recording appears to contain authentic speech.")

        parts.append(f"Average spoof score: {avg_spoof:.0%}, peak: {max_spoof:.0%}.")
        parts.append(f"{suspicious_segments}/{total_segments} segments flagged as suspicious.")

        if avg_speaker is not None:
            parts.append(f"Speaker similarity: {avg_speaker:.2f}.")

        return " ".join(parts)

