"""
Alert service for VoiceShield.

Generates structured alerts based on risk assessments.
Designed to be extended with SMS, email, push notifications.
"""

import logging
import random
from datetime import datetime
from typing import Optional

from app.schemas.analysis import (
    AlertPayload, RiskLevel, Decision, RiskAssessment
)

logger = logging.getLogger(__name__)

# Curated challenge phrases for verification
CHALLENGE_PHRASES = [
    "The blue train arrives at seven.",
    "My favorite color is bright orange.",
    "The weather forecast calls for sunshine.",
    "Please open the window on the left side.",
    "Three cats are sitting on the red fence.",
    "The morning coffee tastes especially good today.",
    "I left my umbrella near the front door.",
    "The library closes at exactly nine o'clock.",
    "Remember to water the plants this evening.",
    "The new bridge connects both sides of the river.",
]


class AlertService:
    """Generates and manages security alerts.

    For the prototype, alerts are sent via WebSocket to the frontend.
    Architecture supports extension to SMS, email, push notifications.
    """

    def __init__(self):
        self._alert_history: list[AlertPayload] = []

    def create_alert(self, risk: RiskAssessment) -> Optional[AlertPayload]:
        """Create an alert from a risk assessment if warranted.

        Only generates alerts for HIGH and CRITICAL risk levels.
        """
        if risk.risk_level == RiskLevel.LOW:
            return None

        if risk.risk_level == RiskLevel.MEDIUM:
            return AlertPayload(
                level=RiskLevel.MEDIUM,
                title="⚠ Monitoring — Moderate Indicators",
                message=f"Synthetic voice probability: {risk.spoof_score_avg:.0%}",
                reason=risk.reason,
                recommended_action=Decision.MONITOR,
                spoof_score=risk.spoof_score_avg,
                speaker_similarity=risk.speaker_similarity_avg,
            )

        if risk.risk_level == RiskLevel.HIGH:
            alert = AlertPayload(
                level=RiskLevel.HIGH,
                title="🔶 HIGH RISK — Possible Synthetic Voice",
                message=(
                    f"Synthetic voice probability: {risk.spoof_score_avg:.0%}. "
                    f"Verification recommended."
                ),
                reason=risk.reason,
                recommended_action=Decision.WARN,
                spoof_score=risk.spoof_score_avg,
                speaker_similarity=risk.speaker_similarity_avg,
            )
            self._alert_history.append(alert)
            logger.warning(f"HIGH RISK alert: {alert.message}")
            return alert

        # CRITICAL
        alert = AlertPayload(
            level=RiskLevel.CRITICAL,
            title="🔴 CRITICAL — Voice Impersonation Likely",
            message=(
                f"Synthetic voice probability: {risk.spoof_score_avg:.0%}. "
                f"Multiple indicators confirm synthetic/cloned speech. "
                f"Immediate verification required."
            ),
            reason=risk.reason,
            recommended_action=Decision.VERIFY,
            spoof_score=risk.spoof_score_avg,
            speaker_similarity=risk.speaker_similarity_avg,
        )
        self._alert_history.append(alert)
        logger.critical(f"CRITICAL alert: {alert.message}")
        return alert

    def generate_challenge(self) -> dict:
        """Generate a verification challenge phrase.

        Returns a dict with the challenge phrase and ID.
        """
        phrase = random.choice(CHALLENGE_PHRASES)
        import uuid
        challenge_id = str(uuid.uuid4())

        return {
            "challenge_id": challenge_id,
            "phrase": phrase,
            "instruction": f'Please repeat this phrase: "{phrase}"',
            "timeout_seconds": 30,
        }

    def get_recent_alerts(self, limit: int = 20) -> list[AlertPayload]:
        """Get recent alerts from the in-memory history."""
        return list(reversed(self._alert_history[-limit:]))

    def clear_history(self):
        """Clear alert history."""
        self._alert_history.clear()


# Singleton
alert_service = AlertService()
