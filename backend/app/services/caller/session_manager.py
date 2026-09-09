"""
VoiceShield — Call Session Manager.

Thread-safe registry for call sessions, lifecycle state tracking,
and chronological event timelines.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict
from app.schemas.caller import (
    CallSession, CallStatus, CallDirection,
    CallTimelineEvent, AudioTelemetry, MediaStreamStatus
)

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CallSessionManager:
    """Thread-safe manager for active and historical call sessions."""

    def __init__(self):
        self._sessions: Dict[str, CallSession] = {}
        self._sid_to_id: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        call_id: str,
        destination: str,
        agent_id: str,
        greeting: Optional[str] = None,
        max_duration_seconds: Optional[int] = None,
    ) -> CallSession:
        """Create a new CallSession in CREATED state."""
        with self._lock:
            session = CallSession(
                call_id=call_id,
                destination=destination,
                direction=CallDirection.OUTBOUND,
                status=CallStatus.CREATED,
                started_at=_now(),
                agent_id=agent_id,
                greeting=greeting,
                max_duration_seconds=max_duration_seconds,
                timeline=[
                    CallTimelineEvent(
                        timestamp=_now(),
                        event_type="CALL_CREATED",
                        description=f"Outbound call session created for verified destination {destination}",
                        metadata={"agent_id": agent_id},
                    )
                ],
                telemetry=AudioTelemetry(),
            )
            self._sessions[call_id] = session
            return session

    def attach_call_sid(self, call_id: str, call_sid: str) -> Optional[CallSession]:
        """Attach Twilio Call SID to the session once dispatched."""
        with self._lock:
            session = self._sessions.get(call_id)
            if not session:
                return None
            session.call_sid = call_sid
            self._sid_to_id[call_sid] = call_id
            session.timeline.append(
                CallTimelineEvent(
                    timestamp=_now(),
                    event_type="TWILIO_DISPATCHED",
                    description=f"Twilio call created with SID {call_sid}",
                    metadata={"call_sid": call_sid},
                )
            )
            return session

    def get_session(self, call_id: str) -> Optional[CallSession]:
        """Retrieve a session by its internal call_id."""
        with self._lock:
            return self._sessions.get(call_id)

    def get_session_by_sid(self, call_sid: str) -> Optional[CallSession]:
        """Retrieve a session by its Twilio Call SID."""
        with self._lock:
            call_id = self._sid_to_id.get(call_sid)
            if not call_id:
                return None
            return self._sessions.get(call_id)

    def list_sessions(self) -> List[CallSession]:
        """Return all call sessions sorted by start time descending."""
        with self._lock:
            return sorted(
                list(self._sessions.values()),
                key=lambda s: s.started_at,
                reverse=True,
            )

    def record_timeline_event(
        self,
        call_id: str,
        event_type: str,
        description: str,
        metadata: Optional[dict] = None,
    ):
        """Append an event to the session's timeline."""
        with self._lock:
            session = self._sessions.get(call_id)
            if session:
                session.timeline.append(
                    CallTimelineEvent(
                        timestamp=_now(),
                        event_type=event_type,
                        description=description,
                        metadata=metadata or {},
                    )
                )

    def mark_termination_requested(self, call_id: str) -> Optional[CallSession]:
        """Mark that a hangup/termination has been requested.

        IMPORTANT (Reviewer Correction B):
        Does NOT prematurely change status to COMPLETED. The terminal state
        must be confirmed by Twilio's actual status callback webhook.
        """
        with self._lock:
            session = self._sessions.get(call_id)
            if not session:
                return None
            session.termination_requested = True
            session.timeline.append(
                CallTimelineEvent(
                    timestamp=_now(),
                    event_type="TERMINATION_REQUESTED",
                    description="Termination requested from VoiceShield backend; awaiting Twilio callback",
                    metadata={"call_sid": session.call_sid},
                )
            )
            return session

    def update_from_twilio_callback(
        self,
        call_sid: str,
        twilio_status: str,
        duration: Optional[int] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[CallSession]:
        """Update session state based on real status callback from Twilio."""
        with self._lock:
            call_id = self._sid_to_id.get(call_sid)
            if not call_id:
                logger.warning(f"Twilio callback received for unknown call_sid: {call_sid}")
                return None

            session = self._sessions.get(call_id)
            if not session:
                return None

            status_normalized = twilio_status.lower().strip()
            old_status = session.status

            if status_normalized == "queued":
                session.status = CallStatus.QUEUED
            elif status_normalized == "ringing":
                session.status = CallStatus.RINGING
            elif status_normalized == "in-progress":
                session.status = CallStatus.IN_PROGRESS
            elif status_normalized == "completed":
                session.status = CallStatus.COMPLETED
                session.ended_at = _now()
            elif status_normalized == "busy":
                session.status = CallStatus.BUSY
                session.ended_at = _now()
            elif status_normalized == "no-answer":
                session.status = CallStatus.NO_ANSWER
                session.ended_at = _now()
            elif status_normalized == "failed":
                session.status = CallStatus.FAILED
                session.ended_at = _now()
                if error_message or error_code:
                    session.error_message = f"Twilio Error {error_code}: {error_message}"
            elif status_normalized == "canceled":
                session.status = CallStatus.CANCELED
                session.ended_at = _now()

            session.timeline.append(
                CallTimelineEvent(
                    timestamp=_now(),
                    event_type=f"TWILIO_STATUS_{session.status.value}",
                    description=f"Twilio reported status change: {old_status.value} -> {session.status.value}",
                    metadata={
                        "twilio_status": twilio_status,
                        "duration_seconds": duration,
                        "error_code": error_code,
                        "error_message": error_message,
                    },
                )
            )

            logger.info(
                f"Session {session.call_id} (SID {call_sid}) status updated to {session.status.value}"
            )
            return session

    def update_telemetry(
        self,
        call_id: str,
        bytes_count: int,
        samples_count: int,
        rms: float,
        peak: float,
        percentage_zero: float,
        nan_count: int,
        inf_count: int,
        media_status: MediaStreamStatus,
    ):
        """Update live audio telemetry (Milestone 2)."""
        with self._lock:
            session = self._sessions.get(call_id)
            if not session:
                return
            t = session.telemetry
            t.packets_received += 1
            t.bytes_received += bytes_count
            t.samples_decoded += samples_count
            t.rms = round(rms, 4)
            t.peak = round(peak, 4)
            t.percentage_zero = round(percentage_zero, 2)
            t.nan_count += nan_count
            t.inf_count += inf_count
            t.last_packet_timestamp = _now()
            t.media_delivery_status = media_status


# Global singleton manager
call_session_manager = CallSessionManager()
