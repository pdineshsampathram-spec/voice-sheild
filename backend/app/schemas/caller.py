"""
VoiceShield — Twilio + ElevenLabs Caller Agent Schemas.

Data models for call sessions, lifecycle state, timeline events,
telemetry, and request/response payloads.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CallStatus(str, Enum):
    """Lifecycle states of a Twilio call."""
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RINGING = "RINGING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUSY = "BUSY"
    NO_ANSWER = "NO_ANSWER"
    CANCELED = "CANCELED"


class CallDirection(str, Enum):
    """Call direction — strictly OUTBOUND for trial accounts."""
    OUTBOUND = "OUTBOUND"


class MediaStreamStatus(str, Enum):
    """Separated status for media streaming.

    In accordance with system rules:
    'MEDIA STREAM IMPLEMENTED' != 'MEDIA STREAM PHYSICALLY VERIFIED'.
    Status remains UNPROVEN until real packets arrive during a physical call.
    """
    UNPROVEN = "UNPROVEN"
    CONNECTED = "CONNECTED"
    RECEIVING_PACKETS = "RECEIVING_PACKETS"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class CallTimelineEvent(BaseModel):
    """An immutable event in the call's chronological lifecycle."""
    timestamp: datetime = Field(default_factory=_utc_now)
    event_type: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioTelemetry(BaseModel):
    """Real-time audio telemetry from Twilio Media Streams (Milestone 2)."""
    packets_received: int = 0
    bytes_received: int = 0
    samples_decoded: int = 0
    rms: float = 0.0
    peak: float = 0.0
    percentage_zero: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    last_packet_timestamp: Optional[datetime] = None
    media_delivery_status: MediaStreamStatus = MediaStreamStatus.UNPROVEN


class CallSession(BaseModel):
    """Represents a single outbound AI call session."""
    call_id: str
    call_sid: Optional[str] = None
    destination: str
    direction: CallDirection = CallDirection.OUTBOUND
    status: CallStatus = CallStatus.CREATED
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: Optional[datetime] = None
    provider: str = "TWILIO_ELEVENLABS"
    agent_id: str
    greeting: Optional[str] = None
    max_duration_seconds: Optional[int] = None
    termination_requested: bool = False
    error_message: Optional[str] = None
    timeline: List[CallTimelineEvent] = Field(default_factory=list)
    telemetry: AudioTelemetry = Field(default_factory=AudioTelemetry)


class OutboundCallRequest(BaseModel):
    """Request payload to initiate an outbound AI call.

    Note: destination number is strictly enforced server-side from
    TWILIO_VERIFIED_DESTINATION_NUMBER. Any user-supplied destination
    is validated or ignored for safety.
    """
    agent_id: Optional[str] = None
    greeting: Optional[str] = None
    destination: Optional[str] = None
    max_duration_seconds: Optional[int] = None


class OutboundCallResponse(BaseModel):
    """Response returned upon dispatching an outbound call."""
    call_id: str
    call_sid: Optional[str] = None
    status: CallStatus
    destination: str
    message: str
    milestone: str = "MILESTONE_1_OUTBOUND"
