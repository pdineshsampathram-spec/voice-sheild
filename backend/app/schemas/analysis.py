"""
Pydantic schemas for VoiceShield analysis results, sessions, and alerts.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnalysisSource(str, Enum):
    LIVE = "live"
    RECORDING = "recording"


class Decision(str, Enum):
    SAFE = "SAFE"
    MONITOR = "MONITOR"
    WARN = "WARN"
    VERIFY = "VERIFY"
    BLOCK = "BLOCK"


class SpoofResult(BaseModel):
    """Result from the Spectra-AASIST anti-spoofing model."""
    model: str = "Spectra-AASIST"
    label: str  # "bonafide" or "spoof"
    spoof_score: float  # Score (0-1)
    bonafide_score: float  # Score (0-1)
    raw_logit_spoof: float  # Raw logit before softmax
    raw_logit_bonafide: float  # Raw logit before softmax
    delta: Optional[float] = None  # Raw logit margin: logit_spoof - logit_bonafide


class MatchStatus(str, Enum):
    MATCH = "MATCH"
    NO_CONFIDENT_MATCH = "NO_CONFIDENT_MATCH"
    NO_ENROLLED_PROFILES = "NO_ENROLLED_PROFILES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNAVAILABLE = "UNAVAILABLE"


class SpeakerCandidate(BaseModel):
    speaker_id: str
    speaker_name: str
    similarity: float  # Cosine similarity (-1.0 to 1.0)
    status: MatchStatus


class SpeakerIdentificationResult(BaseModel):
    best_match: Optional[SpeakerCandidate] = None
    alternatives: list[SpeakerCandidate] = Field(default_factory=list)
    status: MatchStatus = MatchStatus.UNAVAILABLE
    reason: Optional[str] = None


class VoiceAuthenticityResult(BaseModel):
    raw_margin: Optional[float] = None  # Delta: logit_spoof - logit_bonafide
    calibrated_probability: Optional[float] = None  # null if calibration unavailable
    status: str = "CALIBRATION_UNAVAILABLE"  # CALIBRATED | CALIBRATION_UNAVAILABLE | ABSTAIN
    evidence: str = "BONAFIDE"  # BONAFIDE | SPOOF | SUSPICIOUS | INSUFFICIENT
    decision: str = "SAFE"


class SignalQuality(BaseModel):
    status: str = "VALID"  # VALID | INSUFFICIENT
    reason: str = "Audio signal within nominal acoustic parameters"
    rms: float = 0.0
    duration_seconds: float = 0.0


class SecurityVerdict(BaseModel):
    risk: RiskLevel
    reason: str
    risk_score: float
    decision: Decision


class SpeakerResult(BaseModel):
    """Result from ECAPA-TDNN speaker verification."""
    model: str = "ECAPA-TDNN"
    speaker_similarity: float  # Cosine similarity score
    match: bool  # Whether similarity exceeds threshold
    threshold_used: float  # The threshold that was applied
    enrolled_speaker_id: Optional[str] = None


class SegmentResult(BaseModel):
    """Analysis result for a single audio segment."""
    segment_index: int
    start_time: float  # seconds
    end_time: float  # seconds
    spoof: SpoofResult
    speaker: Optional[SpeakerResult] = None


class WindowResult(BaseModel):
    """Result for a single analysis window (live mode)."""
    window_index: int
    timestamp: datetime
    spoof: SpoofResult
    speaker: Optional[SpeakerResult] = None
    risk_score: float
    risk_level: RiskLevel
    decision: Decision
    reason: str


class RiskAssessment(BaseModel):
    """Overall risk assessment combining all signals."""
    risk_score: float  # 0.0 - 1.0
    risk_level: RiskLevel
    decision: Decision
    reason: str
    spoof_score_avg: float
    speaker_similarity_avg: Optional[float] = None
    window_count: int
    suspicious_window_count: int
    temporal_consistency: float  # Variance of scores across windows


class RecordingReference(BaseModel):
    """Explicit lineage reference to an analyzed recording artifact."""
    recording_id: str
    original_filename: str
    mime_type: str
    duration_seconds: float
    playback_available: bool = True
    playback_url: str
    sha256: str


class EnrollmentSampleReference(BaseModel):
    """Explicit lineage reference to an enrollment audio sample."""
    sample_id: str
    original_filename: str
    mime_type: str
    duration_seconds: float
    sha256: str
    playback_available: bool = True
    playback_url: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalysisResult(BaseModel):
    """Complete analysis result for a recording."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: AnalysisSource
    duration_seconds: float
    segments: list[SegmentResult]
    risk: RiskAssessment
    processing_time_ms: float
    # Structured Part 10 Contract Fields
    spoof: Optional[VoiceAuthenticityResult] = None
    speaker: Optional[SpeakerIdentificationResult] = None
    quality: Optional[SignalQuality] = None
    verdict: Optional[SecurityVerdict] = None
    recording: Optional[RecordingReference] = None


class AlertPayload(BaseModel):
    """Alert sent to the frontend."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: RiskLevel
    title: str
    message: str
    reason: str
    recommended_action: Decision
    spoof_score: Optional[float] = None
    speaker_similarity: Optional[float] = None


class EnrollmentRequest(BaseModel):
    """Request to enroll a speaker profile."""
    speaker_name: str
    speaker_id: Optional[str] = None


class EnrollmentResponse(BaseModel):
    """Response after enrolling a speaker."""
    speaker_id: str
    speaker_name: str
    created_at: datetime
    message: str
    sample_count: int = 1
    quality_status: str = "VALID"


class SpeakerProfile(BaseModel):
    """Stored speaker profile metadata."""
    speaker_id: str
    speaker_name: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    embedding_path: str
    sample_count: int = 1
    quality_status: str = "VALID"
    samples: list[EnrollmentSampleReference] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """Summary of an analysis session."""
    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    source: AnalysisSource
    status: str
    final_risk_level: Optional[RiskLevel] = None
    total_windows: int = 0
    suspicious_windows: int = 0


class SecurityEvent(BaseModel):
    """A logged security event."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spoof_score: float
    speaker_similarity: Optional[float] = None
    risk_score: float
    risk_level: RiskLevel
    decision: Decision
    reason: str
    source: AnalysisSource
    metadata: Optional[dict] = None


class ModelStatus(BaseModel):
    """Health status of a single model."""
    name: str
    status: str  # "ONLINE", "LOADING", "ERROR", "UNAVAILABLE"
    message: Optional[str] = None
    load_time_ms: Optional[float] = None


class SystemHealth(BaseModel):
    """Overall system health."""
    status: str
    models: list[ModelStatus]
    database: str  # "ONLINE", "OFFLINE", "UNAVAILABLE"
    redis: str  # "ONLINE", "OFFLINE", "UNAVAILABLE"
    uptime_seconds: float


class LiveCommand(BaseModel):
    """Commands sent from frontend over WebSocket."""
    action: str  # "start", "stop", "verify_challenge", "dismiss", "block"
    data: Optional[dict] = None


class LiveSpeakerStatus(str, Enum):
    MATCH = "MATCH"
    NO_CONFIDENT_MATCH = "NO_CONFIDENT_MATCH"
    NO_ENROLLED_PROFILES = "NO_ENROLLED_PROFILES"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNAVAILABLE = "UNAVAILABLE"


class LiveSpeakerBestMatch(BaseModel):
    speaker_id: str
    speaker_name: str
    similarity: float


class LiveSpeakerCandidate(BaseModel):
    speaker_id: str
    speaker_name: str
    similarity: float
    status: str


class LiveSpeakerPayload(BaseModel):
    """Event contract for live speaker identity telemetry (Part 17)."""
    type: str = "speaker_identity"
    session_id: str
    status: LiveSpeakerStatus
    best_match: Optional[LiveSpeakerBestMatch] = None
    candidates: list[LiveSpeakerCandidate] = Field(default_factory=list)
    profiles_checked: int = 0
    speech_window_id: str = ""
    updated_at: str
    reason: Optional[str] = None
