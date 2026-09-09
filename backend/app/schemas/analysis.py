"""
Pydantic schemas for VoiceShield analysis results, sessions, and alerts.
"""

from datetime import datetime
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
    spoof_score: float  # Probability of spoof (0-1)
    bonafide_score: float  # Probability of bonafide (0-1)
    raw_logit_spoof: float  # Raw logit before softmax
    raw_logit_bonafide: float  # Raw logit before softmax


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


class AnalysisResult(BaseModel):
    """Complete analysis result for a recording."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: AnalysisSource
    duration_seconds: float
    segments: list[SegmentResult]
    risk: RiskAssessment
    processing_time_ms: float


class AlertPayload(BaseModel):
    """Alert sent to the frontend."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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


class EnrollmentResponse(BaseModel):
    """Response after enrolling a speaker."""
    speaker_id: str
    speaker_name: str
    created_at: datetime
    message: str


class SpeakerProfile(BaseModel):
    """Stored speaker profile metadata."""
    speaker_id: str
    speaker_name: str
    created_at: datetime
    embedding_path: str
    sample_count: int = 1


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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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
