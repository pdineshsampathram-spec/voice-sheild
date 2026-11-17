export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type Decision = "SAFE" | "MONITOR" | "WARN" | "VERIFY" | "BLOCK";
export type AnalysisSource = "live" | "recording";

export type MatchStatus =
  | "MATCH"
  | "NO_CONFIDENT_MATCH"
  | "NO_ENROLLED_PROFILES"
  | "INSUFFICIENT_EVIDENCE"
  | "UNAVAILABLE";

export interface SpeakerCandidate {
  speaker_id: string;
  speaker_name: string;
  similarity: number;
  status: MatchStatus;
}

export interface SpeakerIdentificationResult {
  best_match?: SpeakerCandidate | null;
  alternatives: SpeakerCandidate[];
  status: MatchStatus;
}

export interface VoiceAuthenticityResult {
  raw_margin?: number | null;
  calibrated_probability?: number | null;
  status: string;
  evidence: string;
  decision: string;
}

export interface SignalQuality {
  status: string;
  reason: string;
  rms: number;
  duration_seconds: number;
}

export interface SecurityVerdict {
  risk: RiskLevel;
  reason: string;
  risk_score: number;
  decision: Decision;
}

export interface SpoofResult {
  model: string;
  label: string;
  spoof_score: number;
  bonafide_score: number;
  raw_logit_spoof: number;
  raw_logit_bonafide: number;
  delta?: number | null;
}

export interface SpeakerResult {
  model: string;
  speaker_similarity: number;
  match: boolean;
  threshold_used: number;
  enrolled_speaker_id?: string;
}

export interface SegmentResult {
  segment_index: number;
  start_time: number;
  end_time: number;
  spoof: SpoofResult;
  speaker?: SpeakerResult;
}

export interface RiskAssessment {
  risk_score: number;
  risk_level: RiskLevel;
  decision: Decision;
  reason: string;
  spoof_score_avg: number;
  speaker_similarity_avg?: number;
  window_count: number;
  suspicious_window_count: number;
  temporal_consistency: number;
}

export interface RecordingReference {
  recording_id: string;
  original_filename: string;
  mime_type: string;
  duration_seconds: number;
  playback_available: boolean;
  playback_url: string;
  sha256: string;
}

export interface EnrollmentSampleReference {
  sample_id: string;
  original_filename: string;
  mime_type: string;
  duration_seconds: number;
  sha256: string;
  playback_available: boolean;
  playback_url: string;
  created_at: string;
}

export interface AnalysisResult {
  session_id: string;
  timestamp: string;
  source: AnalysisSource;
  duration_seconds: number;
  segments: SegmentResult[];
  risk: RiskAssessment;
  processing_time_ms: number;
  spoof?: VoiceAuthenticityResult | null;
  speaker?: SpeakerIdentificationResult | null;
  quality?: SignalQuality | null;
  verdict?: SecurityVerdict | null;
  recording?: RecordingReference | null;
}

export interface WindowResult {
  window_index: number;
  timestamp: string;
  spoof: SpoofResult;
  speaker?: SpeakerResult;
  risk_score: number;
  risk_level: RiskLevel;
  decision: Decision;
  reason: string;
}

export interface AlertPayload {
  alert_id: string;
  timestamp: string;
  level: RiskLevel;
  title: string;
  message: string;
  reason: string;
  recommended_action: Decision;
  spoof_score?: number;
  speaker_similarity?: number;
}

export interface ModelStatus {
  name: string;
  status: string;
  message?: string;
  load_time_ms?: number;
}

export interface SystemHealth {
  status: string;
  models: ModelStatus[];
  database: string;
  redis: string;
  uptime_seconds: number;
}

export interface SpeakerProfile {
  speaker_id: string;
  speaker_name: string;
  created_at: string;
  updated_at?: string;
  embedding_path: string;
  sample_count: number;
  quality_status?: string;
  samples?: EnrollmentSampleReference[];
}

export type LiveSpeakerStatus =
  | "MATCH"
  | "NO_CONFIDENT_MATCH"
  | "NO_ENROLLED_PROFILES"
  | "INSUFFICIENT_EVIDENCE"
  | "UNAVAILABLE";

export interface LiveSpeakerCandidate {
  speaker_id: string;
  speaker_name: string;
  similarity: number;
  status: string;
}

export interface SpeakerIdentityEvent {
  type: "speaker" | "speaker_identity";
  session_id?: string;
  status?: LiveSpeakerStatus;
  best_match?: {
    speaker_id: string;
    speaker_name: string;
    similarity: number;
  } | null;
  candidates?: LiveSpeakerCandidate[];
  profiles_checked?: number;
  speech_window_id?: string;
  updated_at?: string;
  // Backward compatibility fields
  speaker_similarity?: number | null;
  speaker_name?: string | null;
  enrolled_speaker_id?: string | null;
  match?: boolean | null;
  sample_start?: number;
  sample_end?: number;
  latency_ms?: number;
  embedding_dim?: number;
  calibration?: string;
  reason?: string | null;
}

export interface EnrollmentResponse {
  speaker_id: string;
  speaker_name: string;
  created_at: string;
  message: string;
  sample_count: number;
  quality_status: string;
}

export interface WebSocketMessage {
  type: string;
  data?: WindowResult;
  alert?: AlertPayload;
  session_id?: string;
  message?: string;
}
