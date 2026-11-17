/**
 * PostCallContracts — Data types for post-call forensic analysis.
 *
 * This is POST-CALL ANALYSIS of user-selected OEM call recordings.
 * NOT real-time cellular audio interception.
 *
 * Privacy: raw audio never persisted, uploaded, or logged.
 * Recording is user-owned — never modified, deleted, or copied.
 */
package com.voiceshield.shared

// ─── Recording Import ───────────────────────────────────────────────

enum class RecordingStatus {
    VERIFIED,
    INVALID,
    UNREADABLE,
    PERMISSION_DENIED,
    UNSUPPORTED_FORMAT
}

/**
 * Explicit forensic lifecycle states for post-call recording discovery and analysis.
 */
enum class PostCallRecordingState {
    RECORDING_FOUND,
    RECORDING_NOT_FOUND,
    RECORDING_NOT_ACCESSIBLE,
    RECORDING_AMBIGUOUS,
    RECORDING_INVALID,
    ANALYSIS_RUNNING,
    ANALYSIS_COMPLETE
}

data class RecordingValidation(
    val status: RecordingStatus,
    val uri: String?,                  // SAF content:// Uri — primary reference
    val displayName: String?,          // from ContentResolver DISPLAY_NAME
    val format: String? = null,        // e.g. "audio/mpeg"
    val codec: String? = null,         // e.g. "mp3", "aac"
    val sampleRate: Int = 0,
    val channelCount: Int = 0,
    val durationMs: Long = 0,
    val decodedSampleCount: Long = 0,
    val fileSizeBytes: Long = 0,
    val errorMessage: String? = null
)

// ─── Quality Metrics (informational, NOT auto-reject) ───────────────

data class RecordingQualityMetrics(
    val durationMs: Long = 0,
    val rms: Float = 0f,
    val peak: Float = 0f,
    val clippingPercent: Float = 0f,
    val silencePercent: Float = 0f,
    val nanCount: Int = 0,
    val infCount: Int = 0,
    val sampleCount: Long = 0
)

// ─── Speech Availability (separate from recording validity) ─────────

data class SpeechAvailability(
    val hasSpeech: Boolean = false,
    val totalSpeechMs: Long = 0,
    val speechSegmentCount: Int = 0,
    val speechNonSpeechRatio: Float = 0f,
    val sufficientForAnalysis: Boolean = false,
    val insufficiencyReason: String? = null
)

// ─── Speech Segments ────────────────────────────────────────────────

data class SpeechSegment(
    val startMs: Long,
    val endMs: Long,
    val durationMs: Long,
    val avgVadProb: Float,
    /** "SPEAKER_A", "SPEAKER_B", or null. NEVER "CALLER" or "USER". */
    val speakerLabel: String? = null
)

// ─── Speaker Separation ─────────────────────────────────────────────

/**
 * Speaker separation confidence.
 * ECAPA is an embedding model used for experimental speaker clustering,
 * NOT a proven diarization engine.
 */
enum class SpeakerSeparationStatus {
    /** Two clusters produced and separation quality checks passed. NOT calibrated confidence. */
    SEPARATED,
    /** Clusters produced but not validated/calibrated. */
    PROVISIONAL,
    /** Ambiguous clustering — cannot reliably distinguish speakers. */
    UNCERTAIN,
    /** Only one cluster found in the recording. */
    SINGLE_SPEAKER,
    /** Clustering encountered an error. */
    FAILED
}

// ─── Per-Window Evidence ────────────────────────────────────────────

data class PostCallWindowEvidence(
    val windowIndex: Int,
    val startMs: Long,
    val endMs: Long,
    /** Starting sample index in recording at 16kHz. */
    val sampleStart: Long = 0L,
    /** Ending sample index in recording at 16kHz. */
    val sampleEnd: Long = 0L,
    /** "SPEAKER_A", "SPEAKER_B", or null. NEVER "CALLER"/"USER". */
    val speakerLabel: String? = null,
    /** True if this window contains audio from both speakers (mixed). */
    val isMixedSpeakerWindow: Boolean = false,
    val vadProb: Float = 0f,
    val aasistScore: Float? = null,
    val aasistStatus: String = "PENDING",  // OK, SKIPPED, ERROR, NO_SPEECH, PENDING
    val ecapaSimilarity: Float? = null,
    val ecapaStatus: String = "PENDING",   // OK, SKIPPED, ERROR, NO_ENROLLMENT, PENDING
    val qualityStatus: String = "UNKNOWN"  // GOOD, LOW_QUALITY, INSUFFICIENT
)

// ─── Analysis Outcome ───────────────────────────────────────────────

enum class PostCallAnalysisOutcome {
    /** Recording exists but contains no usable audio data. */
    NO_USABLE_AUDIO,
    /** No speech detected in the recording. */
    NO_SPEECH,
    /** Speech detected but not enough for reliable analysis. */
    INSUFFICIENT_SPEECH,
    /** Analyzed speech shows no spoof-like evidence. */
    NORMAL_SPEECH_EVIDENCE,
    /** Analyzed speech shows suspicious synthetic patterns. */
    SUSPICIOUS_SYNTHETIC_SPEECH_EVIDENCE,
    /** Speaker identity does not match enrolled profile. */
    SPEAKER_IDENTITY_MISMATCH_EVIDENCE,
    /** Evidence is conflicting, insufficient, or speaker separation uncertain. */
    INCONCLUSIVE,
    /** An error occurred during analysis. */
    ANALYSIS_ERROR,
    /** Analysis has not started yet. */
    NOT_STARTED,
    /** Analysis is in progress. */
    IN_PROGRESS
}

// ─── Analysis Session ───────────────────────────────────────────────

data class PostCallAnalysisSession(
    val id: String = java.util.UUID.randomUUID().toString(),
    /** Associated call session ID, if correlation was established. */
    val callSessionId: String? = null,
    /** SAF content:// Uri string — primary recording reference. */
    val recordingUri: String? = null,
    /** User-friendly filename from ContentResolver. */
    val recordingDisplayName: String? = null,
    val recordingValidation: RecordingValidation? = null,
    val qualityMetrics: RecordingQualityMetrics? = null,
    val speechAvailability: SpeechAvailability? = null,
    val speechSegments: List<SpeechSegment> = emptyList(),
    val windowEvidence: List<PostCallWindowEvidence> = emptyList(),
    val speakerSeparationStatus: SpeakerSeparationStatus = SpeakerSeparationStatus.UNCERTAIN,
    val outcome: PostCallAnalysisOutcome = PostCallAnalysisOutcome.NOT_STARTED,
    val summaryText: String = "",
    val startedAt: Long = 0,
    val completedAt: Long? = null,
    val errorMessage: String? = null,
    /** Progress 0.0–1.0 for UI display during analysis. */
    val progress: Float = 0f
)

// ─── Analysis Error Codes ───────────────────────────────────────────

enum class PostCallError {
    RECORDING_NOT_FOUND,
    RECORDING_PERMISSION_DENIED,
    RECORDING_DECODE_FAILED,
    UNSUPPORTED_FORMAT,
    INVALID_SAMPLE_RATE,
    INSUFFICIENT_SPEECH,
    MODEL_UNAVAILABLE,
    SPEAKER_SEPARATION_FAILED,
    ANALYSIS_TIMEOUT,
    OUT_OF_MEMORY,
    UNKNOWN_ANALYSIS_ERROR
}
