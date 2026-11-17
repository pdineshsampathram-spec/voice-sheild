/**
 * VoiceShield Model Contracts — Shared type definitions.
 *
 * These types are used by BOTH detection engines (PythonBackend + OnDevice)
 * to ensure consistent behavior and result schemas.
 */
package com.voiceshield.shared

import java.util.UUID

// ─── Audio Types ────────────────────────────────────────────────────

enum class AudioSourceType {
    /** We own the media stream (WebRTC/VoIP) — guaranteed remote audio. */
    WEBRTC_DIRECT,
    /** Verified direct cellular downlink stream from remote caller. */
    CELLULAR_DOWNLINK,
    /** Verified bidirectional cellular call stream (both parties). */
    CELLULAR_MIXED,
    /** Platform-specific supported path (OEM API, future Android feature). */
    PLATFORM_SUPPORTED,
    /**
     * Acoustic capture via device microphone during an active cellular call.
     * Captures whatever the microphone hears, which MAY include caller audio
     * from the earpiece/speaker. Acoustically degraded — NOT digital downlink.
     * Caller speech content is NOT guaranteed — depends on speaker mode,
     * ambient noise, and earpiece coupling.
     */
    ACOUSTIC_CAPTURE,
    /** No legitimate remote audio path exists. */
    UNAVAILABLE
}

enum class AudioCategory {
    REMOTE_DOWNLINK,
    LOCAL_UPLINK,
    BIDIRECTIONAL_CALL,
    NEAR_END_MIC,
    UNKNOWN
}

enum class RemoteAudioVerification {
    /** Reproducible evidence that captured stream contains remote caller side. */
    VERIFIED_TRUE,
    /** Audio was captured from a near-end microphone only (not remote caller audio). */
    UNVERIFIED_MIC_ONLY,
    /** Source failed to initialize, was permission-restricted, or produced zero frames. */
    FAILED_OR_RESTRICTED,
    /** SIM remote audio has not been physically tested yet. Platform-dependent. */
    NOT_YET_VERIFIED
}

data class AudioSourceProbeResult(
    val sourceId: Int,
    val sourceName: String,
    val category: AudioCategory,
    val initialized: Boolean,
    val recordingStarted: Boolean,
    val permissionGranted: Boolean,
    val sampleRate: Int = 16000,
    val channelConfig: Int = 1,
    val encoding: Int = 2, // AudioFormat.ENCODING_PCM_16BIT
    val framesRead: Long = 0,
    val bytesRead: Long = 0,
    val hasNonZeroPcm: Boolean = false,
    val peakRmsDb: Float? = null,
    val verification: RemoteAudioVerification = RemoteAudioVerification.FAILED_OR_RESTRICTED,
    val error: String? = null
)

data class CellularCapabilityReport(
    val timestamp: Long = System.currentTimeMillis(),
    val callState: String = "IDLE",
    val audioMode: Int = 0,
    val isSpeakerphoneOn: Boolean = false,
    val isBluetoothScoOn: Boolean = false,
    val availableDevices: List<String> = emptyList(),
    val testedSources: List<AudioSourceProbeResult> = emptyList(),
    val usableRemotePcm: RemoteAudioVerification = RemoteAudioVerification.NOT_YET_VERIFIED,
    val detectedRemoteSource: String? = null,
    val technicalLimitationReason: String? = null
)

data class TelecomAudioDiagnostics(
    val callState: String = "IDLE",
    val currentRoute: String = "EARPIECE",
    val supportedRoutes: List<String> = emptyList(),
    val isMuted: Boolean = false,
    val isMicrophoneMuted: Boolean = false,
    val audioManagerMode: Int = 0,
    val isSpeakerphoneOn: Boolean = false,
    val isBluetoothScoOn: Boolean = false,
    // Uplink-specific investigation fields
    val uplinkStatus: String = "UNVERIFIED",  // UNVERIFIED, VERIFIED, FAILED
    val uplinkNotes: String = ""
)

enum class LocalDisconnectReason {
    USER_END,
    THREAT_ALERT_END,
    USER_REJECT,
    UNKNOWN_LOCAL_PATH
}

/**
 * Captures DisconnectCause metadata from Android Telecom.
 * Used to diagnose why calls are dropping — whether initiated by
 * VoiceShield, remote party, carrier, or error.
 */
data class LastDisconnectInfo(
    /** DisconnectCause.getCode() — one of LOCAL, REMOTE, ERROR, etc. */
    val code: Int = -1,
    /** DisconnectCause.getLabel() */
    val label: String = "",
    /** DisconnectCause.getDescription() */
    val description: String = "",
    /** DisconnectCause.getReason() */
    val reason: String = "",
    /** True only when VoiceShield's disconnectCall() or rejectCall() was explicitly called */
    val localDisconnectRequested: Boolean = false,
    /** Categorized reason if local disconnect was requested */
    val localDisconnectReason: LocalDisconnectReason? = null,
    /** Whether endActiveCall() was invoked before disconnect */
    val endActiveCallInvoked: Boolean = false,
    /** Whether an AlertManager action triggered the disconnect */
    val alertActionTriggered: Boolean = false,
    /** Last known Telecom call state before disconnect */
    val lastTelecomState: Int = -1,
    /** Last known InCallService lifecycle state */
    val serviceLifecycleState: String = "ACTIVE",
    /** AI state at the time of disconnect */
    val aiState: String = "AUDIO_UNAVAILABLE",
    /** Timestamp of disconnect */
    val timestamp: Long = 0
)

data class AudioFrame(
    val samples: FloatArray,
    val sampleRate: Int = 16000,
    val timestamp: Long = System.currentTimeMillis(),
    val sourceType: AudioSourceType,
    val channelCount: Int = 1
) {
    val durationMs: Long get() = (samples.size * 1000L) / sampleRate
}

// ─── Model Status ───────────────────────────────────────────────────

enum class ModelStatus {
    READY,
    LOADING,
    UNAVAILABLE,
    ERROR,
    NOT_CALIBRATED,
    /** Model engine class and/or ONNX asset do not exist. Not a runtime failure. */
    NOT_IMPLEMENTED
}

/**
 * Granular forensic failure codes for on-device model initialization and inference.
 * Replaces generic UNAVAILABLE with identifiable root causes.
 */
enum class ModelFailureReason {
    NONE,
    NOT_IMPLEMENTED,
    NOT_LOADED_YET,
    ASSET_MISSING,
    ASSET_OPEN_FAILED,
    ONNX_RUNTIME_INIT_FAILED,
    MODEL_LOAD_FAILED,
    INPUT_CONTRACT_INVALID,
    OUTPUT_CONTRACT_INVALID,
    NATIVE_LIBRARY_LOAD_FAILED,
    OUT_OF_MEMORY,
    STARTUP_TEST_INFERENCE_FAILED,
    INFERENCE_FAILED,
    PREPROCESSING_FAILED,
    PERMISSION_OR_ENVIRONMENT_ERROR,
    UNKNOWN_ERROR
}

data class DetailedModelHealth(
    val name: String,
    val status: ModelStatus,
    val failureReason: ModelFailureReason = ModelFailureReason.NONE,
    val detailMessage: String? = null,
    val loadTimeMs: Long? = null,
    val lastInferenceMs: Long? = null
)

data class ModelHealth(
    val name: String,
    val status: ModelStatus,
    val loadTimeMs: Long? = null,
    val error: String? = null
)

// ─── Detection Results ──────────────────────────────────────────────

data class AntiSpoofResult(
    val modelName: String,           // "AASIST-L" or "Spectra-AASIST"
    val spoofScore: Float,           // 0.0 = genuine, 1.0 = spoof
    val bonafideScore: Float,        // 1.0 - spoofScore
    val rawLogits: FloatArray? = null,
    val label: String                // "ANTI-SPOOF MODEL SCORE" — not "% CLONED"
)

data class SpeakerResult(
    val modelName: String,           // "ECAPA-TDNN"
    val similarity: Float,           // cosine similarity
    val enrolledSpeakerId: String?,
    val match: Boolean?,             // null if no enrolled speaker
    val thresholdUsed: Float? = null  // null when CALIBRATION_NOT_AVAILABLE
)

data class VadResult(
    val speechProbability: Float,
    val isSpeech: Boolean,
    val firstProb: Float = 0f,
    val minProb: Float = 0f,
    val maxProb: Float = 0f,
    val meanProb: Float = 0f,
    val speechFramesCount: Int = 0,
    val totalFramesCount: Int = 0,
    val inputMetrics: AudioSignalMetrics? = null
)

data class WindowResult(
    val windowIndex: Int,
    val timestamp: Long,
    val vad: VadResult,
    val antiSpoof: AntiSpoofResult?,  // null if no speech detected
    val speaker: SpeakerResult?,      // null if no enrolled speaker
    val processingTimeMs: Long
)

// ─── Threat / Risk ──────────────────────────────────────────────────

enum class ThreatLevel {
    MONITORING,
    NORMAL,
    ELEVATED,
    HIGH_RISK,
    CRITICAL,
    INSUFFICIENT_EVIDENCE,
    MODEL_UNAVAILABLE,
    AUDIO_UNAVAILABLE
}

data class ThreatState(
    val level: ThreatLevel,
    val reason: String,
    val windowsAnalyzed: Int,
    val suspiciousWindows: Int,
    val avgSpoofScore: Float?,
    val avgSpeakerSimilarity: Float?,
    val audioSourceType: AudioSourceType,
    val modelAvailability: Map<String, ModelStatus>,
    val timestamp: Long = System.currentTimeMillis()
)

// ─── Evidence ───────────────────────────────────────────────────────

data class WindowEvidence(
    val windowIndex: Int,
    val timestamp: Long,
    val spoofScore: Float,
    val speakerSimilarity: Float? = null,
    val vadConfidence: Float,
    val modelAvailability: Map<String, ModelStatus>
)

// ─── Call Session ───────────────────────────────────────────────────

data class CallSession(
    val id: String = UUID.randomUUID().toString(),
    val startTime: Long = System.currentTimeMillis(),
    val phoneNumber: String? = null,
    val audioSourceType: AudioSourceType = AudioSourceType.UNAVAILABLE,
    val isActive: Boolean = true
)

// ─── Calibration ────────────────────────────────────────────────────

data class CalibrationArtifact(
    val modelName: String,
    val modelRevision: String,
    val datasetIdentifier: String,
    val evaluationDate: String,
    val sampleCount: Int,
    val operatingPoint: OperatingPoint
)

data class OperatingPoint(
    val threshold: Float,
    val eer: Float,          // equal error rate
    val farAtThreshold: Float?,
    val frrAtThreshold: Float?
)

// ─── Detection Engine Interface ─────────────────────────────────────

enum class EngineMode {
    /** Send audio to FastAPI backend for inference. */
    DEVELOPMENT,
    /** Run ONNX inference on-device. No audio leaves the phone. */
    PRIVACY
}

data class EngineStatus(
    val mode: EngineMode,
    val models: List<ModelHealth>,
    val calibrationAvailable: Boolean
)

// ─── Cellular Analysis State Machine ────────────────────────────────

enum class CellularAnalysisState {
    /** No audio stream is connected or available. */
    NO_AUDIO_PATH,
    /** An audio source was detected but caller-audio presence is not verified. */
    AUDIO_PATH_UNVERIFIED,
    /** Legitimate caller audio stream verified. */
    AUDIO_PATH_VERIFIED,
    /** Analysis active via acoustic capture (device microphone). */
    ACOUSTIC_CAPTURE_ACTIVE,
    /** Analysis pipeline actively processing audio. */
    ANALYSIS_ACTIVE,
    /** Audio is present but speech/vad duration is insufficient for evaluation. */
    INSUFFICIENT_EVIDENCE,
    /** Required model is not implemented or failed to load. */
    MODEL_UNAVAILABLE,
    /** Calibration data is not available. */
    CALIBRATION_UNAVAILABLE,
    /** Non-fatal analysis error occurred. Call remains active. */
    ANALYSIS_ERROR
}

// ─── Signal Metrics Contracts ───────────────────────────────────────

/**
 * Aggregate audio signal metrics computed over a window or frame.
 * Strict privacy: NO raw PCM samples are stored or exposed.
 */
data class AudioSignalMetrics(
    val sampleCount: Long = 0L,
    val sampleRate: Int = 16000,
    val channelCount: Int = 1,
    val minSample: Float = 0f,
    val maxSample: Float = 0f,
    val rms: Float = 0f,
    val peakAbs: Float = 0f,
    val peakDb: Float = -100f,
    val mean: Float = 0f,
    val stdDev: Float = 0f,
    val zeroOrNearZeroPercentage: Float = 0f,
    val nanCount: Int = 0,
    val infCount: Int = 0,
    val classification: String = "UNKNOWN" // SILENT_ZERO, VERY_LOW_AMPLITUDE, NORMAL_SIGNAL, MALFORMED_NAN_INF, POTENTIALLY_CLIPPED
)

/**
 * Raw Short-domain PCM metrics computed BEFORE any Short→Float conversion.
 * Proves whether zeros originate inside AudioRecord or inside our code.
 * Strict privacy: NO raw PCM samples are stored or exposed.
 */
data class RawShortPcmMetrics(
    val sampleCount: Int = 0,
    val minShort: Short = 0,
    val maxShort: Short = 0,
    val meanShort: Float = 0f,
    val rmsShort: Float = 0f,
    val peakAbsShort: Int = 0,       // max(abs(sample)) in Short domain
    val nonZeroCount: Int = 0,
    val zeroCount: Int = 0,
    val zeroPercent: Float = 100f,
    // Conversion verification: expected vs actual Float RMS
    val expectedFloatRms: Float = 0f, // rmsShort / 32768f
    val actualFloatRms: Float = 0f,
    val conversionError: Float = 0f   // |expected - actual|
)

/**
 * Compute aggregate signal metrics over a FloatArray without persisting audio.
 */
fun computeAudioSignalMetrics(
    samples: FloatArray,
    sampleRate: Int = 16000,
    channelCount: Int = 1
): AudioSignalMetrics {
    if (samples.isEmpty()) {
        return AudioSignalMetrics(sampleRate = sampleRate, channelCount = channelCount, classification = "EMPTY")
    }
    val count = samples.size
    var minVal = Float.MAX_VALUE
    var maxVal = -Float.MAX_VALUE
    var sum = 0.0
    var sumSq = 0.0
    var peakAbs = 0f
    var zeroNearZeroCount = 0
    var nanCount = 0
    var infCount = 0

    for (s in samples) {
        if (s.isNaN()) {
            nanCount++
            continue
        }
        if (s.isInfinite()) {
            infCount++
            continue
        }
        if (s < minVal) minVal = s
        if (s > maxVal) maxVal = s
        val absS = kotlin.math.abs(s)
        if (absS > peakAbs) peakAbs = absS
        if (absS < 1e-5f) zeroNearZeroCount++
        sum += s.toDouble()
        sumSq += s.toDouble() * s.toDouble()
    }

    val validCount = count - nanCount - infCount
    if (validCount <= 0) {
        return AudioSignalMetrics(
            sampleCount = count.toLong(),
            sampleRate = sampleRate,
            channelCount = channelCount,
            nanCount = nanCount,
            infCount = infCount,
            classification = "MALFORMED_NAN_INF"
        )
    }

    val mean = (sum / validCount).toFloat()
    val rms = kotlin.math.sqrt(sumSq / validCount).toFloat()
    val variance = kotlin.math.max(0.0, (sumSq / validCount) - (mean.toDouble() * mean.toDouble())).toFloat()
    val stdDev = kotlin.math.sqrt(variance)
    val peakDb = if (peakAbs > 0f) (20f * kotlin.math.log10(peakAbs.toDouble())).toFloat().coerceIn(-100f, 0f) else -100f
    val zeroPct = (zeroNearZeroCount.toFloat() / validCount) * 100f

    val classification = when {
        nanCount > 0 || infCount > 0 -> "MALFORMED_NAN_INF"
        peakAbs < 1e-5f || zeroPct >= 99.9f -> "SILENT_ZERO"
        rms < 0.005f -> "VERY_LOW_AMPLITUDE"
        peakAbs >= 0.999f -> "POTENTIALLY_CLIPPED"
        else -> "NORMAL_SIGNAL"
    }

    return AudioSignalMetrics(
        sampleCount = count.toLong(),
        sampleRate = sampleRate,
        channelCount = channelCount,
        minSample = if (minVal == Float.MAX_VALUE) 0f else minVal,
        maxSample = if (maxVal == -Float.MAX_VALUE) 0f else maxVal,
        rms = rms,
        peakAbs = peakAbs,
        peakDb = peakDb,
        mean = mean,
        stdDev = stdDev,
        zeroOrNearZeroPercentage = zeroPct,
        nanCount = nanCount,
        infCount = infCount,
        classification = classification
    )
}

// ─── Controlled Self-Test Result Contracts ──────────────────────────

data class SingleFixtureTestResult(
    val fixtureName: String,
    val sampleCount: Int,
    val sampleRate: Int,
    val durationSeconds: Float,
    val audioMetrics: AudioSignalMetrics,
    val minProb: Float,
    val maxProb: Float,
    val meanProb: Float,
    val speechFramesCount: Int,
    val totalFramesCount: Int,
    val executionTimeMs: Long,
    val errorMessage: String? = null
)

data class ControlledTestResult(
    val timestamp: Long = System.currentTimeMillis(),
    val silenceResult: SingleFixtureTestResult,
    val nonSpeechControlResult: SingleFixtureTestResult,
    val speechFixtureResult: SingleFixtureTestResult,
    val contractValid: Boolean,
    val speechDivergesFromSilence: Boolean,
    val verdict: String,
    val detailedNotes: String
)

// ─── Acoustic Pipeline Telemetry Contracts ──────────────────────────

enum class AcousticAnalysisStatus {
    OFF,
    INITIALIZING,
    RECEIVING_AUDIO,
    BUFFERING,
    WAITING_FOR_SPEECH,
    PROCESSING_SPEECH,
    ERROR
}

/**
 * Stage-by-stage non-audio telemetry for acoustic capture and live analysis pipeline.
 * NO raw audio PCM is ever stored, transmitted, or logged.
 */
data class AcousticPipelineTelemetry(
    val status: AcousticAnalysisStatus = AcousticAnalysisStatus.OFF,
    val statusDetail: String = "Inactive",
    val stopReason: String? = null,
    val stopCaseCode: String? = null,

    // Stage 1: AudioRecord Hardware
    val audioRecordInitialized: Boolean = false,
    val recordingState: String = "STOPPED",
    val selectedAudioSource: String = "MIC",
    val framesRead: Long = 0L,
    val bytesRead: Long = 0L,
    val readErrors: Long = 0L,
    val lastReadError: String? = null,
    val currentRms: Float = 0f,
    val peakRms: Float = 0f,
    val silentFrames: Int = 0,
    val rawRecordMetrics: AudioSignalMetrics? = null,

    // Raw Short-domain forensics (measured BEFORE Short→Float conversion)
    val rawShortMetrics: RawShortPcmMetrics? = null,

    // Stage 2: Adapter Delivery
    val framesDelivered: Long = 0L,
    val micActivityState: String = "MIC_INACTIVE", // MIC_INACTIVE, MIC_SILENT_ZERO, MIC_ACTIVE_LOW_AMPLITUDE, MIC_ACTIVE_NORMAL_SIGNAL

    // Stage 3: Normalizer
    val normalizerFramesReceived: Long = 0L,
    val normalizerSamplesProduced: Long = 0L,
    val normalizerInMetrics: AudioSignalMetrics? = null,
    val normalizerOutMetrics: AudioSignalMetrics? = null,
    val normalizerGain: Float = 1.0f, // Output RMS / Input RMS

    // Stage 4: Buffer (64,600 samples = ~4.04s for AASIST-L)
    val bufferCurrentSamples: Int = 0,
    val bufferRequiredSamples: Int = 64600,
    val bufferFillPercentage: Float = 0f,
    val windowsEmitted: Int = 0,
    val bufferWindowMetrics: AudioSignalMetrics? = null,

    // Stage 5: Silero VAD
    val vadFramesProcessed: Int = 0,
    val vadSpeechFrames: Int = 0,
    val lastVadProbability: Float = 0f,
    val vadMinProbability: Float = 0f,
    val vadMaxProbability: Float = 0f,
    val vadMeanProbability: Float = 0f,
    val vadFirstProbability: Float = 0f,
    val vadTotalFramesInLastWindow: Int = 0,
    val sileroInputMetrics: AudioSignalMetrics? = null,

    // Stage 6: AASIST-L
    val aasistInferences: Int = 0,
    val aasistFailures: Int = 0,
    val lastAasistScore: Float? = null,
    val lastAasistLatencyMs: Long = 0L,

    // Stage 7: ECAPA-TDNN
    val ecapaInferences: Int = 0,
    val ecapaFailures: Int = 0,
    val lastEcapaSimilarity: Float? = null,
    val lastEcapaLatencyMs: Long = 0L,

    // Stage 8: RiskEngine
    val evidenceWindows: Int = 0,
    val lastRiskState: String = "MONITORING",

    // Section 8 Controlled Self-Test Report
    val controlledTestResult: ControlledTestResult? = null,

    // ─── Signal Layer States (never conflated, UI display heuristics only) ──
    // These NEVER influence VAD, AASIST, RiskEngine, alerts, or detection decisions.
    val rawPcmSignalState: String = "UNKNOWN",      // ZERO / LOW / NONZERO
    val normalizedSignalState: String = "UNKNOWN",   // ZERO / LOW / NONZERO
    val vadDecisionState: String = "UNKNOWN",        // SPEECH / NON_SPEECH / UNKNOWN
    val shortFloatMatch: String = "UNKNOWN"           // MATCH_SIGNAL_ALIVE / MATCH_BOTH_ZERO / MISMATCH_CONVERSION_DEFECT
) {
    /**
     * Determines the exact stage where the audio analysis pipeline stopped (Case A through J).
     */
    fun diagnoseStopCase(): String {
        return when {
            status == AcousticAnalysisStatus.OFF -> "OFF: Acoustic mode not enabled"
            !audioRecordInitialized -> "CASE_A: AudioRecord failed to initialize"
            framesRead == 0L -> "CASE_B: AudioRecord initialized but framesRead = 0 (mic blocked/held by OS)"
            framesDelivered == 0L -> "CASE_C: framesRead > 0 but framesDelivered = 0 (adapter delivery failed)"
            normalizerSamplesProduced == 0L -> "CASE_D: framesDelivered > 0 but normalizer produced 0 samples"
            windowsEmitted == 0 -> "CASE_E: Normalizer producing samples but AudioBuffer filling ($bufferCurrentSamples/$bufferRequiredSamples)"
            vadFramesProcessed == 0 -> "CASE_F: Windows emitted but VAD processed zero"
            aasistInferences == 0 -> "CASE_G: VAD running but AASIST never ran (no speech detected yet, lastProb=${String.format("%.3f", lastVadProbability)})"
            ecapaInferences == 0 && aasistInferences > 0 -> "CASE_H: AASIST ran but ECAPA did not run (speaker not enrolled or ECAPA unready)"
            evidenceWindows == 0 && aasistInferences > 0 -> "CASE_I: Models ran but evidence not integrated into RiskEngine"
            else -> "CASE_J: Full pipeline operational (windows=$windowsEmitted, VAD=$vadSpeechFrames, AASIST=$aasistInferences)"
        }
    }
}

/**
 * Diagnostics captured directly by CellularAcousticAudioAdapter.
 */
data class AcousticCaptureDiagnostics(
    val isEnabled: Boolean = false,
    val isRunning: Boolean = false,
    val audioRecordInitialized: Boolean = false,
    val recordingState: String = "STOPPED",
    val selectedSource: String = "MIC",
    val framesRead: Long = 0L,
    val bytesRead: Long = 0L,
    val readErrors: Long = 0L,
    val lastReadError: String? = null,
    val framesDelivered: Long = 0L,
    val currentRms: Float = 0f,
    val peakRms: Float = 0f,
    val silentFrames: Int = 0,
    val stopReason: String? = null,
    val rawMetrics: AudioSignalMetrics? = null,
    val rawShortMetrics: RawShortPcmMetrics? = null
)

/**
 * Result from standalone diagnostic microphone capture (Experiment D).
 * Runs AudioRecord → raw Short metrics + Float metrics WITHOUT any ML pipeline.
 */
data class StandaloneMicDiagnosticResult(
    val timestamp: Long = System.currentTimeMillis(),
    val durationMs: Long = 0L,
    val framesRead: Int = 0,
    val totalSamplesRead: Int = 0,
    val rawShortMetrics: RawShortPcmMetrics = RawShortPcmMetrics(),
    val floatMetrics: AudioSignalMetrics = AudioSignalMetrics(),
    val shortFloatMatch: String = "UNKNOWN",
    val errorMessage: String? = null
)

