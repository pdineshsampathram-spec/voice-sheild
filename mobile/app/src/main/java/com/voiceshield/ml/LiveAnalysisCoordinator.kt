/**
 * LiveAnalysisCoordinator — Orchestrates the full live analysis loop.
 *
 * AudioBuffer → VAD → AASIST-L → ECAPA → WindowResult → RiskEngine → AlertManager
 *
 * Runs on a dedicated coroutine scope tied to the call lifecycle.
 * Consumes AudioFrames from the AudioSourceAdapter and feeds complete
 * windows through the detection engine.
 */
package com.voiceshield.ml

import android.util.Log
import com.voiceshield.VoiceShieldApp
import com.voiceshield.audio.AudioBuffer
import com.voiceshield.audio.AudioNormalizer
import com.voiceshield.shared.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class LiveAnalysisCoordinator(
    private val engine: VoiceShieldDetectionEngine,
    private val audioSourceType: AudioSourceType = AudioSourceType.WEBRTC_DIRECT
) {
    companion object {
        private const val TAG = "VoiceShieldLiveAnalysis"
    }

    private val audioBuffer = AudioBuffer(
        windowSize = AasistLEngine.REQUIRED_SAMPLES, // 64600 samples
        overlapRatio = 0.5f
    )
    private val normalizer = AudioNormalizer()

    private val riskEngine = VoiceShieldApp.riskEngine
    private val alertManager = VoiceShieldApp.alertManager

    private var analysisScope: CoroutineScope? = null

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    private val _windowsProcessed = MutableStateFlow(0)
    val windowsProcessed: StateFlow<Int> = _windowsProcessed.asStateFlow()

    private val _diagnostics = MutableStateFlow(AudioDiagnostics())
    val diagnostics: StateFlow<AudioDiagnostics> = _diagnostics.asStateFlow()

    private val _pipelineTelemetry = MutableStateFlow(AcousticPipelineTelemetry())
    val pipelineTelemetry: StateFlow<AcousticPipelineTelemetry> = _pipelineTelemetry.asStateFlow()

    // Internal stage counters
    private var vadFramesCount = 0
    private var vadSpeechCount = 0
    private var lastVadProb = 0f
    private var lastVadMinProb = 0f
    private var lastVadMaxProb = 0f
    private var lastVadMeanProb = 0f
    private var lastVadFirstProb = 0f
    private var lastVadTotalFrames = 0
    private var lastSileroInputMetrics: AudioSignalMetrics? = null
    private var aasistInferencesCount = 0
    private var aasistFailuresCount = 0
    private var lastAasistScoreVal: Float? = null
    private var lastAasistLatencyVal = 0L
    private var ecapaInferencesCount = 0
    private var ecapaFailuresCount = 0
    private var lastEcapaSimVal: Float? = null
    private var lastEcapaLatencyVal = 0L
    private var evidenceWindowsCount = 0
    private var latestCaptureDiagnostics = AcousticCaptureDiagnostics()

    /**
     * Start the live analysis loop.
     * Must be called after the detection engine is initialized.
     */
    fun start() {
        if (_isAnalyzing.value) {
            Log.w(TAG, "Analysis already running")
            return
        }

        analysisScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        _isAnalyzing.value = true
        engine.reset()
        riskEngine.reset()
        audioBuffer.reset()
        normalizer.resetMetrics()

        vadFramesCount = 0
        vadSpeechCount = 0
        lastVadProb = 0f
        lastVadMinProb = 0f
        lastVadMaxProb = 0f
        lastVadMeanProb = 0f
        lastVadFirstProb = 0f
        lastVadTotalFrames = 0
        lastSileroInputMetrics = null
        aasistInferencesCount = 0
        aasistFailuresCount = 0
        lastAasistScoreVal = null
        lastAasistLatencyVal = 0L
        ecapaInferencesCount = 0
        ecapaFailuresCount = 0
        lastEcapaSimVal = null
        lastEcapaLatencyVal = 0L
        evidenceWindowsCount = 0

        updateTelemetryState(
            status = AcousticAnalysisStatus.INITIALIZING,
            detail = "Starting acoustic capture pipeline..."
        )

        Log.i(TAG, "Live analysis started (engine=${engine.getMode()}, audio=$audioSourceType)")
    }

    /**
     * Update telemetry from the audio capture adapter (Stage 1 & 2).
     */
    fun updateCaptureDiagnostics(captureDiag: AcousticCaptureDiagnostics) {
        latestCaptureDiagnostics = captureDiag

        val newStatus = when {
            !_isAnalyzing.value -> AcousticAnalysisStatus.OFF
            captureDiag.stopReason != null && captureDiag.stopReason != "NORMAL_STOP" -> AcousticAnalysisStatus.ERROR
            !captureDiag.audioRecordInitialized -> AcousticAnalysisStatus.INITIALIZING
            captureDiag.framesRead == 0L -> AcousticAnalysisStatus.INITIALIZING
            audioBuffer.windowsEmitted == 0 -> AcousticAnalysisStatus.BUFFERING
            vadSpeechCount == 0 -> AcousticAnalysisStatus.WAITING_FOR_SPEECH
            else -> AcousticAnalysisStatus.PROCESSING_SPEECH
        }

        val detail = when (newStatus) {
            AcousticAnalysisStatus.OFF -> "Acoustic analysis inactive"
            AcousticAnalysisStatus.ERROR -> "Error: ${captureDiag.stopReason}"
            AcousticAnalysisStatus.INITIALIZING ->
                if (!captureDiag.audioRecordInitialized) "Initializing AudioRecord..."
                else "Waiting for audio input (0 frames read)"
            AcousticAnalysisStatus.BUFFERING ->
                "Buffering audio (${audioBuffer.currentSize}/${audioBuffer.requiredSamples} samples)"
            AcousticAnalysisStatus.WAITING_FOR_SPEECH ->
                "Audio received. Waiting for speech (VAD: silence)"
            AcousticAnalysisStatus.PROCESSING_SPEECH ->
                "Processing speech (windows=${audioBuffer.windowsEmitted}, speech=$vadSpeechCount)"
            else -> "Active"
        }

        updateTelemetryState(newStatus, detail)
    }

    /**
     * Feed an audio frame from the audio adapter.
     * Thread-safe — can be called from any coroutine or callback.
     */
    fun onAudioFrame(frame: AudioFrame) {
        if (!_isAnalyzing.value) return

        // Update legacy diagnostics (metadata only — NO raw audio)
        val diag = _diagnostics.value
        _diagnostics.value = diag.copy(
            framesReceived = diag.framesReceived + 1,
            totalBytesReceived = diag.totalBytesReceived + (frame.samples.size * 4L),
            lastTimestamp = frame.timestamp,
            sampleRate = frame.sampleRate,
            channels = frame.channelCount,
            audioSource = frame.sourceType.name,
            isMonotonic = frame.timestamp >= diag.lastTimestamp
        )

        // Stage 3: Normalize to 16kHz mono
        val normalized = normalizer.normalize(frame)

        // Stage 4: Add to buffer and check for complete windows
        val windows = audioBuffer.addSamples(normalized.samples)

        // Update status if currently buffering
        if (audioBuffer.windowsEmitted == 0) {
            updateTelemetryState(
                status = AcousticAnalysisStatus.BUFFERING,
                detail = "Buffering audio (${audioBuffer.currentSize}/${audioBuffer.requiredSamples} samples)"
            )
        }

        for (windowSamples in windows) {
            val scope = analysisScope ?: return
            scope.launch {
                analyzeWindow(windowSamples, normalized.sourceType)
            }
        }
    }

    private suspend fun analyzeWindow(samples: FloatArray, sourceType: AudioSourceType) {
        try {
            val audioFrame = AudioFrame(
                samples = samples,
                sampleRate = AudioNormalizer.TARGET_SAMPLE_RATE,
                sourceType = sourceType
            )

            val startTime = System.currentTimeMillis()
            vadFramesCount++
            val result = engine.analyzeWindow(audioFrame)
            val elapsed = System.currentTimeMillis() - startTime

            if (result != null) {
                _windowsProcessed.value = _windowsProcessed.value + 1

                // Stage 5: VAD
                val vad = result.vad
                lastVadProb = vad.speechProbability
                lastVadMinProb = vad.minProb
                lastVadMaxProb = vad.maxProb
                lastVadMeanProb = vad.meanProb
                lastVadFirstProb = vad.firstProb
                lastVadTotalFrames = vad.totalFramesCount
                lastSileroInputMetrics = vad.inputMetrics
                if (vad.isSpeech) {
                    vadSpeechCount++
                }

                // Stage 6: AASIST-L
                val antiSpoof = result.antiSpoof
                if (antiSpoof != null) {
                    aasistInferencesCount++
                    lastAasistScoreVal = antiSpoof.spoofScore
                    lastAasistLatencyVal = elapsed
                } else {
                    val onDevice = engine as? OnDeviceDetectionEngine
                    val isAasistReady = onDevice?.let { it.getModelHealth()["AASIST-L"] == ModelStatus.READY } ?: true
                    if (isAasistReady) {
                        aasistFailuresCount++
                    }
                }

                // Stage 7: ECAPA-TDNN
                val speaker = result.speaker
                if (speaker != null) {
                    ecapaInferencesCount++
                    lastEcapaSimVal = speaker.similarity
                    lastEcapaLatencyVal = elapsed
                } else {
                    // Only count failure if ECAPA was ready and user was enrolled (eligible for verification)
                    val onDevice = engine as? OnDeviceDetectionEngine
                    val isEnrolled = onDevice?.getEnrollmentManager()?.isEnrolled() == true
                    val isEcapaReady = onDevice?.getEcapaEngine()?.isReady() == true
                    if (isEnrolled && isEcapaReady) {
                        ecapaFailuresCount++
                    }
                }

                // Stage 8: RiskEngine
                riskEngine.onWindowResult(result, audioSourceType)
                evidenceWindowsCount++

                // Check if alert is warranted
                alertManager.evaluateThreatState(riskEngine.threatState.value)

                val newStatus = if (vadSpeechCount > 0) {
                    AcousticAnalysisStatus.PROCESSING_SPEECH
                } else {
                    AcousticAnalysisStatus.WAITING_FOR_SPEECH
                }

                val detail = if (vadSpeechCount > 0) {
                    "Processing speech (windows=${_windowsProcessed.value}, speech=$vadSpeechCount)"
                } else {
                    "Audio received. Waiting for speech (VAD: silence)"
                }

                updateTelemetryState(newStatus, detail)

                Log.d(TAG, "Window ${result.windowIndex} analyzed: " +
                        "speech=$vadSpeechCount, " +
                        "spoof=${antiSpoof?.spoofScore?.let { String.format("%.4f", it) } ?: "N/A"}, " +
                        "threat=${riskEngine.threatState.value.level}")
            } else {
                // VAD evaluated window as silence.
                // DO NOT increment aasistFailuresCount! Silence is not an AASIST failure.
                val lastVad = (engine as? OnDeviceDetectionEngine)?.lastVadResult
                lastVadProb = lastVad?.speechProbability ?: 0f
                lastVadMinProb = lastVad?.minProb ?: 0f
                lastVadMaxProb = lastVad?.maxProb ?: 0f
                lastVadMeanProb = lastVad?.meanProb ?: 0f
                lastVadFirstProb = lastVad?.firstProb ?: 0f
                lastVadTotalFrames = lastVad?.totalFramesCount ?: 0
                lastSileroInputMetrics = lastVad?.inputMetrics
                updateTelemetryState(
                    status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
                    detail = "Audio received. Waiting for speech (VAD: silence)"
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Window analysis error: ${e.message}", e)
            updateTelemetryState(
                status = AcousticAnalysisStatus.ERROR,
                detail = "Window analysis error: ${e.message}"
            )
        }
    }

    private fun updateTelemetryState(status: AcousticAnalysisStatus, detail: String) {
        val diag = latestCaptureDiagnostics
        val micState = when {
            !diag.audioRecordInitialized || diag.framesRead == 0L -> "MIC_INACTIVE"
            diag.currentRms < 1e-5f || diag.rawMetrics?.classification == "SILENT_ZERO" -> "MIC_SILENT_ZERO"
            diag.currentRms < 0.005f -> "MIC_ACTIVE_LOW_AMPLITUDE"
            vadSpeechCount > 0 -> "MIC_STRONG_SPEECH_DETECTED"
            else -> "MIC_ACTIVE_NORMAL_SIGNAL"
        }

        // ── Signal Layer States (UI display heuristics ONLY) ──────────
        // These NEVER influence VAD, AASIST, RiskEngine, alerts, or detection.
        val rawShort = diag.rawShortMetrics
        val rawSignal = when {
            rawShort == null || rawShort.peakAbsShort == 0 -> "ZERO"
            rawShort.rmsShort < 50f -> "LOW"  // ~50/32768 ≈ 0.0015 float — descriptive only
            else -> "NONZERO"
        }
        val normalizedSignal = when {
            diag.currentRms < 1e-5f -> "ZERO"
            diag.currentRms < 0.005f -> "LOW"
            else -> "NONZERO"
        }
        val vadDecision = when {
            vadSpeechCount > 0 -> "SPEECH"
            vadFramesCount > 0 -> "NON_SPEECH"
            else -> "UNKNOWN"
        }
        val shortFloatCorrelation = when {
            rawShort == null -> "UNKNOWN"
            rawShort.rmsShort > 50f && rawShort.actualFloatRms > 1e-5f -> "MATCH_SIGNAL_ALIVE"
            rawShort.rmsShort < 1f && rawShort.actualFloatRms < 1e-5f -> "MATCH_BOTH_ZERO"
            rawShort.rmsShort > 50f && rawShort.actualFloatRms < 1e-5f -> "MISMATCH_CONVERSION_DEFECT"
            else -> "LOW_SIGNAL"
        }

        val telemetry = AcousticPipelineTelemetry(
            status = status,
            statusDetail = detail,
            stopReason = diag.stopReason,
            stopCaseCode = null, // Computed via diagnoseStopCase()

            // Stage 1
            audioRecordInitialized = diag.audioRecordInitialized,
            recordingState = diag.recordingState,
            selectedAudioSource = diag.selectedSource,
            framesRead = diag.framesRead,
            bytesRead = diag.bytesRead,
            readErrors = diag.readErrors,
            lastReadError = diag.lastReadError,
            currentRms = diag.currentRms,
            peakRms = diag.peakRms,
            silentFrames = diag.silentFrames,
            rawRecordMetrics = diag.rawMetrics,

            // Raw Short forensics
            rawShortMetrics = rawShort,

            // Stage 2
            framesDelivered = diag.framesDelivered,
            micActivityState = micState,

            // Stage 3
            normalizerFramesReceived = normalizer.framesReceived,
            normalizerSamplesProduced = normalizer.samplesProduced,
            normalizerInMetrics = normalizer.lastInputMetrics,
            normalizerOutMetrics = normalizer.lastOutputMetrics,
            normalizerGain = normalizer.normalizerGain,

            // Stage 4
            bufferCurrentSamples = audioBuffer.currentSize,
            bufferRequiredSamples = audioBuffer.requiredSamples,
            bufferFillPercentage = audioBuffer.fillPercentage,
            windowsEmitted = audioBuffer.windowsEmitted,
            bufferWindowMetrics = audioBuffer.lastWindowMetrics,

            // Stage 5
            vadFramesProcessed = vadFramesCount,
            vadSpeechFrames = vadSpeechCount,
            lastVadProbability = lastVadProb,
            vadMinProbability = lastVadMinProb,
            vadMaxProbability = lastVadMaxProb,
            vadMeanProbability = lastVadMeanProb,
            vadFirstProbability = lastVadFirstProb,
            vadTotalFramesInLastWindow = lastVadTotalFrames,
            sileroInputMetrics = lastSileroInputMetrics,

            // Stage 6
            aasistInferences = aasistInferencesCount,
            aasistFailures = aasistFailuresCount,
            lastAasistScore = lastAasistScoreVal,
            lastAasistLatencyMs = lastAasistLatencyVal,

            // Stage 7
            ecapaInferences = ecapaInferencesCount,
            ecapaFailures = ecapaFailuresCount,
            lastEcapaSimilarity = lastEcapaSimVal,
            lastEcapaLatencyMs = lastEcapaLatencyVal,

            // Stage 8
            evidenceWindows = evidenceWindowsCount,
            lastRiskState = riskEngine.threatState.value.level.name,

            // Signal Layer States (UI heuristics only)
            rawPcmSignalState = rawSignal,
            normalizedSignalState = normalizedSignal,
            vadDecisionState = vadDecision,
            shortFloatMatch = shortFloatCorrelation
        )

        _pipelineTelemetry.value = telemetry.copy(stopCaseCode = telemetry.diagnoseStopCase())
    }

    /**
     * Stop the analysis loop and clean up.
     */
    fun stop() {
        _isAnalyzing.value = false
        analysisScope?.cancel()
        analysisScope = null
        audioBuffer.reset()
        normalizer.resetMetrics()
        _windowsProcessed.value = 0
        updateTelemetryState(AcousticAnalysisStatus.OFF, "Acoustic analysis stopped")
        Log.i(TAG, "Live analysis stopped")
    }
}

/**
 * Debug-only audio diagnostics — metadata only, NO raw audio content.
 */
data class AudioDiagnostics(
    val audioSource: String = "UNKNOWN",
    val sampleRate: Int = 0,
    val channels: Int = 0,
    val framesReceived: Long = 0,
    val totalBytesReceived: Long = 0,
    val lastTimestamp: Long = 0,
    val isMonotonic: Boolean = true
)
