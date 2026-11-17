/**
 * OnDeviceDetectionEngine — Privacy Mode detection engine.
 *
 * Runs full AI pipeline on-device using ONNX Runtime:
 *   AudioFrame → Silero VAD → AASIST-L → ECAPA-TDNN → WindowResult
 *
 * No audio leaves the device. All inference is local.
 * Raw audio is never logged, stored, or transmitted.
 *
 * CRITICAL SAFETY RULES:
 * 1. Any AI exception is caught — NEVER affects the active call.
 * 2. ECAPA is optional — pipeline works without it (spoof-only evidence).
 * 3. Speaker comparison requires active enrollment — no enrollment = no speaker result.
 * 4. All thresholds are EXPERIMENTAL until calibrated.
 */
package com.voiceshield.ml

import android.content.Context
import android.util.Log
import com.voiceshield.shared.*

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class OnDeviceDetectionEngine(
    private val context: Context
) : VoiceShieldDetectionEngine {

    companion object {
        private const val TAG = "VoiceShieldOnDevice"
    }

    private val vadEngine = SileroVadEngine(context)
    private val aasistEngine = AasistLEngine(context)
    private val ecapaEngine = EcapaTdnnEngine(context)
    private val enrollmentManager = SpeakerEnrollmentManager(context, ecapaEngine)

    private val _detailedModelHealth = MutableStateFlow<Map<String, DetailedModelHealth>>(emptyMap())
    val detailedModelHealth: StateFlow<Map<String, DetailedModelHealth>> = _detailedModelHealth.asStateFlow()
    init {
        updateHealth()
    }

    private fun updateHealth() {
        _detailedModelHealth.value = mapOf(
            "Silero VAD" to vadEngine.getDetailedHealth(),
            "AASIST-L" to aasistEngine.getDetailedHealth(),
            "ECAPA-TDNN" to ecapaEngine.getDetailedHealth(),
            "Calibration" to DetailedModelHealth(
                name = "Calibration",
                status = ModelStatus.NOT_CALIBRATED,
                failureReason = ModelFailureReason.NONE,
                detailMessage = "Operating points uncalibrated"
            )
        )
    }

    private var windowIndex = 0
    private var modelsInitialized = false

    // ─── Per-stage timing (metadata only) ────────────────────────────
    private var lastVadMs: Long = 0
    private var lastAasistMs: Long = 0
    private var lastEcapaMs: Long = 0

    override fun getMode(): EngineMode = EngineMode.PRIVACY

    override fun isCalibrated(): Boolean = false // CALIBRATION_NOT_AVAILABLE

    fun getDetailedModelHealth(): Map<String, DetailedModelHealth> {
        updateHealth()
        return _detailedModelHealth.value
    }

    override fun getModelHealth(): Map<String, ModelStatus> {
        return getDetailedModelHealth().mapValues { it.value.status }
    }

    /**
     * Get the speaker enrollment manager for enrollment operations.
     */
    fun getEnrollmentManager(): SpeakerEnrollmentManager = enrollmentManager

    /**
     * Get the ECAPA engine for direct access (e.g., enrollment embedding extraction).
     */
    fun getEcapaEngine(): EcapaTdnnEngine = ecapaEngine

    /**
     * Get the AASIST-L anti-spoof engine for direct scoring.
     */
    fun getAasistEngine(): AasistLEngine = aasistEngine

    /**
     * Get the Silero VAD engine for direct health or diagnostic access.
     */
    fun getVadEngine(): SileroVadEngine = vadEngine

    var lastVadResult: VadResult? = null
        private set

    /**
     * Get per-stage pipeline latency from the last analysis window.
     */
    fun getLastLatency(): PipelineLatency = PipelineLatency(
        vadMs = lastVadMs,
        aasistMs = lastAasistMs,
        ecapaMs = lastEcapaMs,
        totalMs = lastVadMs + lastAasistMs + lastEcapaMs
    )

    override suspend fun initialize(): EngineStatus = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
        if (modelsInitialized && vadEngine.isReady() && aasistEngine.isReady() && ecapaEngine.isReady()) {
            Log.d(TAG, "On-device detection engine models already initialized.")
            return@withContext EngineStatus(
                mode = EngineMode.PRIVACY,
                models = getModelHealth().map { (name, status) ->
                    ModelHealth(name = name, status = status)
                },
                calibrationAvailable = false
            )
        }

        Log.i(TAG, "Initializing on-device detection engine (independent load)...")

        // 1. Silero VAD — load independently
        val vadLoaded = try {
            vadEngine.load()
        } catch (e: Exception) {
            Log.e(TAG, "Silero VAD load exception: ${e.message}", e)
            false
        }
        updateHealth()

        // 2. AASIST-L — load independently (failure does not block VAD or ECAPA)
        val aasistLoaded = try {
            aasistEngine.load()
        } catch (e: Exception) {
            Log.e(TAG, "AASIST-L load exception: ${e.message}", e)
            false
        }
        updateHealth()

        // 3. ECAPA-TDNN — load independently
        val ecapaLoaded = try {
            ecapaEngine.load()
        } catch (e: Exception) {
            Log.w(TAG, "ECAPA-TDNN load exception: ${e.message}", e)
            false
        }
        updateHealth()

        modelsInitialized = vadLoaded || aasistLoaded || ecapaLoaded

        Log.i(TAG, "On-device engine initialization complete: " +
                "VAD=$vadLoaded, AASIST-L=$aasistLoaded, ECAPA=$ecapaLoaded")

        EngineStatus(
            mode = EngineMode.PRIVACY,
            models = getModelHealth().map { (name, status) ->
                ModelHealth(name = name, status = status)
            },
            calibrationAvailable = false
        )
    }

    /**
     * Analyze one audio window through the full on-device pipeline.
     *
     * Pipeline:
     * 1. Silero VAD — detect speech
     * 2. AASIST-L — anti-spoof inference (only on speech)
     * 3. ECAPA-TDNN — speaker verification (only when enrolled + model ready)
     *
     * Returns null if VAD detects no speech.
     *
     * SAFETY: All model exceptions are caught. A model failure NEVER
     * propagates to the call layer or disconnects the call.
     */
    override suspend fun analyzeWindow(audio: AudioFrame): WindowResult? {
        val startTime = System.nanoTime()

        // Step 1: VAD
        val vadStart = System.nanoTime()
        val vadResult = vadEngine.process(audio.samples, audio.sampleRate)
        lastVadMs = (System.nanoTime() - vadStart) / 1_000_000
        lastVadResult = vadResult

        if (!vadResult.isSpeech) {
            Log.d(TAG, "Window ${windowIndex}: No speech detected " +
                    "(prob=${String.format("%.3f", vadResult.speechProbability)})")
            windowIndex++
            return null // Don't waste compute on silence
        }

        // Step 2: Anti-spoof inference
        var antiSpoofResult: AntiSpoofResult? = null
        val aasistStart = System.nanoTime()
        if (aasistEngine.isReady()) {
            try {
                // AASIST-L requires exactly 64600 samples
                val window = if (audio.samples.size == AasistLEngine.REQUIRED_SAMPLES) {
                    audio.samples
                } else if (audio.samples.size > AasistLEngine.REQUIRED_SAMPLES) {
                    // Take the last 64600 samples
                    audio.samples.copyOfRange(
                        audio.samples.size - AasistLEngine.REQUIRED_SAMPLES,
                        audio.samples.size
                    )
                } else {
                    // Pad with zeros at the beginning
                    FloatArray(AasistLEngine.REQUIRED_SAMPLES).also {
                        audio.samples.copyInto(it, AasistLEngine.REQUIRED_SAMPLES - audio.samples.size)
                    }
                }

                antiSpoofResult = aasistEngine.score(window)
            } catch (e: Exception) {
                // SAFETY: AASIST failure does NOT affect the call
                Log.e(TAG, "AASIST-L inference error (non-fatal): ${e.message}", e)
            }
        }
        lastAasistMs = (System.nanoTime() - aasistStart) / 1_000_000

        // Step 3: Speaker verification (ECAPA-TDNN)
        var speakerResult: SpeakerResult? = null
        val ecapaStart = System.nanoTime()
        if (ecapaEngine.isReady() && enrollmentManager.isEnrolled()) {
            try {
                // ECAPA needs enough audio for meaningful embedding
                if (audio.samples.size >= EcapaTdnnEngine.MIN_AUDIO_SAMPLES) {
                    val embedding = ecapaEngine.extractEmbedding(audio.samples)
                    val reference = enrollmentManager.getReference()
                    
                    if (reference != null) {
                        val similarity = ecapaEngine.compareSpeakers(embedding, reference)
                        speakerResult = ecapaEngine.createSpeakerResult(
                            similarity = similarity,
                            enrolledSpeakerId = enrollmentManager.getSpeakerId() ?: "unknown"
                        )
                    }
                }
            } catch (e: Exception) {
                // SAFETY: ECAPA failure does NOT affect the call
                Log.e(TAG, "ECAPA-TDNN inference error (non-fatal): ${e.message}", e)
            }
        }
        lastEcapaMs = (System.nanoTime() - ecapaStart) / 1_000_000

        val processingTimeMs = (System.nanoTime() - startTime) / 1_000_000

        val result = WindowResult(
            windowIndex = windowIndex++,
            timestamp = System.currentTimeMillis(),
            vad = vadResult,
            antiSpoof = antiSpoofResult,
            speaker = speakerResult,
            processingTimeMs = processingTimeMs
        )

        Log.d(TAG, "Window ${result.windowIndex}: " +
                "speech=${String.format("%.3f", vadResult.speechProbability)}, " +
                "spoof=${antiSpoofResult?.let { String.format("%.4f", it.spoofScore) } ?: "N/A"}, " +
                "speaker=${speakerResult?.let { String.format("%.4f", it.similarity) } ?: "N/A"}, " +
                "time=${processingTimeMs}ms (vad=${lastVadMs}ms, aasist=${lastAasistMs}ms, ecapa=${lastEcapaMs}ms)")

        return result
    }

    override fun reset() {
        vadEngine.reset()
        windowIndex = 0
        lastVadResult = null
        lastVadMs = 0
        lastAasistMs = 0
        lastEcapaMs = 0
        Log.d(TAG, "On-device detection engine session reset")
    }

    override fun shutdown() {
        vadEngine.release()
        aasistEngine.release()
        ecapaEngine.release()
        vadEngine.reset()
        windowIndex = 0
        modelsInitialized = false
        lastVadResult = null
        lastVadMs = 0
        lastAasistMs = 0
        lastEcapaMs = 0
        Log.i(TAG, "On-device detection engine shut down")
    }
}

/**
 * Per-stage pipeline latency data. For diagnostics only.
 */
data class PipelineLatency(
    val vadMs: Long = 0,
    val aasistMs: Long = 0,
    val ecapaMs: Long = 0,
    val totalMs: Long = 0
)
