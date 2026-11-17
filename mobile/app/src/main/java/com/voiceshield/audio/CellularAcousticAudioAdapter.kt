/**
 * CellularAcousticAudioAdapter — Acoustic Capture via Device Microphone.
 *
 * Captures audio from the device microphone during an active cellular call.
 * This is the ACOUSTIC FALLBACK path — not direct digital PCM of the caller.
 *
 * What it captures:
 *   - Device microphone input during a call
 *   - MAY include caller audio from earpiece/speaker (acoustic coupling)
 *   - ALWAYS includes the local user's voice
 *   - Quality depends on speaker mode, ambient noise, earpiece coupling
 *
 * What it is NOT:
 *   - NOT direct cellular downlink PCM
 *   - NOT guaranteed to contain caller speech
 *   - NOT a replacement for VOICE_DOWNLINK (which is system-only)
 *
 * SAFETY RULES (BINDING):
 * 1. OFF by default — requires explicit user activation
 * 2. If call audio degrades → immediately stop → restore normal telephony
 * 3. If call disconnects → immediately stop
 * 4. If uplink fails → immediately stop
 * 5. Working phone call ALWAYS wins over analysis
 * 6. Never touch AudioManager mode, audio focus, or speaker routing
 * 7. Never persist raw audio
 */
package com.voiceshield.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import com.voiceshield.shared.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt

class CellularAcousticAudioAdapter(
    private val context: Context? = null
) : AudioSourceAdapter {

    companion object {
        private const val TAG = "VoiceShieldAcoustic"
        const val SAMPLE_RATE = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val READ_CHUNK_SAMPLES = 1024
        private const val MAX_CONSECUTIVE_READ_ERRORS = 20
    }

    // ─── State ──────────────────────────────────────────────────────────

    private val isRunning = AtomicBoolean(false)
    private val isEnabled = AtomicBoolean(false) // OFF by default
    private var recordingJob: Job? = null
    private var audioRecord: AudioRecord? = null

    // ─── Stage 1 & 2 Live Metrics (metadata only — NO raw audio) ───────

    private val framesRead = AtomicLong(0)
    private val bytesRead = AtomicLong(0)
    private val readErrors = AtomicLong(0)
    private var lastReadError: String? = null
    private val framesDelivered = AtomicLong(0)
    private var consecutiveReadErrors = 0
    private var consecutiveSilentFrames = 0
    private var currentRms: Float = 0f
    private var peakRms: Float = 0f
    private var stopReason: String? = null
    private var audioRecordStateInt: Int = AudioRecord.STATE_UNINITIALIZED
    private var recordingStateStr: String = "STOPPED"
    private var selectedSourceStr: String = "MIC"
    private var latestRawMetrics: AudioSignalMetrics? = null
    private var latestRawShortMetrics: RawShortPcmMetrics? = null

    // Debug-only source selection: MIC (default) or VOICE_COMMUNICATION
    // Only takes effect on next start(). Does NOT create concurrent recorders.
    private var configuredAudioSource: Int = MediaRecorder.AudioSource.MIC
    private var configuredAudioSourceName: String = "MIC"

    private val _diagnosticsFlow = MutableStateFlow(AcousticCaptureDiagnostics())
    val diagnosticsFlow: StateFlow<AcousticCaptureDiagnostics> = _diagnosticsFlow.asStateFlow()

    // ─── Public API ─────────────────────────────────────────────────────

    /**
     * Whether acoustic capture is currently enabled by the user.
     * OFF by default. Must be explicitly toggled.
     */
    fun isUserEnabled(): Boolean = isEnabled.get()

    /**
     * Enable or disable acoustic capture.
     * When disabled, any active capture is immediately stopped.
     */
    fun setEnabled(enabled: Boolean) {
        isEnabled.set(enabled)
        if (!enabled && isRunning.get()) {
            Log.i(TAG, "User disabled acoustic capture — stopping immediately")
            CoroutineScope(Dispatchers.Default).launch {
                safeStop("USER_DISABLED")
            }
        }
        updateDiagnostics()
    }

    /**
     * Get the reason for stop, if acoustic capture was stopped.
     */
    fun getStopReason(): String? = stopReason

    /**
     * Debug-only: set the AudioSource for the next start() call.
     * ONLY MIC and VOICE_COMMUNICATION are allowed.
     * Does NOT create a new AudioRecord — takes effect on next start().
     * If currently running, caller must stop() first, then start() again.
     */
    fun setDebugAudioSource(source: Int) {
        when (source) {
            MediaRecorder.AudioSource.MIC -> {
                configuredAudioSource = source
                configuredAudioSourceName = "MIC"
            }
            MediaRecorder.AudioSource.VOICE_COMMUNICATION -> {
                configuredAudioSource = source
                configuredAudioSourceName = "VOICE_COMMUNICATION"
            }
            else -> {
                Log.w(TAG, "setDebugAudioSource: rejected source $source (only MIC and VOICE_COMMUNICATION allowed)")
                return
            }
        }
        Log.i(TAG, "Debug audio source set to: $configuredAudioSourceName (takes effect on next start)")
    }

    fun getConfiguredAudioSourceName(): String = configuredAudioSourceName

    /**
     * Legacy getter for backward compatibility.
     */
    fun getAutoKillReason(): String? = stopReason

    /**
     * Get capture diagnostics (metadata only).
     */
    fun getDiagnostics(): AcousticCaptureDiagnostics {
        return _diagnosticsFlow.value
    }

    private fun updateDiagnostics() {
        _diagnosticsFlow.value = AcousticCaptureDiagnostics(
            isEnabled = isEnabled.get(),
            isRunning = isRunning.get(),
            audioRecordInitialized = audioRecordStateInt == AudioRecord.STATE_INITIALIZED,
            recordingState = recordingStateStr,
            selectedSource = selectedSourceStr,
            framesRead = framesRead.get(),
            bytesRead = bytesRead.get(),
            readErrors = readErrors.get(),
            lastReadError = lastReadError,
            framesDelivered = framesDelivered.get(),
            currentRms = currentRms,
            peakRms = peakRms,
            silentFrames = consecutiveSilentFrames,
            stopReason = stopReason,
            rawMetrics = latestRawMetrics,
            rawShortMetrics = latestRawShortMetrics
        )
    }

    // ─── AudioSourceAdapter ─────────────────────────────────────────────

    override fun isAvailable(): Boolean =
        isRunning.get() && isEnabled.get()

    override fun sourceType(): AudioSourceType = AudioSourceType.ACOUSTIC_CAPTURE

    override suspend fun start(session: CallSession, onAudioFrame: (AudioFrame) -> Unit) {
        if (!isEnabled.get()) {
            Log.i(TAG, "Acoustic capture not enabled by user — skipping")
            stopReason = "NOT_USER_ENABLED"
            updateDiagnostics()
            return
        }

        if (isRunning.get()) {
            Log.w(TAG, "Acoustic capture already running")
            return
        }

        val ctx = context ?: run {
            Log.e(TAG, "Context is null — cannot initialize AudioRecord")
            stopReason = "CONTEXT_NULL"
            updateDiagnostics()
            return
        }

        // Check runtime microphone permission
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "RECORD_AUDIO permission not granted at runtime")
            stopReason = "PERMISSION_DENIED"
            updateDiagnostics()
            return
        }

        // Reset stage metrics
        stopReason = null
        consecutiveSilentFrames = 0
        consecutiveReadErrors = 0
        framesRead.set(0)
        bytesRead.set(0)
        readErrors.set(0)
        lastReadError = null
        framesDelivered.set(0)
        peakRms = 0f
        currentRms = 0f

        try {
            val minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
                stopReason = "MIN_BUFFER_SIZE_INVALID ($minBufferSize)"
                updateDiagnostics()
                Log.e(TAG, "AudioRecord minBufferSize invalid: $minBufferSize")
                return
            }

            val bufferSize = maxOf(minBufferSize, READ_CHUNK_SAMPLES * 2 * 2)

            @Suppress("MissingPermission")
            val record = AudioRecord(
                configuredAudioSource,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            audioRecord = record
            audioRecordStateInt = record.state
            selectedSourceStr = configuredAudioSourceName

            if (record.state != AudioRecord.STATE_INITIALIZED) {
                stopReason = "AUDIORECORD_INIT_FAILED (state=${record.state})"
                Log.e(TAG, "AudioRecord failed to initialize: state=${record.state}")
                updateDiagnostics()
                record.release()
                audioRecord = null
                return
            }

            record.startRecording()
            recordingStateStr = "RECORDING"
            isRunning.set(true)
            updateDiagnostics()

            Log.i(TAG, "Acoustic capture started ($configuredAudioSourceName, ${SAMPLE_RATE}Hz, mono, bufferSize=$bufferSize)")

            // Start capture loop
            recordingJob = CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
                captureLoop(session, onAudioFrame)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start acoustic capture: ${e.message}", e)
            safeStop("INIT_FAILED: ${e.message}")
        }
    }

    override suspend fun stop() {
        safeStop("NORMAL_STOP")
    }

    // ─── Capture Loop ───────────────────────────────────────────────────

    private suspend fun captureLoop(session: CallSession, onAudioFrame: (AudioFrame) -> Unit) {
        val shortBuffer = ShortArray(READ_CHUNK_SAMPLES)
        val floatBuffer = FloatArray(READ_CHUNK_SAMPLES)

        while (isRunning.get() && isEnabled.get()) {
            try {
                val recorder = audioRecord ?: break
                val shortsRead = recorder.read(shortBuffer, 0, READ_CHUNK_SAMPLES)

                if (shortsRead < 0) {
                    readErrors.incrementAndGet()
                    consecutiveReadErrors++
                    val errCode = when (shortsRead) {
                        AudioRecord.ERROR_INVALID_OPERATION -> "ERROR_INVALID_OPERATION"
                        AudioRecord.ERROR_BAD_VALUE -> "ERROR_BAD_VALUE"
                        AudioRecord.ERROR_DEAD_OBJECT -> "ERROR_DEAD_OBJECT"
                        else -> "ERROR_CODE_$shortsRead"
                    }
                    lastReadError = errCode
                    Log.w(TAG, "AudioRecord.read returned error: $errCode (count=$consecutiveReadErrors)")
                    updateDiagnostics()

                    // Deterministic stop on persistent HAL/AudioRecord error
                    if (consecutiveReadErrors >= MAX_CONSECUTIVE_READ_ERRORS) {
                        safeStop("PERSISTENT_READ_ERROR: $errCode")
                        return
                    }
                    delay(50)
                    continue
                }

                if (shortsRead == 0) {
                    readErrors.incrementAndGet()
                    lastReadError = "ZERO_SAMPLES_READ"
                    updateDiagnostics()
                    delay(20)
                    continue
                }

                // Successful read
                consecutiveReadErrors = 0
                framesRead.incrementAndGet()
                bytesRead.addAndGet(shortsRead * 2L)

                // ── RAW SHORT FORENSICS (computed BEFORE any conversion) ────
                // Single O(N) pass over raw ShortArray using Long accumulators.
                // Zero allocation. Proves what AudioRecord actually delivered.
                var rawMinShort: Short = Short.MAX_VALUE
                var rawMaxShort: Short = Short.MIN_VALUE
                var rawSumLong: Long = 0L
                var rawSumSqLong: Long = 0L
                var rawPeakAbs: Int = 0
                var rawNonZero: Int = 0

                for (i in 0 until shortsRead) {
                    val v = shortBuffer[i]
                    if (v < rawMinShort) rawMinShort = v
                    if (v > rawMaxShort) rawMaxShort = v
                    val absV = if (v >= 0) v.toInt() else -v.toInt()
                    if (absV > rawPeakAbs) rawPeakAbs = absV
                    if (v.toInt() != 0) rawNonZero++
                    rawSumLong += v.toLong()
                    rawSumSqLong += v.toLong() * v.toLong()
                }

                val rawZeroCount = shortsRead - rawNonZero
                val rawMean = rawSumLong.toFloat() / shortsRead
                val rawRms = sqrt(rawSumSqLong.toDouble() / shortsRead).toFloat()

                // Convert short to float [-1.0, 1.0] with zero-allocation scalar accumulation
                var sumSq = 0.0
                var sum = 0.0
                var minSample = 1.0f
                var maxSample = -1.0f
                var peakAbs = 0.0f
                var zeroCount = 0

                for (i in 0 until shortsRead) {
                    val s = shortBuffer[i].toFloat() / 32768f
                    floatBuffer[i] = s
                    val absS = kotlin.math.abs(s)
                    if (absS > peakAbs) peakAbs = absS
                    if (s < minSample) minSample = s
                    if (s > maxSample) maxSample = s
                    if (absS < 1e-5f) zeroCount++
                    sum += s.toDouble()
                    sumSq += s.toDouble() * s.toDouble()
                }

                // Compute RMS for diagnostic observability (no raw audio stored)
                val rms = sqrt(sumSq / shortsRead).toFloat()
                currentRms = rms
                if (rms > peakRms) peakRms = rms

                if (rms < 1e-5f) {
                    consecutiveSilentFrames++
                } else {
                    consecutiveSilentFrames = 0
                }

                // Deliver frame
                val frame = AudioFrame(
                    samples = floatBuffer.copyOf(shortsRead),
                    sampleRate = SAMPLE_RATE,
                    timestamp = System.currentTimeMillis(),
                    sourceType = AudioSourceType.ACOUSTIC_CAPTURE,
                    channelCount = 1
                )

                onAudioFrame(frame)
                framesDelivered.incrementAndGet()

                // Periodic telemetry update every 16 frames (~1s / 16,384 samples)
                if (framesRead.get() % 16L == 0L) {
                    val mean = (sum / shortsRead).toFloat()
                    val variance = kotlin.math.max(0.0, (sumSq / shortsRead) - (mean.toDouble() * mean.toDouble())).toFloat()
                    val stdDev = kotlin.math.sqrt(variance)
                    val peakDb = if (peakAbs > 0f) (20f * kotlin.math.log10(peakAbs.toDouble())).toFloat().coerceIn(-100f, 0f) else -100f
                    val zeroPct = (zeroCount.toFloat() / shortsRead) * 100f
                    val classification = when {
                        peakAbs < 1e-5f || zeroPct >= 99.9f -> "SILENT_ZERO"
                        rms < 0.005f -> "VERY_LOW_AMPLITUDE"
                        peakAbs >= 0.999f -> "POTENTIALLY_CLIPPED"
                        else -> "NORMAL_SIGNAL"
                    }

                    latestRawMetrics = AudioSignalMetrics(
                        sampleCount = framesRead.get() * shortsRead,
                        sampleRate = SAMPLE_RATE,
                        channelCount = 1,
                        minSample = minSample,
                        maxSample = maxSample,
                        rms = rms,
                        peakAbs = peakAbs,
                        peakDb = peakDb,
                        mean = mean,
                        stdDev = stdDev,
                        zeroOrNearZeroPercentage = zeroPct,
                        nanCount = 0,
                        infCount = 0,
                        classification = classification
                    )

                    // Raw Short forensic metrics + conversion verification
                    val expectedFloatRms = rawRms / 32768f
                    val conversionError = kotlin.math.abs(expectedFloatRms - rms)

                    latestRawShortMetrics = RawShortPcmMetrics(
                        sampleCount = shortsRead,
                        minShort = rawMinShort,
                        maxShort = rawMaxShort,
                        meanShort = rawMean,
                        rmsShort = rawRms,
                        peakAbsShort = rawPeakAbs,
                        nonZeroCount = rawNonZero,
                        zeroCount = rawZeroCount,
                        zeroPercent = (rawZeroCount.toFloat() / shortsRead) * 100f,
                        expectedFloatRms = expectedFloatRms,
                        actualFloatRms = rms,
                        conversionError = conversionError
                    )
                    updateDiagnostics()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Capture loop error: ${e.message}", e)
                safeStop("CAPTURE_LOOP_ERROR: ${e.message}")
                return
            }
        }

        Log.i(TAG, "Capture loop ended (running=${isRunning.get()}, enabled=${isEnabled.get()})")
        updateDiagnostics()
    }

    // ─── Safe Stop ──────────────────────────────────────────────────────

    /**
     * Safely stop acoustic capture and release all resources.
     * This is designed to NEVER throw and NEVER affect the cellular call.
     */
    private suspend fun safeStop(reason: String) {
        stopReason = reason
        isRunning.set(false)
        recordingStateStr = "STOPPED"
        audioRecordStateInt = AudioRecord.STATE_UNINITIALIZED

        Log.i(TAG, "Acoustic capture stopping: $reason")

        try {
            val job = recordingJob
            recordingJob = null
            job?.cancelAndJoin()
        } catch (e: Exception) {
            Log.e(TAG, "Error canceling recording job: ${e.message}")
        }

        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping AudioRecord: ${e.message}")
        }

        try {
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing AudioRecord: ${e.message}")
        }

        audioRecord = null
        updateDiagnostics()

        Log.i(TAG, "Acoustic capture stopped (framesRead=${framesRead.get()}, delivered=${framesDelivered.get()}, peakRms=$peakRms, reason=$reason)")
    }

    /**
     * Experiment D: Standalone diagnostic microphone capture.
     *
     * Opens a fresh AudioRecord(MIC, 16kHz, mono, PCM16) — NO call active.
     * Reads for [durationMs] milliseconds, computes raw Short + Float metrics,
     * and returns a StandaloneMicDiagnosticResult.
     *
     * Does NOT invoke VAD, AASIST, ECAPA, normalizer, buffer, or risk engine.
     * Does NOT persist raw audio samples.
     * Must NOT be called while [isRunning] is true (concurrent AudioRecord hazard).
     */
    suspend fun runStandaloneDiagnosticCapture(
        ctx: Context,
        durationMs: Long = 3000L
    ): StandaloneMicDiagnosticResult = withContext(Dispatchers.IO) {
        if (isRunning.get()) {
            return@withContext StandaloneMicDiagnosticResult(
                errorMessage = "REFUSED: Production AudioRecord is active. Experiment D requires no active call."
            )
        }

        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            return@withContext StandaloneMicDiagnosticResult(
                errorMessage = "RECORD_AUDIO permission not granted"
            )
        }

        var diagRecord: AudioRecord? = null
        try {
            val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBuf == AudioRecord.ERROR || minBuf == AudioRecord.ERROR_BAD_VALUE) {
                return@withContext StandaloneMicDiagnosticResult(
                    errorMessage = "minBufferSize invalid: $minBuf"
                )
            }

            val bufSize = maxOf(minBuf, READ_CHUNK_SAMPLES * 2 * 2)

            @Suppress("MissingPermission")
            diagRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufSize
            )

            if (diagRecord.state != AudioRecord.STATE_INITIALIZED) {
                val state = diagRecord.state
                diagRecord.release()
                return@withContext StandaloneMicDiagnosticResult(
                    errorMessage = "AudioRecord init failed: state=$state"
                )
            }

            diagRecord.startRecording()
            Log.i(TAG, "Experiment D: standalone diagnostic capture started (${durationMs}ms)")

            val shortBuf = ShortArray(READ_CHUNK_SAMPLES)
            val startTime = System.currentTimeMillis()
            var totalFrames = 0
            var totalSamples = 0

            // Aggregate accumulators (raw Short domain)
            var aggMinShort: Short = Short.MAX_VALUE
            var aggMaxShort: Short = Short.MIN_VALUE
            var aggSumLong: Long = 0L
            var aggSumSqLong: Long = 0L
            var aggPeakAbs: Int = 0
            var aggNonZero: Int = 0

            // Aggregate accumulators (Float domain)
            var aggFloatSumSq: Double = 0.0
            var aggFloatPeakAbs: Float = 0f

            while (System.currentTimeMillis() - startTime < durationMs) {
                val read = diagRecord.read(shortBuf, 0, READ_CHUNK_SAMPLES)
                if (read <= 0) {
                    delay(20)
                    continue
                }
                totalFrames++
                totalSamples += read

                // Raw Short pass
                for (i in 0 until read) {
                    val v = shortBuf[i]
                    if (v < aggMinShort) aggMinShort = v
                    if (v > aggMaxShort) aggMaxShort = v
                    val absV = if (v >= 0) v.toInt() else -v.toInt()
                    if (absV > aggPeakAbs) aggPeakAbs = absV
                    if (v.toInt() != 0) aggNonZero++
                    aggSumLong += v.toLong()
                    aggSumSqLong += v.toLong() * v.toLong()
                }

                // Float pass (conversion verification)
                for (i in 0 until read) {
                    val f = shortBuf[i].toFloat() / 32768f
                    val absF = kotlin.math.abs(f)
                    if (absF > aggFloatPeakAbs) aggFloatPeakAbs = absF
                    aggFloatSumSq += f.toDouble() * f.toDouble()
                }
            }

            diagRecord.stop()
            diagRecord.release()
            diagRecord = null

            val elapsed = System.currentTimeMillis() - startTime
            Log.i(TAG, "Experiment D: captured $totalFrames frames, $totalSamples samples in ${elapsed}ms")

            if (totalSamples == 0) {
                return@withContext StandaloneMicDiagnosticResult(
                    durationMs = elapsed,
                    errorMessage = "Zero samples read in ${elapsed}ms"
                )
            }

            val rawRms = sqrt(aggSumSqLong.toDouble() / totalSamples).toFloat()
            val rawMean = aggSumLong.toFloat() / totalSamples
            val rawZeroCount = totalSamples - aggNonZero
            val rawZeroPct = (rawZeroCount.toFloat() / totalSamples) * 100f

            val floatRms = sqrt(aggFloatSumSq / totalSamples).toFloat()
            val expectedFloatRms = rawRms / 32768f
            val convError = kotlin.math.abs(expectedFloatRms - floatRms)

            val rawShort = RawShortPcmMetrics(
                sampleCount = totalSamples,
                minShort = aggMinShort,
                maxShort = aggMaxShort,
                meanShort = rawMean,
                rmsShort = rawRms,
                peakAbsShort = aggPeakAbs,
                nonZeroCount = aggNonZero,
                zeroCount = rawZeroCount,
                zeroPercent = rawZeroPct,
                expectedFloatRms = expectedFloatRms,
                actualFloatRms = floatRms,
                conversionError = convError
            )

            val floatPeakDb = if (aggFloatPeakAbs > 0f)
                (20f * kotlin.math.log10(aggFloatPeakAbs.toDouble())).toFloat().coerceIn(-100f, 0f)
            else -100f

            val floatClassification = when {
                aggFloatPeakAbs < 1e-5f || rawZeroPct >= 99.9f -> "SILENT_ZERO"
                floatRms < 0.005f -> "VERY_LOW_AMPLITUDE"
                aggFloatPeakAbs >= 0.999f -> "POTENTIALLY_CLIPPED"
                else -> "NORMAL_SIGNAL"
            }

            val floatMetrics = AudioSignalMetrics(
                sampleCount = totalSamples.toLong(),
                sampleRate = SAMPLE_RATE,
                channelCount = 1,
                rms = floatRms,
                peakAbs = aggFloatPeakAbs,
                peakDb = floatPeakDb,
                classification = floatClassification
            )

            val match = when {
                rawRms > 50f && floatRms > 1e-5f -> "MATCH_SIGNAL_ALIVE"
                rawRms < 1f && floatRms < 1e-5f -> "MATCH_BOTH_ZERO"
                rawRms > 50f && floatRms < 1e-5f -> "MISMATCH_CONVERSION_DEFECT"
                else -> "LOW_SIGNAL"
            }

            StandaloneMicDiagnosticResult(
                durationMs = elapsed,
                framesRead = totalFrames,
                totalSamplesRead = totalSamples,
                rawShortMetrics = rawShort,
                floatMetrics = floatMetrics,
                shortFloatMatch = match
            )

        } catch (e: Exception) {
            Log.e(TAG, "Experiment D failed: ${e.message}", e)
            try { diagRecord?.stop() } catch (_: Exception) {}
            try { diagRecord?.release() } catch (_: Exception) {}
            StandaloneMicDiagnosticResult(
                errorMessage = "Experiment D exception: ${e.message}"
            )
        }
    }
}
