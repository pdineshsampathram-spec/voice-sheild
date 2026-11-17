/**
 * AudioSourceAdapter — The most critical architectural component.
 *
 * Separates CALL MANAGEMENT ACCESS from CALL AUDIO ACCESS.
 * The Android dialer layer handles call lifecycle.
 * Audio acquisition must explicitly report whether an actual audio stream is available.
 *
 * CRITICAL HONESTY RULE:
 * If remote audio cannot be accessed → NoAudioAdapter → UI shows
 * "VOICE ANALYSIS: NOT AVAILABLE FOR THIS AUDIO PATH"
 * Never generate fake analysis.
 */
package com.voiceshield.audio

import com.voiceshield.shared.AudioFrame
import com.voiceshield.shared.AudioSignalMetrics
import com.voiceshield.shared.AudioSourceType
import com.voiceshield.shared.CallSession
import com.voiceshield.shared.computeAudioSignalMetrics

/**
 * Common interface for all audio source adapters.
 */
interface AudioSourceAdapter {

    /** Whether this adapter can provide actual audio for the given session. */
    fun isAvailable(): Boolean

    /** The type of audio source this adapter represents. */
    fun sourceType(): AudioSourceType

    /**
     * Start capturing audio from this source.
     * @param session The active call session.
     * @param onAudioFrame Callback invoked for each captured audio frame.
     */
    suspend fun start(session: CallSession, onAudioFrame: (AudioFrame) -> Unit)

    /** Stop capturing audio and release resources. */
    suspend fun stop()
}

/**
 * NoAudioAdapter — Used when no legitimate remote audio path exists.
 *
 * This is the DEFAULT adapter for SIM calls where Android does not provide
 * access to the remote party's audio stream.
 *
 * It NEVER generates fake audio or fake analysis results.
 */
class NoAudioAdapter : AudioSourceAdapter {

    override fun isAvailable(): Boolean = false

    override fun sourceType(): AudioSourceType = AudioSourceType.UNAVAILABLE

    override suspend fun start(session: CallSession, onAudioFrame: (AudioFrame) -> Unit) {
        // No-op. No audio is available.
        // The UI layer must show:
        //   "Call Protection Active"
        //   "Call Lifecycle Monitoring: ON"
        //   "Live Voice Analysis: NOT AVAILABLE FOR THIS AUDIO PATH"
    }

    override suspend fun stop() {
        // No-op.
    }
}

/**
 * AudioNormalizer — Ensures all audio reaches the detection pipeline
 * in the correct format: mono float32, 16kHz.
 */
class AudioNormalizer {

    companion object {
        const val TARGET_SAMPLE_RATE = 16000
        const val TARGET_CHANNELS = 1
    }

    var framesReceived: Long = 0L
        private set

    var samplesProduced: Long = 0L
        private set

    var lastInputMetrics: AudioSignalMetrics? = null
        private set

    var lastOutputMetrics: AudioSignalMetrics? = null
        private set

    /**
     * Gain applied by the normalizer: output RMS / max(1e-9, input RMS).
     * Empirically proves whether normalizer modifies signal amplitude.
     */
    val normalizerGain: Float
        get() {
            val inRms = lastInputMetrics?.rms ?: return 1.0f
            val outRms = lastOutputMetrics?.rms ?: return 1.0f
            return outRms / kotlin.math.max(1e-9f, inRms)
        }

    fun resetMetrics() {
        framesReceived = 0L
        samplesProduced = 0L
        lastInputMetrics = null
        lastOutputMetrics = null
    }

    /**
     * Normalize an audio frame to 16kHz mono float32.
     * If already in the correct format, returns the input unchanged.
     */
    fun normalize(frame: AudioFrame): AudioFrame {
        framesReceived++

        // Snapshot input metrics periodically (every 16 frames to keep overhead near-zero)
        val shouldComputeMetrics = framesReceived % 16L == 0L || lastInputMetrics == null

        if (frame.sampleRate == TARGET_SAMPLE_RATE && frame.channelCount == TARGET_CHANNELS) {
            samplesProduced += frame.samples.size
            if (shouldComputeMetrics) {
                val metrics = computeAudioSignalMetrics(frame.samples, frame.sampleRate, frame.channelCount)
                lastInputMetrics = metrics
                lastOutputMetrics = metrics
            }
            return frame
        }

        if (shouldComputeMetrics) {
            lastInputMetrics = computeAudioSignalMetrics(frame.samples, frame.sampleRate, frame.channelCount)
        }

        var samples = frame.samples

        // Stereo → mono: average channels
        if (frame.channelCount == 2) {
            val mono = FloatArray(samples.size / 2)
            for (i in mono.indices) {
                mono[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2f
            }
            samples = mono
        }

        // Resample if needed (simple linear interpolation)
        if (frame.sampleRate != TARGET_SAMPLE_RATE) {
            samples = resample(samples, frame.sampleRate, TARGET_SAMPLE_RATE)
        }

        samplesProduced += samples.size

        val outFrame = frame.copy(
            samples = samples,
            sampleRate = TARGET_SAMPLE_RATE,
            channelCount = TARGET_CHANNELS
        )

        if (shouldComputeMetrics) {
            lastOutputMetrics = computeAudioSignalMetrics(outFrame.samples, outFrame.sampleRate, outFrame.channelCount)
        }

        return outFrame
    }

    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val ratio = fromRate.toDouble() / toRate.toDouble()
        val outputLength = (input.size / ratio).toInt()
        val output = FloatArray(outputLength)

        for (i in output.indices) {
            val srcPos = i * ratio
            val srcIndex = srcPos.toInt()
            val frac = (srcPos - srcIndex).toFloat()

            output[i] = if (srcIndex + 1 < input.size) {
                input[srcIndex] * (1f - frac) + input[srcIndex + 1] * frac
            } else {
                input[srcIndex.coerceAtMost(input.size - 1)]
            }
        }

        return output
    }
}

/**
 * AudioBuffer — Sliding window buffer for accumulating audio frames
 * into AASIST-L-sized windows (64,600 samples at 16kHz ≈ 4.04 seconds).
 *
 * Uses 50% overlap to avoid missing speech at segment boundaries.
 * Mirrors the backend's StreamingBuffer logic.
 */
class AudioBuffer(
    val windowSize: Int = 64600,  // AASIST-L requirement
    val overlapRatio: Float = 0.5f
) {
    private val buffer = mutableListOf<Float>()
    val hopSize: Int = (windowSize * (1f - overlapRatio)).toInt()

    var windowsEmitted: Int = 0
        private set

    var lastWindowMetrics: AudioSignalMetrics? = null
        private set

    /**
     * Add new audio samples to the buffer.
     * @return List of complete windows ready for analysis.
     */
    fun addSamples(samples: FloatArray): List<FloatArray> {
        for (sample in samples) {
            buffer.add(sample)
        }

        val windows = mutableListOf<FloatArray>()
        while (buffer.size >= windowSize) {
            val window = FloatArray(windowSize)
            for (i in 0 until windowSize) {
                window[i] = buffer[i]
            }
            windows.add(window)
            lastWindowMetrics = computeAudioSignalMetrics(window, AudioNormalizer.TARGET_SAMPLE_RATE, 1)

            // Efficient bulk slide by hop size (O(N) instead of O(N^2))
            if (buffer.size >= hopSize) {
                buffer.subList(0, hopSize).clear()
            } else {
                buffer.clear()
            }
        }

        windowsEmitted += windows.size
        return windows
    }

    /** Reset buffer state (e.g. between calls). */
    fun reset() {
        buffer.clear()
        windowsEmitted = 0
        lastWindowMetrics = null
    }

    /** Current number of samples in the buffer. */
    val currentSize: Int get() = buffer.size

    /** Required samples for a complete window. */
    val requiredSamples: Int get() = windowSize

    /** Percentage fill toward next complete window. */
    val fillPercentage: Float get() = if (windowSize > 0) (currentSize.toFloat() / windowSize).coerceIn(0f, 1f) else 0f
}
