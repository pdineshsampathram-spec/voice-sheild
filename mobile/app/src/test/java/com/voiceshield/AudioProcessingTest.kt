package com.voiceshield

import com.voiceshield.audio.AudioBuffer
import com.voiceshield.audio.AudioNormalizer
import com.voiceshield.shared.AudioFrame
import com.voiceshield.shared.AudioSourceType
import org.junit.Assert.*
import org.junit.Test

class AudioProcessingTest {

    @Test
    fun testAudioNormalizer_passthrough_whenAlready16kHzMono() {
        val normalizer = AudioNormalizer()
        val samples = FloatArray(1600) { 0.5f }
        val frame = AudioFrame(
            samples = samples,
            sampleRate = 16000,
            channelCount = 1,
            sourceType = AudioSourceType.WEBRTC_DIRECT
        )

        val normalized = normalizer.normalize(frame)
        assertEquals(16000, normalized.sampleRate)
        assertEquals(1, normalized.channelCount)
        assertEquals(1600, normalized.samples.size)
        assertEquals(0.5f, normalized.samples[0], 1e-5f)
    }

    @Test
    fun testAudioNormalizer_stereoToMonoAveraging() {
        val normalizer = AudioNormalizer()
        // Left: 0.2, Right: 0.8 -> Expected mono: 0.5
        val stereoSamples = floatArrayOf(0.2f, 0.8f, 0.4f, 0.6f)
        val frame = AudioFrame(
            samples = stereoSamples,
            sampleRate = 16000,
            channelCount = 2,
            sourceType = AudioSourceType.WEBRTC_DIRECT
        )

        val normalized = normalizer.normalize(frame)
        assertEquals(1, normalized.channelCount)
        assertEquals(2, normalized.samples.size)
        assertEquals(0.5f, normalized.samples[0], 1e-4f)
        assertEquals(0.5f, normalized.samples[1], 1e-4f)
    }

    @Test
    fun testAudioBuffer_accumulatesAndEmits64600SampleWindows() {
        val windowSize = 64600
        val buffer = AudioBuffer(windowSize = windowSize, overlapRatio = 0.5f)

        // Feed chunks of 16,000 samples (1 second each)
        val chunk = FloatArray(16000) { 0.1f }

        // 1s -> 16,000 samples (0 windows)
        val w1 = buffer.addSamples(chunk)
        assertEquals(0, w1.size)
        assertEquals(16000, buffer.currentSize)

        // 2s -> 32,000 samples (0 windows)
        val w2 = buffer.addSamples(chunk)
        assertEquals(0, w2.size)
        assertEquals(32000, buffer.currentSize)

        // 3s -> 48,000 samples (0 windows)
        val w3 = buffer.addSamples(chunk)
        assertEquals(0, w3.size)
        assertEquals(48000, buffer.currentSize)

        // 4s -> 64,000 samples (0 windows)
        val w4 = buffer.addSamples(chunk)
        assertEquals(0, w4.size)
        assertEquals(64000, buffer.currentSize)

        // 5s -> 80,000 samples -> exceeds 64,600!
        // Should yield 1 window (64,600 samples) and slide by 32,300
        val w5 = buffer.addSamples(chunk)
        assertTrue("Expected at least 1 window when reaching 80,000 samples", w5.isNotEmpty())
        assertEquals(windowSize, w5[0].size)

        // Buffer size after slide: 80,000 - 32,300 = 47,700
        assertEquals(80000 - 32300, buffer.currentSize)
    }

    @Test
    fun testAudioBuffer_resetClearsState() {
        val buffer = AudioBuffer(windowSize = 64600)
        buffer.addSamples(FloatArray(10000) { 0.2f })
        assertEquals(10000, buffer.currentSize)

        buffer.reset()
        assertEquals(0, buffer.currentSize)
        assertEquals(0f, buffer.fillPercentage, 1e-5f)
    }
}
