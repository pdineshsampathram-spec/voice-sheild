package com.voiceshield

import com.voiceshield.ml.MelFeatureExtractor
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

class MelFeatureExtractorTest {

    @Test
    fun testExtraction_dimensionsMatchContract() {
        val extractor = MelFeatureExtractor()
        assertEquals(80, extractor.nMels)
        assertEquals(16000, extractor.sampleRate)
        assertEquals(400, extractor.winLengthSamples)
        assertEquals(160, extractor.hopLengthSamples)

        // 1 second of audio at 16kHz
        val samples = FloatArray(16000) { i ->
            (sin(2.0 * Math.PI * 440.0 * i / 16000.0)).toFloat()
        }

        val features = extractor.extract(samples)
        
        // Expected frames: 1 + (16000 - 400) / 160 = 1 + 15600 / 160 = 1 + 97 = 98 frames
        val expectedFrames = 1 + (16000 - 400) / 160
        assertEquals(expectedFrames, features.size)
        assertEquals(80, features[0].size)

        // Verify all values are finite (no NaN, no Inf)
        for (frame in features) {
            assertEquals(80, frame.size)
            for (value in frame) {
                assertFalse("Value must not be NaN", value.isNaN())
                assertFalse("Value must not be Infinite", value.isInfinite())
            }
        }
    }

    @Test
    fun testShortAudio_returnsEmptyArray() {
        val extractor = MelFeatureExtractor()
        // Audio shorter than winLengthSamples (400)
        val shortSamples = FloatArray(350) { 0.5f }
        val features = extractor.extract(shortSamples)
        assertEquals(0, features.size)
    }

    @Test
    fun testCmvn_centersMeanAroundZero() {
        val extractor = MelFeatureExtractor(applyCmvn = true)
        val samples = FloatArray(16000) { i ->
            (0.5 * sin(2.0 * Math.PI * 1000.0 * i / 16000.0)).toFloat()
        }

        val features = extractor.extract(samples)
        assertTrue(features.isNotEmpty())

        // Check mean per mel band across frames is close to 0
        for (m in 0 until extractor.nMels) {
            var sum = 0f
            for (frame in features) {
                sum += frame[m]
            }
            val mean = sum / features.size
            assertTrue("Mel band $m mean after CMVN should be ~0, got $mean", abs(mean) < 1e-4f)
        }
    }

    @Test
    fun testFactoryFromContract_preservesParameters() {
        val extractor = MelFeatureExtractor.fromContract(
            sampleRate = 16000,
            nMels = 80,
            nFft = 400,
            winLengthSamples = 400,
            hopLengthSamples = 160,
            fMin = 0f,
            fMax = 8000f
        )
        assertEquals(16000, extractor.sampleRate)
        assertEquals(80, extractor.nMels)
        assertEquals(400, extractor.nFft)
        assertEquals(400, extractor.winLengthSamples)
        assertEquals(160, extractor.hopLengthSamples)
        assertEquals(0f, extractor.fMin, 1e-4f)
        assertEquals(8000f, extractor.fMax, 1e-4f)
    }
}
