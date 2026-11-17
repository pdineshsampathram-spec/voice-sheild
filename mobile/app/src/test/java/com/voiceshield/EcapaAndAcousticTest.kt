package com.voiceshield

import com.voiceshield.audio.CellularAcousticAudioAdapter
import com.voiceshield.ml.EcapaTdnnEngine
import com.voiceshield.security.TemporalEvidenceEngine
import com.voiceshield.shared.*
import org.junit.Assert.*
import org.junit.Test

class EcapaAndAcousticTest {

    @Test
    fun testCosineSimilarity_mathematicalProperties() {
        val engine = EcapaTdnnEngine()

        // 1. Identical vectors -> 1.0
        val v1 = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        val similaritySelf = engine.compareSpeakers(v1, v1)
        assertEquals(1.0f, similaritySelf, 1e-5f)

        // 2. Orthogonal vectors -> 0.0
        val vOrthogonalA = floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f)
        val vOrthogonalB = floatArrayOf(0.0f, 1.0f, 0.0f, 0.0f)
        val similarityOrthogonal = engine.compareSpeakers(vOrthogonalA, vOrthogonalB)
        assertEquals(0.0f, similarityOrthogonal, 1e-5f)

        // 3. Opposite vectors -> -1.0
        val vOpposite = floatArrayOf(-0.5f, -0.5f, -0.5f, -0.5f)
        val similarityOpposite = engine.compareSpeakers(v1, vOpposite)
        assertEquals(-1.0f, similarityOpposite, 1e-5f)
    }

    @Test
    fun testCosineSimilarity_dimensionMismatch_throwsIllegalArgument() {
        val engine = EcapaTdnnEngine()
        val v1 = FloatArray(192) { 0.1f }
        val v2 = FloatArray(128) { 0.1f }

        assertThrows(IllegalArgumentException::class.java) {
            engine.compareSpeakers(v1, v2)
        }
    }

    @Test
    fun testSpeakerResult_allThresholdsLabeledExperimental() {
        val engine = EcapaTdnnEngine()
        val result = engine.createSpeakerResult(
            similarity = 0.82f,
            enrolledSpeakerId = "owner"
        )

        assertEquals("ECAPA-TDNN", result.modelName)
        assertEquals(0.82f, result.similarity, 1e-5f)
        assertEquals("owner", result.enrolledSpeakerId)
        // CRITICAL BINDING RULE: No arbitrary threshold applied without calibration
        assertNull("Match boolean must be null until calibrated", result.match)
        assertNull("Threshold used must be null until calibrated", result.thresholdUsed)
    }

    @Test
    fun testEcapaEngine_uninitialized_reportsNotImplemented() {
        val engine = EcapaTdnnEngine()
        assertFalse(engine.isReady())
        assertEquals(ModelStatus.NOT_IMPLEMENTED, engine.getStatus())
    }

    @Test
    fun testAcousticAdapter_sourceTypeIsAcousticCapture() {
        val adapter = CellularAcousticAudioAdapter()

        // BINDING HONESTY RULE: Must report ACOUSTIC_CAPTURE, never CELLULAR_DOWNLINK
        assertEquals(AudioSourceType.ACOUSTIC_CAPTURE, adapter.sourceType())
        assertNotEquals(AudioSourceType.CELLULAR_DOWNLINK, adapter.sourceType())
        assertNotEquals(AudioSourceType.CELLULAR_MIXED, adapter.sourceType())
    }

    @Test
    fun testAcousticAdapter_offByDefault_userMustExplicitlyEnable() {
        val adapter = CellularAcousticAudioAdapter()

        // BINDING SAFETY RULE: OFF by default
        assertFalse("Acoustic capture must be OFF by default", adapter.isUserEnabled())
        assertFalse("Acoustic adapter must not be available when disabled", adapter.isAvailable())

        // Explicit toggle ON
        adapter.setEnabled(true)
        assertTrue("Acoustic capture should be enabled after user toggle", adapter.isUserEnabled())

        // Explicit toggle OFF
        adapter.setEnabled(false)
        assertFalse("Acoustic capture should be disabled after user toggle", adapter.isUserEnabled())
    }

    @Test
    fun testRiskFusion_dualFactorEvidence_whenSpoofHighAndSpeakerMismatch() {
        val engine = TemporalEvidenceEngine(minWindowsForAssessment = 3, calibration = null)

        // Add 3 windows with high spoof (0.85) AND low speaker similarity (0.15)
        for (i in 0 until 3) {
            engine.addEvidence(
                WindowEvidence(
                    windowIndex = i,
                    timestamp = System.currentTimeMillis() + i * 1000,
                    spoofScore = 0.85f,
                    speakerSimilarity = 0.15f, // low similarity = mismatch
                    vadConfidence = 0.95f,
                    modelAvailability = mapOf(
                        "AASIST-L" to ModelStatus.READY,
                        "ECAPA-TDNN" to ModelStatus.READY
                    )
                )
            )
        }

        val threatState = engine.evaluate(AudioSourceType.ACOUSTIC_CAPTURE)
        assertEquals(ThreatLevel.MONITORING, threatState.level)
        assertTrue(
            "Reason must report dual-factor evidence",
            threatState.reason.contains("EXPERIMENTAL DUAL-FACTOR")
        )
        assertTrue(
            "Reason must mention speaker mismatch",
            threatState.reason.contains("speaker mismatch")
        )
    }

    @Test
    fun testRiskFusion_speakerAnomaly_whenSpoofLowAndSpeakerMismatch() {
        val engine = TemporalEvidenceEngine(minWindowsForAssessment = 3, calibration = null)

        // Add 3 windows with LOW spoof (0.10) BUT low speaker similarity (0.20)
        // Meaning: someone else is speaking, but their voice is human, not synthetic
        for (i in 0 until 3) {
            engine.addEvidence(
                WindowEvidence(
                    windowIndex = i,
                    timestamp = System.currentTimeMillis() + i * 1000,
                    spoofScore = 0.10f,
                    speakerSimilarity = 0.20f,
                    vadConfidence = 0.95f,
                    modelAvailability = mapOf(
                        "AASIST-L" to ModelStatus.READY,
                        "ECAPA-TDNN" to ModelStatus.READY
                    )
                )
            )
        }

        val threatState = engine.evaluate(AudioSourceType.ACOUSTIC_CAPTURE)
        assertTrue(
            "Reason must identify identity anomaly, not automated clone",
            threatState.reason.contains("EXPERIMENTAL IDENTITY ANOMALY")
        )
    }

    @Test
    fun testRiskFusion_spoofOnly_whenNoEnrolledSpeaker() {
        val engine = TemporalEvidenceEngine(minWindowsForAssessment = 3, calibration = null)

        // Add 3 windows with high spoof (0.80) but speakerSimilarity = null (no enrollment)
        for (i in 0 until 3) {
            engine.addEvidence(
                WindowEvidence(
                    windowIndex = i,
                    timestamp = System.currentTimeMillis() + i * 1000,
                    spoofScore = 0.80f,
                    speakerSimilarity = null, // No enrollment
                    vadConfidence = 0.95f,
                    modelAvailability = mapOf("AASIST-L" to ModelStatus.READY)
                )
            )
        }

        val threatState = engine.evaluate(AudioSourceType.ACOUSTIC_CAPTURE)
        assertTrue(
            "Reason must indicate spoof-only evidence",
            threatState.reason.contains("EXPERIMENTAL SPOOF-ONLY")
        )
    }
}
