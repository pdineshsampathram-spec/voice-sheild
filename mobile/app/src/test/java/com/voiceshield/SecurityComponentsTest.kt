package com.voiceshield

import com.voiceshield.security.AlertManager
import com.voiceshield.security.RiskEngine
import com.voiceshield.security.TemporalEvidenceEngine
import com.voiceshield.shared.*
import org.junit.Assert.*
import org.junit.Test

class SecurityComponentsTest {

    @Test
    fun testAudioUnavailable_reportsHonestlyWithoutFakeScore() {
        val engine = TemporalEvidenceEngine()
        val state = engine.evaluate(AudioSourceType.UNAVAILABLE)

        assertEquals(ThreatLevel.AUDIO_UNAVAILABLE, state.level)
        assertNull("Spoof score must be null when audio is unavailable", state.avgSpoofScore)
        assertNull("Speaker similarity must be null when audio is unavailable", state.avgSpeakerSimilarity)
        assertTrue(state.reason.contains("Remote call audio is not available"))
    }

    @Test
    fun testInsufficientEvidence_whenFewerThanMinWindows() {
        val engine = TemporalEvidenceEngine(minWindowsForAssessment = 3)

        // Add only 1 window
        engine.addEvidence(
            WindowEvidence(
                windowIndex = 0,
                timestamp = System.currentTimeMillis(),
                spoofScore = 0.85f,
                speakerSimilarity = null,
                vadConfidence = 0.9f,
                modelAvailability = mapOf("AASIST-L" to ModelStatus.READY)
            )
        )

        val state = engine.evaluate(AudioSourceType.WEBRTC_DIRECT)
        assertEquals(ThreatLevel.INSUFFICIENT_EVIDENCE, state.level)
        assertEquals(1, state.windowsAnalyzed)
        assertTrue(state.reason.contains("1/3 windows collected"))
    }

    @Test
    fun testUncalibratedState_reportsCalibrationNotAvailable() {
        val engine = TemporalEvidenceEngine(minWindowsForAssessment = 3, calibration = null)

        // Add 3 windows with scores
        for (i in 0 until 3) {
            engine.addEvidence(
                WindowEvidence(
                    windowIndex = i,
                    timestamp = System.currentTimeMillis() + i * 1000,
                    spoofScore = 0.75f,
                    speakerSimilarity = null,
                    vadConfidence = 0.95f,
                    modelAvailability = mapOf("AASIST-L" to ModelStatus.READY)
                )
            )
        }

        val state = engine.evaluate(AudioSourceType.WEBRTC_DIRECT)
        // Without calibration, system must not claim definitive fraud/safe level
        assertEquals(ThreatLevel.MONITORING, state.level)
        assertTrue("Reason must mention CALIBRATION NOT AVAILABLE", state.reason.contains("CALIBRATION NOT AVAILABLE"))
        assertNotNull(state.avgSpoofScore)
        assertEquals(0.75f, state.avgSpoofScore!!, 1e-4f)
    }

    @Test
    fun testCalibratedState_escalatesToCritical_whenPersistentSuspiciousAudio() {
        val calibration = CalibrationArtifact(
            modelName = "AASIST-L",
            modelRevision = "test-rev",
            datasetIdentifier = "ASVspoof2019-LA",
            evaluationDate = "2026-09-02",
            sampleCount = 1000,
            operatingPoint = OperatingPoint(
                threshold = 0.5f,
                eer = 0.05f,
                farAtThreshold = 0.05f,
                frrAtThreshold = 0.05f
            )
        )

        val engine = TemporalEvidenceEngine(minWindowsForAssessment = 3, calibration = calibration)

        // Add 5 consecutive high spoof windows
        for (i in 0 until 5) {
            engine.addEvidence(
                WindowEvidence(
                    windowIndex = i,
                    timestamp = System.currentTimeMillis() + i * 1000,
                    spoofScore = 0.92f, // clearly above 0.5 threshold
                    speakerSimilarity = null,
                    vadConfidence = 0.99f,
                    modelAvailability = mapOf("AASIST-L" to ModelStatus.READY)
                )
            )
        }

        val state = engine.evaluate(AudioSourceType.WEBRTC_DIRECT)
        assertEquals(ThreatLevel.CRITICAL, state.level)
        assertTrue(state.reason.contains("Persistent anti-spoof evidence"))
        assertEquals(5, state.suspiciousWindows)
    }

    @Test
    fun testAlertManager_triggersAlertOnCritical_andRespectsUserAction() {
        val alertManager = AlertManager()

        val threatState = ThreatState(
            level = ThreatLevel.CRITICAL,
            reason = "Persistent synthetic audio detected across 5 windows",
            windowsAnalyzed = 5,
            suspiciousWindows = 5,
            avgSpoofScore = 0.92f,
            avgSpeakerSimilarity = null,
            audioSourceType = AudioSourceType.WEBRTC_DIRECT,
            modelAvailability = mapOf("AASIST-L" to ModelStatus.READY)
        )

        alertManager.evaluateThreatState(threatState)
        val active = alertManager.activeAlert.value
        assertNotNull("Active alert should be present for CRITICAL state", active)
        assertEquals(ThreatLevel.CRITICAL, active!!.threatState.level)

        // User chooses to verify identity
        alertManager.onUserAction(active.id, AlertManager.UserAction.VERIFY_IDENTITY)
        assertNull("Alert should be dismissed after user action", alertManager.activeAlert.value)
    }
}
