package com.voiceshield

import com.voiceshield.audio.AudioNormalizer
import com.voiceshield.shared.*
import com.voiceshield.telecom.CallSessionManager
import com.voiceshield.telecom.CallState
import org.junit.Assert.*
import org.junit.Test

class CellularAudioTest {

    @Test
    fun testCellularCapabilityManager_sourceCategorization() {
        // 1. Verify AudioSource categorization contract
        val downlinkProbe = AudioSourceProbeResult(
            sourceId = 3, // VOICE_DOWNLINK
            sourceName = "VOICE_DOWNLINK",
            category = AudioCategory.REMOTE_DOWNLINK,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            hasNonZeroPcm = true,
            verification = RemoteAudioVerification.VERIFIED_TRUE
        )
        assertEquals(AudioCategory.REMOTE_DOWNLINK, downlinkProbe.category)
        assertEquals(RemoteAudioVerification.VERIFIED_TRUE, downlinkProbe.verification)

        val callProbe = AudioSourceProbeResult(
            sourceId = 4, // VOICE_CALL
            sourceName = "VOICE_CALL",
            category = AudioCategory.BIDIRECTIONAL_CALL,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            hasNonZeroPcm = true,
            verification = RemoteAudioVerification.VERIFIED_TRUE
        )
        assertEquals(AudioCategory.BIDIRECTIONAL_CALL, callProbe.category)

        val uplinkProbe = AudioSourceProbeResult(
            sourceId = 2, // VOICE_UPLINK
            sourceName = "VOICE_UPLINK",
            category = AudioCategory.LOCAL_UPLINK,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            hasNonZeroPcm = true,
            verification = RemoteAudioVerification.FAILED_OR_RESTRICTED
        )
        assertEquals(AudioCategory.LOCAL_UPLINK, uplinkProbe.category)
    }

    @Test
    fun testRejectionOfMicSourcesAsRemoteAudio() {
        // Strict Honesty Rule:
        // Non-zero PCM or RMS energy on MIC or VOICE_COMMUNICATION proves only that local audio
        // exists—it does NOT prove it is the remote caller. They MUST be classified as NEAR_END_MIC.
        val micProbe = AudioSourceProbeResult(
            sourceId = 1, // MIC
            sourceName = "MIC",
            category = AudioCategory.NEAR_END_MIC,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            framesRead = 1024,
            bytesRead = 2048,
            hasNonZeroPcm = true,
            peakRmsDb = -22.4f,
            verification = RemoteAudioVerification.UNVERIFIED_MIC_ONLY
        )

        val voiceCommProbe = AudioSourceProbeResult(
            sourceId = 7, // VOICE_COMMUNICATION
            sourceName = "VOICE_COMMUNICATION",
            category = AudioCategory.NEAR_END_MIC,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            framesRead = 1024,
            bytesRead = 2048,
            hasNonZeroPcm = true,
            peakRmsDb = -18.2f,
            verification = RemoteAudioVerification.UNVERIFIED_MIC_ONLY
        )

        val report = CellularCapabilityReport(
            callState = "ACTIVE",
            testedSources = listOf(micProbe, voiceCommProbe),
            usableRemotePcm = RemoteAudioVerification.UNVERIFIED_MIC_ONLY,
            detectedRemoteSource = null,
            technicalLimitationReason = "Only near-end microphone sources available"
        )

        assertNotEquals(RemoteAudioVerification.VERIFIED_TRUE, report.usableRemotePcm)
        assertEquals(RemoteAudioVerification.UNVERIFIED_MIC_ONLY, report.usableRemotePcm)
        assertNull("Detected remote source must be null when only mic sources work", report.detectedRemoteSource)
    }

    @Test
    fun testPermissionDenial_returnsSafeFailure() {
        val deniedProbe = AudioSourceProbeResult(
            sourceId = 3,
            sourceName = "VOICE_DOWNLINK",
            category = AudioCategory.REMOTE_DOWNLINK,
            initialized = false,
            recordingStarted = false,
            permissionGranted = false,
            error = "RECORD_AUDIO permission not granted"
        )

        assertFalse(deniedProbe.permissionGranted)
        assertFalse(deniedProbe.initialized)
        assertEquals("RECORD_AUDIO permission not granted", deniedProbe.error)
        assertEquals(RemoteAudioVerification.FAILED_OR_RESTRICTED, deniedProbe.verification)
    }

    @Test
    fun testAudioRecordInitializationFailure_handling() {
        val failedProbe = AudioSourceProbeResult(
            sourceId = 3,
            sourceName = "VOICE_DOWNLINK",
            category = AudioCategory.REMOTE_DOWNLINK,
            initialized = false,
            recordingStarted = false,
            permissionGranted = true,
            error = "STATE_UNINITIALIZED (OS security policy or device restriction)"
        )

        assertFalse(failedProbe.initialized)
        assertFalse(failedProbe.recordingStarted)
        assertTrue(failedProbe.error!!.contains("STATE_UNINITIALIZED"))
        assertEquals(RemoteAudioVerification.FAILED_OR_RESTRICTED, failedProbe.verification)
    }

    @Test
    fun testZeroFrameSource_markedFailedOrRestricted() {
        val zeroFrameProbe = AudioSourceProbeResult(
            sourceId = 4,
            sourceName = "VOICE_CALL",
            category = AudioCategory.BIDIRECTIONAL_CALL,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            framesRead = 0,
            bytesRead = 0,
            hasNonZeroPcm = false,
            verification = RemoteAudioVerification.FAILED_OR_RESTRICTED,
            error = "AudioRecord failed to read frames"
        )

        assertFalse(zeroFrameProbe.hasNonZeroPcm)
        assertEquals(0L, zeroFrameProbe.framesRead)
        assertEquals(RemoteAudioVerification.FAILED_OR_RESTRICTED, zeroFrameProbe.verification)
    }

    @Test
    fun testDirectDownlink_whenVerified_triggersVerifiedTrue() {
        val verifiedDownlink = AudioSourceProbeResult(
            sourceId = 3,
            sourceName = "VOICE_DOWNLINK",
            category = AudioCategory.REMOTE_DOWNLINK,
            initialized = true,
            recordingStarted = true,
            permissionGranted = true,
            framesRead = 1024,
            bytesRead = 2048,
            hasNonZeroPcm = true,
            peakRmsDb = -20.0f,
            verification = RemoteAudioVerification.VERIFIED_TRUE
        )

        val report = CellularCapabilityReport(
            callState = "ACTIVE",
            testedSources = listOf(verifiedDownlink),
            usableRemotePcm = RemoteAudioVerification.VERIFIED_TRUE,
            detectedRemoteSource = "VOICE_DOWNLINK"
        )

        assertEquals(RemoteAudioVerification.VERIFIED_TRUE, report.usableRemotePcm)
        assertEquals("VOICE_DOWNLINK", report.detectedRemoteSource)
        assertNull(report.technicalLimitationReason)
    }

    @Test
    fun testSampleRateNormalization_variousRates() {
        val normalizer = AudioNormalizer()

        // 1. 8000 Hz narrowband telephony mono -> 16000 Hz
        val frame8k = AudioFrame(
            samples = FloatArray(800) { 0.5f },
            sampleRate = 8000,
            sourceType = AudioSourceType.CELLULAR_DOWNLINK,
            channelCount = 1
        )
        val normalized8k = normalizer.normalize(frame8k)
        assertEquals(16000, normalized8k.sampleRate)
        assertEquals(1600, normalized8k.samples.size) // Doubled size for 2x upsampling
        assertEquals(AudioSourceType.CELLULAR_DOWNLINK, normalized8k.sourceType)

        // 2. 48000 Hz HD audio stereo -> 16000 Hz mono
        val stereoSamples = FloatArray(4800) { if (it % 2 == 0) 0.6f else 0.4f }
        val frame48k = AudioFrame(
            samples = stereoSamples,
            sampleRate = 48000,
            sourceType = AudioSourceType.CELLULAR_MIXED,
            channelCount = 2
        )
        val normalized48k = normalizer.normalize(frame48k)
        assertEquals(16000, normalized48k.sampleRate)
        assertEquals(1, normalized48k.channelCount)
        assertEquals(800, normalized48k.samples.size) // 2400 mono samples / 3 = 800
        assertEquals(AudioSourceType.CELLULAR_MIXED, normalized48k.sourceType)
    }

    @Test
    fun testCallSessionManager_cellularReportAndAudioSourceUpdate() {
        val manager = CallSessionManager()
        val session = manager.startWebRtcSession("test-room")

        val report = CellularCapabilityReport(
            callState = "ACTIVE",
            usableRemotePcm = RemoteAudioVerification.FAILED_OR_RESTRICTED,
            technicalLimitationReason = "Direct downlink restricted"
        )

        // Update capability report
        manager.updateCapabilityReport(session.callSession.id, report)
        val updated = manager.getSession(session.callSession.id)
        assertNotNull(updated)
        assertEquals(report, updated!!.capabilityReport)

        // Switch to CELLULAR_DOWNLINK
        manager.updateAudioSource(session.callSession.id, AudioSourceType.CELLULAR_DOWNLINK)
        val withDownlink = manager.getSession(session.callSession.id)
        assertEquals(AudioSourceType.CELLULAR_DOWNLINK, withDownlink!!.callSession.audioSourceType)
        assertEquals(AudioSourceType.CELLULAR_DOWNLINK, withDownlink.threatState.audioSourceType)

        // Fallback to UNAVAILABLE
        manager.updateAudioSource(session.callSession.id, AudioSourceType.UNAVAILABLE)
        val withUnavailable = manager.getSession(session.callSession.id)
        assertEquals(AudioSourceType.UNAVAILABLE, withUnavailable!!.callSession.audioSourceType)
    }

    @Test
    fun testAudioUnavailable_threatStatePropagation() {
        val manager = CallSessionManager()
        val session = manager.startWebRtcSession("cellular-call-1")

        val threatState = ThreatState(
            level = ThreatLevel.AUDIO_UNAVAILABLE,
            reason = "Call Protection Active. Voice analysis unavailable: Direct downlink restricted by platform security policy.",
            windowsAnalyzed = 0,
            suspiciousWindows = 0,
            avgSpoofScore = null,
            avgSpeakerSimilarity = null,
            audioSourceType = AudioSourceType.UNAVAILABLE,
            modelAvailability = emptyMap()
        )

        manager.updateThreatState(session.callSession.id, threatState)
        val updated = manager.getSession(session.callSession.id)

        assertNotNull(updated)
        assertEquals(ThreatLevel.AUDIO_UNAVAILABLE, updated!!.threatState.level)
        assertNull(updated.threatState.avgSpoofScore)
        assertNull(updated.threatState.avgSpeakerSimilarity)
        assertEquals(0, updated.threatState.windowsAnalyzed)
        assertTrue(updated.threatState.reason.contains("Voice analysis unavailable"))
    }
}
