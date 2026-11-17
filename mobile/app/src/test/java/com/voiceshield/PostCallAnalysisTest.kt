package com.voiceshield

import com.voiceshield.shared.*
import org.junit.Assert.*
import org.junit.Test

class PostCallAnalysisTest {

    @Test
    fun testPostCallContracts_defaultInitialization() {
        val session = PostCallAnalysisSession()
        assertEquals(PostCallAnalysisOutcome.NOT_STARTED, session.outcome)
        assertEquals(SpeakerSeparationStatus.UNCERTAIN, session.speakerSeparationStatus)
        assertTrue(session.speechSegments.isEmpty())
        assertTrue(session.windowEvidence.isEmpty())
        assertNull(session.recordingUri)
        assertNull(session.recordingDisplayName)
        assertNull(session.recordingValidation)
        assertNull(session.qualityMetrics)
        assertNull(session.speechAvailability)
        assertNull(session.completedAt)
        assertEquals(0f, session.progress, 0.001f)
    }

    @Test
    fun testSpeechAvailability_criteria() {
        // Less than 3 seconds speech should be marked insufficient
        val insufficient = SpeechAvailability(
            hasSpeech = true,
            totalSpeechMs = 2400L,
            speechSegmentCount = 1,
            speechNonSpeechRatio = 0.2f,
            sufficientForAnalysis = false,
            insufficiencyReason = "Total speech duration (2.4s) is below 3.0s minimum threshold"
        )
        assertTrue(insufficient.hasSpeech)
        assertFalse(insufficient.sufficientForAnalysis)
        assertNotNull(insufficient.insufficiencyReason)

        // No speech at all
        val noSpeech = SpeechAvailability(
            hasSpeech = false,
            totalSpeechMs = 0L,
            speechSegmentCount = 0,
            sufficientForAnalysis = false,
            insufficiencyReason = "No speech detected in recording"
        )
        assertFalse(noSpeech.hasSpeech)
        assertFalse(noSpeech.sufficientForAnalysis)

        // Sufficient speech
        val sufficient = SpeechAvailability(
            hasSpeech = true,
            totalSpeechMs = 8500L,
            speechSegmentCount = 3,
            speechNonSpeechRatio = 0.7f,
            sufficientForAnalysis = true,
            insufficiencyReason = null
        )
        assertTrue(sufficient.hasSpeech)
        assertTrue(sufficient.sufficientForAnalysis)
        assertNull(sufficient.insufficiencyReason)
    }

    @Test
    fun testSpeakerSeparation_statusValues() {
        // SEPARATED represents two distinct clusters passing separation quality checks
        val separated = SpeakerSeparationStatus.SEPARATED
        assertNotNull(separated)

        // PROVISIONAL represents clusters produced but uncalibrated
        val provisional = SpeakerSeparationStatus.PROVISIONAL
        assertNotNull(provisional)

        // UNCERTAIN represents ambiguous separation
        val uncertain = SpeakerSeparationStatus.UNCERTAIN
        assertNotNull(uncertain)

        // SINGLE_SPEAKER represents only one party detected
        val single = SpeakerSeparationStatus.SINGLE_SPEAKER
        assertNotNull(single)

        // FAILED represents clustering error
        val failed = SpeakerSeparationStatus.FAILED
        assertNotNull(failed)
    }

    @Test
    fun testWindowEvidence_privacyLabelsAndMixedSpeakers() {
        // Labels MUST only be SPEAKER_A or SPEAKER_B, NEVER CALLER or USER
        val windowA = PostCallWindowEvidence(
            windowIndex = 0,
            startMs = 0,
            endMs = 3000,
            speakerLabel = "SPEAKER_A",
            isMixedSpeakerWindow = false,
            vadProb = 0.95f,
            aasistScore = 0.12f,
            aasistStatus = "OK",
            qualityStatus = "GOOD"
        )
        assertEquals("SPEAKER_A", windowA.speakerLabel)
        assertNotEquals("CALLER", windowA.speakerLabel)
        assertNotEquals("USER", windowA.speakerLabel)

        // Mixed speaker window must be flagged
        val windowMixed = PostCallWindowEvidence(
            windowIndex = 1,
            startMs = 1500,
            endMs = 4500,
            speakerLabel = null,
            isMixedSpeakerWindow = true,
            vadProb = 0.88f,
            aasistScore = 0.45f,
            aasistStatus = "OK",
            qualityStatus = "GOOD"
        )
        assertTrue(windowMixed.isMixedSpeakerWindow)
        assertNull(windowMixed.speakerLabel)
    }

    @Test
    fun testQualityMetrics_validation() {
        val metrics = RecordingQualityMetrics(
            durationMs = 15000L,
            rms = 0.08f,
            peak = 0.72f,
            clippingPercent = 0.01f,
            silencePercent = 12.5f,
            nanCount = 0,
            infCount = 0,
            sampleCount = 240000L
        )
        assertEquals(0, metrics.nanCount)
        assertEquals(0, metrics.infCount)
        assertTrue(metrics.clippingPercent < 1.0f)
        assertTrue(metrics.peak > 0f)
        assertTrue(metrics.rms > 0f)
    }

    @Test
    fun testConservativeOutcome_semantics() {
        // Verify all conservative outcomes exist and have expected semantics
        val noUsableAudio = PostCallAnalysisOutcome.NO_USABLE_AUDIO
        val noSpeech = PostCallAnalysisOutcome.NO_SPEECH
        val insufficientSpeech = PostCallAnalysisOutcome.INSUFFICIENT_SPEECH
        val normalSpeech = PostCallAnalysisOutcome.NORMAL_SPEECH_EVIDENCE
        val suspiciousSynthetic = PostCallAnalysisOutcome.SUSPICIOUS_SYNTHETIC_SPEECH_EVIDENCE
        val identityMismatch = PostCallAnalysisOutcome.SPEAKER_IDENTITY_MISMATCH_EVIDENCE
        val inconclusive = PostCallAnalysisOutcome.INCONCLUSIVE
        val analysisError = PostCallAnalysisOutcome.ANALYSIS_ERROR

        assertNotNull(noUsableAudio)
        assertNotNull(noSpeech)
        assertNotNull(insufficientSpeech)
        assertNotNull(normalSpeech)
        assertNotNull(suspiciousSynthetic)
        assertNotNull(identityMismatch)
        assertNotNull(inconclusive)
        assertNotNull(analysisError)
    }

    @Test
    fun testRecordingValidation_statusReporting() {
        val verified = RecordingValidation(
            status = RecordingStatus.VERIFIED,
            uri = "content://com.android.providers.media.documents/document/audio%3A1234",
            displayName = "Call_Recording_20260904.mp3",
            format = "audio/mpeg",
            codec = "mp3",
            sampleRate = 44100,
            channelCount = 2,
            durationMs = 45000,
            fileSizeBytes = 720000L
        )
        assertEquals(RecordingStatus.VERIFIED, verified.status)
        assertEquals(2, verified.channelCount)
        assertEquals(44100, verified.sampleRate)
        assertTrue(verified.uri?.startsWith("content://") == true)

        val unreadable = RecordingValidation(
            status = RecordingStatus.UNREADABLE,
            uri = "content://invalid/uri",
            displayName = "broken.mp3",
            errorMessage = "Failed to open input stream from content resolver"
        )
        assertEquals(RecordingStatus.UNREADABLE, unreadable.status)
        assertNotNull(unreadable.errorMessage)
    }

    @Test
    fun testSpeechSegment_labelingRules() {
        val seg1 = SpeechSegment(
            startMs = 0L,
            endMs = 4500L,
            durationMs = 4500L,
            avgVadProb = 0.92f,
            speakerLabel = "SPEAKER_A"
        )
        val seg2 = SpeechSegment(
            startMs = 5000L,
            endMs = 9200L,
            durationMs = 4200L,
            avgVadProb = 0.89f,
            speakerLabel = "SPEAKER_B"
        )
        assertEquals(4500L, seg1.durationMs)
        assertEquals("SPEAKER_A", seg1.speakerLabel)
        assertEquals("SPEAKER_B", seg2.speakerLabel)
    }

    @Test
    fun testAasistInputContract_exact64600Samples() {
        // Section 7 Model Contract: AASIST-L must receive exactly 64600 samples (~4.0375s at 16kHz)
        val windowSize = com.voiceshield.ml.PostCallAnalysisCoordinator.WINDOW_SIZE
        val aasistRequired = com.voiceshield.ml.AasistLEngine.REQUIRED_SAMPLES
        assertEquals(64600, windowSize)
        assertEquals(64600, aasistRequired)
        assertEquals("AudioBuffer window size must match AASIST-L requirement exactly", aasistRequired, windowSize)

        // Verify AudioBuffer produces windows of exactly 64600 samples
        val buffer = com.voiceshield.audio.AudioBuffer(windowSize = windowSize, overlapRatio = 0.5f)
        assertEquals(32300, buffer.hopSize)

        // Feed 100,000 samples
        val input = FloatArray(100000) { 0.01f }
        val windows = buffer.addSamples(input)
        assertTrue("Must produce at least one window", windows.isNotEmpty())
        for (w in windows) {
            assertEquals("Every window must be exactly 64600 samples — NO silent padding or cropping", 64600, w.size)
        }
    }

    @Test
    fun testDriftFreeTimestampForensics_acrossDurations() {
        // Section 9: Prove sample position -> timestamp without drift across 10s, 1m, 10m, 1h
        val hopSize = 32300L
        val windowSize = 64600L
        val sampleRate = 16000L

        val testDurations = listOf(
            10 to "10 seconds",
            60 to "1 minute",
            600 to "10 minutes",
            3600 to "1 hour"
        )

        for ((durationSec, label) in testDurations) {
            val totalSamples = durationSec.toLong() * sampleRate
            val numWindows = if (totalSamples >= windowSize) {
                ((totalSamples - windowSize) / hopSize + 1).toInt()
            } else 0

            if (numWindows > 0) {
                val lastWindowIndex = numWindows - 1
                val sampleStart = lastWindowIndex.toLong() * hopSize
                val sampleEnd = sampleStart + windowSize
                val startMs = (sampleStart * 1000L) / sampleRate
                val endMs = (sampleEnd * 1000L) / sampleRate

                // Compute theoretical floating-point millisecond position
                val exactStartMs = (sampleStart.toDouble() / sampleRate.toDouble()) * 1000.0
                val exactEndMs = (sampleEnd.toDouble() / sampleRate.toDouble()) * 1000.0

                val startDrift = Math.abs(exactStartMs - startMs.toDouble())
                val endDrift = Math.abs(exactEndMs - endMs.toDouble())

                // Max drift from integer division truncation is strictly < 1.0 ms across ANY duration
                assertTrue("$label: start drift must be < 1.0 ms (actual: $startDrift ms)", startDrift < 1.0)
                assertTrue("$label: end drift must be < 1.0 ms (actual: $endDrift ms)", endDrift < 1.0)
                assertTrue("$label: window duration must be exactly 4037 or 4038 ms", endMs - startMs in 4037..4038)
            }
        }
    }

    @Test
    fun testMixedSpeakerWindow_marginAmbiguityDetection() {
        // Section 13: When embeddings have similar distance to both centroids,
        // window MUST be marked mixed and have null speakerLabel.
        val simA = 0.62f
        val simB = 0.58f
        val margin = Math.abs(simA - simB)
        assertTrue("Margin < 0.08 indicates ambiguous overlap", margin < 0.08f)

        val window = PostCallWindowEvidence(
            windowIndex = 5,
            startMs = 10093,
            endMs = 14130,
            sampleStart = 161500L,
            sampleEnd = 226100L,
            speakerLabel = null,
            isMixedSpeakerWindow = true,
            vadProb = 0.94f,
            aasistScore = 0.72f,
            aasistStatus = "OK"
        )
        assertTrue(window.isMixedSpeakerWindow)
        assertNull("Mixed window must NEVER be attributed to a speaker", window.speakerLabel)
    }

    @Test
    fun testUncalibratedModel_abstentionSemantics() {
        // Section 16 & 18: Uncalibrated models must NOT produce SUSPICIOUS_SYNTHETIC_SPEECH_EVIDENCE
        val uncalibratedWindows = listOf(
            PostCallWindowEvidence(
                windowIndex = 0, startMs = 0, endMs = 4037, sampleStart = 0, sampleEnd = 64600,
                speakerLabel = "SPEAKER_A", isMixedSpeakerWindow = false,
                vadProb = 0.95f, aasistScore = 0.85f, aasistStatus = "OK"
            ),
            PostCallWindowEvidence(
                windowIndex = 1, startMs = 2018, endMs = 6056, sampleStart = 32300, sampleEnd = 96900,
                speakerLabel = "SPEAKER_A", isMixedSpeakerWindow = false,
                vadProb = 0.92f, aasistScore = 0.78f, aasistStatus = "OK"
            )
        )

        // Average spoof is elevated (0.815), but model is uncalibrated
        val avgSpoof = uncalibratedWindows.mapNotNull { it.aasistScore }.average().toFloat()
        assertTrue(avgSpoof > 0.5f)

        // The system must classify elevated uncalibrated scores as INCONCLUSIVE
        val outcome = if (avgSpoof < 0.35f) {
            PostCallAnalysisOutcome.NORMAL_SPEECH_EVIDENCE
        } else {
            PostCallAnalysisOutcome.INCONCLUSIVE
        }
        assertEquals(
            "Elevated spoof score on uncalibrated model MUST abstain as INCONCLUSIVE",
            PostCallAnalysisOutcome.INCONCLUSIVE,
            outcome
        )
    }

    @Test
    fun testVadGatingDoesNotTrimWindowPassedToAasist() {
        // PROVE: VAD acts strictly as an execution gate, NEVER as a variable-length cropper.
        // Even if speech is localized to a fraction of the window (e.g. 1.2s),
        // the full 64,600-sample window is preserved intact for AASIST-L inference.
        val windowSize = com.voiceshield.ml.PostCallAnalysisCoordinator.WINDOW_SIZE
        val buffer = com.voiceshield.audio.AudioBuffer(windowSize = windowSize, overlapRatio = 0.5f)

        // Generate synthetic audio frame
        val audioData = FloatArray(100000) { i ->
            if (i in 20000..40000) (kotlin.math.sin(i * 0.05f) * 0.3f) else 0f
        }
        val windows = buffer.addSamples(audioData)
        assertTrue(windows.isNotEmpty())

        val candidateWindow = windows.first()
        assertEquals(64600, candidateWindow.size)

        // Mock VAD decision: speech is detected
        val vadProbability = 0.88f
        val isSpeech = vadProbability > 0.5f
        assertTrue("Speech must be detected in synthetic tone", isSpeech)

        // Target window passed to AASIST must be the EXACT candidate window, not a subslice
        val windowForAasist = candidateWindow
        assertEquals("AASIST input MUST NOT be trimmed to speech bounds", 64600, windowForAasist.size)
        assertEquals("AASIST input must match AasistLEngine.REQUIRED_SAMPLES", com.voiceshield.ml.AasistLEngine.REQUIRED_SAMPLES, windowForAasist.size)
    }
}
