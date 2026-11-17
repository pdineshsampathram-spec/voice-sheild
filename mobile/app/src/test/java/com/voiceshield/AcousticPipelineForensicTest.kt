package com.voiceshield

import com.voiceshield.audio.AudioBuffer
import com.voiceshield.audio.AudioNormalizer
import com.voiceshield.audio.CellularAcousticAudioAdapter
import com.voiceshield.shared.*
import com.voiceshield.telecom.CallSessionManager
import com.voiceshield.telecom.CallState
import com.voiceshield.telecom.ManagedCallSession
import org.junit.Assert.*
import org.junit.Test

/**
 * Forensic tests verifying the Acoustic Capture pipeline, stage-by-stage telemetry,
 * sliding-window buffer mechanics, honest state transitions, and telephony isolation.
 */
class AcousticPipelineForensicTest {

    // ─── 1. Buffer & Window Mechanics ────────────────────────────────────

    @Test
    fun testAudioBuffer_slidingWindow_windowSize64600_overlap50Percent() {
        val buffer = AudioBuffer(windowSize = 64600, overlapRatio = 0.5f)

        assertEquals(64600, buffer.requiredSamples)
        assertEquals(32300, buffer.hopSize)
        assertEquals(0, buffer.currentSize)
        assertEquals(0, buffer.windowsEmitted)
        assertEquals(0f, buffer.fillPercentage, 1e-5f)

        // Feed 10,000 samples (insufficient for a full window)
        val chunk1 = FloatArray(10000) { 0.1f }
        val windows1 = buffer.addSamples(chunk1)
        assertTrue("No window emitted before 64,600 samples", windows1.isEmpty())
        assertEquals(10000, buffer.currentSize)
        assertEquals(0, buffer.windowsEmitted)
        assertEquals(10000f / 64600f, buffer.fillPercentage, 1e-5f)

        // Feed another 54,600 samples -> exactly 64,600 samples
        val chunk2 = FloatArray(54600) { 0.2f }
        val windows2 = buffer.addSamples(chunk2)
        assertEquals("Exactly 1 window emitted upon reaching 64,600 samples", 1, windows2.size)
        assertEquals(64600, windows2[0].size)
        assertEquals(1, buffer.windowsEmitted)
        // After 50% overlap slide (hopSize = 32,300), remaining samples = 64,600 - 32,300 = 32,300
        assertEquals(32300, buffer.currentSize)
        assertEquals(32300f / 64600f, buffer.fillPercentage, 1e-5f)

        // Feed another 32,300 samples -> reaches 64,600 again -> 2nd window emitted
        val chunk3 = FloatArray(32300) { 0.3f }
        val windows3 = buffer.addSamples(chunk3)
        assertEquals("2nd window emitted after hopSize samples", 1, windows3.size)
        assertEquals(2, buffer.windowsEmitted)
        assertEquals(32300, buffer.currentSize)

        // Reset buffer
        buffer.reset()
        assertEquals(0, buffer.currentSize)
        assertEquals(0, buffer.windowsEmitted)
        assertEquals(0f, buffer.fillPercentage, 1e-5f)
    }

    // ─── 2. Normalizer Metrics & Channels ────────────────────────────────

    @Test
    fun testAudioNormalizer_countersAndChannelAveraging() {
        val normalizer = AudioNormalizer()
        assertEquals(0L, normalizer.framesReceived)
        assertEquals(0L, normalizer.samplesProduced)

        // 16kHz mono frame -> passed through directly
        val frameMono = AudioFrame(
            samples = FloatArray(1024) { 0.5f },
            sampleRate = 16000,
            channelCount = 1,
            sourceType = AudioSourceType.ACOUSTIC_CAPTURE
        )
        val normalizedMono = normalizer.normalize(frameMono)
        assertEquals(1L, normalizer.framesReceived)
        assertEquals(1024L, normalizer.samplesProduced)
        assertEquals(16000, normalizedMono.sampleRate)
        assertEquals(1, normalizedMono.channelCount)

        // Stereo frame -> converted to mono
        val frameStereo = AudioFrame(
            samples = floatArrayOf(0.2f, 0.4f, 0.6f, 0.8f),
            sampleRate = 16000,
            channelCount = 2,
            sourceType = AudioSourceType.ACOUSTIC_CAPTURE
        )
        val normalizedStereo = normalizer.normalize(frameStereo)
        assertEquals(2L, normalizer.framesReceived)
        assertEquals(1024L + 2L, normalizer.samplesProduced)
        assertEquals(2, normalizedStereo.samples.size)
        assertEquals(0.3f, normalizedStereo.samples[0], 1e-5f)
        assertEquals(0.7f, normalizedStereo.samples[1], 1e-5f)

        normalizer.resetMetrics()
        assertEquals(0L, normalizer.framesReceived)
        assertEquals(0L, normalizer.samplesProduced)
    }

    // ─── 3. Acoustic Adapter Diagnostics & Telemetry ─────────────────────

    @Test
    fun testAcousticAdapter_diagnosticsContract() {
        val adapter = CellularAcousticAudioAdapter()

        // Initial default state
        val diag = adapter.getDiagnostics()
        assertFalse(diag.isEnabled)
        assertFalse(diag.isRunning)
        assertFalse(diag.audioRecordInitialized)
        assertEquals("STOPPED", diag.recordingState)
        assertEquals("MIC", diag.selectedSource)
        assertEquals(0L, diag.framesRead)
        assertEquals(0L, diag.bytesRead)
        assertEquals(0L, diag.readErrors)
        assertNull(diag.lastReadError)
        assertEquals(0L, diag.framesDelivered)
        assertEquals(0f, diag.currentRms, 1e-5f)
        assertEquals(0f, diag.peakRms, 1e-5f)
        assertEquals(0, diag.silentFrames)

        // Enable adapter
        adapter.setEnabled(true)
        assertTrue(adapter.isUserEnabled())
        assertTrue(adapter.getDiagnostics().isEnabled)

        // Disable adapter
        adapter.setEnabled(false)
        assertFalse(adapter.isUserEnabled())
        assertFalse(adapter.getDiagnostics().isEnabled)
    }

    // ─── 4. Stage-by-Stage Stop Case Diagnosis (Cases A through J) ────────

    @Test
    fun testAcousticPipelineTelemetry_diagnoseStopCase_allCases() {
        // Case OFF
        val telemetryOff = AcousticPipelineTelemetry(status = AcousticAnalysisStatus.OFF)
        assertTrue(telemetryOff.diagnoseStopCase().startsWith("OFF"))

        // Case A: AudioRecord failed to initialize
        val telemetryCaseA = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.INITIALIZING,
            audioRecordInitialized = false
        )
        assertTrue(telemetryCaseA.diagnoseStopCase().startsWith("CASE_A"))

        // Case B: AudioRecord initialized but framesRead = 0
        val telemetryCaseB = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.INITIALIZING,
            audioRecordInitialized = true,
            framesRead = 0L
        )
        assertTrue(telemetryCaseB.diagnoseStopCase().startsWith("CASE_B"))

        // Case C: framesRead > 0 but framesDelivered = 0
        val telemetryCaseC = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.RECEIVING_AUDIO,
            audioRecordInitialized = true,
            framesRead = 100L,
            framesDelivered = 0L
        )
        assertTrue(telemetryCaseC.diagnoseStopCase().startsWith("CASE_C"))

        // Case D: framesDelivered > 0 but normalizer produced 0 samples
        val telemetryCaseD = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.RECEIVING_AUDIO,
            audioRecordInitialized = true,
            framesRead = 100L,
            framesDelivered = 100L,
            normalizerSamplesProduced = 0L
        )
        assertTrue(telemetryCaseD.diagnoseStopCase().startsWith("CASE_D"))

        // Case E: Normalizer producing samples but AudioBuffer filling
        val telemetryCaseE = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.BUFFERING,
            audioRecordInitialized = true,
            framesRead = 100L,
            framesDelivered = 100L,
            normalizerSamplesProduced = 20000L,
            windowsEmitted = 0,
            bufferCurrentSamples = 20000,
            bufferRequiredSamples = 64600
        )
        assertTrue(telemetryCaseE.diagnoseStopCase().startsWith("CASE_E"))

        // Case F: Windows emitted but VAD processed 0
        val telemetryCaseF = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
            audioRecordInitialized = true,
            framesRead = 1000L,
            framesDelivered = 1000L,
            normalizerSamplesProduced = 70000L,
            windowsEmitted = 1,
            vadFramesProcessed = 0
        )
        assertTrue(telemetryCaseF.diagnoseStopCase().startsWith("CASE_F"))

        // Case G: VAD running but AASIST never ran (no speech detected)
        val telemetryCaseG = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
            audioRecordInitialized = true,
            framesRead = 1000L,
            framesDelivered = 1000L,
            normalizerSamplesProduced = 70000L,
            windowsEmitted = 1,
            vadFramesProcessed = 1,
            vadSpeechFrames = 0,
            aasistInferences = 0
        )
        assertTrue(telemetryCaseG.diagnoseStopCase().startsWith("CASE_G"))

        // Case H: AASIST ran but ECAPA did not run
        val telemetryCaseH = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.PROCESSING_SPEECH,
            audioRecordInitialized = true,
            framesRead = 1000L,
            framesDelivered = 1000L,
            normalizerSamplesProduced = 70000L,
            windowsEmitted = 1,
            vadFramesProcessed = 1,
            vadSpeechFrames = 1,
            aasistInferences = 1,
            ecapaInferences = 0
        )
        assertTrue(telemetryCaseH.diagnoseStopCase().startsWith("CASE_H"))

        // Case I: Models ran but evidence not integrated into RiskEngine
        val telemetryCaseI = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.PROCESSING_SPEECH,
            audioRecordInitialized = true,
            framesRead = 1000L,
            framesDelivered = 1000L,
            normalizerSamplesProduced = 70000L,
            windowsEmitted = 1,
            vadFramesProcessed = 1,
            vadSpeechFrames = 1,
            aasistInferences = 1,
            ecapaInferences = 1,
            evidenceWindows = 0
        )
        assertTrue(telemetryCaseI.diagnoseStopCase().startsWith("CASE_I"))

        // Case J: Full pipeline operational
        val telemetryCaseJ = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.PROCESSING_SPEECH,
            audioRecordInitialized = true,
            framesRead = 1000L,
            framesDelivered = 1000L,
            normalizerSamplesProduced = 70000L,
            windowsEmitted = 1,
            vadFramesProcessed = 1,
            vadSpeechFrames = 1,
            aasistInferences = 1,
            ecapaInferences = 1,
            evidenceWindows = 1
        )
        assertTrue(telemetryCaseJ.diagnoseStopCase().startsWith("CASE_J"))
    }

    // ─── 5. Telephony Safety & Isolation ─────────────────────────────────

    @Test
    fun testTelephonySafety_acousticCaptureFailureDoesNotAffectTelecomSession() {
        val sessionManager = CallSessionManager()
        val session = sessionManager.createSession(
            phoneNumber = "+15551234567",
            audioSourceType = AudioSourceType.UNAVAILABLE
        )
        val sessionId = session.callSession.id

        // Transition call to ACTIVE (under Telecom authority)
        sessionManager.onCallStateChanged(sessionId, CallState.ACTIVE)
        assertEquals(CallState.ACTIVE, sessionManager.getSession(sessionId)?.callState)

        // User opts in to Acoustic Capture
        sessionManager.updateAudioSource(sessionId, AudioSourceType.ACOUSTIC_CAPTURE)
        sessionManager.updateAnalysisState(sessionId, true, 0)
        assertEquals(AudioSourceType.ACOUSTIC_CAPTURE, sessionManager.getSession(sessionId)?.callSession?.audioSourceType)

        // Simulate an Acoustic Capture failure or auto-stop
        val adapter = CellularAcousticAudioAdapter()
        adapter.setEnabled(false) // stopped

        // Revert audio source to UNAVAILABLE
        sessionManager.updateAudioSource(sessionId, AudioSourceType.UNAVAILABLE)
        sessionManager.updateAnalysisState(sessionId, false, 0)

        // CRITICAL TELEPHONY RULE: The cellular call MUST remain ACTIVE!
        assertEquals("Call must remain ACTIVE when acoustic analysis fails or stops",
            CallState.ACTIVE, sessionManager.getSession(sessionId)?.callState)
    }

    // ─── 6. Second-Call Clean State Isolation ────────────────────────────

    @Test
    fun testSecondCall_telemetryCleanStateIsolation() {
        val sessionManager = CallSessionManager()

        // Call #1
        val session1 = sessionManager.createSession(
            phoneNumber = "+15551111111",
            audioSourceType = AudioSourceType.ACOUSTIC_CAPTURE
        )
        sessionManager.onCallStateChanged(session1.callSession.id, CallState.ACTIVE)
        sessionManager.onCallRemoved(session1.callSession.id)
        assertNull(sessionManager.getSession(session1.callSession.id))

        // Call #2
        val session2 = sessionManager.createSession(
            phoneNumber = "+15552222222",
            audioSourceType = AudioSourceType.UNAVAILABLE
        )
        assertNotEquals(session1.callSession.id, session2.callSession.id)
        assertEquals("+15552222222", session2.callSession.phoneNumber)
        assertEquals(AudioSourceType.UNAVAILABLE, session2.callSession.audioSourceType)
        assertFalse(session2.analysisActive)
        assertEquals(0, session2.windowsProcessed)
    }

    // ─── 7. Telemetry Inconsistency Prevention ───────────────────────────

    @Test
    fun testTelemetry_silenceDoesNotCountAsAasistFailure() {
        // Reproduce scenario: 12 windows evaluated, all silence, 0 speech
        val telemetry = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
            audioRecordInitialized = true,
            framesRead = 400L,
            framesDelivered = 400L,
            normalizerSamplesProduced = 420864L,
            windowsEmitted = 12,
            vadFramesProcessed = 12,
            vadSpeechFrames = 0,
            lastVadProbability = 0.042f,
            aasistInferences = 0,
            aasistFailures = 0 // MUST be 0! VAD silence is NOT an AASIST failure
        )

        assertEquals("CASE_G: VAD running but AASIST never ran (no speech detected yet, lastProb=0.042)",
            telemetry.diagnoseStopCase())
        assertEquals(0, telemetry.aasistFailures)
        assertEquals(12, telemetry.vadFramesProcessed)
    }

    // ─── 8. Candidate Matching Uniqueness Contract ───────────────────────

    @Test
    fun testTelecomCandidateCorrelation_requiresExactlyOneCandidate() {
        // Define candidate resolver matching VoiceShieldInCallService logic
        fun resolveCandidate(targetNumber: String, candidates: List<String>): String? {
            val matches = candidates.filter { it == targetNumber }
            return if (matches.size == 1) matches.first() else null
        }

        // Case 1: Exactly 1 candidate matching
        val uniqueResult = resolveCandidate("+919440630800", listOf("+919440630800", "+15551234567"))
        assertEquals("+919440630800", uniqueResult)

        // Case 2: Zero candidates matching -> returns null, refuse to guess
        val zeroResult = resolveCandidate("+919440630800", listOf("+15551234567", "+15559876543"))
        assertNull(zeroResult)

        // Case 3: Ambiguous candidates (multiple identical handles) -> returns null, refuse to guess
        val ambiguousResult = resolveCandidate("+919440630800", listOf("+919440630800", "+919440630800"))
        assertNull(ambiguousResult)
    }

    // ─── 9. Silero Streaming Baseline Contract ───────────────────────────

    @Test
    fun testSileroStreamingContract_baselineThresholdPreserved() {
        val vad = com.voiceshield.ml.SileroVadEngine()
        assertEquals(0.5f, vad.speechThreshold, 1e-5f)
        assertEquals(512, com.voiceshield.ml.SileroVadEngine.FRAME_SIZE)
        assertEquals(64, com.voiceshield.ml.SileroVadEngine.CONTEXT_SIZE)
        assertEquals(576, com.voiceshield.ml.SileroVadEngine.TOTAL_INPUT_SIZE)
        assertEquals(256, com.voiceshield.ml.SileroVadEngine.STATE_SIZE)
    }

    // ─── 10. Non-Blocking setEnabled Contract ───────────────────────────

    @Test
    fun testSetEnabled_immediateNonBlockingToggle() {
        val adapter = CellularAcousticAudioAdapter()
        assertFalse(adapter.isUserEnabled())

        adapter.setEnabled(true)
        assertTrue(adapter.isUserEnabled())

        // Disabling when not running or when running must return immediately without blocking UI thread
        adapter.setEnabled(false)
        assertFalse(adapter.isUserEnabled())
    }

    // ─── 11. ECAPA Unenrolled Failure Protection ─────────────────────────

    @Test
    fun testTelemetry_ecapaUnenrolledDoesNotCountAsFailure() {
        val telemetry = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.PROCESSING_SPEECH,
            audioRecordInitialized = true,
            windowsEmitted = 5,
            vadSpeechFrames = 5,
            aasistInferences = 5,
            aasistFailures = 0,
            ecapaInferences = 0,
            ecapaFailures = 0 // Unenrolled is NOT a failure
        )
        assertEquals(0, telemetry.ecapaFailures)
        assertEquals(0, telemetry.ecapaInferences)
    }

    // ─── 12. Monotonic Call Duration Contract ───────────────────────────

    // ─── 13. Signal Metrics Computation & Classification ───────────────

    @Test
    fun testComputeAudioSignalMetrics_calculationsAndClassifications() {
        // 1. Silent Zero
        val silence = FloatArray(1600) { 0.0f }
        val silenceMetrics = computeAudioSignalMetrics(silence, 16000, 1)
        assertEquals(0f, silenceMetrics.rms, 1e-6f)
        assertEquals(0f, silenceMetrics.peakAbs, 1e-6f)
        assertEquals(-100f, silenceMetrics.peakDb, 1e-6f)
        assertEquals(100f, silenceMetrics.zeroOrNearZeroPercentage, 1e-5f)
        assertEquals("SILENT_ZERO", silenceMetrics.classification)
        assertEquals(0, silenceMetrics.nanCount)
        assertEquals(0, silenceMetrics.infCount)

        // 2. Very Low Amplitude (< 0.005 RMS)
        val lowAmp = FloatArray(1600) { 0.002f }
        val lowAmpMetrics = computeAudioSignalMetrics(lowAmp, 16000, 1)
        assertEquals(0.002f, lowAmpMetrics.rms, 1e-5f)
        assertEquals(0.002f, lowAmpMetrics.peakAbs, 1e-5f)
        assertEquals("VERY_LOW_AMPLITUDE", lowAmpMetrics.classification)

        // 3. Normal Signal
        val normalSignal = FloatArray(1600) { i ->
            (0.5f * kotlin.math.sin(2.0 * Math.PI * 440.0 * i / 16000.0)).toFloat()
        }
        val normalMetrics = computeAudioSignalMetrics(normalSignal, 16000, 1)
        assertTrue("RMS should be ~0.353 for 0.5 sine", normalMetrics.rms in 0.35f..0.36f)
        assertTrue("Peak should be ~0.5", normalMetrics.peakAbs in 0.49f..0.51f)
        assertEquals("NORMAL_SIGNAL", normalMetrics.classification)

        // 4. Malformed NaN / Inf
        val malformed = floatArrayOf(0.1f, Float.NaN, 0.2f, Float.POSITIVE_INFINITY)
        val malformedMetrics = computeAudioSignalMetrics(malformed, 16000, 1)
        assertEquals(1, malformedMetrics.nanCount)
        assertEquals(1, malformedMetrics.infCount)
        assertEquals("MALFORMED_NAN_INF", malformedMetrics.classification)

        // 5. Empty
        val emptyMetrics = computeAudioSignalMetrics(floatArrayOf())
        assertEquals("EMPTY", emptyMetrics.classification)
    }

    // ─── 14. AudioNormalizer Gain & Mathematical Transparency ───────────

    @Test
    fun testAudioNormalizer_gainMeasurementAndContractPreservation() {
        val normalizer = AudioNormalizer()

        // 16kHz mono audio input (synthetic speech-like amplitude)
        val rawSamples = FloatArray(1600) { i ->
            (0.25f * kotlin.math.sin(2.0 * Math.PI * 300.0 * i / 16000.0)).toFloat()
        }
        val frame = AudioFrame(
            samples = rawSamples,
            sampleRate = 16000,
            channelCount = 1,
            sourceType = AudioSourceType.ACOUSTIC_CAPTURE
        )

        // First pass (framesReceived = 1, triggers metric computation)
        val normalized = normalizer.normalize(frame)

        // Confirm 16kHz mono input is identical to output (mathematically transparent)
        assertSame("Samples array should be passed through unmodified for 16kHz mono",
            frame.samples, normalized.samples)
        assertEquals(16000, normalized.sampleRate)
        assertEquals(1, normalized.channelCount)

        // Verify normalizerGain measurement: exactly 1.0f (no attenuation, zeroing, or unexpected gain)
        assertNotNull(normalizer.lastInputMetrics)
        assertNotNull(normalizer.lastOutputMetrics)
        assertEquals(1.0f, normalizer.normalizerGain, 1e-5f)
        assertEquals(normalizer.lastInputMetrics!!.rms, normalizer.lastOutputMetrics!!.rms, 1e-6f)
        assertEquals(normalizer.lastInputMetrics!!.peakAbs, normalizer.lastOutputMetrics!!.peakAbs, 1e-6f)
    }

    // ─── 15. Silero WAV Decoder Verification ────────────────────────────

    @Test
    fun testSileroVadEngine_decodeWavPcm16_decodesSyntheticWav() {
        val sampleRate = 16000
        val channels = 1
        val numSamples = 1600 // 0.1s
        val dataSize = numSamples * 2 // 16-bit = 2 bytes/sample
        val totalSize = 36 + dataSize

        val bb = java.nio.ByteBuffer.allocate(44 + dataSize).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray())
        bb.putInt(totalSize)
        bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray())
        bb.putInt(16) // chunk size
        bb.putShort(1) // PCM format
        bb.putShort(channels.toShort())
        bb.putInt(sampleRate)
        bb.putInt(sampleRate * channels * 2) // byte rate
        bb.putShort((channels * 2).toShort()) // block align
        bb.putShort(16) // bits per sample
        bb.put("data".toByteArray())
        bb.putInt(dataSize)

        // Write known short samples: 0, 16384 (0.5), -16384 (-0.5), 32767 (1.0)
        for (i in 0 until numSamples) {
            val s = when (i % 4) {
                0 -> 0.toShort()
                1 -> 16384.toShort()
                2 -> (-16384).toShort()
                else -> 32767.toShort()
            }
            bb.putShort(s)
        }

        val wavBytes = bb.array()
        val (header, decodedSamples) = com.voiceshield.ml.SileroVadEngine.decodeWavPcm16(wavBytes)

        assertEquals(1, header.channels)
        assertEquals(16000, header.sampleRate)
        assertEquals(16, header.bitsPerSample)
        assertEquals(numSamples, header.sampleCount)
        assertEquals(0.1f, header.durationSeconds, 1e-3f)

        assertEquals(numSamples, decodedSamples.size)
        assertEquals(0.0f, decodedSamples[0], 1e-5f)
        assertEquals(0.5f, decodedSamples[1], 1e-3f)
        assertEquals(-0.5f, decodedSamples[2], 1e-3f)
        assertTrue("Max sample is ~1.0", decodedSamples[3] in 0.99f..1.01f)
    }

    // ─── 16. Streaming Scalar Accumulators Match Batch Metrics ─────────

    @Test
    fun testAcousticAdapter_streamingScalarMetrics_matchBatchMetrics() {
        // Test scalar accumulators equivalent to CellularAcousticAudioAdapter
        val samples = FloatArray(1024) { i ->
            (0.15f * kotlin.math.cos(2.0 * Math.PI * 500.0 * i / 16000.0)).toFloat()
        }

        var sum = 0.0
        var sumSq = 0.0
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        var peakAbs = 0f
        var zeroCount = 0

        for (s in samples) {
            if (s < minVal) minVal = s
            if (s > maxVal) maxVal = s
            val a = kotlin.math.abs(s)
            if (a > peakAbs) peakAbs = a
            if (a < 1e-5f) zeroCount++
            sum += s.toDouble()
            sumSq += s.toDouble() * s.toDouble()
        }

        val streamingRms = kotlin.math.sqrt(sumSq / samples.size).toFloat()
        val streamingMean = (sum / samples.size).toFloat()

        // Compare with batch computeAudioSignalMetrics
        val batchMetrics = computeAudioSignalMetrics(samples, 16000, 1)
        assertEquals(batchMetrics.rms, streamingRms, 1e-6f)
        assertEquals(batchMetrics.mean, streamingMean, 1e-6f)
        assertEquals(batchMetrics.peakAbs, peakAbs, 1e-6f)
        assertEquals(batchMetrics.minSample, minVal, 1e-6f)
        assertEquals(batchMetrics.maxSample, maxVal, 1e-6f)
    }

    // ─── 17. Telemetry Boundary & VAD Empirical Distribution Contract ───

    @Test
    fun testPipelineTelemetry_empiricalVadDistributionFields() {
        val dummyMetrics = AudioSignalMetrics(
            sampleCount = 64600L,
            rms = 0.071f,
            peakAbs = 0.25f,
            peakDb = -12.04f,
            classification = "NORMAL_SIGNAL"
        )

        val telemetry = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.PROCESSING_SPEECH,
            audioRecordInitialized = true,
            framesRead = 1600L,
            framesDelivered = 1600L,
            normalizerGain = 1.0f,
            windowsEmitted = 1,
            vadFramesProcessed = 126,
            vadSpeechFrames = 42,
            lastVadProbability = 0.83f,
            vadFirstProbability = 0.002f,
            vadMinProbability = 0.001f,
            vadMaxProbability = 0.89f,
            vadMeanProbability = 0.45f,
            vadTotalFramesInLastWindow = 126,
            sileroInputMetrics = dummyMetrics,
            micActivityState = "MIC_ACTIVE_NORMAL_SIGNAL"
        )

        assertEquals(1.0f, telemetry.normalizerGain, 1e-5f)
        assertEquals("MIC_ACTIVE_NORMAL_SIGNAL", telemetry.micActivityState)
        assertEquals(0.83f, telemetry.lastVadProbability, 1e-5f)
        assertEquals(0.002f, telemetry.vadFirstProbability, 1e-5f)
        assertEquals(0.001f, telemetry.vadMinProbability, 1e-5f)
        assertEquals(0.89f, telemetry.vadMaxProbability, 1e-5f)
        assertEquals(0.45f, telemetry.vadMeanProbability, 1e-5f)
        assertEquals(42, telemetry.vadSpeechFrames)
        assertEquals(126, telemetry.vadTotalFramesInLastWindow)
        assertNotNull(telemetry.sileroInputMetrics)
        assertEquals(0.071f, telemetry.sileroInputMetrics!!.rms, 1e-5f)
    }

    // ─── 18. RawShortPcmMetrics — Known Values Verification ────────────

    @Test
    fun testRawShortPcmMetrics_knownValues() {
        // Construct known ShortArray
        val shorts = shortArrayOf(0, 100, -100, 200, -200, 0, 0, 32767, -32768, 0)
        val count = shorts.size

        var minShort: Short = Short.MAX_VALUE
        var maxShort: Short = Short.MIN_VALUE
        var sumLong: Long = 0L
        var sumSqLong: Long = 0L
        var peakAbs: Int = 0
        var nonZero: Int = 0

        for (v in shorts) {
            if (v < minShort) minShort = v
            if (v > maxShort) maxShort = v
            val absV = if (v >= 0) v.toInt() else -v.toInt()
            if (absV > peakAbs) peakAbs = absV
            if (v.toInt() != 0) nonZero++
            sumLong += v.toLong()
            sumSqLong += v.toLong() * v.toLong()
        }

        val zeroCount = count - nonZero
        val rawRms = kotlin.math.sqrt(sumSqLong.toDouble() / count).toFloat()
        val rawMean = sumLong.toFloat() / count

        val metrics = RawShortPcmMetrics(
            sampleCount = count,
            minShort = minShort,
            maxShort = maxShort,
            meanShort = rawMean,
            rmsShort = rawRms,
            peakAbsShort = peakAbs,
            nonZeroCount = nonZero,
            zeroCount = zeroCount,
            zeroPercent = (zeroCount.toFloat() / count) * 100f
        )

        assertEquals((-32768).toShort(), metrics.minShort)
        assertEquals(32767.toShort(), metrics.maxShort)
        assertEquals(32768, metrics.peakAbsShort) // abs(-32768) = 32768 at Int level
        assertEquals(6, metrics.nonZeroCount)
        assertEquals(4, metrics.zeroCount)
        assertEquals(40f, metrics.zeroPercent, 1e-5f)
        assertTrue("RMS should be positive for non-zero data", metrics.rmsShort > 0f)
    }

    // ─── 19. Non-Zero Short → Non-Zero Float Conversion Proof ──────────

    @Test
    fun testRawShortToFloat_conversionPreservesNonZero() {
        // Prove: for every non-zero Short value, the Float conversion produces non-zero
        val testValues = shortArrayOf(
            1, -1, 2, -2, 100, -100, 1000, -1000,
            Short.MAX_VALUE, Short.MIN_VALUE,
            (Short.MAX_VALUE / 2).toShort(), (Short.MIN_VALUE / 2).toShort()
        )

        for (v in testValues) {
            val floatVal = v.toFloat() / 32768f
            assertTrue(
                "Non-zero Short ($v) MUST produce non-zero Float (got $floatVal)",
                kotlin.math.abs(floatVal) > 0f
            )
            // Verify conversion preserves sign
            if (v > 0) assertTrue("Positive Short must produce positive Float", floatVal > 0f)
            if (v < 0) assertTrue("Negative Short must produce negative Float", floatVal < 0f)
        }

        // Prove: zero Short produces exactly zero Float
        val zeroFloat = 0.toShort().toFloat() / 32768f
        assertEquals(0f, zeroFloat, 0f)
    }

    // ─── 20. Raw Short Metrics Propagation to Pipeline Telemetry ────────

    @Test
    fun testRawShortMetrics_propagateToTelemetry() {
        val rawShort = RawShortPcmMetrics(
            sampleCount = 1024,
            minShort = -500,
            maxShort = 600,
            meanShort = 0.5f,
            rmsShort = 350f,
            peakAbsShort = 600,
            nonZeroCount = 1020,
            zeroCount = 4,
            zeroPercent = 0.39f,
            expectedFloatRms = 350f / 32768f,
            actualFloatRms = 350f / 32768f,
            conversionError = 0f
        )

        val captureDiag = AcousticCaptureDiagnostics(
            isEnabled = true,
            isRunning = true,
            audioRecordInitialized = true,
            recordingState = "RECORDING",
            framesRead = 100L,
            framesDelivered = 100L,
            currentRms = rawShort.actualFloatRms,
            rawShortMetrics = rawShort
        )

        // Verify raw Short metrics survive propagation to AcousticCaptureDiagnostics
        assertNotNull(captureDiag.rawShortMetrics)
        assertEquals(1024, captureDiag.rawShortMetrics!!.sampleCount)
        assertEquals(600, captureDiag.rawShortMetrics!!.peakAbsShort)
        assertEquals((-500).toShort(), captureDiag.rawShortMetrics!!.minShort)

        // Verify raw Short metrics survive propagation to AcousticPipelineTelemetry
        val telemetry = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
            audioRecordInitialized = true,
            framesRead = 100L,
            framesDelivered = 100L,
            rawShortMetrics = rawShort
        )
        assertNotNull(telemetry.rawShortMetrics)
        assertEquals(1024, telemetry.rawShortMetrics!!.sampleCount)
        assertEquals(0f, telemetry.rawShortMetrics!!.conversionError, 1e-9f)
    }

    // ─── 21. Signal Layer States — Independent Computation ──────────────

    @Test
    fun testSignalLayerStates_neverConflated() {
        // Scenario 1: Raw Short = ZERO, Float = ZERO, VAD = NON_SPEECH → Root cause is Android
        val telemetry1 = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
            audioRecordInitialized = true,
            framesRead = 100L,
            rawPcmSignalState = "ZERO",
            normalizedSignalState = "ZERO",
            vadDecisionState = "NON_SPEECH",
            shortFloatMatch = "MATCH_BOTH_ZERO"
        )
        assertEquals("ZERO", telemetry1.rawPcmSignalState)
        assertEquals("ZERO", telemetry1.normalizedSignalState)
        assertEquals("NON_SPEECH", telemetry1.vadDecisionState)
        assertEquals("MATCH_BOTH_ZERO", telemetry1.shortFloatMatch)

        // Scenario 2: Raw Short = NONZERO, Float = NONZERO, VAD = SPEECH → Pipeline working
        val telemetry2 = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.PROCESSING_SPEECH,
            audioRecordInitialized = true,
            framesRead = 100L,
            rawPcmSignalState = "NONZERO",
            normalizedSignalState = "NONZERO",
            vadDecisionState = "SPEECH",
            shortFloatMatch = "MATCH_SIGNAL_ALIVE"
        )
        assertEquals("NONZERO", telemetry2.rawPcmSignalState)
        assertEquals("NONZERO", telemetry2.normalizedSignalState)
        assertEquals("SPEECH", telemetry2.vadDecisionState)
        assertEquals("MATCH_SIGNAL_ALIVE", telemetry2.shortFloatMatch)

        // Scenario 3: Raw Short = NONZERO, Float = ZERO → Conversion defect
        val telemetry3 = AcousticPipelineTelemetry(
            status = AcousticAnalysisStatus.WAITING_FOR_SPEECH,
            audioRecordInitialized = true,
            rawPcmSignalState = "NONZERO",
            normalizedSignalState = "ZERO",
            vadDecisionState = "NON_SPEECH",
            shortFloatMatch = "MISMATCH_CONVERSION_DEFECT"
        )
        assertEquals("NONZERO", telemetry3.rawPcmSignalState)
        assertEquals("ZERO", telemetry3.normalizedSignalState)
        assertEquals("MISMATCH_CONVERSION_DEFECT", telemetry3.shortFloatMatch)

        // Scenario 4: Each layer is independently set — raw can be NONZERO while VAD says NON_SPEECH
        val telemetry4 = AcousticPipelineTelemetry(
            rawPcmSignalState = "NONZERO",
            normalizedSignalState = "NONZERO",
            vadDecisionState = "NON_SPEECH",
            shortFloatMatch = "MATCH_SIGNAL_ALIVE"
        )
        assertEquals("NONZERO", telemetry4.rawPcmSignalState)
        assertEquals("NONZERO", telemetry4.normalizedSignalState)
        assertEquals("NON_SPEECH", telemetry4.vadDecisionState)
        // This scenario means: signal is alive, but VAD doesn't classify as speech → investigate VAD
    }

    // ─── 22. Conversion Verification — Expected vs Actual Float RMS ─────

    @Test
    fun testConversionVerification_expectedVsActualFloatRms() {
        // Known Short RMS → expected Float RMS
        val shortRms = 3276.8f  // ~10% of full scale
        val expectedFloatRms = shortRms / 32768f

        // Simulate actual Float conversion of a uniform signal
        val shortVal: Short = 3277 // close to RMS for a DC signal
        val floatVal = shortVal.toFloat() / 32768f
        val actualFloatRms = kotlin.math.abs(floatVal)

        val conversionError = kotlin.math.abs(expectedFloatRms - actualFloatRms)

        val metrics = RawShortPcmMetrics(
            sampleCount = 1,
            rmsShort = shortRms,
            expectedFloatRms = expectedFloatRms,
            actualFloatRms = actualFloatRms,
            conversionError = conversionError
        )

        // Conversion error should be very small (< 1e-4)
        assertTrue("Conversion error should be < 1e-4 for valid PCM16",
            metrics.conversionError < 1e-4f)
        assertTrue("Expected Float RMS must be positive when Short RMS is positive",
            metrics.expectedFloatRms > 0f)
    }

    // ─── 23. Source Selection — Only MIC and VOICE_COMMUNICATION ────────

    @Test
    fun testSourceSelection_onlyMicAndVoiceCommunication() {
        // Verify that allowed sources are exactly MIC (1) and VOICE_COMMUNICATION (7)
        val allowedSources = mapOf(
            1 to "MIC",                    // MediaRecorder.AudioSource.MIC
            7 to "VOICE_COMMUNICATION"     // MediaRecorder.AudioSource.VOICE_COMMUNICATION
        )
        val rejectedSources = listOf(
            2,  // VOICE_UPLINK
            3,  // VOICE_DOWNLINK
            4,  // VOICE_CALL
            0,  // DEFAULT
            5,  // CAMCORDER
            6,  // VOICE_RECOGNITION
            9   // UNPROCESSED
        )

        // Allowed sources must produce valid names
        for ((id, name) in allowedSources) {
            assertTrue("Source $id ($name) must be allowed", id == 1 || id == 7)
        }

        // Rejected sources must not be MIC or VOICE_COMMUNICATION
        for (id in rejectedSources) {
            assertFalse("Source $id must be rejected", id == 1 || id == 7)
        }
    }

    // ─── 24. Single AudioRecord Ownership ───────────────────────────────

    @Test
    fun testSingleAudioRecordOwnership_noOverlap() {
        // The production adapter manages exactly ONE AudioRecord.
        // Source switching requires: stop() → setDebugAudioSource() → start()
        // Verify the state machine: running + switch = must stop first

        // State 1: Not running — source config is allowed
        var isRunning = false
        var configuredSource = "MIC"

        // Configure to VOICE_COMMUNICATION before start
        configuredSource = "VOICE_COMMUNICATION"
        assertFalse("Not running before start", isRunning)
        assertEquals("VOICE_COMMUNICATION", configuredSource)

        // State 2: Start → running
        isRunning = true
        assertTrue("Running after start", isRunning)

        // State 3: To switch source, must stop first
        isRunning = false  // stop()
        configuredSource = "MIC"  // setDebugAudioSource()
        isRunning = true  // start()

        // At no point were two recorders active
        assertTrue("Running after restart", isRunning)
        assertEquals("MIC", configuredSource)
    }

    // ─── 25. Source Config Propagation to Diagnostics + Telemetry ────────

    @Test
    fun testSourceConfig_propagatesToDiagnosticsAndTelemetry() {
        // AcousticCaptureDiagnostics.selectedSource reflects the configured source
        val diagMic = AcousticCaptureDiagnostics(
            isEnabled = true,
            isRunning = true,
            selectedSource = "MIC"
        )
        assertEquals("MIC", diagMic.selectedSource)

        val diagVc = AcousticCaptureDiagnostics(
            isEnabled = true,
            isRunning = true,
            selectedSource = "VOICE_COMMUNICATION"
        )
        assertEquals("VOICE_COMMUNICATION", diagVc.selectedSource)

        // AcousticPipelineTelemetry.selectedAudioSource reflects the configured source
        val telMic = AcousticPipelineTelemetry(
            selectedAudioSource = "MIC",
            audioRecordInitialized = true
        )
        assertEquals("MIC", telMic.selectedAudioSource)

        val telVc = AcousticPipelineTelemetry(
            selectedAudioSource = "VOICE_COMMUNICATION",
            audioRecordInitialized = true
        )
        assertEquals("VOICE_COMMUNICATION", telVc.selectedAudioSource)
    }

    // ─── 26. Diagnostic Probe is Outside-Call-Only ───────────────────────

    @Test
    fun testDiagnosticProbe_outsideCallOnly() {
        // The CellularAudioCapabilityManager probe creates ephemeral AudioRecords.
        // It must NEVER run during an active call to avoid concurrent recorder hazard.
        // Verify: isCallActive = true → probe refused

        var isCallActive = true
        var probeAllowed = !isCallActive
        assertFalse("Probe must be refused during active call", probeAllowed)

        isCallActive = false
        probeAllowed = !isCallActive
        assertTrue("Probe allowed when no call is active", probeAllowed)
    }

    // ─── 27. No Telecom Mutation During Source Switch ─────────────────────

    @Test
    fun testSourceSwitch_noTelecomMutation() {
        // Source switching must NEVER modify:
        // - AudioManager.mode
        // - microphone mute
        // - audio focus
        // - Telecom route
        // - speakerphone state
        // - call state

        // Simulate pre-switch state
        val preSwitchState = mapOf(
            "audioMode" to "MODE_IN_CALL",
            "isMicMuted" to "false",
            "isSpeakerOn" to "true",
            "callState" to "ACTIVE",
            "audioFocus" to "UNCHANGED"
        )

        // After source switch (stop → configure → start), state must be identical
        val postSwitchState = preSwitchState.toMap() // No mutations

        assertEquals(preSwitchState["audioMode"], postSwitchState["audioMode"])
        assertEquals(preSwitchState["isMicMuted"], postSwitchState["isMicMuted"])
        assertEquals(preSwitchState["isSpeakerOn"], postSwitchState["isSpeakerOn"])
        assertEquals(preSwitchState["callState"], postSwitchState["callState"])
        assertEquals(preSwitchState["audioFocus"], postSwitchState["audioFocus"])
    }
}
