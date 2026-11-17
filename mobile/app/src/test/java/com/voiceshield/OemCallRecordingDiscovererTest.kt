package com.voiceshield

import com.voiceshield.audio.OemCallRecordingDiscoverer
import com.voiceshield.audio.OemDiscoveryResult
import com.voiceshield.telecom.CallSessionManager
import com.voiceshield.telecom.CallState
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class OemCallRecordingDiscovererTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun testPrototypeDirectory_unavailableWhenMissing() = runBlocking {
        val nonExistentDir = File(tempFolder.root, "non_existent_miui_dir")
        val discoverer = OemCallRecordingDiscoverer(customDirectory = nonExistentDir)

        assertNull("Directory should be unresolved if path does not exist", discoverer.resolvePrototypeDirectory())

        val result = discoverer.discoverRecordingWithRetry(
            phoneNumber = "+919876543210",
            callStartTime = System.currentTimeMillis() - 30_000L,
            callEndTime = System.currentTimeMillis(),
            maxWaitMs = 50L,
            pollIntervalMs = 10L
        )

        assertTrue(result is OemDiscoveryResult.Unavailable)
        val unavailable = result as OemDiscoveryResult.Unavailable
        assertEquals(OemDiscoveryResult.Unavailable.ERROR_CODE_UNAVAILABLE, unavailable.code)
        assertTrue(unavailable.reason.contains("POST_CALL_RECORDING_DISCOVERY_UNAVAILABLE"))
    }

    @Test
    fun testCandidateMatching_exactPhoneNumberAndTimingCorrelation() {
        val callDir = tempFolder.newFolder("miui_call_rec")
        val now = System.currentTimeMillis()
        val callStart = now - 45_000L
        val callEnd = now

        // Create a matching recording file (created 2 seconds after call ended)
        val matchingFile = File(callDir, "+919876543210(20260904_210000).mp3")
        matchingFile.writeBytes(ByteArray(4096) { 0x55 })
        matchingFile.setLastModified(callEnd + 2000L)

        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)
        val candidate = discoverer.findBestCandidate(
            directory = callDir,
            phoneNumber = "+919876543210",
            callStartTime = callStart,
            callEndTime = callEnd
        )

        assertNotNull("Should find matching candidate", candidate)
        assertEquals(matchingFile.name, candidate!!.file.name)
        assertTrue("Score should reflect phone number and close end time correlation", candidate.score >= 80)
        assertTrue("Reason should mention phone number", candidate.reason.contains("9876543210"))
    }

    @Test
    fun testCandidateMatching_rejectsOldRecordings() {
        val callDir = tempFolder.newFolder("miui_call_rec_old")
        val now = System.currentTimeMillis()
        val callStart = now - 30_000L
        val callEnd = now

        // Create an old recording from yesterday with the same phone number
        val oldFile = File(callDir, "9876543210_yesterday.mp3")
        oldFile.writeBytes(ByteArray(4096) { 0x55 })
        oldFile.setLastModified(now - 86_400_000L) // 24 hours ago

        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)
        val candidate = discoverer.findBestCandidate(
            directory = callDir,
            phoneNumber = "+919876543210",
            callStartTime = callStart,
            callEndTime = callEnd
        )

        assertNull("Old recording must not be selected as a candidate", candidate)
    }

    @Test
    fun testCandidateMatching_rejectsTinyOrCorruptedFiles() {
        val callDir = tempFolder.newFolder("miui_call_rec_tiny")
        val now = System.currentTimeMillis()
        val callStart = now - 30_000L
        val callEnd = now

        // 100 bytes is below the 1024 bytes minimum threshold
        val tinyFile = File(callDir, "9876543210_empty.mp3")
        tinyFile.writeBytes(ByteArray(100) { 0x00 })
        tinyFile.setLastModified(callEnd + 1000L)

        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)
        val candidate = discoverer.findBestCandidate(
            directory = callDir,
            phoneNumber = "9876543210",
            callStartTime = callStart,
            callEndTime = callEnd
        )

        assertNull("Tiny/empty file (<1024 bytes) must be rejected", candidate)
    }

    @Test
    fun testCandidateMatching_rejectsUnsupportedExtensions() {
        val callDir = tempFolder.newFolder("miui_call_rec_ext")
        val now = System.currentTimeMillis()
        val callStart = now - 30_000L
        val callEnd = now

        val logFile = File(callDir, "9876543210.log")
        logFile.writeBytes(ByteArray(2048) { 0x01 })
        logFile.setLastModified(callEnd + 1000L)

        val tmpFile = File(callDir, "9876543210.tmp")
        tmpFile.writeBytes(ByteArray(2048) { 0x01 })
        tmpFile.setLastModified(callEnd + 1000L)

        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)
        val candidate = discoverer.findBestCandidate(
            directory = callDir,
            phoneNumber = "9876543210",
            callStartTime = callStart,
            callEndTime = callEnd
        )

        assertNull("Non-audio extension files must be ignored", candidate)
    }

    @Test
    fun testNonDestructiveInspection_guarantee() {
        val callDir = tempFolder.newFolder("miui_call_rec_safety")
        val now = System.currentTimeMillis()
        val callStart = now - 20_000L
        val callEnd = now

        val file1 = File(callDir, "old_call.mp3")
        file1.writeBytes(byteArrayOf(1, 2, 3, 4, 5) + ByteArray(2000))
        file1.setLastModified(now - 100_000L)

        val file2 = File(callDir, "+918888888888@call.m4a")
        file2.writeBytes(byteArrayOf(9, 8, 7, 6) + ByteArray(3000))
        file2.setLastModified(callEnd + 1000L)

        val file1OriginalLength = file1.length()
        val file2OriginalLength = file2.length()
        val file1OriginalBytes = file1.readBytes()
        val file2OriginalBytes = file2.readBytes()

        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)
        val candidate = discoverer.findBestCandidate(
            directory = callDir,
            phoneNumber = "+918888888888",
            callStartTime = callStart,
            callEndTime = callEnd
        )

        assertNotNull(candidate)
        assertEquals(file2.name, candidate!!.file.name)

        // Safety assertion: files must NOT be deleted, moved, renamed, or modified
        assertTrue("File 1 must still exist", file1.exists())
        assertTrue("File 2 must still exist", file2.exists())
        assertEquals("File 1 length must be untouched", file1OriginalLength, file1.length())
        assertEquals("File 2 length must be untouched", file2OriginalLength, file2.length())
        assertArrayEquals("File 1 contents must be intact", file1OriginalBytes, file1.readBytes())
        assertArrayEquals("File 2 contents must be intact", file2OriginalBytes, file2.readBytes())
    }

    @Test
    fun testCallSessionManager_completedCallSessionEmitted() {
        val manager = CallSessionManager()
        val session = manager.startWebRtcSession("room-456")
        val sessionId = session.callSession.id

        // Mark active
        manager.onCallStateChanged(sessionId, CallState.ACTIVE)
        assertEquals(CallState.ACTIVE, manager.activeSession.value?.callState)

        // Remove call (call ended)
        val removed = manager.onCallRemoved(sessionId)
        assertNotNull(removed)

        val completed = manager.lastCompletedCall.value
        assertNotNull("lastCompletedCall must be populated on call end", completed)
        assertEquals(sessionId, completed!!.sessionId)
        assertEquals("WebRTC: room-456", completed.phoneNumber)
        assertTrue("wasActive must be true", completed.wasActive)
        assertFalse("WebRTC calls should have isCellular = false", completed.isCellular)

        // Clear completed call
        manager.clearLastCompletedCall()
        assertNull(manager.lastCompletedCall.value)
    }

    @Test
    fun testCandidateMatching_detectsAmbiguousWhenTied() = runBlocking {
        val callDir = tempFolder.newFolder("miui_call_rec_ambiguous")
        val now = System.currentTimeMillis()
        val callStart = now - 30_000L
        val callEnd = now

        // Create two files with identical timestamp and correlation score
        val file1 = File(callDir, "9876543210_call_A.mp3")
        file1.writeBytes(ByteArray(4096) { 0x11 })
        file1.setLastModified(callEnd + 1000L)

        val file2 = File(callDir, "9876543210_call_B.mp3")
        file2.writeBytes(ByteArray(4096) { 0x22 })
        file2.setLastModified(callEnd + 1000L)

        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)
        val result = discoverer.discoverRecordingWithRetry(
            phoneNumber = "+919876543210",
            callStartTime = callStart,
            callEndTime = callEnd,
            maxWaitMs = 50L,
            pollIntervalMs = 10L
        )

        assertTrue("Result should be Ambiguous when tied", result is OemDiscoveryResult.Ambiguous)
        val ambiguous = result as OemDiscoveryResult.Ambiguous
        assertEquals(2, ambiguous.candidateCount)
        assertEquals(com.voiceshield.shared.PostCallRecordingState.RECORDING_AMBIGUOUS, ambiguous.recordingState)
    }

    @Test
    fun testCandidateMatching_recordingStateMapping() {
        val callDir = tempFolder.newFolder("miui_call_rec_states")
        val discoverer = OemCallRecordingDiscoverer(customDirectory = callDir)

        val notFound = OemDiscoveryResult.NotFound("not found", callDir.absolutePath)
        assertEquals(com.voiceshield.shared.PostCallRecordingState.RECORDING_NOT_FOUND, notFound.recordingState)

        val unavailable = OemDiscoveryResult.Unavailable("storage inaccessible")
        assertEquals(com.voiceshield.shared.PostCallRecordingState.RECORDING_NOT_ACCESSIBLE, unavailable.recordingState)

        val invalid = OemDiscoveryResult.Invalid("corrupted audio", null)
        assertEquals(com.voiceshield.shared.PostCallRecordingState.RECORDING_INVALID, invalid.recordingState)
    }
}
