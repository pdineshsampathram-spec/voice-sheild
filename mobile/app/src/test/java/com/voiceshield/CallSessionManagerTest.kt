package com.voiceshield

import com.voiceshield.shared.AudioSourceType
import com.voiceshield.shared.ThreatLevel
import com.voiceshield.telecom.CallSessionManager
import com.voiceshield.telecom.CallState
import org.junit.Assert.*
import org.junit.Test

class CallSessionManagerTest {

    @Test
    fun testWebRtcSession_lifecycle() {
        val manager = CallSessionManager()

        // 1. Start WebRTC session
        val session = manager.startWebRtcSession("room-123")
        assertEquals(AudioSourceType.WEBRTC_DIRECT, session.callSession.audioSourceType)
        assertEquals(CallState.ACTIVE, session.callState)
        assertTrue(session.analysisActive)
        assertEquals("WebRTC: room-123", session.callSession.phoneNumber)
        assertEquals(session, manager.activeSession.value)

        // 2. Update analysis state
        manager.updateAnalysisState(session.callSession.id, isAnalyzing = true, windowsProcessed = 4)
        val updated = manager.getSession(session.callSession.id)
        assertNotNull(updated)
        assertEquals(4, updated!!.windowsProcessed)
        assertTrue(updated.analysisActive)

        // 3. End session
        manager.endSession(session.callSession.id)
        assertNull("Active session should be null after ending the only call", manager.activeSession.value)
        assertEquals(0, manager.allSessions.value.size)
    }

    @Test
    fun testSessionStateChange_propagatesToActiveSessionFlow() {
        val manager = CallSessionManager()
        val session = manager.startWebRtcSession("test-room")

        manager.onCallStateChanged(session.callSession.id, CallState.HOLDING)
        assertEquals(CallState.HOLDING, manager.activeSession.value?.callState)

        manager.onCallStateChanged(session.callSession.id, CallState.ACTIVE)
        assertEquals(CallState.ACTIVE, manager.activeSession.value?.callState)
    }
}
