/**
 * CallSessionManager — Tracks all active call sessions and their states.
 *
 * Each call gets a CallSession object that tracks:
 * - Call lifecycle (ringing, active, disconnected)
 * - Audio source availability
 * - Detection engine assignment
 * - Threat state
 */
package com.voiceshield.telecom

import android.telecom.Call
import com.voiceshield.shared.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages the lifecycle of all active call sessions.
 * Thread-safe via ConcurrentHashMap + StateFlow.
 */
class CallSessionManager {

    private val _sessions = ConcurrentHashMap<String, ManagedCallSession>()
    private val _activeSessionFlow = MutableStateFlow<ManagedCallSession?>(null)
    val activeSession: StateFlow<ManagedCallSession?> = _activeSessionFlow.asStateFlow()

    private val _allSessionsFlow = MutableStateFlow<List<ManagedCallSession>>(emptyList())
    val allSessions: StateFlow<List<ManagedCallSession>> = _allSessionsFlow.asStateFlow()

    private val _lastCompletedCallFlow = MutableStateFlow<CompletedCallSession?>(null)
    val lastCompletedCall: StateFlow<CompletedCallSession?> = _lastCompletedCallFlow.asStateFlow()

    /**
     * Create a new session when a call is added.
     */
    fun onCallAdded(call: Call): ManagedCallSession {
        val phoneNumber = call.details?.handle?.schemeSpecificPart
        val session = ManagedCallSession(
            callSession = CallSession(
                phoneNumber = phoneNumber,
                audioSourceType = AudioSourceType.UNAVAILABLE  // Default: no audio assumed
            ),
            callState = CallState.RINGING,
            threatState = ThreatState(
                level = ThreatLevel.MONITORING,
                reason = "Call started. Determining audio availability.",
                windowsAnalyzed = 0,
                suspiciousWindows = 0,
                avgSpoofScore = null,
                avgSpeakerSimilarity = null,
                audioSourceType = AudioSourceType.UNAVAILABLE,
                modelAvailability = emptyMap()
            )
        )

        _sessions[session.callSession.id] = session
        _activeSessionFlow.value = session
        updateAllSessions()

        return session
    }

    /**
     * Programmatically create and track a session (used for testing and headless session registration).
     */
    fun createSession(
        sessionId: String = java.util.UUID.randomUUID().toString(),
        phoneNumber: String? = null,
        audioSourceType: AudioSourceType = AudioSourceType.UNAVAILABLE
    ): ManagedCallSession {
        val session = ManagedCallSession(
            callSession = CallSession(
                id = sessionId,
                phoneNumber = phoneNumber,
                audioSourceType = audioSourceType
            ),
            callState = CallState.RINGING,
            threatState = ThreatState(
                level = ThreatLevel.MONITORING,
                reason = "Call session initialized.",
                windowsAnalyzed = 0,
                suspiciousWindows = 0,
                avgSpoofScore = null,
                avgSpeakerSimilarity = null,
                audioSourceType = audioSourceType,
                modelAvailability = emptyMap()
            )
        )

        _sessions[sessionId] = session
        _activeSessionFlow.value = session
        updateAllSessions()

        return session
    }

    /**
     * Update a session when the call state changes.
     */
    fun onCallStateChanged(sessionId: String, newState: CallState) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(callState = newState)
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    /**
     * Update audio source availability for a session.
     */
    fun updateAudioSource(sessionId: String, sourceType: AudioSourceType) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(
                callSession = session.callSession.copy(audioSourceType = sourceType),
                threatState = session.threatState.copy(audioSourceType = sourceType)
            )
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    /**
     * Update threat state for a session.
     */
    fun updateThreatState(sessionId: String, threatState: ThreatState) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(threatState = threatState)
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    /**
     * Remove a session when the call ends.
     */
    fun onCallRemoved(sessionId: String): ManagedCallSession? {
        val removed = _sessions.remove(sessionId)
        if (removed != null) {
            val endTime = System.currentTimeMillis()
            val duration = (endTime - removed.startTime).coerceAtLeast(0L)
            val isCellular = removed.callSession.audioSourceType != AudioSourceType.WEBRTC_DIRECT
            val wasActive = removed.callState == CallState.ACTIVE ||
                            removed.callState == CallState.HOLDING ||
                            removed.callState == CallState.DISCONNECTING ||
                            removed.callState == CallState.DISCONNECTED
            _lastCompletedCallFlow.value = CompletedCallSession(
                sessionId = sessionId,
                phoneNumber = removed.callSession.phoneNumber,
                startTime = removed.startTime,
                endTime = endTime,
                durationMs = duration,
                wasActive = wasActive,
                isCellular = isCellular
            )
        }
        if (_activeSessionFlow.value?.callSession?.id == sessionId) {
            _activeSessionFlow.value = _sessions.values.firstOrNull()
        }
        updateAllSessions()
        return removed
    }

    fun clearLastCompletedCall() {
        _lastCompletedCallFlow.value = null
    }

    /**
     * Start a direct WebRTC call session for testing or in-app VoIP.
     */
    fun startWebRtcSession(roomId: String = "voiceshield-test"): ManagedCallSession {
        val session = ManagedCallSession(
            callSession = CallSession(
                phoneNumber = "WebRTC: $roomId",
                audioSourceType = AudioSourceType.WEBRTC_DIRECT
            ),
            callState = CallState.ACTIVE,
            analysisActive = true,
            threatState = ThreatState(
                level = ThreatLevel.MONITORING,
                reason = "WebRTC call connected. Waiting for remote caller audio.",
                windowsAnalyzed = 0,
                suspiciousWindows = 0,
                avgSpoofScore = null,
                avgSpeakerSimilarity = null,
                audioSourceType = AudioSourceType.WEBRTC_DIRECT,
                modelAvailability = emptyMap()
            )
        )
        _sessions[session.callSession.id] = session
        _activeSessionFlow.value = session
        updateAllSessions()
        return session
    }

    /**
     * Update analysis state and windows count.
     */
    fun updateAnalysisState(sessionId: String, isAnalyzing: Boolean, windowsProcessed: Int) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(analysisActive = isAnalyzing, windowsProcessed = windowsProcessed)
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    /**
     * End a call session by id.
     */
    fun endSession(sessionId: String): ManagedCallSession? {
        onCallStateChanged(sessionId, CallState.DISCONNECTED)
        return onCallRemoved(sessionId)
    }

    fun updateCapabilityReport(sessionId: String, report: CellularCapabilityReport) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(capabilityReport = report)
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    fun updateTelecomAudioDiagnostics(sessionId: String, diagnostics: TelecomAudioDiagnostics) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(telecomAudioDiagnostics = diagnostics)
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    fun updateLastDisconnectInfo(sessionId: String, info: LastDisconnectInfo) {
        _sessions[sessionId]?.let { session ->
            val updated = session.copy(lastDisconnectInfo = info)
            _sessions[sessionId] = updated
            if (_activeSessionFlow.value?.callSession?.id == sessionId) {
                _activeSessionFlow.value = updated
            }
            updateAllSessions()
        }
    }

    fun getSession(sessionId: String): ManagedCallSession? = _sessions[sessionId]

    private fun updateAllSessions() {
        _allSessionsFlow.value = _sessions.values.toList()
    }
}

/**
 * A managed call session wrapping CallSession with UI-observable state.
 */
data class ManagedCallSession(
    val callSession: CallSession,
    val callState: CallState,
    val threatState: ThreatState,
    val startTime: Long = System.currentTimeMillis(),
    val analysisActive: Boolean = false,
    val windowsProcessed: Int = 0,
    val capabilityReport: CellularCapabilityReport? = null,
    val telecomAudioDiagnostics: TelecomAudioDiagnostics = TelecomAudioDiagnostics(),
    val lastDisconnectInfo: LastDisconnectInfo? = null
)

enum class CallState {
    RINGING,
    CONNECTING,
    DIALING,
    ACTIVE,
    HOLDING,
    DISCONNECTING,
    DISCONNECTED
}

/**
 * Metadata captured when a call session completes, used for OEM recording discovery.
 */
data class CompletedCallSession(
    val sessionId: String,
    val phoneNumber: String?,
    val startTime: Long,
    val endTime: Long,
    val durationMs: Long,
    val wasActive: Boolean,
    val isCellular: Boolean
)
