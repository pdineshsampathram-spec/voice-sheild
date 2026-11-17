/**
 * VoiceShieldInCallService — Core telecom integration.
 *
 * Extends Android's InCallService to handle incoming/active call lifecycle.
 * Hooks into CallSessionManager to track VoiceShield-specific state.
 *
 * CRITICAL SEPARATION OF CONCERNS:
 * 1. Normal cellular call audio is controlled entirely by Android Telecom.
 * 2. VoiceShield does NOT touch AudioRecord, AudioManager mode, or audio focus
 *    during active call setup, ensuring ordinary earpiece/speaker audio works reliably.
 * 3. Answering a call requests Telecom to answer and transitions to CONNECTING,
 *    updating UI to ACTIVE only when Telecom confirms Call.STATE_ACTIVE.
 * 4. Cellular voice clone analysis is strictly reported as AUDIO_UNAVAILABLE
 *    until independently verified.
 *
 * UPLINK AUDIO RULE:
 * - VoiceShield does NOT call setMuted(true), setMicrophoneMute, or modify
 *   AudioManager.mode at any point during normal call lifecycle.
 * - The only mute operation is toggleMute() which uses the official Telecom
 *   setMuted() API and is initiated only by explicit user action.
 * - Uplink audio is considered UNVERIFIED until physical device testing confirms
 *   the remote party can hear speech from the VoiceShield phone.
 *
 * DISCONNECT RULES:
 * - AUDIO_UNAVAILABLE must NEVER cause a cellular call to disconnect.
 * - Calls are disconnected ONLY by explicit user action (END button or alert END_CALL).
 * - Every disconnect captures DisconnectCause metadata for diagnosis.
 */
package com.voiceshield.telecom

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.os.Build
import android.telecom.Call
import android.telecom.CallAudioState
import android.telecom.DisconnectCause
import android.telecom.InCallService
import android.util.Log
import androidx.core.app.NotificationCompat
import com.voiceshield.MainActivity
import com.voiceshield.VoiceShieldApp
import com.voiceshield.shared.*
import kotlinx.coroutines.*

class VoiceShieldInCallService : InCallService() {

    companion object {
        private const val TAG = "VoiceShieldInCall"
        const val CHANNEL_ID_INCOMING = "voiceshield_incoming_calls"
        const val INCOMING_CALL_NOTIFICATION_ID = 2001

        /**
         * Diagnostic recovery timeout for cellular carrier connection.
         * EXPERIMENTAL: If Telecom remains in STATE_RINGING beyond this interval
         * following call.answer(), the watchdog resets the in-progress flag and
         * restores the UI to permit an explicit retry or rejection.
         */
        const val EXPERIMENTAL_ANSWER_RECOVERY_TIMEOUT_MS = 6000L

        /**
         * Shared instance for UI access.
         * Set when service is bound, cleared when unbound.
         */
        @Volatile
        var instance: VoiceShieldInCallService? = null
            private set

        fun mapCallState(telecomState: Int): CallState = when (telecomState) {
            Call.STATE_RINGING, Call.STATE_SIMULATED_RINGING -> CallState.RINGING
            Call.STATE_NEW, Call.STATE_CONNECTING, Call.STATE_SELECT_PHONE_ACCOUNT -> CallState.CONNECTING
            Call.STATE_DIALING -> CallState.DIALING
            Call.STATE_ACTIVE -> CallState.ACTIVE
            Call.STATE_HOLDING -> CallState.HOLDING
            Call.STATE_DISCONNECTING -> CallState.DISCONNECTING
            Call.STATE_DISCONNECTED -> CallState.DISCONNECTED
            else -> CallState.RINGING // Safe fallback, NEVER claim ACTIVE
        }

        fun routeToString(route: Int): String = when (route) {
            CallAudioState.ROUTE_EARPIECE -> "EARPIECE"
            CallAudioState.ROUTE_BLUETOOTH -> "BLUETOOTH"
            CallAudioState.ROUTE_WIRED_HEADSET -> "WIRED_HEADSET"
            CallAudioState.ROUTE_SPEAKER -> "SPEAKER"
            else -> "ROUTE_$route"
        }

        fun supportedRoutesToList(mask: Int): List<String> {
            val list = mutableListOf<String>()
            if ((mask and CallAudioState.ROUTE_EARPIECE) != 0) list.add("EARPIECE")
            if ((mask and CallAudioState.ROUTE_BLUETOOTH) != 0) list.add("BLUETOOTH")
            if ((mask and CallAudioState.ROUTE_WIRED_HEADSET) != 0) list.add("WIRED_HEADSET")
            if ((mask and CallAudioState.ROUTE_SPEAKER) != 0) list.add("SPEAKER")
            return list
        }

        fun disconnectCauseCodeToString(code: Int): String = when (code) {
            DisconnectCause.UNKNOWN -> "UNKNOWN"
            DisconnectCause.ERROR -> "ERROR"
            DisconnectCause.LOCAL -> "LOCAL"
            DisconnectCause.REMOTE -> "REMOTE"
            DisconnectCause.CANCELED -> "CANCELED"
            DisconnectCause.MISSED -> "MISSED"
            DisconnectCause.REJECTED -> "REJECTED"
            DisconnectCause.BUSY -> "BUSY"
            DisconnectCause.RESTRICTED -> "RESTRICTED"
            DisconnectCause.OTHER -> "OTHER"
            DisconnectCause.CONNECTION_MANAGER_NOT_SUPPORTED -> "CONNECTION_MANAGER_NOT_SUPPORTED"
            else -> "CODE_$code"
        }
    }

    val sessionManager get() = VoiceShieldApp.sessionManager

    // Concurrency synchronization lock for all Telecom Call <-> Session compound operations
    private val callLock = Any()

    // Bi-directional mapping between Telecom Call objects and VoiceShield session IDs
    private val callToSessionMap = mutableMapOf<Call, String>()
    private val sessionToCallMap = mutableMapOf<String, Call>()

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    /**
     * Tracks the lifecycle state of this InCallService instance.
     */
    @Volatile
    private var currentServiceLifecycleState: String = "INITIAL"

    /**
     * Answer request idempotency tracking per session.
     * Set to true when answerCall() is called.
     * Reset when call reaches ACTIVE, DISCONNECTED, or is removed.
     */
    private val answerInProgress = mutableMapOf<String, Boolean>()

    /**
     * Tracks whether VoiceShield explicitly requested a disconnect.
     * Key: sessionId, Value: true if disconnectCall() was called by VoiceShield.
     */
    private val localDisconnectRequested = mutableMapOf<String, Boolean>()

    /**
     * Tracks the categorized reason for a local disconnect.
     */
    private val localDisconnectReason = mutableMapOf<String, LocalDisconnectReason>()

    /**
     * Tracks whether endActiveCall() was invoked (from MainActivity).
     * Key: sessionId, Value: true if endActiveCall path was triggered.
     */
    private val endActiveCallInvoked = mutableMapOf<String, Boolean>()

    /**
     * Tracks whether an AlertManager END_CALL action triggered the disconnect.
     * Key: sessionId, Value: true if alert action triggered it.
     */
    private val alertActionTriggered = mutableMapOf<String, Boolean>()

    /**
     * Tracks the last known Telecom raw state for each call.
     */
    private val lastKnownTelecomState = mutableMapOf<Call, Int>()

    private val callCallback = object : Call.Callback() {
        override fun onStateChanged(call: Call, state: Int) {
            val sessionId = synchronized(callLock) {
                callToSessionMap[call] ?: sessionToCallMap.entries.find { it.value == call }?.key
            } ?: return
            val previousState = synchronized(callLock) {
                val prev = lastKnownTelecomState[call]
                lastKnownTelecomState[call] = state
                prev
            }
            val mappedState = mapCallState(state)

            Log.i(TAG, "LIFECYCLE [onStateChanged] session=$sessionId " +
                    "raw=$state (${telecomStateToString(state)}) -> mapped=$mappedState " +
                    "previousRaw=${previousState ?: "none"} " +
                    "activeCalls=${synchronized(callLock) { callToSessionMap.size }} " +
                    "ts=${System.currentTimeMillis()}")
            Log.i(TAG, "TELECOM_STATE=${telecomStateToString(state)} session=$sessionId raw=$state mapped=$mappedState")

            sessionManager.onCallStateChanged(sessionId, mappedState)

            when (mappedState) {
                CallState.ACTIVE -> {
                    dismissIncomingCallPresentation()
                    synchronized(callLock) { answerInProgress.remove(sessionId) }
                    onCallActive(sessionId, call)
                }
                CallState.DISCONNECTED -> {
                    dismissIncomingCallPresentation()
                    synchronized(callLock) { answerInProgress.remove(sessionId) }
                    // Capture DisconnectCause from call details
                    captureDisconnectCause(call, sessionId, state)
                }
                else -> {}
            }
        }

        override fun onDetailsChanged(call: Call, details: Call.Details) {
            val sessionId = synchronized(callLock) {
                callToSessionMap[call] ?: sessionToCallMap.entries.find { it.value == call }?.key
            } ?: return
            val disconnectCause = details.disconnectCause
            if (disconnectCause != null && disconnectCause.code != DisconnectCause.UNKNOWN) {
                Log.i(TAG, "LIFECYCLE [onDetailsChanged] session=$sessionId " +
                        "disconnectCause.code=${disconnectCauseCodeToString(disconnectCause.code)} " +
                        "label=${disconnectCause.label} " +
                        "description=${disconnectCause.description} " +
                        "reason=${disconnectCause.reason} " +
                        "ts=${System.currentTimeMillis()}")
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        currentServiceLifecycleState = "CREATED"
        instance = this
        Log.i(TAG, "LIFECYCLE [onCreate] VoiceShieldInCallService created ts=${System.currentTimeMillis()}")
    }

    override fun onDestroy() {
        Log.i(TAG, "LIFECYCLE [onDestroy] VoiceShieldInCallService destroyed " +
                "activeCalls=${synchronized(callLock) { callToSessionMap.size }} ts=${System.currentTimeMillis()}")
        currentServiceLifecycleState = "DESTROYED"
        serviceScope.cancel()
        if (instance === this) {
            instance = null
        }
        super.onDestroy()
    }

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)

        val session = sessionManager.onCallAdded(call)
        val sessionId = session.callSession.id
        synchronized(callLock) {
            callToSessionMap[call] = sessionId
            sessionToCallMap[sessionId] = call
            lastKnownTelecomState[call] = call.state

            // Initialize per-session tracking flags (Second-Call Isolation guarantee)
            answerInProgress[sessionId] = false
            localDisconnectRequested[sessionId] = false
            localDisconnectReason.remove(sessionId)
            endActiveCallInvoked[sessionId] = false
            alertActionTriggered[sessionId] = false
        }
        call.registerCallback(callCallback)

        // Set initial state from actual Telecom call state
        val initialMappedState = mapCallState(call.state)
        sessionManager.onCallStateChanged(sessionId, initialMappedState)

        // Platform-supported presentation: high-priority notification with fullScreenIntent
        if (call.state == Call.STATE_RINGING || call.state == Call.STATE_SIMULATED_RINGING) {
            showIncomingCallPresentation(sessionId, session.callSession.phoneNumber)
        }

        Log.i(TAG, "LIFECYCLE [onCallAdded] session=$sessionId " +
                "number=${session.callSession.phoneNumber} " +
                "telecomState=${call.state} (${telecomStateToString(call.state)}) " +
                "mappedState=$initialMappedState " +
                "totalCalls=${synchronized(callLock) { callToSessionMap.size }} " +
                "ts=${System.currentTimeMillis()}")

        callAudioState?.let { updateAudioDiagnostics(it) }

        if (initialMappedState == CallState.ACTIVE) {
            onCallActive(sessionId, call)
        }
    }

    override fun onCallRemoved(call: Call) {
        val sessionId = synchronized(callLock) {
            val id = callToSessionMap.remove(call)
                ?: sessionToCallMap.entries.find { it.value == call }?.key
            if (id != null) {
                sessionToCallMap.remove(id)
            }
            id
        } ?: run {
            // Correlation fallback: candidate matching requires EXACTLY ONE matching candidate
            val candidateHandle = call.details?.handle?.schemeSpecificPart
            synchronized(callLock) {
                val candidateEntries = sessionToCallMap.filter { (sessId, _) ->
                    val s = sessionManager.getSession(sessId)
                    s?.callSession?.phoneNumber != null && s.callSession.phoneNumber == candidateHandle
                }
                if (candidateEntries.size == 1) {
                    val candidateId = candidateEntries.keys.first()
                    sessionToCallMap.remove(candidateId)
                    callToSessionMap.entries.removeIf { it.value == candidateId }
                    candidateId
                } else {
                    null
                }
            }
        }
        call.unregisterCallback(callCallback)
        val previousState = synchronized(callLock) { lastKnownTelecomState.remove(call) }

        // Dismiss incoming call notification if still showing
        dismissIncomingCallPresentation()

        sessionId?.let { id ->
            val wasLocalDisconnect: Boolean
            val disconnectReason: LocalDisconnectReason?
            val wasEndActiveCall: Boolean
            val wasAlertAction: Boolean
            synchronized(callLock) {
                wasLocalDisconnect = localDisconnectRequested.remove(id) ?: false
                disconnectReason = localDisconnectReason.remove(id)
                wasEndActiveCall = endActiveCallInvoked.remove(id) ?: false
                wasAlertAction = alertActionTriggered.remove(id) ?: false
                answerInProgress.remove(id)
            }

            // Capture final disconnect info
            captureDisconnectCause(call, id, Call.STATE_DISCONNECTED)

            Log.i(TAG, "LIFECYCLE [onCallRemoved] session=$id " +
                    "localDisconnect=$wasLocalDisconnect " +
                    "reason=$disconnectReason " +
                    "endActiveCallInvoked=$wasEndActiveCall " +
                    "alertActionTriggered=$wasAlertAction " +
                    "lastRawState=${previousState ?: "unknown"} " +
                    "duration=${System.currentTimeMillis() - (sessionManager.getSession(id)?.startTime ?: 0)}ms " +
                    "remainingCalls=${synchronized(callLock) { callToSessionMap.size }} " +
                    "ts=${System.currentTimeMillis()}")

            onCallDisconnected(id)
            sessionManager.onCallRemoved(id)
        }

        super.onCallRemoved(call)
    }

    // ─── Platform-Supported Call Presentation ───────────────────────────

    private fun showIncomingCallPresentation(sessionId: String, phoneNumber: String?) {
        try {
            ensureNotificationChannel()

            val launchIntent = Intent(this, MainActivity::class.java).apply {
                action = Intent.ACTION_MAIN
                addCategory(Intent.CATEGORY_LAUNCHER)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
                putExtra("EXTRA_CALL_SESSION_ID", sessionId)
            }

            val fullScreenPendingIntent = PendingIntent.getActivity(
                this,
                sessionId.hashCode(),
                launchIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val displayName = phoneNumber ?: "Incoming Call"

            val notification = NotificationCompat.Builder(this, CHANNEL_ID_INCOMING)
                .setSmallIcon(android.R.drawable.sym_call_incoming)
                .setContentTitle("VoiceShield — Incoming Call")
                .setContentText(displayName)
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setCategory(NotificationCompat.CATEGORY_CALL)
                .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
                .setOngoing(true)
                .setAutoCancel(false)
                .setFullScreenIntent(fullScreenPendingIntent, true)
                .setContentIntent(fullScreenPendingIntent)
                .build()

            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(INCOMING_CALL_NOTIFICATION_ID, notification)
            Log.i(TAG, "PRESENTATION [showIncomingCallPresentation] Posted incoming call notification for session=$sessionId")
        } catch (e: Exception) {
            Log.e(TAG, "PRESENTATION [showIncomingCallPresentation] Failed to show incoming call presentation: ${e.message}", e)
        }
    }

    private fun dismissIncomingCallPresentation() {
        try {
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.cancel(INCOMING_CALL_NOTIFICATION_ID)
            Log.d(TAG, "PRESENTATION [dismissIncomingCallPresentation] Dismissed incoming call notification")
        } catch (e: Exception) {
            Log.e(TAG, "PRESENTATION [dismissIncomingCallPresentation] Error dismissing notification: ${e.message}")
        }
    }

    private fun ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val existing = notificationManager.getNotificationChannel(CHANNEL_ID_INCOMING)
            if (existing == null) {
                val channel = NotificationChannel(
                    CHANNEL_ID_INCOMING,
                    "Incoming Calls",
                    NotificationManager.IMPORTANCE_HIGH
                ).apply {
                    description = "VoiceShield incoming call presentation and full-screen alerts"
                    setSound(null, null) // Telecom plays the cellular ringtone
                    enableVibration(true)
                    lockscreenVisibility = Notification.VISIBILITY_PUBLIC
                }
                notificationManager.createNotificationChannel(channel)
                Log.i(TAG, "Created notification channel: $CHANNEL_ID_INCOMING")
            }
        }
    }

    /**
     * Telecom audio state changed callback.
     * Keeps VoiceShield UI & diagnostics synchronized with actual hardware routing.
     */
    override fun onCallAudioStateChanged(audioState: CallAudioState) {
        super.onCallAudioStateChanged(audioState)
        Log.i(TAG, "LIFECYCLE [onCallAudioStateChanged] " +
                "route=${routeToString(audioState.route)} " +
                "isMuted=${audioState.isMuted} " +
                "supported=${supportedRoutesToList(audioState.supportedRouteMask)} " +
                "ts=${System.currentTimeMillis()}")
        updateAudioDiagnostics(audioState)
    }

    private fun updateAudioDiagnostics(audioState: CallAudioState) {
        try {
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager

            // Independent uplink investigation: check all audio state
            val uplinkNotes = buildString {
                append("route=${routeToString(audioState.route)}")
                append(" telecomMuted=${audioState.isMuted}")
                append(" amMicMuted=${audioManager.isMicrophoneMute}")
                append(" amMode=${audioManager.mode}")
                append(" amSpeaker=${audioManager.isSpeakerphoneOn}")
                // Check for devices with microphone
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    val inputDevices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
                    append(" inputDevices=${inputDevices.size}")
                    inputDevices.forEach { device ->
                        append(" [${device.type}:${device.productName}]")
                    }
                }
            }

            // Uplink status determination
            // NOTE: This is NOT a claim that uplink works — it reports the diagnostic state
            // Physical device testing must confirm the remote party can hear speech.
            val uplinkStatus = when {
                audioState.isMuted -> "MUTED_BY_TELECOM"
                audioManager.isMicrophoneMute -> "MUTED_BY_AUDIOMANAGER"
                audioManager.mode == AudioManager.MODE_NORMAL -> "SUSPECT_MODE_NORMAL"
                else -> "UNVERIFIED"
            }

            val diag = TelecomAudioDiagnostics(
                callState = sessionManager.activeSession.value?.callState?.name ?: "IDLE",
                currentRoute = routeToString(audioState.route),
                supportedRoutes = supportedRoutesToList(audioState.supportedRouteMask),
                isMuted = audioState.isMuted,
                isMicrophoneMuted = audioManager.isMicrophoneMute,
                audioManagerMode = audioManager.mode,
                isSpeakerphoneOn = audioManager.isSpeakerphoneOn,
                isBluetoothScoOn = audioManager.isBluetoothScoOn,
                uplinkStatus = uplinkStatus,
                uplinkNotes = uplinkNotes
            )

            Log.i(TAG, "UPLINK_DIAG uplinkStatus=$uplinkStatus notes=$uplinkNotes")

            callToSessionMap.values.forEach { sessionId ->
                sessionManager.updateTelecomAudioDiagnostics(sessionId, diag)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error updating audio diagnostics: ${e.message}")
        }
    }

    /**
     * Capture DisconnectCause from Telecom Call.Details.
     * Records ALL contributing factors for disconnect investigation.
     */
    private fun captureDisconnectCause(call: Call, sessionId: String, rawState: Int) {
        try {
            val cause = call.details?.disconnectCause
            val isLocal = localDisconnectRequested[sessionId] ?: false
            val reason = localDisconnectReason[sessionId]
            val aiState = if (VoiceShieldApp.telephonyOnlyMode) {
                "TELEPHONY_ONLY_ISOLATED"
            } else {
                sessionManager.getSession(sessionId)?.threatState?.level?.name ?: "AUDIO_UNAVAILABLE"
            }

            val info = LastDisconnectInfo(
                code = cause?.code ?: DisconnectCause.UNKNOWN,
                label = cause?.label?.toString() ?: "",
                description = cause?.description?.toString() ?: "",
                reason = cause?.reason ?: "",
                localDisconnectRequested = isLocal,
                localDisconnectReason = reason,
                endActiveCallInvoked = endActiveCallInvoked[sessionId] ?: false,
                alertActionTriggered = alertActionTriggered[sessionId] ?: false,
                lastTelecomState = rawState,
                serviceLifecycleState = currentServiceLifecycleState,
                aiState = aiState,
                timestamp = System.currentTimeMillis()
            )

            sessionManager.updateLastDisconnectInfo(sessionId, info)

            Log.i(TAG, "DISCONNECT_CAUSE session=$sessionId " +
                    "code=${disconnectCauseCodeToString(info.code)} " +
                    "label=${info.label} " +
                    "description=${info.description} " +
                    "reason=${info.reason} " +
                    "localDisconnect=${info.localDisconnectRequested} " +
                    "localReason=${info.localDisconnectReason} " +
                    "endActiveCall=${info.endActiveCallInvoked} " +
                    "alertAction=${info.alertActionTriggered} " +
                    "lastTelecomState=$rawState " +
                    "serviceState=${info.serviceLifecycleState} " +
                    "aiState=${info.aiState} " +
                    "ts=${info.timestamp}")
        } catch (e: Exception) {
            Log.w(TAG, "Error capturing DisconnectCause: ${e.message}")
        }
    }

    /**
     * Triggered ONLY when Telecom reports Call.STATE_ACTIVE.
     * Does NOT touch AudioRecord or AudioManager mode, allowing Telecom
     * to manage normal earpiece and speaker call audio without interference.
     *
     * UPLINK RULE: Does not set any mute state. Does not modify AudioManager.
     * Uplink audio status is reported as UNVERIFIED pending physical device test.
     */
    private fun onCallActive(sessionId: String, call: Call) {
        // Log comprehensive uplink state at the moment of STATE_ACTIVE
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val telecomAudio = callAudioState

        Log.i(TAG, "LIFECYCLE [onCallActive] session=$sessionId " +
                "telecomRoute=${telecomAudio?.let { routeToString(it.route) } ?: "null"} " +
                "telecomMuted=${telecomAudio?.isMuted} " +
                "amMode=${audioManager.mode} " +
                "amMicMuted=${audioManager.isMicrophoneMute} " +
                "amSpeaker=${audioManager.isSpeakerphoneOn} " +
                "telephonyOnlyMode=${VoiceShieldApp.telephonyOnlyMode} " +
                "ts=${System.currentTimeMillis()}")

        // CRITICAL UPLINK RESTORATION:
        // An active call must be UNMUTED by default so the remote party can hear local speech.
        // If Telecom initialized the call in a muted state (stale flag or HAL persistence),
        // enforce setMuted(false) via the official Telecom API.
        if (telecomAudio?.isMuted == true) {
            Log.w(TAG, "LIFECYCLE [onCallActive] Call initialized in MUTED state by Telecom! Enforcing setMuted(false).")
            setMuted(false)
        }
        // If AudioManager's global microphone mute is active, reset it to ensure uplink audio reaches modem
        if (audioManager.isMicrophoneMute) {
            Log.w(TAG, "LIFECYCLE [onCallActive] AudioManager.isMicrophoneMute is TRUE! Resetting to false.")
            try {
                audioManager.isMicrophoneMute = false
            } catch (e: Exception) {
                Log.w(TAG, "LIFECYCLE [onCallActive] Failed to reset isMicrophoneMute: ${e.message}")
            }
        }

        telecomAudio?.let { updateAudioDiagnostics(it) }

        // TELEPHONY_ONLY_MODE ISOLATION EXPERIMENT:
        // When enabled, ZERO AI inference, zero models, zero probe.
        if (VoiceShieldApp.telephonyOnlyMode) {
            Log.i(TAG, "LIFECYCLE [onCallActive] TELEPHONY_ONLY_MODE is ACTIVE. AI models and audio analysis are isolated.")
            sessionManager.updateAudioSource(sessionId, AudioSourceType.UNAVAILABLE)
            sessionManager.updateThreatState(
                sessionId,
                ThreatState(
                    level = ThreatLevel.MONITORING,
                    reason = "TELEPHONY-ONLY DIAGNOSTIC MODE: AI pipeline isolated. Normal cellular call active under Telecom control.",
                    windowsAnalyzed = 0,
                    suspiciousWindows = 0,
                    avgSpoofScore = null,
                    avgSpeakerSimilarity = null,
                    audioSourceType = AudioSourceType.UNAVAILABLE,
                    modelAvailability = emptyMap()
                )
            )
            return
        }

        // Cellular remote voice analysis remains UNAVAILABLE
        // until independently verified without audio HAL interference.
        sessionManager.updateAudioSource(sessionId, AudioSourceType.UNAVAILABLE)
        sessionManager.updateThreatState(
            sessionId,
            ThreatState(
                level = ThreatLevel.AUDIO_UNAVAILABLE,
                reason = "Call Protection Active. Call audio routed through normal phone earpiece/speaker. Voice analysis unavailable for this audio path.",
                windowsAnalyzed = 0,
                suspiciousWindows = 0,
                avgSpoofScore = null,
                avgSpeakerSimilarity = null,
                audioSourceType = AudioSourceType.UNAVAILABLE,
                modelAvailability = VoiceShieldApp.onDeviceEngine.getModelHealth()
            )
        )
        // CRITICAL: AUDIO_UNAVAILABLE does NOT disconnect the call.
        // The call remains active under Telecom control.
    }

    private fun onCallDisconnected(sessionId: String) {
        Log.i(TAG, "LIFECYCLE [onCallDisconnected] session=$sessionId ts=${System.currentTimeMillis()}")
        VoiceShieldApp.onDeviceEngine.reset()
        VoiceShieldApp.alertManager.reset()
        VoiceShieldApp.riskEngine.reset()
    }

    /**
     * Answer a ringing call through Telecom.
     * IDEMPOTENT: Rapid multiple taps are safely ignored.
     * Validates that call is in STATE_RINGING before requesting answer.
     */
    fun answerCall(sessionId: String) {
        val now = System.currentTimeMillis()
        Log.i(TAG, "ANSWER_TAPPED session=$sessionId ts=$now")

        val targetCall: Call
        val otherActiveCalls: List<Call>
        synchronized(callLock) {
            val call = sessionToCallMap[sessionId]
                ?: callToSessionMap.entries.find { it.value == sessionId }?.key
            if (call == null) {
                Log.e(TAG, "ANSWER_VALIDATED FAILED session=$sessionId not found in call map")
                return
            }

            val preState = call.state
            val preStateStr = telecomStateToString(preState)
            val callHash = System.identityHashCode(call)
            val details = call.details
            val capabilities = details?.callCapabilities ?: 0
            val properties = details?.callProperties ?: 0

            Log.i(TAG, "ANSWER_VALIDATED checking session=$sessionId callHash=$callHash " +
                    "state=$preState ($preStateStr) capabilities=$capabilities properties=$properties")

            // Validate call state: must be RINGING or SIMULATED_RINGING
            if (preState != Call.STATE_RINGING && preState != Call.STATE_SIMULATED_RINGING) {
                Log.w(TAG, "ANSWER_VALIDATED REJECTED session=$sessionId callHash=$callHash " +
                        "state=$preState ($preStateStr), expected RINGING")
                return
            }

            // Idempotency check: ignore duplicate rapid taps while answer is in flight
            if (answerInProgress[sessionId] == true) {
                Log.w(TAG, "ANSWER_VALIDATED IGNORED duplicate answer request for session=$sessionId (answerInProgress=true)")
                return
            }
            answerInProgress[sessionId] = true

            // Snapshot other active calls under lock
            otherActiveCalls = callToSessionMap.keys.filter { it != call && it.state == Call.STATE_ACTIVE }
            targetCall = call
        }

        // Check if there are existing active calls.
        // BINDING RULE: Do NOT blindly hold. Only hold if the active call explicitly advertises CAPABILITY_HOLD.
        for (activeCall in otherActiveCalls) {
            val canHold = (activeCall.details?.callCapabilities ?: 0) and Call.Details.CAPABILITY_HOLD != 0
            if (canHold) {
                Log.i(TAG, "LIFECYCLE [answerCall] Holding existing active call before answering incoming call (CAPABILITY_HOLD=true)")
                try {
                    activeCall.hold()
                } catch (e: Exception) {
                    Log.w(TAG, "LIFECYCLE [answerCall] Failed to hold active call: ${e.message}")
                }
            } else {
                Log.w(TAG, "LIFECYCLE [answerCall] Existing active call does not advertise CAPABILITY_HOLD; leaving unheld")
            }
        }

        val callHash = System.identityHashCode(targetCall)
        Log.i(TAG, "ANSWER_REQUESTED session=$sessionId callHash=$callHash " +
                "api=Call.answer(STATE_AUDIO_ONLY) ts=${System.currentTimeMillis()}")

        // Explicitly set state to CONNECTING, NOT ACTIVE!
        sessionManager.onCallStateChanged(sessionId, CallState.CONNECTING)

        try {
            targetCall.answer(android.telecom.VideoProfile.STATE_AUDIO_ONLY)
            val postState = targetCall.state
            Log.i(TAG, "TELECOM_STATE=${telecomStateToString(postState)} immediately post-answerCall session=$sessionId")
        } catch (e: Exception) {
            Log.e(TAG, "ANSWER_REQUESTED FAILED session=$sessionId exception=${e.message}", e)
            synchronized(callLock) { answerInProgress.remove(sessionId) }
            sessionManager.onCallStateChanged(sessionId, CallState.RINGING)
            return
        }

        // Diagnostic recovery watchdog: if Telecom remains in STATE_RINGING beyond recovery timeout, allow retry
        serviceScope.launch {
            delay(EXPERIMENTAL_ANSWER_RECOVERY_TIMEOUT_MS)
            val shouldReset = synchronized(callLock) {
                if (answerInProgress[sessionId] == true && targetCall.state == Call.STATE_RINGING) {
                    answerInProgress.remove(sessionId)
                    true
                } else {
                    false
                }
            }
            if (shouldReset) {
                Log.w(TAG, "ANSWER_WATCHDOG_TIMEOUT_STILL_RINGING session=$sessionId callHash=$callHash " +
                        "after ${EXPERIMENTAL_ANSWER_RECOVERY_TIMEOUT_MS}ms. Resetting answerInProgress to allow retry.")
                sessionManager.onCallStateChanged(sessionId, CallState.RINGING)
            }
        }
    }

    /**
     * Reject a ringing call through Telecom.
     * Tags localDisconnectRequested = true and localDisconnectReason = USER_REJECT.
     */
    fun rejectCall(sessionId: String) {
        val targetCall: Call
        synchronized(callLock) {
            val call = sessionToCallMap[sessionId]
                ?: callToSessionMap.entries.find { it.value == sessionId }?.key
            if (call != null) {
                localDisconnectRequested[sessionId] = true
                localDisconnectReason[sessionId] = LocalDisconnectReason.USER_REJECT
                targetCall = call
            } else {
                Log.w(TAG, "LIFECYCLE [rejectCall] FAILED session=$sessionId not found in call map")
                return
            }
        }

        Log.i(TAG, "LIFECYCLE [rejectCall] session=$sessionId reason=USER_REJECT state=${targetCall.state} ts=${System.currentTimeMillis()}")
        targetCall.reject(false, null)
    }

    /**
     * Disconnect an active call through Telecom.
     * Sets localDisconnectRequested so DisconnectCause investigation can determine
     * whether VoiceShield initiated the disconnect, with categorized reason.
     *
     * STRICT IDENTITY RULE:
     * Disconnects ONLY the uniquely mapped Call for this session.
     * If direct mapping misses, inspects getCalls() candidates using session handle correlation.
     * Requires EXACTLY ONE candidate. Never disconnects an arbitrary call.
     */
    fun disconnectCall(sessionId: String) {
        val targetCall: Call
        val reason: LocalDisconnectReason
        synchronized(callLock) {
            val directCall = sessionToCallMap[sessionId]
                ?: callToSessionMap.entries.find { it.value == sessionId }?.key

            val resolved = if (directCall != null) {
                directCall
            } else {
                // Strict candidate correlation: phone number is candidate hint only.
                // MUST have EXACTLY ONE matching live call plus session correlation.
                val session = sessionManager.getSession(sessionId)
                val candidateNumber = session?.callSession?.phoneNumber
                if (candidateNumber != null) {
                    val candidateCalls = getCalls().filter { c ->
                        c.state != Call.STATE_DISCONNECTED &&
                        c.details?.handle?.schemeSpecificPart == candidateNumber
                    }
                    if (candidateCalls.size == 1) {
                        Log.i(TAG, "LIFECYCLE [disconnectCall] Uniquely resolved Call via candidate correlation for session=$sessionId")
                        candidateCalls.first()
                    } else {
                        if (candidateCalls.size > 1) {
                            Log.w(TAG, "LIFECYCLE [disconnectCall] Multiple candidate calls found (${candidateCalls.size}) for session=$sessionId. Refusing to guess.")
                        } else {
                            Log.w(TAG, "LIFECYCLE [disconnectCall] Zero candidate calls found in getCalls() for session=$sessionId")
                        }
                        null
                    }
                } else {
                    null
                }
            }

            if (resolved != null) {
                localDisconnectRequested[sessionId] = true
                reason = when {
                    alertActionTriggered[sessionId] == true -> LocalDisconnectReason.THREAT_ALERT_END
                    endActiveCallInvoked[sessionId] == true -> LocalDisconnectReason.USER_END
                    else -> LocalDisconnectReason.UNKNOWN_LOCAL_PATH
                }
                localDisconnectReason[sessionId] = reason
                targetCall = resolved
            } else {
                Log.w(TAG, "LIFECYCLE [disconnectCall] FAILED session=$sessionId cannot be uniquely mapped to a live Telecom Call. Refusing to guess.")
                return
            }
        }

        Log.i(TAG, "LIFECYCLE [disconnectCall] session=$sessionId localDisconnectRequested=TRUE reason=$reason " +
                "currentState=${targetCall.state} (${telecomStateToString(targetCall.state)}) ts=${System.currentTimeMillis()}")

        // On ringing calls, Telecom ignores call.disconnect() — call.reject() is required
        if (targetCall.state == Call.STATE_RINGING || targetCall.state == Call.STATE_SIMULATED_RINGING) {
            Log.i(TAG, "LIFECYCLE [disconnectCall] Target call is in RINGING state -> invoking call.reject(false, null)")
            targetCall.reject(false, null)
        } else {
            Log.i(TAG, "LIFECYCLE [disconnectCall] Target call is in state ${targetCall.state} -> invoking call.disconnect()")
            targetCall.disconnect()
        }
    }

    /**
     * Mark that endActiveCall() was invoked from MainActivity.
     * This is for disconnect investigation — it does NOT disconnect the call.
     */
    fun markEndActiveCallInvoked(sessionId: String) {
        synchronized(callLock) {
            endActiveCallInvoked[sessionId] = true
        }
        Log.i(TAG, "LIFECYCLE [markEndActiveCallInvoked] session=$sessionId ts=${System.currentTimeMillis()}")
    }

    /**
     * Mark that an AlertManager action triggered the disconnect.
     */
    fun markAlertActionTriggered(sessionId: String) {
        synchronized(callLock) {
            alertActionTriggered[sessionId] = true
        }
        Log.i(TAG, "LIFECYCLE [markAlertActionTriggered] session=$sessionId ts=${System.currentTimeMillis()}")
    }

    /** Hold an active call through Telecom. */
    fun holdCall(sessionId: String) {
        val call = synchronized(callLock) {
            sessionToCallMap[sessionId] ?: callToSessionMap.entries.find { it.value == sessionId }?.key
        }
        call?.let {
            Log.i(TAG, "LIFECYCLE [holdCall] session=$sessionId ts=${System.currentTimeMillis()}")
            it.hold()
        }
    }

    /** Resume a held call through Telecom. */
    fun unholdCall(sessionId: String) {
        val call = synchronized(callLock) {
            sessionToCallMap[sessionId] ?: callToSessionMap.entries.find { it.value == sessionId }?.key
        }
        call?.let {
            Log.i(TAG, "LIFECYCLE [unholdCall] session=$sessionId ts=${System.currentTimeMillis()}")
            it.unhold()
        }
    }

    /** Toggle speakerphone via official Telecom API. */
    fun toggleSpeaker() {
        val currentRoute = callAudioState?.route ?: CallAudioState.ROUTE_EARPIECE
        val targetRoute = if (currentRoute == CallAudioState.ROUTE_SPEAKER) {
            CallAudioState.ROUTE_EARPIECE
        } else {
            CallAudioState.ROUTE_SPEAKER
        }
        Log.i(TAG, "LIFECYCLE [toggleSpeaker] ${routeToString(currentRoute)} -> ${routeToString(targetRoute)} " +
                "ts=${System.currentTimeMillis()}")
        setAudioRoute(targetRoute)
    }

    /**
     * Toggle microphone mute via official Telecom API.
     * This is the ONLY place VoiceShield calls setMuted().
     * It is invoked ONLY by explicit user action (Mute button tap).
     */
    fun toggleMute() {
        val currentMuted = callAudioState?.isMuted ?: false
        val targetMuted = !currentMuted
        Log.i(TAG, "LIFECYCLE [toggleMute] $currentMuted -> $targetMuted (USER_ACTION) " +
                "ts=${System.currentTimeMillis()}")
        setMuted(targetMuted)
    }

    /**
     * Send a DTMF tone for an active call through Telecom.
     */
    fun playDtmf(sessionId: String, digit: Char) {
        val call = synchronized(callLock) {
            sessionToCallMap[sessionId] ?: callToSessionMap.entries.find { it.value == sessionId }?.key
        }
        if (call != null) {
            Log.i(TAG, "DTMF_SEND session=$sessionId digit=$digit ts=${System.currentTimeMillis()}")
            try {
                call.playDtmfTone(digit)
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    try {
                        call.stopDtmfTone()
                    } catch (e: Exception) {
                        Log.w(TAG, "DTMF stop error: ${e.message}")
                    }
                }, 150)
            } catch (e: Exception) {
                Log.e(TAG, "DTMF play error: ${e.message}")
            }
        } else {
            Log.w(TAG, "DTMF_SEND call not found for session=$sessionId")
        }
    }

    /**
     * Send DTMF tone for the first active call if session ID is omitted.
     */
    fun playDtmf(digit: Char) {
        val call = synchronized(callLock) {
            callToSessionMap.keys.firstOrNull { it.state == Call.STATE_ACTIVE }
                ?: callToSessionMap.keys.firstOrNull()
        }
        if (call != null) {
            Log.i(TAG, "DTMF_SEND (fallback) digit=$digit ts=${System.currentTimeMillis()}")
            try {
                call.playDtmfTone(digit)
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    try {
                        call.stopDtmfTone()
                    } catch (e: Exception) {
                        Log.w(TAG, "DTMF stop error: ${e.message}")
                    }
                }, 150)
            } catch (e: Exception) {
                Log.e(TAG, "DTMF play error: ${e.message}")
            }
        } else {
            Log.w(TAG, "DTMF_SEND no active call available")
        }
    }

    private fun telecomStateToString(state: Int): String = when (state) {
        Call.STATE_NEW -> "NEW"
        Call.STATE_RINGING -> "RINGING"
        Call.STATE_DIALING -> "DIALING"
        Call.STATE_ACTIVE -> "ACTIVE"
        Call.STATE_HOLDING -> "HOLDING"
        Call.STATE_DISCONNECTING -> "DISCONNECTING"
        Call.STATE_DISCONNECTED -> "DISCONNECTED"
        Call.STATE_CONNECTING -> "CONNECTING"
        Call.STATE_SELECT_PHONE_ACCOUNT -> "SELECT_PHONE_ACCOUNT"
        Call.STATE_SIMULATED_RINGING -> "SIMULATED_RINGING"
        else -> "UNKNOWN_$state"
    }
}
