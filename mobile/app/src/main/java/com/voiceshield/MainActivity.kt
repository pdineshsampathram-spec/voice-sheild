/**
 * MainActivity — Primary activity for VoiceShield Mobile.
 *
 * Dynamically switches UI between:
 * 1. HomeScreen (idle / protection dashboard + WebRTC call launcher)
 * 2. IncomingCallScreen (when receiving call)
 * 3. ActiveCallScreen (during active/monitored call with live AI analysis)
 *
 * Manages:
 * - DialerRoleManager (Android Telecom default dialer)
 * - LiveAnalysisCoordinator & WebRtcAudioAdapter
 * - ThreatAlertOverlay (real-time voice clone threat alerts)
 */
package com.voiceshield

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import com.voiceshield.shared.AcousticPipelineTelemetry
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.voiceshield.audio.CellularAcousticAudioAdapter
import com.voiceshield.audio.CellularAudioCapabilityManager
import com.voiceshield.audio.RecordingImporter
import com.voiceshield.audio.WebRtcAudioAdapter
import com.voiceshield.ml.LiveAnalysisCoordinator
import com.voiceshield.ml.PostCallAnalysisCoordinator
import com.voiceshield.security.AlertManager
import com.voiceshield.shared.AudioSourceType
import com.voiceshield.shared.CellularCapabilityReport
import com.voiceshield.shared.ModelStatus
import com.voiceshield.shared.PostCallAnalysisSession
import com.voiceshield.telecom.CallState
import com.voiceshield.telecom.DialerRoleManager
import com.voiceshield.telecom.VoiceShieldInCallService
import com.voiceshield.ui.activecall.ActiveCallScreen
import com.voiceshield.ui.alert.ThreatAlertOverlay
import com.voiceshield.ui.diagnostics.CellularAudioDiagnosticsScreen
import com.voiceshield.ui.home.HomeScreen
import com.voiceshield.ui.incoming.IncomingCallScreen
import com.voiceshield.ui.postcall.PostCallAnalysisScreen
import com.voiceshield.ui.theme.VoiceShieldTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "VoiceShieldMain"
    }

    private lateinit var dialerRoleManager: DialerRoleManager

    // Live WebRTC and Analysis components
    private var webRtcAudioAdapter: WebRtcAudioAdapter? = null
    private var acousticAudioAdapter: CellularAcousticAudioAdapter? = null
    private var liveCoordinator: LiveAnalysisCoordinator? = null
    private val isAcousticCaptureActive = kotlinx.coroutines.flow.MutableStateFlow(false)
    private val toggleMutex = kotlinx.coroutines.sync.Mutex()

    // Post-call analysis
    private var pendingPostCallUri: android.net.Uri? = null

    // Permission request launcher
    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        if (!allGranted) {
            Toast.makeText(
                this,
                "VoiceShield requires phone & microphone permissions to provide call protection",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    // Role request launcher
    private val roleRequestLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        dialerRoleManager.onRoleRequestResult(result.resultCode)
        if (dialerRoleManager.isCurrentlyDefaultDialer()) {
            Toast.makeText(this, "VoiceShield is now your default dialer!", Toast.LENGTH_SHORT).show()
        }
    }

    // SAF file picker for post-call recording selection
    private val recordingPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            Log.i(TAG, "Recording selected: $uri")
            try {
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (e: Exception) {
                Log.d(TAG, "Could not take persistable URI permission: ${e.message}")
            }

            pendingPostCallUri = uri
            VoiceShieldApp.pendingPostCallUri.value = uri
            VoiceShieldApp.postCallCoordinator.prepareRecording(uri)
        } else {
            Log.i(TAG, "Recording selection cancelled")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Allow presenting incoming call UI when locked
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }

        dialerRoleManager = DialerRoleManager(this)
        dialerRoleManager.checkDefaultDialerStatus()

        checkAndRequestPermissions()

        setContent {
            VoiceShieldTheme {
                val isDefaultDialer by dialerRoleManager.isDefaultDialer.collectAsState()

                // Unified active session from VoiceShieldApp (Telecom SIM + WebRTC)
                val activeSession by VoiceShieldApp.sessionManager.activeSession.collectAsState()
                val lastCompletedCall by VoiceShieldApp.sessionManager.lastCompletedCall.collectAsState()
                val inCallService = VoiceShieldInCallService.instance
                val detailedModelHealth by VoiceShieldApp.onDeviceEngine.detailedModelHealth.collectAsState()

                // Threat state & Alerts
                val riskEngine = VoiceShieldApp.riskEngine
                val threatState by riskEngine.threatState.collectAsState()
                val activeAlert by VoiceShieldApp.alertManager.activeAlert.collectAsState()

                // Call duration timer
                var callDurationMs by remember { mutableLongStateOf(0L) }
                LaunchedEffect(activeSession?.callState) {
                    if (activeSession?.callState == CallState.ACTIVE) {
                        val start = activeSession?.callSession?.startTime ?: System.currentTimeMillis()
                        while (true) {
                            callDurationMs = (System.currentTimeMillis() - start).coerceAtLeast(0L)
                            delay(500)
                        }
                    } else {
                        callDurationMs = 0L
                        // Auto-stop acoustic capture & coordinator if call is no longer ACTIVE or HOLDING (e.g. remote disconnect)
                        if (activeSession?.callState != CallState.HOLDING) {
                            if (acousticAudioAdapter != null || liveCoordinator != null) {
                                Log.i(TAG, "Call no longer active — cleaning up acoustic capture resources")
                                acousticAudioAdapter?.stop()
                                acousticAudioAdapter = null
                                liveCoordinator?.stop()
                                liveCoordinator = null
                                isAcousticCaptureActive.value = false
                            }
                        }
                    }
                }

                // Diagnostics & Post-Call State
                var showDiagnostics by remember { mutableStateOf(false) }
                var showPostCallAnalysis by remember { mutableStateOf(false) }
                var manualReport by remember { mutableStateOf<CellularCapabilityReport?>(null) }
                var isProbing by remember { mutableStateOf(false) }
                var telephonyOnlyModeState by remember { mutableStateOf(VoiceShieldApp.telephonyOnlyMode) }
                val oemDiscoveryState by VoiceShieldApp.oemDiscoveryResult.collectAsState()
                val isDiscoveringOemRecording by VoiceShieldApp.isDiscoveringOemRecording.collectAsState()
                val postCallCoordinator = VoiceShieldApp.postCallCoordinator
                val postCallSession by postCallCoordinator.session.collectAsState()
                val coroutineScope = rememberCoroutineScope()

                // When a completed cellular call occurs, or discovery/analysis is active, automatically show post-call screen
                LaunchedEffect(lastCompletedCall, isDiscoveringOemRecording, postCallSession.outcome) {
                    val completed = lastCompletedCall
                    if ((completed != null && completed.isCellular && completed.wasActive) ||
                        isDiscoveringOemRecording ||
                        postCallSession.outcome != com.voiceshield.shared.PostCallAnalysisOutcome.NOT_STARTED) {
                        showPostCallAnalysis = true
                    }
                }

                Box(modifier = Modifier.fillMaxSize()) {
                    when {
                        // Incoming call ringing or connecting state (Telecom) — HIGHEST PRIORITY
                        activeSession != null && (activeSession?.callState == CallState.RINGING || activeSession?.callState == CallState.CONNECTING) -> {
                            IncomingCallScreen(
                                phoneNumber = activeSession?.callSession?.phoneNumber,
                                isConnecting = activeSession?.callState == CallState.CONNECTING,
                                screeningStatus = if (activeSession?.callState == CallState.CONNECTING) "CONNECTING TO CARRIER CALL..." else "SCREENED BY VOICESHIELD",
                                onAnswer = {
                                    activeSession?.callSession?.id?.let { id ->
                                        inCallService?.answerCall(id)
                                    }
                                },
                                onReject = {
                                    activeSession?.callSession?.id?.let { id ->
                                        inCallService?.rejectCall(id)
                                    }
                                }
                            )
                        }

                        // Active call state (Telecom or WebRTC) — active, holding, dialing, or disconnecting — HIGHEST PRIORITY
                        activeSession != null && (activeSession?.callState == CallState.ACTIVE || activeSession?.callState == CallState.HOLDING || activeSession?.callState == CallState.DIALING || activeSession?.callState == CallState.DISCONNECTING) -> {
                            val telecomDiag = activeSession?.telecomAudioDiagnostics
                            val isMuted = telecomDiag?.isMuted ?: false
                            val isSpeaker = telecomDiag?.currentRoute == "SPEAKER"

                            ActiveCallScreen(
                                phoneNumber = activeSession?.callSession?.phoneNumber,
                                callDuration = callDurationMs,
                                audioSourceType = activeSession?.callSession?.audioSourceType
                                    ?: AudioSourceType.UNAVAILABLE,
                                analysisActive = activeSession?.analysisActive ?: false,
                                threatState = threatState,
                                windowsProcessed = activeSession?.windowsProcessed ?: 0,
                                onEndCall = {
                                    activeSession?.callSession?.id?.let { id ->
                                        endActiveCall(id)
                                    }
                                },
                                onHold = {
                                    activeSession?.callSession?.id?.let { id ->
                                        if (activeSession?.callState == CallState.HOLDING) {
                                            inCallService?.unholdCall(id)
                                        } else {
                                            inCallService?.holdCall(id)
                                        }
                                    }
                                },
                                onMute = {
                                    inCallService?.toggleMute()
                                },
                                onSpeaker = {
                                    inCallService?.toggleSpeaker()
                                },
                                isMuted = isMuted,
                                isSpeaker = isSpeaker,
                                isTelephonyOnlyMode = VoiceShieldApp.telephonyOnlyMode,
                                isAcousticCaptureEnabled = isAcousticCaptureActive.collectAsState().value,
                                onToggleAcousticCapture = { enable ->
                                    activeSession?.callSession?.id?.let { id ->
                                        toggleAcousticCapture(id, enable)
                                    }
                                },
                                currentAudioSource = acousticAudioAdapter?.getConfiguredAudioSourceName() ?: "MIC",
                                onSwitchAudioSource = { newSource ->
                                    activeSession?.callSession?.id?.let { sessionId ->
                                        switchAcousticAudioSource(sessionId, newSource)
                                    }
                                },
                                onOpenDiagnostics = { showDiagnostics = true },
                                pipelineTelemetry = liveCoordinator?.pipelineTelemetry?.collectAsState()?.value ?: AcousticPipelineTelemetry(),
                                onSendDtmf = { digit ->
                                    activeSession?.callSession?.id?.let { id ->
                                        inCallService?.playDtmf(id, digit)
                                    } ?: inCallService?.playDtmf(digit)
                                }
                            )
                        }

                        // Post-Call Analysis Screen
                        showPostCallAnalysis -> {
                            PostCallAnalysisScreen(
                                session = postCallSession,
                                onSelectRecording = {
                                    recordingPickerLauncher.launch(RecordingImporter.ACCEPTED_MIME_TYPES)
                                },
                                onStartAnalysis = {
                                    val uri = pendingPostCallUri ?: VoiceShieldApp.pendingPostCallUri.value
                                    if (uri != null) {
                                        coroutineScope.launch(kotlinx.coroutines.Dispatchers.Default) {
                                            postCallCoordinator.analyze(uri)
                                        }
                                    } else {
                                        Toast.makeText(this@MainActivity, "Select a recording first", Toast.LENGTH_SHORT).show()
                                    }
                                },
                                onBack = { showPostCallAnalysis = false },
                                oemDiscoveryResult = oemDiscoveryState,
                                isDiscoveringOemRecording = isDiscoveringOemRecording,
                                onRetryOemDiscovery = {
                                    val completed = lastCompletedCall
                                    if (completed != null) {
                                        VoiceShieldApp.isDiscoveringOemRecording.value = true
                                        VoiceShieldApp.oemDiscoveryResult.value = null
                                        coroutineScope.launch {
                                            val discoverer = com.voiceshield.audio.OemCallRecordingDiscoverer(this@MainActivity)
                                            val result = discoverer.discoverRecordingWithRetry(
                                                phoneNumber = completed.phoneNumber,
                                                callStartTime = completed.startTime,
                                                callEndTime = completed.endTime,
                                                maxWaitMs = 5000L
                                            )
                                            VoiceShieldApp.oemDiscoveryResult.value = result
                                            VoiceShieldApp.isDiscoveringOemRecording.value = false
                                            if (result is com.voiceshield.audio.OemDiscoveryResult.Discovered) {
                                                pendingPostCallUri = result.uri
                                                VoiceShieldApp.pendingPostCallUri.value = result.uri
                                                postCallCoordinator.prepareRecording(result.uri)
                                                postCallCoordinator.analyze(result.uri)
                                            }
                                        }
                                    }
                                }
                            )
                        }

                        // Diagnostics Screen
                        showDiagnostics -> {
                            val isCallActive = activeSession != null &&
                                    (activeSession?.callState == CallState.ACTIVE ||
                                     activeSession?.callState == CallState.CONNECTING ||
                                     activeSession?.callState == CallState.RINGING)

                            CellularAudioDiagnosticsScreen(
                                report = activeSession?.capabilityReport ?: manualReport,
                                telecomDiagnostics = activeSession?.telecomAudioDiagnostics,
                                lastDisconnectInfo = activeSession?.lastDisconnectInfo,
                                isProbing = isProbing,
                                isCallActive = isCallActive,
                                telephonyOnlyMode = telephonyOnlyModeState,
                                onToggleTelephonyOnlyMode = { enabled ->
                                    VoiceShieldApp.telephonyOnlyMode = enabled
                                    telephonyOnlyModeState = enabled
                                    Toast.makeText(
                                        this@MainActivity,
                                        if (enabled) "Telephony-Only Diagnostic Mode: AI Isolated"
                                        else "AI Monitoring Restored (Default)",
                                        Toast.LENGTH_SHORT
                                    ).show()
                                },
                                onRunProbe = {
                                    if (isCallActive) {
                                        Toast.makeText(
                                            this@MainActivity,
                                            "Probe refused: Audio probing is disabled during active calls to protect call audio routing.",
                                            Toast.LENGTH_LONG
                                        ).show()
                                    } else {
                                        coroutineScope.launch {
                                            isProbing = true
                                            val lab = CellularAudioCapabilityManager(this@MainActivity)
                                            val res = lab.probeAllSources(callState = activeSession?.callState?.name ?: "IDLE")
                                            manualReport = res
                                            activeSession?.callSession?.id?.let { id ->
                                                VoiceShieldApp.sessionManager.updateCapabilityReport(id, res)
                                            }
                                            isProbing = false
                                        }
                                    }
                                },
                                onBack = { showDiagnostics = false }
                            )
                        }

                        // Idle / Dashboard State
                        else -> {
                            HomeScreen(
                                isDefaultDialer = isDefaultDialer,
                                callScreeningEnabled = true,
                                engineMode = "PRIVACY (ON-DEVICE)",
                                modelHealth = detailedModelHealth.mapValues { it.value.status },
                                detailedHealth = detailedModelHealth,
                                calibrationAvailable = false,
                                onRequestDialerRole = {
                                    val intent = dialerRoleManager.createRequestIntent(this@MainActivity)
                                    if (intent != null) {
                                        roleRequestLauncher.launch(intent)
                                    }
                                },
                                onStartWebRtcCall = { roomId ->
                                    startWebRtcCall(roomId)
                                },
                                onOpenDiagnostics = { showDiagnostics = true },
                                onOpenPostCallAnalysis = { showPostCallAnalysis = true }
                            )
                        }
                    }

                    // Threat Alert Dialog overlay
                    activeAlert?.let { alert ->
                        ThreatAlertOverlay(
                            threatState = alert.threatState,
                            onAction = { action ->
                                VoiceShieldApp.alertManager.onUserAction(alert.id, action)
                                if (action == AlertManager.UserAction.END_CALL) {
                                    activeSession?.callSession?.id?.let { id ->
                                        endActiveCall(id, isAlertAction = true)
                                    }
                                }
                            }
                        )
                    }
                }
            }
        }
    }

    /**
     * Start a real WebRTC call session and attach the live AI pipeline.
     */
    private fun startWebRtcCall(roomId: String) {
        lifecycleScope.launch {
            try {
                Log.i(TAG, "Starting WebRTC Live Call session for room: $roomId")

                // 1. Initialize On-Device Detection Engine (ONNX)
                val engine = VoiceShieldApp.onDeviceEngine
                engine.initialize()

                // 2. Create and start call session
                val session = VoiceShieldApp.sessionManager.startWebRtcSession(roomId)

                // 3. Start LiveAnalysisCoordinator
                val coordinator = LiveAnalysisCoordinator(engine, AudioSourceType.WEBRTC_DIRECT)
                liveCoordinator = coordinator
                coordinator.start()

                // 4. Start WebRtcAudioAdapter to connect to room & capture remote AudioTrack
                val adapter = WebRtcAudioAdapter(this@MainActivity)
                webRtcAudioAdapter = adapter
                adapter.start(session.callSession) { frame ->
                    coordinator.onAudioFrame(frame)
                }

                // 5. Track windows processed for UI
                launch {
                    coordinator.windowsProcessed.collect { count ->
                        VoiceShieldApp.sessionManager.updateAnalysisState(
                            session.callSession.id,
                            coordinator.isAnalyzing.value,
                            count
                        )
                    }
                }

                Toast.makeText(this@MainActivity, "Connected to room $roomId. Audio analysis active.", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "Error starting WebRTC call: ${e.message}", e)
                Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    /**
     * End active call session and release all audio & ML resources.
     *
     * TELECOM CALLS: Only requests Telecom disconnect. Session cleanup
     * happens in VoiceShieldInCallService.onCallRemoved() — the authoritative
     * lifecycle callback. DO NOT call sessionManager.endSession() here.
     *
     * WEBRTC CALLS: No Telecom lifecycle, so clean up directly.
     *
     * @param isAlertAction If true, the disconnect was triggered by an AlertManager action
     */
    private fun endActiveCall(sessionId: String, isAlertAction: Boolean = false) {
        val session = VoiceShieldApp.sessionManager.getSession(sessionId)
        val isWebRtc = session?.callSession?.audioSourceType == AudioSourceType.WEBRTC_DIRECT

        if (!isWebRtc) {
            // Stop acoustic capture if running
            if (acousticAudioAdapter != null) {
                lifecycleScope.launch {
                    acousticAudioAdapter?.stop()
                    acousticAudioAdapter = null
                    liveCoordinator?.stop()
                    liveCoordinator = null
                    isAcousticCaptureActive.value = false
                }
            }

            // Telecom call: mark investigation flags, then let Telecom disconnect
            val inCallService = VoiceShieldInCallService.instance
            if (inCallService != null) {
                inCallService.markEndActiveCallInvoked(sessionId)
                if (isAlertAction) {
                    inCallService.markAlertActionTriggered(sessionId)
                }
                inCallService.disconnectCall(sessionId)
                Log.i(TAG, "Telecom call $sessionId: disconnect requested. " +
                        "Session cleanup will happen in onCallRemoved(). " +
                        "isAlertAction=$isAlertAction")
            } else {
                Log.w(TAG, "Telecom call $sessionId: endActiveCall invoked but VoiceShieldInCallService.instance is null! " +
                        "Leaving session intact until Telecom unbinds or removes call.")
            }
            // CRITICAL: DO NOT call sessionManager.endSession() here for Telecom calls!
            // Session cleanup for Telecom calls must happen ONLY via onCallRemoved().
        } else {
            // WebRTC call: no Telecom lifecycle, clean up directly
            lifecycleScope.launch {
                liveCoordinator?.stop()
                liveCoordinator = null
                webRtcAudioAdapter?.stop()
                webRtcAudioAdapter = null
                VoiceShieldApp.sessionManager.endSession(sessionId)
                Log.i(TAG, "WebRTC call $sessionId ended and cleaned up")
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        val sessionId = intent.getStringExtra("EXTRA_CALL_SESSION_ID")
        if (sessionId != null) {
            Log.i(TAG, "onNewIntent received with sessionId=$sessionId")
        }
    }

    /**
     * Toggle acoustic capture on or off for the given call session.
     * OFF by default. If enabled, starts CellularAcousticAudioAdapter and LiveAnalysisCoordinator.
     * If call degrades or user toggles off, immediately stops and restores normal state.
     */
    private fun toggleAcousticCapture(sessionId: String, enable: Boolean) {
        lifecycleScope.launch(kotlinx.coroutines.Dispatchers.Default) {
            if (!toggleMutex.tryLock()) {
                Log.w(TAG, "Acoustic toggle ignored: previous transition in progress")
                return@launch
            }
            try {
                if (enable) {
                    // Check runtime microphone permission before starting
                    if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
                        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "Microphone permission required for acoustic analysis", Toast.LENGTH_LONG).show()
                            requestPermissionsLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                        }
                        return@launch
                    }

                    try {
                        // 1. Ensure any previous adapter/coordinator is completely stopped
                        val oldAdapter = acousticAudioAdapter
                        val oldCoordinator = liveCoordinator
                        acousticAudioAdapter = null
                        liveCoordinator = null
                        oldAdapter?.stop()
                        oldCoordinator?.stop()

                        // 2. Initialize models on Dispatchers.IO (NON-BLOCKING to UI thread)
                        val engine = VoiceShieldApp.onDeviceEngine
                        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                            engine.initialize()
                        }

                        // 3. Create fresh adapter & coordinator
                        val adapter = CellularAcousticAudioAdapter(this@MainActivity)
                        acousticAudioAdapter = adapter
                        adapter.setEnabled(true)

                        val coordinator = LiveAnalysisCoordinator(engine, AudioSourceType.ACOUSTIC_CAPTURE)
                        liveCoordinator = coordinator
                        coordinator.start()

                        // Observe adapter diagnostics and feed to coordinator
                        launch {
                            adapter.diagnosticsFlow.collect { diag ->
                                coordinator.updateCaptureDiagnostics(diag)
                            }
                        }

                        val session = VoiceShieldApp.sessionManager.getSession(sessionId)
                        if (session != null) {
                            adapter.start(session.callSession) { frame ->
                                coordinator.onAudioFrame(frame)
                            }
                            VoiceShieldApp.sessionManager.updateAudioSource(sessionId, AudioSourceType.ACOUSTIC_CAPTURE)
                            VoiceShieldApp.sessionManager.updateAnalysisState(sessionId, true, 0)
                            isAcousticCaptureActive.value = true

                            launch {
                                coordinator.windowsProcessed.collect { count ->
                                    VoiceShieldApp.sessionManager.updateAnalysisState(
                                        sessionId,
                                        coordinator.isAnalyzing.value,
                                        count
                                    )
                                }
                            }

                            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                                Toast.makeText(this@MainActivity, "Acoustic mode enabled (speakerphone recommended)", Toast.LENGTH_SHORT).show()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error starting acoustic capture: ${e.message}", e)
                        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                            Toast.makeText(this@MainActivity, "Failed to start acoustic analysis: ${e.message}", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    val adapter = acousticAudioAdapter
                    val coordinator = liveCoordinator
                    acousticAudioAdapter = null
                    liveCoordinator = null
                    isAcousticCaptureActive.value = false
                    adapter?.stop()
                    coordinator?.stop()
                    VoiceShieldApp.sessionManager.updateAudioSource(sessionId, AudioSourceType.UNAVAILABLE)
                    VoiceShieldApp.sessionManager.updateAnalysisState(sessionId, false, 0)
                    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Acoustic analysis stopped", Toast.LENGTH_SHORT).show()
                    }
                }
            } finally {
                toggleMutex.unlock()
            }
        }
    }

    /**
     * Switch the audio source for acoustic capture during an active call.
     * Performs: stop → setDebugAudioSource → start cycle on the SAME adapter.
     * No concurrent AudioRecord instances are created.
     */
    private fun switchAcousticAudioSource(sessionId: String, sourceName: String) {
        val sourceId = when (sourceName) {
            "MIC" -> android.media.MediaRecorder.AudioSource.MIC
            "VOICE_COMMUNICATION" -> android.media.MediaRecorder.AudioSource.VOICE_COMMUNICATION
            else -> {
                Log.w(TAG, "switchAcousticAudioSource: rejected source $sourceName")
                return
            }
        }

        lifecycleScope.launch(kotlinx.coroutines.Dispatchers.Default) {
            if (!toggleMutex.tryLock()) {
                Log.w(TAG, "Source switch ignored: transition in progress")
                return@launch
            }
            try {
                val adapter = acousticAudioAdapter
                if (adapter == null) {
                    Log.w(TAG, "Source switch ignored: no active adapter")
                    return@launch
                }

                val session = VoiceShieldApp.sessionManager.getSession(sessionId)
                if (session == null) {
                    Log.w(TAG, "Source switch ignored: session $sessionId not found")
                    return@launch
                }

                Log.i(TAG, "Switching audio source to $sourceName (stop → configure → start)")

                // 1. Stop current capture
                adapter.stop()

                // 2. Small delay for AudioRecord release
                kotlinx.coroutines.delay(200)

                // 3. Configure new source
                adapter.setDebugAudioSource(sourceId)
                adapter.setEnabled(true)

                // 4. Restart with new source
                val coordinator = liveCoordinator
                adapter.start(session.callSession) { frame ->
                    coordinator?.onAudioFrame(frame)
                }

                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Source: $sourceName", Toast.LENGTH_SHORT).show()
                }
            } finally {
                toggleMutex.unlock()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        dialerRoleManager.checkDefaultDialerStatus()
    }

    private fun checkAndRequestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.CALL_PHONE,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.RECORD_AUDIO
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
            permissions.add(Manifest.permission.READ_MEDIA_AUDIO)
        } else {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isNotEmpty()) {
            requestPermissionsLauncher.launch(missing.toTypedArray())
        }
    }

    override fun onDestroy() {
        val adapter = acousticAudioAdapter
        val coordinator = liveCoordinator
        acousticAudioAdapter = null
        liveCoordinator = null
        if (adapter != null || coordinator != null) {
            kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Default).launch {
                adapter?.stop()
                coordinator?.stop()
            }
        }
        super.onDestroy()
    }
}
