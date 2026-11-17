/**
 * VoiceShieldApp — Application class.
 *
 * Initializes shared resources used across the app:
 * - CallSessionManager
 * - RiskEngine
 * - AlertManager
 * - DetectionEngines (On-Device ONNX + Backend WebSocket)
 */
package com.voiceshield

import android.app.Application
import android.util.Log
import android.net.Uri
import com.voiceshield.audio.OemCallRecordingDiscoverer
import com.voiceshield.audio.OemDiscoveryResult
import com.voiceshield.ml.OnDeviceDetectionEngine
import com.voiceshield.ml.PostCallAnalysisCoordinator
import com.voiceshield.ml.PythonBackendDetectionEngine
import com.voiceshield.security.AlertManager
import com.voiceshield.security.RiskEngine
import com.voiceshield.telecom.CallSessionManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow

class VoiceShieldApp : Application() {

    companion object {
        private const val TAG = "VoiceShieldApp"

        lateinit var instance: VoiceShieldApp
            private set

        /** Global call session manager for Telecom & WebRTC calls. */
        val sessionManager: CallSessionManager by lazy { CallSessionManager() }

        /** Global risk engine — shared between telecom services and UI. */
        val riskEngine: RiskEngine by lazy { RiskEngine() }

        /** Global alert manager. */
        val alertManager: AlertManager by lazy { AlertManager() }

        /** Global on-device detection engine (ONNX Runtime). */
        val onDeviceEngine: OnDeviceDetectionEngine by lazy { OnDeviceDetectionEngine(instance) }

        /** Global backend detection engine (WebSocket). */
        val backendEngine: PythonBackendDetectionEngine by lazy { PythonBackendDetectionEngine() }

        /** Global post-call coordinator (lives at Application level, surviving Activity recreation/backgrounding). */
        val postCallCoordinator: PostCallAnalysisCoordinator by lazy { PostCallAnalysisCoordinator(instance) }

        /** Live OEM recording discovery state */
        val oemDiscoveryResult = MutableStateFlow<OemDiscoveryResult?>(null)
        val isDiscoveringOemRecording = MutableStateFlow(false)
        val pendingPostCallUri = MutableStateFlow<Uri?>(null)

        /**
         * Telephony-Only Diagnostic Mode flag.
         * Default: false (AI is enabled by default).
         * Used ONLY for diagnostic isolation comparison.
         * When true, disables all AI / ML inference and audio probing during cellular calls.
         */
        @Volatile
        var telephonyOnlyMode: Boolean = false
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        Log.i(TAG, "VoiceShield Mobile initialized")
        Log.i(TAG, "Calibration: NOT AVAILABLE (prototype)")

        // Crash / ANR forensics: structured uncaught exception logging
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            Log.e(TAG, "CRASH_FORENSICS FATAL_UNCAUGHT_EXCEPTION " +
                    "thread=${thread.name} id=${thread.id} " +
                    "type=${throwable.javaClass.name} msg=${throwable.message} " +
                    "activeSession=${sessionManager.activeSession.value?.callSession?.id} " +
                    "activeState=${sessionManager.activeSession.value?.callState}", throwable)
            defaultHandler?.uncaughtException(thread, throwable)
        }

        // Asynchronously initialize on-device detection engine in background.
        // NON-BLOCKING: Telephony and Telecom binding are never blocked.
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.i(TAG, "Starting background AI engine initialization...")
                onDeviceEngine.initialize()
                Log.i(TAG, "Background AI engine initialization completed")
            } catch (e: Exception) {
                Log.e(TAG, "Background AI engine initialization failed", e)
            }
        }

        // Automatic Post-Call OEM Recording Discovery Flow (Application-scoped)
        // NOT dependent on MainActivity remaining alive or visible!
        // Decoupled & Non-blocking: Telecom call cleanup completes first without interference.
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            sessionManager.lastCompletedCall.collect { completed ->
                if (completed != null && completed.isCellular && completed.wasActive) {
                    Log.i(TAG, "Completed cellular call detected (App-level): ${completed.phoneNumber} duration=${completed.durationMs}ms. Initiating OEM discovery.")
                    isDiscoveringOemRecording.value = true
                    oemDiscoveryStateUpdated(null)

                    // 1. Telecom call cleanup happens first (delay ensures all Telecom binder threads finish)
                    delay(500)

                    // 2. Discover recording in MIUI prototype directory
                    val discoverer = OemCallRecordingDiscoverer(instance)
                    val result = discoverer.discoverRecordingWithRetry(
                        phoneNumber = completed.phoneNumber,
                        callStartTime = completed.startTime,
                        callEndTime = completed.endTime,
                        maxWaitMs = 5000L
                    )

                    oemDiscoveryStateUpdated(result)
                    isDiscoveringOemRecording.value = false

                    when (result) {
                        is OemDiscoveryResult.Discovered -> {
                            Log.i(TAG, "OEM recording discovered automatically: ${result.displayName}. Starting post-call analysis pipeline.")
                            pendingPostCallUri.value = result.uri
                            postCallCoordinator.prepareRecording(result.uri)
                            postCallCoordinator.analyze(result.uri)
                        }
                        is OemDiscoveryResult.Unavailable -> {
                            Log.w(TAG, "Automatic OEM recording discovery unavailable: ${result.reason}")
                        }
                        is OemDiscoveryResult.NotFound -> {
                            Log.i(TAG, "No matching OEM recording discovered: ${result.reason}")
                        }
                        is OemDiscoveryResult.Ambiguous -> {
                            Log.w(TAG, "Automatic OEM recording discovery ambiguous: ${result.reason}")
                        }
                        is OemDiscoveryResult.Invalid -> {
                            Log.w(TAG, "Automatic OEM recording discovery invalid: ${result.reason}")
                        }
                    }
                }
            }
        }
    }

    private fun oemDiscoveryStateUpdated(result: OemDiscoveryResult?) {
        oemDiscoveryResult.value = result
    }
}
