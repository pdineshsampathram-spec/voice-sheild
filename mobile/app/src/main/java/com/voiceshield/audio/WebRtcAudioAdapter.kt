/**
 * WebRtcAudioAdapter — Adapter A: Direct WebRTC / VoIP Audio Path.
 *
 * For calls where VoiceShield legitimately receives the remote participant's
 * real media stream (e.g. secure VoIP, WebRTC in-app demo calls).
 *
 * Pipeline:
 * Remote WebRTC Audio → WebRtcCallManager → AudioTrack.addSink → PCM frames
 * → AudioNormalizer → onAudioFrame callback → Detection Engine
 *
 * This is the primary guaranteed live-audio path for VoiceShield real-time
 * voice clone detection demonstration.
 */
package com.voiceshield.audio

import android.content.Context
import android.util.Log
import com.voiceshield.shared.AudioFrame
import com.voiceshield.shared.AudioSourceType
import com.voiceshield.shared.CallSession
import com.voiceshield.webrtc.WebRtcCallManager
import java.util.concurrent.atomic.AtomicBoolean

class WebRtcAudioAdapter(
    private val context: Context,
    private val signalingBaseUrl: String = "ws://10.0.2.2:8000/api/rtc/ws"
) : AudioSourceAdapter {

    companion object {
        private const val TAG = "VoiceShieldWebRTC"
    }

    private val isRunning = AtomicBoolean(false)
    private var callManager: WebRtcCallManager? = null
    private val normalizer = AudioNormalizer()

    override fun isAvailable(): Boolean = isRunning.get()

    override fun sourceType(): AudioSourceType = AudioSourceType.WEBRTC_DIRECT

    /**
     * Start capturing remote audio from WebRTC media stream.
     * The WebRtcCallManager establishes a real peer connection and
     * delivers remote audio frames via AudioTrack.addSink.
     */
    override suspend fun start(
        session: CallSession,
        onAudioFrame: (AudioFrame) -> Unit
    ) {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "WebRtcAudioAdapter already running")
            return
        }

        Log.i(TAG, "Starting WebRTC audio capture for session ${session.id}")

        // Create WebRTC call manager with the audio frame callback
        callManager = WebRtcCallManager(
            context = context,
            signalingUrl = signalingBaseUrl,
            onRemoteAudioFrame = { frame ->
                if (isRunning.get()) {
                    val normalized = normalizer.normalize(frame)
                    onAudioFrame(normalized)
                }
            }
        )

        // Initialize WebRTC and join the room
        callManager?.initialize()
        val roomId = session.id.take(8) // Use session ID prefix as room ID
        callManager?.joinRoom(roomId)

        Log.i(TAG, "WebRTC adapter started, room=$roomId")
    }

    /**
     * Get the room ID for this session (so the browser peer can join).
     */
    fun getRoomId(session: CallSession): String = session.id.take(8)

    override suspend fun stop() {
        isRunning.set(false)
        callManager?.disconnect()
        callManager = null
        Log.i(TAG, "WebRtcAudioAdapter stopped")
    }
}
