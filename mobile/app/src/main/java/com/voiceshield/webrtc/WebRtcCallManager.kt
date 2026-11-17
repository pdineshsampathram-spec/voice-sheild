/**
 * WebRtcCallManager — Manages WebRTC peer connection for VoiceShield.
 *
 * Handles:
 * - Signaling via WebSocket to FastAPI backend
 * - SDP offer/answer exchange
 * - ICE candidate exchange
 * - Remote audio track reception
 * - PCM frame extraction from remote AudioTrack
 *
 * The critical flow:
 * Remote AudioTrack → AudioTrackSink → PCM frames → onRemoteAudioFrame callback
 *
 * PRIVACY: Raw audio is NEVER logged, stored, or transmitted in on-device mode.
 * Only metadata (sample rate, frame count, timestamps) is observable.
 */
package com.voiceshield.webrtc

import android.content.Context
import android.util.Log
import com.voiceshield.shared.AudioFrame
import com.voiceshield.shared.AudioSourceType
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONObject
import org.webrtc.*
import org.webrtc.audio.JavaAudioDeviceModule
import java.util.concurrent.atomic.AtomicBoolean

class WebRtcCallManager(
    private val context: Context,
    private val signalingUrl: String = "ws://10.0.2.2:8000/api/rtc/ws",
    private val onRemoteAudioFrame: (AudioFrame) -> Unit
) {
    companion object {
        private const val TAG = "VoiceShieldWebRTC"
        private val ICE_SERVERS = listOf(
            PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer()
        )
    }

    private var peerConnectionFactory: PeerConnectionFactory? = null
    private var peerConnection: PeerConnection? = null
    private var signalingSocket: WebSocket? = null
    private val client = OkHttpClient()
    private val isConnected = AtomicBoolean(false)
    private var roomId: String = ""

    // Audio track extraction
    private var remoteAudioTrack: AudioTrack? = null
    private var audioTrackSink: AudioTrackSink? = null

    /**
     * Initialize WebRTC infrastructure.
     */
    fun initialize() {
        Log.i(TAG, "Initializing WebRTC...")

        // Initialize PeerConnectionFactory
        val options = PeerConnectionFactory.InitializationOptions.builder(context)
            .setEnableInternalTracer(false)
            .createInitializationOptions()
        PeerConnectionFactory.initialize(options)

        val audioDeviceModule = JavaAudioDeviceModule.builder(context)
            .createAudioDeviceModule()

        peerConnectionFactory = PeerConnectionFactory.builder()
            .setAudioDeviceModule(audioDeviceModule)
            .createPeerConnectionFactory()

        Log.i(TAG, "WebRTC PeerConnectionFactory initialized")
    }

    /**
     * Join a signaling room and wait for the remote peer.
     */
    fun joinRoom(roomId: String) {
        this.roomId = roomId
        val wsUrl = "$signalingUrl/$roomId"
        Log.i(TAG, "Joining room: $roomId via $wsUrl")

        val request = Request.Builder().url(wsUrl).build()
        signalingSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                Log.i(TAG, "Signaling WebSocket connected to room $roomId")
                isConnected.set(true)
            }

            override fun onMessage(ws: WebSocket, text: String) {
                handleSignalingMessage(text)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Signaling failure: ${t.message}", t)
                isConnected.set(false)
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "Signaling closed: $code $reason")
                isConnected.set(false)
            }
        })
    }

    private fun handleSignalingMessage(text: String) {
        try {
            val json = JSONObject(text)
            val type = json.optString("type")
            val payload = json.optJSONObject("payload") ?: return

            Log.d(TAG, "Signaling message: type=$type")

            when (type) {
                "offer" -> handleOffer(payload)
                "answer" -> handleAnswer(payload)
                "candidate" -> handleCandidate(payload)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling signaling message: ${e.message}", e)
        }
    }

    private fun ensurePeerConnection() {
        if (peerConnection != null) return

        val factory = peerConnectionFactory ?: return
        val rtcConfig = PeerConnection.RTCConfiguration(ICE_SERVERS).apply {
            sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
        }

        peerConnection = factory.createPeerConnection(rtcConfig, object : PeerConnection.Observer {
            override fun onSignalingChange(state: PeerConnection.SignalingState) {
                Log.d(TAG, "Signaling state: $state")
            }

            override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {
                Log.i(TAG, "ICE connection state: $state")
            }

            override fun onIceConnectionReceivingChange(receiving: Boolean) {}

            override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
                Log.d(TAG, "ICE gathering: $state")
            }

            override fun onIceCandidate(candidate: IceCandidate) {
                // Send ICE candidate to remote peer
                val json = JSONObject().apply {
                    put("type", "candidate")
                    put("payload", JSONObject().apply {
                        put("sdpMid", candidate.sdpMid)
                        put("sdpMLineIndex", candidate.sdpMLineIndex)
                        put("candidate", candidate.sdp)
                    })
                }
                signalingSocket?.send(json.toString())
                Log.d(TAG, "Sent ICE candidate: ${candidate.sdpMid}")
            }

            override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {}

            override fun onAddStream(stream: MediaStream) {
                Log.i(TAG, "Remote stream added: ${stream.id}, audio tracks: ${stream.audioTracks.size}")
                if (stream.audioTracks.isNotEmpty()) {
                    setupRemoteAudioSink(stream.audioTracks[0])
                }
            }

            override fun onRemoveStream(stream: MediaStream) {
                Log.i(TAG, "Remote stream removed: ${stream.id}")
            }

            override fun onTrack(transceiver: RtpTransceiver) {
                val track = transceiver.receiver?.track()
                if (track is AudioTrack) {
                    Log.i(TAG, "Remote audio track received via onTrack: ${track.id()}")
                    setupRemoteAudioSink(track)
                }
            }

            override fun onDataChannel(channel: DataChannel) {}
            override fun onRenegotiationNeeded() {
                Log.d(TAG, "Renegotiation needed")
            }
        })

        // Add audio transceiver to receive remote audio
        peerConnection?.addTransceiver(
            MediaStreamTrack.MediaType.MEDIA_TYPE_AUDIO,
            RtpTransceiver.RtpTransceiverInit(RtpTransceiver.RtpTransceiverDirection.RECV_ONLY)
        )

        Log.i(TAG, "PeerConnection created")
    }

    /**
     * Set up the audio sink to extract PCM frames from the remote audio track.
     * This is THE critical path — where remote caller audio enters VoiceShield.
     */
    private fun setupRemoteAudioSink(track: AudioTrack) {
        remoteAudioTrack = track
        track.setEnabled(true)

        val sink = AudioTrackSink { audioBuffer, bitsPerSample, sampleRate, channels, frames, _ ->
            // Convert to float32 PCM
            val floatSamples = FloatArray(frames * channels)

            if (bitsPerSample == 16) {
                // 16-bit PCM → float32 [-1.0, 1.0]
                val shortBuffer = audioBuffer.asShortBuffer()
                for (i in 0 until (frames * channels).coerceAtMost(shortBuffer.remaining())) {
                    floatSamples[i] = shortBuffer.get(i).toFloat() / 32768f
                }
            } else {
                // Assume float32 already or handle other formats
                val floatBuffer = audioBuffer.asFloatBuffer()
                for (i in 0 until (frames * channels).coerceAtMost(floatBuffer.remaining())) {
                    floatSamples[i] = floatBuffer.get(i)
                }
            }

            val audioFrame = AudioFrame(
                samples = floatSamples,
                sampleRate = sampleRate,
                timestamp = System.currentTimeMillis(),
                sourceType = AudioSourceType.WEBRTC_DIRECT,
                channelCount = channels
            )

            onRemoteAudioFrame(audioFrame)
        }

        audioTrackSink = sink
        track.addSink(sink)

        Log.i(TAG, "Remote audio sink attached — PCM frames will flow to VoiceShield pipeline")
    }

    private fun handleOffer(payload: JSONObject) {
        ensurePeerConnection()
        val pc = peerConnection ?: return

        val sdp = SessionDescription(
            SessionDescription.Type.OFFER,
            payload.getString("sdp")
        )

        pc.setRemoteDescription(object : SdpObserver {
            override fun onSetSuccess() {
                Log.i(TAG, "Remote offer set successfully")
                // Create answer
                pc.createAnswer(object : SdpObserver {
                    override fun onCreateSuccess(answer: SessionDescription) {
                        pc.setLocalDescription(object : SdpObserver {
                            override fun onSetSuccess() {
                                // Send answer to remote peer
                                val json = JSONObject().apply {
                                    put("type", "answer")
                                    put("payload", JSONObject().apply {
                                        put("sdp", answer.description)
                                        put("type", "answer")
                                    })
                                }
                                signalingSocket?.send(json.toString())
                                Log.i(TAG, "Answer sent to remote peer")
                            }
                            override fun onSetFailure(error: String) {
                                Log.e(TAG, "Set local description failed: $error")
                            }
                            override fun onCreateSuccess(p0: SessionDescription?) {}
                            override fun onCreateFailure(p0: String?) {}
                        }, answer)
                    }
                    override fun onCreateFailure(error: String) {
                        Log.e(TAG, "Create answer failed: $error")
                    }
                    override fun onSetSuccess() {}
                    override fun onSetFailure(p0: String?) {}
                }, MediaConstraints())
            }
            override fun onSetFailure(error: String) {
                Log.e(TAG, "Set remote offer failed: $error")
            }
            override fun onCreateSuccess(p0: SessionDescription?) {}
            override fun onCreateFailure(p0: String?) {}
        }, sdp)
    }

    private fun handleAnswer(payload: JSONObject) {
        val pc = peerConnection ?: return
        val sdp = SessionDescription(
            SessionDescription.Type.ANSWER,
            payload.getString("sdp")
        )
        pc.setRemoteDescription(object : SdpObserver {
            override fun onSetSuccess() {
                Log.i(TAG, "Remote answer set successfully")
            }
            override fun onSetFailure(error: String) {
                Log.e(TAG, "Set remote answer failed: $error")
            }
            override fun onCreateSuccess(p0: SessionDescription?) {}
            override fun onCreateFailure(p0: String?) {}
        }, sdp)
    }

    private fun handleCandidate(payload: JSONObject) {
        val pc = peerConnection ?: return
        val candidate = IceCandidate(
            payload.getString("sdpMid"),
            payload.getInt("sdpMLineIndex"),
            payload.getString("candidate")
        )
        pc.addIceCandidate(candidate)
        Log.d(TAG, "Added ICE candidate: ${candidate.sdpMid}")
    }

    /**
     * Create an offer to initiate a call (when VoiceShield is the caller).
     */
    fun createOffer() {
        ensurePeerConnection()
        val pc = peerConnection ?: return

        pc.createOffer(object : SdpObserver {
            override fun onCreateSuccess(offer: SessionDescription) {
                pc.setLocalDescription(object : SdpObserver {
                    override fun onSetSuccess() {
                        val json = JSONObject().apply {
                            put("type", "offer")
                            put("payload", JSONObject().apply {
                                put("sdp", offer.description)
                                put("type", "offer")
                            })
                        }
                        signalingSocket?.send(json.toString())
                        Log.i(TAG, "Offer sent to signaling server")
                    }
                    override fun onSetFailure(error: String) {
                        Log.e(TAG, "Set local description failed: $error")
                    }
                    override fun onCreateSuccess(p0: SessionDescription?) {}
                    override fun onCreateFailure(p0: String?) {}
                }, offer)
            }
            override fun onCreateFailure(error: String) {
                Log.e(TAG, "Create offer failed: $error")
            }
            override fun onSetSuccess() {}
            override fun onSetFailure(p0: String?) {}
        }, MediaConstraints())
    }

    /**
     * Disconnect and clean up all WebRTC resources.
     */
    fun disconnect() {
        audioTrackSink?.let { sink ->
            remoteAudioTrack?.removeSink(sink)
        }
        audioTrackSink = null
        remoteAudioTrack = null
        peerConnection?.close()
        peerConnection = null
        signalingSocket?.close(1000, "Disconnecting")
        signalingSocket = null
        isConnected.set(false)
        Log.i(TAG, "WebRTC disconnected and cleaned up")
    }

    fun isConnected(): Boolean = isConnected.get()
}
