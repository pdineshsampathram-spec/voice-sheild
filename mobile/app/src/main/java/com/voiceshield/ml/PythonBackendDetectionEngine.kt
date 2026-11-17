/**
 * PythonBackendDetectionEngine — Development Mode detection engine.
 *
 * Streams audio frames via WebSocket to the existing FastAPI VoiceShield backend:
 * ws://<host>:8000/ws/voice-analysis
 *
 * Parses responses into standardized WindowResult objects so that
 * the exact same UI and RiskEngine can consume results from either
 * the Python backend or on-device ONNX runtime.
 */
package com.voiceshield.ml

import android.util.Log
import com.voiceshield.shared.*
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit

class PythonBackendDetectionEngine(
    private val backendWsUrl: String = "ws://10.0.2.2:8000/ws/voice-analysis" // 10.0.2.2 points to host from Android emulator
) : VoiceShieldDetectionEngine {

    companion object {
        private const val TAG = "VoiceShieldBackendEngine"
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .pingInterval(10, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var isConnected = false
    private var isCalibrated = false
    private var windowIndexCounter = 0

    private val resultChannel = Channel<WindowResult>(Channel.BUFFERED)

    private val modelHealthMap = mutableMapOf<String, ModelStatus>(
        "Spectra-AASIST" to ModelStatus.LOADING,
        "ECAPA-TDNN" to ModelStatus.LOADING,
        "Silero VAD" to ModelStatus.LOADING
    )

    override fun getMode(): EngineMode = EngineMode.DEVELOPMENT

    override fun isCalibrated(): Boolean = isCalibrated

    override fun getModelHealth(): Map<String, ModelStatus> = modelHealthMap.toMap()

    override suspend fun initialize(): EngineStatus = withContext(Dispatchers.IO) {
        val request = Request.Builder().url(backendWsUrl).build()

        val listener = object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                Log.i(TAG, "Connected to VoiceShield FastAPI backend at $backendWsUrl")
                isConnected = true
                modelHealthMap["Spectra-AASIST"] = ModelStatus.READY
                modelHealthMap["ECAPA-TDNN"] = ModelStatus.READY
                modelHealthMap["Silero VAD"] = ModelStatus.READY
            }

            override fun onMessage(ws: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    val type = json.optString("type")
                    if (type == "window_result") {
                        val payload = json.getJSONObject("payload")
                        val result = parseWindowResult(payload)
                        resultChannel.trySend(result)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing backend message: $text", e)
                }
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}", t)
                isConnected = false
                modelHealthMap["Spectra-AASIST"] = ModelStatus.ERROR
                modelHealthMap["ECAPA-TDNN"] = ModelStatus.ERROR
                modelHealthMap["Silero VAD"] = ModelStatus.ERROR
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: $code $reason")
                isConnected = false
            }
        }

        webSocket = client.newWebSocket(request, listener)

        // Wait brief moment for connection handshake
        delay(500)

        EngineStatus(
            mode = EngineMode.DEVELOPMENT,
            models = modelHealthMap.map { (name, status) ->
                ModelHealth(name = name, status = status)
            },
            calibrationAvailable = isCalibrated
        )
    }

    override suspend fun analyzeWindow(audio: AudioFrame): WindowResult? = withContext(Dispatchers.IO) {
        val ws = webSocket
        if (ws == null || !isConnected) {
            Log.w(TAG, "Cannot analyze window: backend WebSocket is not connected")
            return@withContext null
        }

        // Convert float32 [-1.0, 1.0] samples to 16-bit PCM bytes
        val pcmBytes = floatArrayToPcm16(audio.samples)
        ws.send(pcmBytes.toByteString())

        // Await the next window result from channel with a 5-second timeout
        try {
            withTimeout(5000) {
                resultChannel.receive()
            }
        } catch (e: TimeoutCancellationException) {
            Log.w(TAG, "Timeout waiting for backend analysis response")
            null
        }
    }

    override fun shutdown() {
        webSocket?.close(1000, "App shutdown")
        webSocket = null
        isConnected = false
        client.dispatcher.executorService.shutdown()
    }

    private fun parseWindowResult(payload: JSONObject): WindowResult {
        val windowIdx = payload.optInt("window_index", windowIndexCounter++)
        val processingTimeMs = payload.optLong("processing_time_ms", 0L)

        // VAD
        val vadObj = payload.optJSONObject("vad")
        val isSpeech = vadObj?.optBoolean("is_speech", true) ?: true
        val speechProb = vadObj?.optDouble("speech_probability", 0.95)?.toFloat() ?: 0.95f
        val vadResult = VadResult(speechProbability = speechProb, isSpeech = isSpeech)

        // Spoof
        val spoofObj = payload.optJSONObject("spoof")
        val antiSpoofResult = if (spoofObj != null) {
            val score = spoofObj.optDouble("spoof_score", 0.0).toFloat()
            AntiSpoofResult(
                modelName = spoofObj.optString("model", "Spectra-AASIST"),
                spoofScore = score,
                bonafideScore = 1f - score,
                label = "ANTI-SPOOF MODEL SCORE"
            )
        } else null

        // Speaker
        val speakerObj = payload.optJSONObject("speaker")
        val speakerResult = if (speakerObj != null) {
            SpeakerResult(
                modelName = speakerObj.optString("model", "ECAPA-TDNN"),
                similarity = speakerObj.optDouble("similarity", 0.0).toFloat(),
                enrolledSpeakerId = speakerObj.optString("speaker_id").takeIf { it.isNotEmpty() },
                match = speakerObj.optBoolean("match")
            )
        } else null

        return WindowResult(
            windowIndex = windowIdx,
            timestamp = System.currentTimeMillis(),
            vad = vadResult,
            antiSpoof = antiSpoofResult,
            speaker = speakerResult,
            processingTimeMs = processingTimeMs
        )
    }

    private fun floatArrayToPcm16(floats: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(floats.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) {
            val clamped = f.coerceIn(-1.0f, 1.0f)
            val shortVal = (clamped * 32767.0f).toInt().toShort()
            buffer.putShort(shortVal)
        }
        return buffer.array()
    }
}
