/**
 * CellularAudioAdapter — Direct Cellular Audio Adapter for VoiceShield.
 *
 * Implements AudioSourceAdapter for ordinary SIM/cellular telephone calls.
 *
 * Responsibilities:
 * 1. Attaches to active cellular call session.
 * 2. Consults CellularAudioCapabilityManager to verify legitimate remote audio.
 * 3. Delivers PCM frames to AudioNormalizer (16kHz mono float32) only if
 *    usable remote PCM is verified.
 * 4. Strictly rejects MIC or near-end microphone sources from masquerading as
 *    remote caller audio.
 * 5. Stops and releases AudioRecord immediately when call ends.
 * 6. Never saves or logs raw call audio.
 */
package com.voiceshield.audio

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import com.voiceshield.shared.*
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

class CellularAudioAdapter(
    private val context: Context,
    private val capabilityManager: CellularAudioCapabilityManager = CellularAudioCapabilityManager(context)
) : AudioSourceAdapter {

    companion object {
        private const val TAG = "VoiceShieldCellularAdapter"
        const val SAMPLE_RATE = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val READ_CHUNK_SAMPLES = 1024
    }

    private val isRunning = AtomicBoolean(false)
    private var recordingJob: Job? = null
    private var audioRecord: AudioRecord? = null
    private val normalizer = AudioNormalizer()

    private var activeSourceType: AudioSourceType = AudioSourceType.UNAVAILABLE
    private var latestReport: CellularCapabilityReport? = null

    override fun isAvailable(): Boolean =
        isRunning.get() && activeSourceType != AudioSourceType.UNAVAILABLE

    override fun sourceType(): AudioSourceType = activeSourceType

    fun getLatestCapabilityReport(): CellularCapabilityReport? = latestReport

    /**
     * Start capturing cellular audio if a legitimate remote route exists.
     */
    override suspend fun start(session: CallSession, onAudioFrame: (AudioFrame) -> Unit) {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "CellularAudioAdapter is already running")
            return
        }

        Log.i(TAG, "Starting Cellular Audio Adapter for session ${session.id}...")

        // 1. Run Capability Lab probe
        val report = capabilityManager.probeAllSources(callState = "ACTIVE")
        latestReport = report

        // 2. Evaluate verification status
        if (report.usableRemotePcm != RemoteAudioVerification.VERIFIED_TRUE) {
            Log.i(TAG, "No legitimate direct remote downlink available (${report.usableRemotePcm}). Falling back safely to UNAVAILABLE.")
            activeSourceType = AudioSourceType.UNAVAILABLE
            isRunning.set(false)
            return
        }

        // 3. Select verified source
        val selectedSourceId = when (report.detectedRemoteSource) {
            "VOICE_DOWNLINK" -> MediaRecorder.AudioSource.VOICE_DOWNLINK
            "VOICE_CALL" -> MediaRecorder.AudioSource.VOICE_CALL
            else -> {
                Log.w(TAG, "Unrecognized remote source ${report.detectedRemoteSource}. Refusing unverified audio.")
                activeSourceType = AudioSourceType.UNAVAILABLE
                isRunning.set(false)
                return
            }
        }

        activeSourceType = if (selectedSourceId == MediaRecorder.AudioSource.VOICE_DOWNLINK) {
            AudioSourceType.CELLULAR_DOWNLINK
        } else {
            AudioSourceType.CELLULAR_MIXED
        }

        // 4. Initialize AudioRecord on verified source
        val minBufSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (minBufSize <= 0) {
            Log.e(TAG, "Invalid minBufferSize: $minBufSize")
            activeSourceType = AudioSourceType.UNAVAILABLE
            isRunning.set(false)
            return
        }

        val bufferSize = (minBufSize * 2).coerceAtLeast(READ_CHUNK_SAMPLES * 2)

        try {
            val record = AudioRecord(selectedSourceId, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize)
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize for source $selectedSourceId")
                record.release()
                activeSourceType = AudioSourceType.UNAVAILABLE
                isRunning.set(false)
                return
            }

            record.startRecording()
            if (record.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                Log.e(TAG, "AudioRecord failed to start recording")
                record.stop()
                record.release()
                activeSourceType = AudioSourceType.UNAVAILABLE
                isRunning.set(false)
                return
            }

            audioRecord = record
            Log.i(TAG, "AudioRecord started on verified source $selectedSourceId ($activeSourceType)")

            // 5. Continuous audio streaming loop
            recordingJob = CoroutineScope(Dispatchers.IO).launch {
                val shortBuffer = ShortArray(READ_CHUNK_SAMPLES)

                while (isRunning.get() && isActive) {
                    val read = record.read(shortBuffer, 0, shortBuffer.size)
                    if (read > 0) {
                        // Convert to float32 [-1.0, 1.0]
                        val floatSamples = FloatArray(read)
                        for (i in 0 until read) {
                            floatSamples[i] = shortBuffer[i].toFloat() / 32768.0f
                            // Zero transient buffer immediately
                            shortBuffer[i] = 0
                        }

                        val frame = AudioFrame(
                            samples = floatSamples,
                            sampleRate = SAMPLE_RATE,
                            timestamp = System.currentTimeMillis(),
                            sourceType = activeSourceType,
                            channelCount = 1
                        )

                        val normalized = normalizer.normalize(frame)
                        onAudioFrame(normalized)
                    } else if (read < 0) {
                        Log.w(TAG, "AudioRecord.read() returned error code: $read")
                        delay(20)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start cellular audio capture: ${e.message}", e)
            activeSourceType = AudioSourceType.UNAVAILABLE
            isRunning.set(false)
            cleanup()
        }
    }

    override suspend fun stop() {
        isRunning.set(false)
        recordingJob?.cancelAndJoin()
        recordingJob = null
        cleanup()
        activeSourceType = AudioSourceType.UNAVAILABLE
        Log.i(TAG, "CellularAudioAdapter stopped and resources released")
    }

    private fun cleanup() {
        try {
            audioRecord?.let {
                if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    it.stop()
                }
                it.release()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error cleaning up AudioRecord: ${e.message}")
        } finally {
            audioRecord = null
        }
    }
}
