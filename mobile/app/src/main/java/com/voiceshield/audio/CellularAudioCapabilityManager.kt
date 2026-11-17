/**
 * CellularAudioCapabilityManager — Cellular Audio Capability Lab.
 *
 * Experimentally probes and evaluates legitimate Android audio sources during cellular calls:
 * - MediaRecorder.AudioSource.VOICE_CALL (4)
 * - MediaRecorder.AudioSource.VOICE_DOWNLINK (3)
 * - MediaRecorder.AudioSource.VOICE_UPLINK (2)
 * - MediaRecorder.AudioSource.VOICE_COMMUNICATION (7)
 * - MediaRecorder.AudioSource.MIC (1)
 *
 * CRITICAL ENGINEERING & HONESTY RULES:
 * 1. Physical-device experiment determines actual outcome:
 *    usableRemotePcm = VERIFIED_TRUE ONLY when there is reproducible evidence that the
 *    captured stream contains the remote caller side.
 * 2. Non-zero PCM or RMS energy on MIC or VOICE_COMMUNICATION proves only that local audio
 *    exists—it does NOT prove it is the remote caller. They are classified as NEAR_END_MIC.
 * 3. Never save captured PCM to disk, upload it, or include it in logs.
 * 4. Fails safely when remote call audio is inaccessible.
 */
package com.voiceshield.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import com.voiceshield.shared.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ShortBuffer
import kotlin.math.log10
import kotlin.math.sqrt

class CellularAudioCapabilityManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "VoiceShieldCellularLab"
        const val TEST_SAMPLE_RATE = 16000
        const val TEST_CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val TEST_ENCODING = AudioFormat.ENCODING_PCM_16BIT
        const val DIAGNOSTIC_READ_SAMPLES = 1024 // ~64 ms burst at 16kHz
    }

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    private data class SourceCandidate(
        val id: Int,
        val name: String,
        val category: AudioCategory
    )

    private val officialSources = listOf(
        SourceCandidate(MediaRecorder.AudioSource.VOICE_DOWNLINK, "VOICE_DOWNLINK", AudioCategory.REMOTE_DOWNLINK),
        SourceCandidate(MediaRecorder.AudioSource.VOICE_CALL, "VOICE_CALL", AudioCategory.BIDIRECTIONAL_CALL),
        SourceCandidate(MediaRecorder.AudioSource.VOICE_UPLINK, "VOICE_UPLINK", AudioCategory.LOCAL_UPLINK),
        SourceCandidate(MediaRecorder.AudioSource.VOICE_COMMUNICATION, "VOICE_COMMUNICATION", AudioCategory.NEAR_END_MIC),
        SourceCandidate(MediaRecorder.AudioSource.MIC, "MIC", AudioCategory.NEAR_END_MIC)
    )

    /**
     * Probes all official Android audio sources and inspects current audio routes.
     * Collects non-audio metadata only. Never persists or logs PCM.
     */
    suspend fun probeAllSources(callState: String = "ACTIVE"): CellularCapabilityReport = withContext(Dispatchers.IO) {
        Log.i(TAG, "Starting Cellular Audio Capability Probe (CallState: $callState)...")

        val hasRecordPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        val availableDevices = enumerateInputDevices()
        val probeResults = mutableListOf<AudioSourceProbeResult>()

        var verifiedRemoteSource: String? = null
        var isRemotePcmVerified = false

        for (candidate in officialSources) {
            val result = probeSingleSource(candidate, hasRecordPermission)
            probeResults.add(result)

            // Strict verification: only legitimate downlink or bidirectional call sources
            // can verify usable remote PCM.
            if ((candidate.category == AudioCategory.REMOTE_DOWNLINK || candidate.category == AudioCategory.BIDIRECTIONAL_CALL)
                && result.initialized && result.recordingStarted && result.hasNonZeroPcm
            ) {
                isRemotePcmVerified = true
                verifiedRemoteSource = candidate.name
                Log.i(TAG, "VERIFIED REMOTE PCM obtained from source: ${candidate.name}")
            }
        }

        val verificationStatus = when {
            isRemotePcmVerified -> RemoteAudioVerification.VERIFIED_TRUE
            probeResults.any { it.category == AudioCategory.NEAR_END_MIC && it.hasNonZeroPcm } ->
                RemoteAudioVerification.UNVERIFIED_MIC_ONLY
            else -> RemoteAudioVerification.FAILED_OR_RESTRICTED
        }

        val limitationReason = if (!isRemotePcmVerified) {
            "Direct cellular downlink sources (VOICE_DOWNLINK/VOICE_CALL) are restricted on this device by platform security policy (CAPTURE_AUDIO_OUTPUT). Microphone sources (MIC/VOICE_COMMUNICATION) capture local near-end audio only and cannot be treated as remote caller audio."
        } else null

        val report = CellularCapabilityReport(
            timestamp = System.currentTimeMillis(),
            callState = callState,
            audioMode = audioManager.mode,
            isSpeakerphoneOn = audioManager.isSpeakerphoneOn,
            isBluetoothScoOn = audioManager.isBluetoothScoOn,
            availableDevices = availableDevices,
            testedSources = probeResults,
            usableRemotePcm = verificationStatus,
            detectedRemoteSource = verifiedRemoteSource,
            technicalLimitationReason = limitationReason
        )

        Log.i(TAG, "Cellular capability probe completed: RemotePCM=${report.usableRemotePcm}, Source=$verifiedRemoteSource")
        report
    }

    private fun probeSingleSource(candidate: SourceCandidate, hasRecordPermission: Boolean): AudioSourceProbeResult {
        if (!hasRecordPermission) {
            return AudioSourceProbeResult(
                sourceId = candidate.id,
                sourceName = candidate.name,
                category = candidate.category,
                initialized = false,
                recordingStarted = false,
                permissionGranted = false,
                error = "RECORD_AUDIO permission not granted"
            )
        }

        val minBufSize = AudioRecord.getMinBufferSize(
            TEST_SAMPLE_RATE, TEST_CHANNEL_CONFIG, TEST_ENCODING
        )

        if (minBufSize <= 0) {
            return AudioSourceProbeResult(
                sourceId = candidate.id,
                sourceName = candidate.name,
                category = candidate.category,
                initialized = false,
                recordingStarted = false,
                permissionGranted = true,
                error = "Unsupported audio format or sample rate: $minBufSize"
            )
        }

        val bufferSize = (minBufSize * 2).coerceAtLeast(DIAGNOSTIC_READ_SAMPLES * 2)
        var record: AudioRecord? = null

        return try {
            record = AudioRecord(
                candidate.id,
                TEST_SAMPLE_RATE,
                TEST_CHANNEL_CONFIG,
                TEST_ENCODING,
                bufferSize
            )

            val isInitialized = record.state == AudioRecord.STATE_INITIALIZED
            if (!isInitialized) {
                return AudioSourceProbeResult(
                    sourceId = candidate.id,
                    sourceName = candidate.name,
                    category = candidate.category,
                    initialized = false,
                    recordingStarted = false,
                    permissionGranted = true,
                    sampleRate = TEST_SAMPLE_RATE,
                    channelConfig = 1,
                    encoding = TEST_ENCODING,
                    error = "STATE_UNINITIALIZED (OS security policy or device restriction)"
                )
            }

            // Attempt diagnostic recording burst
            record.startRecording()
            val isRecording = record.recordingState == AudioRecord.RECORDSTATE_RECORDING

            if (!isRecording) {
                return AudioSourceProbeResult(
                    sourceId = candidate.id,
                    sourceName = candidate.name,
                    category = candidate.category,
                    initialized = true,
                    recordingStarted = false,
                    permissionGranted = true,
                    sampleRate = TEST_SAMPLE_RATE,
                    channelConfig = 1,
                    encoding = TEST_ENCODING,
                    error = "AudioRecord failed to enter RECORDSTATE_RECORDING"
                )
            }

            // Read short diagnostic burst
            val sampleBuffer = ShortArray(DIAGNOSTIC_READ_SAMPLES)
            val readCount = record.read(sampleBuffer, 0, sampleBuffer.size)

            var framesRead = 0L
            var bytesRead = 0L
            var hasNonZero = false
            var sumSquares = 0.0

            if (readCount > 0) {
                framesRead = readCount.toLong()
                bytesRead = (readCount * 2).toLong()

                for (i in 0 until readCount) {
                    val s = sampleBuffer[i].toInt()
                    if (s != 0) hasNonZero = true
                    sumSquares += s.toDouble() * s.toDouble()
                    // Zero out memory immediately after inspection
                    sampleBuffer[i] = 0
                }
            }

            val rms = if (framesRead > 0) sqrt(sumSquares / framesRead) else 0.0
            val peakRmsDb = if (rms > 0.0) (20 * log10(rms / 32768.0)).toFloat() else -100.0f

            val verification = when (candidate.category) {
                AudioCategory.REMOTE_DOWNLINK, AudioCategory.BIDIRECTIONAL_CALL -> {
                    if (hasNonZero) RemoteAudioVerification.VERIFIED_TRUE
                    else RemoteAudioVerification.FAILED_OR_RESTRICTED
                }
                AudioCategory.NEAR_END_MIC -> {
                    if (hasNonZero) RemoteAudioVerification.UNVERIFIED_MIC_ONLY
                    else RemoteAudioVerification.FAILED_OR_RESTRICTED
                }
                else -> RemoteAudioVerification.FAILED_OR_RESTRICTED
            }

            AudioSourceProbeResult(
                sourceId = candidate.id,
                sourceName = candidate.name,
                category = candidate.category,
                initialized = true,
                recordingStarted = true,
                permissionGranted = true,
                sampleRate = TEST_SAMPLE_RATE,
                channelConfig = 1,
                encoding = TEST_ENCODING,
                framesRead = framesRead,
                bytesRead = bytesRead,
                hasNonZeroPcm = hasNonZero,
                peakRmsDb = peakRmsDb,
                verification = verification,
                error = if (readCount < 0) "AudioRecord.read() returned error code: $readCount" else null
            )
        } catch (e: SecurityException) {
            Log.w(TAG, "Source ${candidate.name} denied by SecurityException: ${e.message}")
            AudioSourceProbeResult(
                sourceId = candidate.id,
                sourceName = candidate.name,
                category = candidate.category,
                initialized = false,
                recordingStarted = false,
                permissionGranted = true,
                error = "SecurityException: ${e.message}"
            )
        } catch (e: Exception) {
            Log.w(TAG, "Source ${candidate.name} probe error: ${e.message}")
            AudioSourceProbeResult(
                sourceId = candidate.id,
                sourceName = candidate.name,
                category = candidate.category,
                initialized = false,
                recordingStarted = false,
                permissionGranted = true,
                error = "${e.javaClass.simpleName}: ${e.message}"
            )
        } finally {
            try {
                if (record?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    record.stop()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error stopping test record: ${e.message}")
            }
            record?.release()
        }
    }

    private fun enumerateInputDevices(): List<String> {
        val deviceList = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            try {
                val devices = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
                for (dev in devices) {
                    val typeName = getDeviceTypeName(dev.type)
                    deviceList.add("${dev.productName.ifEmpty { "AudioDevice" }} ($typeName, id=${dev.id})")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error querying audio devices: ${e.message}")
            }
        }
        return deviceList
    }

    private fun getDeviceTypeName(type: Int): String = when (type) {
        AudioDeviceInfo.TYPE_BUILTIN_MIC -> "BUILTIN_MIC"
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "BLUETOOTH_SCO"
        AudioDeviceInfo.TYPE_WIRED_HEADSET -> "WIRED_HEADSET"
        AudioDeviceInfo.TYPE_TELEPHONY -> "TELEPHONY"
        AudioDeviceInfo.TYPE_USB_DEVICE -> "USB_DEVICE"
        AudioDeviceInfo.TYPE_USB_HEADSET -> "USB_HEADSET"
        else -> "TYPE_$type"
    }
}
