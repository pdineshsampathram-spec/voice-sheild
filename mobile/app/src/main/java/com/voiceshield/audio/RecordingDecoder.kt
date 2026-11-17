/**
 * RecordingDecoder — Decode-only audio file decoder via MediaExtractor + MediaCodec.
 *
 * DECODE ONLY. Does NOT:
 * - Downmix stereo→mono (coordinator's job)
 * - Resample to 16kHz (coordinator's job)
 * - Apply any normalization
 *
 * Outputs raw decoded PCM in the codec's native format.
 * Streams decoded audio in chunks to bound memory usage.
 * Works from SAF Uri via ContentResolver — no filesystem path needed.
 */
package com.voiceshield.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import com.voiceshield.shared.RecordingStatus
import com.voiceshield.shared.RecordingValidation
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decoded audio chunk — raw PCM in native format from the codec.
 */
data class DecodedAudioChunk(
    val samples: FloatArray,
    val sampleRate: Int,
    val channelCount: Int,
    /** Presentation timestamp in milliseconds from recording start. */
    val timestampMs: Long,
    /** Duration of this chunk in milliseconds. */
    val durationMs: Long
)

/**
 * Callback invoked for each decoded chunk.
 * Return false to stop decoding early.
 */
typealias DecodedChunkCallback = suspend (DecodedAudioChunk) -> Boolean

class RecordingDecoder(private val context: Context) {

    companion object {
        private const val TAG = "RecordingDecoder"
        /** Decode in chunks of approximately this many seconds. */
        private const val CHUNK_DURATION_SECONDS = 10
        private const val TIMEOUT_US = 10_000L
    }

    /**
     * Validate a recording Uri and discover its audio metadata.
     * Does NOT decode the full file — only reads format metadata.
     */
    fun validate(uri: Uri): RecordingValidation {
        val importResult = RecordingImporter.queryRecordingMetadata(context, uri)
            ?: return RecordingValidation(
                status = RecordingStatus.UNREADABLE,
                uri = uri.toString(),
                displayName = null,
                errorMessage = "Cannot read recording metadata"
            )

        if (!RecordingImporter.canRead(context, uri)) {
            return RecordingValidation(
                status = RecordingStatus.PERMISSION_DENIED,
                uri = uri.toString(),
                displayName = importResult.displayName,
                errorMessage = "Cannot open recording for reading"
            )
        }

        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, uri, null)

            // Find audio track
            var audioTrackIndex = -1
            var audioFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = i
                    audioFormat = format
                    break
                }
            }

            if (audioTrackIndex < 0 || audioFormat == null) {
                return RecordingValidation(
                    status = RecordingStatus.UNSUPPORTED_FORMAT,
                    uri = uri.toString(),
                    displayName = importResult.displayName,
                    fileSizeBytes = importResult.sizeBytes,
                    errorMessage = "No audio track found in file"
                )
            }

            val mime = audioFormat.getString(MediaFormat.KEY_MIME) ?: "unknown"
            val sampleRate = audioFormat.getIntSafe(MediaFormat.KEY_SAMPLE_RATE, 0)
            val channelCount = audioFormat.getIntSafe(MediaFormat.KEY_CHANNEL_COUNT, 0)
            val durationUs = audioFormat.getLongSafe(MediaFormat.KEY_DURATION, 0L)
            val durationMs = durationUs / 1000

            if (sampleRate <= 0) {
                return RecordingValidation(
                    status = RecordingStatus.INVALID,
                    uri = uri.toString(),
                    displayName = importResult.displayName,
                    format = mime,
                    fileSizeBytes = importResult.sizeBytes,
                    errorMessage = "Invalid sample rate: $sampleRate"
                )
            }

            if (durationMs <= 0) {
                return RecordingValidation(
                    status = RecordingStatus.INVALID,
                    uri = uri.toString(),
                    displayName = importResult.displayName,
                    format = mime,
                    sampleRate = sampleRate,
                    channelCount = channelCount,
                    fileSizeBytes = importResult.sizeBytes,
                    errorMessage = "Recording duration is zero"
                )
            }

            val codec = mime.substringAfter("audio/", "unknown")

            Log.i(TAG, "Recording validated: $codec, ${sampleRate}Hz, ${channelCount}ch, ${durationMs}ms")

            return RecordingValidation(
                status = RecordingStatus.VERIFIED,
                uri = uri.toString(),
                displayName = importResult.displayName,
                format = mime,
                codec = codec,
                sampleRate = sampleRate,
                channelCount = channelCount,
                durationMs = durationMs,
                fileSizeBytes = importResult.sizeBytes
            )
        } catch (e: Exception) {
            Log.e(TAG, "Recording validation failed: ${e.message}", e)
            return RecordingValidation(
                status = RecordingStatus.UNREADABLE,
                uri = uri.toString(),
                displayName = importResult.displayName,
                errorMessage = "Validation error: ${e.message}"
            )
        } finally {
            extractor.release()
        }
    }

    /**
     * Decode the recording in streaming chunks.
     * Calls [onChunk] for each decoded chunk.
     * The chunk contains raw decoded PCM in the codec's native sample rate and channel count.
     *
     * Returns total decoded sample count, or -1 on error.
     */
    suspend fun decodeStreaming(uri: Uri, onChunk: DecodedChunkCallback): Long {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var totalSamples = 0L

        try {
            extractor.setDataSource(context, uri, null)

            // Find audio track
            var audioTrackIndex = -1
            var audioFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = i
                    audioFormat = format
                    break
                }
            }

            if (audioTrackIndex < 0 || audioFormat == null) {
                Log.e(TAG, "No audio track found")
                return -1
            }

            extractor.selectTrack(audioTrackIndex)

            val mime = audioFormat.getString(MediaFormat.KEY_MIME) ?: return -1
            val sampleRate = audioFormat.getIntSafe(MediaFormat.KEY_SAMPLE_RATE, 16000)
            val channelCount = audioFormat.getIntSafe(MediaFormat.KEY_CHANNEL_COUNT, 1)
            val chunkSampleCount = sampleRate * CHUNK_DURATION_SECONDS * channelCount

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(audioFormat, null, null, 0)
            codec.start()

            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            // Accumulate decoded samples for chunking using primitive array (no boxing)
            var accumulator = FloatArray(chunkSampleCount)
            var accumulatorSize = 0
            var chunkStartTimestampUs = 0L
            var currentTimestampUs = 0L

            while (!outputDone) {
                // Feed input
                if (!inputDone) {
                    val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputIndex)
                        if (inputBuffer != null) {
                            val bytesRead = extractor.readSampleData(inputBuffer, 0)
                            if (bytesRead < 0) {
                                codec.queueInputBuffer(
                                    inputIndex, 0, 0, 0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM
                                )
                                inputDone = true
                            } else {
                                val presentationTimeUs = extractor.sampleTime
                                codec.queueInputBuffer(
                                    inputIndex, 0, bytesRead, presentationTimeUs, 0
                                )
                                extractor.advance()
                            }
                        }
                    }
                }

                // Read output
                val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                when {
                    outputIndex >= 0 -> {
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputDone = true
                        }

                        val outputBuffer = codec.getOutputBuffer(outputIndex)
                        if (outputBuffer != null && bufferInfo.size > 0) {
                            currentTimestampUs = bufferInfo.presentationTimeUs

                            val samples = extractPcmSamples(outputBuffer, bufferInfo.size)
                            if (accumulatorSize + samples.size > accumulator.size) {
                                accumulator = accumulator.copyOf(maxOf(accumulator.size * 2, accumulatorSize + samples.size))
                            }
                            System.arraycopy(samples, 0, accumulator, accumulatorSize, samples.size)
                            accumulatorSize += samples.size
                            totalSamples += samples.size

                            // Emit chunk when accumulated enough
                            if (accumulatorSize >= chunkSampleCount) {
                                val chunkArray = accumulator.copyOfRange(0, accumulatorSize)
                                val chunkDurationMs = (chunkArray.size.toLong() * 1000) / (sampleRate * channelCount)
                                val chunk = DecodedAudioChunk(
                                    samples = chunkArray,
                                    sampleRate = sampleRate,
                                    channelCount = channelCount,
                                    timestampMs = chunkStartTimestampUs / 1000,
                                    durationMs = chunkDurationMs
                                )

                                val shouldContinue = onChunk(chunk)
                                if (!shouldContinue) {
                                    outputDone = true
                                }

                                chunkStartTimestampUs = currentTimestampUs
                                accumulatorSize = 0
                            }
                        }

                        codec.releaseOutputBuffer(outputIndex, false)
                    }
                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        Log.d(TAG, "Output format changed: ${codec.outputFormat}")
                    }
                }
            }

            // Emit remaining samples
            if (accumulatorSize > 0) {
                val chunkArray = accumulator.copyOfRange(0, accumulatorSize)
                val chunkDurationMs = (chunkArray.size.toLong() * 1000) / (sampleRate * channelCount)
                val chunk = DecodedAudioChunk(
                    samples = chunkArray,
                    sampleRate = sampleRate,
                    channelCount = channelCount,
                    timestampMs = chunkStartTimestampUs / 1000,
                    durationMs = chunkDurationMs
                )
                onChunk(chunk)
            }

            Log.i(TAG, "Decoding complete: $totalSamples samples")
            return totalSamples

        } catch (e: Exception) {
            Log.e(TAG, "Decoding failed: ${e.message}", e)
            return -1
        } finally {
            try { codec?.stop() } catch (_: Exception) {}
            try { codec?.release() } catch (_: Exception) {}
            extractor.release()
        }
    }

    /**
     * Extract PCM samples from a MediaCodec output buffer.
     * Assumes 16-bit PCM output (standard for Android MediaCodec audio decoders).
     */
    private fun extractPcmSamples(buffer: ByteBuffer, size: Int): FloatArray {
        buffer.position(0)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        val shortCount = size / 2
        val samples = FloatArray(shortCount)
        for (i in 0 until shortCount) {
            if (buffer.remaining() >= 2) {
                samples[i] = buffer.short.toFloat() / 32768f
            }
        }
        return samples
    }

    private fun MediaFormat.getIntSafe(key: String, default: Int): Int {
        return try { getInteger(key) } catch (_: Exception) { default }
    }

    private fun MediaFormat.getLongSafe(key: String, default: Long): Long {
        return try { getLong(key) } catch (_: Exception) { default }
    }
}
