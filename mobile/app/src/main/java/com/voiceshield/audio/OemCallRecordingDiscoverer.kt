/**
 * OemCallRecordingDiscoverer — Prototype Automatic OEM Call Recording Discovery.
 *
 * PROTOTYPE IMPLEMENTATION NOTE:
 * "Automatic discovery currently optimized for Xiaomi MIUI/POCO devices."
 *
 * Known verified path on POCO F1 / Xiaomi MIUI:
 *   MIUI/sound_recorder/call_rec/
 *
 * TODO (FUTURE MILESTONE — UNIVERSAL DISCOVERY ARCHITECTURE):
 * - MediaStore-based discovery where possible (Scoped Storage / MediaStore.Audio.Media)
 * - OEM-specific discovery adapters (Samsung Voice Recorder, OnePlus/Oppo Record, Pixel Call Screen)
 * - SAF fallback for arbitrary user-selected storage providers
 * - Broader Android version compatibility (Android 11+ scoped storage access frameworks)
 *
 * STRICT SAFETY RULES:
 * 1. Read-only inspection — NEVER delete, move, rename, or overwrite OEM recordings.
 * 2. No false automation — if directory is inaccessible or permissions missing,
 *    return POST_CALL_RECORDING_DISCOVERY_UNAVAILABLE and fallback to SAF SELECT RECORDING.
 * 3. Strict correlation — only select candidate files created/modified around the just-ended call.
 */
package com.voiceshield.audio

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.util.Log
import com.voiceshield.shared.RecordingStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.abs

/**
 * Result of OEM recording discovery.
 */
sealed class OemDiscoveryResult(val recordingState: com.voiceshield.shared.PostCallRecordingState) {

    /**
     * Successfully discovered and validated candidate recording (RECORDING_FOUND).
     */
    data class Discovered(
        val file: File,
        val uri: Uri,
        val displayName: String,
        val sizeBytes: Long,
        val durationMs: Long,
        val mimeType: String,
        val confidenceReason: String
    ) : OemDiscoveryResult(com.voiceshield.shared.PostCallRecordingState.RECORDING_FOUND)

    /**
     * Directory could not be accessed, permissions missing, or storage unavailable (RECORDING_NOT_ACCESSIBLE).
     * Must fallback to SAF SELECT RECORDING.
     */
    data class Unavailable(
        val reason: String,
        val code: String = ERROR_CODE_UNAVAILABLE
    ) : OemDiscoveryResult(com.voiceshield.shared.PostCallRecordingState.RECORDING_NOT_ACCESSIBLE) {
        companion object {
            const val ERROR_CODE_UNAVAILABLE = "POST_CALL_RECORDING_DISCOVERY_UNAVAILABLE"
        }
    }

    /**
     * Directory is accessible, but no candidate file matched the call criteria (RECORDING_NOT_FOUND).
     */
    data class NotFound(
        val reason: String,
        val searchedDir: String
    ) : OemDiscoveryResult(com.voiceshield.shared.PostCallRecordingState.RECORDING_NOT_FOUND)

    /**
     * Multiple candidate files matched with equal confidence (RECORDING_AMBIGUOUS).
     */
    data class Ambiguous(
        val reason: String,
        val candidateCount: Int
    ) : OemDiscoveryResult(com.voiceshield.shared.PostCallRecordingState.RECORDING_AMBIGUOUS)

    /**
     * Candidate file was located but failed audio validation / corrupted (RECORDING_INVALID).
     */
    data class Invalid(
        val reason: String,
        val file: File?
    ) : OemDiscoveryResult(com.voiceshield.shared.PostCallRecordingState.RECORDING_INVALID)
}

class OemCallRecordingDiscoverer(
    private val context: Context? = null,
    private val customDirectory: File? = null,
    decoder: RecordingDecoder? = null
) {
    companion object {
        private const val TAG = "OemRecordingDiscoverer"

        const val OPTIMIZATION_NOTICE = "Automatic discovery currently optimized for Xiaomi MIUI/POCO devices."

        /** Known verified MIUI call recording directory paths */
        private const val MIUI_CALL_REC_RELATIVE_PATH = "MIUI/sound_recorder/call_rec"
        private const val MIUI_FALLBACK_ABSOLUTE_PATH = "/storage/emulated/0/MIUI/sound_recorder/call_rec"

        /** Supported audio recording extensions in MIUI */
        val SUPPORTED_EXTENSIONS = setOf("mp3", "m4a", "aac", "wav", "amr", "3gp", "ogg", "flac")

        /** Tolerance window before call start (e.g. clock drift or early recorder trigger) */
        const val PRE_CALL_TOLERANCE_MS = 10_000L

        /** Tolerance window after call end for file flush to disk */
        const val POST_CALL_TOLERANCE_MS = 20_000L
    }

    private val decoder: RecordingDecoder? = decoder ?: context?.let { RecordingDecoder(it) }

    /**
     * Resolve the candidate MIUI directory, verifying existence and readability.
     */
    fun resolvePrototypeDirectory(): File? {
        if (customDirectory != null) {
            return if (customDirectory.exists() && customDirectory.isDirectory && customDirectory.canRead()) {
                customDirectory
            } else {
                null
            }
        }

        val externalStorage = Environment.getExternalStorageDirectory()
        val primaryDir = File(externalStorage, MIUI_CALL_REC_RELATIVE_PATH)
        if (primaryDir.exists() && primaryDir.isDirectory && primaryDir.canRead()) {
            return primaryDir
        }

        val fallbackDir = File(MIUI_FALLBACK_ABSOLUTE_PATH)
        if (fallbackDir.exists() && fallbackDir.isDirectory && fallbackDir.canRead()) {
            return fallbackDir
        }

        return null
    }

    /**
     * Discover recording for a recently ended call with asynchronous polling.
     *
     * @param phoneNumber Cleaned phone number or caller ID
     * @param callStartTime Call activation timestamp (ms)
     * @param callEndTime Call termination timestamp (ms)
     * @param maxWaitMs Maximum time to wait for OEM recorder to flush file
     * @param pollIntervalMs Interval between directory scans
     */
    suspend fun discoverRecordingWithRetry(
        phoneNumber: String?,
        callStartTime: Long,
        callEndTime: Long,
        maxWaitMs: Long = 5000L,
        pollIntervalMs: Long = 1000L
    ): OemDiscoveryResult = withContext(Dispatchers.IO) {
        val targetDir = resolvePrototypeDirectory()
        if (targetDir == null) {
            val err = "POST_CALL_RECORDING_DISCOVERY_UNAVAILABLE: Cannot access prototype directory " +
                    "($MIUI_CALL_REC_RELATIVE_PATH). Storage permissions may be missing or device is not MIUI."
            Log.w(TAG, err)
            return@withContext OemDiscoveryResult.Unavailable(err)
        }

        Log.i(TAG, "Starting OEM recording discovery in: ${targetDir.absolutePath} " +
                "for call: number=$phoneNumber start=$callStartTime end=$callEndTime")

        val deadline = System.currentTimeMillis() + maxWaitMs
        var lastSearchReason = "No matching candidate found in ${targetDir.absolutePath}"

        while (System.currentTimeMillis() <= deadline) {
            val evaluation = evaluateCandidates(targetDir, phoneNumber, callStartTime, callEndTime)
            when (evaluation) {
                is CandidateEvaluation.Ambiguous -> {
                    val reason = "Multiple candidate recordings match this call (${evaluation.candidates.size} candidates with score=${evaluation.candidates.first().score}). Disambiguation failed."
                    Log.w(TAG, reason)
                    return@withContext OemDiscoveryResult.Ambiguous(
                        reason = reason,
                        candidateCount = evaluation.candidates.size
                    )
                }
                is CandidateEvaluation.Unique -> {
                    val candidate = evaluation.candidate
                    val uri = Uri.fromFile(candidate.file)
                    val validation = decoder?.validate(uri)
                    if (validation != null && validation.status != RecordingStatus.VERIFIED) {
                        val reason = "Found candidate ${candidate.file.name} but validation failed: ${validation.status} (${validation.errorMessage ?: "invalid format or corrupted"})"
                        Log.w(TAG, reason)
                        return@withContext OemDiscoveryResult.Invalid(
                            reason = reason,
                            file = candidate.file
                        )
                    }
                    val duration = validation?.durationMs ?: (callEndTime - callStartTime).coerceAtLeast(1000L)
                    val mime = validation?.format ?: "audio/mpeg"
                    Log.i(TAG, "Successfully discovered and validated OEM recording: ${candidate.file.name} " +
                            "(${candidate.file.length()} bytes, ${duration}ms, ${candidate.reason})")
                    return@withContext OemDiscoveryResult.Discovered(
                        file = candidate.file,
                        uri = uri,
                        displayName = candidate.file.name,
                        sizeBytes = candidate.file.length(),
                        durationMs = duration,
                        mimeType = mime,
                        confidenceReason = candidate.reason
                    )
                }
                is CandidateEvaluation.None -> {
                    // Continue polling until deadline
                }
            }
            delay(pollIntervalMs)
        }

        Log.i(TAG, "OEM recording discovery finished without match: $lastSearchReason")
        return@withContext OemDiscoveryResult.NotFound(
            reason = lastSearchReason,
            searchedDir = targetDir.absolutePath
        )
    }

    internal sealed class CandidateEvaluation {
        data class Unique(val candidate: Candidate) : CandidateEvaluation()
        data class Ambiguous(val candidates: List<Candidate>) : CandidateEvaluation()
        object None : CandidateEvaluation()
    }

    internal data class Candidate(val file: File, val score: Int, val reason: String)

    /**
     * Evaluate candidates and determine whether a unique, ambiguous, or no candidate exists.
     */
    internal fun evaluateCandidates(
        directory: File,
        phoneNumber: String?,
        callStartTime: Long,
        callEndTime: Long
    ): CandidateEvaluation {
        val candidates = findAllCandidates(directory, phoneNumber, callStartTime, callEndTime)
        if (candidates.isEmpty()) return CandidateEvaluation.None

        val maxScore = candidates.maxOf { it.score }
        val topCandidates = candidates.filter { it.score == maxScore }

        return if (topCandidates.size == 1) {
            CandidateEvaluation.Unique(topCandidates.first())
        } else {
            CandidateEvaluation.Ambiguous(topCandidates)
        }
    }

    /**
     * Inspect directory and identify candidate recordings created/modified around the call.
     */
    internal fun findBestCandidate(
        directory: File,
        phoneNumber: String?,
        callStartTime: Long,
        callEndTime: Long
    ): Candidate? {
        val evaluation = evaluateCandidates(directory, phoneNumber, callStartTime, callEndTime)
        return when (evaluation) {
            is CandidateEvaluation.Unique -> evaluation.candidate
            is CandidateEvaluation.Ambiguous -> evaluation.candidates.first()
            is CandidateEvaluation.None -> null
        }
    }

    /**
     * Find all candidate recordings within time boundaries with non-zero correlation score.
     */
    internal fun findAllCandidates(
        directory: File,
        phoneNumber: String?,
        callStartTime: Long,
        callEndTime: Long
    ): List<Candidate> {
        val files = try {
            directory.listFiles() ?: return emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "Error listing files in ${directory.absolutePath}: ${e.message}")
            return emptyList()
        }

        val minTime = callStartTime - PRE_CALL_TOLERANCE_MS
        val maxTime = callEndTime + POST_CALL_TOLERANCE_MS

        val cleanNumber = phoneNumber?.filter { it.isDigit() }?.takeLast(10)
        val candidates = mutableListOf<Candidate>()

        for (file in files) {
            if (!file.isFile) continue
            val ext = file.extension.lowercase()
            if (!SUPPORTED_EXTENSIONS.contains(ext)) continue

            // Must be non-empty
            if (file.length() < 1024L) continue

            val modTime = file.lastModified()
            // Verify modification time is bounded to the call lifecycle
            if (modTime < minTime || modTime > maxTime) continue

            var score = 0
            val reasons = mutableListOf<String>()

            // Factor 1: Phone number correlation in filename
            if (!cleanNumber.isNullOrBlank() && cleanNumber.length >= 7) {
                if (file.name.contains(cleanNumber)) {
                    score += 50
                    reasons.add("Phone number ($cleanNumber) in filename")
                }
            }

            // Factor 2: Proximity of modification time to call end time
            val deltaFromEnd = abs(modTime - callEndTime)
            if (deltaFromEnd <= 5000L) {
                score += 30
                reasons.add("File modified within 5s of call end (${deltaFromEnd}ms)")
            } else if (deltaFromEnd <= 15000L) {
                score += 15
                reasons.add("File modified within 15s of call end (${deltaFromEnd}ms)")
            }

            // Factor 3: Creation/start correlation
            val deltaFromStart = modTime - callStartTime
            if (deltaFromStart >= 0) {
                score += 10
            }

            if (score > 0) {
                candidates.add(Candidate(file, score, reasons.joinToString(", ")))
            }
        }

        return candidates
    }
}
