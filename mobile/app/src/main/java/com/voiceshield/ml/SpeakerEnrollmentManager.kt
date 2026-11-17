/**
 * SpeakerEnrollmentManager — Secure local storage for speaker reference embeddings.
 *
 * Enrollment protocol:
 * 1. Collect N utterances from the user
 * 2. Extract ECAPA-TDNN embedding for each
 * 3. Average embeddings → reference embedding
 * 4. L2-normalize the reference
 * 5. Store ONLY the embedding (never raw audio)
 *
 * ALL DEFAULTS ARE LABELED EXPERIMENTAL:
 * - Number of utterances: EXPERIMENTAL_DEFAULT
 * - Utterance duration: EXPERIMENTAL_DEFAULT
 * - Averaging strategy: EXPERIMENTAL_DEFAULT
 * These values need evaluation against real speaker verification benchmarks.
 *
 * Privacy:
 * - No raw audio is ever stored
 * - Only the embedding vector is persisted
 * - Storage uses Android SharedPreferences (can be upgraded to encrypted storage)
 * - Clear enrollment deletes the embedding permanently
 */
package com.voiceshield.ml

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlin.math.sqrt

class SpeakerEnrollmentManager(
    private val context: Context,
    private val ecapaEngine: EcapaTdnnEngine
) {
    companion object {
        private const val TAG = "VoiceShieldEnrollment"
        private const val PREFS_NAME = "voiceshield_enrollment"
        private const val KEY_EMBEDDING = "reference_embedding"
        private const val KEY_SPEAKER_ID = "speaker_id"
        private const val KEY_ENROLLMENT_TIME = "enrollment_time"
        private const val KEY_UTTERANCE_COUNT = "utterance_count"
        private const val KEY_EMBEDDING_DIM = "embedding_dim"
        
        /**
         * EXPERIMENTAL_DEFAULT: Number of enrollment utterances.
         * Needs evaluation against speaker verification benchmarks.
         */
        const val RECOMMENDED_UTTERANCES = 3 // EXPERIMENTAL_DEFAULT
        
        /**
         * EXPERIMENTAL_DEFAULT: Minimum utterance duration in milliseconds.
         * Needs evaluation against speaker verification benchmarks.
         */
        const val MIN_UTTERANCE_DURATION_MS = 3000L // EXPERIMENTAL_DEFAULT (3 seconds)
        
        /**
         * EXPERIMENTAL_DEFAULT: Minimum audio samples per utterance.
         * = MIN_UTTERANCE_DURATION_MS / 1000 * SAMPLE_RATE
         */
        const val MIN_UTTERANCE_SAMPLES = 48000 // EXPERIMENTAL_DEFAULT (3s at 16kHz)
    }

    private val prefs: SharedPreferences = 
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    // In-memory cache of reference embedding
    private var cachedReference: FloatArray? = null
    private var cachedSpeakerId: String? = null

    // ─── Enrollment State ───────────────────────────────────────────────

    /**
     * Whether a speaker profile has been enrolled.
     */
    fun isEnrolled(): Boolean {
        if (cachedReference != null) return true
        return prefs.contains(KEY_EMBEDDING)
    }

    /**
     * Get the enrolled speaker ID, if any.
     */
    fun getSpeakerId(): String? {
        return cachedSpeakerId ?: prefs.getString(KEY_SPEAKER_ID, null)
    }

    /**
     * Get the reference embedding for the enrolled speaker.
     * Returns null if not enrolled.
     */
    fun getReference(): FloatArray? {
        cachedReference?.let { return it }
        
        val embeddingStr = prefs.getString(KEY_EMBEDDING, null) ?: return null
        try {
            val embedding = embeddingStr.split(",")
                .map { it.trim().toFloat() }
                .toFloatArray()
            
            cachedReference = embedding
            cachedSpeakerId = prefs.getString(KEY_SPEAKER_ID, null)
            return embedding
        } catch (e: Exception) {
            Log.e(TAG, "Failed to deserialize enrollment embedding: ${e.message}")
            return null
        }
    }

    // ─── Enrollment Process ─────────────────────────────────────────────

    /**
     * Enroll a speaker from multiple audio utterances.
     *
     * @param utterances List of raw PCM audio samples (16kHz mono float32).
     *                   Each utterance should be at least [MIN_UTTERANCE_SAMPLES] long.
     * @param speakerId An identifier for the enrolled speaker (e.g., "owner").
     * @return true if enrollment succeeded, false otherwise.
     */
    suspend fun enroll(
        utterances: List<FloatArray>,
        speakerId: String = "owner"
    ): EnrollmentResult {
        if (!ecapaEngine.isReady()) {
            return EnrollmentResult(
                success = false,
                error = "ECAPA-TDNN engine not ready"
            )
        }

        if (utterances.isEmpty()) {
            return EnrollmentResult(
                success = false,
                error = "No utterances provided"
            )
        }

        // Validate utterance lengths
        for ((i, utterance) in utterances.withIndex()) {
            if (utterance.size < MIN_UTTERANCE_SAMPLES) {
                return EnrollmentResult(
                    success = false,
                    error = "Utterance $i too short: ${utterance.size} samples " +
                            "(min: $MIN_UTTERANCE_SAMPLES = ${MIN_UTTERANCE_DURATION_MS}ms)"
                )
            }
        }

        try {
            // Extract embedding for each utterance
            val embeddings = mutableListOf<FloatArray>()
            for ((i, utterance) in utterances.withIndex()) {
                Log.i(TAG, "Extracting embedding for utterance ${i+1}/${utterances.size}...")
                val embedding = ecapaEngine.extractEmbedding(utterance)
                embeddings.add(embedding)
            }

            // Average embeddings
            val dim = embeddings[0].size
            val averaged = FloatArray(dim)
            for (embedding in embeddings) {
                require(embedding.size == dim) {
                    "Embedding dimension mismatch: expected $dim, got ${embedding.size}"
                }
                for (j in 0 until dim) {
                    averaged[j] += embedding[j]
                }
            }
            for (j in 0 until dim) {
                averaged[j] /= embeddings.size
            }

            // L2 normalize the averaged embedding
            val reference = l2Normalize(averaged)

            // Validate — no NaN or Inf
            for (i in reference.indices) {
                if (reference[i].isNaN() || reference[i].isInfinite()) {
                    return EnrollmentResult(
                        success = false,
                        error = "Reference embedding contains NaN/Inf at index $i"
                    )
                }
            }

            // Store (embedding only — never raw audio)
            storeEmbedding(reference, speakerId, utterances.size, dim)
            
            cachedReference = reference
            cachedSpeakerId = speakerId

            Log.i(TAG, "Enrollment complete: speaker=$speakerId, " +
                    "utterances=${utterances.size}, dim=$dim")

            return EnrollmentResult(
                success = true,
                speakerId = speakerId,
                embeddingDim = dim,
                utterancesUsed = utterances.size
            )

        } catch (e: Exception) {
            Log.e(TAG, "Enrollment failed: ${e.message}", e)
            return EnrollmentResult(
                success = false,
                error = "Enrollment failed: ${e.message}"
            )
        }
    }

    /**
     * Clear the enrolled speaker profile.
     * Permanently deletes the stored embedding.
     */
    fun clearEnrollment() {
        prefs.edit()
            .remove(KEY_EMBEDDING)
            .remove(KEY_SPEAKER_ID)
            .remove(KEY_ENROLLMENT_TIME)
            .remove(KEY_UTTERANCE_COUNT)
            .remove(KEY_EMBEDDING_DIM)
            .apply()
        
        cachedReference = null
        cachedSpeakerId = null
        
        Log.i(TAG, "Enrollment cleared")
    }

    // ─── Internal ───────────────────────────────────────────────────────

    private fun storeEmbedding(
        embedding: FloatArray,
        speakerId: String,
        utteranceCount: Int,
        dim: Int
    ) {
        val embeddingStr = embedding.joinToString(",") { it.toString() }
        
        prefs.edit()
            .putString(KEY_EMBEDDING, embeddingStr)
            .putString(KEY_SPEAKER_ID, speakerId)
            .putLong(KEY_ENROLLMENT_TIME, System.currentTimeMillis())
            .putInt(KEY_UTTERANCE_COUNT, utteranceCount)
            .putInt(KEY_EMBEDDING_DIM, dim)
            .apply()
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vector) sumSq += v * v
        val norm = sqrt(sumSq)
        if (norm < 1e-10f) return vector
        return FloatArray(vector.size) { vector[it] / norm }
    }
}

/**
 * Result of a speaker enrollment attempt.
 */
data class EnrollmentResult(
    val success: Boolean,
    val speakerId: String? = null,
    val embeddingDim: Int? = null,
    val utterancesUsed: Int? = null,
    val error: String? = null
)
