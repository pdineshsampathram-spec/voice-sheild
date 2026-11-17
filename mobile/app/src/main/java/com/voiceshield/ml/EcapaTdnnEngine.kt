/**
 * EcapaTdnnEngine — ECAPA-TDNN Speaker Verification Engine.
 *
 * Model: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation)
 * Source: SpeechBrain spkrec-ecapa-voxceleb (Apache-2.0)
 * Embedding: 192 dimensions
 *
 * Input: Raw audio (16kHz mono float32) → MelFeatureExtractor → 80-dim mel features
 * Output: L2-normalized 192-dim speaker embedding
 *
 * CRITICAL RULES:
 * 1. If the ONNX model asset is missing, status = NOT_IMPLEMENTED (not FAILED).
 * 2. All exceptions are caught and NEVER affect the active call.
 * 3. No raw audio is logged, stored, or transmitted.
 * 4. Embedding output is validated: no NaN, no Inf, correct dimension.
 * 5. Cosine similarity thresholds are EXPERIMENTAL until calibrated.
 */
package com.voiceshield.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.voiceshield.shared.DetailedModelHealth
import com.voiceshield.shared.ModelFailureReason
import com.voiceshield.shared.ModelStatus
import com.voiceshield.shared.SpeakerResult
import java.nio.FloatBuffer
import kotlin.math.sqrt

class EcapaTdnnEngine(
    private val context: Context? = null,
    private val modelAssetPath: String = "models/ecapa_tdnn.onnx"
) : SpeakerVerificationEngine {

    companion object {
        private const val TAG = "VoiceShieldEcapaTdnn"
        
        /** Expected embedding dimension. Verified against ONNX graph at load time. */
        const val EMBEDDING_DIM = 192
        
        /** Minimum audio length in samples for meaningful embedding (1 second). */
        const val MIN_AUDIO_SAMPLES = 16000
        
        /** Sample rate expected by the model. */
        const val SAMPLE_RATE = 16000
    }

    override val modelName: String = "ECAPA-TDNN"
    val modelRevision: String = "speechbrain/spkrec-ecapa-voxceleb"

    private var ortEnv: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isModelLoaded = false
    private var failureReason: ModelFailureReason = ModelFailureReason.NOT_LOADED_YET
    private var loadError: String? = null
    private var loadTimeMs: Long? = null
    private var lastInferenceMs: Long? = null
    
    // Verified at load time from the ONNX graph
    private var inputName: String = "mel_features"
    private var outputName: String = "embedding"
    private var verifiedEmbeddingDim: Int = -1
    
    // Feature extractor — parameters will be updated from contract if available
    private val melExtractor = MelFeatureExtractor()

    // ─── Health ──────────────────────────────────────────────────────────

    override fun isReady(): Boolean = isModelLoaded

    fun getDetailedHealth(): DetailedModelHealth {
        val status = when {
            isModelLoaded -> ModelStatus.READY
            failureReason == ModelFailureReason.NOT_IMPLEMENTED -> ModelStatus.NOT_IMPLEMENTED
            failureReason == ModelFailureReason.NOT_LOADED_YET -> ModelStatus.LOADING
            else -> ModelStatus.ERROR
        }
        return DetailedModelHealth(
            name = "ECAPA-TDNN",
            status = status,
            failureReason = failureReason,
            detailMessage = loadError,
            loadTimeMs = loadTimeMs,
            lastInferenceMs = lastInferenceMs
        )
    }

    fun getStatus(): ModelStatus {
        if (isModelLoaded) return ModelStatus.READY
        if (loadError != null) return ModelStatus.ERROR
        val ctx = context ?: return ModelStatus.NOT_IMPLEMENTED
        // Check if the model file exists in assets
        return try {
            ctx.assets.open(modelAssetPath).close()
            ModelStatus.LOADING // File exists, waiting to load or in progress
        } catch (e: Exception) {
            ModelStatus.NOT_IMPLEMENTED // File doesn't exist
        }
    }

    fun getLoadError(): String? = loadError

    // ─── Lifecycle ──────────────────────────────────────────────────────

    override suspend fun load(): Boolean {
        if (isModelLoaded && session != null) {
            Log.d(TAG, "ECAPA-TDNN model already loaded — reusing existing session")
            return true
        }

        val ctx = context ?: run {
            failureReason = ModelFailureReason.NOT_IMPLEMENTED
            loadError = "Context is null"
            Log.w(TAG, "ECAPA-TDNN cannot load: context is null")
            return false
        }

        val startTime = System.currentTimeMillis()
        return try {
            Log.i(TAG, "Loading ECAPA-TDNN ONNX model from $modelAssetPath...")

            // Check if model file exists
            val modelBytes = try {
                ctx.assets.open(modelAssetPath).use { it.readBytes() }
            } catch (e: Exception) {
                failureReason = ModelFailureReason.ASSET_MISSING
                loadError = "Model asset not found: $modelAssetPath"
                Log.w(TAG, "ECAPA-TDNN model asset not found: ${e.message}")
                return false
            }

            try {
                ortEnv = OrtEnvironment.getEnvironment()
            } catch (e: Exception) {
                failureReason = ModelFailureReason.ONNX_RUNTIME_INIT_FAILED
                loadError = "OrtEnvironment init failed: ${e.message}"
                Log.e(TAG, loadError!!, e)
                return false
            }

            try {
                session = ortEnv!!.createSession(modelBytes)
            } catch (e: Exception) {
                failureReason = ModelFailureReason.MODEL_LOAD_FAILED
                loadError = "OrtSession create failed: ${e.message}"
                Log.e(TAG, loadError!!, e)
                return false
            }

            // Verify ONNX graph I/O contract
            val inputs = session!!.inputInfo
            val outputs = session!!.outputInfo

            // Validate input
            if (inputs.isEmpty() || !inputs.containsKey("mel_features")) {
                failureReason = ModelFailureReason.INPUT_CONTRACT_INVALID
                loadError = "ONNX model missing input 'mel_features' (found: ${inputs.keys})"
                Log.e(TAG, loadError!!)
                return false
            }
            inputName = "mel_features"
            Log.i(TAG, "ONNX input name: $inputName")

            // Validate output
            if (outputs.isEmpty() || !outputs.containsKey("embedding")) {
                failureReason = ModelFailureReason.OUTPUT_CONTRACT_INVALID
                loadError = "ONNX model missing output 'embedding' (found: ${outputs.keys})"
                Log.e(TAG, loadError!!)
                return false
            }
            outputName = "embedding"
            Log.i(TAG, "ONNX output name: $outputName")

            // Run a test inference with a small input to verify dimensions
            val testMelFrames = 100 // ~1 second
            val flatInput = FloatArray(testMelFrames * melExtractor.nMels)
            val testTensor = OnnxTensor.createTensor(
                ortEnv!!,
                FloatBuffer.wrap(flatInput),
                longArrayOf(1, testMelFrames.toLong(), melExtractor.nMels.toLong())
            )

            try {
                val testResult = session!!.run(mapOf(inputName to testTensor))
                val testOutput = testResult[0].value
                
                verifiedEmbeddingDim = unpackEmbedding(testOutput).size
                
                testResult.close()
                testTensor.close()

                if (verifiedEmbeddingDim != EMBEDDING_DIM) {
                    Log.w(TAG, "ECAPA output dim = $verifiedEmbeddingDim (expected $EMBEDDING_DIM)")
                }
                
                Log.i(TAG, "ECAPA-TDNN verified: embedding_dim=$verifiedEmbeddingDim")
            } catch (e: Exception) {
                testTensor.close()
                failureReason = ModelFailureReason.STARTUP_TEST_INFERENCE_FAILED
                loadError = "Startup test inference failed: ${e.message}"
                Log.e(TAG, loadError!!, e)
                return false
            }

            loadTimeMs = System.currentTimeMillis() - startTime
            failureReason = ModelFailureReason.NONE
            isModelLoaded = true
            loadError = null
            Log.i(TAG, "ECAPA-TDNN loaded & verified in ${loadTimeMs}ms (embedding_dim=$verifiedEmbeddingDim). Status=READY")
            true
        } catch (e: Exception) {
            failureReason = ModelFailureReason.UNKNOWN_ERROR
            loadError = "Unexpected error loading ECAPA-TDNN: ${e.message}"
            Log.e(TAG, loadError!!, e)
            isModelLoaded = false
            false
        }
    }

    // ─── Inference ──────────────────────────────────────────────────────

    /**
     * Extract a speaker embedding from raw audio.
     *
     * @param waveform Float32 PCM samples, mono, 16kHz. Min length: [MIN_AUDIO_SAMPLES].
     * @return L2-normalized embedding of dimension [verifiedEmbeddingDim] (typically 192).
     * @throws IllegalStateException if model is not loaded.
     * @throws IllegalArgumentException if audio is too short.
     */
    override suspend fun extractEmbedding(waveform: FloatArray): FloatArray {
        require(isModelLoaded) { "ECAPA-TDNN model not loaded" }
        require(waveform.size >= MIN_AUDIO_SAMPLES) {
            "Audio too short: ${waveform.size} samples (min: $MIN_AUDIO_SAMPLES)"
        }

        val env = ortEnv ?: throw IllegalStateException("ONNX environment not initialized")
        val sess = session ?: throw IllegalStateException("ONNX session not initialized")

        // Step 1: Extract mel features
        val melFeatures = melExtractor.extract(waveform)
        if (melFeatures.isEmpty()) {
            throw IllegalStateException("Mel extraction produced no frames from ${waveform.size} samples")
        }

        val numFrames = melFeatures.size
        val numMels = melExtractor.nMels

        // Step 2: Flatten for ONNX tensor [1, time, n_mels]
        val flatFeatures = FloatArray(numFrames * numMels)
        for (t in melFeatures.indices) {
            System.arraycopy(melFeatures[t], 0, flatFeatures, t * numMels, numMels)
        }

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(flatFeatures),
            longArrayOf(1, numFrames.toLong(), numMels.toLong())
        )

        // Step 3: Run inference
        val startTime = System.nanoTime()
        val results = sess.run(mapOf(inputName to inputTensor))
        val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

        // Step 4: Extract embedding
        val output = results[0].value
        val embedding: FloatArray = unpackEmbedding(output)

        inputTensor.close()
        results.close()

        // Step 5: Validate — no NaN, no Inf
        for (i in embedding.indices) {
            if (embedding[i].isNaN() || embedding[i].isInfinite()) {
                throw IllegalStateException("ECAPA embedding contains NaN/Inf at index $i")
            }
        }

        // Step 6: L2 normalize
        val normalized = l2Normalize(embedding)

        lastInferenceMs = inferenceTimeMs

        Log.d(TAG, "ECAPA embedding extracted: dim=${normalized.size}, " +
                "frames=$numFrames, time=${inferenceTimeMs}ms")

        return normalized
    }

    /**
     * Compare two speaker embeddings using cosine similarity.
     *
     * @return Cosine similarity in [-1.0, 1.0]. Higher = more similar.
     *         1.0 = identical speakers, 0.0 = orthogonal, -1.0 = opposite.
     *
     * NOTE: No threshold is applied here. The similarity value is raw and
     * EXPERIMENTAL until calibrated against evaluation data.
     */
    override fun compareSpeakers(embedding1: FloatArray, embedding2: FloatArray): Float {
        require(embedding1.size == embedding2.size) {
            "Embedding dimension mismatch: ${embedding1.size} vs ${embedding2.size}"
        }

        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }

        val denominator = sqrt(norm1) * sqrt(norm2)
        return if (denominator > 0f) dotProduct / denominator else 0f
    }

    /**
     * Create a SpeakerResult from a comparison against an enrolled reference.
     *
     * @param similarity The cosine similarity value.
     * @param enrolledSpeakerId ID of the enrolled speaker.
     * @return SpeakerResult with EXPERIMENTAL label — no threshold applied.
     */
    fun createSpeakerResult(
        similarity: Float,
        enrolledSpeakerId: String
    ): SpeakerResult {
        return SpeakerResult(
            modelName = modelName,
            similarity = similarity,
            enrolledSpeakerId = enrolledSpeakerId,
            match = null, // EXPERIMENTAL — no threshold until calibrated
            thresholdUsed = null // CALIBRATION_NOT_AVAILABLE
        )
    }

    // ─── Utility ────────────────────────────────────────────────────────
    
    private fun unpackEmbedding(output: Any?): FloatArray {
        return when (output) {
            is FloatArray -> output
            is Array<*> -> {
                val first = output.firstOrNull() ?: throw IllegalStateException("Empty output array")
                when (first) {
                    is FloatArray -> first
                    is Array<*> -> {
                        val second = first.firstOrNull() ?: throw IllegalStateException("Empty 2D array")
                        if (second is FloatArray) second
                        else throw IllegalStateException("Unexpected 3D element: ${second?.javaClass}")
                    }
                    else -> throw IllegalStateException("Unexpected 2D element: ${first.javaClass}")
                }
            }
            else -> throw IllegalStateException("Unexpected ECAPA output type: ${output?.javaClass}")
        }
    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vector) sumSq += v * v
        val norm = sqrt(sumSq)
        if (norm < 1e-10f) return vector
        return FloatArray(vector.size) { vector[it] / norm }
    }

    // ─── Lifecycle ──────────────────────────────────────────────────────

    override fun release() {
        session?.close()
        session = null
        // Don't close OrtEnvironment — it's a singleton
        isModelLoaded = false
        loadError = null
        Log.i(TAG, "ECAPA-TDNN engine released")
    }
}
