/**
 * AasistLEngine — Anti-spoofing engine using AASIST-L on-device.
 *
 * Model: AASIST-L (Jung et al., ICASSP 2022)
 * Pretrained on: ASVspoof2019 LA
 * Parameters: ~85,306
 *
 * Input: 64,600 float32 samples (16kHz mono, ~4.04s)
 * Output: [bona_fide_logit, spoof_logit]
 *
 * CRITICAL RULE:
 * Probability calibration is required before labeling as "% CLONED".
 * This engine outputs "ANTI-SPOOF MODEL SCORE" with raw softmax probability.
 */
package com.voiceshield.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.voiceshield.shared.AntiSpoofResult
import com.voiceshield.shared.DetailedModelHealth
import com.voiceshield.shared.ModelFailureReason
import com.voiceshield.shared.ModelStatus
import java.nio.FloatBuffer

class AasistLEngine(
    private val context: Context? = null,
    private val modelAssetPath: String = "models/aasist_l.onnx"
) : AntiSpoofEngine {

    companion object {
        private const val TAG = "VoiceShieldAasistL"
        const val REQUIRED_SAMPLES = 64600
        const val SAMPLE_RATE = 16000
    }

    override val modelName: String = "AASIST-L"
    override val modelRevision: String = "SpeechAntiSpoofingBenchmarks/AASIST-L@c928fa"

    private var ortEnv: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isModelLoaded = false
    private var failureReason: ModelFailureReason = ModelFailureReason.NOT_LOADED_YET
    private var loadError: String? = null
    private var loadTimeMs: Long? = null
    private var lastInferenceMs: Long? = null

    override fun isReady(): Boolean = isModelLoaded

    fun getDetailedHealth(): DetailedModelHealth {
        val status = when {
            isModelLoaded -> ModelStatus.READY
            failureReason == ModelFailureReason.NOT_IMPLEMENTED -> ModelStatus.NOT_IMPLEMENTED
            failureReason == ModelFailureReason.NOT_LOADED_YET -> ModelStatus.LOADING
            else -> ModelStatus.ERROR
        }
        return DetailedModelHealth(
            name = "AASIST-L",
            status = status,
            failureReason = failureReason,
            detailMessage = loadError,
            loadTimeMs = loadTimeMs,
            lastInferenceMs = lastInferenceMs
        )
    }

    override suspend fun load(): Boolean {
        if (isModelLoaded && session != null) {
            Log.d(TAG, "AASIST-L model already loaded — reusing existing session")
            return true
        }

        val ctx = context ?: run {
            failureReason = ModelFailureReason.NOT_IMPLEMENTED
            loadError = "Context is null"
            Log.w(TAG, "AASIST-L cannot load: context is null")
            return false
        }

        val startTime = System.currentTimeMillis()
        return try {
            Log.i(TAG, "Loading AASIST-L ONNX model ($modelRevision) from $modelAssetPath...")

            val modelBytes = try {
                ctx.assets.open(modelAssetPath).use { it.readBytes() }
            } catch (e: Exception) {
                failureReason = ModelFailureReason.ASSET_MISSING
                loadError = "Asset open failed: ${e.message}"
                Log.e(TAG, loadError!!, e)
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

            // Log & verify model metadata
            val inputInfo = session!!.inputInfo
            val outputInfo = session!!.outputInfo
            Log.i(TAG, "AASIST-L loaded. Inputs: ${inputInfo.keys}, Outputs: ${outputInfo.keys}")

            if (!inputInfo.containsKey("audio")) {
                failureReason = ModelFailureReason.INPUT_CONTRACT_INVALID
                loadError = "Missing required input 'audio' (found: ${inputInfo.keys})"
                Log.e(TAG, loadError!!)
                return false
            }
            if (!outputInfo.containsKey("logits")) {
                failureReason = ModelFailureReason.OUTPUT_CONTRACT_INVALID
                loadError = "Missing required output 'logits' (found: ${outputInfo.keys})"
                Log.e(TAG, loadError!!)
                return false
            }

            // Deterministic startup health test inference (zeroed audio buffer)
            try {
                val dummyAudio = FloatArray(REQUIRED_SAMPLES)
                val inTensor = OnnxTensor.createTensor(
                    ortEnv!!,
                    FloatBuffer.wrap(dummyAudio),
                    longArrayOf(1, REQUIRED_SAMPLES.toLong())
                )
                val testRes = session!!.run(mapOf("audio" to inTensor))
                val outputVal = testRes[0].value
                val testLogits = when (outputVal) {
                    is Array<*> -> {
                        @Suppress("UNCHECKED_CAST")
                        (outputVal as Array<FloatArray>)[0]
                    }
                    else -> throw IllegalStateException("Unexpected output type: ${outputVal?.javaClass}")
                }
                inTensor.close()
                testRes.close()

                if (testLogits.size != 2 || !testLogits[0].isFinite() || !testLogits[1].isFinite()) {
                    failureReason = ModelFailureReason.STARTUP_TEST_INFERENCE_FAILED
                    loadError = "Startup inference produced invalid logits: ${testLogits.contentToString()}"
                    return false
                }
            } catch (e: Exception) {
                failureReason = ModelFailureReason.STARTUP_TEST_INFERENCE_FAILED
                loadError = "Startup test inference failed: ${e.message}"
                Log.e(TAG, loadError!!, e)
                return false
            }

            loadTimeMs = System.currentTimeMillis() - startTime
            failureReason = ModelFailureReason.NONE
            loadError = null
            isModelLoaded = true
            Log.i(TAG, "AASIST-L ONNX model loaded & verified in ${loadTimeMs}ms. Status=READY")
            true
        } catch (e: Exception) {
            failureReason = ModelFailureReason.UNKNOWN_ERROR
            loadError = "Unexpected error loading AASIST-L: ${e.message}"
            Log.e(TAG, loadError!!, e)
            isModelLoaded = false
            false
        }
    }

    /**
     * Score a 64,600-sample audio window using real ONNX inference.
     */
    override suspend fun score(waveform: FloatArray): AntiSpoofResult {
        require(waveform.size == REQUIRED_SAMPLES) {
            "AASIST-L requires exactly $REQUIRED_SAMPLES samples, got ${waveform.size}"
        }

        val env = ortEnv ?: throw IllegalStateException("ONNX environment not initialized")
        val sess = session ?: throw IllegalStateException("ONNX session not initialized")

        // Create input tensor: shape [1, 64600]
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(waveform),
            longArrayOf(1, REQUIRED_SAMPLES.toLong())
        )

        // Run inference
        val startTime = System.nanoTime()
        val results = sess.run(mapOf("audio" to inputTensor))
        val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

        // Extract logits [1, 2] → [bona_fide_logit, spoof_logit]
        val output = results[0].value
        val logits: FloatArray = when (output) {
            is Array<*> -> {
                @Suppress("UNCHECKED_CAST")
                (output as Array<FloatArray>)[0]
            }
            else -> throw IllegalStateException("Unexpected AASIST-L output type: ${output?.javaClass}")
        }

        val bonaFideLogit = logits[0]
        val spoofLogit = logits[1]

        // Stable softmax
        val maxLogit = maxOf(bonaFideLogit, spoofLogit)
        val expBona = Math.exp((bonaFideLogit - maxLogit).toDouble()).toFloat()
        val expSpoof = Math.exp((spoofLogit - maxLogit).toDouble()).toFloat()
        val sumExp = expBona + expSpoof

        val spoofProb = expSpoof / sumExp
        val bonaProb = expBona / sumExp

        Log.d(TAG, "AASIST-L inference: logits=[${bonaFideLogit}, ${spoofLogit}], " +
                "spoof=${String.format("%.4f", spoofProb)}, " +
                "time=${inferenceTimeMs}ms")

        // Clean up tensor
        inputTensor.close()
        results.close()

        lastInferenceMs = inferenceTimeMs

        return AntiSpoofResult(
            modelName = modelName,
            spoofScore = spoofProb,
            bonafideScore = bonaProb,
            rawLogits = floatArrayOf(bonaFideLogit, spoofLogit),
            label = "ANTI-SPOOF MODEL SCORE"
        )
    }

    override fun release() {
        session?.close()
        session = null
        // Don't close OrtEnvironment — it's a singleton
        isModelLoaded = false
        Log.i(TAG, "AASIST-L engine released")
    }
}
