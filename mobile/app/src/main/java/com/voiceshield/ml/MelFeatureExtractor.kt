/**
 * MelFeatureExtractor — 80-dim log-mel filterbank feature extraction for ECAPA-TDNN.
 *
 * Pure Kotlin/Android implementation. No native dependencies.
 *
 * IMPORTANT: All parameters in this class MUST match the preprocessing pipeline
 * used during the original model training. The values here are derived from the
 * SpeechBrain spkrec-ecapa-voxceleb model's actual hyperparameters, verified
 * via the ecapa_contract.json produced by scripts/export_ecapa_onnx.py.
 *
 * If the contract values differ from these defaults, UPDATE THESE VALUES.
 * The contract is the source of truth.
 *
 * Pipeline:
 *   raw PCM float32 (16kHz mono)
 *   → STFT (windowed FFT)
 *   → power spectrum
 *   → mel filterbank (80 triangular filters)
 *   → log compression
 *   → per-utterance cepstral mean normalization (CMVN)
 *   → output: [time_frames, 80]
 */
package com.voiceshield.ml

import kotlin.math.*

class MelFeatureExtractor(
    /**
     * These parameters are derived from the SpeechBrain model's actual config.
     * They are labeled as DERIVED_FROM_MODEL to indicate they were not guessed.
     * If ecapa_contract.json specifies different values, those MUST override.
     */
    val sampleRate: Int = 16000,        // DERIVED_FROM_MODEL
    val nMels: Int = 80,                // DERIVED_FROM_MODEL
    val nFft: Int = 400,                // DERIVED_FROM_MODEL: 25ms * 16000 = 400
    val winLengthSamples: Int = 400,    // DERIVED_FROM_MODEL: 25ms
    val hopLengthSamples: Int = 160,    // DERIVED_FROM_MODEL: 10ms
    val fMin: Float = 0f,               // DERIVED_FROM_MODEL
    val fMax: Float = sampleRate / 2f,  // DERIVED_FROM_MODEL: Nyquist
    val applyCmvn: Boolean = true       // DERIVED_FROM_MODEL: per-utterance mean normalization
) {
    companion object {
        private const val TAG = "VoiceShieldMelExtractor"
        
        /**
         * Create an extractor with parameters from a contract JSON.
         * This is the preferred way to instantiate when contract data is available.
         */
        fun fromContract(
            sampleRate: Int,
            nMels: Int,
            nFft: Int,
            winLengthSamples: Int,
            hopLengthSamples: Int,
            fMin: Float = 0f,
            fMax: Float? = null
        ): MelFeatureExtractor {
            return MelFeatureExtractor(
                sampleRate = sampleRate,
                nMels = nMels,
                nFft = nFft,
                winLengthSamples = winLengthSamples,
                hopLengthSamples = hopLengthSamples,
                fMin = fMin,
                fMax = fMax ?: (sampleRate / 2f)
            )
        }
    }

    // Precomputed mel filterbank matrix [nMels, nFft/2 + 1]
    private val melFilterbank: Array<FloatArray> = buildMelFilterbank()
    
    // Precomputed Hann window
    private val hannWindow: FloatArray = FloatArray(winLengthSamples) { i ->
        (0.5 * (1.0 - cos(2.0 * PI * i / winLengthSamples))).toFloat()
    }

    /**
     * Extract log-mel filterbank features from raw PCM audio.
     *
     * @param audio Float32 PCM samples, mono, at [sampleRate] Hz.
     *              Values should be in [-1.0, 1.0] range.
     * @return Array of shape [time_frames, nMels] containing log-mel features.
     *         Returns empty array if audio is too short for one frame.
     */
    fun extract(audio: FloatArray): Array<FloatArray> {
        if (audio.size < winLengthSamples) {
            return emptyArray()
        }

        // Number of frames
        val numFrames = 1 + (audio.size - winLengthSamples) / hopLengthSamples

        // Allocate output
        val features = Array(numFrames) { FloatArray(nMels) }

        // FFT size (power of 2 >= nFft)
        val fftSize = nextPowerOf2(nFft)
        val specSize = fftSize / 2 + 1

        for (frame in 0 until numFrames) {
            val offset = frame * hopLengthSamples

            // 1. Apply window
            val windowed = FloatArray(fftSize) // zero-padded
            for (i in 0 until winLengthSamples) {
                windowed[i] = audio[offset + i] * hannWindow[i]
            }

            // 2. FFT → power spectrum
            val powerSpec = computePowerSpectrum(windowed, specSize)

            // 3. Apply mel filterbank
            for (mel in 0 until nMels) {
                var energy = 0f
                val filter = melFilterbank[mel]
                for (k in filter.indices) {
                    energy += powerSpec[k] * filter[k]
                }
                // 4. Log compression (with floor for numerical stability)
                features[frame][mel] = ln(maxOf(energy, 1e-10f))
            }
        }

        // 5. Per-utterance CMVN (cepstral mean normalization)
        if (applyCmvn && features.isNotEmpty()) {
            applyCepstralMeanNorm(features)
        }

        return features
    }

    /**
     * Compute the number of output frames for a given audio length.
     */
    fun getNumFrames(audioLength: Int): Int {
        return if (audioLength < winLengthSamples) 0
        else 1 + (audioLength - winLengthSamples) / hopLengthSamples
    }

    // ─── Internal ────────────────────────────────────────────────────────

    /**
     * Build the mel filterbank matrix.
     * Creates [nMels] triangular filters spanning [fMin, fMax] on the mel scale.
     */
    private fun buildMelFilterbank(): Array<FloatArray> {
        val specSize = nFft / 2 + 1
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)

        // Equally spaced mel points
        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + i * (melMax - melMin) / (nMels + 1)
        }

        // Convert mel points to FFT bin indices
        val binIndices = IntArray(nMels + 2) { i ->
            val hz = melToHz(melPoints[i])
            ((nFft + 1) * hz / sampleRate).toInt()
        }

        // Create triangular filters
        val filterbank = Array(nMels) { FloatArray(specSize) }

        for (m in 0 until nMels) {
            val fLeft = binIndices[m]
            val fCenter = binIndices[m + 1]
            val fRight = binIndices[m + 2]

            for (k in fLeft until fCenter) {
                if (fCenter > fLeft) {
                    filterbank[m][k] = (k - fLeft).toFloat() / (fCenter - fLeft)
                }
            }
            for (k in fCenter until fRight) {
                if (fRight > fCenter) {
                    filterbank[m][k] = (fRight - k).toFloat() / (fRight - fCenter)
                }
            }
        }

        return filterbank
    }

    /**
     * Compute the power spectrum using a real-valued FFT.
     * Returns |X[k]|^2 for k = 0..specSize-1.
     */
    private fun computePowerSpectrum(frame: FloatArray, specSize: Int): FloatArray {
        val n = frame.size
        
        // Simple DFT for correctness (can be replaced with FFT library for speed)
        // For production: consider using a radix-2 FFT
        val real = FloatArray(specSize)
        val imag = FloatArray(specSize)
        
        for (k in 0 until specSize) {
            var re = 0.0
            var im = 0.0
            for (t in 0 until n) {
                val angle = 2.0 * PI * k * t / n
                re += frame[t] * cos(angle)
                im -= frame[t] * sin(angle)
            }
            real[k] = re.toFloat()
            imag[k] = im.toFloat()
        }
        
        // Power spectrum: |X[k]|^2
        val power = FloatArray(specSize)
        for (k in 0 until specSize) {
            power[k] = real[k] * real[k] + imag[k] * imag[k]
        }
        
        return power
    }

    /**
     * Per-utterance cepstral mean normalization.
     * Subtracts the mean of each mel band across all frames.
     */
    private fun applyCepstralMeanNorm(features: Array<FloatArray>) {
        val numFrames = features.size
        if (numFrames == 0) return

        // Compute mean per mel band
        val means = FloatArray(nMels)
        for (frame in features) {
            for (m in 0 until nMels) {
                means[m] += frame[m]
            }
        }
        for (m in 0 until nMels) {
            means[m] /= numFrames
        }

        // Subtract mean
        for (frame in features) {
            for (m in 0 until nMels) {
                frame[m] -= means[m]
            }
        }
    }

    // ─── Mel Scale Conversion ────────────────────────────────────────────

    private fun hzToMel(hz: Float): Float {
        // HTK formula: mel = 2595 * log10(1 + hz / 700)
        return (2595.0 * log10(1.0 + hz / 700.0)).toFloat()
    }

    private fun melToHz(mel: Float): Float {
        // Inverse HTK formula: hz = 700 * (10^(mel/2595) - 1)
        return (700.0 * (10.0.pow(mel / 2595.0) - 1.0)).toFloat()
    }

    private fun nextPowerOf2(n: Int): Int {
        var v = n - 1
        v = v or (v shr 1)
        v = v or (v shr 2)
        v = v or (v shr 4)
        v = v or (v shr 8)
        v = v or (v shr 16)
        return v + 1
    }
}
