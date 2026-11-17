/**
 * VoiceShieldDetectionEngine — Common interface for AI detection.
 *
 * Two implementations:
 *   - PythonBackendDetectionEngine (Development Mode: WebSocket → FastAPI)
 *   - OnDeviceDetectionEngine (Privacy Mode: ONNX Runtime, all local)
 *
 * Both produce identical WindowResult / ThreatState types so results
 * are directly comparable on the same audio samples.
 */
package com.voiceshield.ml

import com.voiceshield.shared.*

/**
 * Abstraction over the full detection pipeline.
 * Each implementation handles its own model loading, inference, and teardown.
 */
interface VoiceShieldDetectionEngine {

    /** Initialize engine and load models. Returns current status. */
    suspend fun initialize(): EngineStatus

    /**
     * Analyze one audio window through the full pipeline:
     * VAD → Anti-Spoof → Speaker Verification.
     *
     * Returns null if VAD detects no speech in this window.
     */
    suspend fun analyzeWindow(audio: AudioFrame): WindowResult?

    /** Current health of all models in the pipeline. */
    fun getModelHealth(): Map<String, ModelStatus>

    /** Whether calibration artifact is loaded and verified. */
    fun isCalibrated(): Boolean

    /** The engine mode (DEVELOPMENT vs PRIVACY). */
    fun getMode(): EngineMode

    /** Reset runtime recurrent/streaming state at session boundary. */
    fun reset() {}

    /** Release all resources. */
    fun shutdown()
}

/**
 * Anti-spoof model abstraction.
 *
 * Architecture:
 *   AntiSpoofEngine
 *        ├── AasistLEngine        ← lightweight, on-device first
 *        └── SpectraAasistEngine  ← benchmark comparison via backend
 *
 * Choose based on measured accuracy + latency, not assumptions.
 */
interface AntiSpoofEngine {
    val modelName: String
    val modelRevision: String

    /** Load model weights. */
    suspend fun load(): Boolean

    /**
     * Score a raw waveform.
     * @param waveform Float32 PCM samples, mono, 16kHz.
     * @return AntiSpoofResult with raw scores — NOT calibrated probabilities
     *         unless CalibrationArtifact is loaded.
     */
    suspend fun score(waveform: FloatArray): AntiSpoofResult

    /** Whether the model is loaded and functional. */
    fun isReady(): Boolean

    /** Release resources. */
    fun release()
}

/**
 * Speaker verification engine abstraction.
 */
interface SpeakerVerificationEngine {
    val modelName: String

    suspend fun load(): Boolean

    /** Extract a 192-dim speaker embedding from audio. */
    suspend fun extractEmbedding(waveform: FloatArray): FloatArray

    /** Compare two embeddings and return cosine similarity. */
    fun compareSpeakers(embedding1: FloatArray, embedding2: FloatArray): Float

    fun isReady(): Boolean
    fun release()
}

/**
 * Voice Activity Detection engine abstraction.
 */
interface VadEngine {
    suspend fun load(): Boolean

    /** Process a short audio frame and return speech probability. */
    suspend fun process(samples: FloatArray, sampleRate: Int = 16000): VadResult

    /** Reset internal state (e.g. between calls). */
    fun reset()

    fun isReady(): Boolean
    fun release()
}
