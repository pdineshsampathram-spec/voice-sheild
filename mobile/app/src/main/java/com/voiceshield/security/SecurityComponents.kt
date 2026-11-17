/**
 * Security components — Evidence fusion, risk engine, and threat management.
 *
 * Design principles:
 * 1. "I don't know" (INSUFFICIENT_EVIDENCE) is better than a false "safe"
 * 2. No hard-coded thresholds — use CalibrationArtifact or show raw scores
 * 3. Every alert explains the ACTUAL evidence that triggered it
 * 4. Never auto-disconnect — user remains in control
 */
package com.voiceshield.security

import com.voiceshield.shared.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * TemporalEvidenceEngine — Maintains rolling evidence window.
 *
 * Decisions are NEVER based on a single audio window.
 * Evidence must persist across multiple analyzed speech segments.
 */
class TemporalEvidenceEngine(
    /** Minimum windows before making any threat assessment. */
    private val minWindowsForAssessment: Int = 3,
    /** Maximum evidence windows to keep in memory. */
    private val maxWindowHistory: Int = 20,
    /** Calibration artifact — null if not calibrated. */
    private var calibration: CalibrationArtifact? = null
) {
    private val evidence = mutableListOf<WindowEvidence>()

    /**
     * Add new evidence from an analyzed window.
     */
    fun addEvidence(windowEvidence: WindowEvidence) {
        evidence.add(windowEvidence)
        if (evidence.size > maxWindowHistory) {
            evidence.removeAt(0)
        }
    }

    /**
     * Evaluate the current accumulated evidence.
     *
     * Returns INSUFFICIENT_EVIDENCE if fewer than minWindowsForAssessment
     * windows have been analyzed.
     */
    fun evaluate(audioSourceType: AudioSourceType): ThreatState {
        if (audioSourceType == AudioSourceType.UNAVAILABLE) {
            return ThreatState(
                level = ThreatLevel.AUDIO_UNAVAILABLE,
                reason = "Remote call audio is not available on this audio path. " +
                        "Voice analysis is disabled for this call.",
                windowsAnalyzed = 0,
                suspiciousWindows = 0,
                avgSpoofScore = null,
                avgSpeakerSimilarity = null,
                audioSourceType = audioSourceType,
                modelAvailability = evidence.lastOrNull()?.modelAvailability ?: emptyMap()
            )
        }

        if (evidence.size < minWindowsForAssessment) {
            return ThreatState(
                level = ThreatLevel.INSUFFICIENT_EVIDENCE,
                reason = "Analyzing speech. ${evidence.size}/$minWindowsForAssessment " +
                        "windows collected so far.",
                windowsAnalyzed = evidence.size,
                suspiciousWindows = 0,
                avgSpoofScore = evidence.mapNotNull { it.spoofScore.takeIf { s -> s > 0f } }
                    .average().toFloat().takeIf { !it.isNaN() },
                avgSpeakerSimilarity = null,
                audioSourceType = audioSourceType,
                modelAvailability = evidence.lastOrNull()?.modelAvailability ?: emptyMap()
            )
        }

        // Compute aggregate statistics from recent evidence
        val recentEvidence = evidence.takeLast(maxWindowHistory)
        val avgSpoofScore = recentEvidence.map { it.spoofScore }.average().toFloat()
        val avgSpeakerSim = recentEvidence.mapNotNull { it.speakerSimilarity }
            .average().toFloat().takeIf { !it.isNaN() }

        // Determine threat level based on calibration or raw evidence
        val (level, reason) = assessThreatLevel(
            avgSpoofScore = avgSpoofScore,
            avgSpeakerSimilarity = avgSpeakerSim,
            recentEvidence = recentEvidence
        )

        val suspiciousCount = if (calibration != null) {
            recentEvidence.count { it.spoofScore > calibration!!.operatingPoint.threshold }
        } else {
            // Without calibration, count windows where spoof score > 0.5
            // (softmax midpoint, no claim of calibrated threshold)
            recentEvidence.count { it.spoofScore > 0.5f }
        }

        return ThreatState(
            level = level,
            reason = reason,
            windowsAnalyzed = recentEvidence.size,
            suspiciousWindows = suspiciousCount,
            avgSpoofScore = avgSpoofScore,
            avgSpeakerSimilarity = avgSpeakerSim,
            audioSourceType = audioSourceType,
            modelAvailability = recentEvidence.lastOrNull()?.modelAvailability ?: emptyMap()
        )
    }

    private fun assessThreatLevel(
        avgSpoofScore: Float,
        avgSpeakerSimilarity: Float?,
        recentEvidence: List<WindowEvidence>
    ): Pair<ThreatLevel, String> {
        val cal = calibration

        if (cal == null) {
            // NO CALIBRATION: Report raw multi-factor evidence honestly
            val spoofPercent = String.format("%.1f", avgSpoofScore * 100)
            val hasSpeakerEvidence = avgSpeakerSimilarity != null

            return when {
                // Case 1: High spoof + Speaker mismatch (strongest impersonation indicator)
                avgSpoofScore > 0.6f && hasSpeakerEvidence && avgSpeakerSimilarity!! < 0.5f -> {
                    val simStr = String.format("%.2f", avgSpeakerSimilarity)
                    Pair(
                        ThreatLevel.MONITORING,
                        "CALIBRATION NOT AVAILABLE. EXPERIMENTAL DUAL-FACTOR: Elevated spoof score ($spoofPercent%) AND " +
                                "speaker mismatch (cos=$simStr). Possible voice clone. " +
                                "(Uncalibrated: verify caller independently)"
                    )
                }
                // Case 2: High spoof + No speaker enrollment
                avgSpoofScore > 0.6f && !hasSpeakerEvidence -> {
                    Pair(
                        ThreatLevel.MONITORING,
                        "CALIBRATION NOT AVAILABLE. EXPERIMENTAL SPOOF-ONLY: Elevated anti-spoof score ($spoofPercent%). " +
                                "No speaker profile enrolled for identity verification."
                    )
                }
                // Case 3: Low spoof + Speaker mismatch (different person, not synthetic)
                avgSpoofScore <= 0.5f && hasSpeakerEvidence && avgSpeakerSimilarity!! < 0.5f -> {
                    val simStr = String.format("%.2f", avgSpeakerSimilarity)
                    Pair(
                        ThreatLevel.MONITORING,
                        "CALIBRATION NOT AVAILABLE. EXPERIMENTAL IDENTITY ANOMALY: Voice differs from enrolled reference " +
                                "(cos=$simStr), but anti-spoof score is normal ($spoofPercent%). " +
                                "Likely different human speaker, not clone."
                    )
                }
                // Case 4: Normal / Consistent
                else -> {
                    val speakerInfo = if (hasSpeakerEvidence) {
                        ", speaker similarity: ${String.format("%.2f", avgSpeakerSimilarity)}"
                    } else ""
                    Pair(
                        ThreatLevel.MONITORING,
                        "CALIBRATION NOT AVAILABLE. Raw anti-spoof score: $spoofPercent%$speakerInfo. " +
                                "${recentEvidence.size} windows analyzed."
                    )
                }
            }
        }

        // WITH CALIBRATION: use verified operating point
        val threshold = cal.operatingPoint.threshold
        val suspiciousRatio = recentEvidence.count { it.spoofScore > threshold }
            .toFloat() / recentEvidence.size

        return when {
            suspiciousRatio >= 0.8f -> Pair(
                ThreatLevel.CRITICAL,
                "Persistent anti-spoof evidence detected across " +
                        "${(suspiciousRatio * 100).toInt()}% of analyzed speech segments " +
                        "(${recentEvidence.size} windows). " +
                        "Average model score: ${String.format("%.1f", avgSpoofScore * 100)}%. " +
                        "Recommended: Verify the caller through an independent channel."
            )
            suspiciousRatio >= 0.5f -> Pair(
                ThreatLevel.HIGH_RISK,
                "Elevated anti-spoof indicators in " +
                        "${(suspiciousRatio * 100).toInt()}% of analyzed segments. " +
                        "Average model score: ${String.format("%.1f", avgSpoofScore * 100)}%."
            )
            suspiciousRatio >= 0.2f -> Pair(
                ThreatLevel.ELEVATED,
                "Some suspicious segments detected " +
                        "(${(suspiciousRatio * 100).toInt()}% of windows). " +
                        "Monitoring continues."
            )
            else -> Pair(
                ThreatLevel.NORMAL,
                "No significant synthetic speech indicators detected. " +
                        "${recentEvidence.size} windows analyzed."
            )
        }
    }

    /** Update calibration artifact. */
    fun setCalibration(cal: CalibrationArtifact?) {
        calibration = cal
    }

    /** Reset all evidence (e.g., between calls). */
    fun reset() {
        evidence.clear()
    }

    /** Current evidence count. */
    val windowCount: Int get() = evidence.size
}

/**
 * RiskEngine — Wraps TemporalEvidenceEngine with observable state.
 */
class RiskEngine(
    minWindowsForAssessment: Int = 3,
    calibration: CalibrationArtifact? = null
) {
    private val evidenceEngine = TemporalEvidenceEngine(
        minWindowsForAssessment = minWindowsForAssessment,
        calibration = calibration
    )

    private val _threatState = MutableStateFlow(
        ThreatState(
            level = ThreatLevel.MONITORING,
            reason = "Initializing...",
            windowsAnalyzed = 0,
            suspiciousWindows = 0,
            avgSpoofScore = null,
            avgSpeakerSimilarity = null,
            audioSourceType = AudioSourceType.UNAVAILABLE,
            modelAvailability = emptyMap()
        )
    )
    val threatState: StateFlow<ThreatState> = _threatState.asStateFlow()

    /**
     * Process a new window result from the detection engine.
     */
    fun onWindowResult(result: WindowResult, audioSourceType: AudioSourceType) {
        val modelAvailability = mutableMapOf<String, ModelStatus>()

        if (result.antiSpoof != null) {
            modelAvailability[result.antiSpoof.modelName] = ModelStatus.READY
        }
        if (result.speaker != null) {
            modelAvailability[result.speaker.modelName] = ModelStatus.READY
        }

        val windowEvidence = WindowEvidence(
            windowIndex = result.windowIndex,
            timestamp = result.timestamp,
            spoofScore = result.antiSpoof?.spoofScore ?: 0f,
            speakerSimilarity = result.speaker?.similarity,
            vadConfidence = result.vad.speechProbability,
            modelAvailability = modelAvailability
        )

        evidenceEngine.addEvidence(windowEvidence)
        _threatState.value = evidenceEngine.evaluate(audioSourceType)
    }

    /**
     * Set audio unavailable — immediately updates threat state.
     */
    fun setAudioUnavailable() {
        _threatState.value = evidenceEngine.evaluate(AudioSourceType.UNAVAILABLE)
    }

    /** Update calibration. */
    fun setCalibration(cal: CalibrationArtifact?) {
        evidenceEngine.setCalibration(cal)
    }

    /** Reset for new call. */
    fun reset() {
        evidenceEngine.reset()
        _threatState.value = ThreatState(
            level = ThreatLevel.MONITORING,
            reason = "Call started. Waiting for audio.",
            windowsAnalyzed = 0,
            suspiciousWindows = 0,
            avgSpoofScore = null,
            avgSpeakerSimilarity = null,
            audioSourceType = AudioSourceType.UNAVAILABLE,
            modelAvailability = emptyMap()
        )
    }
}

/**
 * AlertManager — Controls when and how threat alerts are shown.
 *
 * Rules:
 * 1. Never auto-disconnect a call based on one model prediction
 * 2. Alerts require persistent evidence across multiple windows
 * 3. User remains in control at all times
 * 4. Every alert explains the actual evidence
 */
class AlertManager {

    data class Alert(
        val id: String = java.util.UUID.randomUUID().toString(),
        val threatState: ThreatState,
        val timestamp: Long = System.currentTimeMillis(),
        val dismissed: Boolean = false,
        val userAction: UserAction? = null
    )

    enum class UserAction {
        CONTINUE_CAREFULLY,
        VERIFY_IDENTITY,
        END_CALL,
        DISMISSED
    }

    private val _activeAlert = MutableStateFlow<Alert?>(null)
    val activeAlert: StateFlow<Alert?> = _activeAlert.asStateFlow()

    private val alertHistory = mutableListOf<Alert>()

    /**
     * Evaluate whether an alert should be shown based on threat state.
     */
    fun evaluateThreatState(threatState: ThreatState) {
        when (threatState.level) {
            ThreatLevel.CRITICAL, ThreatLevel.HIGH_RISK -> {
                // Only show alert if we don't already have an active one
                if (_activeAlert.value == null || _activeAlert.value?.dismissed == true) {
                    _activeAlert.value = Alert(threatState = threatState)
                }
            }
            else -> {
                // Don't clear alert automatically — let user dismiss it
            }
        }
    }

    /**
     * Record the user's response to an alert.
     */
    fun onUserAction(alertId: String, action: UserAction) {
        _activeAlert.value?.let { alert ->
            if (alert.id == alertId) {
                val updated = alert.copy(
                    dismissed = true,
                    userAction = action
                )
                alertHistory.add(updated)
                _activeAlert.value = null
            }
        }
    }

    /** Reset for new call. */
    fun reset() {
        _activeAlert.value = null
    }
}
