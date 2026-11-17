/**
 * PostCallAnalysisCoordinator — Full post-call forensic analysis pipeline.
 *
 * POST-CALL ANALYSIS of user-selected OEM call recordings.
 * NOT real-time cellular audio interception.
 *
 * Pipeline:
 *   RecordingDecoder (raw PCM chunks, primitive buffering)
 *   → AudioNormalizer (own instance — channel conversion + resample to 16kHz mono)
 *   → AudioBuffer (own instance — 64,600-sample windows, 50% overlap)
 *   → SileroVadEngine (dedicated isolated instance — speech detection)
 *   → AasistLEngine (direct scoring on exact 64,600-sample windows)
 *   → EcapaTdnnEngine (speaker embeddings for experimental clustering)
 *   → Speaker separation (2-means with margin-based mixed window detection)
 *   → Conservative outcome evaluation (strictly abstains on uncalibrated models)
 *
 * Session isolation:
 * - Dedicated VAD recurrent state (never shares hidden state with live dialer)
 * - Dedicated audio buffer and normalizer
 * - Concurrency guard prevents duplicate overlapping analysis
 * - Absolute sample-index timestamping eliminates timeline drift
 * - Privacy: decoded PCM discarded after each chunk; user recording never altered
 */
package com.voiceshield.ml

import android.content.Context
import android.net.Uri
import android.util.Log
import com.voiceshield.VoiceShieldApp
import com.voiceshield.audio.AudioBuffer
import com.voiceshield.audio.AudioNormalizer
import com.voiceshield.audio.RecordingDecoder
import com.voiceshield.audio.RecordingImporter
import com.voiceshield.security.TemporalEvidenceEngine
import com.voiceshield.shared.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

class PostCallAnalysisCoordinator(
    private val context: Context
) {
    companion object {
        private const val TAG = "PostCallAnalysis"
        /** AASIST-L input sample count requirement (~4.0375 seconds at 16kHz). */
        const val WINDOW_SIZE = 64600
        const val SAMPLE_RATE = 16000
    }

    private val _session = MutableStateFlow(PostCallAnalysisSession())
    val session: StateFlow<PostCallAnalysisSession> = _session.asStateFlow()

    private val decoder = RecordingDecoder(context)

    // Session-isolated instances — never shared with live pipeline
    private val normalizer = AudioNormalizer()
    private val audioBuffer = AudioBuffer(windowSize = WINDOW_SIZE, overlapRatio = 0.5f)
    private val evidenceEngine = TemporalEvidenceEngine(
        minWindowsForAssessment = 3,
        maxWindowHistory = 50
    )

    // Dedicated VAD instance for post-call analysis — never shares recurrent state with live dialer
    private val vadEngine = SileroVadEngine(context)

    // Shared model engines for stateless inference (thread-safe)
    private val engine: OnDeviceDetectionEngine get() = VoiceShieldApp.onDeviceEngine

    /**
     * Pre-validate recording metadata and update session state immediately upon file selection.
     */
    fun prepareRecording(uri: Uri) {
        reset()
        val validation = decoder.validate(uri)
        val importResult = RecordingImporter.queryRecordingMetadata(context, uri)
        updateSession { it.copy(
            recordingUri = uri.toString(),
            recordingDisplayName = importResult?.displayName ?: validation.displayName,
            recordingValidation = validation,
            outcome = PostCallAnalysisOutcome.NOT_STARTED,
            progress = 0f
        )}
    }

    /**
     * Run full post-call analysis on a user-selected recording.
     * Must be called from a background coroutine — never Main/Telecom thread.
     */
    suspend fun analyze(uri: Uri): PostCallAnalysisSession = withContext(Dispatchers.Default) {
        // Concurrency guard: reject duplicate runs on same coordinator
        if (_session.value.outcome == PostCallAnalysisOutcome.IN_PROGRESS) {
            Log.w(TAG, "Analysis already in progress for session ${_session.value.id} — ignoring duplicate request")
            return@withContext _session.value
        }

        val sessionId = java.util.UUID.randomUUID().toString()
        val startTime = System.currentTimeMillis()

        updateSession { it.copy(
            id = sessionId,
            recordingUri = uri.toString(),
            outcome = PostCallAnalysisOutcome.IN_PROGRESS,
            startedAt = startTime,
            errorMessage = null,
            progress = 0.02f
        )}

        try {
            // Phase 1: Validate recording
            Log.i(TAG, "Phase 1: Validating recording")
            val validation = decoder.validate(uri)
            updateSession { it.copy(
                recordingValidation = validation,
                recordingDisplayName = validation.displayName,
                progress = 0.05f
            )}

            if (validation.status != RecordingStatus.VERIFIED) {
                return@withContext finishWithOutcome(
                    PostCallAnalysisOutcome.NO_USABLE_AUDIO,
                    "Recording validation failed: ${validation.errorMessage ?: "unsupported or corrupt audio"}",
                    validation.errorMessage
                )
            }

            // Phase 2: Initialize isolated DSP & VAD state
            Log.i(TAG, "Phase 2: Streaming decode + analysis")
            vadEngine.load()
            vadEngine.reset() // Guarantee fresh recurrent state with ZERO leakage
            normalizer.resetMetrics()
            audioBuffer.reset()
            evidenceEngine.reset()

            val allWindowEvidence = mutableListOf<PostCallWindowEvidence>()
            val speechSegments = mutableListOf<SpeechSegment>()
            val speakerEmbeddings = mutableListOf<Pair<Int, FloatArray>>() // windowIndex -> embedding
            var windowIndex = 0
            var totalDecodedSamples = 0L
            var speechWindowCount = 0

            // Quality metrics accumulators
            var totalRms = 0.0
            var totalPeak = 0f
            var sampleCount = 0L
            var nanCount = 0
            var infCount = 0
            var nearZeroCount = 0L
            var clipCount = 0L

            // Speech segment tracking
            var currentSpeechStart = -1L
            var currentSpeechVadSum = 0f
            var currentSpeechVadCount = 0

            val recordingDurationMs = validation.durationMs

            val decodedTotal = decoder.decodeStreaming(uri) { chunk ->
                // Normalize: channel conversion + resample to 16kHz mono
                val frame = AudioFrame(
                    samples = chunk.samples,
                    sampleRate = chunk.sampleRate,
                    channelCount = chunk.channelCount,
                    sourceType = AudioSourceType.PLATFORM_SUPPORTED,
                    timestamp = chunk.timestampMs
                )

                val normalized = normalizer.normalize(frame)

                // Quality metrics calculation
                for (s in normalized.samples) {
                    sampleCount++
                    if (s.isNaN()) nanCount++
                    if (s.isInfinite()) infCount++
                    if (kotlin.math.abs(s) < 1e-6f) nearZeroCount++
                    if (kotlin.math.abs(s) > 0.99f) clipCount++
                    totalRms += (s * s).toDouble()
                    if (kotlin.math.abs(s) > totalPeak) totalPeak = kotlin.math.abs(s)
                }

                // Feed into analysis buffer (produces exact 64,600-sample windows)
                val windows = audioBuffer.addSamples(normalized.samples)

                for (window in windows) {
                    // Absolute sample position tracking guarantees drift-free timestamps
                    val sampleStart = windowIndex.toLong() * audioBuffer.hopSize
                    val sampleEnd = sampleStart + audioBuffer.windowSize
                    val windowStartMs = (sampleStart * 1000L) / SAMPLE_RATE
                    val windowEndMs = (sampleEnd * 1000L) / SAMPLE_RATE

                    // Step 1: Run isolated Silero VAD
                    val vadResult = vadEngine.process(window, SAMPLE_RATE)

                    val evidence = if (vadResult.isSpeech) {
                        speechWindowCount++

                        // Track speech segments
                        if (currentSpeechStart < 0) {
                            currentSpeechStart = windowStartMs
                        }
                        currentSpeechVadSum += vadResult.speechProbability
                        currentSpeechVadCount++

                        // Step 2: Run AASIST-L on exactly 64,600 samples
                        val aasistResult = try {
                            if (engine.getAasistEngine().isReady()) {
                                engine.getAasistEngine().score(window)
                            } else null
                        } catch (e: Exception) {
                            Log.e(TAG, "AASIST-L inference error on window $windowIndex: ${e.message}")
                            null
                        }

                        // Step 3: Extract ECAPA-TDNN speaker embedding for clustering
                        val embedding = try {
                            if (engine.getEcapaEngine().isReady() && window.size >= EcapaTdnnEngine.MIN_AUDIO_SAMPLES) {
                                engine.getEcapaEngine().extractEmbedding(window)
                            } else null
                        } catch (e: Exception) {
                            Log.d(TAG, "ECAPA embedding extraction failed for window $windowIndex: ${e.message}")
                            null
                        }

                        if (embedding != null) {
                            speakerEmbeddings.add(windowIndex to embedding)
                        }

                        PostCallWindowEvidence(
                            windowIndex = windowIndex,
                            startMs = windowStartMs,
                            endMs = windowEndMs,
                            sampleStart = sampleStart,
                            sampleEnd = sampleEnd,
                            vadProb = vadResult.speechProbability,
                            aasistScore = aasistResult?.spoofScore,
                            aasistStatus = if (aasistResult != null) "OK" else "SKIPPED",
                            ecapaSimilarity = null, // Populated in Phase 5 during speaker clustering
                            ecapaStatus = if (embedding != null) "OK" else "SKIPPED",
                            qualityStatus = "GOOD"
                        )
                    } else {
                        // Non-speech window
                        if (currentSpeechStart >= 0 && currentSpeechVadCount > 0) {
                            speechSegments.add(SpeechSegment(
                                startMs = currentSpeechStart,
                                endMs = windowStartMs,
                                durationMs = windowStartMs - currentSpeechStart,
                                avgVadProb = currentSpeechVadSum / currentSpeechVadCount
                            ))
                            currentSpeechStart = -1L
                            currentSpeechVadSum = 0f
                            currentSpeechVadCount = 0
                        }

                        PostCallWindowEvidence(
                            windowIndex = windowIndex,
                            startMs = windowStartMs,
                            endMs = windowEndMs,
                            sampleStart = sampleStart,
                            sampleEnd = sampleEnd,
                            vadProb = vadResult.speechProbability,
                            aasistStatus = "NO_SPEECH",
                            ecapaStatus = "SKIPPED",
                            qualityStatus = "GOOD"
                        )
                    }

                    allWindowEvidence.add(evidence)
                    windowIndex++

                    // Update progress (bounded between 0.05 and 0.85 during decode)
                    val progress = if (recordingDurationMs > 0) {
                        (windowEndMs.toFloat() / recordingDurationMs).coerceIn(0.05f, 0.85f)
                    } else 0.5f
                    updateSession { it.copy(progress = progress) }
                }

                true // continue decoding
            }

            // Close final speech segment if open
            if (currentSpeechStart >= 0 && currentSpeechVadCount > 0) {
                val endMs = ((windowIndex.toLong() * audioBuffer.hopSize + audioBuffer.windowSize) * 1000L) / SAMPLE_RATE
                speechSegments.add(SpeechSegment(
                    startMs = currentSpeechStart,
                    endMs = endMs,
                    durationMs = endMs - currentSpeechStart,
                    avgVadProb = currentSpeechVadSum / currentSpeechVadCount
                ))
            }

            totalDecodedSamples = if (decodedTotal > 0) decodedTotal else sampleCount

            // Phase 3: Compute quality metrics (informational)
            val rmsValue = if (sampleCount > 0) kotlin.math.sqrt(totalRms / sampleCount).toFloat() else 0f
            val silencePercent = if (sampleCount > 0) (nearZeroCount.toFloat() / sampleCount) * 100f else 100f
            val clippingPercent = if (sampleCount > 0) (clipCount.toFloat() / sampleCount) * 100f else 0f

            val qualityMetrics = RecordingQualityMetrics(
                durationMs = recordingDurationMs,
                rms = rmsValue,
                peak = totalPeak,
                clippingPercent = clippingPercent,
                silencePercent = silencePercent,
                nanCount = nanCount,
                infCount = infCount,
                sampleCount = sampleCount
            )

            // Phase 4: Speech availability assessment
            val totalSpeechMs = speechSegments.sumOf { it.durationMs }
            val speechAvailability = SpeechAvailability(
                hasSpeech = speechSegments.isNotEmpty(),
                totalSpeechMs = totalSpeechMs,
                speechSegmentCount = speechSegments.size,
                speechNonSpeechRatio = if (recordingDurationMs > 0) {
                    totalSpeechMs.toFloat() / recordingDurationMs
                } else 0f,
                sufficientForAnalysis = speechWindowCount >= 3 && totalSpeechMs >= 3000L,
                insufficiencyReason = when {
                    speechSegments.isEmpty() -> "No speech detected in recording"
                    totalSpeechMs < 3000L -> "Total speech duration (${totalSpeechMs}ms) is below 3.0s minimum threshold"
                    speechWindowCount < 3 -> "Only $speechWindowCount speech windows (minimum 3 required)"
                    else -> null
                }
            )

            // Phase 5: Speaker separation (experimental, uncalibrated)
            Log.i(TAG, "Phase 5: Experimental speaker clustering")
            val speakerSepStatus = performSpeakerClustering(speakerEmbeddings, speechSegments, allWindowEvidence)

            // Phase 6: Determine outcome conservatively
            Log.i(TAG, "Phase 6: Determining conservative outcome")
            val outcome = determineOutcome(
                speechAvailability = speechAvailability,
                windowEvidence = allWindowEvidence,
                speakerSeparation = speakerSepStatus,
                speechWindowCount = speechWindowCount
            )

            val summaryText = generateSummary(outcome, speechAvailability, speakerSepStatus, allWindowEvidence)

            updateSession { it.copy(
                recordingValidation = validation.copy(decodedSampleCount = totalDecodedSamples),
                qualityMetrics = qualityMetrics,
                speechAvailability = speechAvailability,
                speechSegments = speechSegments,
                windowEvidence = allWindowEvidence,
                speakerSeparationStatus = speakerSepStatus,
                outcome = outcome,
                summaryText = summaryText,
                completedAt = System.currentTimeMillis(),
                progress = 1f
            )}

            Log.i(TAG, "Analysis complete: $outcome, ${allWindowEvidence.size} windows, " +
                    "$speechWindowCount speech, ${speechSegments.size} segments")

            return@withContext _session.value

        } catch (e: Exception) {
            Log.e(TAG, "Analysis failed: ${e.message}", e)
            return@withContext finishWithOutcome(
                PostCallAnalysisOutcome.ANALYSIS_ERROR,
                "Analysis error: ${e.message}",
                e.message
            )
        }
    }

    /**
     * Experimental speaker clustering using ECAPA embeddings.
     * Evaluates 2-means clustering with margin-based mixed window detection.
     */
    private fun performSpeakerClustering(
        embeddings: List<Pair<Int, FloatArray>>,
        speechSegments: MutableList<SpeechSegment>,
        windowEvidence: MutableList<PostCallWindowEvidence>
    ): SpeakerSeparationStatus {
        if (embeddings.size < 4) {
            Log.d(TAG, "Not enough embeddings for clustering: ${embeddings.size}")
            for (i in windowEvidence.indices) {
                val we = windowEvidence[i]
                if (we.aasistStatus != "NO_SPEECH") {
                    windowEvidence[i] = we.copy(isMixedSpeakerWindow = true, speakerLabel = null)
                }
            }
            return SpeakerSeparationStatus.UNCERTAIN
        }

        try {
            var maxDist = -1f
            var centroid1Idx = 0
            var centroid2Idx = 1

            for (i in embeddings.indices) {
                for (j in i + 1 until embeddings.size) {
                    val dist = 1f - cosineSimilarity(embeddings[i].second, embeddings[j].second)
                    if (dist > maxDist) {
                        maxDist = dist
                        centroid1Idx = i
                        centroid2Idx = j
                    }
                }
            }

            // If max distance between any two embeddings is very small, single speaker detected
            if (maxDist < 0.12f) {
                for (i in windowEvidence.indices) {
                    val we = windowEvidence[i]
                    if (we.aasistStatus != "NO_SPEECH") {
                        windowEvidence[i] = we.copy(speakerLabel = "SPEAKER_A", isMixedSpeakerWindow = false)
                    }
                }
                for (i in speechSegments.indices) {
                    speechSegments[i] = speechSegments[i].copy(speakerLabel = "SPEAKER_A")
                }
                return SpeakerSeparationStatus.SINGLE_SPEAKER
            }

            var centroid1 = embeddings[centroid1Idx].second.clone()
            var centroid2 = embeddings[centroid2Idx].second.clone()

            // 5 iterations of K-means
            for (iter in 0 until 5) {
                val clA = mutableListOf<FloatArray>()
                val clB = mutableListOf<FloatArray>()
                for ((_, emb) in embeddings) {
                    val sA = cosineSimilarity(emb, centroid1)
                    val sB = cosineSimilarity(emb, centroid2)
                    if (sA >= sB) clA.add(emb) else clB.add(emb)
                }
                if (clA.isNotEmpty()) centroid1 = averageEmbedding(clA)
                if (clB.isNotEmpty()) centroid2 = averageEmbedding(clB)
            }

            val interClusterSim = cosineSimilarity(centroid1, centroid2)

            // If centroids are almost identical, separation failed
            if (interClusterSim > 0.85f) {
                for (i in windowEvidence.indices) {
                    val we = windowEvidence[i]
                    if (we.aasistStatus != "NO_SPEECH") {
                        windowEvidence[i] = we.copy(isMixedSpeakerWindow = true, speakerLabel = null)
                    }
                }
                return SpeakerSeparationStatus.UNCERTAIN
            }

            // Assign windows with margin-based ambiguity detection
            val clusterASet = mutableSetOf<Int>()
            val clusterBSet = mutableSetOf<Int>()
            val mixedSet = mutableSetOf<Int>()

            for ((idx, emb) in embeddings) {
                val simA = cosineSimilarity(emb, centroid1)
                val simB = cosineSimilarity(emb, centroid2)
                // If similarity margin is narrow, window contains mixed speech or ambiguous features
                if (kotlin.math.abs(simA - simB) < 0.08f) {
                    mixedSet.add(idx)
                } else if (simA > simB) {
                    clusterASet.add(idx)
                } else {
                    clusterBSet.add(idx)
                }
            }

            for (i in windowEvidence.indices) {
                val we = windowEvidence[i]
                if (we.aasistStatus == "NO_SPEECH") continue
                windowEvidence[i] = when {
                    we.windowIndex in mixedSet -> we.copy(isMixedSpeakerWindow = true, speakerLabel = null)
                    we.windowIndex in clusterASet -> we.copy(speakerLabel = "SPEAKER_A", isMixedSpeakerWindow = false)
                    we.windowIndex in clusterBSet -> we.copy(speakerLabel = "SPEAKER_B", isMixedSpeakerWindow = false)
                    else -> we.copy(isMixedSpeakerWindow = true, speakerLabel = null)
                }
            }

            // Label speech segments based on dominant window label
            for (i in speechSegments.indices) {
                val seg = speechSegments[i]
                val segWindows = windowEvidence.filter {
                    it.startMs < seg.endMs && it.endMs > seg.startMs && it.speakerLabel != null
                }
                val cntA = segWindows.count { it.speakerLabel == "SPEAKER_A" }
                val cntB = segWindows.count { it.speakerLabel == "SPEAKER_B" }
                val label = when {
                    cntA > cntB -> "SPEAKER_A"
                    cntB > cntA -> "SPEAKER_B"
                    else -> null
                }
                speechSegments[i] = seg.copy(speakerLabel = label)
            }

            val bothNonEmpty = clusterASet.isNotEmpty() && clusterBSet.isNotEmpty()
            val reasonableSize = clusterASet.size >= 2 && clusterBSet.size >= 2

            return when {
                !bothNonEmpty -> SpeakerSeparationStatus.SINGLE_SPEAKER
                !reasonableSize -> SpeakerSeparationStatus.UNCERTAIN
                interClusterSim < 0.5f && reasonableSize -> SpeakerSeparationStatus.SEPARATED
                else -> SpeakerSeparationStatus.PROVISIONAL
            }
        } catch (e: Exception) {
            Log.e(TAG, "Speaker clustering failed: ${e.message}", e)
            return SpeakerSeparationStatus.FAILED
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB)
        return if (denom > 0f) dot / denom else 0f
    }

    private fun averageEmbedding(embeddings: List<FloatArray>): FloatArray {
        if (embeddings.isEmpty()) return FloatArray(0)
        val dim = embeddings[0].size
        val avg = FloatArray(dim)
        for (emb in embeddings) {
            for (i in emb.indices) avg[i] += emb[i]
        }
        for (i in avg.indices) avg[i] /= embeddings.size
        return avg
    }

    /**
     * Determine the analysis outcome conservatively.
     * Abstains on uncalibrated models, ambiguous speaker separation, or insufficient speech.
     * Never fabricates confidence scores.
     */
    private fun determineOutcome(
        speechAvailability: SpeechAvailability,
        windowEvidence: List<PostCallWindowEvidence>,
        speakerSeparation: SpeakerSeparationStatus,
        speechWindowCount: Int
    ): PostCallAnalysisOutcome {
        if (!speechAvailability.hasSpeech) return PostCallAnalysisOutcome.NO_SPEECH
        if (!speechAvailability.sufficientForAnalysis) return PostCallAnalysisOutcome.INSUFFICIENT_SPEECH

        val speechWindows = windowEvidence.filter { it.aasistStatus == "OK" }
        if (speechWindows.size < 2) return PostCallAnalysisOutcome.INSUFFICIENT_SPEECH

        // If speaker separation is uncertain in a multi-speaker recording, abstain
        if (speakerSeparation == SpeakerSeparationStatus.UNCERTAIN || speakerSeparation == SpeakerSeparationStatus.FAILED) {
            Log.w(TAG, "Speaker separation uncertain: abstaining from speaker attribution")
            return PostCallAnalysisOutcome.INCONCLUSIVE
        }

        // Isolate single-speaker windows (mixed windows strictly excluded)
        val speakerAWindows = windowEvidence.filter { it.speakerLabel == "SPEAKER_A" && !it.isMixedSpeakerWindow && it.aasistStatus == "OK" }
        val speakerBWindows = windowEvidence.filter { it.speakerLabel == "SPEAKER_B" && !it.isMixedSpeakerWindow && it.aasistStatus == "OK" }

        val testWindows = when {
            speakerSeparation == SpeakerSeparationStatus.SINGLE_SPEAKER -> speechWindows
            speakerAWindows.isNotEmpty() || speakerBWindows.isNotEmpty() -> speakerAWindows + speakerBWindows
            else -> emptyList()
        }

        if (testWindows.isEmpty()) return PostCallAnalysisOutcome.INCONCLUSIVE

        // Uncalibrated gatekeeping:
        // When operating point calibration is not available, we CANNOT make a definitive synthetic speech claim.
        // We report INCONCLUSIVE with explicit uncalibrated reasoning if elevated scores exist,
        // or NORMAL_SPEECH_EVIDENCE if all analyzed windows show low spoof scores (< 0.35 avg, < 0.50 max).
        val maxSpoofScore = testWindows.mapNotNull { it.aasistScore }.maxOrNull() ?: 0f
        val avgSpoofScore = testWindows.mapNotNull { it.aasistScore }.average().toFloat()

        return when {
            avgSpoofScore < 0.35f && maxSpoofScore < 0.50f ->
                PostCallAnalysisOutcome.NORMAL_SPEECH_EVIDENCE
            else ->
                // Uncalibrated: abstain from automated clone verdict
                PostCallAnalysisOutcome.INCONCLUSIVE
        }
    }

    private fun generateSummary(
        outcome: PostCallAnalysisOutcome,
        speechAvailability: SpeechAvailability,
        speakerSep: SpeakerSeparationStatus,
        evidence: List<PostCallWindowEvidence>
    ): String {
        val speechInfo = "${speechAvailability.totalSpeechMs / 1000}s speech in ${speechAvailability.speechSegmentCount} segments"
        val windowInfo = "${evidence.count { it.aasistStatus == "OK" }} speech windows analyzed"
        val sepInfo = "Speaker separation: ${speakerSep.name}"

        val outcomeText = when (outcome) {
            PostCallAnalysisOutcome.NO_USABLE_AUDIO -> "No usable audio in recording."
            PostCallAnalysisOutcome.NO_SPEECH -> "No speech detected in recording."
            PostCallAnalysisOutcome.INSUFFICIENT_SPEECH ->
                "Insufficient speech for reliable analysis (${speechAvailability.insufficiencyReason ?: "under 3.0s"})."
            PostCallAnalysisOutcome.NORMAL_SPEECH_EVIDENCE ->
                "Analyzed speech shows normal acoustic patterns (no elevated synthetic indicators)."
            PostCallAnalysisOutcome.SUSPICIOUS_SYNTHETIC_SPEECH_EVIDENCE ->
                "Suspicious synthetic-speech evidence detected (calibrated operating point exceeded)."
            PostCallAnalysisOutcome.SPEAKER_IDENTITY_MISMATCH_EVIDENCE ->
                "Speaker identity mismatch detected against enrolled reference."
            PostCallAnalysisOutcome.INCONCLUSIVE -> {
                if (speakerSep == SpeakerSeparationStatus.UNCERTAIN || speakerSep == SpeakerSeparationStatus.FAILED) {
                    "Analysis inconclusive: speaker separation uncertain. Cannot isolate individual speakers from mixed audio."
                } else {
                    "Analysis inconclusive: models uncalibrated for cellular audio. Raw evidence available for forensic review; clone verdict abstained."
                }
            }
            PostCallAnalysisOutcome.ANALYSIS_ERROR -> "Analysis error occurred during processing."
            PostCallAnalysisOutcome.NOT_STARTED -> "Analysis not started."
            PostCallAnalysisOutcome.IN_PROGRESS -> "Analysis in progress."
        }

        return "$outcomeText\n$speechInfo. $windowInfo. $sepInfo."
    }

    private fun finishWithOutcome(
        outcome: PostCallAnalysisOutcome,
        summary: String,
        error: String?
    ): PostCallAnalysisSession {
        updateSession { it.copy(
            outcome = outcome,
            summaryText = summary,
            errorMessage = error,
            completedAt = System.currentTimeMillis(),
            progress = 1f
        )}
        return _session.value
    }

    private fun updateSession(transform: (PostCallAnalysisSession) -> PostCallAnalysisSession) {
        _session.value = transform(_session.value)
    }

    /** Reset coordinator for a new analysis. */
    fun reset() {
        vadEngine.reset()
        normalizer.resetMetrics()
        audioBuffer.reset()
        evidenceEngine.reset()
        _session.value = PostCallAnalysisSession()
    }
}
