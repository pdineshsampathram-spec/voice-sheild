/**
 * PostCallAnalysisScreen — UI for post-call forensic analysis.
 *
 * POST-CALL ANALYSIS — SOURCE: USER/OEM CALL RECORDING
 * NOT real-time cellular audio interception.
 *
 * Shows: recording import → validation → quality metrics →
 *        speech availability → speaker separation → analysis timeline →
 *        conservative outcome.
 */
package com.voiceshield.ui.postcall

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.voiceshield.shared.*
import com.voiceshield.ui.theme.*

@Composable
fun PostCallAnalysisScreen(
    session: PostCallAnalysisSession,
    onSelectRecording: () -> Unit,
    onStartAnalysis: () -> Unit,
    onBack: () -> Unit,
    oemDiscoveryResult: com.voiceshield.audio.OemDiscoveryResult? = null,
    isDiscoveringOemRecording: Boolean = false,
    onRetryOemDiscovery: (() -> Unit)? = null
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Header
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = DarkOnSurface)
            }
            Spacer(modifier = Modifier.width(8.dp))
            Column {
                Text(
                    "POST-CALL ANALYSIS",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = ShieldCyan,
                    letterSpacing = 1.sp
                )
                Text(
                    "SOURCE: USER / OEM CALL RECORDING",
                    style = MaterialTheme.typography.bodySmall,
                    color = DarkOnSurfaceVariant,
                    fontSize = 10.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Prototype Device Optimization Banner
        Card(
            colors = CardDefaults.cardColors(containerColor = DarkSurfaceVariant.copy(alpha = 0.5f)),
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(Icons.Default.Info, contentDescription = null, tint = ShieldCyan, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = com.voiceshield.audio.OemCallRecordingDiscoverer.OPTIMIZATION_NOTICE,
                    style = MaterialTheme.typography.bodySmall,
                    color = DarkOnSurfaceVariant,
                    fontSize = 11.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Automatic Discovery Status Card
        if (isDiscoveringOemRecording || oemDiscoveryResult != null) {
            OemDiscoveryStatusCard(
                isDiscovering = isDiscoveringOemRecording,
                result = oemDiscoveryResult,
                onRetry = onRetryOemDiscovery
            )
            Spacer(modifier = Modifier.height(12.dp))
        }

        // Section 1: Recording Import (with SAF SELECT RECORDING fallback)
        RecordingImportCard(
            session = session,
            onSelectRecording = onSelectRecording,
            onStartAnalysis = onStartAnalysis
        )

        // Section 2: Recording Validation
        session.recordingValidation?.let { validation ->
            Spacer(modifier = Modifier.height(12.dp))
            RecordingValidationCard(validation)
        }

        // Section 3: Quality Metrics (informational)
        session.qualityMetrics?.let { metrics ->
            Spacer(modifier = Modifier.height(12.dp))
            QualityMetricsCard(metrics)
        }

        // Section 4: Speech Availability
        session.speechAvailability?.let { speech ->
            Spacer(modifier = Modifier.height(12.dp))
            SpeechAvailabilityCard(speech)
        }

        // Section 5: Speaker Separation
        if (session.outcome != PostCallAnalysisOutcome.NOT_STARTED) {
            Spacer(modifier = Modifier.height(12.dp))
            SpeakerSeparationCard(session.speakerSeparationStatus)
        }

        // Section 6: Analysis Timeline
        if (session.windowEvidence.isNotEmpty()) {
            Spacer(modifier = Modifier.height(12.dp))
            AnalysisTimelineCard(session.windowEvidence)
        }

        // Section 7: Final Result (strictly guarded: NEVER show without an actual recording)
        if (session.recordingUri != null &&
            session.outcome != PostCallAnalysisOutcome.NOT_STARTED &&
            session.outcome != PostCallAnalysisOutcome.IN_PROGRESS) {
            Spacer(modifier = Modifier.height(12.dp))
            FinalResultCard(session)
        }

        // Section 8: Progress
        if (session.outcome == PostCallAnalysisOutcome.IN_PROGRESS) {
            Spacer(modifier = Modifier.height(12.dp))
            ProgressCard(session.progress)
        }

        // User Guidance
        Spacer(modifier = Modifier.height(16.dp))
        UserGuidanceCard()

        Spacer(modifier = Modifier.height(32.dp))
    }
}

@Composable
private fun RecordingImportCard(
    session: PostCallAnalysisSession,
    onSelectRecording: () -> Unit,
    onStartAnalysis: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "CALL RECORDING",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = ShieldYellow,
                letterSpacing = 1.sp
            )
            Spacer(modifier = Modifier.height(8.dp))

            if (session.recordingUri != null) {
                Text(
                    session.recordingDisplayName ?: "Recording selected",
                    style = MaterialTheme.typography.bodyMedium,
                    color = DarkOnSurface,
                    fontWeight = FontWeight.SemiBold
                )
                session.recordingValidation?.let { v ->
                    if (v.durationMs > 0) {
                        Text(
                            "${v.codec ?: "?"} · ${v.sampleRate}Hz · ${v.channelCount}ch · ${formatDuration(v.durationMs)}",
                            style = MaterialTheme.typography.bodySmall,
                            color = DarkOnSurfaceVariant
                        )
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = onSelectRecording,
                    colors = ButtonDefaults.buttonColors(containerColor = ShieldCyan),
                    shape = RoundedCornerShape(8.dp),
                    enabled = session.outcome != PostCallAnalysisOutcome.IN_PROGRESS
                ) {
                    Icon(Icons.Default.FolderOpen, null, tint = Color.Black, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("SELECT RECORDING", color = Color.Black, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }

                if (session.recordingUri != null &&
                    session.recordingValidation?.status == RecordingStatus.VERIFIED &&
                    session.outcome != PostCallAnalysisOutcome.IN_PROGRESS) {
                    Button(
                        onClick = onStartAnalysis,
                        colors = ButtonDefaults.buttonColors(containerColor = ShieldGreen),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.PlayArrow, null, tint = Color.Black, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("ANALYZE", color = Color.Black, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

@Composable
private fun RecordingValidationCard(validation: RecordingValidation) {
    val statusColor = when (validation.status) {
        RecordingStatus.VERIFIED -> ShieldGreen
        else -> ShieldRed
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            SectionLabel("RECORDING VALIDATION")
            Spacer(modifier = Modifier.height(6.dp))
            MetricRow("Status", validation.status.name, statusColor)
            MetricRow("Format", validation.format ?: "Unknown")
            MetricRow("Duration", formatDuration(validation.durationMs))
            MetricRow("Sample Rate", "${validation.sampleRate} Hz")
            MetricRow("Channels", "${validation.channelCount}")
            if (validation.decodedSampleCount > 0) {
                MetricRow("Decoded Samples", "${validation.decodedSampleCount}")
            }
            validation.errorMessage?.let {
                MetricRow("Error", it, ShieldRed)
            }
        }
    }
}

@Composable
private fun QualityMetricsCard(metrics: RecordingQualityMetrics) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            SectionLabel("QUALITY METRICS")
            Text(
                "Informational — does not auto-reject analysis",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant,
                fontSize = 9.sp
            )
            Spacer(modifier = Modifier.height(6.dp))
            MetricRow("RMS", String.format("%.5f", metrics.rms))
            MetricRow("Peak", String.format("%.5f", metrics.peak))
            MetricRow("Silence", String.format("%.1f%%", metrics.silencePercent))
            MetricRow("Clipping", String.format("%.1f%%", metrics.clippingPercent))
            if (metrics.nanCount > 0) MetricRow("NaN", "${metrics.nanCount}", ShieldRed)
            if (metrics.infCount > 0) MetricRow("Inf", "${metrics.infCount}", ShieldRed)
        }
    }
}

@Composable
private fun SpeechAvailabilityCard(speech: SpeechAvailability) {
    val statusColor = if (speech.sufficientForAnalysis) ShieldGreen else ShieldYellow
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            SectionLabel("SPEECH AVAILABILITY")
            Spacer(modifier = Modifier.height(6.dp))
            MetricRow("Speech Detected", if (speech.hasSpeech) "YES" else "NO",
                if (speech.hasSpeech) ShieldGreen else ShieldRed)
            MetricRow("Total Speech", "${speech.totalSpeechMs / 1000}s")
            MetricRow("Segments", "${speech.speechSegmentCount}")
            MetricRow("Speech Ratio", String.format("%.1f%%", speech.speechNonSpeechRatio * 100))
            MetricRow("Sufficient for Analysis",
                if (speech.sufficientForAnalysis) "YES" else "UNCERTAIN", statusColor)
            speech.insufficiencyReason?.let {
                MetricRow("Note", it, ShieldYellow)
            }
        }
    }
}

@Composable
private fun SpeakerSeparationCard(status: SpeakerSeparationStatus) {
    val statusColor = when (status) {
        SpeakerSeparationStatus.SEPARATED -> ShieldGreen
        SpeakerSeparationStatus.PROVISIONAL -> ShieldYellow
        SpeakerSeparationStatus.UNCERTAIN -> ShieldOrange
        SpeakerSeparationStatus.SINGLE_SPEAKER -> DarkOnSurfaceVariant
        SpeakerSeparationStatus.FAILED -> ShieldRed
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            SectionLabel("SPEAKER SEPARATION")
            Text(
                "Experimental — ECAPA embeddings, uncalibrated clustering",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant,
                fontSize = 9.sp
            )
            Spacer(modifier = Modifier.height(6.dp))
            MetricRow("Status", status.name, statusColor)
            Text(
                "Speaker labels (A/B) are cluster IDs, NOT caller/user identification",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant,
                fontSize = 9.sp
            )
        }
    }
}

@Composable
private fun AnalysisTimelineCard(evidence: List<PostCallWindowEvidence>) {
    val speechWindows = evidence.filter { it.aasistStatus != "NO_SPEECH" && it.aasistStatus != "PENDING" }
    if (speechWindows.isEmpty()) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            SectionLabel("ANALYSIS TIMELINE")
            Spacer(modifier = Modifier.height(6.dp))

            // Show up to 20 windows to avoid overwhelming the UI
            val displayWindows = speechWindows.take(20)
            for (w in displayWindows) {
                val timeRange = "${formatTimestamp(w.startMs)}–${formatTimestamp(w.endMs)}"
                val speakerText = when {
                    w.isMixedSpeakerWindow -> "MIXED"
                    w.speakerLabel != null -> w.speakerLabel
                    else -> "—"
                }
                val aasistText = w.aasistScore?.let { String.format("%.4f", it) } ?: "N/A"
                val aasistColor = when {
                    w.aasistScore == null -> DarkOnSurfaceVariant
                    w.aasistScore > 0.5f -> ShieldRed
                    w.aasistScore > 0.3f -> ShieldYellow
                    else -> ShieldGreen
                }

                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        timeRange,
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        color = DarkOnSurfaceVariant,
                        fontSize = 10.sp,
                        modifier = Modifier.width(100.dp)
                    )
                    Text(
                        speakerText,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium,
                        color = if (speakerText == "MIXED") ShieldOrange else ShieldCyan,
                        fontSize = 10.sp,
                        modifier = Modifier.width(80.dp)
                    )
                    Text(
                        "AASIST: $aasistText",
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        color = aasistColor,
                        fontSize = 10.sp
                    )
                }
            }

            if (speechWindows.size > 20) {
                Text(
                    "... and ${speechWindows.size - 20} more windows",
                    style = MaterialTheme.typography.bodySmall,
                    color = DarkOnSurfaceVariant,
                    fontSize = 9.sp
                )
            }
        }
    }
}

@Composable
private fun FinalResultCard(session: PostCallAnalysisSession) {
    val outcomeColor = when (session.outcome) {
        PostCallAnalysisOutcome.NORMAL_SPEECH_EVIDENCE -> ShieldGreen
        PostCallAnalysisOutcome.SUSPICIOUS_SYNTHETIC_SPEECH_EVIDENCE -> ShieldRed
        PostCallAnalysisOutcome.SPEAKER_IDENTITY_MISMATCH_EVIDENCE -> ShieldOrange
        PostCallAnalysisOutcome.INCONCLUSIVE -> ShieldYellow
        else -> DarkOnSurfaceVariant
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            SectionLabel("FINAL ASSESSMENT")
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                session.outcome.name.replace("_", " "),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = outcomeColor
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                session.summaryText,
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurface
            )
            session.errorMessage?.let {
                Spacer(modifier = Modifier.height(4.dp))
                Text(it, style = MaterialTheme.typography.bodySmall, color = ShieldRed, fontSize = 10.sp)
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Evidence-based assessment with conservative abstention. " +
                "Raw model evidence — not calibrated probability.",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant,
                fontSize = 9.sp
            )
        }
    }
}

@Composable
private fun ProgressCard(progress: Float) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("ANALYZING...", color = ShieldCyan, fontWeight = FontWeight.Bold, fontSize = 14.sp)
            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth().height(6.dp),
                color = ShieldCyan,
                trackColor = DarkSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "${(progress * 100).toInt()}%",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant
            )
        }
    }
}

@Composable
private fun UserGuidanceCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurfaceVariant),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                "HOW TO USE",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface,
                letterSpacing = 1.sp
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                "1. Enable your phone's built-in call recording feature\n" +
                "2. Make a call — the Phone app records both sides\n" +
                "3. After the call, open VoiceShield → Post-Call Analysis\n" +
                "4. Select the call recording via file picker\n" +
                "5. VoiceShield analyzes the recording locally",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant,
                lineHeight = 18.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "If your phone's call recording is disabled, VoiceShield cannot " +
                "analyze this cellular call. Enable it in your Phone app settings.",
                style = MaterialTheme.typography.bodySmall,
                color = ShieldYellow,
                fontSize = 10.sp
            )
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────

@Composable
private fun SectionLabel(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Bold,
        color = ShieldCyan,
        letterSpacing = 1.sp
    )
}

@Composable
private fun MetricRow(label: String, value: String, valueColor: Color = DarkOnSurface) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 1.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = DarkOnSurfaceVariant, fontSize = 11.sp)
        Text(value, style = MaterialTheme.typography.bodySmall, color = valueColor,
            fontFamily = FontFamily.Monospace, fontSize = 11.sp)
    }
}

private fun formatDuration(ms: Long): String {
    val totalSeconds = ms / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return "${minutes}m ${seconds}s"
}

private fun formatTimestamp(ms: Long): String {
    val totalSeconds = ms / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return String.format("%02d:%02d", minutes, seconds)
}

@Composable
private fun OemDiscoveryStatusCard(
    isDiscovering: Boolean,
    result: com.voiceshield.audio.OemDiscoveryResult?,
    onRetry: (() -> Unit)? = null
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "AUTOMATIC RECORDING DISCOVERY",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = ShieldCyan,
                    letterSpacing = 1.sp
                )
                if (isDiscovering) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = ShieldCyan,
                        strokeWidth = 2.dp
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            when {
                isDiscovering -> {
                    Text(
                        text = "Searching for OEM call recording in MIUI/sound_recorder/call_rec/...",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Waiting for Xiaomi MIUI Sound Recorder to flush file...",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 10.sp
                    )
                }
                result is com.voiceshield.audio.OemDiscoveryResult.Discovered -> {
                    Text(
                        text = "✓ Discovered OEM Recording: ${result.displayName}",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = ShieldGreen
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Matched by: ${result.confidenceReason}",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 11.sp
                    )
                }
                result is com.voiceshield.audio.OemDiscoveryResult.Unavailable -> {
                    Text(
                        text = "STATUS: ${result.recordingState.name}",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold,
                        color = ShieldYellow
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Automatic OEM call recording was not available for this call.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = result.reason,
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 11.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Fallback: Please tap SELECT RECORDING below to import the recording file.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ShieldCyan,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
                result is com.voiceshield.audio.OemDiscoveryResult.NotFound -> {
                    Text(
                        text = "STATUS: ${result.recordingState.name}",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold,
                        color = ShieldYellow
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Automatic OEM call recording was not available for this call.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "No recording file was found in MIUI/sound_recorder/call_rec/ for this call. " +
                               "MIUI's built-in recorder did not create a file while VoiceShield was the active dialer.",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 11.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Please tap SELECT RECORDING below to choose a recording file manually.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ShieldCyan,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
                result is com.voiceshield.audio.OemDiscoveryResult.Ambiguous -> {
                    Text(
                        text = "STATUS: ${result.recordingState.name}",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold,
                        color = ShieldOrange
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Automatic OEM call recording was not available for this call.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = result.reason,
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 11.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Please tap SELECT RECORDING below to choose the correct recording file.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ShieldCyan,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
                result is com.voiceshield.audio.OemDiscoveryResult.Invalid -> {
                    Text(
                        text = "STATUS: ${result.recordingState.name}",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold,
                        color = ShieldRed
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Automatic OEM call recording was not available for this call.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = result.reason,
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 11.sp
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Please tap SELECT RECORDING below to choose a valid recording file.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ShieldCyan,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
                else -> {}
            }
        }
    }
}
