/**
 * ActiveCallScreen — Shown during an active call.
 *
 * Displays:
 * - Call duration and caller info
 * - Audio source availability (AVAILABLE / UNAVAILABLE)
 * - Voice analysis status
 * - Real-time threat state with evidence
 * - Call control buttons
 *
 * CRITICAL: Only shows metrics produced by actual models.
 * If audio is unavailable, clearly shows analysis is inactive.
 */
package com.voiceshield.ui.activecall

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.voiceshield.shared.*
import com.voiceshield.ui.theme.*

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll

@Composable
fun ActiveCallScreen(
    phoneNumber: String?,
    callDuration: Long,
    audioSourceType: AudioSourceType,
    analysisActive: Boolean,
    threatState: ThreatState,
    windowsProcessed: Int,
    onEndCall: () -> Unit,
    onHold: () -> Unit,
    onMute: () -> Unit,
    onSpeaker: () -> Unit,
    isMuted: Boolean = false,
    isSpeaker: Boolean = false,
    isTelephonyOnlyMode: Boolean = false,
    isAcousticCaptureEnabled: Boolean = false,
    onToggleAcousticCapture: ((Boolean) -> Unit)? = null,
    currentAudioSource: String = "MIC",
    onSwitchAudioSource: ((String) -> Unit)? = null,
    onOpenDiagnostics: () -> Unit = {},
    pipelineTelemetry: AcousticPipelineTelemetry = AcousticPipelineTelemetry(),
    onSendDtmf: ((Char) -> Unit)? = null
) {
    var showAcousticWarningDialog by remember { mutableStateOf(false) }
    var showKeypadDialog by remember { mutableStateOf(false) }
    var dialedDigits by remember { mutableStateOf("") }

    if (showAcousticWarningDialog) {
        AlertDialog(
            onDismissRequest = { showAcousticWarningDialog = false },
            title = {
                Text(
                    text = "Enable Acoustic Analysis?",
                    color = DarkOnSurface,
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Text(
                    text = "Acoustic analysis captures audio via your phone's microphone to analyze sound from the speaker. It works best on speakerphone.\n\n" +
                            "⚠️ CALL PRIVACY & STABILITY GUARANTEE:\n" +
                            "• Normal phone call audio is never blocked\n" +
                            "• If audio degrades or call drops, capture stops immediately\n" +
                            "• No audio is saved or transmitted\n" +
                            "• Call always takes priority over analysis",
                    color = DarkOnSurfaceVariant,
                    style = MaterialTheme.typography.bodyMedium
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showAcousticWarningDialog = false
                        onToggleAcousticCapture?.invoke(true)
                    }
                ) {
                    Text("Enable (Experimental)", color = ShieldCyan, fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showAcousticWarningDialog = false }) {
                    Text("Cancel", color = DarkOnSurfaceVariant)
                }
            },
            containerColor = DarkSurface
        )
    }

    Scaffold(
        containerColor = DarkBackground,
        bottomBar = {
            Surface(
                color = DarkBackground,
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 24.dp, vertical = 16.dp)
                ) {
                    CallControls(
                        onEndCall = onEndCall,
                        onHold = onHold,
                        onMute = onMute,
                        onSpeaker = onSpeaker,
                        isMuted = isMuted,
                        isSpeaker = isSpeaker,
                        onToggleKeypad = { showKeypadDialog = !showKeypadDialog },
                        isKeypadOpen = showKeypadDialog
                    )
                }
            }
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Top: Protection badge & Diagnostics button
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                ProtectionBadge(audioSourceType = audioSourceType)

                IconButton(onClick = onOpenDiagnostics) {
                    Icon(
                        Icons.Default.Analytics,
                        contentDescription = "Audio Diagnostics",
                        tint = ShieldCyan
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Caller info
            Text(
                phoneNumber ?: "Unknown Number",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )

            Text(
                text = formatDuration(callDuration),
                style = MaterialTheme.typography.bodyLarge,
                color = DarkOnSurfaceVariant
            )

            Spacer(modifier = Modifier.height(20.dp))

            // Audio & Analysis Status
            AudioStatusBanner(
                audioSourceType = audioSourceType,
                analysisActive = analysisActive,
                isTelephonyOnlyMode = isTelephonyOnlyMode,
                pipelineTelemetry = pipelineTelemetry
            )

            // Acoustic Capture Opt-in Card (for Cellular calls)
            if (!isTelephonyOnlyMode && onToggleAcousticCapture != null &&
                (audioSourceType == AudioSourceType.UNAVAILABLE || audioSourceType == AudioSourceType.ACOUSTIC_CAPTURE)) {
                Spacer(modifier = Modifier.height(12.dp))
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = DarkSurface),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Acoustic Voice Analysis",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = DarkOnSurface
                            )
                            Text(
                                text = if (isAcousticCaptureEnabled)
                                    "Microphone capture active (works best on speakerphone)"
                                else
                                    "OFF by default. Analyze caller via device microphone.",
                                style = MaterialTheme.typography.bodySmall,
                                color = DarkOnSurfaceVariant
                            )
                        }
                        Switch(
                            checked = isAcousticCaptureEnabled,
                            onCheckedChange = { enable ->
                                if (enable) {
                                    showAcousticWarningDialog = true
                                } else {
                                    onToggleAcousticCapture(false)
                                }
                            },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = ShieldCyan,
                                checkedTrackColor = ShieldCyan.copy(alpha = 0.3f)
                            )
                        )
                    }
                }
            }

            // Live Pipeline Telemetry Card when Acoustic Mode is enabled
            if (isAcousticCaptureEnabled || audioSourceType == AudioSourceType.ACOUSTIC_CAPTURE) {
                // Debug Audio Source Switcher
                Spacer(modifier = Modifier.height(8.dp))
                AudioSourceSwitcherCard(
                    currentSource = currentAudioSource,
                    onSwitch = { newSource -> onSwitchAudioSource?.invoke(newSource) },
                    isEnabled = isAcousticCaptureEnabled
                )

                Spacer(modifier = Modifier.height(8.dp))
                PipelineTelemetryCard(telemetry = pipelineTelemetry)
                Spacer(modifier = Modifier.height(8.dp))
                RawPcmForensicCard(telemetry = pipelineTelemetry)
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Threat state display — only if analysis is active
            if (analysisActive) {
                ThreatStateCard(
                    threatState = threatState,
                    windowsProcessed = windowsProcessed
                )
            }
        }
    }

    if (showKeypadDialog) {
        InCallKeypadDialog(
            dialedDigits = dialedDigits,
            onDigitClick = { digit ->
                dialedDigits += digit
                onSendDtmf?.invoke(digit)
            },
            onClear = {
                if (dialedDigits.isNotEmpty()) {
                    dialedDigits = dialedDigits.dropLast(1)
                }
            },
            onDismiss = { showKeypadDialog = false }
        )
    }
}

@Composable
private fun PipelineTelemetryCard(telemetry: AcousticPipelineTelemetry) {
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
                    text = "PIPELINE TELEMETRY",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = ShieldCyan,
                    letterSpacing = 1.sp
                )
                Text(
                    text = telemetry.status.name,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = when (telemetry.status) {
                        AcousticAnalysisStatus.PROCESSING_SPEECH -> ShieldGreen
                        AcousticAnalysisStatus.RECEIVING_AUDIO,
                        AcousticAnalysisStatus.BUFFERING -> ShieldCyan
                        AcousticAnalysisStatus.ERROR -> ShieldRed
                        else -> ShieldYellow
                    }
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Diagnostic Stop Case
            Text(
                text = telemetry.diagnoseStopCase(),
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurface,
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium
            )

            Spacer(modifier = Modifier.height(8.dp))
            Divider(color = DarkSurfaceVariant.copy(alpha = 0.5f), thickness = 0.5.dp)
            Spacer(modifier = Modifier.height(8.dp))

            // Stage counters & Boundary Telemetry
            TelemetryLine("AudioRecord", "${if (telemetry.audioRecordInitialized) "INIT_OK" else "NOT_INIT"} | ${telemetry.recordingState} | Src: ${telemetry.selectedAudioSource}")
            TelemetryLine("Mic Activity", "${telemetry.micActivityState} (Cur RMS: ${String.format("%.5f", telemetry.currentRms)}, Peak: ${String.format("%.4f", telemetry.peakRms)})")
            TelemetryLine("Frames Read", "${telemetry.framesRead} (${telemetry.bytesRead} B) | Err: ${telemetry.readErrors}")
            TelemetryLine("Frames Delivered", "${telemetry.framesDelivered} frames")
            TelemetryLine("Normalizer", "${telemetry.normalizerSamplesProduced} samples | Gain: ${String.format("%.2f", telemetry.normalizerGain)}x")
            TelemetryLine("AASIST Buffer", "${telemetry.bufferCurrentSamples}/${telemetry.bufferRequiredSamples} (${(telemetry.bufferFillPercentage * 100).toInt()}%) | Win: ${telemetry.windowsEmitted}")
            TelemetryLine("Window Signal", "RMS: ${String.format("%.5f", telemetry.bufferWindowMetrics?.rms ?: 0f)} | Class: ${telemetry.bufferWindowMetrics?.classification ?: "BUFFERING"}")
            TelemetryLine("Silero VAD", "maxProb: ${String.format("%.4f", telemetry.vadMaxProbability)}, mean: ${String.format("%.4f", telemetry.vadMeanProbability)}, min: ${String.format("%.4f", telemetry.vadMinProbability)}")
            TelemetryLine("VAD Speech Frames", "${telemetry.vadSpeechFrames} speech frames | Win Frames: ${telemetry.vadTotalFramesInLastWindow}")
            
            // Amplitude vs VAD correlation indicator
            val windowRms = telemetry.sileroInputMetrics?.rms ?: telemetry.currentRms
            val correlationStatus = when {
                windowRms > 0.02f && telemetry.vadMaxProbability < 0.1f -> "MISMATCH (High RMS, Low VAD)"
                windowRms < 0.005f -> "CORRELATED (Low RMS = Silence)"
                telemetry.vadMaxProbability >= 0.5f -> "CORRELATED (Speech Detected)"
                else -> "MONITORING"
            }
            TelemetryLine("Signal vs VAD", correlationStatus)

            TelemetryLine("AASIST-L", "Inf: ${telemetry.aasistInferences} | Fail: ${telemetry.aasistFailures} | Score: ${telemetry.lastAasistScore?.let { String.format("%.3f", it) } ?: "N/A"}")
            TelemetryLine("ECAPA-TDNN", "Inf: ${telemetry.ecapaInferences} | Sim: ${telemetry.lastEcapaSimilarity?.let { String.format("%.3f", it) } ?: "N/A"}")
            TelemetryLine("RiskEngine", "Evidence Win: ${telemetry.evidenceWindows} | State: ${telemetry.lastRiskState}")
        }
    }
}

@Composable
private fun AudioSourceSwitcherCard(
    currentSource: String,
    onSwitch: (String) -> Unit,
    isEnabled: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "AUDIO SOURCE TEST",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = ShieldYellow,
                letterSpacing = 1.sp
            )
            Text(
                text = "Switch source → stop + restart (same AudioRecord owner, no concurrent instances)",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant,
                fontSize = 9.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val sources = listOf("MIC", "VOICE_COMMUNICATION")
                sources.forEach { source ->
                    val isActive = source == currentSource
                    Button(
                        onClick = { if (!isActive) onSwitch(source) },
                        modifier = Modifier.weight(1f),
                        enabled = isEnabled && !isActive,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isActive) ShieldCyan else DarkSurfaceVariant,
                            disabledContainerColor = if (isActive) ShieldCyan.copy(alpha = 0.7f) else DarkSurfaceVariant.copy(alpha = 0.5f)
                        ),
                        shape = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = if (source == "VOICE_COMMUNICATION") "VOICE_COMM" else source,
                            fontSize = 10.sp,
                            fontWeight = if (isActive) FontWeight.Bold else FontWeight.Medium,
                            color = if (isActive) Color.Black else DarkOnSurface
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
            TelemetryLine("Active Source", currentSource, if (currentSource == "VOICE_COMMUNICATION") ShieldYellow else ShieldCyan)
        }
    }
}

@Composable
private fun RawPcmForensicCard(telemetry: AcousticPipelineTelemetry) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = "RAW PCM FORENSIC TRACE",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = ShieldYellow,
                letterSpacing = 1.sp
            )

            Spacer(modifier = Modifier.height(8.dp))
            Divider(color = DarkSurfaceVariant.copy(alpha = 0.5f), thickness = 0.5.dp)
            Spacer(modifier = Modifier.height(6.dp))

            // LAYER 1: AudioRecord Stream
            ForensicSectionLabel("LAYER 1: AUDIORECORD STREAM")
            TelemetryLine(
                "Stream",
                if (telemetry.audioRecordInitialized && telemetry.framesRead > 0)
                    "ACTIVE (${telemetry.framesRead} frames)"
                else "INACTIVE"
            )

            Spacer(modifier = Modifier.height(6.dp))

            // LAYER 2: Raw Short PCM
            ForensicSectionLabel("LAYER 2: RAW SHORT PCM")
            val rawShort = telemetry.rawShortMetrics
            if (rawShort != null) {
                val shortSignalColor = when (telemetry.rawPcmSignalState) {
                    "ZERO" -> ShieldRed
                    "LOW" -> ShieldYellow
                    "NONZERO" -> ShieldGreen
                    else -> DarkOnSurface
                }
                TelemetryLine("Signal", telemetry.rawPcmSignalState, shortSignalColor)
                TelemetryLine("RMS", String.format("%.1f", rawShort.rmsShort))
                TelemetryLine("Peak", "${rawShort.peakAbsShort}")
                TelemetryLine("Min / Max", "${rawShort.minShort} / ${rawShort.maxShort}")
                TelemetryLine("NonZero", "${rawShort.nonZeroCount} / ${rawShort.sampleCount} (${String.format("%.1f", 100f - rawShort.zeroPercent)}%)")
            } else {
                TelemetryLine("Signal", "AWAITING DATA", DarkOnSurfaceVariant)
            }

            Spacer(modifier = Modifier.height(6.dp))

            // LAYER 3: Float32 (post-conversion)
            ForensicSectionLabel("LAYER 3: FLOAT32 (post-convert)")
            val floatSignalColor = when (telemetry.normalizedSignalState) {
                "ZERO" -> ShieldRed
                "LOW" -> ShieldYellow
                "NONZERO" -> ShieldGreen
                else -> DarkOnSurface
            }
            TelemetryLine("Signal", telemetry.normalizedSignalState, floatSignalColor)
            TelemetryLine("RMS", String.format("%.5f", telemetry.currentRms))
            TelemetryLine("Peak", String.format("%.5f", telemetry.peakRms))

            Spacer(modifier = Modifier.height(6.dp))

            // LAYER 4: Normalizer Output
            ForensicSectionLabel("LAYER 4: NORMALIZER OUTPUT")
            TelemetryLine("Gain", String.format("%.4f", telemetry.normalizerGain) + "x")

            Spacer(modifier = Modifier.height(6.dp))

            // LAYER 5: VAD Decision
            ForensicSectionLabel("LAYER 5: VAD DECISION")
            val vadColor = when (telemetry.vadDecisionState) {
                "SPEECH" -> ShieldGreen
                "NON_SPEECH" -> ShieldYellow
                else -> DarkOnSurfaceVariant
            }
            TelemetryLine("Decision", telemetry.vadDecisionState, vadColor)
            TelemetryLine("maxProb", String.format("%.4f", telemetry.vadMaxProbability))

            Spacer(modifier = Modifier.height(8.dp))
            Divider(color = DarkSurfaceVariant.copy(alpha = 0.5f), thickness = 0.5.dp)
            Spacer(modifier = Modifier.height(6.dp))

            // CORRELATION
            ForensicSectionLabel("CORRELATION")
            if (rawShort != null) {
                TelemetryLine("Expected Float RMS", String.format("%.6f", rawShort.expectedFloatRms))
                TelemetryLine("Actual Float RMS", String.format("%.6f", rawShort.actualFloatRms))
                TelemetryLine("Conversion Error", String.format("%.8f", rawShort.conversionError))

                val matchColor = when (telemetry.shortFloatMatch) {
                    "MATCH_SIGNAL_ALIVE" -> ShieldGreen
                    "MATCH_BOTH_ZERO" -> ShieldYellow
                    "MISMATCH_CONVERSION_DEFECT" -> ShieldRed
                    else -> DarkOnSurface
                }
                TelemetryLine("Short→Float Match", telemetry.shortFloatMatch, matchColor)

                // Root cause pointer
                val rootCause = when {
                    telemetry.rawPcmSignalState == "ZERO" -> "ROOT CAUSE: Android delivers zeros"
                    telemetry.rawPcmSignalState != "ZERO" && telemetry.normalizedSignalState == "ZERO" -> "ROOT CAUSE: Conversion defect"
                    telemetry.normalizedSignalState != "ZERO" && telemetry.vadDecisionState == "NON_SPEECH" -> "INVESTIGATE: Silero VAD"
                    telemetry.rawPcmSignalState != "ZERO" && telemetry.vadDecisionState == "SPEECH" -> "PIPELINE OPERATIONAL"
                    else -> "COLLECTING DATA..."
                }
                val rootColor = when {
                    rootCause.startsWith("ROOT CAUSE") -> ShieldRed
                    rootCause.startsWith("INVESTIGATE") -> ShieldYellow
                    rootCause.startsWith("PIPELINE") -> ShieldGreen
                    else -> DarkOnSurfaceVariant
                }
                Spacer(modifier = Modifier.height(4.dp))
                TelemetryLine("Root Cause Layer", rootCause, rootColor)
            } else {
                TelemetryLine("Short→Float Match", "AWAITING DATA", DarkOnSurfaceVariant)
            }
        }
    }
}

@Composable
private fun ForensicSectionLabel(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.Bold,
        color = ShieldCyan.copy(alpha = 0.7f),
        fontSize = 9.sp,
        letterSpacing = 0.5.sp,
        modifier = Modifier.padding(vertical = 2.dp)
    )
}

@Composable
private fun TelemetryLine(label: String, value: String, valueColor: Color = DarkOnSurface) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 1.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = DarkOnSurfaceVariant,
            fontSize = 10.sp
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            color = valueColor,
            fontSize = 10.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun ProtectionBadge(audioSourceType: AudioSourceType) {
    val (text, color) = when (audioSourceType) {
        AudioSourceType.WEBRTC_DIRECT -> "📞 CALL PROTECTED BY VOICESHIELD" to ShieldGreen
        AudioSourceType.CELLULAR_DOWNLINK -> "📞 CALL PROTECTED BY VOICESHIELD (CELLULAR)" to ShieldGreen
        AudioSourceType.CELLULAR_MIXED -> "📞 CALL PROTECTED BY VOICESHIELD (CELLULAR MIXED)" to ShieldGreen
        AudioSourceType.PLATFORM_SUPPORTED -> "📞 CALL PROTECTED BY VOICESHIELD" to ShieldGreen
        AudioSourceType.ACOUSTIC_CAPTURE -> "🎤 ACOUSTIC CAPTURE (MIC ANALYSIS)" to ShieldYellow
        AudioSourceType.UNAVAILABLE -> "📞 CALL LIFECYCLE MONITORING" to DarkOnSurfaceVariant
    }

    Card(
        colors = CardDefaults.cardColors(
            containerColor = color.copy(alpha = 0.1f)
        ),
        shape = RoundedCornerShape(8.dp)
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            color = color,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun AudioStatusBanner(
    audioSourceType: AudioSourceType,
    analysisActive: Boolean,
    isTelephonyOnlyMode: Boolean,
    pipelineTelemetry: AcousticPipelineTelemetry = AcousticPipelineTelemetry()
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            StatusLine(
                "Audio Source",
                when (audioSourceType) {
                    AudioSourceType.WEBRTC_DIRECT -> "WEBRTC DIRECT PCM"
                    AudioSourceType.PLATFORM_SUPPORTED -> "AVAILABLE"
                    AudioSourceType.ACOUSTIC_CAPTURE -> "ACOUSTIC CAPTURE (MIC)"
                    AudioSourceType.UNAVAILABLE -> "NOT YET VERIFIED"
                    else -> "AVAILABLE"
                },
                when (audioSourceType) {
                    AudioSourceType.UNAVAILABLE -> ShieldYellow
                    AudioSourceType.ACOUSTIC_CAPTURE -> ShieldYellow
                    else -> ShieldGreen
                }
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Honest, Context-aware analysis status
            val (analysisLabel, analysisColor) = when {
                isTelephonyOnlyMode -> "DISABLED — Diagnostic mode" to ShieldYellow
                audioSourceType == AudioSourceType.ACOUSTIC_CAPTURE -> {
                    when (pipelineTelemetry.status) {
                        AcousticAnalysisStatus.OFF -> "ACOUSTIC MODE ENABLED" to ShieldYellow
                        AcousticAnalysisStatus.INITIALIZING -> "CAPTURE INITIALIZING..." to ShieldYellow
                        AcousticAnalysisStatus.RECEIVING_AUDIO -> "CAPTURE ACTIVE — RECEIVING AUDIO" to ShieldGreen
                        AcousticAnalysisStatus.BUFFERING -> "BUFFERING AUDIO (${pipelineTelemetry.bufferCurrentSamples}/${pipelineTelemetry.bufferRequiredSamples})" to ShieldCyan
                        AcousticAnalysisStatus.WAITING_FOR_SPEECH -> "WAITING FOR SPEECH (VAD: silence)" to ShieldYellow
                        AcousticAnalysisStatus.PROCESSING_SPEECH -> "ANALYSIS ACTIVE — PROCESSING SPEECH" to ShieldGreen
                        AcousticAnalysisStatus.ERROR -> "ANALYSIS ERROR (${pipelineTelemetry.stopReason ?: "Unknown"})" to ShieldRed
                    }
                }
                analysisActive -> "ACTIVE — Analyzing speech" to ShieldGreen
                audioSourceType == AudioSourceType.UNAVAILABLE ->
                    "STANDBY — No remote audio path" to DarkOnSurfaceVariant
                else -> "INACTIVE" to DarkOnSurfaceVariant
            }

            StatusLine(
                "Voice Analysis",
                analysisLabel,
                analysisColor
            )

            if (audioSourceType == AudioSourceType.ACOUSTIC_CAPTURE) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Acoustic mode captures sound via device microphone. Speakerphone recommended.",
                    style = MaterialTheme.typography.bodySmall,
                    color = DarkOnSurfaceVariant,
                    fontSize = 11.sp
                )
            }
        }
    }
}

@Composable
private fun StatusLine(label: String, value: String, valueColor: Color) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = DarkOnSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.Bold,
            color = valueColor
        )
    }
}

@Composable
fun ThreatStateCard(
    threatState: ThreatState,
    windowsProcessed: Int
) {
    val stateColor = threatLevelColor(threatState.level)
    val stateLabel = threatLevelLabel(threatState.level)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = stateColor.copy(alpha = 0.08f)
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(CircleShape)
                        .background(stateColor)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stateLabel,
                    fontWeight = FontWeight.Bold,
                    color = stateColor,
                    fontSize = 14.sp
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = threatState.reason,
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurface,
                lineHeight = 18.sp
            )

            if (threatState.avgSpoofScore != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Anti-spoof score: ${String.format("%.1f", threatState.avgSpoofScore * 100)}% " +
                            "| Windows: ${threatState.windowsAnalyzed}",
                    style = MaterialTheme.typography.bodySmall,
                    color = DarkOnSurfaceVariant,
                    fontSize = 11.sp
                )
            }
        }
    }
}

@Composable
fun CallControls(
    onEndCall: () -> Unit,
    onHold: () -> Unit,
    onMute: () -> Unit,
    onSpeaker: () -> Unit,
    isMuted: Boolean,
    isSpeaker: Boolean,
    onToggleKeypad: () -> Unit = {},
    isKeypadOpen: Boolean = false
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Mute
        CallControlButton(
            icon = if (isMuted) Icons.Default.MicOff else Icons.Default.Mic,
            label = if (isMuted) "Unmute" else "Mute",
            onClick = onMute,
            tint = if (isMuted) ShieldYellow else DarkOnSurface
        )

        // Hold
        CallControlButton(
            icon = Icons.Default.Pause,
            label = "Hold",
            onClick = onHold,
            tint = DarkOnSurface
        )

        // End Call
        IconButton(
            onClick = onEndCall,
            modifier = Modifier
                .size(64.dp)
                .clip(CircleShape)
                .background(ShieldRed)
        ) {
            Icon(
                Icons.Default.CallEnd,
                contentDescription = "End Call",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }

        // Speaker
        CallControlButton(
            icon = if (isSpeaker) Icons.Default.VolumeUp else Icons.Default.VolumeDown,
            label = if (isSpeaker) "Earpiece" else "Speaker",
            onClick = onSpeaker,
            tint = if (isSpeaker) ShieldCyan else DarkOnSurface
        )

        // Keypad button
        CallControlButton(
            icon = Icons.Default.Dialpad,
            label = "Keypad",
            onClick = onToggleKeypad,
            tint = if (isKeypadOpen) ShieldCyan else DarkOnSurface
        )
    }
}

@Composable
private fun InCallKeypadDialog(
    dialedDigits: String,
    onDigitClick: (Char) -> Unit,
    onClear: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Keypad",
                    style = MaterialTheme.typography.titleMedium,
                    color = DarkOnSurface,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(DarkSurfaceVariant)
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (dialedDigits.isEmpty()) "Tap keys..." else dialedDigits,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (dialedDigits.isEmpty()) DarkOnSurfaceVariant else ShieldCyan,
                        maxLines = 1
                    )
                    if (dialedDigits.isNotEmpty()) {
                        IconButton(
                            onClick = onClear,
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(
                                Icons.Default.Backspace,
                                contentDescription = "Clear",
                                tint = DarkOnSurfaceVariant,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
            }
        },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val keyRows = listOf(
                    listOf("1" to "", "2" to "ABC", "3" to "DEF"),
                    listOf("4" to "GHI", "5" to "JKL", "6" to "MNO"),
                    listOf("7" to "PQRS", "8" to "TUV", "9" to "WXYZ"),
                    listOf("*" to "", "0" to "+", "#" to "")
                )

                for (row in keyRows) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        for ((digit, letters) in row) {
                            KeypadButton(
                                digit = digit,
                                letters = letters,
                                onClick = { onDigitClick(digit[0]) }
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Hide", color = ShieldCyan, fontWeight = FontWeight.Bold)
            }
        },
        containerColor = DarkSurface,
        shape = RoundedCornerShape(16.dp)
    )
}

@Composable
private fun KeypadButton(
    digit: String,
    letters: String,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = Modifier.size(68.dp),
        shape = CircleShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = DarkSurfaceVariant,
            contentColor = DarkOnSurface
        ),
        contentPadding = PaddingValues(0.dp)
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = digit,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )
            if (letters.isNotEmpty()) {
                Text(
                    text = letters,
                    fontSize = 9.sp,
                    color = DarkOnSurfaceVariant,
                    lineHeight = 10.sp
                )
            }
        }
    }
}

@Composable
private fun CallControlButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
    tint: Color
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        IconButton(
            onClick = onClick,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(DarkSurfaceVariant)
        ) {
            Icon(icon, contentDescription = label, tint = tint)
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = label,
            fontSize = 10.sp,
            color = DarkOnSurfaceVariant
        )
    }
}

private fun formatDuration(ms: Long): String {
    val totalSeconds = ms / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return String.format("%02d:%02d", minutes, seconds)
}
