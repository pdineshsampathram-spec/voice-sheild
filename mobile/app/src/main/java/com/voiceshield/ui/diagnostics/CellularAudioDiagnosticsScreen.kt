/**
 * CellularAudioDiagnosticsScreen — Debug-only screen showing real-time
 * cellular audio capability metrics, tested sources, and technical limitations.
 */
package com.voiceshield.ui.diagnostics

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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalContext
import com.voiceshield.ml.SileroVadEngine
import com.voiceshield.shared.*
import com.voiceshield.ui.theme.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CellularAudioDiagnosticsScreen(
    report: CellularCapabilityReport?,
    telecomDiagnostics: TelecomAudioDiagnostics? = null,
    lastDisconnectInfo: LastDisconnectInfo? = null,
    isProbing: Boolean,
    isCallActive: Boolean = false,
    telephonyOnlyMode: Boolean = false,
    onToggleTelephonyOnlyMode: (Boolean) -> Unit = {},
    onRunProbe: () -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var selfTestResult by remember { mutableStateOf<ControlledTestResult?>(null) }
    var isRunningSelfTest by remember { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Cellular Audio Diagnostics",
                        color = ShieldCyan,
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = DarkOnSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkBackground
                )
            )
        },
        containerColor = DarkBackground
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header summary card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = DarkSurface),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                text = "Call State",
                                style = MaterialTheme.typography.bodySmall,
                                color = DarkOnSurfaceVariant
                            )
                            Text(
                                text = report?.callState ?: "IDLE",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = DarkOnSurface
                            )
                        }

                        val usableStatus = report?.usableRemotePcm ?: RemoteAudioVerification.NOT_YET_VERIFIED
                        val (statusText, statusColor) = when (usableStatus) {
                            RemoteAudioVerification.VERIFIED_TRUE -> "VERIFIED TRUE" to ShieldGreen
                            RemoteAudioVerification.UNVERIFIED_MIC_ONLY -> "MIC ONLY (NOT REMOTE)" to ShieldYellow
                            RemoteAudioVerification.FAILED_OR_RESTRICTED -> "RESTRICTED / FAILED" to ShieldRed
                            RemoteAudioVerification.NOT_YET_VERIFIED -> "NOT YET VERIFIED" to ShieldYellow
                        }

                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(6.dp))
                                .background(statusColor.copy(alpha = 0.15f))
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = statusText,
                                color = statusColor,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider(color = DarkSurfaceVariant)
                    Spacer(modifier = Modifier.height(12.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column {
                            Text("Audio Route", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                            Text("CELLULAR", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = ShieldCyan)
                        }
                        Column {
                            Text("Speakerphone", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                            Text(
                                if (report?.isSpeakerphoneOn == true) "ON" else "OFF",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = DarkOnSurface
                            )
                        }
                        Column {
                            Text("Bluetooth SCO", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                            Text(
                                if (report?.isBluetoothScoOn == true) "ON" else "OFF",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = DarkOnSurface
                            )
                        }
                    }
                }
            }

            // AI Isolation Mode Card (Diagnostic Experiment)
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (telephonyOnlyMode) ShieldYellow.copy(alpha = 0.12f) else DarkSurface
                ),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Telephony-Only Diagnostic Mode",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = if (telephonyOnlyMode) ShieldYellow else DarkOnSurface
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = if (telephonyOnlyMode)
                                    "AI PIPELINE ISOLATED: Zero ML/audio interference during cellular calls."
                                else
                                    "AI ENABLED (Default): Full VoiceShield threat monitoring pipeline active.",
                                style = MaterialTheme.typography.bodySmall,
                                color = DarkOnSurfaceVariant,
                                fontSize = 11.sp
                            )
                        }
                        Switch(
                            checked = telephonyOnlyMode,
                            onCheckedChange = onToggleTelephonyOnlyMode,
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = Color.Black,
                                checkedTrackColor = ShieldYellow
                            )
                        )
                    }
                }
            }

            // Telecom Call Audio Routing Card
            TelecomAudioRoutingCard(diagnostics = telecomDiagnostics)

            // Uplink Investigation Card
            UplinkInvestigationCard(diagnostics = telecomDiagnostics)

            // Disconnect Cause Investigation Card
            DisconnectCauseCard(lastDisconnectInfo = lastDisconnectInfo)

            // Isolated Silero VAD Self-Test Card (Section 8)
            SileroSelfTestCard(
                result = selfTestResult,
                isRunning = isRunningSelfTest,
                onRunTest = {
                    if (!isRunningSelfTest) {
                        isRunningSelfTest = true
                        coroutineScope.launch(Dispatchers.IO) {
                            val res = SileroVadEngine.runIsolatedControlledSelfTest(context)
                            selfTestResult = res
                            isRunningSelfTest = false
                        }
                    }
                }
            )

            // Experiment D: Standalone Mic Diagnostic (no-call only)
            ExperimentDCard(
                isCallActive = isCallActive,
                context = context,
                coroutineScope = coroutineScope
            )

            // Tested Audio Sources Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = DarkSurface),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Tested Audio Sources",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Probed via official AudioRecord API contracts",
                        fontSize = 12.sp,
                        color = DarkOnSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    if (report == null || report.testedSources.isEmpty()) {
                        Text(
                            text = "No capability probe executed yet. Tap 'Run Probe' below.",
                            color = DarkOnSurfaceVariant,
                            fontSize = 13.sp
                        )
                    } else {
                        report.testedSources.forEach { source ->
                            SourceProbeRow(source = source)
                            if (source != report.testedSources.last()) {
                                HorizontalDivider(
                                    color = DarkSurfaceVariant.copy(alpha = 0.5f),
                                    modifier = Modifier.padding(vertical = 8.dp)
                                )
                            }
                        }
                    }
                }
            }

            // Available Input Devices Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = DarkSurface),
                shape = RoundedCornerShape(16.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Available Audio Input Devices",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    Spacer(modifier = Modifier.height(8.dp))

                    if (report?.availableDevices.isNullOrEmpty()) {
                        Text("No external input devices detected", color = DarkOnSurfaceVariant, fontSize = 13.sp)
                    } else {
                        report?.availableDevices?.forEach { dev ->
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.Mic,
                                    contentDescription = null,
                                    tint = ShieldCyan,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(dev, fontSize = 12.sp, color = DarkOnSurface)
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                        }
                    }
                }
            }

            // Technical Limitation & Rationale Card
            if (report?.technicalLimitationReason != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = DarkSurface),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Info, contentDescription = null, tint = ShieldYellow, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                "Technical Platform Boundaries",
                                fontWeight = FontWeight.Bold,
                                color = ShieldYellow,
                                fontSize = 14.sp
                            )
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = report.technicalLimitationReason!!,
                            fontSize = 12.sp,
                            color = DarkOnSurface,
                            lineHeight = 18.sp
                        )
                    }
                }
            }

            // Action: Re-probe button with safety guard
            if (isCallActive) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = ShieldYellow.copy(alpha = 0.15f)),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Warning,
                            contentDescription = null,
                            tint = ShieldYellow,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = "Audio capability probing is disabled during an active call to prevent AudioRecord from interfering with Telecom hardware audio routing.",
                            color = ShieldYellow,
                            fontSize = 12.sp,
                            lineHeight = 16.sp
                        )
                    }
                }
            }

            Button(
                onClick = onRunProbe,
                enabled = !isProbing && !isCallActive,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isCallActive) DarkSurfaceVariant else ShieldCyan,
                    disabledContainerColor = DarkSurfaceVariant
                ),
                shape = RoundedCornerShape(10.dp)
            ) {
                if (isProbing) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        color = Color.Black,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Probing Audio Sources...", color = Color.Black, fontWeight = FontWeight.Bold)
                } else if (isCallActive) {
                    Icon(Icons.Default.Lock, contentDescription = null, tint = DarkOnSurfaceVariant)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Probing Disabled During Call", color = DarkOnSurfaceVariant, fontWeight = FontWeight.Bold)
                } else {
                    Icon(Icons.Default.Refresh, contentDescription = null, tint = Color.Black)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Run Capability Probe Now", color = Color.Black, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
fun SourceProbeRow(source: AudioSourceProbeResult) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = source.sourceName,
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                    fontFamily = FontFamily.Monospace,
                    color = DarkOnSurface
                )
                Text(
                    text = "Category: ${source.category.name}",
                    fontSize = 11.sp,
                    color = when (source.category) {
                        AudioCategory.REMOTE_DOWNLINK, AudioCategory.BIDIRECTIONAL_CALL -> ShieldCyan
                        AudioCategory.NEAR_END_MIC -> ShieldYellow
                        else -> DarkOnSurfaceVariant
                    }
                )
            }

            val statusText = when {
                !source.permissionGranted -> "NO PERM"
                !source.initialized -> "FAILED"
                !source.recordingStarted -> "INIT ONLY"
                source.hasNonZeroPcm -> "SUCCESS"
                else -> "SILENCE (0)"
            }

            val statusColor = when {
                source.hasNonZeroPcm && (source.category == AudioCategory.REMOTE_DOWNLINK || source.category == AudioCategory.BIDIRECTIONAL_CALL) -> ShieldGreen
                source.hasNonZeroPcm && source.category == AudioCategory.NEAR_END_MIC -> ShieldYellow
                !source.initialized || !source.permissionGranted -> ShieldRed
                else -> DarkOnSurfaceVariant
            }

            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(4.dp))
                    .background(statusColor.copy(alpha = 0.15f))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text(
                    text = statusText,
                    color = statusColor,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(4.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Frames: ${source.framesRead} | Bytes: ${source.bytesRead}",
                fontSize = 11.sp,
                color = DarkOnSurfaceVariant
            )
            Text(
                text = if (source.peakRmsDb != null && source.peakRmsDb > -99f) "${String.format("%.1f", source.peakRmsDb)} dB" else "N/A",
                fontSize = 11.sp,
                color = DarkOnSurfaceVariant
            )
        }

        if (source.error != null) {
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = "Error: ${source.error}",
                fontSize = 10.sp,
                color = ShieldRed.copy(alpha = 0.8f)
            )
        }
    }
}

@Composable
fun TelecomAudioRoutingCard(diagnostics: TelecomAudioDiagnostics?) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.PhoneInTalk,
                    contentDescription = null,
                    tint = ShieldCyan,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Telecom Call Audio Diagnostics",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Live audio route managed directly by Android Telecom framework",
                fontSize = 12.sp,
                color = DarkOnSurfaceVariant
            )
            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("Current Route", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                    Text(
                        diagnostics?.currentRoute ?: "EARPIECE",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = ShieldGreen
                    )
                }
                Column {
                    Text("Call Muted", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                    Text(
                        if (diagnostics?.isMuted == true) "YES" else "NO",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (diagnostics?.isMuted == true) ShieldYellow else DarkOnSurface
                    )
                }
                Column {
                    Text("Mic Muted", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                    Text(
                        if (diagnostics?.isMicrophoneMuted == true) "YES" else "NO",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = if (diagnostics?.isMicrophoneMuted == true) ShieldYellow else DarkOnSurface
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            HorizontalDivider(color = DarkSurfaceVariant.copy(alpha = 0.5f))
            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text("AudioManager Mode", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                    val modeName = when (diagnostics?.audioManagerMode) {
                        2 -> "MODE_IN_CALL (2)"
                        3 -> "MODE_IN_COMMUNICATION (3)"
                        1 -> "MODE_RINGTONE (1)"
                        0 -> "MODE_NORMAL (0)"
                        else -> "MODE_${diagnostics?.audioManagerMode ?: 0}"
                    }
                    Text(
                        modeName,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = DarkOnSurface
                    )
                }
                Column {
                    Text("Supported Routes", fontSize = 11.sp, color = DarkOnSurfaceVariant)
                    Text(
                        diagnostics?.supportedRoutes?.joinToString(", ")?.ifEmpty { "EARPIECE" } ?: "EARPIECE",
                        fontSize = 12.sp,
                        color = DarkOnSurface
                    )
                }
            }
        }
    }
}

@Composable
fun UplinkInvestigationCard(diagnostics: TelecomAudioDiagnostics?) {
    if (diagnostics == null) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.Mic,
                    contentDescription = "Uplink",
                    tint = ShieldYellow,
                    modifier = Modifier.size(20.dp)
                )
                Text(
                    text = "Uplink (Microphone) Investigation",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Physical device test required to verify remote party can hear speech.",
                fontSize = 12.sp,
                color = DarkOnSurfaceVariant
            )
            Spacer(modifier = Modifier.height(12.dp))

            val uplinkColor = when (diagnostics.uplinkStatus) {
                "VERIFIED" -> ShieldGreen
                "UNVERIFIED" -> ShieldYellow
                "MUTED_BY_TELECOM", "MUTED_BY_AUDIOMANAGER" -> ShieldRed
                "SUSPECT_MODE_NORMAL" -> ShieldYellow
                else -> DarkOnSurface
            }

            DiagnosticRow("Uplink Status", diagnostics.uplinkStatus, uplinkColor)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = diagnostics.uplinkNotes.ifEmpty { "No uplink data" },
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
                color = DarkOnSurfaceVariant,
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(DarkBackground)
                    .padding(8.dp)
            )
        }
    }
}

@Composable
fun DisconnectCauseCard(lastDisconnectInfo: LastDisconnectInfo?) {
    if (lastDisconnectInfo == null) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.CallEnd,
                    contentDescription = "Disconnect",
                    tint = ShieldRed,
                    modifier = Modifier.size(20.dp)
                )
                Text(
                    text = "Last Disconnect Cause",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            }
            Spacer(modifier = Modifier.height(12.dp))

            val causeCodeStr = when (lastDisconnectInfo.code) {
                0 -> "UNKNOWN (0)"
                1 -> "ERROR (1)"
                2 -> "LOCAL (2)"
                3 -> "REMOTE (3)"
                4 -> "CANCELED (4)"
                5 -> "MISSED (5)"
                6 -> "REJECTED (6)"
                7 -> "BUSY (7)"
                8 -> "RESTRICTED (8)"
                9 -> "OTHER (9)"
                else -> "CODE_${lastDisconnectInfo.code}"
            }

            DiagnosticRow("Cause Code", causeCodeStr, ShieldRed)
            if (lastDisconnectInfo.label.isNotEmpty()) {
                DiagnosticRow("Label", lastDisconnectInfo.label, DarkOnSurface)
            }
            if (lastDisconnectInfo.description.isNotEmpty()) {
                DiagnosticRow("Description", lastDisconnectInfo.description, DarkOnSurface)
            }
            if (lastDisconnectInfo.reason.isNotEmpty()) {
                DiagnosticRow("Reason", lastDisconnectInfo.reason, DarkOnSurface)
            }

            Spacer(modifier = Modifier.height(8.dp))
            HorizontalDivider(color = DarkSurfaceVariant.copy(alpha = 0.5f))
            Spacer(modifier = Modifier.height(8.dp))

            DiagnosticRow(
                "Local Disconnect Requested",
                if (lastDisconnectInfo.localDisconnectRequested) "YES" else "NO",
                if (lastDisconnectInfo.localDisconnectRequested) ShieldYellow else DarkOnSurface
            )
            DiagnosticRow(
                "endActiveCall() Invoked",
                if (lastDisconnectInfo.endActiveCallInvoked) "YES" else "NO",
                if (lastDisconnectInfo.endActiveCallInvoked) ShieldYellow else DarkOnSurface
            )
            DiagnosticRow(
                "Alert Action Triggered",
                if (lastDisconnectInfo.alertActionTriggered) "YES" else "NO",
                if (lastDisconnectInfo.alertActionTriggered) ShieldRed else DarkOnSurface
            )
        }
    }
}

@Composable
fun DiagnosticRow(label: String, value: String, valueColor: Color) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, fontSize = 12.sp, color = DarkOnSurfaceVariant)
        Text(
            value,
            fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold,
            color = valueColor
        )
    }
}

@Composable
fun SileroSelfTestCard(
    result: ControlledTestResult?,
    isRunning: Boolean,
    onRunTest: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Science,
                        contentDescription = "Test",
                        tint = ShieldCyan,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Silero VAD Controlled Self-Test",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Section 8 Isolated Harness: Runs fresh ONNX session on Silence, 440Hz Tone, and Verified 16kHz Speech. Does NOT touch live call state.",
                fontSize = 11.sp,
                color = DarkOnSurfaceVariant,
                lineHeight = 15.sp
            )
            Spacer(modifier = Modifier.height(12.dp))

            Button(
                onClick = onRunTest,
                enabled = !isRunning,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = ShieldCyan,
                    disabledContainerColor = DarkSurfaceVariant
                ),
                shape = RoundedCornerShape(10.dp)
            ) {
                if (isRunning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = Color.Black,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Running Isolated Self-Test...", color = Color.Black, fontWeight = FontWeight.Bold)
                } else {
                    Icon(Icons.Default.PlayArrow, contentDescription = null, tint = Color.Black)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Run Silero VAD Self-Test", color = Color.Black, fontWeight = FontWeight.Bold)
                }
            }

            if (result != null) {
                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider(color = DarkSurfaceVariant.copy(alpha = 0.5f))
                Spacer(modifier = Modifier.height(8.dp))

                val verdictColor = if (result.speechDivergesFromSilence) ShieldGreen else ShieldYellow
                DiagnosticRow("Verdict", if (result.speechDivergesFromSilence) "VERIFIED" else "ATTENTION", verdictColor)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = result.verdict,
                    fontSize = 11.sp,
                    color = verdictColor,
                    fontWeight = FontWeight.Medium
                )

                Spacer(modifier = Modifier.height(8.dp))
                FixtureResultRow(result.silenceResult)
                Spacer(modifier = Modifier.height(4.dp))
                FixtureResultRow(result.nonSpeechControlResult)
                Spacer(modifier = Modifier.height(4.dp))
                FixtureResultRow(result.speechFixtureResult)

                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = result.detailedNotes,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    color = DarkOnSurfaceVariant,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(DarkBackground)
                        .padding(8.dp)
                )
            }
        }
    }
}

@Composable
private fun FixtureResultRow(fixture: SingleFixtureTestResult) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .background(DarkBackground.copy(alpha = 0.5f))
            .padding(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(fixture.fixtureName, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = ShieldCyan)
            Text(
                "maxProb: ${String.format("%.4f", fixture.maxProb)}",
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                color = if (fixture.maxProb >= 0.5f) ShieldGreen else DarkOnSurface
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("RMS: ${String.format("%.5f", fixture.audioMetrics.rms)} | Peak: ${String.format("%.4f", fixture.audioMetrics.peakAbs)}", fontSize = 10.sp, color = DarkOnSurfaceVariant)
            Text("meanProb: ${String.format("%.4f", fixture.meanProb)}", fontSize = 10.sp, color = DarkOnSurfaceVariant)
        }
    }
}

@Composable
fun ExperimentDCard(
    isCallActive: Boolean,
    context: android.content.Context,
    coroutineScope: kotlinx.coroutines.CoroutineScope
) {
    var diagResult by remember { mutableStateOf<StandaloneMicDiagnosticResult?>(null) }
    var isRunningDiag by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = "Experiment D",
                    tint = ShieldYellow,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Experiment D: Standalone Mic Diagnostic",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Opens fresh AudioRecord (NO call active). Captures 3s of mic audio. Measures raw Short + Float metrics with conversion verification. Does NOT invoke VAD/AASIST/ECAPA.",
                fontSize = 11.sp,
                color = DarkOnSurfaceVariant,
                lineHeight = 15.sp
            )
            Spacer(modifier = Modifier.height(12.dp))

            Button(
                onClick = {
                    if (!isRunningDiag) {
                        isRunningDiag = true
                        coroutineScope.launch(Dispatchers.IO) {
                            val adapter = com.voiceshield.audio.CellularAcousticAudioAdapter(context)
                            val res = adapter.runStandaloneDiagnosticCapture(context)
                            diagResult = res
                            isRunningDiag = false
                        }
                    }
                },
                enabled = !isRunningDiag && !isCallActive,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isCallActive) DarkSurfaceVariant else ShieldYellow,
                    disabledContainerColor = DarkSurfaceVariant
                ),
                shape = RoundedCornerShape(10.dp)
            ) {
                if (isRunningDiag) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = Color.Black,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Capturing Audio (3s)...", color = Color.Black, fontWeight = FontWeight.Bold)
                } else if (isCallActive) {
                    Icon(Icons.Default.Lock, contentDescription = null, tint = DarkOnSurfaceVariant)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Disabled During Call (Use Live Trace)", color = DarkOnSurfaceVariant, fontWeight = FontWeight.Bold)
                } else {
                    Icon(Icons.Default.PlayArrow, contentDescription = null, tint = Color.Black)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Run Experiment D (No-Call Mic Test)", color = Color.Black, fontWeight = FontWeight.Bold)
                }
            }

            if (diagResult != null) {
                val result = diagResult!!
                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider(color = DarkSurfaceVariant.copy(alpha = 0.5f))
                Spacer(modifier = Modifier.height(8.dp))

                if (result.errorMessage != null) {
                    DiagnosticRow("Error", result.errorMessage!!, ShieldRed)
                } else {
                    DiagnosticRow("Duration", "${result.durationMs}ms", DarkOnSurface)
                    DiagnosticRow("Frames Read", "${result.framesRead}", DarkOnSurface)
                    DiagnosticRow("Total Samples", "${result.totalSamplesRead}", DarkOnSurface)

                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "RAW SHORT PCM",
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        color = ShieldCyan.copy(alpha = 0.7f),
                        letterSpacing = 0.5.sp
                    )
                    val rs = result.rawShortMetrics
                    DiagnosticRow("RMS", String.format("%.1f", rs.rmsShort), DarkOnSurface)
                    DiagnosticRow("Peak", "${rs.peakAbsShort}", DarkOnSurface)
                    DiagnosticRow("Min / Max", "${rs.minShort} / ${rs.maxShort}", DarkOnSurface)
                    DiagnosticRow("NonZero", "${rs.nonZeroCount} / ${rs.sampleCount} (${String.format("%.1f", 100f - rs.zeroPercent)}%)", DarkOnSurface)

                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "FLOAT32 CONVERSION",
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        color = ShieldCyan.copy(alpha = 0.7f),
                        letterSpacing = 0.5.sp
                    )
                    DiagnosticRow("Float RMS", String.format("%.6f", result.floatMetrics.rms), DarkOnSurface)
                    DiagnosticRow("Float Peak", String.format("%.6f", result.floatMetrics.peakAbs), DarkOnSurface)
                    DiagnosticRow("Expected Float RMS", String.format("%.6f", rs.expectedFloatRms), DarkOnSurface)
                    DiagnosticRow("Conversion Error", String.format("%.8f", rs.conversionError), DarkOnSurface)

                    Spacer(modifier = Modifier.height(6.dp))
                    val matchColor = when (result.shortFloatMatch) {
                        "MATCH_SIGNAL_ALIVE" -> ShieldGreen
                        "MATCH_BOTH_ZERO" -> ShieldYellow
                        "MISMATCH_CONVERSION_DEFECT" -> ShieldRed
                        else -> DarkOnSurface
                    }
                    DiagnosticRow("Short→Float Match", result.shortFloatMatch, matchColor)
                }
            }

            Spacer(modifier = Modifier.navigationBarsPadding().height(24.dp))
        }
    }
}
