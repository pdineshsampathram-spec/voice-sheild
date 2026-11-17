/**
 * HomeScreen — Main dashboard showing VoiceShield protection status.
 *
 * Displays:
 * - Default Dialer status (ON/OFF)
 * - Call Screening status (ON/OFF)
 * - Voice Analysis readiness (READY/UNAVAILABLE)
 * - Model health (Silero VAD, AASIST-L, ECAPA-TDNN)
 * - Detection engine mode (Development/Privacy)
 */
package com.voiceshield.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.voiceshield.shared.*
import com.voiceshield.ui.theme.*

@Composable
fun HomeScreen(
    isDefaultDialer: Boolean,
    callScreeningEnabled: Boolean,
    engineMode: String,
    modelHealth: Map<String, ModelStatus>,
    detailedHealth: Map<String, DetailedModelHealth> = emptyMap(),
    calibrationAvailable: Boolean,
    onRequestDialerRole: () -> Unit,
    onStartWebRtcCall: (String) -> Unit = {},
    onOpenDiagnostics: () -> Unit = {},
    onOpenPostCallAnalysis: () -> Unit = {}
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Text(
            text = "VoiceShield",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = ShieldCyan
        )
        Text(
            text = "AI-Powered Call Protection",
            style = MaterialTheme.typography.bodyLarge,
            color = DarkOnSurfaceVariant
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Cellular Audio Capability Lab Card
        CellularAudioLabCard(onOpenDiagnostics = onOpenDiagnostics)

        // Post-Call Analysis Card
        PostCallAnalysisCard(onOpenPostCallAnalysis = onOpenPostCallAnalysis)

        // WebRTC Live Call Demo Card
        WebRtcCallCard(onStartCall = onStartWebRtcCall)

        // Protection Status Card
        ShieldStatusCard(
            isDefaultDialer = isDefaultDialer,
            callScreeningEnabled = callScreeningEnabled,
            onRequestDialerRole = onRequestDialerRole
        )

        // Detection Engine Card
        EngineStatusCard(
            engineMode = engineMode,
            calibrationAvailable = calibrationAvailable
        )

        // Model Health Card
        ModelHealthCard(modelHealth = modelHealth)
    }
}

@Composable
fun CellularAudioLabCard(
    onOpenDiagnostics: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.Science,
                    contentDescription = "Capability Lab",
                    tint = ShieldCyan,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text = "Cellular Audio Capability Lab",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = DarkOnSurface
                )
            }

            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Inspect and test real hardware audio capture capabilities during cellular calls. Probe VOICE_CALL, VOICE_DOWNLINK, and mic sources.",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant
            )

            Spacer(modifier = Modifier.height(14.dp))
            OutlinedButton(
                onClick = onOpenDiagnostics,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = ShieldCyan),
                shape = RoundedCornerShape(10.dp)
            ) {
                Icon(Icons.Default.Analytics, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Open Audio Diagnostics", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
fun WebRtcCallCard(
    onStartCall: (String) -> Unit
) {
    var calleeId by remember { mutableStateOf("") }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = "WebRTC Live Call (Demo)",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = DarkOnSurface
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Connect to an agent or remote caller for live voice clone detection.",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant
            )

            Spacer(modifier = Modifier.height(12.dp))

            OutlinedTextField(
                value = calleeId,
                onValueChange = { calleeId = it },
                label = { Text("Peer / Callee ID") },
                placeholder = { Text("e.g. agent-1, test-peer") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = ShieldCyan,
                    unfocusedBorderColor = DarkSurfaceVariant,
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            Button(
                onClick = { onStartCall(calleeId.trim()) },
                enabled = calleeId.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = ShieldCyan,
                    contentColor = DarkBackground,
                    disabledContainerColor = DarkSurfaceVariant,
                    disabledContentColor = DarkOnSurfaceVariant
                )
            ) {
                Icon(
                    imageVector = Icons.Default.Call,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Start WebRTC Call", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
fun ShieldStatusCard(
    isDefaultDialer: Boolean,
    callScreeningEnabled: Boolean,
    onRequestDialerRole: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = "Protection Status",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = DarkOnSurface
            )

            Spacer(modifier = Modifier.height(16.dp))

            StatusRow(
                icon = Icons.Default.Phone,
                label = "Default Dialer",
                status = if (isDefaultDialer) "ON" else "OFF",
                statusColor = if (isDefaultDialer) ShieldGreen else ShieldRed,
                action = if (!isDefaultDialer) onRequestDialerRole else null
            )

            Spacer(modifier = Modifier.height(12.dp))

            StatusRow(
                icon = Icons.Default.Shield,
                label = "Call Screening",
                status = if (callScreeningEnabled) "ON" else "OFF",
                statusColor = if (callScreeningEnabled) ShieldGreen else ShieldYellow
            )

            Spacer(modifier = Modifier.height(12.dp))

            StatusRow(
                icon = Icons.Default.GraphicEq,
                label = "Voice Analysis",
                status = "READY",
                statusColor = ShieldGreen
            )
        }
    }
}

@Composable
fun EngineStatusCard(
    engineMode: String,
    calibrationAvailable: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = "Detection Engine",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = DarkOnSurface
            )

            Spacer(modifier = Modifier.height(16.dp))

            StatusRow(
                icon = Icons.Default.Lock,
                label = "Mode",
                status = engineMode.uppercase(),
                statusColor = if (engineMode == "privacy") ShieldGreen else ShieldCyan
            )

            Spacer(modifier = Modifier.height(12.dp))

            StatusRow(
                icon = Icons.Default.Settings,
                label = "Calibration",
                status = if (calibrationAvailable) "AVAILABLE" else "NOT CALIBRATED",
                statusColor = if (calibrationAvailable) ShieldGreen else ShieldYellow
            )
        }
    }
}

@Composable
fun ModelHealthCard(
    modelHealth: Map<String, ModelStatus>,
    detailedHealth: Map<String, DetailedModelHealth> = emptyMap()
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                text = "Model Health",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = DarkOnSurface
            )

            Spacer(modifier = Modifier.height(16.dp))

            val models = listOf("Silero VAD", "AASIST-L", "ECAPA-TDNN", "Calibration")
            models.forEach { modelName ->
                val detail = detailedHealth[modelName]
                val status = detail?.status ?: (modelHealth[modelName] ?: ModelStatus.UNAVAILABLE)
                val displayStatus = when {
                    status == ModelStatus.READY -> "READY"
                    status == ModelStatus.LOADING -> "INITIALIZING..."
                    status == ModelStatus.NOT_CALIBRATED -> "NOT CALIBRATED"
                    status == ModelStatus.NOT_IMPLEMENTED -> "NOT IMPLEMENTED"
                    detail != null && detail.failureReason != ModelFailureReason.NONE -> "FAILED: ${detail.failureReason.name}"
                    status == ModelStatus.ERROR -> "ERROR"
                    else -> "UNAVAILABLE"
                }
                StatusRow(
                    icon = Icons.Default.Psychology,
                    label = modelName,
                    status = displayStatus,
                    statusColor = when (status) {
                        ModelStatus.READY -> ShieldGreen
                        ModelStatus.LOADING -> ShieldCyan
                        ModelStatus.NOT_CALIBRATED -> ShieldYellow
                        ModelStatus.ERROR -> ShieldRed
                        ModelStatus.UNAVAILABLE -> DarkOnSurfaceVariant
                        ModelStatus.NOT_IMPLEMENTED -> DarkOnSurfaceVariant
                    }
                )
                if (modelName != models.last()) {
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
fun StatusRow(
    icon: ImageVector,
    label: String,
    status: String,
    statusColor: Color,
    action: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = DarkOnSurfaceVariant,
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = DarkOnSurface,
            modifier = Modifier.weight(1f)
        )

        if (action != null) {
            TextButton(onClick = action) {
                Text("Enable", color = ShieldCyan, fontSize = 12.sp)
            }
        }

        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(4.dp))
                .background(statusColor.copy(alpha = 0.15f))
                .padding(horizontal = 8.dp, vertical = 4.dp)
        ) {
            Text(
                text = status,
                color = statusColor,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun PostCallAnalysisCard(
    onOpenPostCallAnalysis: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.Analytics,
                    contentDescription = null,
                    tint = ShieldPurple,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "POST-CALL ANALYSIS",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface,
                    letterSpacing = 1.sp
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Analyze a call recording for synthetic voice evidence. " +
                "Select an OEM call recording after the call ends.",
                style = MaterialTheme.typography.bodySmall,
                color = DarkOnSurfaceVariant
            )
            Spacer(modifier = Modifier.height(12.dp))
            Button(
                onClick = onOpenPostCallAnalysis,
                colors = ButtonDefaults.buttonColors(containerColor = ShieldPurple),
                shape = RoundedCornerShape(8.dp)
            ) {
                Icon(Icons.Default.FolderOpen, null, tint = Color.White, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("OPEN POST-CALL ANALYSIS", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}
