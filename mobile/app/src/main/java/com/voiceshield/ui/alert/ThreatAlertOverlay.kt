/**
 * ThreatAlertOverlay — Non-intrusive alert modal displayed during active calls
 * when accumulated AI evidence indicates potential voice cloning or impersonation.
 *
 * User remains in control:
 * - [ Continue Carefully ]
 * - [ Verify Identity ]
 * - [ End Call ]
 */
package com.voiceshield.ui.alert

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.voiceshield.security.AlertManager
import com.voiceshield.shared.ThreatState
import com.voiceshield.ui.theme.*

@Composable
fun ThreatAlertOverlay(
    threatState: ThreatState,
    onAction: (AlertManager.UserAction) -> Unit
) {
    Dialog(
        onDismissRequest = { onAction(AlertManager.UserAction.DISMISSED) },
        properties = DialogProperties(dismissOnBackPress = true, dismissOnClickOutside = false)
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(containerColor = DarkSurface),
            shape = RoundedCornerShape(20.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Warning Icon Header
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(RoundedCornerShape(28.dp))
                        .background(ShieldRed.copy(alpha = 0.15f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Warning,
                        contentDescription = "Threat Alert",
                        tint = ShieldRed,
                        modifier = Modifier.size(32.dp)
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "POSSIBLE VOICE IMPERSONATION",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = ShieldRed,
                    textAlign = TextAlign.Center,
                    letterSpacing = 0.5.sp
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Evidence explanation
                Text(
                    text = threatState.reason,
                    style = MaterialTheme.typography.bodyMedium,
                    color = DarkOnSurface,
                    textAlign = TextAlign.Center,
                    lineHeight = 20.sp
                )

                if (threatState.avgSpoofScore != null) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Confidence: ${String.format("%.1f", threatState.avgSpoofScore * 100)}% synthetic indicators across ${threatState.windowsAnalyzed} analyzed windows",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurfaceVariant,
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Action Buttons
                // 1. Verify Identity
                Button(
                    onClick = { onAction(AlertManager.UserAction.VERIFY_IDENTITY) },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = ShieldCyan),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = null,
                        tint = Color.Black,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Verify Caller Identity", color = Color.Black, fontWeight = FontWeight.Bold)
                }

                Spacer(modifier = Modifier.height(10.dp))

                // 2. End Call
                OutlinedButton(
                    onClick = { onAction(AlertManager.UserAction.END_CALL) },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = ShieldRed),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CallEnd,
                        contentDescription = null,
                        tint = ShieldRed,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("End Call Immediately", fontWeight = FontWeight.SemiBold)
                }

                Spacer(modifier = Modifier.height(6.dp))

                // 3. Continue Carefully
                TextButton(
                    onClick = { onAction(AlertManager.UserAction.CONTINUE_CAREFULLY) },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Continue Carefully", color = DarkOnSurfaceVariant, fontSize = 13.sp)
                }
            }
        }
    }
}
