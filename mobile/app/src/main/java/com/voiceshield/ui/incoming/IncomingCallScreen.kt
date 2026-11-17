/**
 * IncomingCallScreen — Full-screen UI displayed when receiving an incoming call.
 *
 * Displays:
 * - Caller phone number / contact name
 * - VoiceShield Call Screening protection badge
 * - Answer & Reject buttons
 * - Pre-call metadata risk indicator (when available)
 */
package com.voiceshield.ui.incoming

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.voiceshield.ui.theme.*

@Composable
fun IncomingCallScreen(
    phoneNumber: String?,
    callerName: String? = null,
    screeningStatus: String = "SCREENED BY VOICESHIELD",
    isConnecting: Boolean = false,
    onAnswer: () -> Unit,
    onReject: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        // Top: Call Screening Header
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(top = 40.dp)
        ) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = ShieldCyan.copy(alpha = 0.12f)
                ),
                shape = RoundedCornerShape(20.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Shield,
                        contentDescription = "Shield",
                        tint = ShieldCyan,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = screeningStatus,
                        color = ShieldCyan,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(48.dp))

            // Caller Avatar / Icon Placeholder
            Box(
                modifier = Modifier
                    .size(96.dp)
                    .clip(CircleShape)
                    .background(DarkSurfaceVariant),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Call,
                    contentDescription = null,
                    tint = DarkOnSurfaceVariant,
                    modifier = Modifier.size(48.dp)
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Caller Info
            Text(
                text = callerName ?: phoneNumber ?: "Incoming Call",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )

            if (callerName != null && phoneNumber != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = phoneNumber,
                    style = MaterialTheme.typography.bodyLarge,
                    color = DarkOnSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "VoiceShield Live Monitoring Ready on Answer",
                style = MaterialTheme.typography.bodySmall,
                color = ShieldGreen,
                fontSize = 12.sp
            )
        }

        // Bottom: Answer & Reject Action Buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 32.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Reject Button
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                IconButton(
                    onClick = onReject,
                    modifier = Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(ShieldRed)
                ) {
                    Icon(
                        imageVector = Icons.Default.CallEnd,
                        contentDescription = "Decline Call",
                        tint = Color.White,
                        modifier = Modifier.size(32.dp)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Decline",
                    color = DarkOnSurfaceVariant,
                    fontSize = 12.sp
                )
            }

            // Answer Button
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                IconButton(
                    onClick = { if (!isConnecting) onAnswer() },
                    enabled = !isConnecting,
                    modifier = Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(if (isConnecting) ShieldGreen.copy(alpha = 0.5f) else ShieldGreen)
                ) {
                    if (isConnecting) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(32.dp),
                            color = Color.Black,
                            strokeWidth = 3.dp
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.Call,
                            contentDescription = "Answer Call",
                            tint = Color.Black,
                            modifier = Modifier.size(32.dp)
                        )
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = if (isConnecting) "Connecting..." else "Answer",
                    color = DarkOnSurfaceVariant,
                    fontSize = 12.sp
                )
            }
        }
    }
}
