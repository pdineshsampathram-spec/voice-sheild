/**
 * VoiceShield Theme — Material3 dark theme with security-focused palette.
 *
 * Colors chosen to convey trust, security, and clear threat indication.
 */
package com.voiceshield.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// ─── Shield Colors ──────────────────────────────────────────────────

val ShieldCyan = Color(0xFF00E5FF)
val ShieldCyanDark = Color(0xFF00B8D4)
val ShieldGreen = Color(0xFF00E676)
val ShieldYellow = Color(0xFFFFD600)
val ShieldOrange = Color(0xFFFF9100)
val ShieldRed = Color(0xFFFF1744)
val ShieldPurple = Color(0xFF7C4DFF)

val DarkBackground = Color(0xFF0A0E14)
val DarkSurface = Color(0xFF111822)
val DarkSurfaceVariant = Color(0xFF1A2332)
val DarkOnSurface = Color(0xFFE0E6ED)
val DarkOnSurfaceVariant = Color(0xFF8B95A5)

private val DarkColorScheme = darkColorScheme(
    primary = ShieldCyan,
    onPrimary = Color.Black,
    primaryContainer = ShieldCyanDark,
    onPrimaryContainer = Color.White,
    secondary = ShieldPurple,
    onSecondary = Color.White,
    tertiary = ShieldGreen,
    background = DarkBackground,
    onBackground = DarkOnSurface,
    surface = DarkSurface,
    onSurface = DarkOnSurface,
    surfaceVariant = DarkSurfaceVariant,
    onSurfaceVariant = DarkOnSurfaceVariant,
    error = ShieldRed,
    onError = Color.White
)

@Composable
fun VoiceShieldTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        typography = Typography(),
        content = content
    )
}

/**
 * Map ThreatLevel to display color.
 */
fun threatLevelColor(level: com.voiceshield.shared.ThreatLevel): Color = when (level) {
    com.voiceshield.shared.ThreatLevel.MONITORING -> DarkOnSurfaceVariant
    com.voiceshield.shared.ThreatLevel.NORMAL -> ShieldGreen
    com.voiceshield.shared.ThreatLevel.ELEVATED -> ShieldYellow
    com.voiceshield.shared.ThreatLevel.HIGH_RISK -> ShieldOrange
    com.voiceshield.shared.ThreatLevel.CRITICAL -> ShieldRed
    com.voiceshield.shared.ThreatLevel.INSUFFICIENT_EVIDENCE -> ShieldCyan
    com.voiceshield.shared.ThreatLevel.MODEL_UNAVAILABLE -> ShieldYellow
    com.voiceshield.shared.ThreatLevel.AUDIO_UNAVAILABLE -> DarkOnSurfaceVariant
}

/**
 * Threat level display text.
 */
fun threatLevelLabel(level: com.voiceshield.shared.ThreatLevel): String = when (level) {
    com.voiceshield.shared.ThreatLevel.MONITORING -> "MONITORING"
    com.voiceshield.shared.ThreatLevel.NORMAL -> "NORMAL"
    com.voiceshield.shared.ThreatLevel.ELEVATED -> "ELEVATED"
    com.voiceshield.shared.ThreatLevel.HIGH_RISK -> "HIGH RISK"
    com.voiceshield.shared.ThreatLevel.CRITICAL -> "CRITICAL"
    com.voiceshield.shared.ThreatLevel.INSUFFICIENT_EVIDENCE -> "ANALYZING..."
    com.voiceshield.shared.ThreatLevel.MODEL_UNAVAILABLE -> "MODEL UNAVAILABLE"
    com.voiceshield.shared.ThreatLevel.AUDIO_UNAVAILABLE -> "AUDIO UNAVAILABLE"
}
