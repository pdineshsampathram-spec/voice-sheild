/**
 * VoiceShieldCallScreeningService — Screens incoming calls.
 *
 * Extends Android's CallScreeningService to evaluate incoming calls
 * before the user sees them. Runs FAST — no heavy AI inference here.
 *
 * CRITICAL: The screening service must NOT pretend to perform voice
 * analysis before caller audio exists. Voice-clone detection begins
 * only after a legitimate audio source becomes available.
 */
package com.voiceshield.telecom

import android.os.Build
import android.telecom.Call
import android.telecom.CallScreeningService
import android.util.Log

class VoiceShieldCallScreeningService : CallScreeningService() {

    companion object {
        private const val TAG = "VoiceShieldScreen"
    }

    override fun onScreenCall(callDetails: Call.Details) {
        val number = callDetails.handle?.schemeSpecificPart ?: "Unknown"
        val isIncoming = callDetails.callDirection == Call.Details.DIRECTION_INCOMING

        Log.i(TAG, "Screening call: number=$number, incoming=$isIncoming")

        // Quick local checks only — no heavy inference.
        // Future: integrate caller reputation DB, known-spam lists.
        val response = buildScreeningResponse(callDetails)

        respondToCall(callDetails, response)
    }

    /**
     * Build the screening response based on local checks.
     *
     * IMPORTANT: This must respond quickly. Android may time out
     * the screening service if it takes too long.
     *
     * Voice analysis is NOT done here — it requires actual audio
     * from an active call, which doesn't exist during screening.
     */
    private fun buildScreeningResponse(callDetails: Call.Details): CallResponse {
        val builder = CallResponse.Builder()

        // Default: allow all calls through. VoiceShield monitors DURING
        // the call, not before it. Pre-call blocking would require a
        // reputation database which is out of scope for the prototype.
        builder.setDisallowCall(false)
        builder.setRejectCall(false)

        // Don't skip the call log — we want call history for review
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            builder.setSkipCallLog(false)
        }

        // Don't skip notifications — user should see incoming calls
        builder.setSkipNotification(false)

        // Future enhancement: if caller is in known-spoof database,
        // could silence the call with:
        // builder.setSilenceCall(true)

        Log.d(TAG, "Screening result: ALLOW (default policy)")

        return builder.build()
    }
}
