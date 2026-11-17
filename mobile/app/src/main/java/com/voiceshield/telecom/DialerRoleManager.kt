/**
 * DialerRoleManager — Handles requesting the default dialer role.
 *
 * Uses RoleManager.ROLE_DIALER on Android Q+ to request the user's
 * explicit approval to become the default dialer. Does NOT attempt
 * to bypass Android system roles.
 */
package com.voiceshield.telecom

import android.app.Activity
import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.telecom.TelecomManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class DialerRoleManager(private val context: Context) {

    companion object {
        const val REQUEST_CODE_SET_DEFAULT_DIALER = 1001
    }

    private val _isDefaultDialer = MutableStateFlow(false)
    val isDefaultDialer: StateFlow<Boolean> = _isDefaultDialer.asStateFlow()

    /**
     * Check whether VoiceShield is currently the default dialer.
     */
    fun checkDefaultDialerStatus() {
        _isDefaultDialer.value = isCurrentlyDefaultDialer()
    }

    /**
     * Returns true if VoiceShield is currently the default dialer app.
     */
    fun isCurrentlyDefaultDialer(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = context.getSystemService(RoleManager::class.java)
            roleManager?.isRoleHeld(RoleManager.ROLE_DIALER) == true
        } else {
            val telecomManager = context.getSystemService(TelecomManager::class.java)
            telecomManager?.defaultDialerPackage == context.packageName
        }
    }

    /**
     * Request the default dialer role from the user.
     * The user must explicitly approve this.
     *
     * @param activity The activity to launch the role request from.
     * @return The intent to launch, or null if already the default dialer.
     */
    fun createRequestIntent(activity: Activity): Intent? {
        if (isCurrentlyDefaultDialer()) {
            _isDefaultDialer.value = true
            return null
        }

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = activity.getSystemService(RoleManager::class.java)
            roleManager?.createRequestRoleIntent(RoleManager.ROLE_DIALER)
        } else {
            // Fallback for pre-Q devices
            Intent(TelecomManager.ACTION_CHANGE_DEFAULT_DIALER).apply {
                putExtra(TelecomManager.EXTRA_CHANGE_DEFAULT_DIALER_PACKAGE_NAME, context.packageName)
            }
        }
    }

    /**
     * Handle the result of the dialer role request.
     * Call this from Activity.onActivityResult().
     */
    fun onRoleRequestResult(resultCode: Int) {
        _isDefaultDialer.value = if (resultCode == Activity.RESULT_OK) {
            isCurrentlyDefaultDialer()
        } else {
            false
        }
    }
}
