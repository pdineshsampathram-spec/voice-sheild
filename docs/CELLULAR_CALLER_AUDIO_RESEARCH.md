# Android Cellular Caller-Audio Technical & Forensic Research Report

## Executive Summary

This document details the architectural, technical, and regulatory landscape governing access to cellular caller audio (downlink) on Android. It specifically addresses why direct cellular PCM capture is restricted by the Android Open Source Project (AOSP), what privileges the Android Default Dialer role actually provides, how OEM call recording systems operate, and how VoiceShield's acoustic analysis mechanism functions within these boundaries.

---

## 1. The 4 Tiers of Audio Access in Android

Android categorizes audio capture into four distinct security and architectural tiers:

### Tier 1: Public Third-Party Application APIs (Standard SDK)
- **Audio Sources**: `MediaRecorder.AudioSource.MIC` (1), `CAMCORDER` (5), `VOICE_RECOGNITION` (6), `VOICE_COMMUNICATION` (7), `UNPROCESSED` (9).
- **Permissions Required**: `android.permission.RECORD_AUDIO` (runtime dangerous permission).
- **Behavior During Cellular Calls (`MODE_IN_CALL`)**:
  - Prior to Android 10 (API 29), opening `MIC` during a call could produce silent buffers or throw `ERROR_INVALID_OPERATION` depending on the device audio HAL.
  - Starting in Android 10, AOSP introduced **Concurrent Audio Input Sharing**:
    - If a cellular phone call is active, telephony audio takes absolute priority.
    - If a third-party app captures from `MIC`, AOSP audio policy may deliver silence (zeros) or muted audio to prevent eavesdropping unless acoustic speakerphone audio is physically picked up by the microphone.
  - Third-party apps **cannot** tap into the internal modem baseband or downlink DSP stream directly via Tier 1 APIs.

### Tier 2: Telephony Audio Sources (Restricted SDK)
- **Audio Sources**:
  - `MediaRecorder.AudioSource.VOICE_UPLINK` (2) — user's outgoing voice.
  - `MediaRecorder.AudioSource.VOICE_DOWNLINK` (3) — incoming remote caller voice.
  - `MediaRecorder.AudioSource.VOICE_CALL` (4) — mixed bidirectional voice.
- **Security Requirement**:
  - Requires `android.permission.CAPTURE_AUDIO_OUTPUT`.
  - Protection Level: `signature|privileged`.
  - In AOSP `AudioPolicyManager.cpp` and `AudioService.java`, requests for `VOICE_DOWNLINK` or `VOICE_CALL` check whether the calling UID holds `CAPTURE_AUDIO_OUTPUT` or is UID 1000 (system).
  - If a non-system third-party app attempts to instantiate `AudioRecord` with `VOICE_DOWNLINK` or `VOICE_CALL`, the Android Audio HAL immediately fails initialization (`AudioRecord.STATE_UNINITIALIZED` / `PERMISSION_DENIED`).

### Tier 3: Default Phone App / Default Dialer Role Privileges
- **Role**: `android.app.role.RoleManager.ROLE_DIALER`
- **What Default Dialer Role Grants**:
  - Binding to `android.telecom.InCallService` with `BIND_INCALL_SERVICE` permission.
  - Full authority over call management: answering (`Call.answer()`), rejecting (`Call.reject()`), holding, disconnecting, muting, and audio routing (`setAudioRoute()`).
  - Access to `CallAudioState` (earpiece, speaker, bluetooth route status).
  - Access to Call Screening (`CallScreeningService`), Call Logs, and Phone Accounts.
  - Lockscreen full-screen intent presentation exemption for incoming calls.
- **What Default Dialer Role DOES NOT Grant**:
  - **It does NOT grant `CAPTURE_AUDIO_OUTPUT`**.
  - Google intentionally decoupled *Call Management* from *Raw Audio Interception*.
  - Even if an app is selected as the default system dialer by the user, Android's `PackageManager` enforces that `CAPTURE_AUDIO_OUTPUT` can only be granted to apps signed with the platform platform-key or pre-installed in the `/system/priv-app/` partition.

### Tier 4: OEM / Internal Call Recording Implementations
Many users observe that native dialers on Samsung (OneUI in certain countries), Xiaomi (MIUI/HyperOS), or Pixel (Google Phone in supported regions) can record calls. This occurs via proprietary OEM mechanisms:
- **Samsung**: Samsung's native Phone app (`com.samsung.android.incallui`) is pre-installed in `/system/priv-app/`, signed with Samsung's OS platform key, and communicates directly with Samsung's proprietary RIL / audio HAL (`audio.primary.exynos.so` or `audio.primary.snapdragon.so`) via hidden vendor extensions (`SecAudioSystem`).
- **Google Pixel**: Google Dialer's call recording uses system-level privileged bindings and carrier/regional policy flags (`ro.carrier`, MCC/MNC checks). It operates as a privileged system app with `CAPTURE_AUDIO_OUTPUT`.
- **Conclusion**: The existence of an OEM call recorder does **not** provide an API for third-party default dialers. Android provides zero public AIDL or binder interfaces for third-party apps to access the OEM recorder engine.

---

## 2. Technical Evaluation of Audio Capture Paths

| Audio Path | API Source | Required Permission | Availability to VoiceShield | Telephony Interference Risk | Viability for Real SIM Analysis |
|---|---|---|---|---|---|
| **Direct Cellular Downlink** | `VOICE_DOWNLINK` / `VOICE_CALL` | `CAPTURE_AUDIO_OUTPUT` (`signature\|privileged`) | **BLOCKED** (AOSP security restriction) | Critical (HAL crash if forced) | Zero without root/custom ROM |
| **Acoustic Device Microphone** | `MIC` (16 kHz mono) | `RECORD_AUDIO` (Runtime) | **AVAILABLE** (Opt-in) | Low (isolated from Telecom routing) | **VIABLE** (requires speakerphone) |
| **Voice Communication** | `VOICE_COMMUNICATION` | `RECORD_AUDIO` (Runtime) | **PLATFORM DEPENDENT** | Medium (may trigger VoIP AEC mode) | Diagnostic fallback only |
| **Voice Recognition** | `VOICE_RECOGNITION` | `RECORD_AUDIO` (Runtime) | **PLATFORM DEPENDENT** | Low | Experimental diagnostic only |
| **WebRTC Direct Stream** | WebRTC AudioTrackSink | In-App WebRTC PeerConnection | **FULLY VERIFIED** | None (separate VoIP stream) | Reference / Demo only |

---

## 3. The Acoustic Capture Analysis Mechanism

Because direct cellular baseband downlink is gated behind `signature|privileged` permissions in all stock Android releases (Android 6 through Android 15), VoiceShield implements an **honest acoustic capture path**:

```text
         REAL CELLULAR CALL (TELECOM)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[Remote Caller Audio]      [User Microphone Audio]
        │                           │
        ▼ (Acoustic Egress)         │
[Device Speakerphone]               │
        │                           │
        ▼ (Acoustic Coupling)       │
 ┌──────────────────────────────────┴──────────────────┐
 │           Device Microphone (`AudioSource.MIC`)      │
 └─────────────────────────┬───────────────────────────┘
                           ▼
          [CellularAcousticAudioAdapter]
                           ▼
                 [AudioNormalizer] (16kHz mono)
                           ▼
                 [AudioBuffer] (64,600 samples)
                           ▼
                 [Silero VAD v5 Engine]
                           ▼
                 [AASIST-L Anti-Spoof Engine]
                           ▼
                 [ECAPA-TDNN Speaker Engine]
                           ▼
              [TemporalEvidenceEngine]
                           ▼
                   [RiskEngine]
                           ▼
                🚨 Threat Alert
```

### Critical Acoustic Realities & Constraints:
1. **Speakerphone Prerequisite**: When the phone is held to the user's ear in earpiece mode, the acoustic isolation between the top earpiece speaker and bottom microphone is high. Caller audio rarely reaches the microphone with sufficient SNR. Therefore, acoustic analysis works reliably **only on speakerphone** or with elevated speaker volume.
2. **Signal Mixture**: The captured stream is NOT pure caller downlink; it is a physical acoustic superposition:
   $$\text{Captured}(t) = \alpha \cdot \text{Caller}(t) * h_{\text{room}}(t) + \beta \cdot \text{UserVoice}(t) + \text{AmbientNoise}(t) + \text{Echo}(t)$$
3. **No False Claims**: VoiceShield's UI strictly displays:
   - `AUDIO SOURCE: ACOUSTIC CAPTURE (MIC)`
   - Never claims `CELLULAR_DOWNLINK`.
   - Explicitly notes: *"Acoustic mode captures sound via device microphone. Speakerphone recommended."*
4. **Deterministic Telephony Isolation**:
   - Acoustic capture is **strictly OFF by default**.
   - If capture fails or encounters errors, it releases `AudioRecord` immediately.
   - It **never** alters Telecom routing, never calls `setMode()`, never mutes the call, and never disconnects the cellular call.

---

## 4. Default Dialer Closed-App Lifecycle & Synchronization

### The Android 10+ Background Restriction Challenge
Starting in Android 10, apps cannot launch activities from the background while the user is outside the app or the screen is locked, with specific exemptions:
- An app with the **Default Dialer** role holding `BIND_INCALL_SERVICE` is exempt.
- However, launching an unconditional `startActivity()` while the user is actively using another app (e.g. typing a message) is disruptive and can be blocked by OEM window managers.

### The Standard Supported Platform Pattern
Per Android Telecom CDD (Compatibility Definition Document) and CTS:
1. **Ringing State**:
   - `VoiceShieldInCallService.onCallAdded(call)` receives the incoming call.
   - It posts an ongoing Notification to `voiceshield_incoming_calls` with `NotificationManager.IMPORTANCE_HIGH`.
   - It attaches a `fullScreenIntent` pointing to `MainActivity`.
   - **When Screen is Locked**: Android OS automatically promotes the `fullScreenIntent` to a full-screen window, waking the screen and presenting `IncomingCallScreen` over the keyguard (`showWhenLocked="true"` and `turnScreenOn="true"`).
   - **When Screen is Unlocked**: Android OS displays a high-priority heads-up banner with Answer and Decline actions, allowing the user to answer without losing their current app context.
2. **Active / Disconnected State**:
   - Once answered or rejected, `VoiceShieldInCallService` dismisses the notification.
   - If `MainActivity` is cold-started or brought to the front after a call is already active, `CallSessionManager.activeSession` reconciles immediately, ensuring the UI always reflects authoritative Telecom state.
