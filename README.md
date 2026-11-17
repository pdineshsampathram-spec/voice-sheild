# VoiceShield 🛡️

**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

*Smart India Hackathon 2026 — Problem Statement PS-104*

---

> [!WARNING]
> ### ⚠️ Project Post-Mortem & Critical Telecom Limitation Notice
> **VoiceShield is an ambitious research and engineering prototype that pushed the absolute limits of Android OS internals, Telecom frameworks, and on-device machine learning.**
>
> While the **WebRTC live calling suite**, **browser microphone detection**, **offline audio forensic analyzer**, and **speaker verification pipeline** are fully functional, **direct real-time detection on live cellular phone calls (SIM / telecom carrier calls) is fundamentally blocked by Android's OS security sandbox and telecom architecture.**
>
> We conducted exhaustive low-level engineering, probing Android Audio HAL, `AudioRecord`, `InCallService`, and OEM system hooks. However, modern telecom standards and Android Open Source Project (AOSP) security policies strictly prohibit third-party apps from intercepting cellular downlink audio streams. See [The Cellular Telecom Audio Wall & Post-Mortem](#-the-cellular-telecom-audio-wall--post-mortem) for our full experimental findings, failed alternatives, and technical autopsy.

---

## 🚀 Live Demos & Downloads

- 🌐 **Live WebRTC Web App (Vercel)**: [https://voice-shield-sepia.vercel.app](https://voice-shield-sepia.vercel.app) *(or updated link below)*
- 📦 **Android App Release APK**: [Download VoiceShield Release APK (v1.0.0)](https://github.com/pdineshsampathram-spec/voice-sheild/releases/download/v1.0.0-prototype/app-release.apk)
- 🧪 **Android App Debug APK**: [Download VoiceShield Debug APK (v1.0.0)](https://github.com/pdineshsampathram-spec/voice-sheild/releases/download/v1.0.0-prototype/app-debug.apk)
- 🏷️ **GitHub Release Page**: [VoiceShield v1.0.0 Releases](https://github.com/pdineshsampathram-spec/voice-sheild/releases/tag/v1.0.0-prototype)

---

## 📋 Table of Contents

1. [Overview & Problem Statement](#-overview--problem-statement)
2. [System Architecture](#-system-architecture)
3. [AI & ML Forensic Pipeline](#-aiml-forensic-pipeline)
4. [The Cellular Telecom Audio Wall & Post-Mortem](#-the-cellular-telecom-audio-wall--post-mortem)
   - [The 4 Tiers of Android Audio Access](#the-4-tiers-of-android-audio-access)
   - [Why Default Dialer Role is Decoupled](#why-default-dialer-role-is-decoupled)
   - [Exhaustive Breakdown of Alternatives Tried & Failed](#exhaustive-breakdown-of-alternatives-tried--failed)
   - [The Acoustic Coupling Fallback (Speakerphone Only)](#the-acoustic-coupling-fallback-speakerphone-only)
   - [Engineering Conclusions](#engineering-conclusions)
5. [Working Modules & Capabilities](#-working-modules--capabilities)
   - [1. WebRTC Live Call Protection Room](#1-webrtc-live-call-protection-room)
   - [2. Live Browser Microphone Guard](#2-live-browser-microphone-guard)
   - [3. Recorded Forensic Analysis](#3-recorded-forensic-analysis)
   - [4. Biometric Speaker Enrollment](#4-biometric-speaker-enrollment)
   - [5. Outbound Twilio + ElevenLabs Fraud Simulator](#5-outbound-twilio--elevenlabs-fraud-simulator)
6. [Android Mobile App Architecture](#-android-mobile-app-architecture)
7. [API Reference](#-api-reference)
8. [Getting Started & Installation](#-getting-started--installation)
   - [Prerequisites](#prerequisites)
   - [Backend Setup (FastAPI)](#backend-setup-fastapi)
   - [Frontend Setup (Next.js)](#frontend-setup-nextjs)
   - [Docker Deployment](#docker-deployment)
   - [Android Build](#android-build)
9. [Configuration Reference](#-configuration-reference)
10. [Authors & Attributions](#-authors--attributions)

---

## 🎯 Overview & Problem Statement

With the rapid emergence of generative voice models (e.g., ElevenLabs, VALL-E, OpenVoice, Bark), bad actors can clone a victim’s voice using as little as 3 seconds of audio. In India and across the globe, **voice cloning financial fraud, CEO impersonation scams, and kidnapping ransom hoaxes** over phone calls have surged dramatically.

**Smart India Hackathon 2026 Problem Statement PS-104** challenges teams to design and implement an AI-powered system capable of real-time detection and mitigation of synthetic voice cloning impersonation attacks during live phone calls.

VoiceShield addresses this threat through:
- **Zero-trust multi-modal acoustic forensics** combining Voice Activity Detection (VAD), deepfake spectral artifact detection, and speaker biometric verification.
- **W3C WebRTC live call room** with a real-time AI security observer tap and streaming conversational captions.
- **A fully functional native Android Default Dialer** featuring native call handling, lockscreen UI, on-device ONNX inference, and cellular diagnostic capabilities.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────────────────────┐
                                  │           VOICESHIELD ECOSYSTEM        │
                                  └───────────────────┬────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐        ┌───────────────────────────────┐
│     Next.js 16 Web Client     │        │     FastAPI Forensic Core     │        │     Android Native Dialer     │
│   (React 19 / TypeScript)     │        │     (Python 3.11 / PyTorch)   │        │     (Kotlin / Jetpack / ONNX) │
├───────────────────────────────┤        ├───────────────────────────────┤        ├───────────────────────────────┤
│ • P2P WebRTC Call Room        │        │ • aiortc Media Receiver       │        │ • Custom Telecom Dialer UI    │
│ • Real-Time Observer Tap      │        │ • AudioBus Ingest Pipeline    │        │ • InCallService Binding       │
│ • Live Audio Truth Panel      │◄──────►│ • Faster-Whisper ASR Engine   │        │ • CallScreeningService        │
│ • Forensic File Analyzer      │  HTTP  │ • Spectra-AASIST Anti-Spoof   │        │ • On-Device Silero VAD        │
│ • Biometric Enrollment UI     │  / WS  │ • ECAPA-TDNN Speaker Engine   │        │ • On-Device AASIST-L ONNX     │
│ • Incident Security Log       │ / WebRTC│ • Calibrated Risk Engine     │        │ • On-Device ECAPA-TDNN ONNX   │
│                               │        │ • Twilio + ElevenLabs Bridge  │        │ • Cellular Capability Lab     │
└───────────────────────────────┘        └───────────────────────────────┘        └───────────────────────────────┘
```

### Full-Stack Audio Ingestion & Verification Loop

```
[ Incoming Audio Stream ]
           │
           ▼
[ Canonical 16 kHz Mono Float32 Resampler ]
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                              ▼                              ▼
  [ Silero VAD v5 ]           [ Spectra-AASIST ]               [ ECAPA-TDNN ]
(Voice Activity Detection)    (Anti-Spoofing Detection)     (Speaker Verification)
  • Speech vs. Silence         • 64,600 sample windows        • 192-dim vector embedding
  • Energy & ZCR Gating        • Graph attention network      • Cosine similarity against
                               • Raw logit calibration          enrolled voice profile
           │                              │                              │
           └──────────────────────────────┼──────────────────────────────┘
                                          ▼
                             [ Calibrated Risk Engine ]
                               • Isotonic / Logistic calibration curve
                               • Sliding window score smoothing (5 windows)
                               • Temporal consistency check (3 escalations)
                               • Debounce hysteresis (2.0s hold)
                                          │
                                          ▼
                             [ Forensic Threat Verdict ]
                            🟢 BONAFIDE (Safe)
                            🟡 SUSPICIOUS (Review Needed)
                            🔴 CRITICAL SPOOF (Cloned Voice Alert)
```

---

## 🧠 AI/ML Forensic Pipeline

VoiceShield employs a defense-in-depth machine learning pipeline to ensure sub-second latency and zero-trust verification:

| Model | Checkpoint / Source | Role | Size | Execution Backend |
|---|---|---|---|---|
| **Spectra-AASIST** | `lab260/Spectra-AASIST` | Synthetic & cloned voice detection (AASIST graph architecture) | ~1.2 GB | PyTorch (MPS / CUDA / CPU) |
| **AASIST-L (Mobile)** | Converted ONNX Model | Light anti-spoofing for on-device Android inference | ~1.3 MB | ONNX Runtime Mobile |
| **ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` | 192-dimensional speaker biometric embeddings | ~80 MB | PyTorch & ONNX Runtime |
| **Silero VAD** | `snakers4/silero-vad` (v5) | Voice activity detection, silence gating, chunk slicing | ~2.2 MB | TorchScript & ONNX |
| **Faster-Whisper** | `Systran/faster-whisper-base` | Low-latency live conversational transcription / captions | ~140 MB | CTranslate2 / int8 |

### Logit Calibration Curve
Raw neural network logits from anti-spoofing models can drift across microphone models, bitrates, and background noise. VoiceShield utilizes an empirical calibration table (`backend/models/calibration_v1.json`) generated from bonafide and deepfake speech samples. Logits are mapped to calibrated probabilities $P(\text{spoof} \mid x) \in [0, 1]$ before passing to the risk engine.

---

## 🚫 The Cellular Telecom Audio Wall & Post-Mortem

### The Core Question
> *"Why can't third-party apps detect cloned voices directly during regular cellular SIM calls?"*

This question represents the core technical hurdle that led to this project being marked as a **failed cellular app** despite months of exhaustive engineering.

### The 4 Tiers of Android Audio Access

Android strictly categorizes audio capture into four isolation tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 4: OEM Proprietary Vendor HAL                                     │
│ (Samsung SecAudioSystem, Pixel System Partition)                       │
│ • Private vendor extensions (audio.primary.exynos.so)                  │
│ • Zero public AIDL / Binder APIs for external apps                     │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 3: Default Dialer Role (ROLE_DIALER)                              │
│ • Grants InCallService: answer(), reject(), setAudioRoute()            │
│ • Deliberately DENIES CAPTURE_AUDIO_OUTPUT                             │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Telephony Audio Sources (VOICE_DOWNLINK / VOICE_CALL)         │
│ • Requires CAPTURE_AUDIO_OUTPUT (signature|privileged)                │
│ • Blocked by AOSP AudioPolicyManager for any non-system app            │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 1: Standard Public SDK (MediaRecorder.AudioSource.MIC)            │
│ • Standard RECORD_AUDIO permission                                     │
│ • During cellular calls (MODE_IN_CALL), delivering zeroed/muted audio   │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Tier 1 (Public SDK - `MIC`)**:
   Standard apps use `AudioSource.MIC`. However, starting in Android 10, AOSP enforces **Concurrent Audio Input Sharing**. When a cellular call is active (`MODE_IN_CALL`), the telephony audio path takes absolute hardware priority. Any third-party app attempting to record from `MIC` is delivered **zeros (silence)** by the Android Audio HAL to prevent eavesdropping and spyware.
2. **Tier 2 (Telephony Sources - `VOICE_DOWNLINK`, `VOICE_CALL`)**:
   These sources specifically exist in the Android SDK to tap incoming remote caller audio or the bidirectional mix. However, both sources are protected by:
   ```xml
   android.permission.CAPTURE_AUDIO_OUTPUT
   ```
   Protection level: `signature|privileged`.
   Android's `AudioPolicyManager.cpp` verifies the caller UID. If the app is not signed with the device manufacturer's platform key or pre-installed in `/system/priv-app/`, the Audio HAL immediately aborts with `AudioRecord.STATE_UNINITIALIZED` / `PERMISSION_DENIED`.
3. **Tier 3 (Default Dialer Role - `ROLE_DIALER`)**:
   We built a complete default phone dialer using `InCallService` and `CallScreeningService`. However, **Google intentionally decoupled call management from audio interception**. Holding the default dialer role gives authority over UI, call routing, and answering/rejecting calls, but **does NOT grant `CAPTURE_AUDIO_OUTPUT`**.
4. **Tier 4 (OEM Call Recorders)**:
   Native call recorders on Samsung (OneUI), Xiaomi (MIUI), or Pixel operate through proprietary OEM vendor libraries (`audio.primary.snapdragon.so`, `SecAudioSystem`). They bypass standard Android APIs entirely through internal vendor hooks. Android provides zero public IPC/AIDL interfaces for third-party dialers to hook into this stream.

---

### Exhaustive Breakdown of Alternatives Tried & Failed

Throughout the development of VoiceShield, we engineered, tested, and documented numerous workarounds. Every single one encountered impenetrable hardware or OS security barriers:

| # | Approach Attempted | Implementation | Failure Reason & Observed Behavior |
|---|---|---|---|
| **1** | **Direct Cellular Downlink Capture** | `AudioRecord` instantiated with `AudioSource.VOICE_DOWNLINK` (3) or `VOICE_CALL` (4) | **FAILED**: Throws `SecurityException` or halts at `AudioRecord.STATE_UNINITIALIZED`. Android HAL enforces `CAPTURE_AUDIO_OUTPUT` (`signature|privileged`). |
| **2** | **Voice Communication Mode** | `AudioSource.VOICE_COMMUNICATION` (7) | **FAILED**: Delivered empty/silent buffers during cellular calls, or engaged VoIP Acoustic Echo Cancellation (AEC) that actively stripped out incoming caller speech. |
| **3** | **Voice Recognition Mode** | `AudioSource.VOICE_RECOGNITION` (6) | **FAILED**: Android AudioPolicyManager prioritized the cellular telephony baseband and muted the capture pipeline. |
| **4** | **Default Dialer Role Escalation** | Full implementation of `RoleManager.ROLE_DIALER` + `InCallService` | **FAILED**: Granted phone UI control, answering, and hangup, but Google CTS explicitly forbids granting audio interception privileges to third-party dialers. |
| **5** | **OEM Call Recording File Scraping** | Post-call scanner (`OemCallRecordingDiscoverer.kt`) probing `MIUI/sound_recorder/call_rec/` | **FAILED**: Highly device-dependent (only worked on select older Xiaomi devices), introduces delayed post-call analysis rather than real-time protection, and is blocked by Scoped Storage on Android 11+. |
| **6** | **Telecom Carrier Baseband Interception** | Attempting to access live VoLTE/VoNR radio packets | **FAILED**: Mobile carriers (Jio, Airtel, Vi, etc.) encrypt radio traffic (AES-based VoLTE ciphering). Baseband processors (Qualcomm/MediaTek) route voice through dedicated DSPs completely isolated from the application processor (AP). |
| **7** | **Acoustic Coupling (Speakerphone)** | `CellularAcousticAudioAdapter.kt` capturing room audio via device microphone | **PARTIAL / RESTRICTED**: Only works if the user manually turns on speakerphone at high volume. Handset earpiece mode is completely inaccessible due to acoustic isolation. Signal is corrupted by room acoustics, echo, and the user's own voice. |

---

### The Acoustic Coupling Fallback (Speakerphone Only)

Because direct cellular baseband downlink is locked behind platform keys, the only non-root mechanism VoiceShield could implement on Android was **acoustic capture**:

```
REAL CELLULAR CALL (TELECOM BASEBAND)
               │
               ▼
      [ Device Earpiece ]  ──(Acoustic Isolation: ~40 dB)──>  [ Device Mic ] (SILENCE)
               │
        (User switches to)
               ▼
     [ Device Speakerphone ]
               │
               ▼ (Acoustic Sound in Room)
               │
      [ Device Microphone ] (`AudioSource.MIC`)
               │
               ▼
     [ VoiceShield Acoustic Pipeline ]
```

#### Acoustic Realities:
1. **Earpiece Failure**: When holding the phone to the ear, physical acoustic isolation prevents caller audio from reaching the bottom microphone. The signal-to-noise ratio (SNR) is near zero.
2. **Superposition Distortion**: The captured audio is a physical mixture:
   $$\text{Captured}(t) = \alpha \cdot \text{Caller}(t) * h_{\text{room}}(t) + \beta \cdot \text{UserVoice}(t) + \text{Noise}(t)$$
   This mixture severely corrupts the high-frequency spectral artifacts needed by models like AASIST, causing false positives and unreliable risk scoring.

### Engineering Conclusions

> **Honest Assessment**: A production-ready, real-time voice clone defense app for regular cellular calls cannot exist as a standard third-party Android or iOS application today. Unless Google and Apple introduce a dedicated, privacy-preserving `CallAudioObserver` API (analogous to accessibility or VPN APIs), or telecom carriers deploy server-side deepfake screening in the core IMS network, consumer cellular calls remain fundamentally unprotectable by third-party software.

VoiceShield successfully demonstrates that **the AI forensic technology works brilliantly over WebRTC, VoIP, and digital audio streams**—the barrier is entirely telecom and OS regulatory architecture.

---

## ✨ Working Modules & Capabilities

### 1. WebRTC Live Call Protection Room
- **Path**: `/call/[roomId]`
- Full peer-to-peer WebRTC video/audio call between two users using W3C Perfect Negotiation.
- **AI Observer Tap**: The application clones the remote peer's audio track into an `aiortc` media session on the FastAPI backend without altering call latency.
- **Real-Time Forensic Dashboards**:
  - **Live Audio Truth Panel**: Displays decoded codec, canonical 16 kHz sample rate, packet counter, RMS, and peak energy.
  - **Live Deepfake Threat Gauge**: Evaluates synthetic voice score in 64,600-sample sliding windows.
  - **Biometric Speaker Verification**: Matches caller against enrolled profiles.
  - **Real-Time Conversational Captions**: Low-latency transcription powered by Faster-Whisper.

### 2. Live Browser Microphone Guard
- **Path**: `/live`
- Captures microphone input directly from the browser via AudioWorklet.
- Streams raw 16 kHz PCM over WebSocket (`/ws/voice-analysis`).
- Real-time VAD voice activity gating and synthetic voice threat alerts.

### 3. Recorded Forensic Analysis
- **Path**: `/analyze`
- Upload multi-format audio files (WAV, MP3, M4A, WebM, OGG, FLAC).
- Performs VAD speech segmentation and produces an interactive forensic timeline.
- Inspects each speech segment for synthetic generation artifacts and speaker identity.

### 4. Biometric Speaker Enrollment
- **Path**: `/enroll`
- Records user voice samples or accepts audio uploads.
- Extracts 192-dimensional embeddings via ECAPA-TDNN.
- Stores profiles in SQLite/filesystem for speaker identity verification during calls.

### 5. Outbound Twilio + ElevenLabs Fraud Simulator
- **Path**: `/caller` (Backend API)
- Triggers automated outbound phone calls through Twilio.
- Streams conversational voice generated by ElevenLabs AI agents into the call.
- Provides an automated live fire benchmark to test whether VoiceShield can detect incoming synthetic calls over telephony bridges.

---

## 📱 Android Mobile App Architecture

The Android application is located in `mobile/`:
- **`dialer-core`**: A standalone Android library implementing:
  - `DefaultInCallService`: Authoritative telecom call state listener.
  - `CallScreeningService`: Incoming number verification and call screening.
  - `CallScreenActivity`: Jetpack Compose active call UI, lockscreen presentation (`showWhenLocked`), and audio routing.
- **`mobile/app`**: The application layer integrating on-device AI:
  - `SileroVadEngine.kt`: On-device VAD running via ONNX Runtime Mobile.
  - `AasistLEngine.kt`: Lightweight anti-spoof model for Android inference.
  - `EcapaTdnnEngine.kt`: On-device speaker verification model.
  - `CellularAudioCapabilityManager.kt`: Probes all 5 Android audio sources during active calls and generates diagnostic reports.
  - `OemCallRecordingDiscoverer.kt`: Probes external directories for OEM call recordings post-call.

---

## 🔌 API Reference

### Health & Diagnostics
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status, GPU availability, and model memory states |
| `GET` | `/` | Root service metadata |

### Audio Analysis & Enrollment
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze/upload` | Upload audio file for multi-segment forensic analysis |
| `POST` | `/api/enrollment` | Enroll a new speaker voice profile |
| `GET` | `/api/enrollment/profiles` | List all registered speaker biometric profiles |
| `DELETE` | `/api/enrollment/{id}` | Delete a registered speaker profile |

### WebRTC & Live Call Monitoring
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/monitor/session` | Initialize real-time observer session for a call |
| `POST` | `/api/monitor/offer` | Exchange WebRTC SDP offer/answer with `aiortc` |
| `GET` | `/api/monitor/telemetry` | Inspect live stream telemetry (RMS, peak, frames) |
| `DELETE` | `/api/monitor/session/{id}` | Terminate and clean up observer session |
| `WS` | `/ws/monitor/{session_id}` | Real-time event stream for captions, spoof scores, and risk alerts |
| `WS` | `/ws/voice-analysis` | Binary PCM live mic streaming endpoint |
| `WS` | `/api/rtc/ws/signaling/{room_id}/{peer_id}` | P2P WebRTC signaling channel |

### Telephony Fraud Simulator
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/caller/outbound` | Initiate Twilio + ElevenLabs synthetic call test |
| `GET` | `/caller/config` | Inspect caller simulator configuration |

---

## 🛠️ Getting Started & Installation

### Prerequisites
- **Python**: 3.11+
- **Node.js**: 18+ (tested on Node 20 / 22)
- **Android Studio**: Ladybug / Meerkat (for mobile module)
- **Hardware**: 8 GB+ RAM recommended (for running local PyTorch models)

---

### 1. Backend Setup (FastAPI)

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp ../.env.example ../.env

# Launch the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend initializes models automatically on startup and exposes interactive Swagger documentation at `http://localhost:8000/docs`.

---

### 2. Frontend Setup (Next.js)

```bash
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```
Open `http://localhost:3000` in your browser.

---

### 3. Docker Deployment

To spin up the entire backend stack with dependencies:

```bash
docker-compose up --build
```

---

### 4. Android Build

Open the `mobile/` directory in Android Studio:
1. Ensure Android SDK 34/35 is installed.
2. Build debug APK:
   ```bash
   cd mobile
   ./gradlew assembleDebug
   ```
3. Built APKs will be located at `mobile/app/build/outputs/apk/`.

---

## ⚙️ Configuration Reference

Key parameters configurable in `.env`:

```env
# Server
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:3000,https://voice-shield-sepia.vercel.app

# Device execution (auto, cpu, cuda, mps)
DEVICE=auto

# Detection parameters
SAMPLE_RATE=16000
SPECTRA_WINDOW_SAMPLES=64600
SLIDING_WINDOW_OVERLAP=0.5

# Risk Engine Thresholds (Calibrated)
SPOOF_THRESHOLD_CRITICAL=0.85
SPOOF_THRESHOLD_HIGH=0.70
SPOOF_THRESHOLD_MEDIUM=0.40
SPEAKER_THRESHOLD_MATCH=0.25
RISK_WINDOW_COUNT=5
RISK_ESCALATION_COUNT=3
RISK_DEBOUNCE_SECONDS=2.0
```

---

## 👥 Authors & Attributions

Developed for **Smart India Hackathon 2026 — Problem Statement PS-104**.

- **Lead Developer**: Dinesh Polimera ([@pdineshsampathram-spec](https://github.com/pdineshsampathram-spec))
- **Models & Research Attributions**:
  - *Spectra-AASIST*: `lab260/Spectra-AASIST`
  - *ECAPA-TDNN*: SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`)
  - *Silero VAD*: Snakers4 (`snakers4/silero-vad`)
  - *Faster-Whisper*: Systran / OpenAI Whisper

---

## 📄 License

Educational and Hackathon Research Use — Smart India Hackathon 2026.
