# VoiceShield Mobile — System Architecture & Implementation

Built for **Smart India Hackathon 2026 — PS-104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**.

---

## 1. Core Principle: Separation of Concerns

```
VOICESHIELD MOBILE
│
├── CALL MANAGEMENT ACCESS (Telephony)
│   ├── Default Dialer Role (RoleManager.ROLE_DIALER)
│   ├── InCallService (VoiceShieldInCallService)
│   ├── CallScreeningService (VoiceShieldCallScreeningService)
│   └── CallSessionManager
│
└── CALL AUDIO ACCESS (Audio Source Adapter Layer)
    ├── WebRtcAudioAdapter (Direct media VoIP/WebRTC calls — GUARANTEED AUDIO)
    ├── PlatformSupportedAudioAdapter (Future platform/OEM APIs)
    └── NoAudioAdapter (SIM calls — HONEST UNAVAILABLE STATE)
```

### Critical Honesty Rule
Becoming Android's default dialer **does not** provide unrestricted raw PCM access to the remote party of cellular/SIM calls. VoiceShield explicitly distinguishes:
- **Call Lifecycle Integration**: Always available for incoming, active, holding, and disconnecting calls.
- **Remote Voice Analysis**: Available ONLY when a legitimate, approved audio stream exists.
- If remote audio cannot be accessed, the UI clearly displays:
  ```
  Call Protection Active
  Call Lifecycle Monitoring: ON
  Voice Analysis: NOT AVAILABLE FOR THIS AUDIO PATH
  ```
  The system **never** simulates or fakes safety metrics.

---

## 2. Detection Engine Architecture

```
                    VoiceShield
                        │
            VoiceShieldDetectionEngine
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Development Mode             Privacy Mode
(PythonBackendDetectionEngine) (OnDeviceDetectionEngine)
          ▼                           ▼
   FastAPI WebSocket             ONNX Runtime
   (Spectra-AASIST,              (AASIST-L, Silero VAD,
    ECAPA-TDNN, Silero)           ECAPA-TDNN)
```

Both engines implement the identical contract:
```kotlin
interface VoiceShieldDetectionEngine {
    suspend fun initialize(): EngineStatus
    suspend fun analyzeWindow(audio: AudioFrame): WindowResult?
    fun getModelHealth(): Map<String, ModelStatus>
    fun isCalibrated(): Boolean
    fun getMode(): EngineMode
    fun shutdown()
}
```
This enables direct side-by-side evaluation of the on-device AASIST-L against the server-side Spectra-AASIST on the exact same audio stream.

---

## 3. On-Device AI Pipeline

```
Audio Stream (16kHz mono)
         │
         ▼
   AudioBuffer (64,600 samples, 50% overlap)
         │
         ▼
    Silero VAD (silero_vad.onnx)
         │
    Speech Detected?
    ├── NO  ──> WindowResult(vad=NoSpeech, antiSpoof=null)
    └── YES ──> Continue
         │
         ├──> AASIST-L (aasist_l.onnx, 85k params, ~197ms latency)
         │        └──> AntiSpoofResult(spoofScore, bonaFideScore)
         │
         └──> ECAPA-TDNN (ecapa_tdnn.onnx, 80-dim log-mel filterbanks)
                  └──> SpeakerResult(similarity, match)
         │
         ▼
   TemporalEvidenceEngine (Multi-window sliding fusion)
         │
         ▼
   RiskEngine (8 Explicit Threat States)
         │
         ▼
    User UI / ThreatAlertOverlay
```

---

## 4. Threat States

| Threat State | Meaning |
| :--- | :--- |
| `MONITORING` | Call active, collecting initial speech samples |
| `NORMAL` | Multi-window evidence shows natural human speech |
| `ELEVATED` | Minor or transient synthetic indicators observed |
| `HIGH_RISK` | Consistent synthetic features detected across multiple segments |
| `CRITICAL` | Persistent high anti-spoof evidence; caller identity suspicious |
| `INSUFFICIENT_EVIDENCE` | Fewer than 3 analyzed speech windows accumulated |
| `MODEL_UNAVAILABLE` | Local model failed to load or runtime error occurred |
| `AUDIO_UNAVAILABLE` | Remote audio stream is not accessible on this device path |

---

## 5. Calibration Methodology

VoiceShield enforces **zero invented thresholds**:
- If a verified evaluation dataset has not been executed, the calibration artifact declares:
  ```json
  "status": "CALIBRATION_NOT_AVAILABLE"
  ```
- The UI reports raw anti-spoof model scores and explicitly informs the user that probabilistic risk mapping is pending calibration.
- Once calibrated on a dataset (e.g. ASVspoof2019 LA), the artifact records model version, sample count, EER, and optimal operating point thresholds.

---

## 6. Open Source Reuse

| Component | Upstream Repository | License | Purpose |
| :--- | :--- | :--- | :--- |
| `dialer-core` | `fayyaztech/dialer-core` | Apache 2.0 | Telephony InCallService and CallScreening |
| `AASIST-L` | `SpeechAntiSpoofingBenchmarks/AASIST-L` | MIT | 85k-parameter lightweight anti-spoof model |
| `ECAPA-TDNN` | `astrolabos/ecapa-tdnn` | Apache 2.0 | Speaker embedding extraction |
| `Silero VAD` | `snakers4/silero-vad` | MIT | Real-time speech activity detection |
