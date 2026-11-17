# Open Source Attribution

VoiceShield Mobile integrates open-source components in full compliance with their respective licenses.

---

## 1. dialer-core
- **Repository**: [https://github.com/fayyaztech/dialer-core](https://github.com/fayyaztech/dialer-core)
- **Author**: Fayyaz Tech
- **License**: Apache License, Version 2.0 (see `dialer-core/LICENSE`)
- **What is reused**:
  - Telephony `InCallService` framework and call state handling
  - `CallScreeningService` structure for spam and metadata evaluation
  - Call routing (speaker, earpiece, bluetooth)
  - Proximity sensor and audio route handling
- **Modifications**:
  - Extended `DefaultInCallService` with VoiceShield `CallSessionManager` and `ThreatState` hooks
  - Enforced strict audio availability reporting (`UNAVAILABLE` by default for SIM calls)
  - Integrated with VoiceShield custom Jetpack Compose security overlays

---

## 2. AASIST-L (Anti-Spoofing Model)
- **Repository**: [https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L](https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L)
- **Authors**: Jee-weon Jung et al. / SpeechAntiSpoofingBenchmarks (Kirill Borodin)
- **License**: MIT License (inherited from `clovaai/aasist`)
- **What is reused**:
  - AASIST-L lightweight graph attention architecture (~85k parameters)
  - ASVspoof2019 LA pretrained checkpoint weights
- **Input Contract**:
  - 16 kHz mono float32, first 64,600 samples (~4.04 seconds)
  - Output: 2-class logits (bona fide vs spoof)

---

## 3. ECAPA-TDNN ONNX (Speaker Verification)
- **Repository**: [https://huggingface.co/astrolabos/ecapa-tdnn](https://huggingface.co/astrolabos/ecapa-tdnn)
- **Source Model**: `speechbrain/spkrec-ecapa-voxceleb` (SpeechBrain)
- **License**: Apache License, Version 2.0
- **What is reused**:
  - Pre-exported ONNX weights for speaker embedding generation
  - 192-dimensional speaker embeddings for cosine similarity verification

---

## 4. Silero VAD
- **Repository**: [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **License**: MIT License
- **What is reused**:
  - Silero VAD ONNX model for real-time speech activity detection

---

## 5. ONNX Runtime Android
- **Repository**: [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **License**: MIT License
- **What is reused**:
  - Runtime library for executing quantized/optimized neural networks on-device
