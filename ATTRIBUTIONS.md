# VoiceShield — Open Source Attributions

## AI Models

### Spectra-AASIST
- **Source**: [lab260/Spectra-AASIST](https://huggingface.co/lab260/Spectra-AASIST)
- **Paper**: "Spectra-AASIST: Spectro-Temporal Representations for Anti-Spoofing"
- **License**: Apache 2.0
- **Purpose**: Anti-spoofing / synthetic voice detection

### ECAPA-TDNN
- **Source**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Paper**: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
- **License**: Apache 2.0
- **Purpose**: Speaker verification / voice embedding extraction

### Silero VAD
- **Source**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **License**: MIT
- **Purpose**: Voice Activity Detection

### wav2vec2-xls-r-300m (Internal to Spectra-AASIST)
- **Source**: [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- **License**: Apache 2.0
- **Purpose**: SSL encoder backbone used internally by Spectra-AASIST

## Frameworks & Libraries

| Library | License | Purpose |
|---------|---------|---------|
| PyTorch | BSD-3 | Deep learning framework |
| FastAPI | MIT | Backend web framework |
| Next.js | MIT | Frontend framework |
| SpeechBrain | Apache 2.0 | Speaker verification toolkit |
| Transformers | Apache 2.0 | Model hub and loading |
| torchaudio | BSD-2 | Audio processing |
| librosa | ISC | Audio analysis |
| Tailwind CSS | MIT | Frontend styling |

## Datasets (Referenced)

- **ASVspoof 2019/2021**: Used for training/evaluating anti-spoofing models
- **VoxCeleb1/2**: Used for training ECAPA-TDNN speaker verification
- **Common Voice**: Referenced for multilingual voice data
