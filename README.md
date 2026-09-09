# VoiceShield 🛡️

**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

*Smart India Hackathon 2026 — PS-104*

---

## Architecture

```
┌────────────────────────────────┐
│ Next.js Frontend (Port 3000)   │
│ Dashboard │ Live │ Analyze     │
│ Enroll    │ History            │
│                                │
│ Browser Mic → PCM Audio        │
│ File Upload → REST API         │
└────────┬──────────────┬────────┘
         │ WebSocket     │ REST
         │ (Binary PCM)  │ (Multipart)
         ▼               ▼
┌────────────────────────────────┐
│ FastAPI Backend (Port 8000)    │
│                                │
│ ┌──────────────────────────┐   │
│ │ StreamingBuffer           │   │
│ │ (Rolling window, 50%     │   │
│ │  overlap, 64600 samples) │   │
│ └────────┬─────────────────┘   │
│          ▼                     │
│ ┌──────────────────────────┐   │
│ │ Analysis Pipeline         │   │
│ │ 1. Silero VAD             │   │
│ │ 2. Spectra-AASIST         │   │
│ │ 3. ECAPA-TDNN (optional)  │   │
│ │ 4. Risk Engine            │   │
│ │ 5. Alert Service          │   │
│ └──────────────────────────┘   │
└────────────────────────────────┘
```

## AI Models Used

| Model | Source | Purpose | Size |
|-------|--------|---------|------|
| **Spectra-AASIST** | `lab260/Spectra-AASIST` | Anti-spoofing (synthetic voice detection) | ~1.2 GB |
| **ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` | Speaker verification | ~90 MB |
| **Silero VAD** | `snakers4/silero-vad` | Voice Activity Detection | ~2 MB |

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- 4GB+ RAM (models are large)

### 1. Backend

```bash
cd voice-shield/backend
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Copy and edit environment config
cp ../.env.example ../.env

# Start the backend (models download automatically on first run)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend

```bash
cd voice-shield/frontend
npm install
npm run dev
```

Visit `http://localhost:3000`

### 3. Docker (optional)

```bash
cd voice-shield
docker-compose up --build
```

## Features

### Real-Time Protection (Live Mode)
- Browser microphone capture → WebSocket → detection → alert
- Rolling buffer with 50% sliding window overlap
- Real-time threat gauge with risk level indicators
- Verify / Dismiss / Block workflow

### Recorded Analysis
- Upload WAV, MP3, M4A, WebM, OGG, FLAC
- VAD-based speech segmentation
- Per-segment spoof scoring with visual timeline
- Optional speaker verification against enrolled profiles

### Speaker Enrollment
- Record voice or upload sample
- Generates 192-dimensional speaker embeddings via ECAPA-TDNN
- Profile management (list, delete)

### Risk Engine
- **Configurable thresholds** (not hardcoded)
- Temporal consistency tracking
- Hysteresis with debounce (prevents flickering alerts)
- Rolling window averaging for live mode

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health and model status |
| POST | `/api/analyze/upload` | Analyze uploaded audio |
| POST | `/api/enrollment` | Enroll speaker profile |
| GET | `/api/enrollment/profiles` | List enrolled speakers |
| DELETE | `/api/enrollment/{id}` | Delete speaker profile |
| WS | `/ws/voice-analysis` | Live audio streaming |
| GET | `/api/sessions` | Security event history |
| GET | `/api/alerts/recent` | Recent alerts |

## Configuration

All thresholds are configurable via `.env` file:

```env
# Risk thresholds — CALIBRATE WITH YOUR DATA
SPOOF_THRESHOLD_CRITICAL=0.85
SPOOF_THRESHOLD_HIGH=0.70
SPOOF_THRESHOLD_MEDIUM=0.40
SPEAKER_THRESHOLD_MATCH=0.25

# Sliding window
SLIDING_WINDOW_OVERLAP=0.5  # 50% overlap

# Risk engine
RISK_WINDOW_COUNT=5
RISK_ESCALATION_COUNT=3
RISK_DEBOUNCE_SECONDS=2.0
```

## Team

*Smart India Hackathon 2026 — PS-104*

## License

Educational use — SIH 2026.
