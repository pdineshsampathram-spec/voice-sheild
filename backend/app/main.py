"""
VoiceShield — AI-Powered Voice Cloning Detection & Prevention

Main FastAPI application entry point.
Models are loaded once at startup via the lifespan context manager.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voiceshield")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — load models at startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("VoiceShield starting — loading AI models...")
    logger.info(f"Device: {settings.get_device()}")
    logger.info("=" * 60)

    start = time.time()

    # Load models in dependency order
    # 1. Silero VAD (fastest, CPU-only)
    from app.services.vad.silero_service import vad_service
    vad_service.load_model()

    # 2. Spectra-AASIST (primary, heaviest)
    from app.services.antispoof.spectra_detector import spectra_detector
    spectra_detector.load_model()

    # 3. ECAPA-TDNN (speaker verification)
    from app.services.speaker.ecapa_service import ecapa_service
    ecapa_service.load_model()

    # 4. Faster-Whisper ASR — loaded ONCE, shared across all sessions (Fix C1)
    from app.api.ws_monitor import _get_or_create_asr
    asr_provider = _get_or_create_asr("base")
    asr_status = "ONLINE" if asr_provider.is_loaded() else "ASR_UNAVAILABLE"

    # 5. Score calibration artifact lifecycle (Fix M3)
    from app.services.antispoof.calibrator import score_calibrator
    from pathlib import Path

    calibration_candidates = [
        Path("models/calibration_v1.json"),
        Path("backend/models/calibration_v1.json"),
        Path(__file__).parent.parent / "models" / "calibration_v1.json",
    ]
    calibration_artifact_path = None
    for cand in calibration_candidates:
        if cand.exists():
            calibration_artifact_path = cand
            break

    if calibration_artifact_path is not None:
        loaded = score_calibrator.load(calibration_artifact_path)
        if loaded:
            calibration_status = f"CALIBRATED (v={score_calibrator.version}, domain={score_calibrator.domain})"
        else:
            calibration_status = score_calibrator.status
            logger.warning(
                f"Calibration artifact exists at {calibration_artifact_path} "
                f"but failed validation. Status: {calibration_status}."
            )
    else:
        calibration_status = "UNCALIBRATED"
        logger.info(
            "No calibration artifact found. "
            "All anti-spoof decisions will report UNCALIBRATED status."
        )

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"All models loaded in {elapsed:.1f}s")
    logger.info(f"  Spectra-AASIST: {spectra_detector.status}")
    logger.info(f"  ECAPA-TDNN:     {ecapa_service.status}")
    logger.info(f"  Silero VAD:     {vad_service.status}")
    logger.info(f"  Faster-Whisper: {asr_status}")
    logger.info(f"  Calibration:    {calibration_status}")
    logger.info("=" * 60)

    yield

    # Shutdown cleanup
    logger.info("VoiceShield shutting down.")


# Create FastAPI app
app = FastAPI(
    title="VoiceShield",
    description="AI-Powered Real-Time Voice Cloning Detection & Prevention",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from app.api.health import router as health_router
from app.api.analyze import router as analyze_router
from app.api.enrollment import router as enrollment_router
from app.api.sessions import router as sessions_router
from app.api.websocket import router as websocket_router
from app.api.webrtc import router as webrtc_router
from app.api.caller import router as caller_router
from app.api.ws_monitor import router as monitor_router

app.include_router(health_router, tags=["Health"])
app.include_router(analyze_router, tags=["Analysis"])
app.include_router(enrollment_router, tags=["Enrollment"])
app.include_router(sessions_router, tags=["Sessions"])
app.include_router(websocket_router, tags=["WebSocket"])
app.include_router(webrtc_router, tags=["WebRTC Signaling"])
app.include_router(caller_router)
app.include_router(monitor_router, tags=["Monitor"])


@app.get("/")
async def root():
    return {
        "name": "VoiceShield",
        "version": "0.1.0",
        "description": "AI-Powered Voice Cloning Detection & Prevention",
        "docs": "/docs",
    }
