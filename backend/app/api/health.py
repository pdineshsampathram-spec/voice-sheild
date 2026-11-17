"""
Health check endpoint — reports model status, DB, and Redis.
"""

import time
from fastapi import APIRouter

from app.schemas.analysis import SystemHealth, ModelStatus
from app.services.antispoof.spectra_detector import spectra_detector
from app.services.vad.silero_service import vad_service
from app.services.speaker.ecapa_service import ecapa_service
from app.services.antispoof.calibrator import score_calibrator
import app.api.ws_monitor as ws_monitor

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check including model status."""
    asr = ws_monitor._shared_asr_provider
    asr_status = "ONLINE" if (asr and asr.is_loaded()) else "ASR_UNAVAILABLE"
    asr_msg = (
        f"model={getattr(asr, '_model_name', 'base')}, backend={getattr(asr, '_backend', 'ctranslate2')}"
        if asr else "Not loaded"
    )

    models = [
        ModelStatus(
            name="Spectra-AASIST",
            status=spectra_detector.status,
            message=spectra_detector.status_message,
            load_time_ms=spectra_detector.load_time_ms,
        ),
        ModelStatus(
            name="ECAPA-TDNN",
            status=ecapa_service.status,
            message=ecapa_service.status_message,
            load_time_ms=ecapa_service.load_time_ms,
        ),
        ModelStatus(
            name="Silero VAD",
            status=vad_service.status,
            message=vad_service.status_message,
            load_time_ms=vad_service.load_time_ms,
        ),
        ModelStatus(
            name="Faster-Whisper",
            status=asr_status,
            message=asr_msg,
            load_time_ms=getattr(asr, "_load_time_ms", None) if asr else None,
        ),
        ModelStatus(
            name="Score-Calibrator",
            status=score_calibrator.status,
            message=f"version={score_calibrator.version}, domain={score_calibrator.domain}, method={score_calibrator.method}",
            load_time_ms=None,
        ),
    ]

    all_online = all(m.status == "ONLINE" for m in models)
    any_error = any(m.status == "ERROR" for m in models)

    if all_online:
        overall = "healthy"
    elif any_error:
        overall = "degraded"
    else:
        overall = "loading"

    # Prototype: using in-memory stores, not actual DB/Redis
    db_status = "IN_MEMORY"
    redis_status = "NOT_CONFIGURED"

    return SystemHealth(
        status=overall,
        models=models,
        database=db_status,
        redis=redis_status,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
