"""
Monitor WebSocket & REST API endpoints.

Provides:
- POST /api/monitor/session     — Create a new monitor session
- POST /api/monitor/offer       — WebRTC SDP offer → answer exchange
- GET  /api/monitor/telemetry   — Session telemetry
- DELETE /api/monitor/session    — Stop session
- WS   /ws/monitor/{session_id} — Real-time event stream to browser

The WebSocket endpoint relays events from both pipelines to the browser.
It does NOT receive audio — audio arrives via WebRTC (aiortc).
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from app.services.monitor.session import (
    MonitorSession, create_session, get_session, remove_session
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitor", tags=["Monitor"])


# --- REST Endpoints ---


class CreateSessionRequest(BaseModel):
    asr_model_size: str = "base"


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str


class SDPOfferRequest(BaseModel):
    session_id: str
    sdp: str
    type: str = "offer"


class SDPAnswerResponse(BaseModel):
    session_id: str
    sdp: str
    type: str = "answer"


# Shared ASR provider — loaded once, reused across sessions
_shared_asr_provider = None


def _get_or_create_asr(model_size: str = "base"):
    """Load ASR once and share across sessions (model is stateless)."""
    global _shared_asr_provider
    if _shared_asr_provider is None:
        from app.services.asr.hf_whisper_provider import HFWhisperProvider
        _shared_asr_provider = HFWhisperProvider(
            model_name=model_size,
            backend="ctranslate2",
            beam_size=1,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        _shared_asr_provider.load_model()
    return _shared_asr_provider


@router.post("/session", response_model=CreateSessionResponse)
async def create_monitor_session(req: CreateSessionRequest = CreateSessionRequest()):
    """Create a new monitor session with all pipelines."""
    try:
        asr = _get_or_create_asr(req.asr_model_size)
        session = create_session(asr_model_size=req.asr_model_size)
        await session.start(asr_provider=asr)

        logger.info(f"Monitor session created: {session.session_id}")
        return CreateSessionResponse(
            session_id=session.session_id,
            status="started",
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/offer", response_model=SDPAnswerResponse)
async def handle_webrtc_offer(req: SDPOfferRequest):
    """Exchange WebRTC SDP offer for answer.

    The browser sends its SDP offer; we create a server-side
    RTCPeerConnection via aiortc and return the answer.
    Audio then flows via WebRTC into the session's AudioBus.
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.webrtc_receiver:
        raise HTTPException(status_code=400, detail="Session not initialized")

    try:
        answer_sdp = await session.webrtc_receiver.handle_offer(
            offer_sdp=req.sdp,
            offer_type=req.type,
        )
        logger.info(f"WebRTC offer handled for session {req.session_id}")
        return SDPAnswerResponse(
            session_id=req.session_id,
            sdp=answer_sdp,
            type="answer",
        )
    except Exception as e:
        logger.error(f"WebRTC offer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/{session_id}")
async def get_telemetry(session_id: str):
    """Get real-time telemetry for a monitor session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get_telemetry()


@router.delete("/session/{session_id}")
async def stop_session(session_id: str):
    """Stop and clean up a monitor session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await remove_session(session_id)
    logger.info(f"Monitor session stopped: {session_id}")
    return {"status": "stopped", "session_id": session_id}


# --- WebSocket Event Stream ---


@router.websocket("/ws/{session_id}")
async def monitor_websocket(websocket: WebSocket, session_id: str):
    """Real-time event stream from pipelines to browser.

    Relays caption, detection, speaker, vad, telemetry, and
    window_progress events. Does NOT receive audio.
    """
    session = get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info(f"Monitor WebSocket connected: {session_id}")

    try:
        # Send initial telemetry
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "telemetry": session.get_telemetry(),
        })

        # Relay events from pipelines to browser
        while True:
            event = await session.get_event(timeout=0.1)
            if event:
                await websocket.send_json(event)
            else:
                # Send periodic telemetry heartbeat
                try:
                    await websocket.send_json({
                        "type": "telemetry",
                        **session.get_telemetry(),
                    })
                except Exception:
                    break

                # Small sleep to avoid busy loop
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(f"Monitor WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"Monitor WebSocket closed: {session_id}; terminating session")
        await remove_session(session_id)
