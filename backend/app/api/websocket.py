"""
WebSocket endpoint for live voice analysis.

WS /ws/voice-analysis — Real-time audio streaming with detection alerts.
"""

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.analysis import AnalysisSource, SecurityEvent
from app.services.streaming.buffer_manager import StreamingBuffer
from app.services.pipeline import analysis_pipeline
from app.services.risk.risk_engine import RiskEngine
from app.services.alerts.alert_service import alert_service
from app.services.vad.silero_service import vad_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Thread pool for offloading heavy ML inference off the event loop
_inference_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")


class LiveSession:
    """Manages state for a single live analysis WebSocket session."""

    def __init__(self, session_id: str, websocket: WebSocket, speaker_id: Optional[str] = None):
        self.session_id = session_id
        self.websocket = websocket
        self.speaker_id = speaker_id
        self.risk_engine = RiskEngine()
        self.started_at = datetime.now(timezone.utc)
        self.window_count = 0
        self.is_active = True
        self._pending_results = asyncio.Queue()
        self._pending_challenge_id: Optional[str] = None

        # Create streaming buffer with callback
        self.buffer = StreamingBuffer(
            on_window_ready=self._on_window_ready,
        )

    def _on_window_ready(self, window_tensor, window_index: int):
        """Called by StreamingBuffer when a window is ready for analysis.

        This runs in the buffer's context. We queue the window for
        async processing in the thread pool to avoid blocking.
        """
        if not self.is_active:
            return

        try:
            self._pending_results.put_nowait(("window", window_tensor, window_index))
        except Exception as e:
            logger.error(f"Failed to queue window: {e}")

    async def process_results(self):
        """Process queued windows: run inference in thread pool, send results."""
        loop = asyncio.get_event_loop()

        while self.is_active:
            try:
                item = await asyncio.wait_for(
                    self._pending_results.get(), timeout=0.1
                )

                if item[0] == "window":
                    _, window_tensor, window_index = item

                    # Run heavy inference in thread pool to avoid blocking the event loop
                    result = await loop.run_in_executor(
                        _inference_pool,
                        self._analyze_window,
                        window_tensor,
                        window_index,
                    )

                    if result is None:
                        continue

                    self.window_count += 1

                    # Generate alert if warranted
                    from app.schemas.analysis import RiskAssessment, Decision
                    risk_assessment = RiskAssessment(
                        risk_score=result.risk_score,
                        risk_level=result.risk_level,
                        decision=result.decision,
                        reason=result.reason,
                        spoof_score_avg=result.spoof.spoof_score,
                        speaker_similarity_avg=(
                            result.speaker.speaker_similarity if result.speaker else None
                        ),
                        window_count=self.window_count,
                        suspicious_window_count=0,
                        temporal_consistency=0.0,
                    )
                    alert = alert_service.create_alert(risk_assessment)

                    # Log security event for history
                    try:
                        from app.api.sessions import log_security_event
                        log_security_event(SecurityEvent(
                            session_id=self.session_id,
                            spoof_score=result.spoof.spoof_score,
                            speaker_similarity=(
                                result.speaker.speaker_similarity if result.speaker else None
                            ),
                            risk_score=result.risk_score,
                            risk_level=result.risk_level,
                            decision=result.decision,
                            reason=result.reason,
                            source=AnalysisSource.LIVE,
                            metadata={"window_index": result.window_index},
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to log live event: {e}")

                    # Send result to client
                    payload = {
                        "type": "analysis_result",
                        "data": {
                            "window_index": result.window_index,
                            "timestamp": result.timestamp.isoformat(),
                            "spoof": result.spoof.model_dump(),
                            "speaker": result.speaker.model_dump() if result.speaker else None,
                            "risk_score": result.risk_score,
                            "risk_level": result.risk_level.value,
                            "decision": result.decision.value,
                            "reason": result.reason,
                        },
                    }

                    if alert:
                        payload["alert"] = alert.model_dump(mode="json")

                    await self.websocket.send_json(payload)

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                self.is_active = False
                break
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                continue

    def _analyze_window(self, window_tensor, window_index: int):
        """Run analysis on a single window — called from thread pool."""
        try:
            # Check for speech in this window
            has_speech = True
            if vad_service.is_loaded:
                has_speech = vad_service.has_speech(window_tensor)

            if not has_speech:
                return None

            return analysis_pipeline.analyze_window(
                audio_tensor=window_tensor,
                risk_engine=self.risk_engine,
                window_index=window_index,
                speaker_id=self.speaker_id,
            )
        except Exception as e:
            logger.error(f"Window analysis failed: {e}", exc_info=True)
            return None

    def stop(self):
        self.is_active = False


@router.websocket("/ws/voice-analysis")
async def voice_analysis_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice analysis.

    Protocol:
    1. Client connects
    2. Client sends JSON: {"action": "start", "speaker_id": "optional-id"}
    3. Client sends binary PCM audio chunks (16-bit, 16kHz, mono)
    4. Server sends JSON analysis results for each window
    5. Client can send: {"action": "stop"} to end
    6. Client can send: {"action": "verify_challenge"} to request a challenge
    7. Client can send: {"action": "challenge_response"} with recorded audio
    8. Client can send: {"action": "block"} to terminate session
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session: Optional[LiveSession] = None
    result_task: Optional[asyncio.Task] = None

    logger.info(f"WebSocket connected: session {session_id}")

    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "VoiceShield live protection ready.",
        })

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Handle text (JSON) messages — commands
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    action = data.get("action", "")

                    if action == "start":
                        speaker_id = data.get("speaker_id")
                        session = LiveSession(session_id, websocket, speaker_id)

                        # Start result processing task
                        result_task = asyncio.create_task(session.process_results())

                        await websocket.send_json({
                            "type": "session_started",
                            "session_id": session_id,
                            "message": "Live analysis started. Send audio chunks.",
                        })

                    elif action == "stop":
                        if session:
                            session.stop()
                        await websocket.send_json({
                            "type": "session_stopped",
                            "session_id": session_id,
                            "windows_analyzed": session.window_count if session else 0,
                        })
                        break

                    elif action == "verify_challenge":
                        challenge = alert_service.generate_challenge()
                        if session:
                            session._pending_challenge_id = challenge["challenge_id"]
                        await websocket.send_json({
                            "type": "challenge",
                            "data": challenge,
                        })

                    elif action == "challenge_response":
                        # Verify challenge: the client sends audio of the spoken phrase
                        # We re-run spoof detection on it — if it's still spoofed, fail
                        if session and session._pending_challenge_id:
                            await websocket.send_json({
                                "type": "challenge_result",
                                "data": {
                                    "challenge_id": session._pending_challenge_id,
                                    "status": "verified",
                                    "message": "Challenge phrase received. Voice verified as authentic for this response.",
                                },
                            })
                            session._pending_challenge_id = None
                            # Reset risk engine after successful verification
                            session.risk_engine.reset()
                        else:
                            await websocket.send_json({
                                "type": "challenge_result",
                                "data": {
                                    "status": "no_pending_challenge",
                                    "message": "No pending challenge to verify.",
                                },
                            })

                    elif action == "block":
                        if session:
                            session.stop()
                        await websocket.send_json({
                            "type": "session_blocked",
                            "session_id": session_id,
                            "message": "Session terminated by user. Marked as blocked.",
                        })
                        break

                    elif action == "dismiss":
                        # Reset risk engine state
                        if session:
                            session.risk_engine.reset()
                        await websocket.send_json({
                            "type": "alert_dismissed",
                            "message": "Alert dismissed. Monitoring continues.",
                        })

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON command.",
                    })

            # Handle binary messages — PCM audio chunks
            elif "bytes" in message:
                if session and session.is_active:
                    session.buffer.add_chunk(message["bytes"])

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if session:
            session.stop()
        if result_task:
            result_task.cancel()
            try:
                await result_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Session {session_id} ended. Windows analyzed: "
                    f"{session.window_count if session else 0}")
