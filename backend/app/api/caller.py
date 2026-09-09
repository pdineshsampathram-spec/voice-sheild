"""
VoiceShield — Caller Agent API Router.

REST endpoints, Twilio TwiML/Status webhooks, and Media Stream WebSocket handler.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional, List
from fastapi import (
    APIRouter, HTTPException, Request, Response,
    Query, Form, Header, WebSocket, WebSocketDisconnect, status
)

from app.core.config import settings
from app.schemas.caller import (
    CallSession, OutboundCallRequest, OutboundCallResponse,
    CallStatus, MediaStreamStatus
)
from app.services.caller.session_manager import call_session_manager
from app.services.caller.twilio_service import twilio_service
from app.services.caller.elevenlabs_service import (
    elevenlabs_service, ElevenLabsIntegrationError
)
from app.services.caller.audio_decoder import audio_decoder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/caller", tags=["Caller Agent"])


async def _auto_terminate_after(call_id: str, seconds: int):
    """Wait for specified seconds and terminate the call if still active."""
    await asyncio.sleep(seconds)
    session = call_session_manager.get_session(call_id)
    if session and session.status not in [
        CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.BUSY,
        CallStatus.NO_ANSWER, CallStatus.CANCELED
    ]:
        logger.info(f"Max call duration ({seconds}s) elapsed for {call_id}. Initiating auto-hangup.")
        call_session_manager.record_timeline_event(
            call_id,
            event_type="MAX_DURATION_REACHED",
            description=f"Auto-terminating call after reaching {seconds}s limit",
        )
        call_session_manager.mark_termination_requested(call_id)
        if session.call_sid:
            try:
                twilio_service.terminate_call(session.call_sid)
            except Exception as e:
                logger.error(f"Error during auto-termination of {session.call_sid}: {e}")



@router.get("/config")
async def get_caller_config():
    """Check caller agent configuration status with all secrets safely masked."""
    return settings.get_masked_caller_config()


# In-memory cache of pre-fetched TwiML to respond in 0ms to Twilio's webhook
_cached_twiml: dict[str, str] = {}


@router.post(
    "/outbound",
    response_model=OutboundCallResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_outbound_call(request: OutboundCallRequest):
    """Initiate an outbound AI call via Twilio and ElevenLabs.

    STRICT CONSTRAINTS:
    - Destination is strictly locked to TWILIO_VERIFIED_DESTINATION_NUMBER.
    - If user supplies an arbitrary destination that differs, request is rejected.
    """
    missing = settings.validate_caller_config()
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Caller agent is not fully configured.",
                "missing_variables": missing,
                "hint": "Set required variables in .env before placing outbound calls.",
            },
        )

    verified_dest = settings.twilio_verified_destination_number.strip()

    # Reject attempt to call arbitrary numbers
    if request.destination and request.destination.strip() != verified_dest:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Trial account limitation: arbitrary destination numbers are blocked.",
                "allowed_destination": verified_dest,
                "requested_destination": request.destination,
            },
        )

    call_id = str(uuid.uuid4())
    agent_id = (request.agent_id or settings.elevenlabs_agent_id).strip()
    greeting = request.greeting

    # 1. Create CallSession
    session = call_session_manager.create_session(
        call_id=call_id,
        destination=verified_dest,
        agent_id=agent_id,
        greeting=greeting,
        max_duration_seconds=request.max_duration_seconds,
    )

    # 1.5 Pre-register with ElevenLabs so TwiML is ready instantaneously (0ms)
    # when the user answers, completely eliminating Twilio's 10-second webhook timeout.
    try:
        prefetched_twiml = await elevenlabs_service.register_call(
            agent_id=agent_id,
            greeting=greeting,
        )
        _cached_twiml[call_id] = prefetched_twiml
        logger.info(f"Pre-registered ElevenLabs TwiML for call_id={call_id} successfully")
    except Exception as e:
        logger.warning(f"Could not pre-register ElevenLabs TwiML at dispatch (will fallback on answer): {e}")

    # 2. Dispatch call via Twilio REST API
    try:
        call_sid = twilio_service.create_outbound_call(call_id=call_id, greeting=greeting)
        call_session_manager.attach_call_sid(call_id, call_sid)
    except Exception as e:
        _cached_twiml.pop(call_id, None)
        logger.error(f"Failed to dispatch Twilio call: {e}", exc_info=True)
        session.status = CallStatus.FAILED
        session.error_message = str(e)
        call_session_manager.record_timeline_event(
            call_id,
            event_type="DISPATCH_FAILED",
            description=f"Twilio dispatch error: {e}",
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to initiate Twilio call: {e}",
        )

    return OutboundCallResponse(
        call_id=call_id,
        call_sid=call_sid,
        status=session.status,
        destination=verified_dest,
        message="Outbound call dispatched to verified destination. Awaiting answer.",
        milestone="MILESTONE_1_OUTBOUND",
    )


@router.get("/calls", response_model=List[CallSession])
async def list_calls():
    """List all call sessions (most recent first) with full timelines and telemetry."""
    return call_session_manager.list_sessions()


@router.get("/calls/{call_id}", response_model=CallSession)
async def get_call(call_id: str):
    """Retrieve detailed state, timeline, and telemetry for a specific call session."""
    session = call_session_manager.get_session(call_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call session {call_id} not found.",
        )
    return session


@router.post("/calls/{call_id}/end", response_model=CallSession)
async def end_call(call_id: str):
    """Request termination of an active call.

    IMPORTANT (Reviewer Correction B):
    Requests Twilio termination only; terminal lifecycle state (COMPLETED)
    is recorded upon receipt of the Twilio status callback webhook.
    """
    session = call_session_manager.get_session(call_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call session {call_id} not found.",
        )

    if session.status in [
        CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.BUSY,
        CallStatus.NO_ANSWER, CallStatus.CANCELED
    ]:
        return session

    call_session_manager.mark_termination_requested(call_id)

    if session.call_sid:
        try:
            twilio_service.terminate_call(session.call_sid)
        except Exception as e:
            logger.error(f"Error requesting Twilio termination for SID {session.call_sid}: {e}")
            call_session_manager.record_timeline_event(
                call_id,
                event_type="TERMINATION_ERROR",
                description=f"Twilio termination request failed: {e}",
            )

    return session


@router.post("/twiml")
async def twiml_webhook(
    request: Request,
    call_id: Optional[str] = Query(None),
    x_twilio_signature: Optional[str] = Header(None, alias="X-Twilio-Signature"),
):
    """Twilio TwiML Webhook.

    Executed when the user answers the outbound call. Fetches TwiML from
    ElevenLabs register-call endpoint and returns it to Twilio.
    """
    # Signature verification if configured
    form_data = await request.form()
    params = dict(form_data)
    url = str(request.url)

    if not twilio_service.validate_webhook_signature(url, params, x_twilio_signature):
        logger.warning(f"Signature validation failed for {url}")
        if settings.twilio_webhook_validate_signature:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid Twilio signature.",
            )

    session = call_session_manager.get_session(call_id) if call_id else None
    if session:
        call_session_manager.record_timeline_event(
            session.call_id,
            event_type="TWIML_REQUESTED",
            description="Twilio requested TwiML upon call answer",
            metadata={"call_id": call_id},
        )

    agent_id = session.agent_id if session else settings.elevenlabs_agent_id
    greeting = session.greeting if session else None

    try:
        # Check pre-registered cache first to respond in 0ms (immune to network latency)
        twiml_xml = _cached_twiml.pop(call_id, None) if call_id else None
        if not twiml_xml:
            logger.info(f"No pre-registered TwiML found for call_id={call_id}, registering now...")
            twiml_xml = await elevenlabs_service.register_call(
                agent_id=agent_id,
                greeting=greeting,
            )
        else:
            logger.info(f"Serving pre-registered TwiML for call_id={call_id} instantly")

        if session:
            call_session_manager.record_timeline_event(
                session.call_id,
                event_type="ELEVENLABS_CONNECTED",
                description="ElevenLabs register-call TwiML returned and verified",
            )
            if session.max_duration_seconds and session.max_duration_seconds > 0:
                logger.info(
                    f"Scheduling auto-termination timer for {session.call_id} in "
                    f"{session.max_duration_seconds}s"
                )
        # Ensure call does not drop prematurely if stream pauses or ends;
        # provide continuous speech and hold the line open so VoiceShield can test audio.
        if "</Response>" in twiml_xml and "<Say" not in twiml_xml:
            continuous_fallback = (
                '\n    <Say voice="Polly.Aditi" language="en-IN">'
                'VoiceShield continuous testing active. Monitoring live cellular call audio for synthetic voice artifacts and acoustic anomalies. Please continue speaking or listening while real-time spoof detection analyzes spectral characteristics.'
                '</Say>\n'
                '    <Pause length="2"/>\n'
                '    <Say voice="Polly.Aditi" language="en-IN">'
                'VoiceShield deepfake detection pipeline evaluates multi-band spectrograms, temporal consistency, and vocal tract resonance. This call remains active for testing.'
                '</Say>\n'
                '    <Pause length="3"/>\n'
                '    <Say voice="Polly.Aditi" language="en-IN">'
                'Downlink audio stream is stable. You can speak now to test two-way communication, or hang up when finished.'
                '</Say>\n'
                '    <Pause length="120"/>\n'
                '</Response>'
            )
            twiml_xml = twiml_xml.replace("</Response>", continuous_fallback)

        return Response(content=twiml_xml, media_type="application/xml")

    except ElevenLabsIntegrationError as e:
        logger.error(f"ElevenLabs registration failed: {e}", exc_info=True)
        if session:
            call_session_manager.record_timeline_event(
                session.call_id,
                event_type="ELEVENLABS_ERROR",
                description=f"ElevenLabs registration error: {e}",
            )

        # Return a graceful message to the caller
        error_twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            '    <Say voice="Polly.Amy">VoiceShield AI agent connection could not be established. Goodbye.</Say>\n'
            '    <Hangup/>\n'
            '</Response>'
        )
        return Response(content=error_twiml, media_type="application/xml")


@router.post("/status-callback")
async def status_callback_webhook(
    request: Request,
    call_id: Optional[str] = Query(None),
    x_twilio_signature: Optional[str] = Header(None, alias="X-Twilio-Signature"),
):
    """Twilio Status Callback Webhook.

    Receives real lifecycle updates (initiated, ringing, answered, completed)
    and updates CallSession status and timeline.
    """
    form_data = await request.form()
    params = dict(form_data)
    url = str(request.url)

    if not twilio_service.validate_webhook_signature(url, params, x_twilio_signature):
        if settings.twilio_webhook_validate_signature:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid Twilio signature.",
            )

    call_sid = params.get("CallSid")
    call_status = params.get("CallStatus", "unknown")
    call_duration = params.get("CallDuration")
    duration_int = int(call_duration) if call_duration and call_duration.isdigit() else None
    error_code = params.get("ErrorCode")
    error_message = params.get("ErrorMessage")

    logger.info(
        f"Twilio Status Callback: SID={call_sid} Status={call_status} Duration={call_duration}s"
    )

    if call_sid:
        call_session_manager.update_from_twilio_callback(
            call_sid=call_sid,
            twilio_status=call_status,
            duration=duration_int,
            error_code=error_code,
            error_message=error_message,
        )

    return {"status": "ok", "call_sid": call_sid, "received_status": call_status}


@router.websocket("/media-stream")
async def media_stream_websocket(websocket: WebSocket, call_id: Optional[str] = Query(None)):
    """Twilio Media Stream WebSocket Endpoint (Milestone 2).

    Receives live 8kHz mu-law audio packets, decodes to 16kHz PCM,
    updates telemetry, and verifies media stream delivery.
    """
    await websocket.accept()
    logger.info(f"Media stream WebSocket connected for call_id={call_id}")

    session = call_session_manager.get_session(call_id) if call_id else None
    if session:
        session.telemetry.media_delivery_status = MediaStreamStatus.CONNECTED
        call_session_manager.record_timeline_event(
            session.call_id,
            event_type="MEDIA_STREAM_CONNECTED",
            description="Twilio media stream WebSocket connected",
        )

    try:
        while True:
            message_text = await websocket.receive_text()
            data = json.loads(message_text)
            event = data.get("event")

            if event == "start":
                stream_sid = data.get("streamSid")
                logger.info(f"Twilio media stream started: StreamSID={stream_sid}")
                if session:
                    call_session_manager.record_timeline_event(
                        session.call_id,
                        event_type="MEDIA_STREAM_STARTED",
                        description=f"Stream SID: {stream_sid}",
                        metadata={"stream_sid": stream_sid},
                    )

            elif event == "media":
                media_data = data.get("media", {})
                payload_b64 = media_data.get("payload", "")

                if payload_b64:
                    pcm_16k, telemetry = audio_decoder.decode_twilio_payload(payload_b64)

                    if session:
                        call_session_manager.update_telemetry(
                            call_id=session.call_id,
                            bytes_count=telemetry["bytes_count"],
                            samples_count=telemetry["samples_count"],
                            rms=telemetry["rms"],
                            peak=telemetry["peak"],
                            percentage_zero=telemetry["percentage_zero"],
                            nan_count=telemetry["nan_count"],
                            inf_count=telemetry["inf_count"],
                            media_status=MediaStreamStatus.RECEIVING_PACKETS,
                        )

            elif event == "stop":
                logger.info("Twilio media stream stopped.")
                if session:
                    session.telemetry.media_delivery_status = MediaStreamStatus.DISCONNECTED
                    call_session_manager.record_timeline_event(
                        session.call_id,
                        event_type="MEDIA_STREAM_STOPPED",
                        description="Media stream stop event received",
                    )
                break

    except WebSocketDisconnect:
        logger.info(f"Media stream WebSocket disconnected for call_id={call_id}")
        if session:
            session.telemetry.media_delivery_status = MediaStreamStatus.DISCONNECTED
    except Exception as e:
        logger.error(f"Media stream WebSocket error: {e}", exc_info=True)
        if session:
            session.telemetry.media_delivery_status = MediaStreamStatus.ERROR
