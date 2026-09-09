"""
VoiceShield — Caller Agent Test Suite.

Automated unit and integration tests covering:
- Configuration validation and credential masking
- Strict verified-destination enforcement (rejecting arbitrary numbers)
- CallSession lifecycle and state machine transitions
- Strict ElevenLabs response validation (XML / Connect / Stream contracts)
- Twilio REST client call dispatch construction
- Twilio webhook signature verification
- Telephony 8kHz mu-law to 16kHz Float32 PCM decoding & telemetry
- REST API endpoint contracts
"""

import base64
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.schemas.caller import CallStatus, CallDirection, MediaStreamStatus
from app.services.caller.session_manager import CallSessionManager
from app.services.caller.elevenlabs_service import (
    ElevenLabsService, ElevenLabsIntegrationError
)
from app.services.caller.twilio_service import TwilioService
from app.services.caller.audio_decoder import TelephonyAudioDecoder


client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Configuration Validation & Secret Masking
# ---------------------------------------------------------------------------

def test_caller_config_validation():
    """Verify that missing variables are identified and secrets are masked."""
    # Test with empty settings
    with patch.object(settings, "twilio_account_sid", ""), \
         patch.object(settings, "twilio_auth_token", ""), \
         patch.object(settings, "twilio_phone_number", ""), \
         patch.object(settings, "twilio_verified_destination_number", ""), \
         patch.object(settings, "elevenlabs_api_key", ""), \
         patch.object(settings, "elevenlabs_agent_id", ""):

        missing = settings.validate_caller_config()
        assert "TWILIO_ACCOUNT_SID" in missing
        assert "TWILIO_AUTH_TOKEN" in missing
        assert "TWILIO_PHONE_NUMBER" in missing
        assert "TWILIO_VERIFIED_DESTINATION_NUMBER" in missing
        assert "ELEVENLABS_API_KEY" in missing
        assert "ELEVENLABS_AGENT_ID" in missing
        assert settings.is_caller_configured() is False

        masked = settings.get_masked_caller_config()
        assert masked["twilio_auth_token"] == "<NOT SET>"
        assert masked["elevenlabs_api_key"] == "<NOT SET>"
        assert masked["configured"] is False

    # Test with configured settings — secrets must be masked
    with patch.object(settings, "twilio_account_sid", "AC1234567890abcdef"), \
         patch.object(settings, "twilio_auth_token", "super_secret_auth_token_xyz"), \
         patch.object(settings, "twilio_phone_number", "+15005550006"), \
         patch.object(settings, "twilio_verified_destination_number", "+919876543210"), \
         patch.object(settings, "elevenlabs_api_key", "xi_api_key_secret_12345"), \
         patch.object(settings, "elevenlabs_agent_id", "agent_abc_123"), \
         patch.object(settings, "backend_public_url", "https://test.ngrok-free.app"):

        assert settings.is_caller_configured() is True
        masked = settings.get_masked_caller_config()
        assert masked["configured"] is True
        assert masked["twilio_auth_token"] == "***"
        assert masked["elevenlabs_api_key"] == "***"
        assert "super_secret" not in json.dumps(masked)
        assert "xi_api_key" not in json.dumps(masked)
        assert masked["twilio_account_sid"] == "AC1...def"


# ---------------------------------------------------------------------------
# 2. Strict Destination Enforcement (Trial Account Protection)
# ---------------------------------------------------------------------------

def test_outbound_call_unconfigured_returns_503():
    """If caller config is missing, return 503 with informative details."""
    with patch.object(settings, "twilio_account_sid", ""):
        res = client.post("/caller/outbound", json={})
        assert res.status_code == 503
        data = res.json()
        assert "missing_variables" in data["detail"]


def test_outbound_call_arbitrary_destination_rejected():
    """Arbitrary destination numbers must be rejected with HTTP 400."""
    with patch.object(settings, "twilio_account_sid", "ACmock"), \
         patch.object(settings, "twilio_auth_token", "mocktoken"), \
         patch.object(settings, "twilio_phone_number", "+15005550006"), \
         patch.object(settings, "twilio_verified_destination_number", "+919876543210"), \
         patch.object(settings, "elevenlabs_api_key", "ximock"), \
         patch.object(settings, "elevenlabs_agent_id", "agentmock"), \
         patch.object(settings, "backend_public_url", "https://test.ngrok-free.app"):

        # Attempt to call an unverified arbitrary number
        res = client.post("/caller/outbound", json={"destination": "+19999999999"})
        assert res.status_code == 400
        assert "Trial account limitation" in res.json()["detail"]["error"]


# ---------------------------------------------------------------------------
# 3. CallSession Lifecycle & State Transitions
# ---------------------------------------------------------------------------

def test_call_session_lifecycle():
    """Verify session transitions and non-premature termination semantics."""
    mgr = CallSessionManager()
    call_id = "test-call-1"
    dest = "+919876543210"

    session = mgr.create_session(call_id=call_id, destination=dest, agent_id="agent-1")
    assert session.status == CallStatus.CREATED
    assert session.direction == CallDirection.OUTBOUND
    assert len(session.timeline) == 1
    assert session.timeline[0].event_type == "CALL_CREATED"

    # Attach Twilio SID
    mgr.attach_call_sid(call_id, "CA12345")
    assert session.call_sid == "CA12345"
    assert mgr.get_session_by_sid("CA12345").call_id == call_id

    # Reviewer Correction B: end_call requests termination only; status NOT completed yet
    mgr.mark_termination_requested(call_id)
    assert session.termination_requested is True
    assert session.status == CallStatus.CREATED  # Not COMPLETED yet!

    # Status callback updates state based on real Twilio events
    mgr.update_from_twilio_callback("CA12345", "ringing")
    assert session.status == CallStatus.RINGING

    mgr.update_from_twilio_callback("CA12345", "in-progress")
    assert session.status == CallStatus.IN_PROGRESS

    mgr.update_from_twilio_callback("CA12345", "completed", duration=25)
    assert session.status == CallStatus.COMPLETED
    assert session.ended_at is not None
    assert any(e.event_type == "TWILIO_STATUS_COMPLETED" for e in session.timeline)


# ---------------------------------------------------------------------------
# 4. Strict ElevenLabs Register-Call XML Validation (Reviewer Correction C)
# ---------------------------------------------------------------------------

def test_elevenlabs_valid_twiml_accepted():
    """Valid TwiML with <Response><Connect><Stream url="..."/></Connect></Response> is accepted."""
    service = ElevenLabsService()
    valid_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Response>\n'
        '    <Connect>\n'
        '        <Stream url="wss://api.elevenlabs.io/v1/convai/conversation?agent_id=123" />\n'
        '    </Connect>\n'
        '</Response>'
    )
    validated = service._validate_twiml(valid_xml)
    assert "<Stream" in validated
    assert "wss://api.elevenlabs.io" in validated


def test_elevenlabs_invalid_twiml_rejected():
    """Malformed or missing structure raises ElevenLabsIntegrationError."""
    service = ElevenLabsService()

    # Missing <Connect>
    with pytest.raises(ElevenLabsIntegrationError, match="missing <Connect>"):
        service._validate_twiml('<Response><Say>Hello</Say></Response>')

    # Missing <Stream> inside <Connect>
    with pytest.raises(ElevenLabsIntegrationError, match="missing <Stream>"):
        service._validate_twiml('<Response><Connect><Room>abc</Room></Connect></Response>')

    # Missing url in <Stream>
    with pytest.raises(ElevenLabsIntegrationError, match="missing 'url'"):
        service._validate_twiml('<Response><Connect><Stream /></Connect></Response>')

    # Invalid XML
    with pytest.raises(ElevenLabsIntegrationError, match="invalid XML"):
        service._validate_twiml('<Response><Connect>')


# ---------------------------------------------------------------------------
# 5. Twilio Outbound Dispatch Parameter Construction
# ---------------------------------------------------------------------------

def test_twilio_outbound_dispatch_parameters():
    """Verify exact Twilio REST API parameters constructed during call creation."""
    twilio_svc = TwilioService()
    mock_client = MagicMock()
    mock_call = MagicMock()
    mock_call.sid = "CA_MOCK_SID_123"
    mock_client.calls.create.return_value = mock_call

    with patch.object(twilio_svc, "_get_client", return_value=mock_client), \
         patch.object(settings, "twilio_phone_number", "+15005550006"), \
         patch.object(settings, "twilio_verified_destination_number", "+919876543210"), \
         patch.object(settings, "backend_public_url", "https://voiceshield.ngrok-free.app"):

        call_sid = twilio_svc.create_outbound_call(call_id="call-999")
        assert call_sid == "CA_MOCK_SID_123"

        mock_client.calls.create.assert_called_once_with(
            to="+919876543210",
            from_="+15005550006",
            url="https://voiceshield.ngrok-free.app/caller/twiml?call_id=call-999",
            status_callback="https://voiceshield.ngrok-free.app/caller/status-callback?call_id=call-999",
        )


# ---------------------------------------------------------------------------
# 6. Telephony Audio Decoder & Validation (Torchaudio / Python 3.13)
# ---------------------------------------------------------------------------

def test_telephony_audio_decoder_mulaw():
    """Test 8kHz mu-law decoding to 16kHz Float32 PCM tensor and telemetry calculation."""
    decoder = TelephonyAudioDecoder()

    # Synthetic mu-law payload (160 samples of 8kHz mu-law = 20ms audio)
    raw_mulaw = bytes([128] * 80 + [255] * 40 + [0] * 40)
    b64_payload = base64.b64encode(raw_mulaw).decode("ascii")

    pcm_16k, telemetry = decoder.decode_twilio_payload(b64_payload)

    # Output should be 16kHz (2x sample count of 8kHz = 320 samples)
    assert pcm_16k.ndim == 1
    assert len(pcm_16k) == 320
    assert telemetry["bytes_count"] == 160
    assert telemetry["samples_count"] == 320
    assert telemetry["nan_count"] == 0
    assert telemetry["inf_count"] == 0
    assert telemetry["rms"] > 0.0
    assert telemetry["peak"] > 0.0


def test_telephony_audio_decoder_empty_payload():
    """Empty payload returns empty tensor and 0 metrics."""
    decoder = TelephonyAudioDecoder()
    pcm, telemetry = decoder.decode_twilio_payload("")
    assert len(pcm) == 0
    assert telemetry["bytes_count"] == 0
    assert telemetry["samples_count"] == 0
    assert telemetry["percentage_zero"] == 100.0


# ---------------------------------------------------------------------------
# 7. End-to-End REST API Endpoints
# ---------------------------------------------------------------------------

def test_rest_api_caller_workflow():
    """Test full caller REST API workflow."""
    with patch.object(settings, "twilio_account_sid", "ACmock"), \
         patch.object(settings, "twilio_auth_token", "mocktoken"), \
         patch.object(settings, "twilio_phone_number", "+15005550006"), \
         patch.object(settings, "twilio_verified_destination_number", "+919876543210"), \
         patch.object(settings, "elevenlabs_api_key", "ximock"), \
         patch.object(settings, "elevenlabs_agent_id", "agentmock"), \
         patch.object(settings, "backend_public_url", "https://test.ngrok-free.app"), \
         patch.object(settings, "twilio_webhook_validate_signature", False), \
         patch("app.api.caller.twilio_service.create_outbound_call", return_value="CA_TEST_SID"):

        # 1. Check Config
        config_res = client.get("/caller/config")
        assert config_res.status_code == 200
        assert config_res.json()["configured"] is True

        # 2. Dispatch Outbound Call
        res = client.post("/caller/outbound", json={"greeting": "Hello, this is VoiceShield."})
        assert res.status_code == 201
        data = res.json()
        call_id = data["call_id"]
        assert data["call_sid"] == "CA_TEST_SID"
        assert data["destination"] == "+919876543210"

        # 3. Get Call by ID
        get_res = client.get(f"/caller/calls/{call_id}")
        assert get_res.status_code == 200
        assert get_res.json()["call_id"] == call_id
        assert get_res.json()["telemetry"]["media_delivery_status"] == MediaStreamStatus.UNPROVEN.value

        # 4. List Calls
        list_res = client.get("/caller/calls")
        assert list_res.status_code == 200
        assert any(c["call_id"] == call_id for c in list_res.json())

        # 5. Status Callback from Twilio
        status_res = client.post(
            f"/caller/status-callback?call_id={call_id}",
            data={"CallSid": "CA_TEST_SID", "CallStatus": "in-progress"},
        )
        assert status_res.status_code == 200

        # Verify updated status
        updated_call = client.get(f"/caller/calls/{call_id}").json()
        assert updated_call["status"] == CallStatus.IN_PROGRESS.value

        # 6. Request End Call
        with patch("app.api.caller.twilio_service.terminate_call") as mock_term:
            end_res = client.post(f"/caller/calls/{call_id}/end")
            assert end_res.status_code == 200
            assert end_res.json()["termination_requested"] is True
            mock_term.assert_called_once_with("CA_TEST_SID")

        # 7. Final Twilio Completed Callback
        client.post(
            f"/caller/status-callback?call_id={call_id}",
            data={"CallSid": "CA_TEST_SID", "CallStatus": "completed", "CallDuration": "30"},
        )
        final_call = client.get(f"/caller/calls/{call_id}").json()
        assert final_call["status"] == CallStatus.COMPLETED.value
        assert final_call["ended_at"] is not None


def test_twiml_endpoint_error_handling():
    """If ElevenLabs registration fails, twiml endpoint returns graceful Hangup TwiML."""
    with patch.object(settings, "twilio_webhook_validate_signature", False), \
         patch("app.api.caller.elevenlabs_service.register_call", side_effect=ElevenLabsIntegrationError("Connection failed")):
        res = client.post("/caller/twiml")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/xml"
        assert "<Hangup/>" in res.text
        assert "could not be established" in res.text


def test_signature_validation_enforcement():
    """When signature validation is enabled, missing or invalid signature returns 403."""
    with patch.object(settings, "twilio_webhook_validate_signature", True), \
         patch.object(settings, "twilio_auth_token", "secret_token_123"):
        # Without signature header -> 403 Forbidden
        res = client.post("/caller/status-callback", data={"CallSid": "CA123", "CallStatus": "ringing"})
        assert res.status_code == 403
        assert "Invalid Twilio signature" in res.json()["detail"]

