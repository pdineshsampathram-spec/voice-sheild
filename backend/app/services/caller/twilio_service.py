"""
VoiceShield — Twilio Outbound Calling Service.

Manages outbound call creation, provider termination requests,
and webhook signature verification.
"""

import logging
from typing import Optional, Dict
from twilio.rest import Client
from twilio.request_validator import RequestValidator

from app.core.config import settings

logger = logging.getLogger(__name__)


class TwilioService:
    """Interface to Twilio REST API for outbound call orchestration."""

    def __init__(self):
        self._client: Optional[Client] = None
        self._validator: Optional[RequestValidator] = None

    def _get_client(self) -> Client:
        """Lazy-initialize and return the Twilio REST client."""
        if not settings.twilio_account_sid:
            raise ValueError(
                "Twilio credentials not configured. Please set TWILIO_ACCOUNT_SID in .env"
            )
        if self._client is None:
            if settings.twilio_api_key_sid and settings.twilio_api_key_secret:
                self._client = Client(
                    settings.twilio_api_key_sid.strip(),
                    settings.twilio_api_key_secret.strip(),
                    account_sid=settings.twilio_account_sid.strip(),
                )
            elif settings.twilio_auth_token:
                self._client = Client(
                    settings.twilio_account_sid.strip(),
                    settings.twilio_auth_token.strip(),
                )
            else:
                raise ValueError(
                    "Neither TWILIO_API_KEY_SID/SECRET nor TWILIO_AUTH_TOKEN configured in .env"
                )
        return self._client

    def _get_validator(self) -> RequestValidator:
        """Lazy-initialize Twilio request validator."""
        if self._validator is None:
            if not settings.twilio_auth_token:
                raise ValueError("TWILIO_AUTH_TOKEN required for signature verification.")
            self._validator = RequestValidator(settings.twilio_auth_token)
        return self._validator

    def create_outbound_call(
        self,
        call_id: str,
        greeting: Optional[str] = None,
    ) -> str:
        """Create an outbound call to the single configured verified number.

        STRICT ENFORCEMENT:
        Destination is unconditionally locked to settings.twilio_verified_destination_number.
        """
        verified_number = settings.twilio_verified_destination_number.strip()
        from_number = settings.twilio_phone_number.strip()
        public_url = settings.backend_public_url.rstrip("/")

        if not verified_number:
            raise ValueError(
                "TWILIO_VERIFIED_DESTINATION_NUMBER is not set. "
                "For trial accounts, calling is strictly restricted to your verified destination."
            )
        if not from_number:
            raise ValueError("TWILIO_PHONE_NUMBER is not configured.")
        if not public_url:
            raise ValueError(
                "BACKEND_PUBLIC_URL is not set. Twilio requires an externally accessible HTTPS "
                "URL (e.g. ngrok tunnel) for TwiML and status callbacks."
            )

        client = self._get_client()

        twiml_url = f"{public_url}/caller/twiml?call_id={call_id}"
        status_callback_url = f"{public_url}/caller/status-callback?call_id={call_id}"

        logger.info(
            f"Initiating Twilio call: From={from_number} -> To={verified_number} "
            f"(call_id={call_id})"
        )

        call = client.calls.create(
            to=verified_number,
            from_=from_number,
            url=twiml_url,
            status_callback=status_callback_url,
        )

        logger.info(f"Twilio call dispatched successfully. Call SID: {call.sid}")
        return call.sid

    def terminate_call(self, call_sid: str) -> None:
        """Request Twilio to hang up / complete an ongoing call."""
        client = self._get_client()
        logger.info(f"Requesting Twilio to terminate call SID: {call_sid}")
        client.calls(call_sid).update(status="completed")

    def validate_webhook_signature(
        self,
        url: str,
        params: Dict[str, str],
        signature: Optional[str],
    ) -> bool:
        """Verify the X-Twilio-Signature header on incoming webhooks."""
        if not settings.twilio_webhook_validate_signature:
            logger.debug("Twilio webhook signature verification is disabled by config.")
            return True

        if not signature:
            logger.warning("Missing X-Twilio-Signature header on webhook request.")
            return False

        try:
            validator = self._get_validator()
            is_valid = validator.validate(url, params, signature)
            if not is_valid:
                logger.warning(f"Invalid Twilio signature for webhook URL: {url}")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating Twilio signature: {e}")
            return False


# Global singleton
twilio_service = TwilioService()
