"""
VoiceShield — ElevenLabs Conversational AI Telephony Service.

Connects Twilio calls to ElevenLabs conversational agents using the official
register-call API with strict response validation.
"""

import json
import logging
import xml.etree.ElementTree as ET
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

ELEVENLABS_REGISTER_CALL_URL = "https://api.elevenlabs.io/v1/convai/twilio/register-call"


class ElevenLabsIntegrationError(Exception):
    """Raised when ElevenLabs register-call fails or returns unexpected TwiML structure."""
    pass


class ElevenLabsService:
    """Interface to ElevenLabs Conversational AI for Twilio integration."""

    async def register_call(
        self,
        agent_id: Optional[str] = None,
        from_number: Optional[str] = None,
        to_number: Optional[str] = None,
        greeting: Optional[str] = None,
    ) -> str:
        """Call ElevenLabs register-call endpoint and validate returned TwiML.

        STRICT VALIDATION (Reviewer Correction C):
        HTTP 200 -> Validate content -> Parse XML -> Validate <Response> root
        -> Validate <Connect> and <Stream> elements -> Return TwiML.
        Raises ElevenLabsIntegrationError if structure is invalid.
        """
        active_agent_id = (agent_id or settings.elevenlabs_agent_id).strip()
        active_from = (from_number or settings.twilio_phone_number).strip()
        active_to = (to_number or settings.twilio_verified_destination_number).strip()
        api_key = settings.elevenlabs_api_key.strip()

        if not api_key:
            raise ElevenLabsIntegrationError(
                "ELEVENLABS_API_KEY is not configured in .env"
            )
        if not active_agent_id:
            raise ElevenLabsIntegrationError(
                "ELEVENLABS_AGENT_ID is not configured in .env"
            )

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "agent_id": active_agent_id,
            "from_number": active_from,
            "to_number": active_to,
            "direction": "outbound",
        }

        if greeting:
            payload["conversation_initiation_client_data"] = {
                "conversation_config_override": {
                    "agent": {
                        "first_message": greeting
                    }
                }
            }

        logger.info(
            f"Calling ElevenLabs register-call for agent_id={active_agent_id[:6]}... "
            f"From={active_from} To={active_to}"
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(
                    ELEVENLABS_REGISTER_CALL_URL,
                    headers=headers,
                    json=payload,
                )
            except httpx.RequestError as e:
                raise ElevenLabsIntegrationError(
                    f"Network error contacting ElevenLabs register-call: {e}"
                )

        if response.status_code != 200:
            raise ElevenLabsIntegrationError(
                f"ElevenLabs register-call failed: HTTP {response.status_code} - {response.text}"
            )

        raw_content = response.text.strip()
        if not raw_content:
            raise ElevenLabsIntegrationError(
                "ElevenLabs returned empty response body from register-call"
            )

        # Some API responses may enclose TwiML in a JSON wrapper {"twiml": "..."}
        twiml_string = raw_content
        if raw_content.startswith("{") and raw_content.endswith("}"):
            try:
                parsed_json = json.loads(raw_content)
                if "twiml" in parsed_json:
                    twiml_string = parsed_json["twiml"].strip()
                elif "response" in parsed_json:
                    twiml_string = parsed_json["response"].strip()
            except Exception as e:
                raise ElevenLabsIntegrationError(
                    f"Failed to parse JSON response from ElevenLabs: {e}"
                )

        # Strict XML Validation
        return self._validate_twiml(twiml_string)

    def _validate_twiml(self, twiml_string: str) -> str:
        """Strictly validate that twiml_string is valid TwiML containing <Response><Connect><Stream>."""
        try:
            root = ET.fromstring(twiml_string)
        except ET.ParseError as e:
            raise ElevenLabsIntegrationError(
                f"ElevenLabs returned invalid XML: {e}\nPayload: {twiml_string[:200]}"
            )

        # Strip XML namespace if present
        tag_name = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag_name.lower() != "response":
            raise ElevenLabsIntegrationError(
                f"Expected root element <Response>, got <{tag_name}>"
            )

        # Look for <Connect> verb
        connect_node = None
        for child in root:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag.lower() == "connect":
                connect_node = child
                break

        if connect_node is None:
            raise ElevenLabsIntegrationError(
                "Invalid TwiML from ElevenLabs: missing <Connect> element"
            )

        # Look for <Stream> inside <Connect>
        stream_node = None
        for subchild in connect_node:
            subchild_tag = subchild.tag.split("}")[-1] if "}" in subchild.tag else subchild.tag
            if subchild_tag.lower() == "stream":
                stream_node = subchild
                break

        if stream_node is None:
            raise ElevenLabsIntegrationError(
                "Invalid TwiML from ElevenLabs: missing <Stream> element inside <Connect>"
            )

        stream_url = stream_node.attrib.get("url") or stream_node.attrib.get("Url")
        if not stream_url:
            raise ElevenLabsIntegrationError(
                "Invalid TwiML from ElevenLabs: <Stream> element missing 'url' attribute"
            )

        logger.info("ElevenLabs register-call TwiML validated successfully.")
        return twiml_string


# Global singleton
elevenlabs_service = ElevenLabsService()
