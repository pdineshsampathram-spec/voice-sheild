"""
VoiceShield configuration module.

All thresholds are configurable and require calibration with your specific
audio data. The defaults are starting points, not validated thresholds.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # Model identifiers
    spectra_model_id: str = "lab260/Spectra-AASIST"
    ecapa_model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_cache_dir: str = "./model_cache"

    # Device selection: "auto", "cpu", "cuda"
    device: str = "auto"

    # Audio settings
    sample_rate: int = 16000
    spectra_window_samples: int = 64600  # ~4.04 seconds at 16kHz
    sliding_window_overlap: float = 0.5  # 50% overlap for live streaming

    # Risk engine thresholds — these are configurable starting defaults
    # Calibrate with your test audio for production use
    spoof_threshold_critical: float = 0.85
    spoof_threshold_high: float = 0.70
    spoof_threshold_medium: float = 0.40
    speaker_threshold_match: float = 0.25
    risk_window_count: int = 5  # Number of recent windows to consider
    risk_escalation_count: int = 3  # Consecutive elevated windows before escalating
    risk_debounce_seconds: float = 2.0  # Minimum time between alert escalations

    # Spectra-AASIST classify threshold (from model's default)
    spectra_classify_threshold: float = -1.0625009

    # Database
    database_url: str = "postgresql://voiceshield:voiceshield@localhost:5432/voiceshield"
    # Set to "sqlite" to use SQLite for development without PostgreSQL
    database_backend: str = "postgresql"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Speaker profiles & enrollment samples
    speaker_profiles_dir: str = "./speaker_profiles"
    enrollment_samples_dir: str = "./enrollment_samples"

    # Analyzed recordings cache
    recordings_cache_dir: str = "./recordings_cache"

    # Challenge verification
    challenge_timeout_seconds: int = 30

    # Twilio & ElevenLabs Outbound Caller Agent
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_api_key_sid: str = ""
    twilio_api_key_secret: str = ""
    twilio_phone_number: str = ""
    twilio_verified_destination_number: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_agent_id: str = ""
    backend_public_url: str = ""
    twilio_webhook_validate_signature: bool = True

    model_config = {"env_file": [".env", "../.env"], "env_file_encoding": "utf-8", "extra": "ignore"}

    def validate_caller_config(self) -> list[str]:
        """Check for missing Twilio/ElevenLabs configuration variables."""
        missing = []
        if not self.twilio_account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not self.twilio_auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not self.twilio_phone_number:
            missing.append("TWILIO_PHONE_NUMBER")
        if not self.twilio_verified_destination_number:
            missing.append("TWILIO_VERIFIED_DESTINATION_NUMBER")
        if not self.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY")
        if not self.elevenlabs_agent_id:
            missing.append("ELEVENLABS_AGENT_ID")
        return missing

    def is_caller_configured(self) -> bool:
        """Returns True if all required caller variables are present."""
        return len(self.validate_caller_config()) == 0

    def get_masked_caller_config(self) -> dict:
        """Return caller configuration with secrets securely masked."""
        def mask(val: str) -> str:
            if not val:
                return "<NOT SET>"
            if len(val) <= 6:
                return "***"
            return f"{val[:3]}...{val[-3:]}"

        return {
            "twilio_account_sid": mask(self.twilio_account_sid),
            "twilio_auth_token": "***" if self.twilio_auth_token else "<NOT SET>",
            "twilio_api_key_sid": mask(self.twilio_api_key_sid) if self.twilio_api_key_sid else "<NOT SET>",
            "twilio_phone_number": self.twilio_phone_number or "<NOT SET>",
            "twilio_verified_destination_number": self.twilio_verified_destination_number or "<NOT SET>",
            "elevenlabs_api_key": "***" if self.elevenlabs_api_key else "<NOT SET>",
            "elevenlabs_agent_id": mask(self.elevenlabs_agent_id),
            "backend_public_url": self.backend_public_url or "<NOT SET>",
            "signature_validation": self.twilio_webhook_validate_signature,
            "configured": self.is_caller_configured(),
            "missing_variables": self.validate_caller_config(),
        }

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    @property
    def profiles_path(self) -> Path:
        path = Path(self.speaker_profiles_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_path(self) -> Path:
        path = Path(self.model_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def recordings_path(self) -> Path:
        path = Path(self.recordings_cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def enrollment_samples_path(self) -> Path:
        path = Path(self.enrollment_samples_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
