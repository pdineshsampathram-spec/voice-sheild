"""
ECAPA-TDNN speaker verification service.

Uses SpeechBrain's pretrained ECAPA-TDNN model for:
- Speaker enrollment (storing voice embeddings)
- Speaker verification (comparing incoming audio against enrolled profiles)
"""

import logging
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from app.core.config import settings
from app.schemas.analysis import SpeakerResult, SpeakerProfile

logger = logging.getLogger(__name__)


class ECAPAService:
    """Speaker verification using SpeechBrain ECAPA-TDNN.

    The model generates 192-dimensional speaker embeddings.
    Cosine similarity is used to compare embeddings.
    The default threshold (0.25) requires calibration for your data.
    """

    def __init__(self):
        self.model = None
        self._loaded = False
        self._load_time_ms: Optional[float] = None
        self._error_message: Optional[str] = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> str:
        if self._loaded:
            return "ONLINE"
        if self._error_message:
            return "ERROR"
        return "UNAVAILABLE"

    @property
    def status_message(self) -> Optional[str]:
        return self._error_message

    @property
    def load_time_ms(self) -> Optional[float]:
        return self._load_time_ms

    def load_model(self) -> bool:
        """Load the ECAPA-TDNN model from SpeechBrain."""
        try:
            start = time.time()
            self._device = settings.get_device()
            logger.info(f"Loading ECAPA-TDNN on device: {self._device}")

            from speechbrain.inference.speaker import EncoderClassifier

            run_opts = {"device": self._device}
            self.model = EncoderClassifier.from_hparams(
                source=settings.ecapa_model_id,
                savedir=str(settings.cache_path / "ecapa-tdnn"),
                run_opts=run_opts,
            )

            elapsed = (time.time() - start) * 1000
            self._load_time_ms = elapsed
            self._loaded = True
            self._error_message = None
            logger.info(f"ECAPA-TDNN loaded in {elapsed:.0f}ms on {self._device}")
            return True

        except Exception as e:
            self._error_message = str(e)
            self._loaded = False
            logger.error(f"Failed to load ECAPA-TDNN: {e}", exc_info=True)
            return False

    @torch.inference_mode()
    def get_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract speaker embedding from audio waveform.

        Args:
            waveform: 1D tensor of audio samples at 16kHz.

        Returns:
            numpy array of shape (192,) — the speaker embedding.
        """
        if not self._loaded:
            raise RuntimeError("ECAPA-TDNN model is not loaded")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        with self._lock:
            embedding = self.model.encode_batch(waveform)
            return embedding.squeeze().cpu().numpy()

    def enroll_speaker(
        self,
        speaker_name: str,
        waveform: torch.Tensor,
    ) -> SpeakerProfile:
        """Enroll a new speaker by storing their voice embedding.

        Args:
            speaker_name: Name for the speaker profile.
            waveform: 1D tensor of audio (at least a few seconds for quality).

        Returns:
            SpeakerProfile with the speaker ID and metadata.
        """
        import uuid

        embedding = self.get_embedding(waveform)
        speaker_id = str(uuid.uuid4())

        # Save embedding
        profiles_dir = settings.profiles_path
        embedding_path = profiles_dir / f"{speaker_id}.npy"
        np.save(str(embedding_path), embedding)

        # Save metadata
        meta_path = profiles_dir / f"{speaker_id}.json"
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            created_at=datetime.utcnow(),
            embedding_path=str(embedding_path),
            sample_count=1,
        )
        with open(meta_path, "w") as f:
            json.dump(profile.model_dump(mode="json"), f, default=str)

        logger.info(f"Enrolled speaker '{speaker_name}' as {speaker_id}")
        return profile

    def verify_speaker(
        self,
        waveform: torch.Tensor,
        speaker_id: str,
        threshold: float = None,
    ) -> SpeakerResult:
        """Verify if audio matches an enrolled speaker.

        Args:
            waveform: 1D tensor of audio samples at 16kHz.
            speaker_id: ID of the enrolled speaker to compare against.
            threshold: Cosine similarity threshold. Default from config.

        Returns:
            SpeakerResult with similarity score and match decision.
        """
        if threshold is None:
            threshold = settings.speaker_threshold_match

        # Load enrolled embedding
        embedding_path = settings.profiles_path / f"{speaker_id}.npy"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Speaker profile not found: {speaker_id}")

        enrolled_embedding = np.load(str(embedding_path))
        incoming_embedding = self.get_embedding(waveform)

        # Cosine similarity
        similarity = float(np.dot(enrolled_embedding, incoming_embedding) / (
            np.linalg.norm(enrolled_embedding) * np.linalg.norm(incoming_embedding) + 1e-8
        ))

        return SpeakerResult(
            model="ECAPA-TDNN",
            speaker_similarity=round(similarity, 4),
            match=similarity >= threshold,
            threshold_used=threshold,
            enrolled_speaker_id=speaker_id,
        )

    def list_profiles(self) -> list[SpeakerProfile]:
        """List all enrolled speaker profiles."""
        profiles = []
        profiles_dir = settings.profiles_path
        for meta_path in profiles_dir.glob("*.json"):
            with open(meta_path) as f:
                data = json.load(f)
                profiles.append(SpeakerProfile(**data))
        return profiles

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get a specific speaker profile."""
        meta_path = settings.profiles_path / f"{speaker_id}.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            data = json.load(f)
            return SpeakerProfile(**data)

    def delete_profile(self, speaker_id: str) -> bool:
        """Delete a speaker profile."""
        profiles_dir = settings.profiles_path
        embedding_path = profiles_dir / f"{speaker_id}.npy"
        meta_path = profiles_dir / f"{speaker_id}.json"

        deleted = False
        if embedding_path.exists():
            embedding_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        return deleted


# Singleton instance
ecapa_service = ECAPAService()
