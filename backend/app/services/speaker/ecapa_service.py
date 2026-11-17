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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import hashlib
import os
import uuid
import torch
import numpy as np

from app.core.config import settings
from app.schemas.analysis import (
    SpeakerResult, SpeakerProfile, SpeakerIdentificationResult, SpeakerCandidate, MatchStatus,
    EnrollmentSampleReference
)

logger = logging.getLogger(__name__)


class ECAPAService:
    """Speaker verification using SpeechBrain ECAPA-TDNN.

    The model generates 192-dimensional speaker embeddings.
    Cosine similarity is used to compare unit-normalized embeddings.
    Multi-sample enrollment creates a normalized centroid representation.
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
        """Extract speaker embedding from audio waveform and normalize to unit norm.

        Args:
            waveform: 1D or 2D tensor of audio samples at 16kHz.

        Returns:
            numpy array of shape (192,) — unit-normalized speaker embedding.
        """
        if not self._loaded:
            raise RuntimeError("ECAPA-TDNN model is not loaded")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        with self._lock:
            embedding = self.model.encode_batch(waveform)
            raw = embedding.squeeze().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(raw)
            if norm > 0:
                raw = raw / (norm + 1e-8)
            return raw

    def enroll_speaker(
        self,
        speaker_name: str,
        waveform: torch.Tensor,
        existing_speaker_id: Optional[str] = None,
        sample_bytes: Optional[bytes] = None,
        sample_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> SpeakerProfile:
        """Enroll a new speaker or update an existing profile with a multi-sample centroid.

        Retains explicit EnrollmentSampleReference for exact sample playback.

        Args:
            speaker_name: Name for the speaker profile.
            waveform: 1D tensor of audio (at least a few seconds for quality).
            existing_speaker_id: Optional ID to append sample to existing profile.
            sample_bytes: Raw audio bytes for persistent exact playback.
            sample_filename: Original filename of the sample.
            mime_type: Audio MIME type.

        Returns:
            SpeakerProfile with updated metadata, samples, and centroid embedding.
        """
        new_embedding = self.get_embedding(waveform)
        profiles_dir = settings.profiles_path
        samples_dir = settings.enrollment_samples_path

        # Save sample artifact if bytes provided
        sample_ref = None
        if sample_bytes and len(sample_bytes) > 0:
            sample_id = f"smp_{uuid.uuid4().hex[:12]}"
            sample_sha256 = hashlib.sha256(sample_bytes).hexdigest()
            ext = os.path.splitext(sample_filename or "sample.webm")[1].lower() or ".webm"
            sample_file_path = samples_dir / f"{sample_id}{ext}"
            with open(sample_file_path, "wb") as sf:
                sf.write(sample_bytes)
            duration_sec = round(float(len(waveform) / 16000), 2)
            sample_emb_path = samples_dir / f"{sample_id}.npy"
            np.save(str(sample_emb_path), new_embedding)
            sample_ref = EnrollmentSampleReference(
                sample_id=sample_id,
                original_filename=sample_filename or f"sample_{sample_id}{ext}",
                mime_type=mime_type or "audio/webm",
                duration_seconds=duration_sec,
                sha256=sample_sha256,
                playback_available=True,
                playback_url=f"/api/enrollment/samples/{sample_id}",
                created_at=datetime.now(timezone.utc),
            )

        target_id = existing_speaker_id
        if not target_id:
            # Check if a profile with the same name already exists
            for existing in self.list_profiles():
                if existing.speaker_name.strip().lower() == speaker_name.strip().lower():
                    target_id = existing.speaker_id
                    break

        if target_id and (profiles_dir / f"{target_id}.npy").exists() and (profiles_dir / f"{target_id}.json").exists():
            # Update existing profile with running centroid
            with open(profiles_dir / f"{target_id}.json") as f:
                prev_data = json.load(f)
            prev_profile = SpeakerProfile(**prev_data)
            prev_embedding = np.load(str(profiles_dir / f"{target_id}.npy"))

            # Running centroid calculation
            n = prev_profile.sample_count
            combined = prev_embedding * n + new_embedding
            norm = np.linalg.norm(combined)
            centroid_embedding = combined / (norm + 1e-8) if norm > 0 else new_embedding

            np.save(str(profiles_dir / f"{target_id}.npy"), centroid_embedding)

            samples = list(prev_profile.samples) if prev_profile.samples else []
            if sample_ref:
                samples.append(sample_ref)

            updated_profile = SpeakerProfile(
                speaker_id=target_id,
                speaker_name=speaker_name,
                created_at=prev_profile.created_at,
                updated_at=datetime.now(timezone.utc),
                embedding_path=str(profiles_dir / f"{target_id}.npy"),
                sample_count=n + 1,
                quality_status="VALID",
                samples=samples,
            )
            with open(profiles_dir / f"{target_id}.json", "w") as f:
                json.dump(updated_profile.model_dump(mode="json"), f, default=str)

            logger.info(
                f"Updated speaker '{speaker_name}' ({target_id}) centroid with sample {n + 1}"
            )
            return updated_profile

        # New speaker profile
        speaker_id = target_id or str(uuid.uuid4())
        embedding_path = profiles_dir / f"{speaker_id}.npy"
        np.save(str(embedding_path), new_embedding)

        meta_path = profiles_dir / f"{speaker_id}.json"
        samples = [sample_ref] if sample_ref else []
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            created_at=datetime.now(timezone.utc),
            updated_at=None,
            embedding_path=str(embedding_path),
            sample_count=1,
            quality_status="VALID",
            samples=samples,
        )
        with open(meta_path, "w") as f:
            json.dump(profile.model_dump(mode="json"), f, default=str)

        logger.info(f"Enrolled new speaker '{speaker_name}' as {speaker_id}")
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

        embedding_path = settings.profiles_path / f"{speaker_id}.npy"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Speaker profile not found: {speaker_id}")

        enrolled_embedding = np.load(str(embedding_path))
        incoming_embedding = self.get_embedding(waveform)

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

    def identify_speaker(
        self,
        waveform_or_embedding: Union[torch.Tensor, np.ndarray],
        threshold: float = None,
    ) -> SpeakerIdentificationResult:
        """Automatically compare audio against ALL enrolled speaker profiles.

        Ranks candidates by cosine similarity. Does NOT pretend a percentage proves identity.

        Args:
            waveform_or_embedding: 1D tensor of audio or precomputed 192-dim embedding.
            threshold: Similarity threshold to mark 'MATCH'. Defaults to config.

        Returns:
            SpeakerIdentificationResult with best_match and ranked alternatives.
        """
        if threshold is None:
            threshold = settings.speaker_threshold_match

        if not self._loaded:
            return SpeakerIdentificationResult(
                best_match=None,
                alternatives=[],
                status=MatchStatus.UNAVAILABLE,
                reason="ECAPA_UNAVAILABLE",
            )

        # Silence / low-quality guard: abstain from speaker identification if signal is insufficient
        if isinstance(waveform_or_embedding, torch.Tensor):
            wf = waveform_or_embedding.squeeze()
            num_samples = wf.numel()
            if num_samples == 0:
                return SpeakerIdentificationResult(
                    best_match=None,
                    alternatives=[],
                    status=MatchStatus.INSUFFICIENT_EVIDENCE,
                    reason="NO_SPEECH",
                )
            rms = float(torch.sqrt(torch.mean(wf.float() ** 2)))
            if rms < 0.0005:
                return SpeakerIdentificationResult(
                    best_match=None,
                    alternatives=[],
                    status=MatchStatus.INSUFFICIENT_EVIDENCE,
                    reason="NO_SPEECH",
                )
            if rms < 0.0015:
                return SpeakerIdentificationResult(
                    best_match=None,
                    alternatives=[],
                    status=MatchStatus.INSUFFICIENT_EVIDENCE,
                    reason="LOW_SIGNAL",
                )
            if num_samples < 4000:
                return SpeakerIdentificationResult(
                    best_match=None,
                    alternatives=[],
                    status=MatchStatus.INSUFFICIENT_EVIDENCE,
                    reason="TOO_SHORT",
                )
            waveform_or_embedding = wf

        profiles = [p for p in self.list_profiles() if p.sample_count > 0 and p.quality_status != "EMPTY"]
        if not profiles:
            return SpeakerIdentificationResult(
                best_match=None,
                alternatives=[],
                status=MatchStatus.NO_ENROLLED_PROFILES,
                reason="NO_ENROLLED_PROFILES",
            )

        if isinstance(waveform_or_embedding, torch.Tensor):
            query_embedding = self.get_embedding(waveform_or_embedding)
        else:
            query_embedding = waveform_or_embedding
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / (norm + 1e-8)

        candidates: list[SpeakerCandidate] = []
        profiles_dir = settings.profiles_path

        for p in profiles:
            emb_path = profiles_dir / f"{p.speaker_id}.npy"
            if not emb_path.exists():
                continue
            try:
                enrolled_emb = np.load(str(emb_path))
                sim = float(np.dot(enrolled_emb, query_embedding) / (
                    np.linalg.norm(enrolled_emb) * np.linalg.norm(query_embedding) + 1e-8
                ))
                sim_rounded = round(sim, 4)
                candidate_status = MatchStatus.MATCH if sim >= threshold else MatchStatus.NO_CONFIDENT_MATCH
                candidates.append(SpeakerCandidate(
                    speaker_id=p.speaker_id,
                    speaker_name=p.speaker_name,
                    similarity=sim_rounded,
                    status=candidate_status,
                ))
            except Exception as e:
                logger.warning(f"Error reading profile {p.speaker_id}: {e}")

        if not candidates:
            return SpeakerIdentificationResult(
                best_match=None,
                alternatives=[],
                status=MatchStatus.NO_ENROLLED_PROFILES,
                reason="NO_ENROLLED_PROFILES",
            )

        # Rank candidates by descending similarity
        candidates.sort(key=lambda c: c.similarity, reverse=True)
        best = candidates[0]
        alternatives = candidates[1:]

        overall_status = best.status

        return SpeakerIdentificationResult(
            best_match=best,
            alternatives=alternatives,
            status=overall_status,
        )

    def list_profiles(self) -> list[SpeakerProfile]:
        """List all enrolled speaker profiles, verifying sample artifact availability."""
        profiles = []
        profiles_dir = settings.profiles_path
        for meta_path in profiles_dir.glob("*.json"):
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                prof = SpeakerProfile(**data)
                # Verify playback availability for each sample
                for s in prof.samples:
                    s.playback_available = (self.get_sample_path(s.sample_id) is not None)
                profiles.append(prof)
            except Exception as e:
                logger.warning(f"Failed to load profile {meta_path}: {e}")
        return sorted(profiles, key=lambda p: p.speaker_name.lower())

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get a single speaker profile by ID."""
        profiles_dir = settings.profiles_path
        meta_path = profiles_dir / f"{speaker_id}.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            data = json.load(f)
        prof = SpeakerProfile(**data)
        for s in prof.samples:
            s.playback_available = (self.get_sample_path(s.sample_id) is not None)
        return prof

    def get_sample_path(self, sample_id: str) -> Optional[Path]:
        """Safely retrieve the path to an enrollment sample artifact with containment validation."""
        import re
        if not sample_id or not re.match(r'^[a-zA-Z0-9_-]+$', sample_id):
            return None

        samples_dir = settings.enrollment_samples_path.resolve()
        for f in samples_dir.glob(f"{sample_id}.*"):
            if f.is_file() and not f.name.endswith(".json") and not f.name.endswith(".npy"):
                try:
                    resolved = f.resolve()
                    if resolved.parent == samples_dir:
                        return resolved
                except (OSError, ValueError):
                    return None
        return None

    def delete_sample(self, sample_id: str) -> Optional[SpeakerProfile]:
        """Delete an individual enrollment voice recording and safely recompute the profile centroid.

        Invariants:
        - Validates sample_id regex
        - Finds the profile that owns the sample
        - Removes the audio artifact and its embedding file
        - Recomputes centroid from remaining valid sample embeddings
        - If sample_count reaches 0, unlinks centroid and sets quality_status="EMPTY"
        - Does NOT leave a fake VALID profile with no underlying samples
        - Returns updated SpeakerProfile, or None if sample_id not found
        """
        import re
        if not sample_id or not re.match(r'^[a-zA-Z0-9_-]+$', sample_id):
            return None

        profiles_dir = settings.profiles_path
        samples_dir = settings.enrollment_samples_path

        # Find the owning profile
        target_profile: Optional[SpeakerProfile] = None
        for p in self.list_profiles():
            if any(s.sample_id == sample_id for s in p.samples):
                target_profile = p
                break

        if not target_profile:
            return None

        # 1. Unlink audio sample artifact
        audio_path = self.get_sample_path(sample_id)
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete sample audio {audio_path}: {e}")

        # 2. Unlink sample embedding if present
        emb_file = samples_dir / f"{sample_id}.npy"
        if emb_file.exists():
            try:
                emb_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete sample embedding {emb_file}: {e}")

        # 3. Update samples list
        remaining_samples = [s for s in target_profile.samples if s.sample_id != sample_id]

        if len(remaining_samples) == 0:
            # All samples deleted -> profile becomes EMPTY (no fake valid centroid)
            centroid_path = profiles_dir / f"{target_profile.speaker_id}.npy"
            if centroid_path.exists():
                try:
                    centroid_path.unlink()
                except OSError:
                    pass

            target_profile.samples = []
            target_profile.sample_count = 0
            target_profile.quality_status = "EMPTY"
            target_profile.updated_at = datetime.now(timezone.utc)

            with open(profiles_dir / f"{target_profile.speaker_id}.json", "w") as f:
                json.dump(target_profile.model_dump(mode="json"), f, default=str)

            logger.info(f"Deleted last sample '{sample_id}' from profile '{target_profile.speaker_name}'. Profile marked EMPTY.")
            return target_profile

        # 4. Recompute centroid from remaining sample embeddings
        rem_embeddings = []
        for s in remaining_samples:
            s_emb_path = samples_dir / f"{s.sample_id}.npy"
            if s_emb_path.exists():
                try:
                    rem_embeddings.append(np.load(str(s_emb_path)))
                except Exception as ex:
                    logger.warning(f"Could not load embedding for {s.sample_id}: {ex}")
            else:
                # If audio artifact exists, re-extract embedding
                s_audio_path = self.get_sample_path(s.sample_id)
                if s_audio_path and self._loaded:
                    try:
                        import soundfile as sf
                        audio_data, sr = sf.read(str(s_audio_path), dtype="float32")
                        import torchaudio.transforms as T
                        tensor = torch.from_numpy(audio_data)
                        if sr != 16000:
                            resampler = T.Resample(orig_freq=sr, new_freq=16000)
                            tensor = resampler(tensor)
                        emb = self.get_embedding(tensor)
                        np.save(str(s_emb_path), emb)
                        rem_embeddings.append(emb)
                    except Exception as ex:
                        logger.warning(f"Could not re-extract embedding for {s.sample_id}: {ex}")

        if rem_embeddings:
            centroid = np.mean(rem_embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / (norm + 1e-8)
            np.save(str(profiles_dir / f"{target_profile.speaker_id}.npy"), centroid)

        target_profile.samples = remaining_samples
        target_profile.sample_count = len(remaining_samples)
        target_profile.quality_status = "VALID"
        target_profile.updated_at = datetime.now(timezone.utc)

        with open(profiles_dir / f"{target_profile.speaker_id}.json", "w") as f:
            json.dump(target_profile.model_dump(mode="json"), f, default=str)

        logger.info(
            f"Deleted sample '{sample_id}'. Recomputed centroid for profile '{target_profile.speaker_name}' "
            f"from {len(remaining_samples)} remaining sample(s)."
        )
        return target_profile

    def delete_profile(self, speaker_id: str) -> bool:
        """Delete a speaker profile and all associated enrollment samples."""
        profiles_dir = settings.profiles_path
        embedding_path = profiles_dir / f"{speaker_id}.npy"
        meta_path = profiles_dir / f"{speaker_id}.json"

        # Check for samples to delete
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                prof = SpeakerProfile(**data)
                for s in prof.samples:
                    p = self.get_sample_path(s.sample_id)
                    if p and p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to delete sample files for profile {speaker_id}: {e}")

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
