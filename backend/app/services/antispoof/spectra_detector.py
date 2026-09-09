"""
Spectra-AASIST anti-spoofing detector service.

Loads the lab260/Spectra-AASIST model using PyTorchModelHubMixin.from_pretrained().
The model uses a Wav2Vec2 XLS-R-300M SSL encoder followed by an AASIST classifier.

Model output: logits (batch, 2) — index 0 = spoof, index 1 = bonafide.
"""

import logging
import time
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.schemas.analysis import SpoofResult

logger = logging.getLogger(__name__)


class SpectraAASISTDetector:
    """Adapter for the Spectra-AASIST voice anti-spoofing model.

    The model is loaded once at startup and reused for all inference requests.
    Supports both GPU (CUDA) and CPU inference.
    """

    def __init__(self):
        self.model = None
        self.device: Optional[str] = None
        self._loaded = False
        self._load_time_ms: Optional[float] = None
        self._error_message: Optional[str] = None
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
        """Load the Spectra-AASIST model from HuggingFace.

        The model repository contains model.py with the SpectraAASIST class
        that uses PyTorchModelHubMixin, allowing direct from_pretrained() loading.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            start = time.time()
            self.device = settings.get_device()
            logger.info(f"Loading Spectra-AASIST on device: {self.device}")

            # The model uses PyTorchModelHubMixin which provides from_pretrained()
            # This downloads model.py + model.safetensors from the HF repo
            # and instantiates SpectraAASIST with the saved weights
            from huggingface_hub import hf_hub_download
            import importlib.util

            model_id = settings.spectra_model_id
            cache_dir = str(settings.cache_path)

            # Download model.py from the repo
            model_py_path = hf_hub_download(
                repo_id=model_id,
                filename="model.py",
                cache_dir=cache_dir,
            )

            # Dynamically import the model module
            spec = importlib.util.spec_from_file_location("spectra_model", model_py_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["spectra_model"] = module
            spec.loader.exec_module(module)

            # Get the SpectraAASIST class and use from_pretrained
            SpectraAASIST = module.SpectraAASIST
            self.model = SpectraAASIST.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            elapsed = (time.time() - start) * 1000
            self._load_time_ms = elapsed
            self._loaded = True
            self._error_message = None
            logger.info(f"Spectra-AASIST loaded successfully in {elapsed:.0f}ms on {self.device}")
            return True

        except Exception as e:
            self._error_message = str(e)
            self._loaded = False
            logger.error(f"Failed to load Spectra-AASIST: {e}", exc_info=True)
            return False

    @torch.inference_mode()
    def predict(self, audio_tensor: torch.Tensor) -> SpoofResult:
        """Run inference on a single audio window.

        Args:
            audio_tensor: 1D tensor of audio samples (should be 64600 samples at 16kHz).
                         Will be padded/truncated if necessary.

        Returns:
            SpoofResult with normalized scores.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Spectra-AASIST model is not loaded")

        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor).float()

        # Ensure correct shape: (batch=1, samples)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Pad or truncate to expected window size
        window_size = settings.spectra_window_samples
        current_len = audio_tensor.shape[1]

        if current_len < window_size:
            padding = torch.zeros(1, window_size - current_len)
            audio_tensor = torch.cat([audio_tensor, padding], dim=1)
        elif current_len > window_size:
            audio_tensor = audio_tensor[:, :window_size]

        with self._lock:
            audio_tensor = audio_tensor.to(self.device)

            # Forward pass — output shape: (batch, 2)
            # ASVspoof / SpectraAASIST convention:
            # Index 0 = spoof logit, Index 1 = bonafide logit
            # In SpectraAASIST.classify(): forward(x)[:, 1] > threshold evaluates bonafide.
            logits = self.model(audio_tensor)

            # Apply softmax to get normalized probabilities
            probs = F.softmax(logits, dim=1)

            spoof_score = probs[0, 0].item()
            bonafide_score = probs[0, 1].item()
            raw_logit_spoof = logits[0, 0].item()
            raw_logit_bonafide = logits[0, 1].item()

        # Determine label
        label = "spoof" if spoof_score > bonafide_score else "bonafide"

        return SpoofResult(
            model="Spectra-AASIST",
            label=label,
            spoof_score=round(spoof_score, 4),
            bonafide_score=round(bonafide_score, 4),
            raw_logit_spoof=round(raw_logit_spoof, 4),
            raw_logit_bonafide=round(raw_logit_bonafide, 4),
        )

    @torch.inference_mode()
    def predict_batch(self, audio_tensors: list[torch.Tensor]) -> list[SpoofResult]:
        """Run inference on multiple audio windows at once.

        Args:
            audio_tensors: List of 1D tensors, each 64600 samples.

        Returns:
            List of SpoofResult, one per input.
        """
        if not self._loaded:
            raise RuntimeError("Spectra-AASIST model is not loaded")

        window_size = settings.spectra_window_samples
        padded = []
        for t in audio_tensors:
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.shape[1] < window_size:
                t = torch.cat([t, torch.zeros(1, window_size - t.shape[1])], dim=1)
            elif t.shape[1] > window_size:
                t = t[:, :window_size]
            padded.append(t)

        with self._lock:
            batch = torch.cat(padded, dim=0).to(self.device)
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)

        results = []
        for i in range(len(audio_tensors)):
            spoof_score = probs[i, 0].item()
            bonafide_score = probs[i, 1].item()
            label = "spoof" if spoof_score > bonafide_score else "bonafide"
            results.append(SpoofResult(
                model="Spectra-AASIST",
                label=label,
                spoof_score=round(spoof_score, 4),
                bonafide_score=round(bonafide_score, 4),
                raw_logit_spoof=round(logits[i, 0].item(), 4),
                raw_logit_bonafide=round(logits[i, 1].item(), 4),
            ))

        return results


# Singleton instance — loaded once at app startup
spectra_detector = SpectraAASISTDetector()
