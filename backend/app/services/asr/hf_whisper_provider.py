"""
Hugging Face ASR Provider for VoiceShield.

Provides local speech-to-text using Hugging Face models (CTranslate2 & Transformers backends).
Does not use external cloud APIs or paid services.
All inference runs locally on canonical 16 kHz Float32 audio.

Supports:
- Systran/faster-whisper-base (Hugging Face Hub model)
- Systran/faster-whisper-tiny (Hugging Face Hub model)
- Hugging Face transformers pipeline fallback (when PyTorch safetensors are cached)
"""

import logging
import time
import threading
from typing import Optional

import torch
import numpy as np

from app.services.asr.asr_provider import ASRProvider, ASRResult, ASRModelInfo

logger = logging.getLogger(__name__)


class HFWhisperProvider(ASRProvider):
    """
    Hugging Face Whisper ASR provider.
    
    Operates on complete speech segments (from upstream Silero VAD),
    preventing hallucination loops and chunk fragmentation.
    """

    def __init__(
        self,
        model_name: str = "base",
        backend: str = "ctranslate2",  # "ctranslate2" or "transformers"
        beam_size: int = 5,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: Optional[float] = -1.0,
        device: str = "auto",
    ):
        self._model_name = model_name
        self._backend = backend
        self._beam_size = beam_size
        self._no_speech_threshold = no_speech_threshold
        self._log_prob_threshold = log_prob_threshold
        self._device = device

        self._model = None
        self._pipeline = None
        self._loaded = False
        self._load_time_ms: float = 0.0
        self._inference_times: list[float] = []
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend(self) -> str:
        return self._backend

    def load_model(self) -> bool:
        """Load the local Hugging Face ASR model."""
        start = time.time()

        if self._backend == "transformers":
            try:
                from transformers import pipeline
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                model_id = f"openai/whisper-{self._model_name}" if not self._model_name.startswith("openai/") else self._model_name
                logger.info(f"Loading Hugging Face Transformers pipeline '{model_id}' on {device}...")
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=device,
                    chunk_length_s=30,
                )
                self._load_time_ms = (time.time() - start) * 1000
                self._loaded = True
                logger.info(f"Hugging Face Transformers '{model_id}' loaded in {self._load_time_ms:.1f}ms on {device}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load HF transformers pipeline: {e}. Falling back to ctranslate2 backend.")
                self._backend = "ctranslate2"

        if self._backend == "ctranslate2":
            try:
                from faster_whisper import WhisperModel

                # Resolve device and quantization
                device = "cpu"
                compute_type = "int8"
                if self._device == "cuda" or (self._device == "auto" and torch.cuda.is_available()):
                    device = "cuda"
                    compute_type = "float16"

                hf_hub_variant = self._model_name
                if hf_hub_variant.startswith("openai/whisper-"):
                    hf_hub_variant = hf_hub_variant.replace("openai/whisper-", "")
                elif hf_hub_variant.startswith("Systran/faster-whisper-"):
                    hf_hub_variant = hf_hub_variant.replace("Systran/faster-whisper-", "")

                logger.info(f"Loading local Hugging Face CTranslate2 Whisper '{hf_hub_variant}' on {device} ({compute_type})...")
                self._model = WhisperModel(
                    hf_hub_variant,
                    device=device,
                    compute_type=compute_type,
                )

                self._load_time_ms = (time.time() - start) * 1000
                self._loaded = True
                logger.info(
                    f"Hugging Face Whisper ({hf_hub_variant}) loaded in "
                    f"{self._load_time_ms:.1f}ms on {device} ({compute_type})"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load Hugging Face CTranslate2 Whisper model: {e}", exc_info=True)
                self._loaded = False
                return False

        return False

    def transcribe_segment(
        self,
        audio: torch.Tensor,
        sample_offset: int = 0,
    ) -> ASRResult:
        """
        Transcribe a validated speech segment.
        
        Args:
            audio: 1D float32 tensor sampled at 16 kHz.
            sample_offset: Absolute timeline sample index.
            
        Returns:
            ASRResult with text, latency, and timing boundaries.
        """
        if not self._loaded:
            raise RuntimeError("Hugging Face ASR provider model is not loaded.")

        start_time = time.time()
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy().copy()
        else:
            audio_np = np.array(audio, dtype=np.float32, copy=True)

        # Ensure 1D float32 contiguous array
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        audio_np = np.ascontiguousarray(audio_np, dtype=np.float32)

        text = ""
        detected_language = "en"

        with self._lock:
            if self._backend == "transformers" and self._pipeline is not None:
                outputs = self._pipeline(audio_np, generate_kwargs={"language": "english", "task": "transcribe"})
                text = outputs.get("text", "").strip()
            elif self._model is not None:
                # Optimized CTranslate2 Hugging Face Whisper decoding
                # CRITICAL: temperature=0.0 prevents temperature fallback to random sampling (0.2, 0.4, 0.8, 1.0)
                # which causes hallucinatory repetition loops on low-confidence or chopped audio.
                segments, info = self._model.transcribe(
                    audio_np,
                    beam_size=self._beam_size,
                    language="en",
                    vad_filter=False,  # External Silero VAD already segmented the speech
                    word_timestamps=False,
                    condition_on_previous_text=False,  # Suppress loop hallucinations across segments
                    temperature=0.0,
                    no_speech_threshold=self._no_speech_threshold,
                    log_prob_threshold=self._log_prob_threshold,
                    compression_ratio_threshold=2.4,
                )
                valid_parts = []
                avg_logprob_sum = 0.0
                max_no_speech_prob = 0.0
                segment_count = 0

                UNCONDITIONAL_HALLUCINATIONS = {
                    "i'll see you in the next video",
                    "see you in the next video",
                    "thank you for watching",
                    "thanks for watching",
                    "don't forget to subscribe",
                    "please subscribe",
                    "subscribe to my channel",
                    "like and subscribe",
                    "subtitles by",
                    "english subtitles",
                    "closed captioning provided by",
                    "transcription by",
                    "translated by",
                    "watching",
                }

                SUSPECT_PHANTOMS = {
                    "i'm sorry",
                    "im sorry",
                    "sorry",
                    "amen",
                    "you",
                    "thank you",
                    "thanks",
                    "bye",
                    "goodbye",
                }

                for s in segments:
                    segment_count += 1
                    s_text = s.text.strip()
                    s_no_speech = getattr(s, "no_speech_prob", 0.0)
                    s_avg_logprob = getattr(s, "avg_logprob", 0.0)
                    avg_logprob_sum += s_avg_logprob
                    max_no_speech_prob = max(max_no_speech_prob, s_no_speech)

                    # Hallucination filter: high no_speech_prob or poor acoustic likelihood
                    if s_no_speech > 0.40:
                        logger.debug(f"[ASR Defense] Rejected segment '{s_text}': high no_speech_prob={s_no_speech:.3f}")
                        continue
                    if s_avg_logprob < -1.0:
                        logger.debug(f"[ASR Defense] Rejected segment '{s_text}': low avg_logprob={s_avg_logprob:.3f}")
                        continue

                    # Filter known training priors (e.g. YouTube outro boilerplate) - unconditionally
                    clean_lower = s_text.lower().strip(" .!?,;:")
                    if clean_lower in UNCONDITIONAL_HALLUCINATIONS or clean_lower.endswith("in the next video") or clean_lower.endswith("for watching"):
                        logger.info(f"[ASR Defense] Filtered training prior hallucination: '{s_text}'")
                        continue

                    # Filter phantom short hallucinations on low-confidence or short audio
                    if clean_lower in SUSPECT_PHANTOMS:
                        if s_no_speech > 0.15 or s_avg_logprob < -0.45 or len(audio_np) < 16000 * 0.75:
                            logger.info(f"[ASR Defense] Filtered phantom short hallucination '{s_text}' (no_speech={s_no_speech:.3f}, logprob={s_avg_logprob:.3f})")
                            continue

                    if s_text:
                        valid_parts.append(s_text)

                text = " ".join(valid_parts).strip()
                avg_logprob = (avg_logprob_sum / segment_count) if segment_count > 0 else 0.0
                no_speech_prob = max_no_speech_prob
                if info and info.language:
                    detected_language = info.language

        latency_ms = (time.time() - start_time) * 1000.0
        self._inference_times.append(latency_ms)

        return ASRResult(
            text=text,
            is_partial=False,
            language=detected_language,
            confidence=round(1.0 - no_speech_prob, 3),
            latency_ms=latency_ms,
            sample_start=sample_offset,
            sample_end=sample_offset + len(audio_np),
            model_name=f"hf-whisper-{self._model_name}-{self._backend}",
            avg_logprob=round(avg_logprob, 3),
            no_speech_prob=round(no_speech_prob, 3),
        )

    def get_model_info(self) -> ASRModelInfo:
        p50 = 0.0
        p95 = 0.0
        if self._inference_times:
            s = sorted(self._inference_times)
            p50 = s[len(s) // 2]
            p95 = s[int(len(s) * 0.95)]

        params_map = {"tiny": "39M", "base": "74M", "small": "244M", "medium": "769M"}
        variant = self._model_name.replace("openai/whisper-", "").replace("Systran/faster-whisper-", "")

        return ASRModelInfo(
            name=f"hf-whisper-{self._backend}",
            variant=variant,
            parameters=params_map.get(variant, "unknown"),
            measured_latency_p50=p50,
            measured_latency_p95=p95,
        )

    def is_loaded(self) -> bool:
        return self._loaded
