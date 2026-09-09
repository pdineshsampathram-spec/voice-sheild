"""
VoiceShield — Telephony Audio Decoder & Validation Service (Milestone 2).

Decodes Twilio 8kHz mu-law audio to 16kHz Float32 PCM tensors, calculates
telemetry metrics (RMS, Peak, Zeros, NaNs, Infs), and prepares audio for
the VoiceShield ML pipeline.
"""

import base64
import logging
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class TelephonyAudioDecoder:
    """Decodes Twilio telephony media streams and computes telemetry."""

    def __init__(self):
        # 8kHz to 16kHz resampler
        self.resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

    def decode_twilio_payload(self, base64_payload: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decode a base64 Twilio mu-law chunk to 16kHz Float32 PCM tensor and compute telemetry.

        Args:
            base64_payload: Base64-encoded 8kHz mu-law audio chunk from Twilio.

        Returns:
            Tuple of:
            - 1D torch.Tensor: 16kHz Float32 PCM audio samples
            - Dict of telemetry metrics (bytes_count, samples_count, rms, peak, zero_percent, nan_count, inf_count)
        """
        raw_bytes = base64.b64decode(base64_payload)
        bytes_count = len(raw_bytes)

        if bytes_count == 0:
            empty_tensor = torch.zeros(0, dtype=torch.float32)
            return empty_tensor, {
                "bytes_count": 0,
                "samples_count": 0,
                "rms": 0.0,
                "peak": 0.0,
                "percentage_zero": 100.0,
                "nan_count": 0,
                "inf_count": 0,
            }

        # Convert raw bytes to uint8 tensor (copy to ensure writability)
        mulaw_np = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
        mulaw_tensor = torch.from_numpy(mulaw_np)

        # Decode mu-law (8-bit) to linear Float32 PCM [-1.0, 1.0]
        # In Python 3.13, torchaudio.functional.mu_law_decoding replaces removed audioop
        pcm_8k = torchaudio.functional.mu_law_decoding(mulaw_tensor, quantization_channels=256)

        # Resample to 16kHz
        # Resample expects [batch, time] or [channels, time]
        pcm_8k_2d = pcm_8k.unsqueeze(0)
        pcm_16k_2d = self.resampler(pcm_8k_2d)
        pcm_16k = pcm_16k_2d.squeeze(0)

        samples_count = len(pcm_16k)

        # Compute Telemetry
        nan_count = int(torch.isnan(pcm_16k).sum().item())
        inf_count = int(torch.isinf(pcm_16k).sum().item())

        if samples_count > 0:
            rms = float(torch.sqrt(torch.mean(pcm_16k ** 2)).item())
            peak = float(torch.max(torch.abs(pcm_16k)).item())
            zero_count = int(torch.sum(pcm_16k == 0).item())
            zero_percent = float((zero_count / samples_count) * 100.0)
        else:
            rms = 0.0
            peak = 0.0
            zero_percent = 0.0

        telemetry = {
            "bytes_count": bytes_count,
            "samples_count": samples_count,
            "rms": rms,
            "peak": peak,
            "percentage_zero": zero_percent,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

        return pcm_16k, telemetry


# Global singleton
audio_decoder = TelephonyAudioDecoder()
