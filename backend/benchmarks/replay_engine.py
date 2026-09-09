"""
Replay engine — feeds audio files through the SAME pipeline as live mode.

Two modes:
  --mode realtime : Streams at real-time pace (measures latency, queue behavior)
  --mode fast     : Streams as fast as inference allows (measures accuracy, throughput)

The replay engine injects audio into the AudioBus exactly like the
WebRTC media receiver does. All downstream pipelines are identical.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from app.services.monitor.audio_bus import AudioBus

logger = logging.getLogger(__name__)

# Chunk size for streaming (matches typical WebRTC frame: 20ms at 16kHz = 320 samples)
REALTIME_CHUNK_SAMPLES = 320
# Fast mode uses larger chunks for throughput
FAST_CHUNK_SAMPLES = 16000  # 1 second


class ReplayEngine:
    """Injects WAV/FLAC files into the AudioBus.

    Uses the SAME pipeline as live WebRTC mode —
    no separate batch inference path.

    Modes:
        realtime: Sleep between chunks to simulate real-time audio.
                  Measures latency, queue depth, streaming behavior.
        fast:     Stream as quickly as inference allows.
                  Measures accuracy, throughput, benchmark reproducibility.
    """

    def __init__(self, audio_bus: AudioBus):
        self._audio_bus = audio_bus
        self._total_replayed = 0
        self._total_files = 0

    async def replay_file(
        self,
        wav_path: str | Path,
        mode: str = "realtime",
    ) -> dict:
        """Stream a single audio file through the AudioBus.

        Args:
            wav_path: Path to WAV/FLAC audio file.
            mode: "realtime" or "fast".

        Returns:
            Telemetry dict with replay statistics.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # Load audio
        waveform, sr = torchaudio.load(str(wav_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # 1D

        # Resample to canonical 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        total_samples = len(waveform)
        chunk_size = REALTIME_CHUNK_SAMPLES if mode == "realtime" else FAST_CHUNK_SAMPLES

        start_time = time.time()
        samples_sent = 0

        # Stream through audio bus in chunks
        for offset in range(0, total_samples, chunk_size):
            chunk = waveform[offset : offset + chunk_size]
            if len(chunk) == 0:
                break

            await self._audio_bus.publish(
                samples=chunk,
                wall_timestamp=time.time(),
            )
            samples_sent += len(chunk)

            # In realtime mode, sleep to simulate real-time delivery
            if mode == "realtime":
                chunk_duration = len(chunk) / 16000.0
                await asyncio.sleep(chunk_duration)

        elapsed = time.time() - start_time
        audio_duration = total_samples / 16000.0

        self._total_replayed += total_samples
        self._total_files += 1

        return {
            "file": str(wav_path),
            "mode": mode,
            "audio_duration_s": round(audio_duration, 3),
            "replay_duration_s": round(elapsed, 3),
            "real_time_factor": round(elapsed / audio_duration, 3) if audio_duration > 0 else 0,
            "samples_sent": samples_sent,
            "source_sr": sr,
            "chunks": (total_samples + chunk_size - 1) // chunk_size,
        }

    async def replay_directory(
        self,
        dir_path: str | Path,
        mode: str = "realtime",
        extensions: tuple = (".wav", ".flac", ".mp3"),
        callback=None,
    ) -> list[dict]:
        """Replay all audio files in a directory.

        Args:
            dir_path: Directory containing audio files.
            mode: "realtime" or "fast".
            extensions: File extensions to include.
            callback: Optional async callback(file_path, result) after each file.

        Returns:
            List of telemetry dicts.
        """
        dir_path = Path(dir_path)
        files = sorted(
            f for f in dir_path.rglob("*")
            if f.suffix.lower() in extensions
        )

        if not files:
            logger.warning(f"No audio files found in {dir_path}")
            return []

        logger.info(f"Replaying {len(files)} files from {dir_path} (mode={mode})")
        results = []

        for i, f in enumerate(files):
            try:
                result = await self.replay_file(f, mode=mode)
                results.append(result)
                if callback:
                    await callback(f, result)
                logger.info(
                    f"[{i+1}/{len(files)}] {f.name}: "
                    f"{result['audio_duration_s']}s audio in "
                    f"{result['replay_duration_s']}s "
                    f"(RTF={result['real_time_factor']})"
                )
            except Exception as e:
                logger.error(f"Replay failed for {f}: {e}")
                results.append({"file": str(f), "error": str(e)})

        return results
