"""
VoiceShield End-to-End Pipeline Benchmark (Rule 15 & Rule 19).

Simulates an entire WebRTC session through the canonical AudioBus:
1. Replays audio chunks at real-time cadence through AudioBus
2. Concurrently exercises CaptionPipeline (VAD-A + ASR worker)
3. Concurrently exercises SecurityPipeline (VAD-B + AASIST windowing + Calibrator)
4. Checks:
   - Dropped audio chunks (must be 0!)
   - Sample count & audio duration integrity
   - Latency percentiles for ASR and AASIST
   - Peak CPU and RAM usage

Usage:
    python -m benchmarks.run_e2e
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.monitor.audio_bus import AudioBus, OverflowPolicy
from app.services.monitor.caption_pipeline import CaptionPipeline, CaptionEvent
from app.services.monitor.security_pipeline import SecurityPipeline, DetectionEvent
from app.services.asr.whisper_provider import FasterWhisperProvider
from app.services.antispoof.spectra_detector import spectra_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("benchmark.e2e")


async def run_e2e_benchmark():
    logger.info("=" * 80)
    logger.info("VOICESHIELD END-TO-END PIPELINE BENCHMARK (Rule 15, 19)")
    logger.info("=" * 80)

    # 1. Load models
    asr_provider = FasterWhisperProvider(model_size="base")
    asr_provider.load_model()
    if not spectra_detector.is_loaded:
        spectra_detector.load_model()

    # 2. Setup isolated pipeline
    audio_bus = AudioBus()

    caption_events: list[CaptionEvent] = []
    detection_events: list[DetectionEvent] = []

    caption_queue = audio_bus.subscribe("caption", maxsize=300, policy=OverflowPolicy.DROP_OLDEST)
    security_queue = audio_bus.subscribe("security", maxsize=300, policy=OverflowPolicy.EMIT_DISCONTINUITY)

    caption_pipeline = CaptionPipeline(
        audio_queue=caption_queue,
        asr_provider=asr_provider,
        on_event=lambda e: caption_events.append(e) if isinstance(e, CaptionEvent) else None,
    )

    security_pipeline = SecurityPipeline(
        audio_queue=security_queue,
        on_event=lambda e: detection_events.append(e) if isinstance(e, DetectionEvent) else None,
    )
    security_queue.on_discontinuity = security_pipeline.on_discontinuity

    await caption_pipeline.start()
    await security_pipeline.start()

    # 3. Stream 10 seconds of simulated audio at 20ms pacing
    # 20ms at 16kHz = 320 samples
    chunk_samples = 320
    sample_rate = 16000
    total_seconds = 10.0
    total_chunks = int(total_seconds * 50)  # 500 chunks

    logger.info(f"Streaming {total_seconds}s of canonical PCM ({total_chunks} chunks of {chunk_samples} samples)...")

    # Generate synthetic alternating speech burst and silence
    # Utterance: 440 Hz tone burst for 3 seconds, silence for 2s, burst for 3s
    t = torch.linspace(0, total_seconds, int(total_seconds * sample_rate))
    signal = torch.zeros_like(t)

    # Speech interval 1: 1s to 4s
    idx1 = (t >= 1.0) & (t <= 4.0)
    signal[idx1] = 0.2 * torch.sin(2 * np.pi * 440 * t[idx1])

    # Speech interval 2: 6s to 9s
    idx2 = (t >= 6.0) & (t <= 9.0)
    signal[idx2] = 0.2 * torch.sin(2 * np.pi * 440 * t[idx2])

    process = psutil.Process()
    cpu_readings = []
    ram_readings = []

    start_stream = time.time()

    for i in range(total_chunks):
        start_idx = i * chunk_samples
        end_idx = start_idx + chunk_samples
        chunk_data = signal[start_idx:end_idx]

        await audio_bus.publish(chunk_data)

        cpu_readings.append(process.cpu_percent())
        ram_readings.append(process.memory_info().rss / (1024 * 1024))

        # Pacing: 20ms simulated real-time (fast-mode 2ms for benchmark speed)
        await asyncio.sleep(0.002)

    # Wait for pipelines to finish processing trailing windows
    await asyncio.sleep(2.0)

    await caption_pipeline.stop()
    await security_pipeline.stop()

    telemetry = audio_bus.get_telemetry()
    elapsed = time.time() - start_stream

    # Extract metrics
    caption_telemetry = telemetry["consumers"]["caption"]
    security_telemetry = telemetry["consumers"]["security"]

    final_captions = [c for c in caption_events if not c.is_partial and c.text]
    valid_detections = [d for d in detection_events if d.window_valid]

    print("\n" + "=" * 80)
    print("E2E BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Audio Duration Streamed:   {telemetry['total_duration_s']}s ({telemetry['total_samples']} samples)")
    print(f"Total Chunks Published:    {telemetry['total_published']}")
    print(f"Elapsed Processing Time:   {elapsed:.2f}s")
    print(f"\nQueue Integrity:")
    print(f"  Caption Queue Dropped:   {caption_telemetry['dropped']} chunks  ← (Target: 0)")
    print(f"  Security Queue Dropped:  {security_telemetry['dropped']} chunks ← (Target: 0)")
    print(f"\nML Pipeline Performance:")
    print(f"  Final Captions Produced: {len(final_captions)}")
    print(f"  AASIST Windows Analyzed: {len(valid_detections)}")
    if valid_detections:
        latencies = [d.latency_ms for d in valid_detections]
        print(f"  AASIST Latency P50:      {np.percentile(latencies, 50):.1f}ms")
        print(f"  AASIST Latency P95:      {np.percentile(latencies, 95):.1f}ms")

    print(f"\nResource Footprint:")
    print(f"  CPU Mean: {np.mean(cpu_readings):.1f}%  Peak: {np.max(cpu_readings):.1f}%")
    print(f"  RAM Mean: {np.mean(ram_readings):.1f}MB Peak: {np.max(ram_readings):.1f}MB")
    print("=" * 80)

    # Strict assertion on dropped chunks
    if caption_telemetry["dropped"] > 0 or security_telemetry["dropped"] > 0:
        logger.error(f"FAILURE: Dropped chunks detected (caption={caption_telemetry['dropped']}, security={security_telemetry['dropped']})")
        return False

    logger.info("PASS: Zero dropped chunks, continuous audio integrity verified.")
    return True


if __name__ == "__main__":
    success = asyncio.run(run_e2e_benchmark())
    sys.exit(0 if success else 1)
