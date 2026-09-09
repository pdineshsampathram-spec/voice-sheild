"""
Quick inference benchmark for VoiceShield models.

Usage:
    cd backend
    source venv/bin/activate
    python -m ml.benchmarks.benchmark_inference

Tests:
1. Spectra-AASIST inference speed (single + batch)
2. ECAPA-TDNN embedding extraction speed
3. Silero VAD speech detection speed
4. Full pipeline throughput
"""

import sys
import time
import torch
import numpy as np

from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(backend_dir))


def benchmark_spectra():
    """Benchmark Spectra-AASIST inference."""
    from app.services.antispoof.spectra_detector import spectra_detector

    print("\n--- Spectra-AASIST Benchmark ---")

    if not spectra_detector.load_model():
        print("  FAILED to load model")
        return

    # Generate test audio (4.04 seconds of white noise at 16kHz)
    test_audio = torch.randn(64600)

    # Warmup
    for _ in range(3):
        spectra_detector.predict(test_audio)

    # Single inference
    times = []
    for _ in range(10):
        start = time.time()
        result = spectra_detector.predict(test_audio)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    print(f"  Single inference: {np.mean(times):.1f}ms (±{np.std(times):.1f}ms)")
    print(f"  Result: label={result.label}, spoof={result.spoof_score:.4f}")

    # Batch inference
    batch = [torch.randn(64600) for _ in range(4)]
    batch_times = []
    for _ in range(5):
        start = time.time()
        results = spectra_detector.predict_batch(batch)
        elapsed = (time.time() - start) * 1000
        batch_times.append(elapsed)

    print(f"  Batch (4x) inference: {np.mean(batch_times):.1f}ms (±{np.std(batch_times):.1f}ms)")
    print(f"  Per-sample in batch: {np.mean(batch_times) / 4:.1f}ms")


def benchmark_ecapa():
    """Benchmark ECAPA-TDNN embedding extraction."""
    from app.services.speaker.ecapa_service import ecapa_service

    print("\n--- ECAPA-TDNN Benchmark ---")

    if not ecapa_service.load_model():
        print("  FAILED to load model")
        return

    test_audio = torch.randn(48000)  # 3 seconds

    # Warmup
    for _ in range(3):
        ecapa_service.get_embedding(test_audio)

    times = []
    for _ in range(10):
        start = time.time()
        emb = ecapa_service.get_embedding(test_audio)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    print(f"  Embedding extraction: {np.mean(times):.1f}ms (±{np.std(times):.1f}ms)")
    print(f"  Embedding shape: {emb.shape}")


def benchmark_vad():
    """Benchmark Silero VAD."""
    from app.services.vad.silero_service import vad_service

    print("\n--- Silero VAD Benchmark ---")

    if not vad_service.load_model():
        print("  FAILED to load model")
        return

    # 10 seconds of audio with some speech-like patterns
    test_audio = torch.randn(160000)

    times = []
    for _ in range(20):
        start = time.time()
        timestamps = vad_service.get_speech_timestamps(test_audio)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    print(f"  VAD (10s audio): {np.mean(times):.1f}ms (±{np.std(times):.1f}ms)")
    print(f"  Segments detected: {len(timestamps)}")


def benchmark_pipeline():
    """Benchmark full analysis pipeline."""
    from app.services.pipeline import analysis_pipeline
    from app.services.audio.preprocessor import AudioPreprocessor

    print("\n--- Full Pipeline Benchmark ---")

    preprocessor = AudioPreprocessor()

    # Generate 10 seconds of test audio
    test_audio = torch.randn(160000)

    # Save to temp file using soundfile
    import tempfile
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, test_audio.numpy(), 16000)

    times = []
    for _ in range(5):
        start = time.time()
        result = analysis_pipeline.analyze_file(file_path=tmp.name)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    import os
    os.unlink(tmp.name)

    print(f"  Full pipeline (10s): {np.mean(times):.1f}ms (±{np.std(times):.1f}ms)")
    print(f"  Segments: {len(result.segments)}")
    print(f"  Risk: {result.risk.risk_level.value} ({result.risk.risk_score:.4f})")


if __name__ == "__main__":
    print("=" * 50)
    print("VoiceShield — Inference Benchmark")
    print("=" * 50)

    from app.core.config import settings
    device = settings.get_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    benchmark_spectra()
    benchmark_ecapa()
    benchmark_vad()
    benchmark_pipeline()

    print("\n" + "=" * 50)
    print("Benchmark complete.")
