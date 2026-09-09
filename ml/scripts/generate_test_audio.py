#!/usr/bin/env python3
"""
Generate test audio samples for VoiceShield demo.

Creates:
  1. bonafide_sample.wav — a speech-like tone sequence (human-like cadence)
  2. synthetic_sample.wav — a constant tone (synthetic-sounding monotone)

These are NOT real speech — they're test signals for demonstrating the pipeline
works end-to-end. Use real bonafide/spoof audio from ASVspoof for calibration.

Usage:
    python3 ml/scripts/generate_test_audio.py
"""

import numpy as np
import os
import sys

SAMPLE_RATE = 16000
DURATION = 5.0  # seconds


def generate_speech_like(duration: float, sr: int) -> np.ndarray:
    """Generate a speech-like signal with varying pitch and amplitude.
    
    Simulates human voice cadence with:
    - Fundamental frequency around 120-250 Hz
    - Amplitude modulation (pauses, emphasis)
    - Harmonic overtones
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    # Varying pitch (120-250 Hz, typical human range)
    pitch = 150 + 50 * np.sin(2 * np.pi * 0.5 * t) + 30 * np.sin(2 * np.pi * 1.3 * t)
    
    # Phase from instantaneous frequency
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    
    # Fundamental + harmonics
    signal = (
        0.6 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.1 * np.sin(3 * phase) +
        0.05 * np.sin(4 * phase)
    )
    
    # Amplitude envelope (speech-like bursts)
    envelope = (
        0.5 + 0.3 * np.sin(2 * np.pi * 3.0 * t) +
        0.2 * np.sin(2 * np.pi * 0.7 * t)
    )
    envelope = np.clip(envelope, 0.0, 1.0)
    
    # Simulate pauses
    for start in np.arange(0, duration, 1.5):
        pause_start = int((start + 0.8) * sr)
        pause_end = int((start + 1.0) * sr)
        if pause_end < len(envelope):
            envelope[pause_start:pause_end] *= 0.1
    
    signal *= envelope
    
    # Add slight noise (natural recording noise)
    signal += np.random.randn(len(signal)).astype(np.float32) * 0.005
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    
    return signal.astype(np.float32)


def generate_synthetic_tone(duration: float, sr: int) -> np.ndarray:
    """Generate a synthetic/artificial-sounding constant tone.
    
    Perfect pitch, no natural variation — the kind of signal that
    should score higher on spoof detection.
    """
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    # Perfect constant pitch (no variation = sounds artificial)
    signal = (
        0.7 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 600 * t)
    )
    
    # Constant amplitude (no natural modulation)
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    
    return signal.astype(np.float32)


def write_wav(path: str, signal: np.ndarray, sr: int):
    """Write a WAV file using only stdlib (no soundfile dependency needed)."""
    import struct
    import wave
    
    # Convert float32 [-1, 1] to int16
    pcm = (signal * 32767).astype(np.int16)
    
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    sample_dir = os.path.join(project_root, "ml", "sample_audio")
    os.makedirs(sample_dir, exist_ok=True)
    
    print("Generating test audio samples...")
    
    # 1. Bonafide-like sample
    bonafide_path = os.path.join(sample_dir, "bonafide_sample.wav")
    bonafide = generate_speech_like(DURATION, SAMPLE_RATE)
    write_wav(bonafide_path, bonafide, SAMPLE_RATE)
    print(f"  ✓ {bonafide_path} ({DURATION}s, {SAMPLE_RATE}Hz, {len(bonafide)} samples)")
    
    # 2. Synthetic-like sample  
    synthetic_path = os.path.join(sample_dir, "synthetic_sample.wav")
    synthetic = generate_synthetic_tone(DURATION, SAMPLE_RATE)
    write_wav(synthetic_path, synthetic, SAMPLE_RATE)
    print(f"  ✓ {synthetic_path} ({DURATION}s, {SAMPLE_RATE}Hz, {len(synthetic)} samples)")
    
    print("\nDone! Use these files with:")
    print(f"  curl -X POST http://localhost:8000/api/analyze/upload -F 'file=@{bonafide_path}'")
    print(f"  curl -X POST http://localhost:8000/api/analyze/upload -F 'file=@{synthetic_path}'")


if __name__ == "__main__":
    main()
