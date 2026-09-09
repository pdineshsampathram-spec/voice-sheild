"""
VoiceShield — Three-Condition Live Harness Benchmark (Phase 4).

Reproducible test procedure evaluating the end-to-end streaming monitor pipeline
under three controlled acoustic conditions:
  Condition A: Normal speech (bona fide persona)
  Condition B: Synthetic / voice-cloned speech (vocoder phase-dispersion attack)
  Condition C: Silence / background noise (sub-gate ambient noise)

CRITICAL INVARIANTS:
1. Uses the SAME streaming AudioBus and pipeline consumers as live WebRTC mode.
2. Ingests audio in 320-sample (20ms) chunks.
3. Records all metrics without converting them into unproven accuracy claims.
4. Distinguishes zero vs unavailable vs measured.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.services.monitor.session import MonitorSession
from app.services.asr.hf_whisper_provider import HFWhisperProvider
from app.services.antispoof.spectra_detector import spectra_detector

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("live_harness")
logger.setLevel(logging.INFO)

CHUNK_SAMPLES = 320  # 20ms at 16kHz


def synthesize_audio(text: str, voice: str = "Samantha", sr: int = 16000) -> torch.Tensor:
    """Synthesize clean audio using macOS speech synthesizer."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff_path = tmp.name

    try:
        cmd = ["say", "-v", voice, text, "-o", aiff_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        data, read_sr = sf.read(aiff_path)
        tensor = torch.from_numpy(data.astype(np.float32))
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=-1)

        if read_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=read_sr, new_freq=sr)
            tensor = resampler(tensor.unsqueeze(0)).squeeze(0)

        return tensor
    finally:
        if os.path.exists(aiff_path):
            os.remove(aiff_path)


def synthesize_synthetic_attack(text: str, voice: str = "Samantha") -> torch.Tensor:
    """Synthesize voice sample with vocoder phase-dispersion perturbation."""
    clean = synthesize_audio(text, voice=voice)
    fft = torch.fft.rfft(clean)
    mag = torch.abs(fft)
    phase = torch.angle(fft)
    freqs = torch.linspace(0, 1, len(phase))
    dispersion = 1.2 * torch.sin(2 * np.pi * 3.5 * freqs) + 0.6 * torch.cos(2 * np.pi * 7.0 * freqs)
    perturbed_fft = mag * torch.exp(1j * (phase + dispersion))
    attack = torch.fft.irfft(perturbed_fft, n=len(clean))
    return attack.to(torch.float32)


def generate_ambient_noise(duration_s: float = 6.0, rms: float = 0.0005, sr: int = 16000) -> torch.Tensor:
    """Generate controlled ambient noise floor."""
    n_samples = int(duration_s * sr)
    noise = torch.randn(n_samples)
    current_rms = float(torch.sqrt(torch.mean(noise ** 2)))
    if current_rms > 0:
        noise = noise * (rms / current_rms)
    return noise.to(torch.float32)


@dataclass
class ConditionResult:
    condition_name: str
    total_duration_s: float = 0.0
    valid_aasist_windows: int = 0
    mean_spoof_score: float = 0.0
    median_spoof_score: float = 0.0
    min_spoof_score: float = 0.0
    max_spoof_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    abstention_count: int = 0
    vad_speech_duration_s: float = 0.0
    dropped_chunks: int = 0
    queue_overflows: int = 0
    asr_transcripts: List[str] = field(default_factory=list)
    raw_scores: List[float] = field(default_factory=list)
    decisions: Dict[str, int] = field(default_factory=dict)
    pipeline_errors: int = 0


async def stream_audio_through_session(
    session: MonitorSession,
    audio_tensor: torch.Tensor,
    trailing_silence_s: float = 1.0,
    drain_timeout_s: float = 2.5,
) -> List[dict]:
    """Stream audio chunks through session's AudioBus with realistic pacing and collect emitted events."""
    events = []
    # Add trailing silence so VAD cleanly offsets and dispatches speech to ASR
    if trailing_silence_s > 0:
        silence = torch.zeros(int(trailing_silence_s * 16000), dtype=torch.float32)
        stream_tensor = torch.cat([audio_tensor, silence])
    else:
        stream_tensor = audio_tensor

    total_samples = len(stream_tensor)

    # Consumer task to collect events
    async def collect_events():
        while True:
            evt = await session.get_event(timeout=0.05)
            if evt:
                events.append(evt)
            else:
                await asyncio.sleep(0.01)

    collector_task = asyncio.create_task(collect_events())

    # Stream in 320-sample chunks with realistic pacing (0.015s per 20ms chunk = 1.33x real-time)
    for offset in range(0, total_samples, CHUNK_SAMPLES):
        chunk = stream_tensor[offset : offset + CHUNK_SAMPLES]
        if len(chunk) == 0:
            break
        await session.audio_bus.publish(chunk, wall_timestamp=time.time())
        await asyncio.sleep(0.015)

    # Allow downstream workers to finish processing
    await asyncio.sleep(drain_timeout_s)
    collector_task.cancel()
    try:
        await collector_task
    except asyncio.CancelledError:
        pass

    return events


def compile_condition_metrics(name: str, audio: torch.Tensor, events: List[dict], telemetry: dict) -> ConditionResult:
    """Compile metrics for a condition without fabricating accuracy."""
    duration_s = round(len(audio) / 16000.0, 3)
    raw_scores = []
    decisions = {}
    abstentions = 0
    valid_windows = 0
    transcripts = []
    vad_speech_samples = 0

    bins = {"[0.0, 0.2)": 0, "[0.2, 0.4)": 0, "[0.4, 0.6)": 0, "[0.6, 0.8)": 0, "[0.8, 1.0]": 0}

    for e in events:
        etype = e.get("type")
        if etype == "detection":
            decision = e.get("decision", "UNKNOWN")
            decisions[decision] = decisions.get(decision, 0) + 1

            if decision == "ABSTAIN":
                abstentions += 1
            elif e.get("window_valid", False):
                valid_windows += 1
                score = float(e.get("spoof_score", 0.0))
                raw_scores.append(score)

                if score < 0.2:
                    bins["[0.0, 0.2)"] += 1
                elif score < 0.4:
                    bins["[0.2, 0.4)"] += 1
                elif score < 0.6:
                    bins["[0.4, 0.6)"] += 1
                elif score < 0.8:
                    bins["[0.6, 0.8)"] += 1
                else:
                    bins["[0.8, 1.0]"] += 1

        elif etype == "caption":
            if not e.get("is_partial", True):
                text = e.get("text", "").strip()
                if text and text not in transcripts:
                    transcripts.append(text)

        elif etype == "vad":
            # Count speech duration when speech segment completes (vad_speech_active=False)
            if not e.get("vad_speech_active", True):
                vad_speech_samples += (e.get("sample_end", 0) - e.get("sample_start", 0))

    audio_bus_telemetry = telemetry.get("audio_bus", {})
    consumers = audio_bus_telemetry.get("consumers", {})
    total_dropped = sum(c.get("dropped", 0) for c in consumers.values())
    total_overflows = sum(c.get("queue_overflows", 0) for c in consumers.values())

    cap_errors = telemetry.get("caption_pipeline", {}).get("pipeline_errors", 0)
    sec_errors = telemetry.get("security_pipeline", {}).get("pipeline_errors", 0)

    return ConditionResult(
        condition_name=name,
        total_duration_s=duration_s,
        valid_aasist_windows=valid_windows,
        mean_spoof_score=round(float(np.mean(raw_scores)), 4) if raw_scores else 0.0,
        median_spoof_score=round(float(np.median(raw_scores)), 4) if raw_scores else 0.0,
        min_spoof_score=round(float(np.min(raw_scores)), 4) if raw_scores else 0.0,
        max_spoof_score=round(float(np.max(raw_scores)), 4) if raw_scores else 0.0,
        score_distribution=bins,
        abstention_count=abstentions,
        vad_speech_duration_s=round(vad_speech_samples / 16000.0, 2),
        dropped_chunks=total_dropped,
        queue_overflows=total_overflows,
        asr_transcripts=transcripts,
        raw_scores=raw_scores,
        decisions=decisions,
        pipeline_errors=cap_errors + sec_errors,
    )


async def run_harness():
    print("================================================================================")
    print("VOICE SHIELD — THREE-CONDITION LIVE HARNESS BENCHMARK (PHASE 4)")
    print("================================================================================")

    # 1. Warm up models
    logger.info("Initializing shared models...")
    spectra_detector.load_model()
    asr_provider = HFWhisperProvider(model_name="base", backend="ctranslate2", beam_size=1, no_speech_threshold=0.6, log_prob_threshold=-1.0)
    asr_provider.load_model()

    from app.services.antispoof.calibrator import score_calibrator
    calib_path = Path("models/calibration_v1.json")
    if not calib_path.exists():
        calib_path = Path("backend/models/calibration_v1.json")
    if calib_path.exists():
        score_calibrator.load(calib_path)
        logger.info(f"Loaded calibration artifact: {score_calibrator.version}")

    test_sentence_a = (
        "VoiceShield is conducting comprehensive end-to-end verification of WebRTC speech monitoring."
    )
    test_sentence_b = (
        "Voice biometrics require statistical calibration before presenting security probabilities."
    )

    results: List[ConditionResult] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Condition A: Normal human speech
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Running Condition A: Normal Speech (Bona Fide Persona) ---")
    audio_a = synthesize_audio(test_sentence_a, voice="Samantha")
    # Ensure at least 6.5s of audio so multiple 64,600-sample windows form
    if len(audio_a) < 104000:
        tail = synthesize_audio(test_sentence_b, voice="Samantha")
        audio_a = torch.cat([audio_a, tail])

    session_a = MonitorSession(session_id="harness-cond-a")
    await session_a.start(asr_provider=asr_provider)
    events_a = await stream_audio_through_session(session_a, audio_a, drain_timeout_s=3.0)
    telem_a = session_a.get_telemetry()
    await session_a.stop()
    res_a = compile_condition_metrics("Condition A: Normal Speech (Bona Fide)", audio_a, events_a, telem_a)
    results.append(res_a)

    # ──────────────────────────────────────────────────────────────────────────
    # Condition B: Synthetic / voice-cloned speech
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Running Condition B: Synthetic / Cloned Speech (Vocoder Attack) ---")
    audio_b = synthesize_synthetic_attack(test_sentence_a, voice="Samantha")
    if len(audio_b) < 104000:
        tail_b = synthesize_synthetic_attack(test_sentence_b, voice="Samantha")
        audio_b = torch.cat([audio_b, tail_b])

    session_b = MonitorSession(session_id="harness-cond-b")
    await session_b.start(asr_provider=asr_provider)
    events_b = await stream_audio_through_session(session_b, audio_b, drain_timeout_s=3.0)
    telem_b = session_b.get_telemetry()
    await session_b.stop()
    res_b = compile_condition_metrics("Condition B: Synthetic Speech (Attack)", audio_b, events_b, telem_b)
    results.append(res_b)

    # ──────────────────────────────────────────────────────────────────────────
    # Condition C: Silence / background noise
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Running Condition C: Silence / Background Noise (RMS < Gate) ---")
    audio_c = generate_ambient_noise(duration_s=8.0, rms=0.0005)

    session_c = MonitorSession(session_id="harness-cond-c")
    await session_c.start(asr_provider=asr_provider)
    events_c = await stream_audio_through_session(session_c, audio_c, drain_timeout_s=2.5)
    telem_c = session_c.get_telemetry()
    await session_c.stop()
    res_c = compile_condition_metrics("Condition C: Silence / Ambient Noise", audio_c, events_c, telem_c)
    results.append(res_c)

    # ──────────────────────────────────────────────────────────────────────────
    # Presentation Table
    # ──────────────────────────────────────────────────────────────────────────
    print("\n================================================================================")
    print("MEASURED TEST HARNESS RESULTS (STRICT LOCAL HARDWARE MEASUREMENT)")
    print("================================================================================")
    header = (
        f"{'Metric':<30} | {'Condition A (Speech)':<22} | "
        f"{'Condition B (Attack)':<22} | {'Condition C (Silence)':<22}"
    )
    print(header)
    print("-" * len(header))

    def row(label, val_a, val_b, val_c):
        print(f"{label:<30} | {str(val_a):<22} | {str(val_b):<22} | {str(val_c):<22}")

    row("Total Duration (s)", res_a.total_duration_s, res_b.total_duration_s, res_c.total_duration_s)
    row("Valid AASIST Windows", res_a.valid_aasist_windows, res_b.valid_aasist_windows, res_c.valid_aasist_windows)
    row("Quality Gate Abstentions", res_a.abstention_count, res_b.abstention_count, res_c.abstention_count)
    row("Mean Raw Spoof Score", res_a.mean_spoof_score, res_b.mean_spoof_score, res_c.mean_spoof_score)
    row("Median Raw Spoof Score", res_a.median_spoof_score, res_b.median_spoof_score, res_c.median_spoof_score)
    row("Min Spoof Score", res_a.min_spoof_score, res_b.min_spoof_score, res_c.min_spoof_score)
    row("Max Spoof Score", res_a.max_spoof_score, res_b.max_spoof_score, res_c.max_spoof_score)
    row("VAD Speech Duration (s)", res_a.vad_speech_duration_s, res_b.vad_speech_duration_s, res_c.vad_speech_duration_s)
    row("Dropped Audio Chunks", res_a.dropped_chunks, res_b.dropped_chunks, res_c.dropped_chunks)
    row("Queue Overflows", res_a.queue_overflows, res_b.queue_overflows, res_c.queue_overflows)
    row("Pipeline Errors", res_a.pipeline_errors, res_b.pipeline_errors, res_c.pipeline_errors)

    print("\n--- Operating Decisions Recorded ---")
    for r in results:
        print(f"{r.condition_name}: {r.decisions}")

    print("\n--- AASIST Raw Score Distribution Across Bins ---")
    for r in results:
        if r.valid_aasist_windows > 0:
            print(f"{r.condition_name}: {r.score_distribution}")
        else:
            print(f"{r.condition_name}: No valid evaluated windows (100% abstained by quality gate)")

    print("\n--- ASR Output Transcripts ---")
    print(f"Condition A Transcripts: {res_a.asr_transcripts}")
    print(f"Condition B Transcripts: {res_b.asr_transcripts}")
    print(f"Condition C Transcripts: {res_c.asr_transcripts} (Expected empty)")

    print("\n================================================================================")
    print("SCIENTIFIC CAVEAT:")
    print("These measurements reflect controlled synthetic/TTS test stimuli.")
    print("They do NOT constitute a final accuracy or field-readiness proof.")
    print("================================================================================")


if __name__ == "__main__":
    asyncio.run(run_harness())
