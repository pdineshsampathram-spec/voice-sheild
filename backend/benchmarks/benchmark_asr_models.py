"""
Hugging Face ASR Model Benchmark for VoiceShield Live Captions.

Evaluates locally runnable Hugging Face speech models
on the canonical 16 kHz Float32 audio dataset and preprocessing path.

Measures:
- Normalized WER & CER
- Exact Match Rate (%)
- Hallucination counts on digital silence and ambient noise
- Latency (Median P50, P95, Max ms)
- Real-Time Factor (RTF)
- Memory usage & Model Load time
"""

import os
import sys
import time
import re
import json
import logging
import tempfile
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.asr.hf_whisper_provider import HFWhisperProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("benchmark.asr_models")

def normalize_text(text: str) -> str:
    """Normalize text for consistent phonetic comparison."""
    if not text:
        return ""
    norm = text.lower()
    norm = re.sub(r"\bvoiceshield\b", "voice shield", norm)
    norm = re.sub(r"[.,!?;:\"'()\[\]{}~`*_\-/\\<>#@$%^&+=]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm

def compute_levenshtein(ref: List[str], hyp: List[str]) -> int:
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[len(ref), len(hyp)]

def compute_wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return compute_levenshtein(ref_words, hyp_words) / len(ref_words)

def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return compute_levenshtein(ref_chars, hyp_chars) / len(ref_chars)


# Multi-Speaker Canonical Benchmark Suite (6 Speakers / Accents)
BENCHMARK_UTTERANCES = [
    # US accent
    {"id": "spk1_us_1", "text": "VoiceShield is verifying the integrity of this encrypted conversation.", "voice": "Samantha"},
    {"id": "spk1_us_2", "text": "Please provide your confirmation code to proceed with authentication.", "voice": "Samantha"},
    # GB accent
    {"id": "spk2_gb_1", "text": "Security analysts require reliable evidence before flagging suspicious transactions.", "voice": "Daniel"},
    {"id": "spk2_gb_2", "text": "The telecommunications gateway reported normal latency throughout the call.", "voice": "Daniel"},
    # AU accent
    {"id": "spk3_au_1", "text": "Our laboratory conducted comprehensive benchmarks under simulated field conditions.", "voice": "Karen"},
    {"id": "spk3_au_2", "text": "Biometric verification systems must maintain low false positive rates.", "voice": "Karen"},
    # IN accent
    {"id": "spk4_in_1", "text": "The national digital identity platform processes millions of authentications daily.", "voice": "Rishi"},
    {"id": "spk4_in_2", "text": "Real time speech recognition must adapt to regional pronunciation variations.", "voice": "Rishi"},
    # IE accent
    {"id": "spk5_ie_1", "text": "Signal processing filters eliminate ambient acoustic reflections effectively.", "voice": "Moira"},
    # ZA accent
    {"id": "spk6_za_1", "text": "Continuous authentication ensures security throughout the entire phone conversation.", "voice": "Tessa"},
    # Short utterance
    {"id": "short_1", "text": "Yes I agree.", "voice": "Samantha"},
    # Long utterance
    {"id": "long_1", "text": "According to the latest security advisory all callers must verify their biometric credentials through the multi-factor authentication portal.", "voice": "Daniel"},
]

def generate_speech_audio(text: str, voice: str) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(["say", "-v", voice, text, "-o", tmp_path], check=True)
        data, sr = sf.read(tmp_path)
        tensor = torch.from_numpy(data.astype(np.float32))
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=-1)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            tensor = resampler(tensor.unsqueeze(0)).squeeze(0)
        return tensor
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_benchmark():
    dataset = []
    logger.info("Synthesizing multi-speaker canonical evaluation corpus...")
    for item in BENCHMARK_UTTERANCES:
        try:
            audio = generate_speech_audio(item["text"], item["voice"])
            dataset.append({
                "id": item["id"],
                "text": item["text"],
                "voice": item["voice"],
                "audio": audio,
                "duration_s": len(audio) / 16000.0,
            })
        except Exception as e:
            logger.warning(f"Could not synthesize {item['id']}: {e}")

    logger.info(f"Loaded {len(dataset)} canonical evaluation speech utterances.")

    # Generate Silence & Noise test signals
    silence_2s = torch.zeros(32000, dtype=torch.float32)
    np.random.seed(42)
    ambient_noise_2s = torch.from_numpy(np.random.normal(0, 0.0005, 32000).astype(np.float32))
    mic_noise_2s = torch.from_numpy(np.random.normal(0, 0.002, 32000).astype(np.float32))

    test_models = [
        {"name": "tiny", "beam_size": 1, "log_prob": -1.0, "desc": "HF Whisper Tiny (Greedy, logprob=-1.0)"},
        {"name": "tiny", "beam_size": 5, "log_prob": -1.0, "desc": "HF Whisper Tiny (Beam-5, logprob=-1.0)"},
        {"name": "base", "beam_size": 1, "log_prob": -1.0, "desc": "HF Whisper Base (Greedy, logprob=-1.0)"},
        {"name": "base", "beam_size": 5, "log_prob": -1.0, "desc": "HF Whisper Base (Beam-5, logprob=-1.0)"},
    ]

    results = []

    for cfg in test_models:
        logger.info(f"\n=======================================================")
        logger.info(f"Evaluating: {cfg['desc']}")
        logger.info(f"=======================================================")
        proc = psutil.Process()
        ram_before = proc.memory_info().rss / (1024 * 1024)

        start_load = time.time()
        provider = HFWhisperProvider(
            model_name=cfg["name"],
            backend="ctranslate2",
            beam_size=cfg["beam_size"],
            log_prob_threshold=cfg["log_prob"],
            no_speech_threshold=0.6,
        )
        loaded = provider.load_model()
        load_time_ms = (time.time() - start_load) * 1000.0

        if not loaded:
            logger.error(f"Failed to load candidate {cfg['desc']}")
            continue

        ram_after = proc.memory_info().rss / (1024 * 1024)
        ram_delta = max(0.0, ram_after - ram_before)

        # 1. Evaluate Speech Accuracy & Latency
        wers, cers, exact_matches, latencies = [], [], [], []
        total_audio_time = 0.0
        total_inference_time = 0.0

        for item in dataset:
            audio = item["audio"]
            dur = item["duration_s"]
            total_audio_time += dur

            res = provider.transcribe_segment(audio)
            latencies.append(res.latency_ms)
            total_inference_time += (res.latency_ms / 1000.0)

            ref_norm = normalize_text(item["text"])
            hyp_norm = normalize_text(res.text)

            wer = compute_wer(ref_norm, hyp_norm)
            cer = compute_cer(ref_norm, hyp_norm)
            exact = 1 if ref_norm == hyp_norm else 0

            wers.append(wer)
            cers.append(cer)
            exact_matches.append(exact)

            logger.info(f"[{item['id']}] [{item['voice']}] REF: '{ref_norm}'")
            logger.info(f"[{item['id']}] [{item['voice']}] HYP: '{hyp_norm}' (WER: {wer:.2f}, Latency: {res.latency_ms:.0f}ms)")

        # 2. Evaluate Silence & Noise Hallucinations
        silence_res = provider.transcribe_segment(silence_2s)
        ambient_res = provider.transcribe_segment(ambient_noise_2s)
        mic_res = provider.transcribe_segment(mic_noise_2s)

        hallucinations_silence = 1 if silence_res.text.strip() else 0
        hallucinations_noise = (1 if ambient_res.text.strip() else 0) + (1 if mic_res.text.strip() else 0)

        mean_wer = float(np.mean(wers)) if wers else 1.0
        mean_cer = float(np.mean(cers)) if cers else 1.0
        exact_rate = float(np.mean(exact_matches)) * 100.0 if exact_matches else 0.0
        p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        rtf = (total_inference_time / total_audio_time) if total_audio_time > 0 else 0.0

        res_entry = {
            "model": cfg["desc"],
            "model_size": cfg["name"],
            "beam_size": cfg["beam_size"],
            "log_prob_threshold": cfg["log_prob"],
            "load_time_ms": load_time_ms,
            "ram_mb": ram_delta,
            "normalized_wer_pct": mean_wer * 100.0,
            "normalized_cer_pct": mean_cer * 100.0,
            "exact_match_pct": exact_rate,
            "latency_p50_ms": p50_lat,
            "latency_p95_ms": p95_lat,
            "rtf": rtf,
            "silence_hallucinations": hallucinations_silence,
            "noise_hallucinations": hallucinations_noise,
        }
        results.append(res_entry)

    print("\n" + "="*80)
    print("VOICESHIELD HUGGING FACE ASR CANDIDATE COMPARISON MATRIX")
    print("="*80)
    for r in results:
        print(f"\nCandidate: {r['model']}")
        print(f"  WER: {r['normalized_wer_pct']:.2f}% | CER: {r['normalized_cer_pct']:.2f}% | Exact Match: {r['exact_match_pct']:.1f}%")
        print(f"  P50 Latency: {r['latency_p50_ms']:.1f}ms | P95 Latency: {r['latency_p95_ms']:.1f}ms | RTF: {r['rtf']:.3f}")
        print(f"  Load Time: {r['load_time_ms']:.0f}ms | Silence Hallucinations: {r['silence_hallucinations']} | Noise Hallucinations: {r['noise_hallucinations']}")

    out_path = Path(__file__).parent / "asr_model_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nBenchmark results saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
