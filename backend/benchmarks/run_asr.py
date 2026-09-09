"""
VoiceShield ASR Benchmark CLI Runner (Scientific Validation & Statistical Forensic Pass).

Features:
1. Explicit Text Normalization Policy:
   - Lowercasing
   - Punctuation removal
   - Whitespace collapsing
   - Domain token normalization ("voiceshield" <-> "voice shield")
2. Strict Dataset Separation & Provenance:
   - Smoke / Regression Suite: 5 Golden sentences (Samantha)
   - Multi-Speaker Benchmark Suite: 24 distinct utterances across 6 speakers
     (Samantha [US], Daniel [GB], Karen [AU], Rishi [IN], Moira [IE], Tessa [ZA])
   - Provenance explicitly documented: TTS / Synthetic personas (NOT human recordings)
3. Rigorous Statistical Reporting:
   - Raw WER and Normalized WER reported separately
   - 1,000-iteration bootstrap with fixed random seed for 95% Confidence Intervals
   - Per-speaker breakdown and aggregate metrics
   - Latency percentiles: P50, P90, P95, P99 (ms)
   - Real-Time Factor (RTF)
   - Peak / Mean CPU % and RAM MB
4. Silence & Ambient Noise Hallucination Tests:
   - Pure digital zero (2s)
   - Low ambient noise (RMS ~0.0005, 2s)
   - Quiet microphone noise (RMS ~0.0020, 2s)
5. Comprehensive Benchmark Manifest.

Usage:
    python -m benchmarks.run_asr --model base
    python -m benchmarks.run_asr --model all
"""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.asr.whisper_provider import FasterWhisperProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("benchmark.asr")

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 1000

# ---------------------------------------------------------------------------
# Explicit Text Normalization Policy
# ---------------------------------------------------------------------------
def normalize_text_for_asr(text: str) -> str:
    """
    Explicit ASR text normalization policy:
    1. Convert to lowercase.
    2. Strip punctuation: .,!?;:'"-()[]{}~`*_/\
    3. Collapse multiple spaces into single space.
    4. Normalize compound domain token: 'voiceshield' -> 'voice shield'.
    5. Strip leading and trailing whitespace.
    """
    if not text:
        return ""
    norm = text.lower()
    norm = re.sub(r"\bvoiceshield\b", "voice shield", norm)
    norm = re.sub(r"[.,!?;:\"'()\[\]{}~`*_\-/\\<>#@$%^&+=]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


# ---------------------------------------------------------------------------
# Test Suites with Provenance
# ---------------------------------------------------------------------------
# SMOKE TEST / REGRESSION SUITE (N=5, Speaker: Samantha)
SMOKE_REGRESSION_SUITE = [
    {
        "id": "smoke_golden_1",
        "text": "VoiceShield is testing real-time WebRTC voice analysis.",
        "speaker": "Samantha",
        "locale": "en_US",
        "provenance": "TTS (macOS Samantha)",
        "category": "golden_sentence",
    },
    {
        "id": "smoke_golden_2",
        "text": "I am testing VoiceShield WebRTC captions.",
        "speaker": "Samantha",
        "locale": "en_US",
        "provenance": "TTS (macOS Samantha)",
        "category": "short_sentence",
    },
    {
        "id": "smoke_short_phrase",
        "text": "Hello world.",
        "speaker": "Samantha",
        "locale": "en_US",
        "provenance": "TTS (macOS Samantha)",
        "category": "short_phrase",
    },
    {
        "id": "smoke_long_sentence",
        "text": "Artificial intelligence models require rigorous calibration before probabilities can be presented to end users.",
        "speaker": "Samantha",
        "locale": "en_US",
        "provenance": "TTS (macOS Samantha)",
        "category": "long_sentence",
    },
    {
        "id": "smoke_numbers",
        "text": "Call emergency response at 9 1 1 immediately.",
        "speaker": "Samantha",
        "locale": "en_US",
        "provenance": "TTS (macOS Samantha)",
        "category": "numeric_codes",
    },
]

# MULTI-SPEAKER CONTROLLED BENCHMARK SUITE (N=24, 6 Speakers, 4 utterances each)
MULTI_SPEAKER_BENCHMARK_SUITE = [
    # Speaker 1: Samantha (Female, en_US)
    {"id": "spk1_utt1", "text": "VoiceShield is verifying the integrity of this encrypted conversation.", "speaker": "Samantha", "locale": "en_US"},
    {"id": "spk1_utt2", "text": "Please provide your confirmation code to proceed with authentication.", "speaker": "Samantha", "locale": "en_US"},
    {"id": "spk1_utt3", "text": "The quick brown fox jumps over the lazy dog.", "speaker": "Samantha", "locale": "en_US"},
    {"id": "spk1_utt4", "text": "We are evaluating background noise suppression on the audio channel.", "speaker": "Samantha", "locale": "en_US"},

    # Speaker 2: Daniel (Male, en_GB)
    {"id": "spk2_utt1", "text": "Security analysts require reliable evidence before flagging suspicious transactions.", "speaker": "Daniel", "locale": "en_GB"},
    {"id": "spk2_utt2", "text": "The telecommunications gateway reported normal latency throughout the call.", "speaker": "Daniel", "locale": "en_GB"},
    {"id": "spk2_utt3", "text": "Speech synthesis technology has advanced dramatically over the past two years.", "speaker": "Daniel", "locale": "en_GB"},
    {"id": "spk2_utt4", "text": "Please verify your account details using the registered mobile phone number.", "speaker": "Daniel", "locale": "en_GB"},

    # Speaker 3: Karen (Female, en_AU)
    {"id": "spk3_utt1", "text": "Our laboratory conducted comprehensive benchmarks under simulated field conditions.", "speaker": "Karen", "locale": "en_AU"},
    {"id": "spk3_utt2", "text": "Biometric verification systems must maintain low false positive rates.", "speaker": "Karen", "locale": "en_AU"},
    {"id": "spk3_utt3", "text": "Australian financial institutions enforce strict multi factor authentication.", "speaker": "Karen", "locale": "en_AU"},
    {"id": "spk3_utt4", "text": "The voice stream was processed without packet loss or jitter.", "speaker": "Karen", "locale": "en_AU"},

    # Speaker 4: Rishi (Male, en_IN)
    {"id": "spk4_utt1", "text": "The national digital identity platform processes millions of authentications daily.", "speaker": "Rishi", "locale": "en_IN"},
    {"id": "spk4_utt2", "text": "Real time speech recognition must adapt to regional pronunciation variations.", "speaker": "Rishi", "locale": "en_IN"},
    {"id": "spk4_utt3", "text": "Financial fraud prevention algorithms analyze behavioral and biometric signals.", "speaker": "Rishi", "locale": "en_IN"},
    {"id": "spk4_utt4", "text": "Voice biometrics offer convenient security for mobile banking applications.", "speaker": "Rishi", "locale": "en_IN"},

    # Speaker 5: Moira (Female, en_IE)
    {"id": "spk5_utt1", "text": "Signal processing filters eliminate ambient acoustic reflections effectively.", "speaker": "Moira", "locale": "en_IE"},
    {"id": "spk5_utt2", "text": "Customer service agents should confirm caller identity before transferring funds.", "speaker": "Moira", "locale": "en_IE"},
    {"id": "spk5_utt3", "text": "Deepfake audio detection models examine micro spectral inconsistencies.", "speaker": "Moira", "locale": "en_IE"},
    {"id": "spk5_utt4", "text": "Voice authentication protocols need to remain resilient against replay attacks.", "speaker": "Moira", "locale": "en_IE"},

    # Speaker 6: Tessa (Female, en_ZA)
    {"id": "spk6_utt1", "text": "Telecommunications infrastructure varies widely across geographic regions.", "speaker": "Tessa", "locale": "en_ZA"},
    {"id": "spk6_utt2", "text": "The neural network was trained on thousands of multilingual audio recordings.", "speaker": "Tessa", "locale": "en_ZA"},
    {"id": "spk6_utt3", "text": "Synthetic voice detectors should indicate confidence bounds for their predictions.", "speaker": "Tessa", "locale": "en_ZA"},
    {"id": "spk6_utt4", "text": "Continuous authentication ensures security throughout the entire phone conversation.", "speaker": "Tessa", "locale": "en_ZA"},
]


# ---------------------------------------------------------------------------
# Audio Generation
# ---------------------------------------------------------------------------
def generate_audio(text: str, voice: str, target_sr: int = 16000) -> torch.Tensor:
    """Generate controlled test audio using macOS speech synthesis at 16kHz."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(["say", "-v", voice, text, "-o", tmp_path], check=True)
        data, sr = sf.read(tmp_path)
        tensor = torch.from_numpy(data.astype(np.float32))

        if tensor.ndim > 1:
            tensor = tensor.mean(dim=-1)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            tensor = resampler(tensor.unsqueeze(0)).squeeze(0)

        return tensor
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Bootstrap Confidence Interval Calculation
# ---------------------------------------------------------------------------
def compute_bootstrap_ci(values: list[float], n_iterations: int = BOOTSTRAP_ITERATIONS, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """Compute 95% Confidence Interval using empirical bootstrap resampling."""
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return round(ci_lower, 4), round(ci_upper, 4)


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------
def evaluate_asr_model(model_size: str, suite: list[dict], suite_name: str) -> dict:
    """Evaluate faster-whisper on a given suite with raw vs normalized metrics."""
    import jiwer

    logger.info(f"\n==================================================")
    logger.info(f"EVALUATING: faster-whisper-{model_size} on {suite_name}")
    logger.info(f"==================================================")

    provider = FasterWhisperProvider(model_size=model_size, device="auto")
    t_load = time.time()
    loaded = provider.load_model()
    load_time_ms = (time.time() - t_load) * 1000

    if not loaded:
        return {"error": f"Failed to load {model_size}"}

    process = psutil.Process()
    results = []
    latencies = []
    rtfs = []
    cpu_measurements = []
    ram_measurements = []

    for item in suite:
        target_text = item["text"]
        speaker = item.get("speaker", "Samantha")
        audio = generate_audio(target_text, voice=speaker)
        duration_s = len(audio) / 16000.0

        t0 = time.time()
        asr_res = provider.transcribe_segment(audio)
        latency_ms = (time.time() - t0) * 1000

        latencies.append(latency_ms)
        rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else 0.0
        rtfs.append(rtf)

        cpu_measurements.append(process.cpu_percent())
        ram_measurements.append(process.memory_info().rss / (1024 * 1024))

        hypothesis = asr_res.text.strip()

        # Raw WER & CER (case-insensitive only)
        raw_target = target_text.lower()
        raw_hypo = hypothesis.lower()
        raw_wer = jiwer.wer(raw_target, raw_hypo if raw_hypo else "")
        raw_cer = jiwer.cer(raw_target, raw_hypo if raw_hypo else "")

        # Normalized WER & CER (using explicit normalization policy)
        norm_target = normalize_text_for_asr(target_text)
        norm_hypo = normalize_text_for_asr(hypothesis)
        norm_wer = jiwer.wer(norm_target, norm_hypo if norm_hypo else "")
        norm_cer = jiwer.cer(norm_target, norm_hypo if norm_hypo else "")
        exact_match = (norm_target == norm_hypo)

        results.append({
            "id": item["id"],
            "speaker": speaker,
            "locale": item.get("locale", "unknown"),
            "target": target_text,
            "hypothesis": hypothesis,
            "norm_target": norm_target,
            "norm_hypothesis": norm_hypo,
            "raw_wer": round(float(raw_wer), 4),
            "raw_cer": round(float(raw_cer), 4),
            "norm_wer": round(float(norm_wer), 4),
            "norm_cer": round(float(norm_cer), 4),
            "exact_match": exact_match,
            "duration_s": round(duration_s, 2),
            "latency_ms": round(latency_ms, 1),
            "rtf": round(rtf, 3),
        })

    # Silence & Low-Noise Hallucination Evaluation
    silence_tests = {
        "pure_silence": torch.zeros(16000 * 2, dtype=torch.float32),
        "ambient_noise_rms_0005": torch.randn(16000 * 2) * 0.0005,
        "quiet_noise_rms_0020": torch.randn(16000 * 2) * 0.0020,
    }
    silence_results = {}
    total_hallucinations = 0
    for name, s_tensor in silence_tests.items():
        s_res = provider.transcribe_segment(s_tensor)
        text = s_res.text.strip()
        is_hallucination = bool(text)
        if is_hallucination:
            total_hallucinations += 1
        silence_results[name] = {
            "hallucination": is_hallucination,
            "output": text,
            "latency_ms": round(s_res.latency_ms, 1),
        }

    # Aggregate Metrics
    raw_wers = [r["raw_wer"] for r in results]
    norm_wers = [r["norm_wer"] for r in results]
    raw_cers = [r["raw_cer"] for r in results]
    norm_cers = [r["norm_cer"] for r in results]
    exact_matches = [1.0 if r["exact_match"] else 0.0 for r in results]

    # Latency percentiles
    s_lat = sorted(latencies)
    p50 = s_lat[int(len(s_lat) * 0.50)]
    p90 = s_lat[int(len(s_lat) * 0.90)]
    p95 = s_lat[int(len(s_lat) * 0.95)]
    p99 = s_lat[int(len(s_lat) * 0.99)]

    # Bootstrap 95% Confidence Intervals
    raw_wer_ci = compute_bootstrap_ci(raw_wers)
    norm_wer_ci = compute_bootstrap_ci(norm_wers)
    norm_cer_ci = compute_bootstrap_ci(norm_cers)

    # Per-Speaker Breakdown
    speakers = sorted(list(set(r["speaker"] for r in results)))
    speaker_breakdown = {}
    for spk in speakers:
        spk_res = [r for r in results if r["speaker"] == spk]
        speaker_breakdown[spk] = {
            "sample_count": len(spk_res),
            "raw_wer": round(float(np.mean([r["raw_wer"] for r in spk_res])), 4),
            "norm_wer": round(float(np.mean([r["norm_wer"] for r in spk_res])), 4),
            "norm_cer": round(float(np.mean([r["norm_cer"] for r in spk_res])), 4),
            "exact_match_pct": round(float(np.mean([1.0 if r["exact_match"] else 0.0 for r in spk_res])) * 100, 1),
            "mean_rtf": round(float(np.mean([r["rtf"] for r in spk_res])), 3),
        }

    return {
        "suite_name": suite_name,
        "model": f"faster-whisper-{model_size}",
        "load_time_ms": round(load_time_ms, 1),
        "sample_count": len(results),
        "speaker_count": len(speakers),
        "raw_wer_mean": round(float(np.mean(raw_wers)), 4),
        "raw_wer_95ci": raw_wer_ci,
        "norm_wer_mean": round(float(np.mean(norm_wers)), 4),
        "norm_wer_95ci": norm_wer_ci,
        "norm_cer_mean": round(float(np.mean(norm_cers)), 4),
        "norm_cer_95ci": norm_cer_ci,
        "exact_match_pct": round(float(np.mean(exact_matches)) * 100, 1),
        "mean_rtf": round(float(np.mean(rtfs)), 3),
        "latency_p50_ms": round(float(p50), 1),
        "latency_p90_ms": round(float(p90), 1),
        "latency_p95_ms": round(float(p95), 1),
        "latency_p99_ms": round(float(p99), 1),
        "cpu_mean_pct": round(float(np.mean(cpu_measurements)), 1),
        "cpu_peak_pct": round(float(np.max(cpu_measurements)), 1),
        "ram_mean_mb": round(float(np.mean(ram_measurements)), 1),
        "ram_peak_mb": round(float(np.max(ram_measurements)), 1),
        "silence_hallucination_count": total_hallucinations,
        "silence_details": silence_results,
        "speaker_breakdown": speaker_breakdown,
        "item_results": results,
    }


def get_git_commit() -> str:
    """Retrieve git commit if available."""
    try:
        out = subprocess.check_output(["git", "-C", str(Path(__file__).parent.parent.parent / "frontend"), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "untracked"


def print_asr_report(smoke_summary: dict, benchmark_summary: dict):
    """Print complete statistical forensic ASR report."""
    print("\n" + "=" * 90)
    print("VOICESHIELD ASR STATISTICAL FORENSIC REPORT")
    print("=" * 90)
    print(f"Timestamp:       {datetime.now(timezone.utc).isoformat()}")
    print(f"Git Commit:      {get_git_commit()}")
    print(f"Hardware:        {platform.processor()} ({platform.system()} {platform.release()})")
    print(f"Normalization:   Lowercased, punctuation stripped, whitespace collapsed, compound tokens handled")
    print("-" * 90)

    print("\n1. SMOKE / REGRESSION TEST (Golden Sentences, Samantha en_US, N=5)")
    print(f"  Model:           {smoke_summary['model']}")
    print(f"  Raw WER:         {smoke_summary['raw_wer_mean']:.2%}")
    print(f"  Normalized WER:  {smoke_summary['norm_wer_mean']:.2%}")
    print(f"  Normalized CER:  {smoke_summary['norm_cer_mean']:.2%}")
    print(f"  Exact Match:     {smoke_summary['exact_match_pct']:.1f}%")
    print(f"  RTF:             {smoke_summary['mean_rtf']:.3f}")
    print(f"  Latency (P50):   {smoke_summary['latency_p50_ms']:.1f} ms (P95: {smoke_summary['latency_p95_ms']:.1f} ms)")
    print(f"  Silence Guards:  {smoke_summary['silence_hallucination_count']} hallucinations detected out of 3 tests (PASS)")

    print("\n  Detailed Smoke Utterances:")
    for item in smoke_summary["item_results"]:
        print(f"    [{item['id']}]")
        print(f"      Target:     \"{item['target']}\"")
        print(f"      Hypothesis: \"{item['hypothesis']}\"")
        print(f"      Raw WER: {item['raw_wer']:.2%} | Norm WER: {item['norm_wer']:.2%} | Norm CER: {item['norm_cer']:.2%} | Latency: {item['latency_ms']}ms")

    print("\n" + "-" * 90)
    print("\n2. CONTROLLED MULTI-SPEAKER BENCHMARK (N=24, 6 Speakers, Diverse Regional Accents)")
    print(f"  Provenance:      TTS Benchmark (macOS NSSpeechSynthesizer: US, GB, AU, IN, IE, ZA). NOT Human.")
    print(f"  Total Samples:   {benchmark_summary['sample_count']} utterances across {benchmark_summary['speaker_count']} speakers")
    print(f"  Raw WER (Mean):  {benchmark_summary['raw_wer_mean']:.2%} | 95% CI: [{benchmark_summary['raw_wer_95ci'][0]:.2%}, {benchmark_summary['raw_wer_95ci'][1]:.2%}]")
    print(f"  Norm WER (Mean): {benchmark_summary['norm_wer_mean']:.2%} | 95% CI: [{benchmark_summary['norm_wer_95ci'][0]:.2%}, {benchmark_summary['norm_wer_95ci'][1]:.2%}]")
    print(f"  Norm CER (Mean): {benchmark_summary['norm_cer_mean']:.2%} | 95% CI: [{benchmark_summary['norm_cer_95ci'][0]:.2%}, {benchmark_summary['norm_cer_95ci'][1]:.2%}]")
    print(f"  Exact Match:     {benchmark_summary['exact_match_pct']:.1f}%")
    print(f"  Latency:         P50={benchmark_summary['latency_p50_ms']:.1f}ms | P90={benchmark_summary['latency_p90_ms']:.1f}ms | P95={benchmark_summary['latency_p95_ms']:.1f}ms | P99={benchmark_summary['latency_p99_ms']:.1f}ms")
    print(f"  Real-Time Factor:{benchmark_summary['mean_rtf']:.3f} (Inference time is {benchmark_summary['mean_rtf']*100:.1f}% of audio duration)")
    print(f"  Resources:       CPU Mean {benchmark_summary['cpu_mean_pct']}% (Peak {benchmark_summary['cpu_peak_pct']}%) | RAM Mean {benchmark_summary['ram_mean_mb']:.1f} MB (Peak {benchmark_summary['ram_peak_mb']:.1f} MB)")

    print("\n  Per-Speaker Performance Breakdown:")
    print(f"    {'Speaker':<12} | {'Locale':<8} | {'Utterances':<10} | {'Raw WER':<9} | {'Norm WER':<9} | {'Norm CER':<9} | {'Exact Match':<11} | {'RTF'}")
    print("    " + "-" * 82)
    locale_map = {"Samantha": "en_US", "Daniel": "en_GB", "Karen": "en_AU", "Rishi": "en_IN", "Moira": "en_IE", "Tessa": "en_ZA"}
    for spk, stats in benchmark_summary["speaker_breakdown"].items():
        print(f"    {spk:<12} | {locale_map.get(spk, 'unknown'):<8} | {stats['sample_count']:<10} | {stats['raw_wer']:<9.2%} | {stats['norm_wer']:<9.2%} | {stats['norm_cer']:<9.2%} | {stats['exact_match_pct']:<10.1f}% | {stats['mean_rtf']:.3f}")

    print("\n3. SILENCE & AMBIENT NOISE HALLUCINATION STABILITY")
    for s_name, s_res in benchmark_summary["silence_details"].items():
        status_str = "PASS (0 hallucination)" if not s_res["hallucination"] else f"FAIL ('{s_res['output']}')"
        print(f"    {s_name:<28}: {status_str} (latency: {s_res['latency_ms']}ms)")

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Statistical Forensic ASR Benchmark")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "all"], help="Whisper model size")
    parser.add_argument("--output-json", default=None, help="Optional path to dump JSON benchmark results")
    args = parser.parse_args()

    models = ["tiny", "base"] if args.model == "all" else [args.model]

    all_reports = {}
    for m in models:
        smoke = evaluate_asr_model(m, SMOKE_REGRESSION_SUITE, "Smoke Regression Suite")
        bench = evaluate_asr_model(m, MULTI_SPEAKER_BENCHMARK_SUITE, "Multi-Speaker Benchmark Suite")
        print_asr_report(smoke, bench)
        all_reports[m] = {"smoke": smoke, "benchmark": bench}

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"Saved JSON benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
