"""
Adversarial & Forensic ASR Benchmark for VoiceShield Live Captions.

Audits speech recognition under harsh, realistic, and adversarial conditions:
1. Spontaneous conversational speech (fillers, hesitations, self-corrections)
2. Short utterances (1-3 words)
3. Long complex multi-clause sentences
4. Intra-sentence pauses (1.5s silence gap)
5. Multi-sentence utterances
6. Multiple accents (US, GB, AU, IN, IE, ZA)
7. Fast speech (+40% speech rate)
8. Slow speech (-30% speech rate)
9. Additive acoustic background noise (SNR 20 dB and SNR 10 dB)
10. Digital silence & low-level microphone hum (hallucination audit)
11. Microphone distance / low amplitude attenuation (-18 dB)

Computes:
- Unvarnished WER (Word Error Rate)
- Unvarnished CER (Character Error Rate)
- Exact Match Rate (%)
- Hallucination counts on silence/hum
- P50, P95, Max Latency (ms)
- Real-Time Factor (RTF)
"""

import os
import sys
import time
import re
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.asr.hf_whisper_provider import HFWhisperProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("benchmark.adversarial_asr")


def normalize_text(text: str) -> str:
    """Standardized phonetic normalization for unvarnished WER/CER."""
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


def generate_speech(text: str, voice: str, rate: int = 175) -> torch.Tensor:
    """Synthesize speech using macOS say at specified speaking rate (words per minute)."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(["say", "-v", voice, "-r", str(rate), text, "-o", tmp_path], check=True)
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


def add_noise(audio: torch.Tensor, target_snr_db: float) -> torch.Tensor:
    """Add Gaussian noise to achieve target SNR (dB)."""
    signal_power = torch.mean(audio ** 2)
    if signal_power <= 0:
        return audio
    noise_power = signal_power / (10 ** (target_snr_db / 10.0))
    noise = torch.randn_like(audio) * torch.sqrt(noise_power)
    return audio + noise


# Comprehensive 18-Test Adversarial Evaluation Corpus
ADVERSARIAL_TESTS = [
    # Category 1: Spontaneous speech (fillers, hesitations)
    {
        "id": "spontaneous_1",
        "category": "spontaneous_speech",
        "voice": "Samantha",
        "rate": 175,
        "text": "Um I was wondering if you could uh check the authorization status.",
        "perturbation": "none",
    },
    {
        "id": "spontaneous_2",
        "category": "spontaneous_speech",
        "voice": "Daniel",
        "rate": 175,
        "text": "Well actually wait no I meant to transfer thirty five dollars.",
        "perturbation": "none",
    },
    # Category 2: Short utterances
    {
        "id": "short_1",
        "category": "short_utterance",
        "voice": "Samantha",
        "rate": 175,
        "text": "Yes confirmed.",
        "perturbation": "none",
    },
    {
        "id": "short_2",
        "category": "short_utterance",
        "voice": "Daniel",
        "rate": 175,
        "text": "Cancel that now.",
        "perturbation": "none",
    },
    # Category 3: Long complex sentence
    {
        "id": "long_1",
        "category": "long_sentence",
        "voice": "Samantha",
        "rate": 175,
        "text": "The telecommunications infrastructure team completed the cryptographic audit and determined that all endpoints complied with the required mutual authentication protocols.",
        "perturbation": "none",
    },
    # Category 4: Intra-sentence pause (1.5s pause inserted)
    {
        "id": "pause_1",
        "category": "pause",
        "voice": "Karen",
        "rate": 175,
        "part1": "I am looking at the dashboard right now",
        "part2": "and the signal looks completely stable.",
        "text": "I am looking at the dashboard right now and the signal looks completely stable.",
        "perturbation": "pause_1.5s",
    },
    # Category 5: Multiple sentences
    {
        "id": "multisentence_1",
        "category": "multiple_sentences",
        "voice": "Daniel",
        "rate": 175,
        "text": "Please confirm the transaction details. We need your security token before authorization.",
        "perturbation": "none",
    },
    # Category 6: Accents
    {
        "id": "accent_in",
        "category": "accent",
        "voice": "Rishi",  # en_IN
        "rate": 175,
        "text": "The digital payment verification requires entering your six digit one time password.",
        "perturbation": "none",
    },
    {
        "id": "accent_ie",
        "category": "accent",
        "voice": "Moira",  # en_IE
        "rate": 175,
        "text": "Security analysts detected anomalous acoustic patterns during the biometric handshake.",
        "perturbation": "none",
    },
    {
        "id": "accent_za",
        "category": "accent",
        "voice": "Tessa",  # en_ZA
        "rate": 175,
        "text": "The communication channel is protected by end to end encryption.",
        "perturbation": "none",
    },
    {
        "id": "accent_au",
        "category": "accent",
        "voice": "Karen",  # en_AU
        "rate": 175,
        "text": "Please speak clearly into the microphone so the audio analyzer can calibrate.",
        "perturbation": "none",
    },
    # Category 7: Fast speech (+40% speed: rate 245)
    {
        "id": "fast_speech_1",
        "category": "fast_speech",
        "voice": "Samantha",
        "rate": 245,
        "text": "We need immediate confirmation of the account balance before the wire transfer closes.",
        "perturbation": "fast_rate_245",
    },
    # Category 8: Slow speech (-30% speed: rate 125)
    {
        "id": "slow_speech_1",
        "category": "slow_speech",
        "voice": "Daniel",
        "rate": 125,
        "text": "Please repeat the security code very slowly.",
        "perturbation": "slow_rate_125",
    },
    # Category 9: Background noise
    {
        "id": "noise_snr20",
        "category": "background_noise_20db",
        "voice": "Samantha",
        "rate": 175,
        "text": "The verification system operates continuously throughout the telephone conversation.",
        "perturbation": "snr_20db",
    },
    {
        "id": "noise_snr10",
        "category": "background_noise_10db",
        "voice": "Daniel",
        "rate": 175,
        "text": "Emergency protocols require dual administrative approval for all wire transfers.",
        "perturbation": "snr_10db",
    },
    # Category 10: Microphone distance / attenuation (-18 dB)
    {
        "id": "quiet_mic_dist",
        "category": "quiet_speech",
        "voice": "Samantha",
        "rate": 175,
        "text": "Can you hear me properly through this headset microphone.",
        "perturbation": "attenuate_minus_18db",
    },
]


def prepare_adversarial_audio(item: dict) -> torch.Tensor:
    """Prepare audio tensor applying specific perturbation."""
    pert = item.get("perturbation", "none")

    if pert == "pause_1.5s":
        p1 = generate_speech(item["part1"], item["voice"], item["rate"])
        p2 = generate_speech(item["part2"], item["voice"], item["rate"])
        silence_1_5s = torch.zeros(int(1.5 * 16000), dtype=torch.float32)
        audio = torch.cat([p1, silence_1_5s, p2])
    else:
        audio = generate_speech(item["text"], item["voice"], item["rate"])

    if pert == "snr_20db":
        audio = add_noise(audio, target_snr_db=20.0)
    elif pert == "snr_10db":
        audio = add_noise(audio, target_snr_db=10.0)
    elif pert == "attenuate_minus_18db":
        # -18 dB = factor of 0.125
        audio = audio * 0.125

    return audio


def run_adversarial_benchmark():
    logger.info("Generating adversarial evaluation dataset...")
    dataset = []
    for item in ADVERSARIAL_TESTS:
        try:
            audio = prepare_adversarial_audio(item)
            dataset.append({
                "id": item["id"],
                "category": item["category"],
                "text": item["text"],
                "voice": item["voice"],
                "rate": item["rate"],
                "perturbation": item["perturbation"],
                "audio": audio,
                "duration_s": len(audio) / 16000.0,
            })
        except Exception as e:
            logger.error(f"Failed to generate {item['id']}: {e}")

    logger.info(f"Generated {len(dataset)} adversarial test samples.")

    # Silence and line-hum tests
    silence_3s = torch.zeros(48000, dtype=torch.float32)
    line_hum_3s = torch.from_numpy(np.random.normal(0, 0.0015, 48000).astype(np.float32))

    models_to_test = [
        {"name": "tiny", "beam_size": 1, "desc": "HF Whisper Tiny (Greedy, beam=1)"},
        {"name": "tiny", "beam_size": 5, "desc": "HF Whisper Tiny (Beam-5)"},
        {"name": "base", "beam_size": 1, "desc": "HF Whisper Base (Greedy, beam=1)"},
        {"name": "base", "beam_size": 5, "desc": "HF Whisper Base (Beam-5)"},
    ]

    all_results = []

    for cfg in models_to_test:
        logger.info(f"\n=======================================================")
        logger.info(f"Running Adversarial Audit: {cfg['desc']}")
        logger.info(f"=======================================================")

        provider = HFWhisperProvider(
            model_name=cfg["name"],
            backend="ctranslate2",
            beam_size=cfg["beam_size"],
            log_prob_threshold=-1.5,
            no_speech_threshold=0.6,
        )
        if not provider.load_model():
            logger.error(f"Failed to load {cfg['desc']}")
            continue

        wers, cers, exacts, latencies = [], [], [], []
        category_breakdown: Dict[str, Dict[str, Any]] = {}
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
            exacts.append(exact)

            cat = item["category"]
            if cat not in category_breakdown:
                category_breakdown[cat] = {"wers": [], "cers": [], "exacts": []}
            category_breakdown[cat]["wers"].append(wer)
            category_breakdown[cat]["cers"].append(cer)
            category_breakdown[cat]["exacts"].append(exact)

            logger.info(
                f"[{item['id']}] ({item['category']}) REF: '{ref_norm}'"
            )
            logger.info(
                f"[{item['id']}] HYP: '{hyp_norm}' | WER={wer:.2f} | Latency={res.latency_ms:.0f}ms"
            )

        # Silence & Hum Hallucination Check
        res_silence = provider.transcribe_segment(silence_3s)
        res_hum = provider.transcribe_segment(line_hum_3s)

        halluc_silence = 1 if res_silence.text.strip() else 0
        halluc_hum = 1 if res_hum.text.strip() else 0

        summary = {
            "model_desc": cfg["desc"],
            "model_size": cfg["name"],
            "beam_size": cfg["beam_size"],
            "overall_wer_pct": round(float(np.mean(wers)) * 100.0, 2),
            "overall_cer_pct": round(float(np.mean(cers)) * 100.0, 2),
            "exact_match_pct": round(float(np.mean(exacts)) * 100.0, 1),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "rtf": round(total_inference_time / total_audio_time, 3) if total_audio_time > 0 else 0.0,
            "silence_hallucination": halluc_silence,
            "line_hum_hallucination": halluc_hum,
            "category_metrics": {
                k: {
                    "mean_wer_pct": round(float(np.mean(v["wers"])) * 100.0, 2),
                    "mean_cer_pct": round(float(np.mean(v["cers"])) * 100.0, 2),
                    "exact_match_pct": round(float(np.mean(v["exacts"])) * 100.0, 1),
                    "count": len(v["wers"]),
                }
                for k, v in category_breakdown.items()
            }
        }
        all_results.append(summary)

    print("\n" + "="*80)
    print("ADVERSARIAL ASR AUDIT RESULTS (REALISTIC & HARSH CONDITIONS)")
    print("="*80)
    for s in all_results:
        print(f"\nModel: {s['model_desc']}")
        print(f"  Overall WER: {s['overall_wer_pct']}% | CER: {s['overall_cer_pct']}% | Exact Match: {s['exact_match_pct']}%")
        print(f"  P50 Latency: {s['latency_p50_ms']}ms | P95 Latency: {s['latency_p95_ms']}ms | RTF: {s['rtf']}")
        print(f"  Silence Hallucination: {s['silence_hallucination']} | Line Hum Hallucination: {s['line_hum_hallucination']}")
        print("  Category WER Breakdown:")
        for cat, m in s["category_metrics"].items():
            print(f"    - {cat:24s}: WER={m['mean_wer_pct']:5.1f}% | Exact={m['exact_match_pct']:5.1f}%")

    out_file = Path(__file__).parent / "adversarial_asr_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved adversarial benchmark report to {out_file}")


if __name__ == "__main__":
    run_adversarial_benchmark()
