"""
VoiceShield Benchmark CLI Runner.

Usage:
    python -m benchmarks.run --dataset custom_wav --path ./datasets/my_data --split val
    python -m benchmarks.run --dataset custom_wav --path ./datasets/my_data --split test --mode fast
    python -m benchmarks.run --dataset custom_wav --path ./datasets/my_data --split val --output results/

No arbitrary filesystem paths via HTTP. This is CLI-only.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import psutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.dataset import get_dataset, BenchmarkSample
from benchmarks.evaluator import BenchmarkEvaluator, Prediction
from benchmarks.replay_engine import ReplayEngine

from app.services.monitor.audio_bus import AudioBus, OverflowPolicy
from app.services.monitor.security_pipeline import (
    SecurityPipeline, DetectionEvent, AASIST_WINDOW_SAMPLES
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("benchmark")


async def run_benchmark(args):
    """Run the benchmark on a registered dataset."""

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} from {args.path}")
    dataset = get_dataset(args.dataset, args.path)
    samples = dataset.get_split(args.split)
    logger.info(f"Split '{args.split}': {len(samples)} samples")

    n_spoof = sum(1 for s in samples if s.label == 1)
    n_bonafide = sum(1 for s in samples if s.label == 0)
    logger.info(f"  Spoof: {n_spoof}, Bonafide: {n_bonafide}")

    # Load AASIST model
    logger.info("Loading AASIST model...")
    from app.services.antispoof.spectra_detector import spectra_detector
    if not spectra_detector.is_loaded:
        spectra_detector.load_model()
    logger.info(f"AASIST: {spectra_detector.status}")

    # Collect predictions
    predictions: list[Prediction] = []
    process = psutil.Process()
    cpu_readings = []
    ram_readings = []

    start_time = time.time()

    for i, sample in enumerate(samples):
        try:
            # Load and preprocess
            from app.services.audio.preprocessor import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            waveform, sr = preprocessor.load_file(sample.file_path)
            waveform = preprocessor.preprocess(waveform, sr)

            # Window the audio (same as streaming pipeline)
            windows = preprocessor.create_windows(waveform)

            # Run AASIST on each window
            for window in windows:
                t0 = time.time()
                result = spectra_detector.predict(window)
                latency = (time.time() - t0) * 1000

                predictions.append(Prediction(
                    file=str(sample.file_path),
                    ground_truth=sample.label,
                    spoof_score=result.spoof_score,
                    raw_logit_spoof=result.raw_logit_spoof,
                    raw_logit_bonafide=result.raw_logit_bonafide,
                    latency_ms=latency,
                ))

            # Resource monitoring
            cpu_readings.append(process.cpu_percent())
            ram_readings.append(process.memory_info().rss / 1024 / 1024)

            if (i + 1) % 50 == 0 or i == len(samples) - 1:
                logger.info(
                    f"[{i+1}/{len(samples)}] "
                    f"Predictions: {len(predictions)}, "
                    f"RAM: {ram_readings[-1]:.0f}MB"
                )

        except Exception as e:
            logger.error(f"Error processing {sample.file_path}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Inference complete: {len(predictions)} predictions in {elapsed:.1f}s")

    # Evaluate
    evaluator = BenchmarkEvaluator()
    metrics = evaluator.evaluate_antispoof(
        predictions=predictions,
        threshold=None,  # Use EER threshold
        threshold_source=f"EER from {args.split} split",
    )

    # Add resource metrics
    import numpy as np
    if cpu_readings:
        metrics.cpu_pct_mean = round(float(np.mean(cpu_readings)), 1)
        metrics.cpu_pct_peak = round(float(np.max(cpu_readings)), 1)
    if ram_readings:
        metrics.ram_mb_mean = round(float(np.mean(ram_readings)), 1)
        metrics.ram_mb_peak = round(float(np.max(ram_readings)), 1)

    # Print results
    print("\n" + "=" * 60)
    print(f"VOICESHIELD BENCHMARK RESULTS")
    print(f"Dataset: {dataset.get_name()}")
    print(f"Split: {args.split}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    print(f"\nAnti-Spoof Detection (AASIST)")
    print(f"  Samples: {metrics.n_spoof} spoof + {metrics.n_bonafide} bonafide")
    print(f"  AUC-ROC:           {metrics.auc_roc}")
    print(f"  EER:               {metrics.eer}")
    print(f"  EER Threshold:     {metrics.eer_threshold}")
    print(f"  FPR @ 1% FNR:      {metrics.fpr_at_1pct_fnr}")
    print(f"  FNR @ 1% FPR:      {metrics.fnr_at_1pct_fpr}")
    print(f"  Precision:         {metrics.precision}")
    print(f"  Recall:            {metrics.recall}")
    print(f"  F1:                {metrics.f1}")
    print(f"  Op. Threshold:     {metrics.operating_threshold} ({metrics.threshold_source})")
    print(f"\nLatency")
    print(f"  P50: {metrics.detection_p50_ms}ms")
    print(f"  P90: {metrics.detection_p90_ms}ms")
    print(f"  P95: {metrics.detection_p95_ms}ms")
    print(f"  P99: {metrics.detection_p99_ms}ms")
    print(f"\nResources")
    print(f"  CPU mean: {metrics.cpu_pct_mean}%  peak: {metrics.cpu_pct_peak}%")
    print(f"  RAM mean: {metrics.ram_mb_mean}MB  peak: {metrics.ram_mb_peak}MB")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)

    # Save results
    output_dir = Path(args.output)
    output_file = output_dir / f"benchmark_{args.split}_{int(time.time())}.json"
    evaluator.save_results(metrics, output_file)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="VoiceShield Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarks.run --dataset custom_wav --path ./datasets/test --split val
    python -m benchmarks.run --dataset custom_wav --path ./datasets/test --split test --mode fast
        """,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Registered dataset name (e.g., custom_wav)",
    )
    parser.add_argument(
        "--path", required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--split", default="val",
        help="Dataset split to evaluate (train/val/test)",
    )
    parser.add_argument(
        "--mode", default="fast", choices=["realtime", "fast"],
        help="Replay mode: realtime (latency test) or fast (accuracy test)",
    )
    parser.add_argument(
        "--output", default="./benchmark_results",
        help="Output directory for results",
    )

    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
