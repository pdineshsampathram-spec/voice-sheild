"""
Benchmark evaluator — computes metrics from collected predictions vs labels.

Score direction contract (from AASIST audit):
  - Positive class = spoof (label=1)
  - Negative class = bonafide (label=0)
  - Score for ROC/AUC = spoof_score (higher → more spoof-like)

Operating-point metrics are derived from VALIDATION split only.
Thresholds are frozen before TEST evaluation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction with ground truth."""
    file: str
    ground_truth: int          # 0 = bonafide, 1 = spoof
    spoof_score: float         # Higher → more spoof-like
    raw_logit_spoof: float = 0.0
    raw_logit_bonafide: float = 0.0
    latency_ms: float = 0.0
    window_valid: bool = True


@dataclass
class ASRPrediction:
    """A single ASR prediction with ground truth."""
    file: str
    hypothesis: str
    reference: str
    latency_ms: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Complete benchmark result."""
    # Anti-spoof
    auc_roc: float = 0.0
    eer: float = 0.0
    eer_threshold: float = 0.0
    fpr_at_1pct_fnr: Optional[float] = None
    fnr_at_1pct_fpr: Optional[float] = None
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    confusion_matrix: list = field(default_factory=list)
    operating_threshold: float = 0.0
    threshold_source: str = ""   # "validation" or "EER"
    n_spoof: int = 0
    n_bonafide: int = 0

    # ASR
    wer: float = 0.0
    cer: float = 0.0
    asr_n_samples: int = 0

    # Latency (anti-spoof)
    detection_p50_ms: float = 0.0
    detection_p90_ms: float = 0.0
    detection_p95_ms: float = 0.0
    detection_p99_ms: float = 0.0

    # Latency (ASR)
    asr_p50_ms: float = 0.0
    asr_p90_ms: float = 0.0
    asr_p95_ms: float = 0.0
    asr_p99_ms: float = 0.0

    # Resource
    cpu_pct_mean: float = 0.0
    cpu_pct_peak: float = 0.0
    ram_mb_mean: float = 0.0
    ram_mb_peak: float = 0.0


class BenchmarkEvaluator:
    """Computes benchmark metrics from predictions.

    Score direction: higher spoof_score → more spoof-like (verified).
    Positive class: spoof (label=1).
    """

    def evaluate_antispoof(
        self,
        predictions: list[Prediction],
        threshold: Optional[float] = None,
        threshold_source: str = "EER",
    ) -> BenchmarkMetrics:
        """Compute anti-spoof metrics.

        Args:
            predictions: List of predictions with ground truth.
            threshold: Operating threshold. If None, uses EER threshold.
            threshold_source: Description of where threshold came from.
        """
        from sklearn.metrics import (
            roc_auc_score, roc_curve, precision_score,
            recall_score, f1_score, confusion_matrix
        )

        # Filter valid windows
        valid = [p for p in predictions if p.window_valid]
        if not valid:
            logger.warning("No valid predictions for evaluation")
            return BenchmarkMetrics()

        y_true = np.array([p.ground_truth for p in valid])
        y_scores = np.array([p.spoof_score for p in valid])
        latencies = np.array([p.latency_ms for p in valid])

        n_spoof = int(np.sum(y_true == 1))
        n_bonafide = int(np.sum(y_true == 0))

        # AUC-ROC
        auc = 0.0
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_scores))
        else:
            logger.warning(
                "Only one class present in predictions — "
                "AUC is undefined"
            )

        # EER
        eer = 0.0
        eer_threshold = 0.5
        if len(np.unique(y_true)) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr
            # Find EER: where FPR ≈ FNR
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
            eer_threshold = float(thresholds[eer_idx])

        # Use provided threshold or EER threshold
        op_threshold = threshold if threshold is not None else eer_threshold

        # FPR @ 1% FNR
        fpr_at_1pct_fnr = None
        fnr_at_1pct_fpr = None
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            fnr = 1 - tpr
            # FPR where FNR ≤ 0.01
            mask = fnr <= 0.01
            if mask.any():
                fpr_at_1pct_fnr = float(fpr[mask][-1])
            # FNR where FPR ≤ 0.01
            mask = fpr <= 0.01
            if mask.any():
                fnr_at_1pct_fpr = float(fnr[mask][-1])

        # Binary predictions at operating point
        y_pred = (y_scores >= op_threshold).astype(int)

        metrics = BenchmarkMetrics(
            auc_roc=round(auc, 4),
            eer=round(eer, 4),
            eer_threshold=round(eer_threshold, 4),
            fpr_at_1pct_fnr=round(fpr_at_1pct_fnr, 4) if fpr_at_1pct_fnr is not None else None,
            fnr_at_1pct_fpr=round(fnr_at_1pct_fpr, 4) if fnr_at_1pct_fpr is not None else None,
            precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            operating_threshold=round(op_threshold, 4),
            threshold_source=threshold_source,
            n_spoof=n_spoof,
            n_bonafide=n_bonafide,
            detection_p50_ms=round(float(np.percentile(latencies, 50)), 2),
            detection_p90_ms=round(float(np.percentile(latencies, 90)), 2),
            detection_p95_ms=round(float(np.percentile(latencies, 95)), 2),
            detection_p99_ms=round(float(np.percentile(latencies, 99)), 2),
        )

        return metrics

    def evaluate_asr(
        self,
        predictions: list[ASRPrediction],
    ) -> dict:
        """Compute ASR metrics (WER, CER, latency)."""
        from jiwer import wer, cer

        refs = [p.reference for p in predictions if p.reference.strip()]
        hyps = [p.hypothesis for p in predictions if p.reference.strip()]
        latencies = np.array([p.latency_ms for p in predictions])

        if not refs:
            return {"wer": 0, "cer": 0, "n_samples": 0}

        return {
            "wer": round(float(wer(refs, hyps)), 4),
            "cer": round(float(cer(refs, hyps)), 4),
            "n_samples": len(refs),
            "asr_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "asr_p90_ms": round(float(np.percentile(latencies, 90)), 2),
            "asr_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "asr_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        }

    def save_results(self, metrics: BenchmarkMetrics, output_path: str | Path):
        """Save benchmark results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        result = {}
        for k in metrics.__dataclass_fields__:
            v = getattr(metrics, k)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            result[k] = v

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to {output_path}")
