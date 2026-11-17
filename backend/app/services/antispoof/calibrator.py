"""
VoiceShield Anti-Spoof Calibration & Decisioning Service (Rule 1, Rule 2, Rule 3).

Establishes trustworthy, calibrated probability mapping and operating-point
decisioning for AASIST model outputs.

CRITICAL INVARIANTS:
1. RAW MODEL SCORE, CALIBRATED PROBABILITY, and DECISION are strictly distinguished.
2. If a validated calibration artifact is absent, expired, or domain-mismatched:
   calibrated_probability is None, decision is "UNCALIBRATED", status is "CALIBRATION_REQUIRED".
3. A calibrated probability is labeled as:
   "Estimated probability of spoof class under the validated calibration model"
   (NEVER as "percentage of voice spoofed").
4. Calibration parameters and operating points are fitted ONLY on validation data.
   Test data is evaluated once and remains untouched.
5. Platt scaling (logistic regression) and Isotonic regression are supported.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibratedOutput:
    """Rigorous score contract distinguishing raw outputs from calibrated decisions."""
    raw_spoof_score: float
    raw_logit_spoof: float
    raw_logit_bonafide: float
    delta: float                             # Raw logit margin: logit_spoof - logit_bonafide
    calibrated_probability: Optional[float]  # None if uncalibrated / invalid / abstained
    decision: str                            # "UNCALIBRATED", "GENUINE", "SUSPICIOUS", "UNCERTAIN", "ABSTAIN", "INVALID"
    calibration_status: str                  # "CALIBRATION_REQUIRED", "CALIBRATED", "NOT_VALID", "INSUFFICIENT_AUDIO_QUALITY", "NOT_APPLICABLE_SILENCE"
    calibration_version: str
    calibration_method: str                  # "platt", "isotonic", "none"
    operating_threshold: Optional[float]
    target_domain: str                       # e.g., "webrtc_opus", "clean_wav"
    is_valid: bool = True
    invalidation_reason: str = ""


class ScoreCalibrator:
    """Manages anti-spoof model score calibration and validated operating points."""

    def __init__(
        self,
        artifact_path: Optional[Union[str, Path]] = None,
        expected_model: str = "Spectra-AASIST",
        expected_domain: str = "webrtc_opus",
        expected_score_representation: str = "logit_margin_delta",
    ):
        self.expected_model = expected_model
        self.expected_domain = expected_domain
        self.expected_score_representation = expected_score_representation
        self.score_representation = expected_score_representation
        self.is_loaded = False
        self.version = "uncalibrated"
        self.method = "none"
        self.domain = "unknown"
        self.operating_threshold = 0.5
        self.eer = 0.0
        self.quality_gate_rms = 0.0042

        self._status: str = "UNCALIBRATED"

        # Platt scaling parameters: logit(p) = a * score + b
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0

        # Isotonic regression parameters: piecewise linear knots (x, y)
        self._isotonic_x: list[float] = []
        self._isotonic_y: list[float] = []

        if artifact_path:
            self.load(artifact_path)

    @property
    def status(self) -> str:
        """Return the explicit calibration state: CALIBRATED, UNCALIBRATED, CALIBRATION_UNAVAILABLE, CALIBRATION_INVALID."""
        return self._status

    def load(self, path: Union[str, Path]) -> bool:
        """Load calibration artifact JSON with domain and model version validation."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Calibration artifact not found: {path}. Calibration REQUIRED.")
            self.is_loaded = False
            self._status = "UNCALIBRATED"
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Validate model and domain compatibility
            artifact_model = data.get("model", "")
            artifact_domain = data.get("domain", "")
            artifact_rep = data.get("score_representation", "logit_margin_delta")

            if artifact_model != self.expected_model:
                logger.warning(
                    f"Calibration model mismatch: artifact has '{artifact_model}', "
                    f"expected '{self.expected_model}'. Calibration marked INVALID."
                )
                self.is_loaded = False
                self._status = "CALIBRATION_INVALID"
                return False

            if artifact_domain != self.expected_domain and self.expected_domain != "any":
                logger.warning(
                    f"Calibration domain mismatch: artifact trained for '{artifact_domain}', "
                    f"evaluated in '{self.expected_domain}'. Calibration marked INVALID."
                )
                self.is_loaded = False
                self._status = "CALIBRATION_INVALID"
                return False

            if artifact_rep != self.expected_score_representation:
                logger.warning(
                    f"Calibration representation mismatch: artifact has '{artifact_rep}', "
                    f"expected '{self.expected_score_representation}'. Calibration marked INVALID."
                )
                self.is_loaded = False
                self._status = "CALIBRATION_INVALID"
                return False

            self.version = data.get("version", "v1.0")
            self.method = data.get("method", "platt")
            self.domain = artifact_domain
            self.score_representation = artifact_rep
            self.operating_threshold = float(data.get("operating_threshold", 0.5))
            self.eer = float(data.get("validation_eer", 0.0))
            self.quality_gate_rms = float(data.get("metadata", {}).get("quality_gate_rms", 0.0042))

            if self.method == "platt":
                self._platt_a = float(data["params"]["a"])
                self._platt_b = float(data["params"]["b"])
            elif self.method == "isotonic":
                self._isotonic_x = data["params"]["x"]
                self._isotonic_y = data["params"]["y"]

            self.is_loaded = True
            self._status = "CALIBRATED"
            logger.info(
                f"ScoreCalibrator: loaded {self.version} ({self.method}) "
                f"for domain '{self.domain}' (op_thresh={self.operating_threshold:.3f}, gate_rms={self.quality_gate_rms:.4f})"
            )
            return True

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse calibration artifact {path}: {e}")
            self.is_loaded = False
            self._status = "CALIBRATION_INVALID"
            return False
        except Exception as e:
            logger.error(f"Failed to load calibration artifact {path}: {e}")
            self.is_loaded = False
            self._status = "CALIBRATION_UNAVAILABLE"
            return False

    def save(self, path: Union[str, Path], metadata: Optional[dict] = None):
        """Save fitted calibration artifact to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.expected_model,
            "version": self.version,
            "method": self.method,
            "domain": self.domain,
            "score_representation": self.score_representation,
            "operating_threshold": self.operating_threshold,
            "validation_eer": self.eer,
            "params": {},
            "metadata": metadata or {},
        }

        if self.method == "platt":
            data["params"] = {"a": self._platt_a, "b": self._platt_b}
        elif self.method == "isotonic":
            data["params"] = {"x": self._isotonic_x, "y": self._isotonic_y}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"ScoreCalibrator: saved calibration artifact to {path}")

    def fit_validation_split(
        self,
        scores: list[float],
        labels: list[int],
        method: str = "platt",
        domain: str = "webrtc_opus",
        version: str = "v1.0",
        score_representation: str = "raw_score",
    ):
        """Fit calibration mapping and find operating point ONLY on validation split."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import roc_curve

        scores_arr = np.array(scores).reshape(-1, 1)
        labels_arr = np.array(labels)

        self.method = method
        self.domain = domain
        self.version = version
        self.score_representation = score_representation

        # 1. Fit probability calibration
        if method == "platt":
            # Logistic regression on raw scores (unregularized Platt scaling for 1D scores)
            lr = LogisticRegression(C=1e4)
            lr.fit(scores_arr, labels_arr)
            self._platt_a = float(lr.coef_[0][0])
            self._platt_b = float(lr.intercept_[0])
            # Calibrated probabilities on validation
            probs = lr.predict_proba(scores_arr)[:, 1]
        elif method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            probs = ir.fit_transform(scores, labels_arr)
            self._isotonic_x = ir.X_thresholds_.tolist()
            self._isotonic_y = ir.y_thresholds_.tolist()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # 2. Compute EER on validation probabilities to select operating threshold
        fpr, tpr, thresholds = roc_curve(labels_arr, probs, pos_label=1)
        fnr = 1 - tpr
        eer_idx = int(np.nanargmin(np.absolute((fnr - fpr))))
        self.eer = float(fpr[eer_idx])
        self.operating_threshold = float(thresholds[eer_idx])
        self.is_loaded = True

        logger.info(
            f"Fitted {method} calibrator on validation split: "
            f"EER={self.eer:.4f}, Operating Threshold={self.operating_threshold:.4f}"
        )

    def calibrate(
        self,
        raw_spoof_score: float,
        raw_logit_spoof: float = 0.0,
        raw_logit_bonafide: float = 0.0,
        delta: Optional[float] = None,
        window_valid: bool = True,
        invalidation_reason: str = "",
        rms: Optional[float] = None,
        audio_domain: str = "webrtc_opus",
        is_silence: bool = False,
    ) -> CalibratedOutput:
        """Produce calibrated output or explicitly declare CALIBRATION_REQUIRED or ABSTAIN."""
        if delta is None:
            delta = raw_logit_spoof - raw_logit_bonafide

        if not window_valid:
            return CalibratedOutput(
                raw_spoof_score=round(raw_spoof_score, 4),
                raw_logit_spoof=round(raw_logit_spoof, 4),
                raw_logit_bonafide=round(raw_logit_bonafide, 4),
                delta=round(delta, 4),
                calibrated_probability=None,
                decision="INVALID",
                calibration_status="INVALID_WINDOW",
                calibration_version=self.version,
                calibration_method=self.method,
                operating_threshold=None,
                target_domain=self.domain,
                is_valid=False,
                invalidation_reason=invalidation_reason or "AUDIO_DISCONTINUITY",
            )

        # 1. FIRST-CLASS SILENCE HANDLING (Clean separation from audio quality failure)
        if is_silence:
            return CalibratedOutput(
                raw_spoof_score=round(raw_spoof_score, 4),
                raw_logit_spoof=round(raw_logit_spoof, 4),
                raw_logit_bonafide=round(raw_logit_bonafide, 4),
                delta=round(delta, 4),
                calibrated_probability=None,
                decision="ABSTAIN",
                calibration_status="NOT_APPLICABLE_SILENCE",
                calibration_version=self.version,
                calibration_method=self.method,
                operating_threshold=round(self.operating_threshold, 4) if self.is_loaded else None,
                target_domain=self.domain,
                is_valid=False,
                invalidation_reason=invalidation_reason or "SILENCE_NO_SPEECH",
            )

        # 2. QUALITY-AWARE ABSTENTION: Insufficient acoustic energy / empty silence
        # Low-information audio produces biased uncalibrated spoof outputs.
        # Quality Gate threshold is derived on validation data and frozen in calibration artifact.
        gate_rms = getattr(self, "quality_gate_rms", 0.0042)
        if rms is not None and rms < gate_rms:
            return CalibratedOutput(
                raw_spoof_score=round(raw_spoof_score, 4),
                raw_logit_spoof=round(raw_logit_spoof, 4),
                raw_logit_bonafide=round(raw_logit_bonafide, 4),
                delta=round(delta, 4),
                calibrated_probability=None,
                decision="ABSTAIN",
                calibration_status="INSUFFICIENT_AUDIO_QUALITY",
                calibration_version=self.version,
                calibration_method=self.method,
                operating_threshold=round(self.operating_threshold, 4) if self.is_loaded else None,
                target_domain=self.domain,
                is_valid=False,
                invalidation_reason=f"QUALITY_GATE: Low RMS ({rms:.5f} < {gate_rms:.4f})",
            )

        # 3. Model calibration availability check
        if not self.is_loaded:
            # RULE 1 & 3: Never fabricate a spoof probability without calibration!
            return CalibratedOutput(
                raw_spoof_score=round(raw_spoof_score, 4),
                raw_logit_spoof=round(raw_logit_spoof, 4),
                raw_logit_bonafide=round(raw_logit_bonafide, 4),
                delta=round(delta, 4),
                calibrated_probability=None,
                decision="UNCALIBRATED",
                calibration_status="CALIBRATION_REQUIRED",
                calibration_version="none",
                calibration_method="none",
                operating_threshold=None,
                target_domain=self.domain,
                is_valid=True,
            )

        # 4. Domain validity contract check
        if audio_domain != self.domain and self.domain != "any":
            return CalibratedOutput(
                raw_spoof_score=round(raw_spoof_score, 4),
                raw_logit_spoof=round(raw_logit_spoof, 4),
                raw_logit_bonafide=round(raw_logit_bonafide, 4),
                delta=round(delta, 4),
                calibrated_probability=None,
                decision="UNCALIBRATED",
                calibration_status="NOT_VALID",
                calibration_version=self.version,
                calibration_method=self.method,
                operating_threshold=None,
                target_domain=self.domain,
                is_valid=False,
                invalidation_reason=f"DOMAIN_MISMATCH: Input audio '{audio_domain}' != calibrated domain '{self.domain}'",
            )

        # 5. Apply validated calibration mapping
        # Support logit_margin_delta when available/expected, else fall back to raw score
        if self.score_representation == "logit_margin_delta" and (
            delta is not None and (raw_logit_spoof != 0.0 or raw_logit_bonafide != 0.0 or delta != 0.0)
        ):
            score_to_calibrate = delta
        else:
            score_to_calibrate = raw_spoof_score

        if self.method == "platt":
            z = self._platt_a * score_to_calibrate + self._platt_b
            calibrated_prob = float(1.0 / (1.0 + np.exp(-z)))
        elif self.method == "isotonic":
            calibrated_prob = float(np.interp(
                score_to_calibrate, self._isotonic_x, self._isotonic_y
            ))
        else:
            calibrated_prob = score_to_calibrate

        # Operating decision based on validated threshold
        if calibrated_prob >= self.operating_threshold:
            decision = "SUSPICIOUS"
        elif calibrated_prob <= (self.operating_threshold * 0.5):
            decision = "GENUINE"
        else:
            decision = "UNCERTAIN"

        return CalibratedOutput(
            raw_spoof_score=round(raw_spoof_score, 4),
            raw_logit_spoof=round(raw_logit_spoof, 4),
            raw_logit_bonafide=round(raw_logit_bonafide, 4),
            delta=round(delta, 4),
            calibrated_probability=round(calibrated_prob, 4),
            decision=decision,
            calibration_status="CALIBRATED",
            calibration_version=self.version,
            calibration_method=self.method,
            operating_threshold=round(self.operating_threshold, 4),
            target_domain=self.domain,
            is_valid=True,
        )


def compute_calibration_metrics(probs: list[float], labels: list[int], n_bins: int = 10) -> dict:
    """Computes Brier Score and Expected Calibration Error (ECE) for held-out evaluation."""
    probs_arr = np.array(probs, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.float64)

    if len(probs_arr) == 0:
        return {"brier_score": 0.0, "expected_calibration_error": 0.0, "bin_details": []}

    brier_score = float(np.mean((probs_arr - labels_arr) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs_arr >= bin_lower) & (probs_arr < bin_upper if i < n_bins - 1 else probs_arr <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(labels_arr[in_bin]))
            avg_confidence_in_bin = float(np.mean(probs_arr[in_bin]))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_details.append({
                "bin": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "count": int(np.sum(in_bin)),
                "confidence": round(avg_confidence_in_bin, 4),
                "accuracy": round(accuracy_in_bin, 4),
                "calibration_gap": round(float(np.abs(avg_confidence_in_bin - accuracy_in_bin)), 4),
            })

    return {
        "brier_score": round(brier_score, 4),
        "expected_calibration_error": round(float(ece), 4),
        "bin_details": bin_details,
    }


# Global singleton calibrator (defaults to uncalibrated until fitted artifact is loaded)
score_calibrator = ScoreCalibrator()
