#!/usr/bin/env python3
"""
Generate verified calibration artifact for VoiceShield Mobile.

CRITICAL HONESTY RULE:
Do not use arbitrary thresholds (e.g. spoof > 0.85, similarity < 0.3) without
evaluation provenance. If evaluation data has not been run, this generator
emits a status of CALIBRATION_NOT_AVAILABLE so the application never pretends
to have calibrated thresholds.
"""

import os
import json
import datetime
import numpy as np


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Compute Equal Error Rate (EER) and the optimal decision threshold.
    """
    thresholds = np.linspace(0.0, 1.0, 1000)
    frr = np.array([np.mean(bonafide_scores > t) for t in thresholds])  # False rejection (bona fide scored as spoof)
    far = np.array([np.mean(spoof_scores <= t) for t in thresholds])   # False acceptance (spoof scored as bona fide)

    idx = np.nanargmin(np.abs(frr - far))
    eer = (frr[idx] + far[idx]) / 2.0
    optimal_threshold = thresholds[idx]

    return float(optimal_threshold), float(eer), float(far[idx]), float(frr[idx])


def generate_calibration(output_path="ml/calibration/calibration_v1.json", bonafide_scores=None, spoof_scores=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if bonafide_scores is None or spoof_scores is None:
        # Default unverified state: Honest reporting of uncalibrated status
        artifact = {
            "status": "CALIBRATION_NOT_AVAILABLE",
            "model_name": "AASIST-L",
            "model_revision": "SpeechAntiSpoofingBenchmarks/AASIST-L",
            "evaluation_dataset": "None (Awaiting benchmark execution)",
            "evaluation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sample_count": 0,
            "operating_point": None,
            "note": "Raw anti-spoof model scores are reported without mapping to probabilistic risk levels until evaluation on target corpus is completed."
        }
    else:
        threshold, eer, far, frr = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
        artifact = {
            "status": "VERIFIED",
            "model_name": "AASIST-L",
            "model_revision": "SpeechAntiSpoofingBenchmarks/AASIST-L",
            "evaluation_dataset": "ASVspoof2019_LA_eval",
            "evaluation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sample_count": len(bonafide_scores) + len(spoof_scores),
            "operating_point": {
                "threshold": round(threshold, 4),
                "eer": round(eer, 4),
                "far_at_threshold": round(far, 4),
                "frr_at_threshold": round(frr, 4)
            },
            "note": "Threshold derived from EER optimization on evaluated benchmark data."
        }

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Generated calibration artifact at: {output_path}")
    print(f"Status: {artifact['status']}")
    return artifact


if __name__ == "__main__":
    generate_calibration()
