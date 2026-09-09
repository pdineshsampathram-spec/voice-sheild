"""
Zero-Trust Forensic Invariant Test Suite.

Verifies the mathematical and runtime invariants required by the Zero-Trust Audit:
1. AASIST model output index alignment (empirical separation on real audio).
2. RiskEngine escalation and reason causality (no contradictory states).
3. ASR hallucination suppression on low-energy speech (no "I'm sorry").
4. CaptionEvent and DetectionEvent runtime lineage completeness.
5. Fail-safe behavior when models are offline (no fake SAFE decisions).
"""

import os
import pytest
import torch
import torchaudio

from app.schemas.analysis import RiskLevel, Decision, SpoofResult
from app.services.risk.risk_engine import RiskEngine
from app.services.antispoof.spectra_detector import spectra_detector
from app.services.asr.hf_whisper_provider import HFWhisperProvider
from app.services.monitor.caption_pipeline import CaptionEvent


def test_aasist_class_index_mapping_empirical():
    """Verify that SpectraAASIST maps index 0 to spoof and index 1 to bonafide.
    
    Empirical ground truth:
    - Real human speech (/tmp/speechbrain_example.wav) must yield bonafide > spoof.
    - AI-generated speech (ml/sample_audio/real_tts_16k.wav) must yield spoof > bonafide.
    """
    if not spectra_detector.is_loaded:
        loaded = spectra_detector.load_model()
        if not loaded:
            pytest.skip("SpectraAASIST model checkpoint could not be loaded")

    import soundfile as sf

    human_wav = "/tmp/speechbrain_example.wav"
    synthetic_wav = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/sample_audio/real_tts_16k.wav"))

    assert os.path.exists(human_wav), f"Missing human test fixture: {human_wav}"
    assert os.path.exists(synthetic_wav), f"Missing synthetic test fixture: {synthetic_wav}"

    data_h, sr_h = sf.read(human_wav)
    data_s, sr_s = sf.read(synthetic_wav)
    human_audio = torch.from_numpy(data_h).float()
    synth_audio = torch.from_numpy(data_s).float()
    if human_audio.ndim > 1:
        human_audio = human_audio[:, 0]
    if synth_audio.ndim > 1:
        synth_audio = synth_audio[:, 0]

    # 1. Test Human speech
    human_window = human_audio[:64600]
    res_human = spectra_detector.predict(human_window)

    print(f"\n[FORENSIC EMPIRICAL] Human: logit_bonafide={res_human.raw_logit_bonafide:.4f}, "
          f"logit_spoof={res_human.raw_logit_spoof:.4f}, spoof_score={res_human.spoof_score:.4f}")

    assert res_human.raw_logit_bonafide > res_human.raw_logit_spoof, \
        f"Human speech inverted! Bonafide {res_human.raw_logit_bonafide} <= Spoof {res_human.raw_logit_spoof}"
    assert res_human.spoof_score < 0.20, \
        f"Human speech spoof score unexpectedly high: {res_human.spoof_score}"
    assert res_human.label == "bonafide"

    # 2. Test AI synthetic speech
    synth_window = synth_audio[:64600]
    res_synth = spectra_detector.predict(synth_window)

    print(f"[FORENSIC EMPIRICAL] Synthetic: logit_bonafide={res_synth.raw_logit_bonafide:.4f}, "
          f"logit_spoof={res_synth.raw_logit_spoof:.4f}, spoof_score={res_synth.spoof_score:.4f}")

    assert res_synth.raw_logit_spoof > res_synth.raw_logit_bonafide, \
        f"Synthetic speech inverted! Spoof {res_synth.raw_logit_spoof} <= Bonafide {res_synth.raw_logit_bonafide}"
    assert res_synth.spoof_score > 0.80, \
        f"Synthetic speech spoof score unexpectedly low: {res_synth.spoof_score}"
    assert res_synth.label == "spoof"


def test_risk_engine_escalation_invariants():
    """Verify that RiskEngine correctly escalates and cannot hold contradictory states."""
    engine = RiskEngine()

    high_spoof = SpoofResult(
        model="Spectra-AASIST",
        label="spoof",
        spoof_score=0.98,
        bonafide_score=0.02,
        raw_logit_spoof=2.5,
        raw_logit_bonafide=-2.5,
    )

    # Invariant 1: Escalation from LOW to CRITICAL is mathematically possible
    # (Previously broken by string comparison 'CRITICAL' > 'LOW' which evaluates to False!)
    assessment = None
    for _ in range(5):
        assessment = engine.assess_live_window(high_spoof)

    # After sustained spoof windows, decision must escalate to BLOCK/CRITICAL
    assert assessment is not None
    assert assessment.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL), \
        f"Expected HIGH or CRITICAL after 5 high-spoof windows, got {assessment.risk_level}"
    assert assessment.decision in (Decision.VERIFY, Decision.BLOCK), \
        f"Expected VERIFY or BLOCK decision, got {assessment.decision}"

    # Invariant 2: 'Sustained alert' text must NEVER accompany Decision.SAFE
    assert not (assessment.decision == Decision.SAFE and "Sustained alert" in assessment.reason), \
        "Contradiction: 'Sustained alert' emitted while decision is SAFE"


def test_risk_engine_single_spike_no_false_block():
    """One single anomalous window must transition from SAFE to MONITOR, never jump directly to BLOCK."""
    engine = RiskEngine()

    # Normal window
    normal_spoof = SpoofResult(
        model="Spectra-AASIST",
        label="bonafide",
        spoof_score=0.03,
        bonafide_score=0.97,
        raw_logit_spoof=-2.5,
        raw_logit_bonafide=2.5,
    )
    a1 = engine.assess_live_window(normal_spoof)
    assert a1.decision == Decision.SAFE

    # Single spike
    spike_spoof = SpoofResult(
        model="Spectra-AASIST",
        label="spoof",
        spoof_score=0.95,
        bonafide_score=0.05,
        raw_logit_spoof=2.5,
        raw_logit_bonafide=-2.5,
    )
    a2 = engine.assess_live_window(spike_spoof)
    # Debouncing prevents premature jump to BLOCK; but it MUST NOT say SAFE when candidate is elevated!
    assert a2.decision == Decision.MONITOR, f"Expected MONITOR during debouncing, got {a2.decision}"
    assert "Potential synthetic voice anomaly" in a2.reason or "Evaluating potential threat" in a2.reason or "elevated window" in a2.reason


def test_risk_engine_offline_model_safe_abstention():
    """When antispoof model is unavailable, assess_aggregate must NOT return SAFE."""
    engine = RiskEngine()

    unavailable_results = [
        SpoofResult(
            model="Spectra-AASIST",
            label="unavailable",
            spoof_score=0.0,
            bonafide_score=0.0,
            raw_logit_spoof=0.0,
            raw_logit_bonafide=0.0,
        )
    ]

    agg = engine.assess_aggregate(unavailable_results)
    assert agg.decision != Decision.SAFE, \
        f"Critical fail-open flaw: Model unavailable declared SAFE! Decision={agg.decision}"
    assert agg.decision == Decision.MONITOR
    assert agg.risk_level == RiskLevel.MEDIUM
    assert "unavailable" in agg.reason.lower()


def test_hf_whisper_hallucination_rejection():
    """Verify that HFWhisperProvider rejects low-confidence decoder loops (like 'I\\'m sorry')."""
    provider = HFWhisperProvider(model_name="base", log_prob_threshold=-1.0)
    loaded = provider.load_model()
    if not loaded:
        pytest.skip("Whisper model could not be loaded on this environment")

    # Feed low-energy noise (RMS ~0.001)
    noise = torch.randn(16000 * 2) * 0.001
    result = provider.transcribe_segment(noise, sample_offset=0)

    # Must be empty or cleanly rejected, NEVER "I'm sorry"
    assert "sorry" not in result.text.lower(), \
        f"Whisper hallucination detected on low-energy audio: '{result.text}'"


def test_caption_event_lineage_fields():
    """Verify that CaptionEvent contains all lineage fields."""
    event = CaptionEvent(
        type="caption",
        text="Hello world",
        sample_start=0,
        sample_end=16000,
        caption_start=0,
        caption_end=16000,
        latency_ms=120.5,
        model_name="hf-whisper-base-ctranslate2",
        asr_provider="HFWhisperProvider",
        asr_model="base",
        asr_backend="ctranslate2",
    )

    assert event.asr_provider == "HFWhisperProvider"
    assert event.asr_model == "base"
    assert event.asr_backend == "ctranslate2"
    assert event.caption_start == 0
    assert event.caption_end == 16000
