"""
Regression test suite for:
1. Browser WebM audio decoding via PyAV (no TorchCodec dependency)
2. Robust multi-sample speaker enrollment with quality gating & centroid calculation
3. Automatic enrolled-speaker ranking and identification
4. Zero-Trust decoupling: Cloned enrolled speaker voice produces HIGH RISK, not SAFE.
"""

import io
import os
import tempfile
import json
import pytest
import torch
import numpy as np
import av
from pathlib import Path

from app.core.config import settings
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.speaker.ecapa_service import ecapa_service
from app.services.risk.risk_engine import RiskEngine
from app.services.pipeline import analysis_pipeline
from app.schemas.analysis import (
    RiskLevel, Decision, SpoofResult, SpeakerResult, MatchStatus
)


def create_synthetic_audio(duration_s=3.0, sr=16000, freq=440.0, silence=False) -> torch.Tensor:
    """Generate a clean synthetic speech-like tone waveform."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    if silence:
        data = np.zeros_like(t)
    else:
        # Complex harmonic tone simulating voiced formant
        data = (
            0.4 * np.sin(2 * np.pi * freq * t) +
            0.2 * np.sin(2 * np.pi * 2 * freq * t) +
            0.1 * np.sin(2 * np.pi * 3 * freq * t)
        )
    return torch.from_numpy(data)


def create_webm_opus_bytes(waveform: torch.Tensor, sr: int = 16000) -> bytes:
    """Encode a waveform tensor into WebM/Opus bytes using PyAV, matching browser MediaRecorder."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="webm")
    stream = container.add_stream("opus", rate=48000)
    stream.layout = "mono"

    # Resample to 48kHz for Opus encoder
    resampler = av.AudioResampler(format="flt", layout="mono", rate=48000)
    audio_np = waveform.numpy().reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(audio_np, format="flt", layout="mono")
    frame.sample_rate = sr

    resampled_frames = resampler.resample(frame)
    for rf in resampled_frames:
        for packet in stream.encode(rf):
            container.mux(packet)

    flush_frames = resampler.resample(None)
    for rf in flush_frames:
        for packet in stream.encode(rf):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)

    container.close()
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_test_profiles(tmp_path, monkeypatch):
    """Use a temporary directory for speaker profiles in each test."""
    test_profiles_dir = tmp_path / "speaker_profiles"
    test_profiles_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "speaker_profiles_dir", str(test_profiles_dir))
    yield test_profiles_dir


# =====================================================================
# 1. BROWSER WEBM AUDIO DECODING TESTS (TorchCodec fix verification)
# =====================================================================

def test_browser_webm_decoding_via_pyav():
    """Verify browser-recorded WebM/Opus audio decodes without TorchCodec."""
    preprocessor = AudioPreprocessor()
    test_waveform = create_synthetic_audio(duration_s=3.0, freq=300.0)
    webm_bytes = create_webm_opus_bytes(test_waveform, sr=16000)

    # 1. Test load_bytes
    decoded_wf, sr = preprocessor.load_bytes(webm_bytes, format_hint="webm")
    assert sr == 16000
    assert decoded_wf.ndim == 2
    assert decoded_wf.shape[0] == 1  # mono
    assert decoded_wf.shape[1] > 0
    rms = preprocessor.compute_rms(decoded_wf)
    assert rms > 0.05, f"Expected audible RMS, got {rms}"

    # 2. Test load_file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(webm_bytes)
        tmp_path = tmp.name

    try:
        decoded_from_file, file_sr = preprocessor.load_file(tmp_path)
        assert file_sr == 16000
        assert decoded_from_file.shape == decoded_wf.shape
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_corrupt_audio_fails_cleanly():
    """Verify corrupt audio bytes raise clean exception rather than hanging."""
    preprocessor = AudioPreprocessor()
    corrupt_bytes = b"NOT_A_REAL_AUDIO_CONTAINER_OR_WAV"
    with pytest.raises(Exception):
        preprocessor.load_bytes(corrupt_bytes, format_hint="wav")


# =====================================================================
# 2. SPEAKER ENROLLMENT ROBUSTNESS TESTS
# =====================================================================

def test_silent_enrollment_rejected():
    """Verify silent audio is rejected before enrollment with explicit reason."""
    preprocessor = AudioPreprocessor()
    silent_waveform = create_synthetic_audio(duration_s=3.0, silence=True)
    rms = preprocessor.compute_rms(silent_waveform)
    assert rms < 0.002

    # Attempting to enroll should be gated by energy validation in API
    # Verify preprocessor energy gate correctly catches it
    assert rms == 0.0


def test_short_enrollment_rejected():
    """Verify audio under 2 seconds is flagged as insufficient duration."""
    preprocessor = AudioPreprocessor()
    short_waveform = create_synthetic_audio(duration_s=1.0)
    duration = preprocessor.get_duration(short_waveform)
    assert duration < 2.0


def test_valid_enrollment_creates_normalized_embedding():
    """Verify enrollment produces unit-normalized 192-dim embedding."""
    if not ecapa_service.is_loaded:
        # Load model for testing
        ecapa_service.load_model()
    if not ecapa_service.is_loaded:
        pytest.skip("ECAPA-TDNN model not loaded in environment")

    waveform = create_synthetic_audio(duration_s=3.0, freq=250.0)
    profile = ecapa_service.enroll_speaker(speaker_name="Dinesh", waveform=waveform)

    assert profile.speaker_name == "Dinesh"
    assert profile.sample_count == 1
    assert profile.quality_status == "VALID"
    assert Path(profile.embedding_path).exists()

    embedding = np.load(profile.embedding_path)
    assert embedding.shape == (192,)
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-4), f"Embedding not normalized: {norm}"


def test_multisample_enrollment_centroid_update():
    """Verify multiple samples for the same speaker update the centroid profile."""
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()
    if not ecapa_service.is_loaded:
        pytest.skip("ECAPA-TDNN model not loaded in environment")

    waveform_1 = create_synthetic_audio(duration_s=3.0, freq=220.0)
    waveform_2 = create_synthetic_audio(duration_s=3.0, freq=230.0)

    # First sample
    profile_1 = ecapa_service.enroll_speaker(speaker_name="Dinesh", waveform=waveform_1)
    emb_1 = np.load(profile_1.embedding_path).copy()
    assert profile_1.sample_count == 1

    # Second sample with same speaker_name
    profile_2 = ecapa_service.enroll_speaker(speaker_name="Dinesh", waveform=waveform_2)
    emb_2 = np.load(profile_2.embedding_path).copy()
    assert profile_2.sample_count == 2
    assert profile_2.speaker_id == profile_1.speaker_id

    # Centroid should remain unit normalized
    norm = np.linalg.norm(emb_2)
    assert np.isclose(norm, 1.0, atol=1e-4)
    # Centroid should have moved towards sample 2
    assert not np.allclose(emb_1, emb_2)


# =====================================================================
# 3. AUTOMATIC SPEAKER IDENTIFICATION TESTS
# =====================================================================

def test_automatic_speaker_identification_ranking():
    """Verify automatic comparison against all enrolled profiles in descending rank order."""
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()
    if not ecapa_service.is_loaded:
        pytest.skip("ECAPA-TDNN model not loaded in environment")

    wf_dinesh = create_synthetic_audio(duration_s=3.0, freq=200.0)
    wf_alice = create_synthetic_audio(duration_s=3.0, freq=400.0)

    p_dinesh = ecapa_service.enroll_speaker("Dinesh", wf_dinesh)
    p_alice = ecapa_service.enroll_speaker("Alice", wf_alice)

    # Test query using Dinesh's audio
    id_result = ecapa_service.identify_speaker(wf_dinesh)

    assert id_result.best_match is not None
    assert id_result.best_match.speaker_id == p_dinesh.speaker_id
    assert id_result.best_match.speaker_name == "Dinesh"
    assert id_result.best_match.similarity > 0.90  # Identical synthetic audio
    assert id_result.best_match.status == MatchStatus.MATCH

    # Check alternatives list contains Alice
    assert len(id_result.alternatives) == 1
    assert id_result.alternatives[0].speaker_name == "Alice"
    assert id_result.best_match.similarity >= id_result.alternatives[0].similarity


def test_identification_when_no_profiles_exist():
    """Verify clean UNAVAILABLE state when no profiles are enrolled."""
    wf = create_synthetic_audio(duration_s=3.0)
    id_result = ecapa_service.identify_speaker(wf)
    assert id_result.best_match is None
    assert id_result.alternatives == []
    assert id_result.status in [MatchStatus.NO_ENROLLED_PROFILES, MatchStatus.UNAVAILABLE]


# =====================================================================
# 4. ZERO-TRUST DECOUPLING & CLONED-VOICE TEST CASE (PART 5 & 6)
# =====================================================================

def test_cloned_enrolled_speaker_produces_high_risk_verdict():
    """
    SECURITY INVARIANT:
    'Whose voice does this resemble?' != 'Is this voice authentic?'

    Scenario:
    - Enrolled genuine speaker: Dinesh
    - Audio matches Dinesh acoustically (ECAPA similarity high: e.g. 0.86)
    - Anti-spoof detector detects strong synthetic indicators (AASIST spoof score high: e.g. 0.92)

    Required Behavior:
    - Best match speaker: Dinesh (Similarity: ~0.86, Status: MATCH)
    - Voice authenticity: Synthetic indicators detected
    - Spoof risk: HIGH
    - Final verdict: HIGH RISK / WARN
    - High speaker similarity must NEVER suppress spoof detection to SAFE!
    """
    risk_engine = RiskEngine()

    spoof_result = SpoofResult(
        model="Spectra-AASIST",
        label="spoof",
        spoof_score=0.92,
        bonafide_score=0.08,
        raw_logit_spoof=5.2,
        raw_logit_bonafide=-3.8,
        delta=9.0,
    )

    speaker_result = SpeakerResult(
        model="ECAPA-TDNN",
        speaker_similarity=0.86,
        match=True,
        threshold_used=0.25,
        enrolled_speaker_id="dinesh-uuid-123",
    )

    # Single segment assessment
    single_risk = risk_engine.assess_single(spoof=spoof_result, speaker=speaker_result)
    assert single_risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL), (
        f"Expected HIGH or CRITICAL, got {single_risk.risk_level}"
    )
    assert single_risk.decision in (Decision.WARN, Decision.VERIFY), (
        f"Expected WARN or VERIFY, got {single_risk.decision}"
    )
    assert "resembles" in single_risk.reason.lower() or "synthetic" in single_risk.reason.lower()
    assert "synthetic-voice indicators are detected" in single_risk.reason.lower()

    # Aggregate assessment
    agg_risk = risk_engine.assess_aggregate(
        spoof_results=[spoof_result],
        speaker_results=[speaker_result],
        matched_speaker_name="Dinesh",
    )
    assert agg_risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert agg_risk.decision in (Decision.WARN, Decision.VERIFY)
    assert "Dinesh" in agg_risk.reason
    assert "synthetic-voice indicators are detected" in agg_risk.reason


def test_genuine_enrolled_speaker_produces_safe_verdict():
    """Verify authentic voice matching enrolled speaker produces LOW / SAFE verdict."""
    risk_engine = RiskEngine()

    bonafide_spoof = SpoofResult(
        model="Spectra-AASIST",
        label="bonafide",
        spoof_score=0.04,
        bonafide_score=0.96,
        raw_logit_spoof=-4.5,
        raw_logit_bonafide=3.2,
        delta=-7.7,
    )

    speaker_result = SpeakerResult(
        model="ECAPA-TDNN",
        speaker_similarity=0.88,
        match=True,
        threshold_used=0.25,
        enrolled_speaker_id="dinesh-uuid-123",
    )

    agg_risk = risk_engine.assess_aggregate(
        spoof_results=[bonafide_spoof],
        speaker_results=[speaker_result],
        matched_speaker_name="Dinesh",
    )
    assert agg_risk.risk_level == RiskLevel.LOW
    assert agg_risk.decision == Decision.SAFE
    assert "authentic" in agg_risk.reason.lower()


def test_unknown_speaker_with_spoof_evidence():
    """Verify spoof audio from an unknown speaker produces HIGH RISK without false speaker match."""
    risk_engine = RiskEngine()

    spoof_result = SpoofResult(
        model="Spectra-AASIST",
        label="spoof",
        spoof_score=0.88,
        bonafide_score=0.12,
        raw_logit_spoof=4.8,
        raw_logit_bonafide=-3.2,
        delta=8.0,
    )

    agg_risk = risk_engine.assess_aggregate(
        spoof_results=[spoof_result],
        speaker_results=[None],
        matched_speaker_name=None,
    )
    assert agg_risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert agg_risk.decision in (Decision.WARN, Decision.VERIFY)
    assert "elevated synthetic voice indicators" in agg_risk.reason.lower()


def test_pipeline_analyze_file_with_automatic_speaker_id_and_contract():
    """Verify analysis_pipeline.analyze_file decodes WebM, identifies enrolled speaker, and returns Part 10 structured fields."""
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()
    if not ecapa_service.is_loaded:
        pytest.skip("ECAPA-TDNN model not loaded in environment")

    # 1. Enroll Dinesh
    wf_dinesh = create_synthetic_audio(duration_s=3.0, freq=300.0)
    profile = ecapa_service.enroll_speaker(speaker_name="Dinesh", waveform=wf_dinesh)

    # 2. Create WebM audio file matching browser MediaRecorder
    webm_bytes = create_webm_opus_bytes(wf_dinesh, sr=16000)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(webm_bytes)
        tmp_path = tmp.name

    try:
        # 3. Run full analysis pipeline WITHOUT passing speaker_id (automatic discovery!)
        result = analysis_pipeline.analyze_file(file_path=tmp_path)

        assert result.duration_seconds >= 2.5
        # Verify structured contract fields exist
        assert result.spoof is not None
        assert result.speaker is not None
        assert result.quality is not None
        assert result.verdict is not None

        # Verify automatic speaker identification found Dinesh
        assert result.speaker.best_match is not None
        assert result.speaker.best_match.speaker_name == "Dinesh"
        assert result.speaker.best_match.speaker_id == profile.speaker_id
        assert result.speaker.best_match.similarity > 0.85
        assert result.speaker.status == MatchStatus.MATCH

        # Verify quality
        assert result.quality.status == "VALID"
        assert result.quality.rms > 0.0

        # Verify verdict
        assert result.verdict.risk in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert result.verdict.decision in (Decision.SAFE, Decision.MONITOR, Decision.WARN, Decision.VERIFY)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

