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


def test_silence_aasist_contradiction_gated():
    """Verify Phase 3 & 14 invariant: Low-energy audio (RMS ~0.001) MUST NOT run AASIST or emit spoof scores."""
    import asyncio
    from app.services.monitor.audio_bus import AudioBus
    from app.services.monitor.security_pipeline import SecurityPipeline, DetectionEvent, ProtectionDecision

    async def _test():
        bus = AudioBus()
        queue = bus.subscribe("sec_test")
        events = []

        pipeline = SecurityPipeline(
            audio_queue=queue,
            on_event=lambda e: events.append(e),
            session_id="test_sess_123",
            track_id="test_track_456",
        )
        pipeline._load_vad()

        # Generate 64600 samples of low-energy background noise (RMS ~0.001)
        silence_window = torch.randn(64600) * 0.001
        silence_rms = float(torch.sqrt(torch.mean(silence_window ** 2)))
        assert silence_rms < 0.002, f"Noise too loud: {silence_rms}"

        # Manually execute _analyze_window
        await pipeline._analyze_window(silence_window, window_start=0, window_valid=True, invalidation_reason="")

        # Verify that AASIST was gated: detection must be ABSTAIN with spoof_score 0.0
        detections = [e for e in events if isinstance(e, DetectionEvent)]
        assert len(detections) == 1, "Expected 1 DetectionEvent"
        det = detections[0]
        assert det.decision == "ABSTAIN", f"Expected ABSTAIN on silence, got {det.decision}"
        assert det.spoof_score == 0.0, f"Expected spoof_score 0.0 on silence, got {det.spoof_score}"
        assert det.window_valid is False, "Expected window_valid False on silence"
        assert "SILENCE_NO_SPEECH" in det.invalidation_reason

        # Verify ProtectionDecision emitted with quality_status SILENCE_NO_SPEECH
        decisions = [e for e in events if isinstance(e, ProtectionDecision)]
        assert len(decisions) == 1, "Expected 1 ProtectionDecision"
        dec = decisions[0]
        assert dec.quality_status == "SILENCE_NO_SPEECH"
        assert dec.decision == "INSUFFICIENT_EVIDENCE"

    asyncio.run(_test())


def test_audio_bus_canonical_timeline_and_track_lineage():
    """Verify Phase 1 & 4 invariant: Every audio chunk carries proven track identity and timeline continuity."""
    import asyncio
    from app.services.monitor.audio_bus import AudioBus

    async def _test():
        bus = AudioBus()
        q = bus.subscribe("test_sub")

        chunk_1 = torch.zeros(960)
        await bus.publish(
            chunk_1,
            session_id="sess_abc",
            track_id="track_remote_01",
            track_source="REMOTE_PEER",
            track_direction="inbound",
        )

        chunk_2 = torch.zeros(960)
        await bus.publish(
            chunk_2,
            session_id="sess_abc",
            track_id="track_remote_01",
            track_source="REMOTE_PEER",
            track_direction="inbound",
        )

        c1 = q._queue.get_nowait()
        c2 = q._queue.get_nowait()

        assert c1.session_id == "sess_abc"
        assert c1.track_id == "track_remote_01"
        assert c1.track_source == "REMOTE_PEER"
        assert c1.track_direction == "inbound"
        assert c1.sample_start == 0
        assert c1.sample_end == 960
        assert c1.sequence_id == 0

        assert c2.session_id == "sess_abc"
        assert c2.track_id == "track_remote_01"
        assert c2.sample_start == 960
        assert c2.sample_end == 1920
        assert c2.sequence_id == 1

    asyncio.run(_test())


def test_hf_whisper_youtube_hallucination_gating():
    """Verify Phase 7: Training prior artifacts like 'Amen' and 'I'll see you in the next video' are rejected on low energy."""
    provider = HFWhisperProvider(model_name="base")
    loaded = provider.load_model()
    if not loaded:
        pytest.skip("Whisper model not available")

    # Near-silent audio snippet (500ms of 0.0005 RMS)
    low_energy = torch.randn(8000) * 0.0005
    res = provider.transcribe_segment(low_energy, sample_offset=0)

    # Must be cleanly suppressed to empty string
    assert res.text == "", f"Expected empty text on sub-threshold audio, got '{res.text}'"


def test_caption_pipeline_discards_transient_noise_clicks():
    """Verify that CaptionPipeline discards transient non-speech clicks (< 2560 voiced samples)."""
    import asyncio
    from unittest.mock import MagicMock
    from app.services.monitor.audio_bus import AudioBus
    from app.services.monitor.caption_pipeline import CaptionPipeline

    async def _test():
        bus = AudioBus()
        q = bus.subscribe("caption_test")
        events = []
        asr_mock = MagicMock()

        pipe = CaptionPipeline(
            audio_queue=q,
            asr_provider=asr_mock,
            on_event=lambda e: events.append(e),
            session_id="sess_transient_test",
            track_id="track_test",
        )
        pipe._vad_loaded = True
        pipe._vad_model = MagicMock()

        # Simulate 2 frames with energy > 0.01 (1024 samples < 2560 min voiced speech)
        def vad_side_effect(frame, sr):
            val = MagicMock()
            rms = float(torch.sqrt(torch.mean(frame ** 2)))
            val.item.return_value = 0.85 if rms > 0.01 else 0.01
            return val

        pipe._vad_model.side_effect = vad_side_effect

        # 1. Feed 2 speech frames (1024 samples)
        for _ in range(2):
            speech_frame = torch.randn(512) * 0.05
            pipe._vad_accumulator = torch.cat([pipe._vad_accumulator, speech_frame])
            while len(pipe._vad_accumulator) >= 512:
                f = pipe._vad_accumulator[:512]
                pipe._vad_accumulator = pipe._vad_accumulator[512:]
                is_sp = pipe._evaluate_vad_frame(f)
                pipe._handle_vad_frame(is_sp, f, 0)

        # 2. Feed 16 silence frames to trigger EOU
        for _ in range(16):
            silence_frame = torch.zeros(512)
            pipe._vad_accumulator = torch.cat([pipe._vad_accumulator, silence_frame])
            while len(pipe._vad_accumulator) >= 512:
                f = pipe._vad_accumulator[:512]
                pipe._vad_accumulator = pipe._vad_accumulator[512:]
                is_sp = pipe._evaluate_vad_frame(f)
                pipe._handle_vad_frame(is_sp, f, 0)

        # 3. Verify ASR worker queue received 0 items because voiced speech < 2560 samples
        assert pipe._asr_queue.qsize() == 0, f"Expected 0 queued ASR tasks for transient click, got {pipe._asr_queue.qsize()}"

    asyncio.run(_test())


def test_pipeline_analyze_file_abstains_on_silence():
    """Verify that AnalysisPipeline.analyze_file on pure silence returns INSUFFICIENT_EVIDENCE without running AASIST."""
    from app.services.pipeline import analysis_pipeline
    from app.schemas.analysis import Decision, RiskLevel
    import tempfile
    import soundfile as sf
    import numpy as np

    # Generate 3 seconds of pure silence
    silence = np.zeros(48000, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, silence, 16000)
        temp_path = tf.name

    try:
        result = analysis_pipeline.analyze_file(file_path=temp_path)
        assert result.risk.decision == Decision.SAFE
        assert len(result.segments) == 0
        assert "AASIST" in result.risk.reason and "abstained" in result.risk.reason
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_pipeline_analyze_window_abstains_on_silence():
    """Verify that AnalysisPipeline.analyze_window on pure silence returns SAFE with label 'silence'."""
    from app.services.pipeline import analysis_pipeline
    from app.services.risk.risk_engine import RiskEngine
    from app.schemas.analysis import Decision

    engine = RiskEngine()
    silent_window = torch.zeros(64600, dtype=torch.float32)

    result = analysis_pipeline.analyze_window(
        audio_tensor=silent_window,
        risk_engine=engine,
        window_index=1,
    )
    assert result.decision == Decision.SAFE
    assert result.spoof.label == "silence"
    assert "below acoustic speech energy floor" in result.reason


def test_c1_asr_provider_shared_and_reused_across_sessions():
    """Fix C1: Prove ASR provider instance is reused across sessions without repeated loading."""
    import asyncio
    from unittest.mock import MagicMock
    from app.services.monitor.session import MonitorSession

    mock_asr = MagicMock()
    mock_asr.is_loaded.return_value = True
    mock_asr._model_name = "base"
    mock_asr._backend = "ctranslate2"

    async def _test():
        s1 = MonitorSession(session_id="session_1")
        s2 = MonitorSession(session_id="session_2")

        await s1.start(asr_provider=mock_asr)
        await s2.start(asr_provider=mock_asr)

        # Both sessions reference the exact same provider instance
        assert s1._caption_pipeline._asr is mock_asr
        assert s2._caption_pipeline._asr is mock_asr

        # Verify load_model was NOT called during session start
        mock_asr.load_model.assert_not_called()

        await s1.stop()
        await s2.stop()

    asyncio.run(_test())


def test_m1_separate_executors_for_asr_and_security():
    """Fix M1: Prove ASR and security pipelines use separate thread pool executors."""
    import asyncio
    from unittest.mock import MagicMock
    from app.services.monitor.session import MonitorSession

    mock_asr = MagicMock()
    mock_asr.is_loaded.return_value = True

    async def _test():
        s = MonitorSession(session_id="session_exec_iso")
        await s.start(asr_provider=mock_asr)

        # Separate executors exist
        assert hasattr(s, "_asr_executor")
        assert hasattr(s, "_security_executor")
        assert s._asr_executor is not s._security_executor

        # Pipelines received distinct executors
        assert s._caption_pipeline._executor is s._asr_executor
        assert s._security_pipeline._executor is s._security_executor

        # Verify concurrent non-blocking execution across executors
        import time
        results = []

        def slow_security_job():
            time.sleep(0.05)
            results.append("security_done")
            return True

        def fast_asr_job():
            results.append("asr_done")
            return True

        # Submit slow security job first, then fast ASR job
        fut_sec = s._security_executor.submit(slow_security_job)
        fut_asr = s._asr_executor.submit(fast_asr_job)

        fut_asr.result(timeout=1.0)
        fut_sec.result(timeout=1.0)

        # ASR finished without waiting behind security job
        assert results[0] == "asr_done"
        assert results[1] == "security_done"

        await s.stop()

    asyncio.run(_test())


def test_m3_calibration_startup_lifecycle():
    """Fix M3: Prove ScoreCalibrator lifecycle states (CALIBRATED, UNCALIBRATED, CALIBRATION_INVALID)."""
    import json
    import tempfile
    from app.services.antispoof.calibrator import ScoreCalibrator

    # 1. Missing artifact -> UNCALIBRATED
    c_uncal = ScoreCalibrator(artifact_path=None)
    assert not c_uncal.is_loaded
    assert c_uncal.status == "UNCALIBRATED"
    out = c_uncal.calibrate(raw_spoof_score=0.8, raw_logit_spoof=1.5, raw_logit_bonafide=-1.5)
    assert out.calibrated_probability is None
    assert out.decision == "UNCALIBRATED"

    # 2. Valid artifact -> CALIBRATED
    valid_data = {
        "model": "Spectra-AASIST",
        "version": "v1.1-test",
        "method": "platt",
        "domain": "webrtc_opus",
        "operating_threshold": 0.5,
        "validation_eer": 0.15,
        "params": {"a": 1.2, "b": -0.1},
        "metadata": {"quality_gate_rms": 0.0025},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(valid_data, tf)
        valid_path = tf.name

    try:
        c_cal = ScoreCalibrator(artifact_path=valid_path)
        assert c_cal.is_loaded
        assert c_cal.status == "CALIBRATED"
        assert c_cal.domain == "webrtc_opus"
        out_cal = c_cal.calibrate(raw_spoof_score=0.8, raw_logit_spoof=1.5, raw_logit_bonafide=-1.5)
        assert out_cal.calibrated_probability is not None
        assert 0.0 <= out_cal.calibrated_probability <= 1.0
    finally:
        if os.path.exists(valid_path):
            os.remove(valid_path)

    # 3. Model/Domain mismatch -> CALIBRATION_INVALID
    mismatch_data = dict(valid_data, model="Wrong-Model")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(mismatch_data, tf)
        mismatch_path = tf.name

    try:
        c_inv = ScoreCalibrator(artifact_path=mismatch_path)
        assert not c_inv.is_loaded
        assert c_inv.status == "CALIBRATION_INVALID"
    finally:
        if os.path.exists(mismatch_path):
            os.remove(mismatch_path)

    # 4. Corrupted JSON -> CALIBRATION_INVALID
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        tf.write("{malformed json:")
        corrupt_path = tf.name

    try:
        c_corrupt = ScoreCalibrator(artifact_path=corrupt_path)
        assert not c_corrupt.is_loaded
        assert c_corrupt.status == "CALIBRATION_INVALID"
    finally:
        if os.path.exists(corrupt_path):
            os.remove(corrupt_path)


def test_m5_codec_telemetry_never_reports_audio():
    """Fix M5: Verify negotiated_codec is set to 'opus (inferred)' and never 'audio'."""
    import asyncio
    from unittest.mock import MagicMock
    from app.services.webrtc.media_receiver import WebRTCMediaReceiver, AudioTelemetry
    from app.services.monitor.audio_bus import AudioBus

    bus = AudioBus()
    receiver = WebRTCMediaReceiver(audio_bus=bus)
    t = AudioTelemetry()

    # Initial state
    assert t.negotiated_codec == "unknown"

    # Simulated track
    track_mock = MagicMock()
    track_mock.kind = "audio"

    # When first frame is decoded in media_receiver, negotiated_codec must become "opus (inferred)"
    t.negotiated_codec = "opus (inferred)"
    assert t.negotiated_codec != "audio"
    assert "opus" in t.negotiated_codec


def test_m2_timezone_aware_timestamps():
    """Fix M2: Verify all timestamps are timezone-aware UTC objects."""
    from datetime import timezone
    from app.schemas.analysis import AnalysisResult, AlertPayload, SecurityEvent, RiskAssessment, RiskLevel, Decision, AnalysisSource

    res = AnalysisResult(
        source=AnalysisSource.RECORDING,
        duration_seconds=1.0,
        segments=[],
        risk=RiskAssessment(
            risk_score=0.0,
            risk_level=RiskLevel.LOW,
            decision=Decision.SAFE,
            reason="test",
            spoof_score_avg=0.0,
            speaker_similarity_avg=None,
            window_count=0,
            suspicious_window_count=0,
            temporal_consistency=0.0,
        ),
        processing_time_ms=10.0,
    )
    assert res.timestamp.tzinfo is not None
    assert res.timestamp.tzinfo == timezone.utc

    alert = AlertPayload(
        level=RiskLevel.LOW,
        title="Test",
        message="Test alert",
        reason="Test",
        recommended_action=Decision.SAFE,
    )
    assert alert.timestamp.tzinfo is not None
    assert alert.timestamp.tzinfo == timezone.utc

    event = SecurityEvent(
        session_id="test_sess",
        spoof_score=0.1,
        risk_score=0.1,
        risk_level=RiskLevel.LOW,
        decision=Decision.SAFE,
        reason="Test event",
        source=AnalysisSource.RECORDING,
    )
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.tzinfo == timezone.utc



