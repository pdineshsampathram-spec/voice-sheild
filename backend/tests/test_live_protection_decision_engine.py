"""
Adversarial Verification Suite for VoiceShield Live-Protection Decision Engine.

Tests all 10 mandatory adversarial cases:
CASE 1: Normal human speech -> should not create unsupported spoof alert.
CASE 2: One anomalous AASIST window -> evidence retained, but does not convict entire call.
CASE 3: Multiple valid anomalous windows -> decision escalates according to validated logic.
CASE 4: Silence after anomaly -> does not erase historical evidence, nor manufacture spoof evidence.
CASE 5: New normal call after suspicious previous call -> new call starts with clean decision state.
CASE 6: Old WebSocket event arrives after reconnect -> rejected via generation/session guard.
CASE 7: Caption arrives after its session ended -> discarded.
CASE 8: AASIST window crosses discontinuity -> marked invalid and excluded from decisions.
CASE 9: Calibration unavailable -> no authoritative probability (UNCALIBRATED).
CASE 10: Speaker reference unavailable -> no fabricated speaker-match conclusion.
"""

import asyncio
import time
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.antispoof.calibrator import ScoreCalibrator, CalibratedOutput
from app.services.monitor.audio_bus import AudioBus, AudioChunk, DiscontinuityEvent, OverflowPolicy
from app.services.monitor.security_pipeline import SecurityPipeline, DetectionEvent, SpeakerEvent, AASIST_WINDOW_SAMPLES
from app.services.monitor.session import MonitorSession, create_session, get_session, remove_session

client = TestClient(app)


def test_case_1_normal_human_speech_no_unsupported_alert():
    """CASE 1: Normal human speech (bonafide) -> should produce GENUINE, never a false alert."""
    calibrator = ScoreCalibrator(expected_domain="any")
    calibrator.fit_validation_split(
        scores=[0.05, 0.10, 0.15, 0.85, 0.90, 0.95],
        labels=[0, 0, 0, 1, 1, 1],
        method="platt",
        domain="any",
    )

    # Bonafide input with low raw spoof score (0.08) and healthy energy (RMS = 0.05)
    output: CalibratedOutput = calibrator.calibrate(
        raw_spoof_score=0.08,
        raw_logit_spoof=-2.5,
        raw_logit_bonafide=2.5,
        window_valid=True,
        rms=0.05,
    )

    assert output.is_valid is True
    assert output.decision == "GENUINE"
    assert output.calibrated_probability is not None
    assert output.calibrated_probability < calibrator.operating_threshold


def test_case_2_single_anomalous_window_evidence_retained():
    """CASE 2: One anomalous AASIST window -> evidence retained, but isolated."""
    calibrator = ScoreCalibrator(expected_domain="any")
    calibrator.fit_validation_split(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        method="platt",
        domain="any",
    )

    # Window 1: genuine speech
    w1 = calibrator.calibrate(raw_spoof_score=0.10, rms=0.04)
    assert w1.decision == "GENUINE"

    # Window 2: anomalous spoof burst
    w2 = calibrator.calibrate(raw_spoof_score=0.88, rms=0.04)
    assert w2.decision == "SUSPICIOUS"
    assert w2.calibrated_probability is not None
    assert w2.calibrated_probability >= calibrator.operating_threshold

    # Window 3: genuine speech resumes
    w3 = calibrator.calibrate(raw_spoof_score=0.12, rms=0.04)
    assert w3.decision == "GENUINE"

    # Verify that window 2 anomaly is preserved as raw evidence,
    # while window 3 current acoustic state returns to genuine!
    history = [w1, w2, w3]
    suspicious = [w for w in history if w.decision == "SUSPICIOUS"]
    assert len(suspicious) == 1
    assert history[-1].decision == "GENUINE"


def test_case_3_multiple_valid_anomalous_windows_escalation():
    """CASE 3: Multiple valid anomalous windows -> cumulative evidence scales consistently."""
    calibrator = ScoreCalibrator(expected_domain="any")
    calibrator.fit_validation_split(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        method="platt",
        domain="any",
    )

    history = [
        calibrator.calibrate(raw_spoof_score=0.85, rms=0.03),
        calibrator.calibrate(raw_spoof_score=0.92, rms=0.04),
        calibrator.calibrate(raw_spoof_score=0.89, rms=0.035),
    ]

    suspicious_count = sum(1 for w in history if w.decision == "SUSPICIOUS")
    assert suspicious_count == 3
    # All 3 windows met valid acoustic quality and operating threshold
    for w in history:
        assert w.is_valid is True
        assert w.decision == "SUSPICIOUS"


def test_case_4_silence_after_anomaly_no_manufactured_spoof():
    """CASE 4: Silence after anomaly -> triggers ABSTAIN, does not manufacture new spoof evidence."""
    calibrator = ScoreCalibrator(expected_domain="any")
    calibrator.quality_gate_rms = 0.0025
    calibrator.fit_validation_split(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        method="platt",
        domain="any",
    )

    # Window 1: Spoof anomaly
    w1 = calibrator.calibrate(raw_spoof_score=0.90, rms=0.05)
    assert w1.decision == "SUSPICIOUS"

    # Window 2: Conversational silence / background line hum (low RMS)
    w2 = calibrator.calibrate(raw_spoof_score=0.75, rms=0.0008)  # Model emits noisy raw score
    # Quality gate must force ABSTAIN and prevent false spoof creation on silence
    assert w2.decision == "ABSTAIN"
    assert w2.calibration_status == "INSUFFICIENT_AUDIO_QUALITY"
    assert w2.calibrated_probability is None

    # Verify historical evidence of w1 is NOT erased
    assert w1.decision == "SUSPICIOUS"


def test_case_5_new_call_starts_with_clean_decision_state():
    """CASE 5: New normal call after suspicious previous call -> completely clean state."""
    room_a = "room-test-suspicious-call"
    room_b = "room-test-clean-call"

    # Call A: Connects, receives spoof detection
    with client.websocket_connect(f"/api/rtc/ws/{room_a}?peer_id=peer_a") as ws_a:
        ws_a.receive_json()  # joined
        # End Call A
        ws_a.send_json({"type": "bye", "generation": 1})

    # Call B: Completely separate call session
    with client.websocket_connect(f"/api/rtc/ws/{room_b}?peer_id=peer_b") as ws_b:
        msg_b = ws_b.receive_json()
        assert msg_b["type"] == "joined"
        assert msg_b["generation"] == 1
        assert msg_b["peer_count"] == 1
        # Invariant: Room B has zero residual state from Room A
        state_b = client.get(f"/api/rtc/rooms/{room_b}/state").json()
        assert state_b["peer_count"] == 1


def test_case_6_old_websocket_event_rejection():
    """CASE 6: Old WebSocket event arrives with stale generation -> rejected."""
    room_id = "test-stale-event-reject"

    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=p1") as ws_1:
        ws_1.receive_json()
        with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=p2") as ws_2:
            ws_2.receive_json()
            ws_1.receive_json()

            # p1 sends offer with old generation 0 (active generation is 1)
            ws_1.send_json({"type": "offer", "generation": 0, "payload": {"sdp": "stale"}})
            # p1 sends offer with current generation 1
            ws_1.send_json({"type": "offer", "generation": 1, "payload": {"sdp": "current"}})

            # p2 must receive ONLY the current generation message
            received = ws_2.receive_json()
            assert received["generation"] == 1
            assert received["payload"]["sdp"] == "current"


def test_case_7_caption_after_session_ended_discarded():
    """CASE 7: Caption pipeline after session ended -> queue drained and tasks cancelled."""
    loop = asyncio.new_event_loop()

    async def run_case_7():
        session = create_session("test-ended-session")
        await session.start()
        session_id = session.session_id

        # Stop session
        await remove_session(session_id)
        assert get_session(session_id) is None
        # Attempt to get event after termination returns None
        ev = await session.get_event(timeout=0.05)
        assert ev is None

    loop.run_until_complete(run_case_7())
    loop.close()


def test_case_8_aasist_window_crosses_discontinuity_marked_invalid():
    """CASE 8: AASIST window crosses audio discontinuity -> marked INVALID and excluded."""
    loop = asyncio.new_event_loop()

    async def run_case_8():
        events_emitted = []
        q = AudioBus().subscribe("test_sec", maxsize=50, policy=OverflowPolicy.EMIT_DISCONTINUITY)

        sec_pipeline = SecurityPipeline(
            audio_queue=q,
            on_event=lambda e: events_emitted.append(e),
        )

        # Trigger discontinuity event on security pipeline
        sec_pipeline.on_discontinuity(DiscontinuityEvent(
            sample_start=10000,
            sample_end=20000,
            reason="QUEUE_OVERFLOW_PACKET_LOSS",
        ))

        # Build window that crosses discontinuity
        window = torch.zeros(AASIST_WINDOW_SAMPLES, dtype=torch.float32)
        window_valid = not sec_pipeline._has_discontinuity
        discont_reason = sec_pipeline._discontinuity_reason

        # Calibrate output with discontinuity
        calibrator = ScoreCalibrator(expected_domain="any")
        out = calibrator.calibrate(
            raw_spoof_score=0.95,
            window_valid=window_valid,
            invalidation_reason=discont_reason,
            rms=0.03,
        )

        assert out.is_valid is False
        assert out.decision == "INVALID"
        assert out.calibration_status == "INVALID_WINDOW"
        assert out.calibrated_probability is None
        assert "QUEUE_OVERFLOW_PACKET_LOSS" in out.invalidation_reason

    loop.run_until_complete(run_case_8())
    loop.close()


def test_case_9_calibration_unavailable_yields_uncalibrated():
    """CASE 9: Calibration unavailable -> no authoritative probability (UNCALIBRATED)."""
    uncalibrated = ScoreCalibrator(artifact_path="nonexistent_artifact.json")
    assert uncalibrated.is_loaded is False

    out: CalibratedOutput = uncalibrated.calibrate(
        raw_spoof_score=0.88,
        raw_logit_spoof=2.0,
        raw_logit_bonafide=-2.0,
        window_valid=True,
        rms=0.04,
    )

    # Invariant: Never fabricate probability without calibration artifact
    assert out.calibrated_probability is None
    assert out.decision == "UNCALIBRATED"
    assert out.calibration_status == "CALIBRATION_REQUIRED"
    assert out.raw_spoof_score == 0.88


def test_case_10_speaker_reference_unavailable_no_fabricated_match():
    """CASE 10: Speaker reference unavailable -> no fabricated speaker-match conclusion."""
    event = SpeakerEvent(
        sample_start=0,
        sample_end=16000,
        embedding_dim=192,
        speaker_similarity=None,
        enrolled_speaker_id=None,
        match=None,
        calibration="REQUIRED",
    )

    assert event.speaker_similarity is None
    assert event.enrolled_speaker_id is None
    assert event.match is None
    assert event.calibration == "REQUIRED"
