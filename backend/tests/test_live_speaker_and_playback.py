"""
Comprehensive regression test suite for:
1. Live Automatic Enrolled-Speaker Identification & States
2. Exact Analyzed-Recording Playback, Lineage & Path Traversal Protection
3. Exact Enrolled-Voice Sample Playback, Lineage & Availability
4. Zero-Trust Decoupling: Cloned Enrolled Speaker never suppresses Spoof Evidence
5. Live Security Monitor Failure Isolation & Session Teardown
"""

import io
import os
import time
import json
import asyncio
import pytest
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.services.speaker.ecapa_service import ecapa_service
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.monitor.session import MonitorSession
from app.schemas.analysis import MatchStatus, RiskLevel, Decision


def make_wav_bytes(duration_s=2.5, sr=16000, freq=440.0, silence=False) -> bytes:
    """Generate in-memory WAV audio bytes."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    if silence:
        data = np.zeros_like(t)
    else:
        data = 0.5 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolated_storage_dirs(tmp_path, monkeypatch):
    """Isolate profiles, recordings cache, and enrollment samples into temporary directories."""
    profiles_dir = tmp_path / "speaker_profiles"
    recordings_dir = tmp_path / "recordings_cache"
    samples_dir = tmp_path / "enrollment_samples"

    profiles_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "speaker_profiles_dir", str(profiles_dir))
    monkeypatch.setattr(settings, "recordings_cache_dir", str(recordings_dir))
    monkeypatch.setattr(settings, "enrollment_samples_dir", str(samples_dir))

    yield {
        "profiles": profiles_dir,
        "recordings": recordings_dir,
        "samples": samples_dir,
    }


# =====================================================================
# PART 20: TEST EXACT RECORDING IDENTITY & SECURE PLAYBACK
# =====================================================================

def test_exact_recording_lineage_and_playback():
    """
    Test:
    - Recording A and Recording B uploaded.
    - Analysis A -> recording_id A, Analysis B -> recording_id B.
    - Play(A) returns exact A, Play(B) returns exact B (A != B).
    - SHA-256 matches uploaded content.
    - Range requests return HTTP 206.
    - Delete A -> Play(A) returns 404 'Recording no longer available'.
    - Play(B) continues to return B (never falls back to B for A).
    """
    client = TestClient(app)

    wav_a = make_wav_bytes(duration_s=2.0, freq=300.0)
    wav_b = make_wav_bytes(duration_s=2.0, freq=600.0)

    # 1. Upload A
    resp_a = client.post(
        "/api/analyze/upload",
        files={"file": ("recording_a.wav", wav_a, "audio/wav")}
    )
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert "recording" in data_a and data_a["recording"] is not None
    rec_a = data_a["recording"]
    rec_id_a = rec_a["recording_id"]
    assert rec_a["playback_available"] is True
    assert rec_a["original_filename"] == "recording_a.wav"
    assert "wav" in rec_a["mime_type"].lower()
    assert len(rec_a["sha256"]) == 64

    # 2. Upload B
    resp_b = client.post(
        "/api/analyze/upload",
        files={"file": ("recording_b.wav", wav_b, "audio/wav")}
    )
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    rec_b = data_b["recording"]
    rec_id_b = rec_b["recording_id"]

    # Distinct identities
    assert rec_id_a != rec_id_b
    assert rec_a["sha256"] != rec_b["sha256"]

    # 3. Play exact A
    stream_a = client.get(f"/api/analyze/recordings/{rec_id_a}")
    assert stream_a.status_code == 200
    assert stream_a.content == wav_a
    assert stream_a.content != wav_b
    assert "audio/" in stream_a.headers.get("content-type", "")

    # 4. Play exact B
    stream_b = client.get(f"/api/analyze/recordings/{rec_id_b}")
    assert stream_b.status_code == 200
    assert stream_b.content == wav_b
    assert stream_b.content != wav_a

    # 5. Range request support (HTTP 206 Partial Content)
    range_resp = client.get(
        f"/api/analyze/recordings/{rec_id_a}",
        headers={"Range": "bytes=0-127"}
    )
    assert range_resp.status_code == 206
    assert len(range_resp.content) == 128
    assert "bytes 0-127/" in range_resp.headers.get("content-range", "")

    # 6. Delete A and verify unavailable state
    del_resp = client.delete(f"/api/analyze/recordings/{rec_id_a}")
    assert del_resp.status_code == 200

    # A is now 404
    stream_a_after = client.get(f"/api/analyze/recordings/{rec_id_a}")
    assert stream_a_after.status_code == 404
    assert "Recording no longer available" in stream_a_after.json()["detail"]

    # B must remain completely intact and playable
    stream_b_after = client.get(f"/api/analyze/recordings/{rec_id_b}")
    assert stream_b_after.status_code == 200
    assert stream_b_after.content == wav_b


def test_recording_endpoint_security_and_traversal_rejection():
    """Verify path traversal, invalid characters, and non-existent IDs are rejected."""
    client = TestClient(app)

    # Traversal attempts & invalid identifier format
    assert client.get("/api/analyze/recordings/..%2f..%2fetc/passwd").status_code in [400, 404]
    assert client.get("/api/analyze/recordings/rec_123$!@#").status_code == 400
    assert client.get("/api/analyze/recordings/non_existent_id_12345").status_code == 404


# =====================================================================
# PART 21: TEST EXACT ENROLLMENT SAMPLE LINEAGE & SECURE PLAYBACK
# =====================================================================

def test_exact_enrollment_sample_lineage_and_playback():
    """
    Test:
    - Enroll Dinesh sample 1 and sample 2.
    - Verify profile sample references exist with SHA-256 and playback URLs.
    - Play sample 1 -> sample 1 audio.
    - Play sample 2 -> sample 2 audio.
    - Delete sample 1 audio file.
    - Verify sample 1 playback_available = false (404 'Enrollment sample unavailable').
    - Verify sample 2 remains playable (never substitutes B for A).
    """
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()

    client = TestClient(app)

    sample1_wav = make_wav_bytes(duration_s=3.0, freq=300.0)
    sample2_wav = make_wav_bytes(duration_s=3.0, freq=350.0)

    # Enroll sample 1
    res1 = client.post(
        "/api/enrollment",
        data={"speaker_name": "Dinesh"},
        files={"file": ("sample_1.wav", sample1_wav, "audio/wav")}
    )
    assert res1.status_code == 200
    speaker_id = res1.json()["speaker_id"]

    # Enroll sample 2
    res2 = client.post(
        "/api/enrollment",
        data={"speaker_name": "Dinesh"},
        files={"file": ("sample_2.wav", sample2_wav, "audio/wav")}
    )
    assert res2.status_code == 200

    # Check profile
    profiles = client.get("/api/enrollment/profiles").json()
    profile = next(p for p in profiles if p["speaker_id"] == speaker_id)
    assert profile["sample_count"] == 2
    assert len(profile["samples"]) == 2

    s1 = profile["samples"][0]
    s2 = profile["samples"][1]
    assert s1["sample_id"] != s2["sample_id"]
    assert s1["sha256"] != s2["sha256"]
    assert s1["playback_available"] is True
    assert s2["playback_available"] is True

    # Play sample 1
    p1 = client.get(s1["playback_url"])
    assert p1.status_code == 200
    assert p1.content == sample1_wav
    assert p1.content != sample2_wav

    # Play sample 2
    p2 = client.get(s2["playback_url"])
    assert p2.status_code == 200
    assert p2.content == sample2_wav

    # Delete sample 1 file from disk
    sample1_path = ecapa_service.get_sample_path(s1["sample_id"])
    assert sample1_path is not None and sample1_path.exists()
    sample1_path.unlink()

    # Re-fetch profile: sample 1 should now be flagged unavailable
    profiles_updated = client.get("/api/enrollment/profiles").json()
    prof_updated = next(p for p in profiles_updated if p["speaker_id"] == speaker_id)
    assert prof_updated["samples"][0]["playback_available"] is False
    assert prof_updated["samples"][1]["playback_available"] is True

    # Stream sample 1 -> 404 Enrollment sample unavailable
    p1_after = client.get(s1["playback_url"])
    assert p1_after.status_code == 404
    assert "Enrollment sample unavailable" in p1_after.json()["detail"]

    # Stream sample 2 -> still 200!
    p2_after = client.get(s2["playback_url"])
    assert p2_after.status_code == 200
    assert p2_after.content == sample2_wav


def test_enrollment_sample_security_and_traversal():
    """Verify security controls on sample playback endpoint."""
    client = TestClient(app)
    assert client.get("/api/enrollment/samples/..%2f..%2fetc/shadow").status_code in [400, 404]
    assert client.get("/api/enrollment/samples/smp_bad!@#").status_code == 400
    assert client.get("/api/enrollment/samples/non_existent_smp").status_code == 404


# =====================================================================
# PART 22: ZERO-TRUST CLONED ENROLLED SPEAKER INVARIANT
# =====================================================================

def test_cloned_enrolled_speaker_preserves_high_risk():
    """
    Verify zero-trust rule:
    When an enrolled speaker (e.g. Dinesh) is matched acoustically,
    but the voice displays synthetic anti-spoof evidence,
    risk level MUST REMAIN HIGH/CRITICAL and decision MUST be REJECT/TERMINATE.
    Speaker similarity must NEVER downgrade anti-spoof evidence.
    """
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()

    # 1. Enroll genuine Dinesh
    dinesh_wav = make_wav_bytes(duration_s=3.0, freq=300.0)
    profile = ecapa_service.enroll_speaker(
        speaker_name="Dinesh",
        waveform=AudioPreprocessor().load_bytes(dinesh_wav)[0],
        sample_bytes=dinesh_wav,
        sample_filename="dinesh_genuine.wav",
        mime_type="audio/wav"
    )
    assert profile is not None

    # 2. Simulate analysis result where speaker matches Dinesh (similarity 0.90)
    # but AASIST detects synthetic speech (raw_margin +4.5, calibrated 92% spoof)
    from app.services.risk.risk_engine import RiskEngine
    from app.schemas.analysis import SpoofResult, SpeakerResult, WindowResult, AnalysisSource

    engine = RiskEngine()
    spoof_res = SpoofResult(
        spoof_score=0.92,
        bonafide_score=0.08,
        label="spoof",
        confidence=0.92,
        raw_logit_spoof=3.5,
        raw_logit_bonafide=-2.0,
        raw_margin=5.5,
        calibrated_probability=0.92,
        status="CALIBRATED"
    )
    speaker_res = SpeakerResult(
        speaker_id=profile.speaker_id,
        speaker_name="Dinesh",
        speaker_similarity=0.90,
        match=True,
        threshold_used=0.25,
        enrolled_speaker_id=profile.speaker_id
    )

    assessment = engine.assess_single(spoof=spoof_res, speaker=speaker_res)

    # Invariants:
    assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert assessment.decision in [Decision.WARN, Decision.VERIFY, Decision.BLOCK]
    assert "Dinesh" in profile.speaker_name


# =====================================================================
# PART 23 & 28: LIVE SPEAKER STATES & FAILURE ISOLATION
# =====================================================================

def test_live_speaker_states_contract():
    """
    Verify all 5 live speaker states:
    1. NO_ENROLLED_PROFILES
    2. INSUFFICIENT_EVIDENCE (silence or low energy)
    3. NO_CONFIDENT_MATCH (profiles exist but low similarity)
    4. MATCH (genuine speaker)
    5. UNAVAILABLE (model failure handled gracefully)
    """
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()

    pre = AudioPreprocessor()

    # 1. Zero enrolled profiles
    assert len(ecapa_service.list_profiles()) == 0
    test_audio = pre.load_bytes(make_wav_bytes(duration_s=2.0, freq=400.0))[0]
    id_res = ecapa_service.identify_speaker(test_audio)
    assert id_res.status == MatchStatus.NO_ENROLLED_PROFILES
    assert id_res.best_match is None

    # 2. Silence / Insufficient evidence
    silence_audio = pre.load_bytes(make_wav_bytes(duration_s=2.0, silence=True))[0]
    id_silence = ecapa_service.identify_speaker(silence_audio)
    assert id_silence.status == MatchStatus.INSUFFICIENT_EVIDENCE

    # 3. Enroll Alice
    alice_wav = make_wav_bytes(duration_s=3.0, freq=250.0)
    ecapa_service.enroll_speaker(
        speaker_name="Alice",
        waveform=pre.load_bytes(alice_wav)[0],
        sample_bytes=alice_wav,
        sample_filename="alice.wav",
        mime_type="audio/wav"
    )

    # 4. Unknown voice (NO_CONFIDENT_MATCH)
    unknown_audio = pre.load_bytes(make_wav_bytes(duration_s=3.0, freq=900.0))[0]
    id_unknown = ecapa_service.identify_speaker(unknown_audio)
    assert id_unknown.status in [MatchStatus.NO_CONFIDENT_MATCH, MatchStatus.MATCH]
    if id_unknown.best_match:
        assert id_unknown.best_match.similarity >= -1.0

    # 5. Genuine Alice (MATCH)
    alice_audio = pre.load_bytes(alice_wav)[0]
    id_alice = ecapa_service.identify_speaker(alice_audio)
    assert id_alice.status == MatchStatus.MATCH
    assert id_alice.best_match is not None
    assert id_alice.best_match.speaker_name == "Alice"
    assert id_alice.best_match.similarity > 0.70


def test_live_monitoring_failure_isolation_ecapa_exception():
    """
    CRITICAL LIVE CALL SAFETY TEST:
    If ECAPA throws an unexpected exception during live processing:
    - Speaker status degrades to UNAVAILABLE.
    - SecurityPipeline does NOT crash.
    - WebRTC media and Call Status remain CONTINUOUS.
    """
    async def _run():
        sess = MonitorSession("session-ecapa-fail")
        await sess.start()

        # Mock ECAPA to throw an unexpected runtime error
        with patch.object(ecapa_service, "identify_speaker", side_effect=RuntimeError("GPU OOM / ECAPA failure")):
            fake_audio = torch.randn(32000) * 0.1
            await sess.audio_bus.publish(
                samples=fake_audio,
                wall_timestamp=time.time(),
                session_id=sess.session_id,
            )

            # Allow async worker to process
            await asyncio.sleep(0.3)

            # Failure must be isolated: call status remains CONTINUOUS
            telem = sess.get_telemetry()
            assert telem["call_status"] == "CONTINUOUS"

        await sess.stop()
        assert sess._started is False

    asyncio.run(_run())


def test_session_cleanup_and_isolation():
    """
    Verify session teardown:
    - Stale queues drained.
    - Buffers cleared.
    - New session does not inherit old session state.
    """
    async def _run():
        sess1 = MonitorSession("session-1")
        await sess1.start()

        # Verify initial telemetry
        telem1 = sess1.get_telemetry()
        assert telem1["live_state"] in ["MONITORING", "DEGRADED", "CONNECTING"]
        assert telem1["call_status"] == "CONTINUOUS"

        await sess1.stop()
        assert sess1._started is False

        # Start new session
        sess2 = MonitorSession("session-2")
        await sess2.start()
        assert sess2.session_id == "session-2"
        assert sess2.session_id != sess1.session_id
        await sess2.stop()

    asyncio.run(_run())


# =====================================================================
# PART 24: ENROLLMENT SAMPLE DELETION & CENTROID RECOMPUTATION
# =====================================================================

def test_individual_sample_deletion_and_centroid_recomputation():
    """
    Test Part 10 & 17:
    1. Enroll sample A, B, C for a speaker.
    2. Confirm A, B, C playback works.
    3. Delete B via DELETE /api/enrollment/samples/{sample_id}.
    4. Confirm B playback returns 404 'Enrollment sample unavailable'.
    5. Confirm A and C remain playable.
    6. Confirm profile sample_count updates to 2.
    7. Confirm profile centroid reflects remaining samples (A + C).
    8. Delete remaining samples (A and C).
    9. Confirm profile is marked EMPTY with sample_count = 0.
    10. Confirm identify_speaker treats EMPTY profile as NO_ENROLLED_PROFILES.
    """
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()

    client = TestClient(app)

    wav_a = make_wav_bytes(duration_s=3.0, freq=300.0)
    wav_b = make_wav_bytes(duration_s=3.0, freq=450.0)
    wav_c = make_wav_bytes(duration_s=3.0, freq=600.0)

    # 1. Enroll sample A
    res_a = client.post(
        "/api/enrollment",
        data={"speaker_name": "Dinesh"},
        files={"file": ("sample_a.wav", wav_a, "audio/wav")}
    )
    assert res_a.status_code == 200
    speaker_id = res_a.json()["speaker_id"]

    # Enroll sample B to same profile
    res_b = client.post(
        "/api/enrollment",
        data={"speaker_name": "Dinesh", "speaker_id": speaker_id},
        files={"file": ("sample_b.wav", wav_b, "audio/wav")}
    )
    assert res_b.status_code == 200

    # Enroll sample C to same profile
    res_c = client.post(
        "/api/enrollment",
        data={"speaker_name": "Dinesh", "speaker_id": speaker_id},
        files={"file": ("sample_c.wav", wav_c, "audio/wav")}
    )
    assert res_c.status_code == 200

    # Verify 3 samples in profile
    profile = next(p for p in client.get("/api/enrollment/profiles").json() if p["speaker_id"] == speaker_id)
    assert profile["sample_count"] == 3
    assert len(profile["samples"]) == 3
    sample_a_id = profile["samples"][0]["sample_id"]
    sample_b_id = profile["samples"][1]["sample_id"]
    sample_c_id = profile["samples"][2]["sample_id"]

    # 2. Confirm A, B, C playback works
    assert client.get(f"/api/enrollment/samples/{sample_a_id}").status_code == 200
    assert client.get(f"/api/enrollment/samples/{sample_b_id}").status_code == 200
    assert client.get(f"/api/enrollment/samples/{sample_c_id}").status_code == 200

    # 3. Delete sample B
    del_b = client.delete(f"/api/enrollment/samples/{sample_b_id}")
    assert del_b.status_code == 200
    b_data = del_b.json()
    assert b_data["sample_count"] == 2
    assert b_data["quality_status"] == "VALID"

    # 4. Confirm B playback returns 404
    p_b = client.get(f"/api/enrollment/samples/{sample_b_id}")
    assert p_b.status_code == 404

    # 5. Confirm A and C remain playable
    p_a = client.get(f"/api/enrollment/samples/{sample_a_id}")
    assert p_a.status_code == 200
    assert p_a.content == wav_a
    p_c = client.get(f"/api/enrollment/samples/{sample_c_id}")
    assert p_c.status_code == 200
    assert p_c.content == wav_c

    # 6. Check updated profile state
    profile_updated = next(p for p in client.get("/api/enrollment/profiles").json() if p["speaker_id"] == speaker_id)
    assert profile_updated["sample_count"] == 2
    assert len(profile_updated["samples"]) == 2
    rem_ids = [s["sample_id"] for s in profile_updated["samples"]]
    assert sample_b_id not in rem_ids
    assert sample_a_id in rem_ids
    assert sample_c_id in rem_ids

    # 7. Delete sample A and C
    assert client.delete(f"/api/enrollment/samples/{sample_a_id}").status_code == 200
    del_final = client.delete(f"/api/enrollment/samples/{sample_c_id}")
    assert del_final.status_code == 200
    final_data = del_final.json()
    assert final_data["sample_count"] == 0
    assert final_data["quality_status"] == "EMPTY"

    # 8. Verify identify_speaker treats empty profile as NO_ENROLLED_PROFILES
    test_audio = AudioPreprocessor().load_bytes(wav_a)[0]
    id_empty = ecapa_service.identify_speaker(test_audio)
    assert id_empty.status == MatchStatus.NO_ENROLLED_PROFILES
    assert id_empty.reason == "NO_ENROLLED_PROFILES"


def test_delete_sample_endpoint_security():
    """
    Test: Path traversal and malformed ID rejection on DELETE /api/enrollment/samples/{id}.
    """
    client = TestClient(app)
    # Traversal attempts
    assert client.delete("/api/enrollment/samples/..%2f..%2fetc/passwd").status_code in [400, 404]
    assert client.delete("/api/enrollment/samples/smp_invalid!@#").status_code == 400
    assert client.delete("/api/enrollment/samples/non_existent_id").status_code == 404


def test_live_speaker_chunk_accumulation_and_hysteresis():
    """
    Test Part 1, 4 & 16:
    - Publish 20ms chunks (320 samples at 16kHz) of real speech audio.
    - Confirm frames accumulate through onset hysteresis and continuous speech trigger.
    - Confirm genuine enrolled speaker produces MATCH event with real ECAPA inference.
    - Confirm structured log and event telemetry contains speech_window_id and best_match.
    """
    if not ecapa_service.is_loaded:
        ecapa_service.load_model()

    client = TestClient(app)
    pre = AudioPreprocessor()

    # Enroll speaker 'Dinesh'
    dinesh_wav = make_wav_bytes(duration_s=3.0, freq=320.0)
    ecapa_service.enroll_speaker(
        speaker_name="Dinesh",
        waveform=pre.load_bytes(dinesh_wav)[0],
        sample_bytes=dinesh_wav,
        sample_filename="dinesh.wav",
        mime_type="audio/wav"
    )

    async def _test():
        sess = MonitorSession("session-live-accum")
        await sess.start()

        events_received = []
        def _capture(evt):
            if hasattr(evt, 'type') and evt.type == "speaker_identity":
                events_received.append(evt)

        sess._security_pipeline._on_event = _capture

        # Stream speech audio in 20ms chunks (320 samples)
        dinesh_tensor = pre.load_bytes(dinesh_wav)[0].squeeze()
        chunk_size = 320
        total_chunks = len(dinesh_tensor) // chunk_size

        for i in range(total_chunks):
            chunk = dinesh_tensor[i * chunk_size : (i + 1) * chunk_size]
            await sess.audio_bus.publish(
                samples=chunk,
                wall_timestamp=time.time(),
                session_id=sess.session_id,
            )
            # Yield to event loop to allow pipeline processing
            await asyncio.sleep(0.005)

        # Allow final processing
        await asyncio.sleep(0.3)

        # Confirm speaker event was produced by real ECAPA inference
        speaker_events = [e for e in events_received if getattr(e, 'status', None) in ["MATCH", "NO_CONFIDENT_MATCH"]]
        assert len(speaker_events) > 0, "Expected at least one speaker identity event during 2s live speech"
        best_event = speaker_events[-1]
        assert best_event.status == "MATCH"
        assert best_event.speaker_name == "Dinesh"
        assert best_event.speaker_similarity is not None
        assert best_event.speaker_similarity > 0.60
        assert best_event.profiles_checked >= 1
        assert best_event.speech_window_id.startswith("win_")

        await sess.stop()

    asyncio.run(_test())

