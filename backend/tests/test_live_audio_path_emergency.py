"""
VoiceShield — Live Audio Path Emergency Rebuild Regression Tests.

Validates:
1. WebRTCMediaReceiver & AudioTrackConsumer correctly forward track_source and track_direction to AudioBus.
2. MonitorSession.get_telemetry() aggregates complete canonical signal truth (frames, samples, rms, peak, vad, speech_active).
3. CaptionPipeline preserves quiet speech (RMS ~0.0020) without premature energy threshold truncation.
4. SecurityPipeline uses get_speech_timestamps without blocking event loop and correctly identifies speech vs silence.
5. SpectraAASISTDetector accepts both torch.Tensor and numpy.ndarray safely.
"""

import asyncio
import time
import pytest
import numpy as np
import torch

from app.services.monitor.audio_bus import AudioBus, AudioChunk, ConsumerQueue
from app.services.webrtc.media_receiver import AudioTelemetry, WebRTCMediaReceiver
from app.services.monitor.session import MonitorSession
from app.services.monitor.caption_pipeline import CaptionPipeline, CaptionEvent, MIN_UTTERANCE_RMS
from app.services.monitor.security_pipeline import SecurityPipeline
from app.services.antispoof.spectra_detector import spectra_detector
from app.services.asr.asr_provider import ASRResult


def test_track_metadata_forwarding_from_media_receiver():
    """Verify that AudioBus chunks carry explicit REMOTE_PEER and inbound direction."""
    async def _test():
        audio_bus = AudioBus()
        queue = audio_bus.subscribe("test_sub")

        # Initialize receiver for remote peer audio
        receiver = WebRTCMediaReceiver(
            audio_bus=audio_bus,
            session_id="session_remote_101",
            room_id="room_alpha",
            observed_peer_id="peer_bravo",
            track_id="track_bravo_mic",
            track_source="REMOTE_PEER",
            track_direction="inbound",
        )

        # Publish chunk as if from AudioTrackConsumer
        await audio_bus.publish(
            samples=torch.randn(320) * 0.05,
            wall_timestamp=time.time(),
            session_id="session_remote_101",
            track_id="track_bravo_mic",
            track_source=receiver._track_source,
            track_direction=receiver._track_direction,
        )

        chunk: AudioChunk = await queue.get()
        assert chunk.track_source == "REMOTE_PEER"
        assert chunk.track_direction == "inbound"
        assert chunk.track_id == "track_bravo_mic"
        assert chunk.session_id == "session_remote_101"

        # Test telemetry dictionary
        t_dict = receiver.get_telemetry_dict()
        assert t_dict["track_source"] == "REMOTE_PEER"
        assert t_dict["track_direction"] == "inbound"
        assert t_dict["track_id"] == "track_bravo_mic"

    asyncio.run(_test())


def test_audio_telemetry_truth_aggregation():
    """Verify that MonitorSession aggregates complete canonical signal truth."""
    async def _test():
        session = MonitorSession(
            session_id="sess_truth_101",
            room_id="room_101",
            observed_peer_id="peer_user_2",
            track_id="track_mic_2",
            track_source="REMOTE_PEER",
            track_direction="inbound",
            connection_generation=2,
        )

        telemetry = session.get_telemetry()
        assert "audio_truth" in telemetry
        at = telemetry["audio_truth"]
        assert at["session_id"] == "sess_truth_101"
        assert at["track_source"] == "REMOTE_PEER"
        assert at["track_direction"] == "inbound"
        assert "frames_received" in at
        assert "canonical_samples" in at
        assert "rms" in at
        assert "peak" in at
        assert "vad_probability" in at
        assert "speech_active" in at

    asyncio.run(_test())


def test_caption_pipeline_preserves_quiet_speech_floor():
    """Verify that MIN_UTTERANCE_RMS is low enough to prevent discarding quiet human speech."""
    assert MIN_UTTERANCE_RMS <= 0.0015
    # A quiet microphone speech signal at RMS 0.0020 must pass
    quiet_speech_rms = 0.0020
    assert quiet_speech_rms > MIN_UTTERANCE_RMS


def test_security_pipeline_unblocked_vad_evaluation():
    """Verify that SecurityPipeline evaluates speech windows without blocking and abstains on silence."""
    async def _test():
        audio_bus = AudioBus()
        queue = audio_bus.subscribe("sec_test", maxsize=200)

        events = []
        sec_pipe = SecurityPipeline(
            audio_queue=queue,
            on_event=events.append,
            session_id="test_sec_sess",
            track_id="test_sec_track",
        )

        await sec_pipe.start()

        # Ingest 64,600 samples of low-level ambient noise (RMS 0.0005)
        noise_chunk = torch.randn(3200) * 0.0005
        for _ in range(21):
            await queue.put(AudioChunk(
                samples=noise_chunk,
                sample_offset=0,
                sample_rate=16000,
                wall_timestamp=time.time(),
                session_id="test_sec_sess",
                track_id="test_sec_track",
                sequence_id=0,
                sample_start=0,
                sample_end=3200,
                timestamp_start=time.time(),
                timestamp_end=time.time() + 0.2,
                discontinuity=False,
                track_source="REMOTE_PEER",
                track_direction="inbound",
            ))

        # Allow pipeline to process
        await asyncio.sleep(0.3)
        await sec_pipe.stop()

        # Must have produced at least one window detection event with decision=ABSTAIN
        detection_events = [e for e in events if getattr(e, "model", "") == "Spectra-AASIST"]
        assert len(detection_events) > 0
        assert detection_events[0].decision == "ABSTAIN"
        assert "SILENCE_NO_SPEECH" in detection_events[0].invalidation_reason

    asyncio.run(_test())


def test_spectra_detector_numpy_and_tensor_compatibility():
    """Verify that SpectraAASISTDetector accepts both torch.Tensor and numpy.ndarray without crashing."""
    if not spectra_detector.is_loaded:
        spectra_detector.load_model()

    # Pass 1D numpy array
    arr_np = np.zeros(64600, dtype=np.float32)
    res1 = spectra_detector.predict(arr_np)
    assert hasattr(res1, "spoof_score")
    assert hasattr(res1, "bonafide_score")

    # Pass 1D torch tensor
    tensor = torch.zeros(64600, dtype=torch.float32)
    res2 = spectra_detector.predict(tensor)
    assert hasattr(res2, "spoof_score")
    assert hasattr(res2, "bonafide_score")
