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


def test_audio_track_consumer_av_frame_consumption_interleaved_and_planar():
    """Phase 6 & 8: Verify AudioTrackConsumer unpacks interleaved s16 and planar s16p without crashing."""
    import av
    from app.services.webrtc.media_receiver import AudioTrackConsumer

    class MockTrack:
        def __init__(self, frames):
            self.frames = list(frames)
            self.kind = "audio"
            self.id = "mock_track_001"

        async def recv(self):
            if self.frames:
                return self.frames.pop(0)
            await asyncio.sleep(100)  # wait until cancelled

    async def _test():
        audio_bus = AudioBus()
        queue = audio_bus.subscribe("test_consumer_queue")

        # 1. Create a 48kHz s16 stereo frame with known sine wave
        t = np.linspace(0, 0.02, 960, endpoint=False)
        sine = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
        interleaved = np.empty((960, 2), dtype=np.int16)
        interleaved[:, 0] = sine
        interleaved[:, 1] = 0  # mono mic in channel 0

        frame_s16 = av.AudioFrame(format='s16', layout='stereo', samples=960)
        frame_s16.sample_rate = 48000
        frame_s16.planes[0].update(interleaved.tobytes())

        # 2. Create a 48kHz s16p planar stereo frame
        frame_s16p = av.AudioFrame(format='s16p', layout='stereo', samples=960)
        frame_s16p.sample_rate = 48000
        frame_s16p.planes[0].update(sine.tobytes())
        frame_s16p.planes[1].update(sine.tobytes())

        mock_track = MockTrack([frame_s16, frame_s16p])
        consumer = AudioTrackConsumer(
            track=mock_track,
            audio_bus=audio_bus,
            session_id="sess_av_test",
            track_id="mock_track_001",
            track_source="LOCAL_MIC",
            track_direction="outbound",
        )

        await consumer.start()
        # Wait briefly for both frames to be received, converted and published
        chunk1: AudioChunk = await queue.get()
        chunk2: AudioChunk = await queue.get()
        await consumer.stop()

        # Verify chunk 1 (interleaved s16 with mono in left channel)
        assert chunk1.track_source == "LOCAL_MIC"
        assert chunk1.track_direction == "outbound"
        assert chunk1.track_id == "mock_track_001"
        assert len(chunk1.samples) == 320  # 960 @ 48k resampled to 16k = 320
        rms1 = float(torch.sqrt(torch.mean(chunk1.samples ** 2)))
        assert 0.15 < rms1 < 0.25  # Expected ~0.216

        # Verify chunk 2 (planar s16p)
        assert len(chunk2.samples) == 320
        rms2 = float(torch.sqrt(torch.mean(chunk2.samples ** 2)))
        assert 0.15 < rms2 < 0.25

        # Verify telemetry statistics
        t_dict = consumer.telemetry
        assert t_dict.frames_decoded == 2
        assert t_dict.total_samples_decoded == 640
        assert t_dict.min_val < -0.1
        assert t_dict.max_val > 0.1
        assert t_dict.zero_crossing_rate > 0.0

    asyncio.run(_test())


def test_known_signal_deterministic_round_trip():
    """Phase 9: Verify known 440 Hz test signal round-trip through media receiver and audio bus."""
    import av
    from app.services.webrtc.media_receiver import AudioTrackConsumer

    class MockSineTrack:
        def __init__(self, num_frames=5):
            self.kind = "audio"
            self.id = "known_signal_track"
            self.frames = []
            t_total = np.linspace(0, 0.02 * num_frames, 960 * num_frames, endpoint=False)
            sine_total = (np.sin(2 * np.pi * 440 * t_total) * 16384).astype(np.int16)
            for i in range(num_frames):
                f = av.AudioFrame(format='s16', layout='mono', samples=960)
                f.sample_rate = 48000
                f.planes[0].update(sine_total[i*960:(i+1)*960].tobytes())
                self.frames.append(f)

        async def recv(self):
            if self.frames:
                return self.frames.pop(0)
            await asyncio.sleep(100)

    async def _test():
        audio_bus = AudioBus()
        queue = audio_bus.subscribe("test_sine_queue")
        mock_track = MockSineTrack(num_frames=3)

        consumer = AudioTrackConsumer(
            track=mock_track,
            audio_bus=audio_bus,
            session_id="sess_sine_test",
            track_id="known_signal_track",
            track_source="LOCAL_MIC",
            track_direction="outbound",
        )

        await consumer.start()
        collected = []
        for _ in range(3):
            chunk = await queue.get()
            collected.append(chunk.samples)
        await consumer.stop()

        all_samples = torch.cat(collected)
        assert len(all_samples) == 960  # 3 * 320 samples at 16kHz
        overall_rms = float(torch.sqrt(torch.mean(all_samples ** 2)))
        # 16384 / 32768 = 0.5 peak amplitude, expected sine RMS = 0.5 / sqrt(2) ≈ 0.3535
        assert 0.33 < overall_rms < 0.37
        assert consumer.telemetry.resampler_active is True

    asyncio.run(_test())
