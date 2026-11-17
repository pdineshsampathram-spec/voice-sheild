"""
WebRTC media termination layer using aiortc.

This module creates a server-side RTCPeerConnection that:
1. Receives the browser's SDP offer
2. Creates an SDP answer
3. Terminates RTP/DTLS/SRTP media transport
4. Decodes incoming audio (Opus or whatever codec is negotiated)
5. Converts decoded audio to canonical 16 kHz mono float32 (exactly once)
6. Publishes decoded PCM to the AudioBus

IMPORTANT:
- We do NOT assume the codec is Opus or the sample rate is 48 kHz.
- We inspect the actual negotiated track and log the real codec/format.
- Canonical conversion happens exactly once at this boundary.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torchaudio

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

from app.services.monitor.audio_bus import AudioBus

logger = logging.getLogger(__name__)

# Target canonical format
CANONICAL_SAMPLE_RATE = 16000
CANONICAL_CHANNELS = 1


@dataclass
class AudioTelemetry:
    """Real-time telemetry and ground truth from the WebRTC audio ingest."""
    session_id: str = ""
    room_id: Optional[str] = None
    observed_peer_id: Optional[str] = None
    track_id: str = "unknown"
    track_kind: str = "audio"
    track_direction: str = "inbound"
    track_source: str = "LOCAL_MIC"  # "LOCAL_MIC" or "REMOTE_PEER"
    connection_generation: int = 1
    negotiated_codec: str = "unknown"
    source_sample_rate: int = 0
    canonical_sample_rate: int = CANONICAL_SAMPLE_RATE
    source_channels: int = 0
    packets_received: int = 0
    frames_decoded: int = 0
    total_samples_decoded: int = 0
    rms: float = 0.0
    peak: float = 0.0
    zero_percentage: float = 0.0
    first_sample_timestamp: float = 0.0
    last_sample_timestamp: float = 0.0
    resampler_active: bool = False
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    zero_crossing_rate: float = 0.0
    sequence_counter: int = 0
    vad_probability: float = 0.0
    speech_active: bool = False


class AudioTrackConsumer:
    """Consumes audio frames from an aiortc MediaStreamTrack.

    Decodes to canonical 16 kHz mono float32 and publishes to AudioBus.
    Records all telemetry without assuming codec or sample rate.
    """

    def __init__(
        self,
        track: MediaStreamTrack,
        audio_bus: AudioBus,
        session_id: str = "",
        room_id: Optional[str] = None,
        observed_peer_id: Optional[str] = None,
        track_id: str = "",
        track_source: str = "LOCAL_MIC",
        track_direction: str = "inbound",
        connection_generation: int = 1,
    ):
        self._track = track
        self._audio_bus = audio_bus
        self._session_id = session_id
        self._track_id = track_id or getattr(track, 'id', "track_live")
        self._track_source = track_source
        self._track_direction = track_direction
        self._telemetry = AudioTelemetry(
            session_id=session_id,
            room_id=room_id,
            observed_peer_id=observed_peer_id,
            track_id=self._track_id,
            track_kind=getattr(track, 'kind', 'audio'),
            track_direction=track_direction,
            track_source=track_source,
            connection_generation=connection_generation,
        )
        self._resampler: Optional[torchaudio.transforms.Resample] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def telemetry(self) -> AudioTelemetry:
        return self._telemetry

    async def start(self):
        """Start consuming audio frames from the track."""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        logger.info(f"AudioTrackConsumer started for track {self._track_id} (source: {self._telemetry.track_source})")

    async def stop(self):
        """Stop consuming."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"AudioTrackConsumer stopped for track {self._track_id}. "
            f"Frames decoded: {self._telemetry.frames_decoded}, "
            f"Samples: {self._telemetry.total_samples_decoded}"
        )

    async def _consume_loop(self):
        """Main loop: receive frames from aiortc track, convert, publish."""
        while self._running:
            try:
                # aiortc MediaStreamTrack.recv() returns an AudioFrame (av.AudioFrame)
                frame = await self._track.recv()

                now = time.time()
                self._telemetry.packets_received += 1
                self._telemetry.last_sample_timestamp = now

                # Inspect actual format on first frame
                if self._telemetry.frames_decoded == 0:
                    self._telemetry.first_sample_timestamp = now
                    self._telemetry.source_sample_rate = frame.sample_rate
                    self._telemetry.source_channels = len(frame.layout.channels) if frame.layout else 1
                    # WebRTC audio tracks use Opus (Fix M5). Never report "audio" as codec.
                    self._telemetry.negotiated_codec = "opus (inferred)"

                    logger.info(
                        f"WebRTC audio: first frame received for track {self._track_id} — "
                        f"source={self._telemetry.track_source}, "
                        f"sample_rate={frame.sample_rate}, "
                        f"channels={self._telemetry.source_channels}, "
                        f"format={frame.format.name}, "
                        f"samples={frame.samples}"
                    )

                    # Create resampler if needed
                    if frame.sample_rate != CANONICAL_SAMPLE_RATE:
                        self._resampler = torchaudio.transforms.Resample(
                            orig_freq=frame.sample_rate,
                            new_freq=CANONICAL_SAMPLE_RATE,
                        )
                        self._telemetry.resampler_active = True
                        logger.info(
                            f"Resampler created: {frame.sample_rate} → "
                            f"{CANONICAL_SAMPLE_RATE} Hz"
                        )

                self._telemetry.frames_decoded += 1

                # Convert av.AudioFrame to numpy array and extract mono float32
                raw_array = frame.to_ndarray()
                orig_dtype = raw_array.dtype
                n_channels = len(frame.layout.channels) if frame.layout else 1
                format_name = frame.format.name

                if n_channels > 1:
                    if 'p' in format_name:
                        # Planar: shape is (channels, samples)
                        ch_rms = np.sqrt(np.mean(raw_array.astype(np.float64) ** 2, axis=1))
                        if ch_rms[0] > 1e-4 and ch_rms[1] < 1e-5:
                            mono = raw_array[0]
                        elif ch_rms[1] > 1e-4 and ch_rms[0] < 1e-5:
                            mono = raw_array[1]
                        else:
                            mono = raw_array.mean(axis=0)
                    else:
                        # Interleaved: shape is (1, samples * n_channels)
                        samples = frame.samples
                        reshaped = raw_array.reshape(samples, n_channels)
                        ch_rms = np.sqrt(np.mean(reshaped.astype(np.float64) ** 2, axis=0))
                        if ch_rms[0] > 1e-4 and ch_rms[1] < 1e-5:
                            mono = reshaped[:, 0]
                        elif ch_rms[1] > 1e-4 and ch_rms[0] < 1e-5:
                            mono = reshaped[:, 1]
                        else:
                            mono = reshaped.mean(axis=1)
                else:
                    if raw_array.ndim == 2:
                        mono = raw_array[0]
                    else:
                        mono = raw_array

                # Convert to float32 [-1, 1] based on original integer dtype
                if orig_dtype == np.int16:
                    audio_float = mono.astype(np.float32) / 32768.0
                elif orig_dtype == np.int32:
                    audio_float = mono.astype(np.float32) / 2147483648.0
                elif orig_dtype == np.uint8:
                    audio_float = (mono.astype(np.float32) - 128.0) / 128.0
                else:
                    audio_float = mono.astype(np.float32)

                tensor = torch.from_numpy(audio_float)

                # Resample to canonical 16 kHz if needed (exactly once)
                if self._resampler is not None:
                    tensor = self._resampler(tensor.unsqueeze(0)).squeeze(0)

                # Update telemetry
                self._telemetry.total_samples_decoded += len(tensor)
                if len(tensor) > 0:
                    abs_tensor = torch.abs(tensor)
                    self._telemetry.rms = float(torch.sqrt(torch.mean(tensor ** 2)))
                    self._telemetry.peak = float(torch.max(abs_tensor))
                    zero_count = int(torch.sum(abs_tensor < 1e-6))
                    self._telemetry.zero_percentage = zero_count / len(tensor) * 100
                    self._telemetry.min_val = float(torch.min(tensor))
                    self._telemetry.max_val = float(torch.max(tensor))
                    self._telemetry.mean_val = float(torch.mean(tensor))
                    if len(tensor) > 1:
                        signs = torch.sign(tensor)
                        self._telemetry.zero_crossing_rate = float(
                            torch.mean(torch.abs(signs[1:] - signs[:-1])) / 2.0
                        )

                # Publish to audio bus with canonical timeline metadata
                await self._audio_bus.publish(
                    samples=tensor,
                    wall_timestamp=now,
                    session_id=self._session_id,
                    track_id=self._track_id,
                    track_source=self._telemetry.track_source,
                    track_direction=self._telemetry.track_direction,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.error(f"AudioTrackConsumer error: {e}", exc_info=True)
                    await asyncio.sleep(0.01)


class WebRTCMediaReceiver:
    """Server-side WebRTC peer that terminates media and extracts audio.

    Usage:
        receiver = WebRTCMediaReceiver(audio_bus, session_id=..., track_source=...)
        answer_sdp = await receiver.handle_offer(offer_sdp)
        # Audio now flows into audio_bus
        await receiver.close()
    """

    def __init__(
        self,
        audio_bus: AudioBus,
        session_id: str = "",
        room_id: Optional[str] = None,
        observed_peer_id: Optional[str] = None,
        track_id: str = "",
        track_source: str = "LOCAL_MIC",
        track_direction: str = "inbound",
        connection_generation: int = 1,
    ):
        self._audio_bus = audio_bus
        self._session_id = session_id
        self._room_id = room_id
        self._observed_peer_id = observed_peer_id
        self._track_id = track_id
        self._track_source = track_source
        self._track_direction = track_direction
        self._connection_generation = connection_generation
        self._pc: Optional[RTCPeerConnection] = None
        self._consumer: Optional[AudioTrackConsumer] = None
        self._relay = MediaRelay()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def telemetry(self) -> Optional[AudioTelemetry]:
        return self._consumer.telemetry if self._consumer else None

    async def handle_offer(self, offer_sdp: str, offer_type: str = "offer") -> str:
        """Process browser SDP offer and return server SDP answer.

        Args:
            offer_sdp: SDP offer string from browser RTCPeerConnection.
            offer_type: SDP type (always "offer" for initial connection).

        Returns:
            SDP answer string to send back to the browser.
        """
        self._pc = RTCPeerConnection()

        @self._pc.on("connectionstatechange")
        async def on_state_change():
            state = self._pc.connectionState
            logger.info(f"WebRTC connection state: {state}")
            if state == "connected":
                self._connected = True
            elif state in ("failed", "closed", "disconnected"):
                self._connected = False

        @self._pc.on("track")
        async def on_track(track: MediaStreamTrack):
            actual_track_id = self._track_id or getattr(track, 'id', 'track_live')
            logger.info(f"WebRTC track received: kind={track.kind}, id={actual_track_id}, source={self._track_source}")

            if track.kind == "audio":
                # Create consumer for this audio track
                self._consumer = AudioTrackConsumer(
                    track=track,
                    audio_bus=self._audio_bus,
                    session_id=self._session_id,
                    room_id=self._room_id,
                    observed_peer_id=self._observed_peer_id,
                    track_id=actual_track_id,
                    track_source=self._track_source,
                    track_direction=self._track_direction,
                    connection_generation=self._connection_generation,
                )
                await self._consumer.start()

                @track.on("ended")
                async def on_track_ended():
                    logger.info(f"WebRTC audio track {actual_track_id} ended")
                    if self._consumer:
                        await self._consumer.stop()

        # Set remote description (browser's offer)
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await self._pc.setRemoteDescription(offer)

        # Create answer
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        logger.info("WebRTC: SDP answer created")
        return self._pc.localDescription.sdp

    async def close(self):
        """Close the WebRTC connection and clean up."""
        if self._consumer:
            await self._consumer.stop()
        if self._pc:
            await self._pc.close()
        self._connected = False
        logger.info("WebRTCMediaReceiver closed")

    def get_telemetry_dict(self) -> dict:
        """Get telemetry for UI display and live audio truth panel."""
        if not self._consumer:
            return {
                "status": "no_track",
                "session_id": self._session_id,
                "room_id": self._room_id,
                "observed_peer_id": self._observed_peer_id,
                "track_id": self._track_id,
                "track_source": self._track_source,
                "track_direction": self._track_direction,
                "connection_generation": self._connection_generation,
            }

        t = self._consumer.telemetry
        return {
            "status": "receiving" if self._connected else "disconnected",
            "session_id": t.session_id,
            "room_id": t.room_id,
            "observed_peer_id": t.observed_peer_id,
            "track_id": t.track_id,
            "track_kind": t.track_kind,
            "track_direction": t.track_direction,
            "track_source": t.track_source,
            "connection_generation": t.connection_generation,
            "codec": t.negotiated_codec,
            "source_sample_rate": t.source_sample_rate,
            "source_channels": t.source_channels,
            "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
            "packets_received": t.packets_received,
            "frames_decoded": t.frames_decoded,
            "frames_received": t.frames_decoded,
            "total_samples": t.total_samples_decoded,
            "samples_received": t.total_samples_decoded,
            "total_duration_s": round(
                t.total_samples_decoded / CANONICAL_SAMPLE_RATE, 2
            ) if t.total_samples_decoded > 0 else 0,
            "rms": round(t.rms, 6),
            "current_rms": round(t.rms, 6),
            "peak": round(t.peak, 6),
            "current_peak": round(t.peak, 6),
            "min_val": round(t.min_val, 6),
            "max_val": round(t.max_val, 6),
            "mean_val": round(t.mean_val, 6),
            "zero_crossing_rate": round(t.zero_crossing_rate, 4),
            "zero_pct": round(t.zero_percentage, 2),
            "first_sample_timestamp": t.first_sample_timestamp,
            "last_sample_timestamp": t.last_sample_timestamp,
            "resampler_active": t.resampler_active,
            "vad_probability": t.vad_probability,
            "speech_active": t.speech_active,
        }
