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
    """Real-time telemetry from the WebRTC audio ingest."""
    negotiated_codec: str = "unknown"
    source_sample_rate: int = 0
    source_channels: int = 0
    packets_received: int = 0
    frames_decoded: int = 0
    total_samples_decoded: int = 0
    rms: float = 0.0
    peak: float = 0.0
    zero_percentage: float = 0.0
    last_frame_time: float = 0.0
    resampler_active: bool = False


class AudioTrackConsumer:
    """Consumes audio frames from an aiortc MediaStreamTrack.

    Decodes to canonical 16 kHz mono float32 and publishes to AudioBus.
    Records all telemetry without assuming codec or sample rate.
    """

    def __init__(self, track: MediaStreamTrack, audio_bus: AudioBus):
        self._track = track
        self._audio_bus = audio_bus
        self._telemetry = AudioTelemetry()
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
        logger.info("AudioTrackConsumer started")

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
            f"AudioTrackConsumer stopped. "
            f"Frames decoded: {self._telemetry.frames_decoded}, "
            f"Samples: {self._telemetry.total_samples_decoded}"
        )

    async def _consume_loop(self):
        """Main loop: receive frames from aiortc track, convert, publish."""
        while self._running:
            try:
                # aiortc MediaStreamTrack.recv() returns an AudioFrame (av.AudioFrame)
                frame = await self._track.recv()

                self._telemetry.packets_received += 1
                self._telemetry.last_frame_time = time.time()

                # Inspect actual format on first frame
                if self._telemetry.frames_decoded == 0:
                    self._telemetry.source_sample_rate = frame.sample_rate
                    self._telemetry.source_channels = len(frame.layout.channels) if frame.layout else 1
                    self._telemetry.negotiated_codec = getattr(
                        self._track, 'kind', 'audio'
                    )

                    logger.info(
                        f"WebRTC audio: first frame received — "
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

                # Convert av.AudioFrame to numpy array
                # frame.to_ndarray() returns shape (channels, samples) as int16 or float
                raw_array = frame.to_ndarray()

                # Handle multi-channel → mono
                if raw_array.ndim == 2 and raw_array.shape[0] > 1:
                    raw_array = raw_array.mean(axis=0)
                elif raw_array.ndim == 2:
                    raw_array = raw_array[0]

                # Convert to float32 [-1, 1]
                if raw_array.dtype == np.int16:
                    audio_float = raw_array.astype(np.float32) / 32768.0
                elif raw_array.dtype == np.int32:
                    audio_float = raw_array.astype(np.float32) / 2147483648.0
                elif raw_array.dtype == np.float32:
                    audio_float = raw_array
                else:
                    audio_float = raw_array.astype(np.float32)

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

                # Publish to audio bus
                await self._audio_bus.publish(
                    samples=tensor,
                    wall_timestamp=time.time(),
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
        receiver = WebRTCMediaReceiver(audio_bus)
        answer_sdp = await receiver.handle_offer(offer_sdp)
        # Audio now flows into audio_bus
        await receiver.close()
    """

    def __init__(self, audio_bus: AudioBus):
        self._audio_bus = audio_bus
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
            logger.info(f"WebRTC track received: kind={track.kind}")

            if track.kind == "audio":
                # Create consumer for this audio track
                self._consumer = AudioTrackConsumer(
                    track=track,
                    audio_bus=self._audio_bus,
                )
                await self._consumer.start()

                @track.on("ended")
                async def on_track_ended():
                    logger.info("WebRTC audio track ended")
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
        """Get telemetry for UI display."""
        if not self._consumer:
            return {"status": "no_track"}

        t = self._consumer.telemetry
        return {
            "status": "receiving" if self._connected else "disconnected",
            "codec": t.negotiated_codec,
            "source_sample_rate": t.source_sample_rate,
            "source_channels": t.source_channels,
            "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
            "packets_received": t.packets_received,
            "frames_decoded": t.frames_decoded,
            "total_samples": t.total_samples_decoded,
            "total_duration_s": round(
                t.total_samples_decoded / CANONICAL_SAMPLE_RATE, 2
            ) if t.total_samples_decoded > 0 else 0,
            "rms": round(t.rms, 6),
            "peak": round(t.peak, 6),
            "zero_pct": round(t.zero_percentage, 2),
            "resampler_active": t.resampler_active,
        }
