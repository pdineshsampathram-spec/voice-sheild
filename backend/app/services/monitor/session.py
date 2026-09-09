"""
Monitor session — per-session isolation for all pipeline state.

Each MonitorSession owns:
- AudioBus (fan-out)
- CaptionPipeline (VAD-A + ASR)
- SecurityPipeline (VAD-B + AASIST + ECAPA)
- WebRTCMediaReceiver (aiortc)
- Timeline (sample-position-based event log)
- Telemetry

Complete isolation between sessions. No shared mutable state.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from app.services.monitor.audio_bus import AudioBus, OverflowPolicy
from app.services.monitor.caption_pipeline import CaptionPipeline, CaptionEvent
from app.services.monitor.security_pipeline import (
    SecurityPipeline, DetectionEvent, SpeakerEvent, WindowProgress
)
from app.services.webrtc.media_receiver import WebRTCMediaReceiver
from app.services.asr.hf_whisper_provider import HFWhisperProvider

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """A single event in the monitor timeline."""
    event_type: str
    sample_start: int
    sample_end: int
    wall_time: float
    data: dict = field(default_factory=dict)


class MonitorSession:
    """Isolated session with all pipeline state.

    Creates and owns:
    - One AudioBus
    - One CaptionPipeline (with VAD-A + ASR)
    - One SecurityPipeline (with VAD-B + AASIST + ECAPA)
    - One WebRTCMediaReceiver
    - Shared ThreadPoolExecutor for ML inference
    """

    def __init__(self, session_id: str = None, asr_model_size: str = "base"):
        self.session_id = session_id or str(uuid.uuid4())
        self._asr_model_size = asr_model_size
        self._started = False
        self._start_time: float = 0.0

        # Infrastructure
        self._audio_bus = AudioBus()
        self._executor = ThreadPoolExecutor(
            max_workers=2,  # One for ASR, one for AASIST/ECAPA
            thread_name_prefix=f"monitor-{self.session_id[:8]}",
        )

        # Pipelines
        self._caption_pipeline: Optional[CaptionPipeline] = None
        self._security_pipeline: Optional[SecurityPipeline] = None
        self._webrtc_receiver: Optional[WebRTCMediaReceiver] = None

        # Event buffer for WebSocket relay
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        # Timeline
        self._timeline: list[TimelineEntry] = []
        self._event_seq: int = 0

    @property
    def audio_bus(self) -> AudioBus:
        return self._audio_bus

    @property
    def webrtc_receiver(self) -> Optional[WebRTCMediaReceiver]:
        return self._webrtc_receiver

    def _on_event(self, event):
        """Unified event handler for both pipelines."""
        # Convert dataclass events to dicts for WebSocket relay
        if hasattr(event, '__dataclass_fields__'):
            event_dict = {
                k: getattr(event, k)
                for k in event.__dataclass_fields__
            }
        else:
            event_dict = {"type": "unknown", "data": str(event)}

        now = time.time()
        self._event_seq += 1
        event_dict["session_id"] = self.session_id
        event_dict["event_id"] = f"{self.session_id[:8]}-{self._event_seq}"
        event_dict["seq"] = self._event_seq
        event_dict["timestamp"] = now

        # Add to timeline
        self._timeline.append(TimelineEntry(
            event_type=event_dict.get("type", "unknown"),
            sample_start=event_dict.get("sample_start", 0),
            sample_end=event_dict.get("sample_end", 0),
            wall_time=now,
            data=event_dict,
        ))

        # Queue for WebSocket relay (non-blocking)
        try:
            self._event_queue.put_nowait(event_dict)
        except asyncio.QueueFull:
            # Drop oldest event
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._event_queue.put_nowait(event_dict)
            except asyncio.QueueFull:
                pass

    async def start(self, asr_provider=None):
        """Initialize and start all pipelines.

        Args:
            asr_provider: Optional pre-loaded ASR provider. If None,
                          creates and loads FasterWhisperProvider.
        """
        self._start_time = time.time()
        logger.info(f"MonitorSession {self.session_id}: starting")
        # Create ASR provider
        if asr_provider is None:
            asr_provider = HFWhisperProvider(
                model_name=self._asr_model_size,
                backend="ctranslate2",
                beam_size=1,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            asr_provider.load_model()

        logger.info(
            f"[Live ASR Init] MonitorSession {self.session_id}: ASR Provider={type(asr_provider).__name__}, "
            f"model={getattr(asr_provider, '_model_name', 'base')}, backend={getattr(asr_provider, '_backend', 'ctranslate2')}"
        )

        # Subscribe consumers to audio bus
        caption_queue = self._audio_bus.subscribe(
            name="caption",
            maxsize=200,
            policy=OverflowPolicy.DROP_OLDEST,
        )

        security_queue = self._audio_bus.subscribe(
            name="security",
            maxsize=200,
            policy=OverflowPolicy.EMIT_DISCONTINUITY,
        )

        # Create pipelines with independent state
        self._caption_pipeline = CaptionPipeline(
            audio_queue=caption_queue,
            asr_provider=asr_provider,
            on_event=self._on_event,
            executor=self._executor,
        )

        self._security_pipeline = SecurityPipeline(
            audio_queue=security_queue,
            on_event=self._on_event,
            executor=self._executor,
        )

        # Wire discontinuity handler
        security_queue.on_discontinuity = self._security_pipeline.on_discontinuity

        # Create WebRTC media receiver
        self._webrtc_receiver = WebRTCMediaReceiver(self._audio_bus)

        # Start pipelines
        await self._caption_pipeline.start()
        await self._security_pipeline.start()

        self._started = True
        logger.info(f"MonitorSession {self.session_id}: all pipelines started")

    async def stop(self):
        """Stop all pipelines and clean up."""
        logger.info(f"MonitorSession {self.session_id}: stopping")

        if self._caption_pipeline:
            await self._caption_pipeline.stop()
        if self._security_pipeline:
            await self._security_pipeline.stop()
        if self._webrtc_receiver:
            await self._webrtc_receiver.close()

        self._audio_bus.reset()
        self._executor.shutdown(wait=False)
        self._started = False

        logger.info(
            f"MonitorSession {self.session_id}: stopped. "
            f"Timeline entries: {len(self._timeline)}"
        )

    async def get_event(self, timeout: float = 0.1) -> Optional[dict]:
        """Get the next event for WebSocket relay."""
        try:
            return await asyncio.wait_for(
                self._event_queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def get_telemetry(self) -> dict:
        """Get comprehensive session telemetry including actual runtime models."""
        from app.services.antispoof.spectra_detector import spectra_detector
        from app.services.speaker.ecapa_service import ecapa_service

        asr_info = {}
        if self._caption_pipeline and hasattr(self._caption_pipeline, "_asr") and self._caption_pipeline._asr:
            asr = self._caption_pipeline._asr
            asr_info = {
                "asr_provider": type(asr).__name__,
                "asr_model": getattr(asr, "_model_name", "base"),
                "asr_backend": getattr(asr, "_backend", "ctranslate2"),
                "asr_loaded": asr.is_loaded() if hasattr(asr, "is_loaded") else True,
            }

        telemetry = {
            "session_id": self.session_id,
            "models": {
                **asr_info,
                "aasist_model": "Spectra-AASIST",
                "aasist_loaded": spectra_detector.is_loaded,
                "ecapa_model": "ECAPA-TDNN",
                "ecapa_loaded": ecapa_service.is_loaded,
                "vad_model": "Silero-VAD",
            },
            "uptime_s": round(time.time() - self._start_time, 1) if self._started else 0,
            "audio_bus": self._audio_bus.get_telemetry(),
            "caption_pipeline": (
                self._caption_pipeline.get_telemetry()
                if self._caption_pipeline is not None
                else {"status": "unavailable"}
            ),
            "security_pipeline": (
                self._security_pipeline.get_telemetry()
                if self._security_pipeline is not None
                else {"status": "unavailable"}
            ),
            "timeline_entries": len(self._timeline),
        }

        if self._webrtc_receiver:
            telemetry["webrtc"] = self._webrtc_receiver.get_telemetry_dict()
        else:
            telemetry["webrtc"] = {"status": "unavailable"}

        return telemetry


# Active sessions registry
_sessions: dict[str, MonitorSession] = {}


def get_session(session_id: str) -> Optional[MonitorSession]:
    return _sessions.get(session_id)


def create_session(session_id: str = None, **kwargs) -> MonitorSession:
    session = MonitorSession(session_id=session_id, **kwargs)
    _sessions[session.session_id] = session
    return session


async def remove_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        await session.stop()
