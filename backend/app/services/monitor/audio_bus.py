"""
Canonical audio distribution bus for VoiceShield monitor pipeline.

Provides bounded fan-out from a single audio producer (WebRTC ingest or
Replay engine) to multiple consumers (Caption pipeline, Security pipeline).

Each consumer gets its own bounded queue with configurable overflow policy.
The Security pipeline queue emits discontinuity events on overflow instead
of silently dropping audio.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Canonical audio representation in the monitor pipeline.

    All downstream consumers receive this exact dataclass.
    The samples are always 16 kHz, mono, float32 after canonical conversion.
    """
    samples: torch.Tensor          # 1D float32 tensor
    sample_offset: int             # Absolute sample position since session start
    sample_rate: int = 16000       # Always 16000 after canonical conversion
    wall_timestamp: float = 0.0    # Wall-clock time for latency measurement only
    session_id: str = ""           # Owning monitor session ID
    track_id: str = ""             # Monitored WebRTC track ID
    sequence_id: int = 0           # Monotonic chunk sequence index
    sample_start: int = 0          # Exact starting sample index
    sample_end: int = 0            # Exact ending sample index
    timestamp_start: float = 0.0   # Wall-clock chunk onset
    timestamp_end: float = 0.0     # Wall-clock chunk offset
    discontinuity: bool = False    # True if audio stream suffered packet loss/gap
    track_source: str = "LOCAL_MIC" # Explicit origin: LOCAL_MIC or REMOTE_PEER
    track_direction: str = "inbound"

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


class OverflowPolicy(Enum):
    """Queue overflow behavior for audio consumers."""
    DROP_OLDEST = "drop_oldest"           # Caption path: acceptable
    EMIT_DISCONTINUITY = "discontinuity"  # Security path: never silent drop


@dataclass
class DiscontinuityEvent:
    """Emitted when audio is lost due to queue overflow."""
    sample_start: int
    sample_end: int
    reason: str
    timestamp: float = field(default_factory=time.time)


class ConsumerQueue:
    """Bounded audio queue for a single consumer with overflow policy."""

    def __init__(
        self,
        name: str,
        maxsize: int = 200,
        policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        on_discontinuity: Optional[Callable[[DiscontinuityEvent], None]] = None,
    ):
        self.name = name
        self.maxsize = maxsize
        self.policy = policy
        self.on_discontinuity = on_discontinuity
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._dropped_count = 0
        self._total_received = 0
        self._overflow_count = 0

    async def put(self, chunk: AudioChunk):
        """Put an audio chunk into the consumer queue.

        If full:
        - DROP_OLDEST: removes oldest chunk, logs warning
        - EMIT_DISCONTINUITY: removes oldest, emits event, never silent
        """
        self._total_received += 1

        if self._queue.full():
            self._overflow_count += 1
            # Remove oldest
            try:
                dropped = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                dropped = None

            self._dropped_count += 1

            if self.policy == OverflowPolicy.EMIT_DISCONTINUITY:
                event = DiscontinuityEvent(
                    sample_start=dropped.sample_offset if dropped else 0,
                    sample_end=chunk.sample_offset,
                    reason=f"QUEUE_OVERFLOW in {self.name} "
                           f"(dropped={self._dropped_count}, "
                           f"depth={self.maxsize})",
                )
                logger.warning(
                    f"Audio discontinuity in {self.name}: "
                    f"samples {event.sample_start}-{event.sample_end} "
                    f"(total dropped: {self._dropped_count})"
                )
                if self.on_discontinuity:
                    self.on_discontinuity(event)
            else:
                if self._dropped_count % 50 == 1:
                    logger.warning(
                        f"Caption queue {self.name} overflow: "
                        f"dropped {self._dropped_count} chunks"
                    )

        await self._queue.put(chunk)

    async def get(self, timeout: float = 0.1) -> Optional[AudioChunk]:
        """Get the next audio chunk, with timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    @property
    def total_received(self) -> int:
        return self._total_received


class AudioBus:
    """Canonical audio distribution layer.

    One producer publishes AudioChunks.
    Multiple consumers subscribe with independent bounded queues.
    Caption and Security paths use different overflow policies.
    """

    def __init__(self):
        self._consumers: dict[str, ConsumerQueue] = {}
        self._total_published = 0
        self._sample_counter = 0

    def subscribe(
        self,
        name: str,
        maxsize: int = 200,
        policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        on_discontinuity: Optional[Callable[[DiscontinuityEvent], None]] = None,
    ) -> ConsumerQueue:
        """Register a consumer. Returns the consumer's queue."""
        if name in self._consumers:
            raise ValueError(f"Consumer '{name}' already subscribed")

        consumer = ConsumerQueue(
            name=name,
            maxsize=maxsize,
            policy=policy,
            on_discontinuity=on_discontinuity,
        )
        self._consumers[name] = consumer
        logger.info(f"AudioBus: consumer '{name}' subscribed (policy={policy.value})")
        return consumer

    def unsubscribe(self, name: str):
        """Remove a consumer."""
        if name in self._consumers:
            del self._consumers[name]
            logger.info(f"AudioBus: consumer '{name}' unsubscribed")

    async def publish(
        self,
        samples: torch.Tensor,
        wall_timestamp: float = None,
        session_id: str = "",
        track_id: str = "",
        discontinuity: bool = False,
        track_source: str = "LOCAL_MIC",
        track_direction: str = "inbound",
    ):
        """Publish a PCM chunk to all consumers with canonical timeline metadata.

        Args:
            samples: 1D float32 tensor at 16 kHz.
            wall_timestamp: Wall-clock time for latency measurement.
            session_id: Monitor session ID.
            track_id: Monitored track ID.
            discontinuity: True if gap/loss occurred.
            track_source: Explicit origin ("LOCAL_MIC" or "REMOTE_PEER").
            track_direction: Direction ("inbound" or "outbound").
        """
        if wall_timestamp is None:
            wall_timestamp = time.time()

        sample_len = len(samples)
        sample_start = self._sample_counter
        sample_end = sample_start + sample_len
        duration_s = sample_len / 16000.0

        chunk = AudioChunk(
            samples=samples,
            sample_offset=sample_start,
            sample_rate=16000,
            wall_timestamp=wall_timestamp,
            session_id=session_id,
            track_id=track_id,
            sequence_id=self._total_published,
            sample_start=sample_start,
            sample_end=sample_end,
            timestamp_start=wall_timestamp,
            timestamp_end=wall_timestamp + duration_s,
            discontinuity=discontinuity,
            track_source=track_source,
            track_direction=track_direction,
        )
        self._sample_counter = sample_end
        self._total_published += 1

        for consumer in self._consumers.values():
            await consumer.put(chunk)

    @property
    def total_published(self) -> int:
        return self._total_published

    @property
    def total_samples(self) -> int:
        return self._sample_counter

    @property
    def total_duration_seconds(self) -> float:
        return self._sample_counter / 16000.0

    def get_telemetry(self) -> dict:
        """Get bus telemetry for UI."""
        return {
            "total_published": self._total_published,
            "total_samples": self._sample_counter,
            "total_duration_s": round(self.total_duration_seconds, 2),
            "consumers": {
                name: {
                    "queue_depth": c.depth,
                    "total_received": c.total_received,
                    "dropped": c.dropped_count,
                    "queue_overflows": c.overflow_count,
                }
                for name, c in self._consumers.items()
            },
        }

    def reset(self):
        """Reset for a new session."""
        self._total_published = 0
        self._sample_counter = 0
        for name in list(self._consumers.keys()):
            self.unsubscribe(name)
