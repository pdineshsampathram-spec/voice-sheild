"""
Caption pipeline — streaming ASR with independent VAD and non-blocking worker.

Consumes audio from the AudioBus, gates through its own VAD instance using a
512-sample accumulator, accumulates speech segments, and dispatches completed
segments to a detached async ASR worker queue.

CRITICAL INVARIANTS:
1. Audio consumer loop (_run) NEVER awaits ASR inference. It continuously
   drains the AudioBus consumer queue so audio chunks containing speech are
   never dropped.
2. Silero VAD is evaluated on exact 512-sample frames via an accumulator.
3. Silero VAD state is reset between utterances to avoid recurrent leakage.
4. Provisional heuristics (min duration, RMS noise floor) are explicitly labeled
   as provisional audio-quality heuristics, not universal speech requirements.
5. Emits partial hypotheses during speech and immutable final transcripts.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from app.services.monitor.audio_bus import ConsumerQueue, AudioChunk
from app.services.asr.asr_provider import ASRProvider, ASRResult

logger = logging.getLogger(__name__)

# Silero VAD frame size required at 16kHz
VAD_FRAME_SAMPLES = 512  # 32ms at 16kHz

# PROVISIONAL AUDIO-QUALITY HEURISTICS (Subject to benchmark validation)
PROVISIONAL_MIN_SPEECH_SAMPLES = 3200    # 0.20s at 16kHz (allows short words: "hello", "hi", "yes")
PROVISIONAL_MIN_RMS = 0.0015             # Noise floor guard to prevent hallucination on quiet noise
PROVISIONAL_SILENCE_FRAMES_OFFSET = 10  # ~320ms continuous silence to trigger segment end
MAX_SEGMENT_SAMPLES = 480000            # 30s at 16kHz max segment before forced finalization


@dataclass
class CaptionEvent:
    """Event emitted by the caption pipeline with verified runtime lineage."""
    type: str = "caption"
    text: str = ""
    is_partial: bool = True
    sample_start: int = 0
    sample_end: int = 0
    caption_start: int = 0
    caption_end: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    asr_provider: str = ""
    asr_model: str = ""
    asr_backend: str = ""
    vad_speech_active: bool = False


@dataclass
class _ASRTask:
    """Internal task dispatched to the detached ASR worker."""
    segment: torch.Tensor
    sample_start: int
    sample_end: int
    dispatch_time: float


class CaptionPipeline:
    """Streaming caption pipeline with independent VAD and decoupled ASR worker.

    Owns its own VAD instance — no shared mutable recurrent state
    with the security pipeline. If ASR stalls, it does NOT affect
    detection or drop incoming audio chunks.
    """

    def __init__(
        self,
        audio_queue: ConsumerQueue,
        asr_provider: ASRProvider,
        on_event: Callable[[CaptionEvent], None],
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self._queue = audio_queue
        self._asr = asr_provider
        self._on_event = on_event
        self._executor = executor or ThreadPoolExecutor(max_workers=1)

        # VAD state — independent instance
        self._vad_model = None
        self._vad_loaded = False
        self._vad_accumulator = torch.tensor([], dtype=torch.float32)

        # Hysteresis state for VAD speech boundary detection
        self._in_speech = False
        self._silence_frame_count = 0
        self._speech_start_offset: int = 0
        self._speech_buffer: list[torch.Tensor] = []
        self._total_speech_samples = 0

        # Detached ASR work queue to decouple audio ingest from inference
        self._asr_queue: asyncio.Queue[_ASRTask] = asyncio.Queue(maxsize=50)

        # Telemetry counters
        self._total_asr_jobs_queued = 0
        self._total_asr_jobs_completed = 0
        self._pipeline_errors = 0

        # Running tasks
        self._running = False
        self._consume_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the caption pipeline and worker tasks."""
        self._load_vad()
        self._running = True
        self._consume_task = asyncio.create_task(self._run_consumer())
        self._worker_task = asyncio.create_task(self._run_asr_worker())
        logger.info("CaptionPipeline started (consumer + detached ASR worker)")

    async def stop(self):
        """Stop the caption pipeline."""
        self._running = False
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass

        # Finalize any pending accumulated speech
        if self._speech_buffer:
            self._dispatch_current_segment()

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("CaptionPipeline stopped")

    def _load_vad(self):
        """Load independent Silero VAD model instance."""
        try:
            self._vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self._vad_loaded = True
            logger.info("CaptionPipeline: VAD-A loaded (independent instance)")
        except Exception as e:
            logger.error(f"CaptionPipeline: VAD-A load failed: {e}")
            self._vad_loaded = False

    def _reset_vad_state(self):
        """Reset Silero VAD internal state between utterances."""
        if self._vad_loaded and hasattr(self._vad_model, 'reset_states'):
            try:
                self._vad_model.reset_states()
            except Exception as e:
                logger.debug(f"Silero VAD reset_states error: {e}")

    async def _run_consumer(self):
        """Continuous audio chunk consumer loop.

        CRITICAL: Never await ASR here! This loop must keep draining
        self._queue at wire-speed so zero audio chunks are dropped.
        """
        while self._running:
            chunk: Optional[AudioChunk] = await self._queue.get(timeout=0.05)
            if chunk is None:
                continue

            # Append chunk samples to VAD accumulator
            self._vad_accumulator = torch.cat([self._vad_accumulator, chunk.samples])

            # Process all complete 512-sample frames in the accumulator
            while len(self._vad_accumulator) >= VAD_FRAME_SAMPLES:
                frame = self._vad_accumulator[:VAD_FRAME_SAMPLES]
                self._vad_accumulator = self._vad_accumulator[VAD_FRAME_SAMPLES:]

                is_speech_frame = self._evaluate_vad_frame(frame)
                self._handle_vad_frame(is_speech_frame, frame, chunk.sample_offset)

            # Max segment duration guard
            if self._in_speech and self._total_speech_samples >= MAX_SEGMENT_SAMPLES:
                self._dispatch_current_segment()

    def _evaluate_vad_frame(self, frame: torch.Tensor) -> bool:
        """Run Silero VAD on exactly 512 samples at 16kHz."""
        if not self._vad_loaded:
            return True  # Fallback: assume speech if VAD unavailable

        try:
            prob = self._vad_model(frame, 16000).item()
            return prob > 0.5
        except Exception as e:
            logger.debug(f"Silero VAD frame evaluation exception: {e}")
            return True

    def _handle_vad_frame(self, is_speech: bool, frame: torch.Tensor, current_sample_offset: int):
        """Manage speech segment accumulation with hysteresis."""
        if is_speech:
            self._silence_frame_count = 0
            if not self._in_speech:
                # Speech onset
                self._in_speech = True
                self._speech_start_offset = current_sample_offset
                self._speech_buffer = []
                self._total_speech_samples = 0

                self._on_event(CaptionEvent(
                    type="vad",
                    vad_speech_active=True,
                    sample_start=self._speech_start_offset,
                    sample_end=self._speech_start_offset + len(frame),
                ))

            self._speech_buffer.append(frame)
            self._total_speech_samples += len(frame)

            # Emit partial status (speech in progress)
            self._on_event(CaptionEvent(
                type="caption",
                text="",
                is_partial=True,
                sample_start=self._speech_start_offset,
                sample_end=self._speech_start_offset + self._total_speech_samples,
                vad_speech_active=True,
            ))

        else:
            if self._in_speech:
                # Keep buffering a small amount of trailing silence for smooth acoustics
                self._speech_buffer.append(frame)
                self._total_speech_samples += len(frame)
                self._silence_frame_count += 1

                # Check if silence sustained long enough to trigger speech offset
                if self._silence_frame_count >= PROVISIONAL_SILENCE_FRAMES_OFFSET:
                    self._dispatch_current_segment()
                    self._reset_vad_state()
                    self._on_event(CaptionEvent(
                        type="vad",
                        vad_speech_active=False,
                        sample_start=self._speech_start_offset,
                        sample_end=self._speech_start_offset + self._total_speech_samples,
                    ))

    def _dispatch_current_segment(self):
        """Hand off accumulated speech to detached ASR worker queue."""
        if not self._speech_buffer:
            self._in_speech = False
            return

        segment = torch.cat(self._speech_buffer)
        start_offset = self._speech_start_offset
        end_offset = start_offset + len(segment)

        # Reset accumulator state immediately
        self._in_speech = False
        self._speech_buffer = []
        self._total_speech_samples = 0
        self._silence_frame_count = 0

        # Check PROVISIONAL AUDIO-QUALITY HEURISTICS
        duration_s = len(segment) / 16000.0
        rms = float(torch.sqrt(torch.mean(segment ** 2))) if len(segment) > 0 else 0.0

        if len(segment) < PROVISIONAL_MIN_SPEECH_SAMPLES:
            logger.debug(
                f"[Provisional Heuristic] Discarding short segment: "
                f"{len(segment)} samples ({duration_s:.2f}s < {PROVISIONAL_MIN_SPEECH_SAMPLES/16000:.2f}s)"
            )
            return

        if rms < PROVISIONAL_MIN_RMS:
            logger.debug(
                f"[Provisional Heuristic] Discarding low-energy noise segment: "
                f"RMS {rms:.5f} < {PROVISIONAL_MIN_RMS}"
            )
            return

        task = _ASRTask(
            segment=segment,
            sample_start=start_offset,
            sample_end=end_offset,
            dispatch_time=time.time(),
        )

        try:
            self._asr_queue.put_nowait(task)
            self._total_asr_jobs_queued += 1
        except asyncio.QueueFull:
            self._pipeline_errors += 1
            logger.warning("CaptionPipeline ASR worker queue full; dropping segment")

    async def _run_asr_worker(self):
        """Detached worker loop executing ASR inference without blocking ingest."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                task: _ASRTask = await self._asr_queue.get()
            except asyncio.CancelledError:
                break

            try:
                # Run ASR inference in thread pool
                result: ASRResult = await loop.run_in_executor(
                    self._executor,
                    self._asr.transcribe_segment,
                    task.segment,
                    task.sample_start,
                )
                self._total_asr_jobs_completed += 1

                logger.info(
                    f"[LIVE TRACE: ASR-{self._total_asr_jobs_completed}] AUDIO: samples=[{task.sample_start}..{task.sample_end}] "
                    f"({len(task.segment)/16000:.2f}s) | ASR: provider={type(self._asr).__name__} "
                    f"model={getattr(self._asr, 'model_name', result.model_name)} latency={result.latency_ms:.1f}ms "
                    f"transcript='{result.text.strip()}'"
                )

                if result.text.strip():
                    self._on_event(CaptionEvent(
                        type="caption",
                        text=result.text.strip(),
                        is_partial=False,
                        sample_start=result.sample_start,
                        sample_end=result.sample_end,
                        caption_start=result.sample_start,
                        caption_end=result.sample_end,
                        latency_ms=result.latency_ms,
                        model_name=result.model_name,
                        asr_provider=type(self._asr).__name__,
                        asr_model=getattr(self._asr, "model_name", result.model_name),
                        asr_backend=getattr(self._asr, "backend", "ctranslate2"),
                        vad_speech_active=False,
                    ))

            except Exception as e:
                self._pipeline_errors += 1
                logger.error(f"ASR inference error: {e}", exc_info=True)
                self._on_event(CaptionEvent(
                    type="caption_error",
                    text=f"ASR error: {e}",
                    sample_start=task.sample_start,
                    sample_end=task.sample_end,
                ))
            finally:
                self._asr_queue.task_done()

    def get_telemetry(self) -> dict:
        """Get caption pipeline telemetry."""
        return {
            "vad_loaded": self._vad_loaded,
            "vad_speech_active": self._in_speech,
            "asr_jobs_queued": self._total_asr_jobs_queued,
            "asr_jobs_completed": self._total_asr_jobs_completed,
            "asr_queue_depth": self._asr_queue.qsize(),
            "pipeline_errors": self._pipeline_errors,
        }
