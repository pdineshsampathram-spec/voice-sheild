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

# Voice-Assistant Streaming & Endpointing Constants
SPEECH_ONSET_FRAMES = 3          # Require 3 consecutive speech frames (~96ms) to trigger onset
PAUSE_ENDPOINT_FRAMES = 16       # ~512ms of continuous silence to trigger End-of-Utterance (EOU)
MIN_UTTERANCE_SAMPLES = 4800     # 0.30s at 16kHz minimum valid utterance duration
MIN_VOICED_SPEECH_SAMPLES = 2560 # ~160ms (at least 5 frames) of verified voiced speech to discard clicks/coughs
MIN_UTTERANCE_RMS = 0.0010       # Rejects pure digital silence / line noise while preserving quiet speech
MAX_SEGMENT_SAMPLES = 240000     # 15s at 16kHz max segment before forced finalization
PARTIAL_INTERVAL_SAMPLES = 12800 # ~800ms of incremental speech to dispatch a partial hypothesis


@dataclass
class CaptionEvent:
    """Event emitted by the caption pipeline with verified runtime lineage."""
    type: str = "caption"
    session_id: str = ""
    track_id: str = ""
    utterance_id: int = 0
    event_id: str = ""
    seq: int = 0
    sequence: int = 0
    text: str = ""
    is_partial: bool = True
    is_final: bool = False
    sample_start: int = 0
    sample_end: int = 0
    caption_start: int = 0
    caption_end: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    timestamp: float = 0.0
    queue_enter_time: float = 0.0
    inference_start: float = 0.0
    inference_end: float = 0.0
    latency_ms: float = 0.0
    analysis_lag_ms: float = 0.0
    model_name: str = ""
    asr_provider: str = ""
    asr_model: str = ""
    asr_backend: str = ""
    vad_speech_active: bool = False
    confidence: float = 0.0
    input_duration_s: float = 0.0


@dataclass
class _ASRTask:
    """Internal task dispatched to the detached ASR worker."""
    segment: torch.Tensor
    sample_start: int
    sample_end: int
    dispatch_time: float
    audio_end_time: float = 0.0
    is_partial: bool = False
    utterance_id: int = 0
    session_id: str = ""
    track_id: str = ""


class CaptionPipeline:
    """Voice-assistant style streaming caption pipeline with endpointing and decoupled ASR worker.

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
        session_id: str = "",
        track_id: str = "",
    ):
        self._queue = audio_queue
        self._asr = asr_provider
        self._on_event = on_event
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._session_id = session_id
        self._track_id = track_id

        # VAD state — independent instance
        self._vad_model = None
        self._vad_loaded = False
        self._vad_accumulator = torch.tensor([], dtype=torch.float32)
        self._last_vad_prob: float = 0.0

        # Utterance lineage & exact monotonic sample tracking
        self._utterance_counter: int = 0
        self._current_utterance_id: int = 0
        self._last_finalized_utterance_id: int = 0
        self._monotonic_samples_processed: int = 0

        # Voice-assistant endpointing & hysteresis state
        self._in_speech = False
        self._consecutive_speech_count = 0
        self._consecutive_pause_speech_count = 0
        self._onset_frames_buffer: list[torch.Tensor] = []
        self._silence_frame_count = 0
        self._speech_start_offset: int = 0
        self._speech_buffer: list[torch.Tensor] = []
        self._total_speech_samples = 0
        self._actual_speech_samples = 0
        self._last_partial_sample_count = 0

        # Endpointing configuration
        self._pause_endpoint_frames = 11  # ~352ms of silence to trigger EOU
        self._speech_resume_frames = 2    # Require 2 frames (64ms) to break an in-progress pause

        # Detached ASR work queue to decouple audio ingest from inference
        self._asr_queue: asyncio.Queue[_ASRTask] = asyncio.Queue(maxsize=50)

        # Telemetry counters
        self._total_asr_jobs_queued = 0
        self._total_asr_jobs_completed = 0
        self._total_partials_emitted = 0
        self._total_finals_emitted = 0
        self._stale_partials_dropped = 0
        self._pipeline_errors = 0
        self._last_analysis_lag_ms: float = 0.0
        self._max_analysis_lag_ms: float = 0.0

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

        # Allow pending final segment to complete inference before teardown
        try:
            await asyncio.wait_for(self._asr_queue.join(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass

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

                frame_sample_start = self._monotonic_samples_processed
                self._monotonic_samples_processed += VAD_FRAME_SAMPLES

                is_speech_frame = self._evaluate_vad_frame(frame)
                self._handle_vad_frame(is_speech_frame, frame, frame_sample_start)

            # Max segment duration guard
            if self._in_speech and self._total_speech_samples >= MAX_SEGMENT_SAMPLES:
                self._dispatch_current_segment()

    def _evaluate_vad_frame(self, frame: torch.Tensor) -> bool:
        """Run Silero VAD on exactly 512 samples at 16kHz."""
        if not self._vad_loaded:
            return True  # Fallback: assume speech if VAD unavailable

        try:
            prob = self._vad_model(frame, 16000).item()
            self._last_vad_prob = prob
            return prob > 0.5
        except Exception as e:
            logger.debug(f"Silero VAD frame evaluation exception: {e}")
            self._last_vad_prob = 0.0
            return True

    def _handle_vad_frame(self, is_speech: bool, frame: torch.Tensor, frame_sample_start: int):
        """Manage speech segment accumulation with voice-assistant streaming & endpointing."""
        if is_speech:
            if not self._in_speech:
                self._consecutive_speech_count += 1
                self._onset_frames_buffer.append(frame)
                if self._consecutive_speech_count >= SPEECH_ONSET_FRAMES:
                    # Verified Speech Onset
                    self._in_speech = True
                    self._utterance_counter += 1
                    self._current_utterance_id = self._utterance_counter
                    # Prepend onset frames: start sample is offset back by (len(onset) - 1) frames
                    self._speech_start_offset = max(0, frame_sample_start - (len(self._onset_frames_buffer) - 1) * VAD_FRAME_SAMPLES)
                    self._speech_buffer = list(self._onset_frames_buffer)
                    self._onset_frames_buffer = []
                    self._total_speech_samples = sum(len(f) for f in self._speech_buffer)
                    self._actual_speech_samples = self._total_speech_samples
                    self._last_partial_sample_count = self._total_speech_samples
                    self._silence_frame_count = 0
                    self._consecutive_pause_speech_count = 0

                    self._on_event(CaptionEvent(
                        type="vad",
                        session_id=self._session_id,
                        track_id=self._track_id,
                        utterance_id=self._current_utterance_id,
                        vad_speech_active=True,
                        sample_start=self._speech_start_offset,
                        sample_end=self._speech_start_offset + self._total_speech_samples,
                    ))
            else:
                # Inside speech: require at least SPEECH_RESUME_FRAMES to break a pause
                self._consecutive_pause_speech_count += 1
                if self._consecutive_pause_speech_count >= self._speech_resume_frames:
                    self._silence_frame_count = 0

                self._speech_buffer.append(frame)
                self._total_speech_samples += len(frame)
                self._actual_speech_samples += len(frame)

                # Incremental partial dispatch if sufficient new speech arrived
                if (self._total_speech_samples - self._last_partial_sample_count) >= PARTIAL_INTERVAL_SAMPLES:
                    self._last_partial_sample_count = self._total_speech_samples
                    self._dispatch_partial_segment()

        else:
            # Silence frame
            self._consecutive_speech_count = 0
            self._consecutive_pause_speech_count = 0
            if not self._in_speech:
                # Keep rolling onset buffer limited to SPEECH_ONSET_FRAMES
                self._onset_frames_buffer.append(frame)
                if len(self._onset_frames_buffer) > SPEECH_ONSET_FRAMES:
                    self._onset_frames_buffer.pop(0)
            else:
                # Inside speech: accumulate pause frames until endpoint threshold
                self._speech_buffer.append(frame)
                self._total_speech_samples += len(frame)
                self._silence_frame_count += 1

                if self._silence_frame_count >= self._pause_endpoint_frames:
                    # END-OF-UTTERANCE (EOU) reached
                    # Prune excess trailing silence frames: retain at most 3 padding frames (~96ms)
                    excess_silence = max(0, self._silence_frame_count - 3)
                    if excess_silence > 0 and len(self._speech_buffer) > excess_silence:
                        self._speech_buffer = self._speech_buffer[:-excess_silence]
                        self._total_speech_samples = sum(len(f) for f in self._speech_buffer)

                    self._dispatch_final_segment()
                    self._reset_vad_state()
                    self._on_event(CaptionEvent(
                        type="vad",
                        session_id=self._session_id,
                        track_id=self._track_id,
                        utterance_id=self._last_finalized_utterance_id,
                        vad_speech_active=False,
                        sample_start=self._speech_start_offset,
                        sample_end=self._speech_start_offset + self._total_speech_samples,
                    ))

    def _purge_stale_partials(self):
        """Purge unstarted partial tasks from the worker queue."""
        if self._asr_queue.empty():
            return
        non_partials = []
        purged = 0
        while not self._asr_queue.empty():
            try:
                t = self._asr_queue.get_nowait()
                if not t.is_partial:
                    non_partials.append(t)
                else:
                    purged += 1
            except asyncio.QueueEmpty:
                break
        for t in non_partials:
            self._asr_queue.put_nowait(t)
        if purged > 0:
            self._stale_partials_dropped += purged

    def _dispatch_partial_segment(self):
        """Dispatch partial speech segment for real-time live preview."""
        if not self._speech_buffer or self._total_speech_samples < MIN_UTTERANCE_SAMPLES:
            return

        # Coalesce: remove any unstarted partial task already in queue
        self._purge_stale_partials()

        segment = torch.cat(self._speech_buffer).clone().detach()
        task = _ASRTask(
            segment=segment,
            sample_start=self._speech_start_offset,
            sample_end=self._speech_start_offset + len(segment),
            dispatch_time=time.time(),
            audio_end_time=self._speech_start_offset + len(segment),
            is_partial=True,
            utterance_id=self._current_utterance_id,
            session_id=self._session_id,
            track_id=self._track_id,
        )
        try:
            self._asr_queue.put_nowait(task)
            self._total_asr_jobs_queued += 1
        except asyncio.QueueFull:
            pass  # Dropping partials under load is safe

    def _dispatch_final_segment(self):
        """Hand off completed utterance to detached ASR worker queue."""
        if not self._speech_buffer:
            self._in_speech = False
            return

        # Purge all pending partial tasks immediately on finalization
        self._purge_stale_partials()

        segment = torch.cat(self._speech_buffer).clone().detach()
        start_offset = self._speech_start_offset
        end_offset = start_offset + len(segment)
        uid = self._current_utterance_id
        self._last_finalized_utterance_id = uid

        # Reset utterance state immediately
        actual_voiced = self._actual_speech_samples
        self._in_speech = False
        self._speech_buffer = []
        self._total_speech_samples = 0
        self._actual_speech_samples = 0
        self._silence_frame_count = 0
        self._consecutive_speech_count = 0
        self._consecutive_pause_speech_count = 0
        self._onset_frames_buffer = []
        self._last_partial_sample_count = 0

        # Acoustic gating to prevent hallucinating on silence or low-energy clicks
        duration_s = len(segment) / 16000.0
        rms = float(torch.sqrt(torch.mean(segment ** 2))) if len(segment) > 0 else 0.0

        if actual_voiced < MIN_VOICED_SPEECH_SAMPLES:
            logger.debug(
                f"[CaptionPipeline] Discarding non-speech transient/click/cough: "
                f"only {actual_voiced} voiced samples ({actual_voiced/16000:.2f}s < {MIN_VOICED_SPEECH_SAMPLES/16000:.2f}s)"
            )
            return

        if len(segment) < MIN_UTTERANCE_SAMPLES:
            logger.debug(
                f"[CaptionPipeline] Discarding short segment: "
                f"{len(segment)} samples ({duration_s:.2f}s < {MIN_UTTERANCE_SAMPLES/16000:.2f}s)"
            )
            return

        if rms < MIN_UTTERANCE_RMS:
            logger.debug(
                f"[CaptionPipeline] Discarding low-energy silence/noise segment: "
                f"RMS {rms:.5f} < {MIN_UTTERANCE_RMS}"
            )
            return

        task = _ASRTask(
            segment=segment,
            sample_start=start_offset,
            sample_end=end_offset,
            dispatch_time=time.time(),
            audio_end_time=end_offset,
            is_partial=False,
            utterance_id=uid,
            session_id=self._session_id,
            track_id=self._track_id,
        )

        try:
            self._asr_queue.put_nowait(task)
            self._total_asr_jobs_queued += 1
        except asyncio.QueueFull:
            self._pipeline_errors += 1
            logger.warning("CaptionPipeline ASR worker queue full; dropping segment")

    def _dispatch_current_segment(self):
        """Alias for final segment dispatch on shutdown or max duration."""
        self._dispatch_final_segment()

    async def _run_asr_worker(self):
        """Detached worker loop executing ASR inference without blocking ingest."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                task: _ASRTask = await self._asr_queue.get()
            except asyncio.CancelledError:
                break

            # Zero-trust queue lag check: discard obsolete partials before inference
            if task.is_partial:
                if task.utterance_id <= self._last_finalized_utterance_id or not self._asr_queue.empty():
                    self._stale_partials_dropped += 1
                    self._asr_queue.task_done()
                    continue

            inference_start = time.time()
            try:
                # Run ASR inference in thread pool
                result: ASRResult = await loop.run_in_executor(
                    self._executor,
                    self._asr.transcribe_segment,
                    task.segment,
                    task.sample_start,
                )
                inference_end = time.time()
                self._total_asr_jobs_completed += 1

                # Calculate analysis lag
                analysis_lag_ms = max(0.0, (inference_end - task.dispatch_time) * 1000.0)
                self._last_analysis_lag_ms = analysis_lag_ms
                if analysis_lag_ms > self._max_analysis_lag_ms:
                    self._max_analysis_lag_ms = analysis_lag_ms

                # Late partial drop check: if utterance finalized while inference was running, reject partial
                if task.is_partial and task.utterance_id <= self._last_finalized_utterance_id:
                    self._stale_partials_dropped += 1
                    continue

                if task.is_partial:
                    self._total_partials_emitted += 1
                else:
                    self._total_finals_emitted += 1

                logger.info(
                    f"[LIVE TRACE: ASR] task={'PARTIAL' if task.is_partial else 'FINAL'} "
                    f"utterance_id={task.utterance_id} track={task.track_id} "
                    f"samples=[{task.sample_start}..{task.sample_end}] "
                    f"({len(task.segment)/16000:.2f}s) | latency={result.latency_ms:.1f}ms "
                    f"lag={analysis_lag_ms:.1f}ms | transcript='{result.text.strip()}'"
                )

                if result.text.strip():
                    audio_dur = len(task.segment) / 16000.0
                    self._on_event(CaptionEvent(
                        type="caption",
                        session_id=task.session_id,
                        track_id=task.track_id,
                        utterance_id=task.utterance_id,
                        text=result.text.strip(),
                        is_partial=task.is_partial,
                        is_final=not task.is_partial,
                        sample_start=result.sample_start,
                        sample_end=result.sample_end,
                        caption_start=result.sample_start,
                        caption_end=result.sample_end,
                        timestamp_start=task.dispatch_time,
                        timestamp_end=inference_end,
                        timestamp=inference_end,
                        queue_enter_time=task.dispatch_time,
                        inference_start=inference_start,
                        inference_end=inference_end,
                        latency_ms=result.latency_ms,
                        analysis_lag_ms=round(analysis_lag_ms, 1),
                        model_name=result.model_name,
                        asr_provider=type(self._asr).__name__,
                        asr_model=getattr(self._asr, "model_name", result.model_name),
                        asr_backend=getattr(self._asr, "backend", "transformers"),
                        vad_speech_active=task.is_partial,
                        confidence=getattr(result, "confidence", 1.0),
                        input_duration_s=round(audio_dur, 3),
                    ))

            except Exception as e:
                self._pipeline_errors += 1
                logger.error(f"ASR inference error: {e}", exc_info=True)
                self._on_event(CaptionEvent(
                    type="caption_error",
                    session_id=task.session_id,
                    track_id=task.track_id,
                    utterance_id=task.utterance_id,
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
            "last_vad_probability": round(self._last_vad_prob, 4),
            "asr_jobs_queued": self._total_asr_jobs_queued,
            "asr_jobs_completed": self._total_asr_jobs_completed,
            "asr_queue_depth": self._asr_queue.qsize(),
            "pipeline_errors": self._pipeline_errors,
            "current_utterance_id": self._current_utterance_id,
            "last_finalized_utterance_id": self._last_finalized_utterance_id,
            "analysis_lag_ms": round(self._last_analysis_lag_ms, 1),
            "max_analysis_lag_ms": round(self._max_analysis_lag_ms, 1),
            "stale_partials_dropped": self._stale_partials_dropped,
        }

