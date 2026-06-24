"""
core/frame_buffer.py
--------------------
Asynchronous CPU/GPU-prefetch ring buffer for smooth video scrubbing and tracking.

Frames originate as CPU BGR ``uint8`` arrays from ``VideoReader``. The buffer
retains that decoded CPU image and lazily attaches a GPU copy when a GPU-based
consumer requests it. During point-colour-only batch runs, GPU prefetch may be
disabled so the fast tracker never performs an unnecessary upload or download.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from time import perf_counter
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cupy as cp
except Exception:  # CPU Point Fast / native D3D11 mode does not require it.
    cp = None
import numpy as np

from core.video_reader import VideoReader
from core.workspace_cache import (
    WorkspaceDiskCache,
    DEFAULT_RAM_LOOKAHEAD_BYTES,
    DEFAULT_LIVE_LOOKAHEAD_FRAMES,
    DEFAULT_NVDEC_TRANSFER_MODE,
    decode_single_workspace_frame,
    iter_live_workspace_frames,
)
from core.workspace_types import WorkspaceFrame, WorkspaceRect


@dataclass
class _CachedFrame:
    cpu: np.ndarray
    gpu: Optional[object] = None


FramePair = Tuple[np.ndarray, object]
PLAYBACK_NVDEC_TRANSFER_MODE = "event_chained"
PLAYBACK_LOOKAHEAD_FRAMES = 96
# Native D3D11 decoders can take a moment to open their next session. Start
# the next range halfway through the present one so this handoff is invisible.
PLAYBACK_REFILL_WATERMARK_FRAMES = 64
PLAYBACK_START_PREBUFFER_FRAMES = 1
_VALID_NVDEC_TRANSFER_MODES = {"fast_async", "event_chained", "strict_sync"}


class FrameBuffer:
    """LRU cache retaining decoded CPU frames and optional GPU copies."""

    def __init__(
        self,
        reader: VideoReader,
        buffer_size: int = 32,
        prefetch_ahead: int = 16,
    ) -> None:
        self._reader = reader
        self._buffer_size = buffer_size
        self._prefetch_ahead = prefetch_ahead

        self._cache: OrderedDict[int, _CachedFrame] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._gpu_available = cp is not None
        self._upload_stream = cp.cuda.Stream(non_blocking=True) if self._gpu_available else None

        # Optional prepared analysis-workspace sources. They are used only for
        # CPU-only batch processing; interactive display still reads full frames.
        self._analysis_cache: Optional[WorkspaceDiskCache] = None
        self._analysis_workspace: Optional[WorkspaceRect] = None
        self._analysis_source_mode: str = "direct"  # direct | live | disk
        self._analysis_prefer_nvdec: bool = True
        self._analysis_decoder_preference: str = "auto"
        self._analysis_gpu_id: int = 0
        self._analysis_cache_ram_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES
        self._analysis_live_lookahead_frames: int = DEFAULT_LIVE_LOOKAHEAD_FRAMES
        self._analysis_nvdec_transfer_mode: str = DEFAULT_NVDEC_TRANSFER_MODE

        # Optional post-tracking playback prebuffer.  When enabled, a background
        # NVDEC-preferred live decoder fills a small RAM cache ahead of the
        # current viewing frame so result review/scrubbing is less laggy.
        self._playback_prebuffer_enabled = False
        self._playback_lookahead_frames = PLAYBACK_LOOKAHEAD_FRAMES
        self._playback_prefer_nvdec = True
        self._playback_decoder_preference = "auto"
        self._playback_gpu_id = 0
        self._playback_nvdec_transfer_mode = PLAYBACK_NVDEC_TRANSFER_MODE
        self._playback_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        # During batch tracking we temporarily restore the original simple
        # RAM topology: no preview/prefetch/playback frame users, only the
        # dedicated batch frame source.  These fields remember the playback
        # state so it can be restored after the batch finishes.
        self._batch_exclusive_active = False
        self._saved_playback_state: Optional[tuple[bool, int, bool, int, str, str]] = None
        self._playback_lock = threading.Lock()
        self._playback_stop = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_range: tuple[int, int] = (-1, -1)
        self._playback_generation = 0

        # Preserve prior GPU-prefetch behavior by default; TrackerManager
        # disables it for CPU-only point-colour batch execution.
        self._gpu_prefetch_enabled = self._gpu_available
        # Interactive prefetch is paused during offline batch decoding. Batch
        # uses its own sequential VideoReader so it cannot fight the scrubber
        # reader with repeated seek/decode operations.
        self._prefetch_paused = threading.Event()

        self._prefetch_head = 0
        self._prefetch_target = 0
        self._stop_event = threading.Event()
        self._seek_event = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name='FrameBuffer-Prefetch',
        )
        self._prefetch_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_analysis_cache(
        self,
        cache: Optional[WorkspaceDiskCache],
        *,
        ram_lookahead_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES,
    ) -> None:
        """Attach a prepared workspace cache for CPU-only batch runs."""
        self._analysis_cache = cache
        if cache is not None:
            self._analysis_workspace = cache.workspace
            self._analysis_source_mode = "disk"
        self._analysis_cache_ram_bytes = max(1, int(ram_lookahead_bytes))

    def set_live_analysis_workspace(
        self,
        workspace: Optional[WorkspaceRect],
        *,
        ram_lookahead_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES,
        max_lookahead_frames: int = DEFAULT_LIVE_LOOKAHEAD_FRAMES,
        prefer_nvdec: bool = True,
        gpu_id: int = 0,
        nvdec_transfer_mode: str = DEFAULT_NVDEC_TRANSFER_MODE,
        decoder_preference: str = "auto",
    ) -> None:
        """Use a RAM-only live decoder/crop queue for CPU-only batch runs."""
        self._analysis_workspace = workspace
        self._analysis_source_mode = "live" if workspace is not None else "direct"
        self._analysis_prefer_nvdec = bool(prefer_nvdec)
        self._analysis_decoder_preference = str(decoder_preference or "auto")
        self._analysis_gpu_id = int(gpu_id)
        self._analysis_cache = None
        self._analysis_cache_ram_bytes = max(1, int(ram_lookahead_bytes))
        self._analysis_live_lookahead_frames = max(1, int(max_lookahead_frames))
        self._analysis_nvdec_transfer_mode = str(nvdec_transfer_mode or DEFAULT_NVDEC_TRANSFER_MODE)

    def clear_analysis_cache(self) -> None:
        """Disable prepared-workspace use while preserving files on disk."""
        self._analysis_cache = None
        self._analysis_workspace = None
        self._analysis_source_mode = "direct"

    def set_playback_prebuffer_enabled(
        self,
        enabled: bool,
        *,
        lookahead_frames: int = PLAYBACK_LOOKAHEAD_FRAMES,
        prefer_nvdec: bool = True,
        gpu_id: int = 0,
        nvdec_transfer_mode: str = PLAYBACK_NVDEC_TRANSFER_MODE,
        decoder_preference: str = "auto",
    ) -> None:
        """Enable/disable NVDEC-preferred RAM prebuffer for result review.

        This is only for interactive video viewing after tracking. Batch
        tracking continues to use its dedicated sequential/live workspace path.
        """
        self._playback_prebuffer_enabled = bool(enabled)
        self._playback_lookahead_frames = max(1, int(lookahead_frames))
        self._playback_prefer_nvdec = bool(prefer_nvdec)
        self._playback_decoder_preference = str(decoder_preference or "auto")
        self._playback_gpu_id = int(gpu_id)
        mode = str(nvdec_transfer_mode or PLAYBACK_NVDEC_TRANSFER_MODE).strip().lower().replace("-", "_")
        self._playback_nvdec_transfer_mode = mode if mode in _VALID_NVDEC_TRANSFER_MODES else PLAYBACK_NVDEC_TRANSFER_MODE
        if not enabled:
            self._stop_playback_prebuffer()
            with self._playback_lock:
                self._playback_cache.clear()

    def start_playback_prebuffer(self, start_index: int) -> None:
        """Begin sequential review decoding at ``start_index`` without a UI read."""
        self._schedule_playback_prebuffer(int(start_index))

    def get_playback_frame(self, index: int) -> Optional[np.ndarray]:
        """Return a review frame only when it is already in the playback cache.

        Playback must never make a synchronous ``VideoReader.read`` call from
        the Qt thread: that blocks rendering and also fights the sequential
        decoder worker.  A cache miss simply holds the displayed frame until
        the worker catches up.
        """
        if self._batch_exclusive_active or not self._playback_prebuffer_enabled:
            return None
        index = int(index)
        with self._playback_lock:
            cached = self._playback_cache.get(index)
            if cached is not None:
                self._playback_cache.move_to_end(index)
        if cached is not None:
            self._maybe_refill_playback_prebuffer(index)
            return cached
        self._schedule_playback_prebuffer(index)
        return None

    @property
    def analysis_cache(self) -> Optional[WorkspaceDiskCache]:
        return self._analysis_cache

    @property
    def analysis_source_mode(self) -> str:
        return self._analysis_source_mode

    def get_analysis_frame(self, index: int) -> Optional[WorkspaceFrame]:
        """Return a workspace frame if an analysis workspace source is active."""
        cache = self._analysis_cache
        if self._analysis_source_mode == "disk" and cache is not None:
            if not cache.covers_range(int(index), int(index) + 1):
                return None
            try:
                return cache.get_frame(int(index))
            except Exception:
                return None
        if self._analysis_source_mode == "live" and self._analysis_workspace is not None:
            try:
                decoded = decode_single_workspace_frame(
                    video_path=self._reader.path,
                    workspace=self._analysis_workspace,
                    frame_index=int(index),
                    prefer_nvdec=self._analysis_prefer_nvdec,
                    gpu_id=self._analysis_gpu_id,
                    decoder_preference=self._analysis_decoder_preference,
                )
                return decoded.frame
            except Exception:
                return None
        return None

    def set_gpu_prefetch_enabled(self, enabled: bool) -> None:
        """Control whether newly prefetched interactive frames use GPU memory."""
        self._gpu_prefetch_enabled = bool(enabled)
        self._seek_event.set()

    def begin_batch(self) -> None:
        """Enter exclusive batch-frame mode.

        The old fast/reliable live-RAM tracker effectively had one active RAM
        frame path during tracking.  Newer UI features add preview and playback
        prebuffers, which are useful after a run but should not compete with or
        hold references to frames while batch tracking is active.  Batch mode
        therefore disables those interactive RAM users, clears old frame caches,
        and leaves only ``iter_batch_frames`` as the active frame source.
        """
        if not self._batch_exclusive_active:
            self._saved_playback_state = (
                bool(self._playback_prebuffer_enabled),
                int(self._playback_lookahead_frames),
                bool(self._playback_prefer_nvdec),
                int(self._playback_gpu_id),
                str(self._playback_nvdec_transfer_mode),
                str(self._playback_decoder_preference),
            )
        self._batch_exclusive_active = True
        self._playback_prebuffer_enabled = False
        self._stop_playback_prebuffer()
        with self._playback_lock:
            self._playback_cache.clear()
        with self._cache_lock:
            self._cache.clear()
        self._prefetch_paused.set()
        self._seek_event.set()

    def end_batch(self) -> None:
        """Leave exclusive batch-frame mode and restore interactive viewing."""
        self._batch_exclusive_active = False
        if self._saved_playback_state is not None:
            enabled, lookahead, prefer_nvdec, gpu_id, transfer_mode, decoder_preference = self._saved_playback_state
            self._playback_prebuffer_enabled = bool(enabled)
            self._playback_lookahead_frames = int(lookahead)
            self._playback_prefer_nvdec = bool(prefer_nvdec)
            self._playback_gpu_id = int(gpu_id)
            self._playback_nvdec_transfer_mode = str(transfer_mode or PLAYBACK_NVDEC_TRANSFER_MODE)
            self._playback_decoder_preference = str(decoder_preference or "auto")
            self._saved_playback_state = None
        self._prefetch_paused.clear()
        self._prefetch_head = self._prefetch_target
        self._seek_event.set()

    def iter_batch_frames(
        self,
        start: int,
        end: int,
        *,
        needs_gpu: bool,
        include_timing: bool = False,
        allow_workspace_cache: bool = True,
    ):
        """Yield sequential batch frames without using the scrub/prefetch cache.

        When ``include_timing`` is true, each yielded item additionally contains
        a dictionary with sequential decode and optional CPU->GPU upload timing.
        This is used only by opt-in profiling and avoids affecting normal runs.
        """
        # A cropped cache cannot satisfy full-frame GPU trackers. Use it only
        # when all active trackers accept CPU frames.
        cache = self._analysis_cache if (
            allow_workspace_cache
            and self._analysis_source_mode == "disk"
            and self._analysis_cache is not None
            and not needs_gpu
        ) else None
        if cache is not None and cache.covers_range(start, end):
            iterator = iter(cache.iter_frames_prefetched(
                start, end, ram_budget_bytes=self._analysis_cache_ram_bytes
            ))
            while True:
                read_start = perf_counter() if include_timing else 0.0
                try:
                    frame_index, frame_cpu = next(iterator)
                except StopIteration:
                    return
                read_ms = (perf_counter() - read_start) * 1000.0 if include_timing else 0.0
                if include_timing:
                    yield frame_index, frame_cpu, None, {
                        "decode_ms": 0.0,
                        "cache_read_ms": float(read_ms),
                        "gpu_upload_ms": 0.0,
                        "frame_source": "workspace_cache",
                    }
                else:
                    yield frame_index, frame_cpu, None
            return

        live_workspace = self._analysis_workspace if (
            allow_workspace_cache
            and self._analysis_source_mode == "live"
            and self._analysis_workspace is not None
            and not needs_gpu
        ) else None
        if live_workspace is not None:
            iterator = iter_live_workspace_frames(
                video_path=self._reader.path,
                workspace=live_workspace,
                start=start,
                end=end,
                prefer_nvdec=self._analysis_prefer_nvdec,
                gpu_id=self._analysis_gpu_id,
                ram_lookahead_bytes=self._analysis_cache_ram_bytes,
                max_lookahead_frames=self._analysis_live_lookahead_frames,
                nvdec_transfer_mode=self._analysis_nvdec_transfer_mode,
                decoder_preference=self._analysis_decoder_preference,
            )
            for frame_index, frame_cpu, source_metrics in iterator:
                if include_timing:
                    metrics = {
                        "decode_ms": 0.0,
                        "cache_read_ms": 0.0,
                        "gpu_upload_ms": 0.0,
                        "frame_source": source_metrics.get("frame_source", "live_ram"),
                        "live_backend": source_metrics.get("live_backend", "unknown"),
                        "live_backend_message": source_metrics.get("live_backend_message", ""),
                        "live_lookahead_capacity_frames": source_metrics.get("live_lookahead_capacity_frames", ""),
                        "live_start_prebuffer_frames": source_metrics.get("live_start_prebuffer_frames", ""),
                        "live_buffer_mode": source_metrics.get("live_buffer_mode", ""),
                        "source_frame_index": source_metrics.get("source_frame_index", ""),
                        "workspace_crc32": source_metrics.get("workspace_crc32", ""),
                        "workspace_crc32_repeat_n_minus_1": source_metrics.get("workspace_crc32_repeat_n_minus_1", ""),
                        "workspace_crc32_repeat_n_minus_2": source_metrics.get("workspace_crc32_repeat_n_minus_2", ""),
                        "workspace_mean": source_metrics.get("workspace_mean", ""),
                        "workspace_std": source_metrics.get("workspace_std", ""),
                        "live_nvdec_transfer_mode": source_metrics.get("live_nvdec_transfer_mode", ""),
                        "nvdec_batch_frames": source_metrics.get("nvdec_batch_frames", ""),
                        "pinned_transfer_buffers": source_metrics.get("pinned_transfer_buffers", ""),
                        "native_d3d11_decode_crop_readback_ms": source_metrics.get("native_d3d11_decode_crop_readback_ms", ""),
                        "live_queue_wait_ms": source_metrics.get("live_queue_wait_ms", ""),
                    }
                    yield frame_index, frame_cpu, None, metrics
                else:
                    yield frame_index, frame_cpu, None
            return

        with VideoReader(self._reader.path) as batch_reader:
            iterator = iter(batch_reader.read_range(start, end))
            while True:
                decode_start = perf_counter() if include_timing else 0.0
                try:
                    frame_index, frame_cpu = next(iterator)
                except StopIteration:
                    return
                decode_ms = (perf_counter() - decode_start) * 1000.0 if include_timing else 0.0

                upload_start = perf_counter() if include_timing else 0.0
                if needs_gpu and not self._gpu_available:
                    raise RuntimeError('This batch includes a CUDA-only tracker, but CuPy is unavailable.')
                frame_gpu = cp.asarray(frame_cpu) if needs_gpu else None
                gpu_upload_ms = (perf_counter() - upload_start) * 1000.0 if include_timing else 0.0

                if include_timing:
                    yield frame_index, frame_cpu, frame_gpu, {
                        "decode_ms": float(decode_ms),
                        "cache_read_ms": 0.0,
                        "gpu_upload_ms": float(gpu_upload_ms),
                        "frame_source": "direct_decode",
                    }
                else:
                    yield frame_index, frame_cpu, frame_gpu

    def iter_batch_frames_reverse(
        self,
        start: int,
        end: int,
        *,
        needs_gpu: bool,
    ):
        """Yield full frames in descending order for opt-in backward tracking.

        Reverse access uses indexed reads because the efficient NVDEC and
        workspace-cache paths are forward-only.  It is intentionally separate
        from the normal batch iterator, so the usual forward path is unchanged.
        """
        start = max(0, int(start))
        end = min(int(end), int(self._reader.frame_count))
        if end <= start:
            return
        with VideoReader(self._reader.path) as reverse_reader:
            for frame_index in range(end - 1, start - 1, -1):
                frame_cpu = reverse_reader.read(frame_index)
                if frame_cpu is None:
                    continue
                if needs_gpu and not self._gpu_available:
                    raise RuntimeError('Backward tracking requires CuPy for the selected tracker.')
                frame_gpu = cp.asarray(frame_cpu) if needs_gpu else None
                yield frame_index, frame_cpu, frame_gpu

    def get_frame_cpu(self, index: int) -> Optional[np.ndarray]:
        """Return decoded CPU BGR data without a GPU round-trip."""
        if self._batch_exclusive_active:
            return None
        index = int(index)
        if self._playback_prebuffer_enabled:
            with self._playback_lock:
                cached = self._playback_cache.get(index)
                if cached is not None:
                    self._playback_cache.move_to_end(index)
                    self._maybe_refill_playback_prebuffer(index)
                    return cached
            self._schedule_playback_prebuffer(index)
        entry = self._get_or_load_cpu(index)
        return None if entry is None else entry.cpu

    def get_frame(self, index: int) -> Optional[cp.ndarray]:
        """Return a GPU BGR array, uploading lazily if needed."""
        if self._batch_exclusive_active:
            return None
        entry = self._get_or_load_cpu(index)
        if entry is None:
            return None
        return self._ensure_gpu(index, entry)

    def get_frame_pair(self, index: int) -> Optional[FramePair]:
        """Return shared CPU/GPU representations using one decode/cache entry."""
        if self._batch_exclusive_active:
            return None
        entry = self._get_or_load_cpu(index)
        if entry is None:
            return None
        return (entry.cpu, self._ensure_gpu(index, entry))

    def seek(self, index: int) -> None:
        self._prefetch_target = index
        self._prefetch_head = index
        self._seek_event.set()

    def close(self) -> None:
        self._stop_playback_prebuffer()
        self._stop_event.set()
        self._seek_event.set()
        self._prefetch_thread.join(timeout=2.0)
        with self._cache_lock:
            self._cache.clear()

    def __enter__(self) -> 'FrameBuffer':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop_playback_prebuffer(self) -> None:
        self._playback_generation += 1
        self._playback_stop.set()
        thread = self._playback_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._playback_thread = None
        self._playback_range = (-1, -1)
        self._playback_stop = threading.Event()

    def _schedule_playback_prebuffer(self, start_index: int) -> None:
        if not self._playback_prebuffer_enabled:
            return
        start_index = max(0, min(int(start_index), int(self._reader.frame_count) - 1))
        end_index = min(int(self._reader.frame_count), start_index + self._playback_lookahead_frames)
        current_start, current_end = self._playback_range
        # If an active producer already covers the requested frame, let it continue.
        if self._playback_thread is not None and self._playback_thread.is_alive():
            if current_start <= start_index < current_end:
                return
            self._playback_generation += 1
            self._playback_stop.set()

        self._playback_generation += 1
        generation = self._playback_generation
        self._playback_range = (start_index, end_index)
        self._playback_stop = threading.Event()
        thread = threading.Thread(
            target=self._playback_prefetch_worker,
            args=(start_index, end_index, self._playback_stop, generation),
            daemon=True,
            name='FrameBuffer-PlaybackNVDEC',
        )
        self._playback_thread = thread
        thread.start()

    def _maybe_refill_playback_prebuffer(self, current_index: int) -> None:
        if not self._playback_prebuffer_enabled:
            return
        next_index = int(current_index) + 1
        frame_count = int(self._reader.frame_count)
        if next_index >= frame_count:
            return

        # A future range may already be decoding while we are displaying
        # cached frames from the preceding range.  Never replace that worker:
        # doing so caused the periodic D3D11 playback pause at each boundary.
        if self._playback_thread is not None and self._playback_thread.is_alive():
            return

        current_start, current_end = self._playback_range
        if next_index < current_start:
            # The completed range is already ahead of the playhead.
            return
        if current_start <= next_index < current_end:
            remaining = current_end - next_index
            watermark = min(
                PLAYBACK_REFILL_WATERMARK_FRAMES,
                max(1, self._playback_lookahead_frames // 3),
            )
            if remaining > watermark:
                return
            if current_end >= frame_count:
                return
            self._schedule_playback_prebuffer(current_end)
            return

        self._schedule_playback_prebuffer(next_index)

    def _playback_prefetch_worker(
        self,
        start: int,
        end: int,
        stop: threading.Event,
        generation: int,
    ) -> None:
        try:
            workspace = WorkspaceRect(0, 0, int(self._reader.width), int(self._reader.height))
            bytes_per_frame = int(self._reader.width * self._reader.height * 3)
            ram_budget = max(bytes_per_frame, bytes_per_frame * self._playback_lookahead_frames)
            for frame_index, frame, _metrics in iter_live_workspace_frames(
                video_path=self._reader.path,
                workspace=workspace,
                start=start,
                end=end,
                prefer_nvdec=self._playback_prefer_nvdec,
                gpu_id=self._playback_gpu_id,
                ram_lookahead_bytes=ram_budget,
                max_lookahead_frames=self._playback_lookahead_frames,
                min_start_prebuffer_frames=PLAYBACK_START_PREBUFFER_FRAMES,
                nvdec_transfer_mode=self._playback_nvdec_transfer_mode,
                decoder_preference=self._playback_decoder_preference,
            ):
                if stop.is_set() or generation != self._playback_generation:
                    break
                # Full-frame workspace has origin (0,0). Copy so the frame is independent of decoder buffers.
                frame_cpu = np.ascontiguousarray(frame.bgr.copy())
                with self._playback_lock:
                    if generation != self._playback_generation:
                        break
                    self._playback_cache[int(frame_index)] = frame_cpu
                    self._playback_cache.move_to_end(int(frame_index))
                    while len(self._playback_cache) > self._playback_lookahead_frames * 2:
                        self._playback_cache.popitem(last=False)
        except Exception:
            # Playback prebuffer is an optimization only; normal OpenCV path remains available.
            return

    def _get_or_load_cpu(self, index: int) -> Optional[_CachedFrame]:
        self._prefetch_target = index
        with self._cache_lock:
            entry = self._cache.get(index)
            if entry is not None:
                self._cache.move_to_end(index)
                return entry

        frame_cpu = self._reader.read(index)
        if frame_cpu is None:
            return None

        entry = _CachedFrame(cpu=frame_cpu)
        self._store(index, entry)

        if abs(index - self._prefetch_head) > self._prefetch_ahead:
            self._prefetch_head = index
            self._seek_event.set()
        return entry

    def _ensure_gpu(self, index: int, entry: _CachedFrame) -> cp.ndarray:
        if not self._gpu_available:
            raise RuntimeError('GPU frame access requires CuPy.')
        with self._cache_lock:
            cached = self._cache.get(index)
            if cached is not None and cached.gpu is not None:
                self._cache.move_to_end(index)
                return cached.gpu

        gpu = self._upload_cpu_frame(entry.cpu)
        with self._cache_lock:
            cached = self._cache.get(index)
            if cached is not None:
                if cached.gpu is None:
                    cached.gpu = gpu
                self._cache.move_to_end(index)
                return cached.gpu
        # Entry may have been evicted while uploading; return usable GPU frame.
        return gpu

    def _upload_cpu_frame(self, frame_cpu: np.ndarray) -> cp.ndarray:
        if not self._gpu_available or self._upload_stream is None:
            raise RuntimeError('GPU frame upload requires CuPy.')
        with self._upload_stream:
            gpu = cp.asarray(frame_cpu)
        return gpu

    def _store(self, index: int, entry: _CachedFrame) -> None:
        with self._cache_lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return
            self._cache[index] = entry
            if len(self._cache) > self._buffer_size:
                self._cache.popitem(last=False)

    def _prefetch_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._prefetch_paused.is_set():
                self._seek_event.wait(timeout=0.1)
                self._seek_event.clear()
                continue

            head = self._prefetch_head
            with self._cache_lock:
                cached = set(self._cache.keys())

            found = False
            next_idx = head
            for index in range(
                head,
                min(head + self._prefetch_ahead, self._reader.frame_count),
            ):
                if index not in cached:
                    next_idx = index
                    found = True
                    break

            if not found:
                self._seek_event.wait(timeout=0.1)
                self._seek_event.clear()
                continue

            frame_cpu = self._reader.read(next_idx)
            if frame_cpu is not None:
                entry = _CachedFrame(cpu=frame_cpu)
                if self._gpu_prefetch_enabled:
                    entry.gpu = self._upload_cpu_frame(frame_cpu)
                self._store(next_idx, entry)

            self._prefetch_head = next_idx + 1
            if self._seek_event.is_set():
                self._seek_event.clear()
                self._prefetch_head = self._prefetch_target
