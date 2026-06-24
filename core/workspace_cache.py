"""
core/workspace_cache.py
-----------------------
Reusable decoded analysis-workspace cache.

Pipeline when NVIDIA PyNvVideoCodec is available and the video is accepted by
NVDEC:

    ThreadedDecoder GPU RGB frame
      -> GPU crop / RGB-to-BGR conversion
      -> asynchronous device-to-host copy into rotating pinned buffers
      -> uint8 BGR memory-mapped workspace cache on disk

If NVDEC cannot be used, the same cache is built with sequential OpenCV/FFmpeg
CPU decoding and the UI is told that the fallback backend is active.

The same decoder/crop/copy path can also be used in live RAM-buffer mode:

    decoder producer thread -> bounded frame-indexed RAM buffer -> batch tracker consumer

This mode avoids writing decoded frames to SSD.  The reusable disk cache remains
available when the user explicitly wants fast repeated reruns.

The cache keeps full-frame coordinate metadata so a point tracker can process
cropped BGR frames while reporting coordinates in the original video plane.
"""
from __future__ import annotations

import os
if hasattr(os, "add_dll_directory"):
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
import hashlib
import json
import queue
import zlib
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core.video_reader import VideoReader
from core.workspace_types import WorkspaceFrame, WorkspaceRect
from core.ffmpeg_decoder import iter_d3d11va_bgr_frames
from core.native_d3d11 import NativeD3D11WorkspaceDecoder, iter_native_d3d11_workspace_frames


CACHE_VERSION = 4  # Native D3D11 GPU ROI frames are a distinct cache source.
DEFAULT_RAM_LOOKAHEAD_BYTES = 1024 * 1024 * 1024  # 1 GiB
DEFAULT_NVDEC_GPU_BUFFER_FRAMES = 12
DEFAULT_NVDEC_BATCH_FRAMES = 6
DEFAULT_PINNED_TRANSFER_BUFFERS = 3
DEFAULT_LIVE_LOOKAHEAD_FRAMES = 127
DEFAULT_LIVE_START_PREBUFFER_FRAMES = 12
DEFAULT_NVDEC_TRANSFER_MODE = "event_chained"  # fast_async | event_chained | strict_sync


@dataclass(frozen=True)
class CacheBuildResult:
    cache_dir: Path
    backend: str
    backend_message: str
    reused_existing: bool


@dataclass(frozen=True)
class SingleFrameDecodeResult:
    """One workspace frame decoded through the preferred analysis-cache path."""

    frame_index: int
    frame: WorkspaceFrame
    backend: str
    backend_message: str


@dataclass(frozen=True)
class LiveFrameDecodeInfo:
    """Metadata reported by live RAM decoding iterators."""

    backend: str
    backend_message: str
    frame_source: str


def _nvdec_surface_is_padded(rgb_gpu, visible_hw: tuple[int, int]) -> bool:
    """Validate an NVDEC RGB surface and report coded-vs-visible padding.

    Some H.264/H.265 streams expose a padded coded surface (for example 864x656
    for an 858x642 visible frame).  PyNvVideoCodec owns the decoder stream; our
    custom crop/copy streams must not read that surface until decode completion.
    """
    if rgb_gpu.ndim != 3 or rgb_gpu.shape[2] != 3:
        raise RuntimeError(f"Unexpected NVDEC RGB frame shape: {tuple(rgb_gpu.shape)}")
    visible_h, visible_w = [int(value) for value in visible_hw]
    coded_h, coded_w = [int(value) for value in rgb_gpu.shape[:2]]
    if coded_h < visible_h or coded_w < visible_w:
        raise RuntimeError(
            f"NVDEC RGB surface {coded_w}x{coded_h} is smaller than visible video {visible_w}x{visible_h}."
        )
    return (coded_h, coded_w) != (visible_h, visible_w)




def _workspace_crc32(frame_bgr: np.ndarray) -> str:
    """Cheap deterministic fingerprint for frame-source diagnostics.

    Use the whole workspace bytes.  Workspaces used for batch tracking are small
    enough that this is cheap relative to decoding/tracking, and it makes stale
    frame reuse visible in profiling/debug output.
    """
    arr = np.ascontiguousarray(frame_bgr)
    return f"{zlib.crc32(arr.view(np.uint8)) & 0xFFFFFFFF:08x}"


def _workspace_basic_metrics(frame_bgr: np.ndarray, *, frame_index: int) -> dict[str, object]:
    arr = np.asarray(frame_bgr)
    return {
        "source_frame_index": int(frame_index),
        "workspace_crc32": _workspace_crc32(arr),
        "workspace_mean": float(np.mean(arr)) if arr.size else 0.0,
        "workspace_std": float(np.std(arr)) if arr.size else 0.0,
        "workspace_shape": tuple(int(v) for v in arr.shape),
    }


class _FrameIndexedLiveBuffer:
    """Bounded producer/consumer buffer addressed by exact frame index.

    A queue only guarantees consumption order.  This buffer enforces the stronger
    invariant needed by the tracker: when the consumer asks for frame N, it can
    only receive a slot whose stored frame_index is exactly N.  The producer still
    decodes ahead up to ``capacity`` frames, but every slot carries its own index,
    owned NumPy array, metadata, and fingerprint diagnostics.
    """

    def __init__(self, capacity: int, stop_event: threading.Event) -> None:
        self.capacity = max(1, int(capacity))
        self._stop_event = stop_event
        self._cond = threading.Condition()
        self._slots: dict[int, tuple[WorkspaceFrame, dict[str, object]]] = {}
        self._recent_hashes: dict[int, str] = {}
        self._closed = False
        self._error: Optional[BaseException] = None

    def put(self, frame_index: int, frame: WorkspaceFrame, metrics: dict[str, object]) -> None:
        frame_index = int(frame_index)
        metrics = dict(metrics)
        if "source_frame_index" not in metrics:
            metrics.update(_workspace_basic_metrics(frame.bgr, frame_index=frame_index))

        crc = str(metrics.get("workspace_crc32", ""))
        repeat_prev = bool(crc and self._recent_hashes.get(frame_index - 1) == crc)
        repeat_n2 = bool(crc and self._recent_hashes.get(frame_index - 2) == crc)
        metrics["workspace_crc32_repeat_n_minus_1"] = repeat_prev
        metrics["workspace_crc32_repeat_n_minus_2"] = repeat_n2
        metrics["live_buffer_mode"] = "frame_indexed"
        # Keep the same diagnostics attached to the frame object for future
        # tracker-side source-error gating without changing public APIs again.
        try:
            frame.metadata.update(metrics)
        except Exception:
            pass

        with self._cond:
            while len(self._slots) >= self.capacity and not self._closed and not self._stop_event.is_set():
                self._cond.wait(timeout=0.05)
            if self._closed or self._stop_event.is_set():
                return
            self._slots[frame_index] = (frame, metrics)
            self._recent_hashes[frame_index] = crc
            # Retain enough history for n-2 diagnostics plus a little slack.
            for old_index in list(self._recent_hashes.keys()):
                if old_index < frame_index - max(8, self.capacity * 2):
                    self._recent_hashes.pop(old_index, None)
            self._cond.notify_all()

    def put_error(self, exc: BaseException) -> None:
        with self._cond:
            self._error = exc
            self._closed = True
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def wait_for_prebuffer(self, required: int) -> None:
        required = max(0, int(required))
        if required <= 0:
            return
        with self._cond:
            while len(self._slots) < required and not self._closed and self._error is None:
                self._cond.wait(timeout=0.05)
            if self._error is not None:
                raise self._error

    def get(self, frame_index: int) -> Optional[tuple[WorkspaceFrame, dict[str, object]]]:
        frame_index = int(frame_index)
        with self._cond:
            while frame_index not in self._slots and not self._closed and self._error is None:
                self._cond.wait(timeout=0.05)
            if self._error is not None:
                raise self._error
            item = self._slots.pop(frame_index, None)
            self._cond.notify_all()
            return item


class WorkspaceDiskCache:
    """Opened, complete memory-mapped BGR workspace frame cache."""

    METADATA_FILENAME = "metadata.json"
    FRAMES_FILENAME = "workspace_frames.dat"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing analysis-cache metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not self.metadata.get("complete", False):
            raise RuntimeError(f"Analysis cache is incomplete: {self.cache_dir}")
        shape = tuple(int(v) for v in self.metadata["shape"])
        self._frames = np.memmap(
            self.cache_dir / self.FRAMES_FILENAME,
            dtype=np.uint8,
            mode="r",
            shape=shape,
        )

    @property
    def frame_count(self) -> int:
        return int(self.metadata["decoded_frames"])

    @property
    def workspace(self) -> WorkspaceRect:
        return WorkspaceRect(**self.metadata["workspace"])

    @property
    def full_frame_hw(self) -> tuple[int, int]:
        return (int(self.metadata["full_height"]), int(self.metadata["full_width"]))

    @property
    def backend(self) -> str:
        return str(self.metadata.get("backend", "unknown"))

    @property
    def backend_message(self) -> str:
        return str(self.metadata.get("backend_message", ""))

    def covers_range(self, start: int, end: int) -> bool:
        return 0 <= start < end <= self.frame_count

    def get_frame(self, frame_index: int) -> WorkspaceFrame:
        """Return one copied cached workspace frame by original frame index."""
        if not 0 <= int(frame_index) < self.frame_count:
            raise IndexError(f"Cached frame index out of range: {frame_index}")
        frame = np.array(self._frames[int(frame_index)], copy=True)
        return WorkspaceFrame(
            bgr=frame,
            origin_xy=(self.workspace.x, self.workspace.y),
            full_frame_hw=self.full_frame_hw,
        )

    def iter_frames_prefetched(
        self,
        start: int,
        end: int,
        *,
        ram_budget_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES,
    ) -> Iterator[tuple[int, WorkspaceFrame]]:
        """Yield copied workspace frames using a bounded producer/consumer RAM queue."""
        start = max(0, int(start))
        end = min(int(end), self.frame_count)
        if start >= end:
            return

        bytes_per_frame = int(self.workspace.width * self.workspace.height * 3)
        capacity = max(1, min(end - start, int(ram_budget_bytes // max(bytes_per_frame, 1))))
        ready: "queue.Queue[object]" = queue.Queue(maxsize=capacity)
        stop = threading.Event()
        sentinel = object()

        def producer() -> None:
            try:
                for frame_index in range(start, end):
                    if stop.is_set():
                        break
                    # Copy out of memmap so downstream use is independent of page turnover.
                    frame = np.array(self._frames[frame_index], copy=True)
                    ready.put((frame_index, frame))
            finally:
                ready.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True, name="WorkspaceCache-Prefetch")
        thread.start()
        try:
            while True:
                item = ready.get()
                if item is sentinel:
                    break
                frame_index, frame = item
                yield frame_index, WorkspaceFrame(
                    bgr=frame,
                    origin_xy=(self.workspace.x, self.workspace.y),
                    full_frame_hw=self.full_frame_hw,
                )
        finally:
            stop.set()
            thread.join(timeout=1.0)


def _cache_key(video_path: Path, reader: VideoReader, workspace: WorkspaceRect) -> str:
    stat = video_path.stat()
    payload = {
        "cache_version": CACHE_VERSION,
        "path": str(video_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "frame_count": int(reader.frame_count),
        "fps": float(reader.fps),
        "width": int(reader.width),
        "height": int(reader.height),
        "workspace": workspace.__dict__,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_directory_for(
    cache_root: Path,
    video_path: Path,
    reader: VideoReader,
    workspace: WorkspaceRect,
) -> Path:
    return Path(cache_root) / f"{video_path.stem}_{_cache_key(video_path, reader, workspace)}"




def decode_single_workspace_frame(
    *,
    video_path: str | Path,
    workspace: WorkspaceRect,
    frame_index: int,
    prefer_nvdec: bool = True,
    gpu_id: int = 0,
    decoder_preference: str = "auto",
) -> SingleFrameDecodeResult:
    """Decode one cropped workspace frame using the same path as cache building.

    This is used for marker initialization so the learned colour model can come
    from the prepared-cache/NVDEC decode path instead of the normal OpenCV
    display frame.  If NVDEC cannot be used, it falls back to OpenCV and
    returns the fallback reason in ``backend_message``.
    """
    video_path = Path(video_path)
    with VideoReader(video_path) as reader:
        workspace = workspace.clamped(reader.width, reader.height)
        frame_index = max(0, min(int(frame_index), int(reader.frame_count) - 1))
        preference = str(decoder_preference or "auto").lower()
        if preference in {"auto", "nvdec"} and prefer_nvdec:
            try:
                frame = _decode_single_workspace_frame_nvdec(
                    video_path=video_path,
                    workspace=workspace,
                    frame_index=frame_index,
                    full_frame_hw=(reader.height, reader.width),
                    gpu_id=gpu_id,
                )
                return SingleFrameDecodeResult(
                    frame_index=frame_index,
                    frame=frame,
                    backend="NVDEC (PyNvVideoCodec)",
                    backend_message="Initialization frame decoded with NVIDIA NVDEC.",
                )
            except Exception as exc:
                fallback_message = (
                    "Initialization NVDEC decode failed; using OpenCV fallback. "
                    f"Reason: {exc}"
                )
        elif preference == "nvdec":
            fallback_message = "NVIDIA NVDEC preference disabled; using OpenCV fallback."
        if preference == "d3d11va":
            try:
                for decoded_index, frame_bgr in iter_d3d11va_bgr_frames(
                    video_path, width=reader.width, height=reader.height,
                    start=frame_index, end=frame_index + 1,
                    crop_xywh=workspace.xywh,
                ):
                    return SingleFrameDecodeResult(
                        frame_index=decoded_index,
                        frame=WorkspaceFrame(bgr=frame_bgr, origin_xy=(workspace.x, workspace.y), full_frame_hw=(reader.height, reader.width)),
                        backend="FFmpeg D3D11VA",
                        backend_message="Initialization frame decoded with FFmpeg D3D11VA.",
                    )
                raise RuntimeError("FFmpeg D3D11VA produced no target frame.")
            except Exception as exc:
                fallback_message = f"FFmpeg D3D11VA failed; using OpenCV fallback. Reason: {exc}"
        elif preference in {"auto", "native_d3d11"}:
            try:
                with NativeD3D11WorkspaceDecoder(
                    video_path, x=workspace.x, y=workspace.y,
                    width=workspace.width, height=workspace.height,
                ) as decoder:
                    while (item := decoder.read()) is not None:
                        decoded_index, frame_bgr = item
                        if decoded_index == frame_index:
                            return SingleFrameDecodeResult(
                                frame_index=decoded_index,
                                frame=WorkspaceFrame(bgr=frame_bgr, origin_xy=(workspace.x, workspace.y), full_frame_hw=(reader.height, reader.width)),
                                backend="Native FFmpeg D3D11VA",
                                backend_message="Initialization frame decoded with native FFmpeg D3D11VA GPU ROI crop.",
                            )
                raise RuntimeError("Native D3D11 decoder produced no target frame.")
            except Exception as exc:
                fallback_message = f"Native FFmpeg D3D11VA failed; using OpenCV fallback. Reason: {exc}"
        elif preference == "cpu":
            fallback_message = "CPU/OpenCV selected."
        elif preference == "auto" and not prefer_nvdec:
            fallback_message = "Native D3D11 preference unavailable; using OpenCV fallback."

        frame_bgr = reader.read(frame_index)
        if frame_bgr is None:
            raise RuntimeError(f"Could not decode initialization frame {frame_index}.")
        x, y, w, h = workspace.xywh
        return SingleFrameDecodeResult(
            frame_index=frame_index,
            frame=WorkspaceFrame(
                bgr=frame_bgr[y:y + h, x:x + w].copy(),
                origin_xy=(workspace.x, workspace.y),
                full_frame_hw=(reader.height, reader.width),
            ),
            backend="OpenCV/FFmpeg CPU fallback",
            backend_message=fallback_message,
        )


def _decode_single_workspace_frame_nvdec(
    *,
    video_path: Path,
    workspace: WorkspaceRect,
    frame_index: int,
    full_frame_hw: tuple[int, int],
    gpu_id: int,
) -> WorkspaceFrame:
    import cupy as cp
    try:
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise RuntimeError(
            "PyNvVideoCodec is not installed. Install it to enable NVIDIA NVDEC."
        ) from exc

    decoder = nvc.ThreadedDecoder(
        enc_file_path=str(video_path),
        buffer_size=DEFAULT_NVDEC_GPU_BUFFER_FRAMES,
        gpu_id=int(gpu_id),
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGB,
    )
    decoded_index = 0
    while True:
        frames = decoder.get_batch_frames(DEFAULT_NVDEC_BATCH_FRAMES)
        if not frames:
            break
        for frame in frames:
            if decoded_index == frame_index:
                try:
                    rgb_gpu = cp.from_dlpack(frame)
                except AttributeError:
                    rgb_gpu = cp.fromDlpack(frame)
                _nvdec_surface_is_padded(rgb_gpu, full_frame_hw)
                x, y, w, h = workspace.xywh
                bgr_workspace_gpu = cp.ascontiguousarray(rgb_gpu[y:y + h, x:x + w, ::-1])
                bgr_workspace_cpu = cp.asnumpy(bgr_workspace_gpu)
                return WorkspaceFrame(
                    bgr=bgr_workspace_cpu,
                    origin_xy=(workspace.x, workspace.y),
                    full_frame_hw=full_frame_hw,
                )
            decoded_index += 1
            if decoded_index > frame_index:
                break
    raise RuntimeError(f"NVDEC produced only {decoded_index} frames before target {frame_index}.")


def iter_live_workspace_frames(
    *,
    video_path: str | Path,
    workspace: WorkspaceRect,
    start: int,
    end: int,
    prefer_nvdec: bool = True,
    gpu_id: int = 0,
    ram_lookahead_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES,
    max_lookahead_frames: int = DEFAULT_LIVE_LOOKAHEAD_FRAMES,
    min_start_prebuffer_frames: int = DEFAULT_LIVE_START_PREBUFFER_FRAMES,
    nvdec_transfer_mode: str = DEFAULT_NVDEC_TRANSFER_MODE,
    decoder_preference: str = "auto",
) -> Iterator[tuple[int, WorkspaceFrame, dict[str, object]]]:
    """Yield live workspace frames through a bounded frame-indexed RAM buffer.

    Unlike :class:`WorkspaceDiskCache`, this function never writes decoded
    frames to disk.  A producer thread decodes sequentially, crops the selected
    workspace, copies it to owned CPU RAM, and stores each ``WorkspaceFrame`` in
    a bounded RAM buffer under its exact frame index.  The tracker consumes
    ``frame_index == requested_index`` only; it no longer trusts queue order as a
    substitute for frame identity.

    NVDEC/PyNvVideoCodec is tried first when requested.  If it fails before any
    frame is emitted, the producer transparently falls back to sequential
    OpenCV/FFmpeg decoding.  If a decoder fails after frames have already been
    emitted, the error is propagated because mixing decoder paths mid-run would
    make colour tracking less reproducible.
    """
    video_path = Path(video_path)
    with VideoReader(video_path) as reader:
        workspace = workspace.clamped(reader.width, reader.height)
        start = max(0, int(start))
        end = min(int(end), int(reader.frame_count))
        if start >= end:
            return

        transfer_mode = str(nvdec_transfer_mode or DEFAULT_NVDEC_TRANSFER_MODE).strip().lower().replace("-", "_")
        valid_transfer_modes = {"fast_async", "event_chained", "strict_sync"}
        if transfer_mode not in valid_transfer_modes:
            transfer_mode = DEFAULT_NVDEC_TRANSFER_MODE

        bytes_per_frame = int(workspace.width * workspace.height * 3)
        memory_capacity = int(ram_lookahead_bytes // max(bytes_per_frame, 1))
        frame_capacity = max(1, int(max_lookahead_frames))
        # Live RAM mode must be paced.  A pure 1 GiB budget can allow hundreds
        # or thousands of small workspace frames to be decoded far ahead, which
        # can contend with the CPU tracker and create long scheduler/memory
        # stalls.  The queue is therefore capped by frame count as well as RAM.
        capacity = max(1, min(end - start, max(1, memory_capacity), frame_capacity))
        start_prebuffer = max(0, min(int(min_start_prebuffer_frames), capacity, end - start))
        stop = threading.Event()
        live_buffer = _FrameIndexedLiveBuffer(capacity=capacity, stop_event=stop)

        emitted_count = 0

        def put_frame(frame_index: int, frame: WorkspaceFrame, metrics: dict[str, object]) -> None:
            nonlocal emitted_count
            live_buffer.put(int(frame_index), frame, metrics)
            emitted_count += 1

        def produce_opencv(message: str) -> None:
            backend = "OpenCV/FFmpeg CPU fallback"
            with VideoReader(video_path) as cv_reader:
                for frame_index, frame_bgr in cv_reader.read_range(start, end):
                    if stop.is_set():
                        break
                    x, y, w, h = workspace.xywh
                    workspace_bgr = frame_bgr[y:y + h, x:x + w].copy()
                    put_frame(
                        frame_index,
                        WorkspaceFrame(
                            bgr=workspace_bgr,
                            origin_xy=(workspace.x, workspace.y),
                            full_frame_hw=(cv_reader.height, cv_reader.width),
                            metadata={"source_frame_index": int(frame_index), "frame_source": "live_ram_opencv"},
                        ),
                        {
                            "frame_source": "live_ram_opencv",
                            "live_backend": backend,
                            "live_backend_message": message,
                            "live_lookahead_capacity_frames": capacity,
                            "live_start_prebuffer_frames": start_prebuffer,
                            "live_nvdec_transfer_mode": transfer_mode,
                            "nvdec_batch_frames": DEFAULT_NVDEC_BATCH_FRAMES,
                            "pinned_transfer_buffers": DEFAULT_PINNED_TRANSFER_BUFFERS,
                        },
                    )

        def produce_d3d11va() -> None:
            for frame_index, frame_bgr in iter_d3d11va_bgr_frames(
                video_path, width=reader.width, height=reader.height, start=start, end=end,
                crop_xywh=workspace.xywh,
            ):
                if stop.is_set():
                    break
                put_frame(
                    frame_index,
                    WorkspaceFrame(bgr=frame_bgr, origin_xy=(workspace.x, workspace.y), full_frame_hw=(reader.height, reader.width), metadata={"source_frame_index": int(frame_index), "frame_source": "live_ram_d3d11va"}),
                    {"frame_source": "live_ram_d3d11va", "live_backend": "FFmpeg D3D11VA", "live_backend_message": "Live RAM mode: FFmpeg D3D11VA decode, CPU-side NV12 crop, and cropped BGR delivery.", "live_lookahead_capacity_frames": capacity, "live_start_prebuffer_frames": start_prebuffer},
                )

        def produce_native_d3d11() -> None:
            for frame_index, frame_bgr, backend_message, native_decode_ms in iter_native_d3d11_workspace_frames(
                video_path, x=workspace.x, y=workspace.y, width=workspace.width, height=workspace.height,
                start=start, end=end,
            ):
                if stop.is_set():
                    break
                put_frame(
                    frame_index,
                    WorkspaceFrame(bgr=frame_bgr, origin_xy=(workspace.x, workspace.y), full_frame_hw=(reader.height, reader.width), metadata={"source_frame_index": int(frame_index), "frame_source": "live_ram_native_d3d11"}),
                    {"frame_source": "live_ram_native_d3d11", "live_backend": "Native FFmpeg D3D11VA", "live_backend_message": backend_message, "native_d3d11_decode_crop_readback_ms": float(native_decode_ms), "live_lookahead_capacity_frames": capacity, "live_start_prebuffer_frames": start_prebuffer},
                )

        def produce_auto_fallback(reason: str) -> None:
            try:
                produce_native_d3d11()
                if emitted_count <= 0:
                    produce_opencv(f"{reason} Native D3D11 produced no frames; using OpenCV fallback.")
            except Exception as exc:
                if emitted_count > 0:
                    raise
                produce_opencv(f"{reason} Native D3D11 failed; using OpenCV fallback. Reason: {exc}")

        def produce_nvdec() -> int:
            import cupy as cp
            try:
                import PyNvVideoCodec as nvc
            except ImportError as exc:
                raise RuntimeError(
                    "PyNvVideoCodec is not installed. Install it to enable NVIDIA NVDEC."
                ) from exc

            decoder = nvc.ThreadedDecoder(
                enc_file_path=str(video_path),
                buffer_size=DEFAULT_NVDEC_GPU_BUFFER_FRAMES,
                gpu_id=int(gpu_id),
                use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
            )
            shape = (workspace.height, workspace.width, 3)
            nbytes = int(np.prod(shape))
            use_pinned_slots = transfer_mode != "strict_sync"
            copy_stream = cp.cuda.Stream(non_blocking=True) if use_pinned_slots else None
            compute_stream = cp.cuda.Stream(non_blocking=True) if transfer_mode == "event_chained" else None

            def make_metrics(mode_message: str) -> dict[str, object]:
                return {
                    "frame_source": "live_ram_nvdec",
                    "live_backend": "NVDEC (PyNvVideoCodec)",
                    "live_backend_message": mode_message,
                    "live_lookahead_capacity_frames": capacity,
                    "live_start_prebuffer_frames": start_prebuffer,
                    "live_nvdec_transfer_mode": transfer_mode,
                    "nvdec_batch_frames": DEFAULT_NVDEC_BATCH_FRAMES,
                    "pinned_transfer_buffers": DEFAULT_PINNED_TRANSFER_BUFFERS if use_pinned_slots else 0,
                }

            class _PinnedSlot:
                def __init__(self) -> None:
                    self.memory = cp.cuda.alloc_pinned_memory(nbytes)
                    self.array = np.frombuffer(self.memory, dtype=np.uint8, count=nbytes).reshape(shape)
                    self.event = cp.cuda.Event()
                    self.index: Optional[int] = None
                    # Keep GPU arrays/events alive until the async copy into this
                    # pinned buffer is known complete.  This is important for
                    # DLPack-backed decoder frames whose lifetime may otherwise
                    # be shorter than the queued GPU->CPU transfer.
                    self.frame_ref = None
                    self.rgb_ref = None
                    self.bgr_ref = None
                    self.crop_done_event = None

            slots = [_PinnedSlot() for _ in range(DEFAULT_PINNED_TRANSFER_BUFFERS)] if use_pinned_slots else []
            emitted = 0
            decoded_index = 0

            def commit(slot: _PinnedSlot) -> None:
                nonlocal emitted
                if slot.index is None:
                    return
                slot.event.synchronize()
                if start <= slot.index < end and not stop.is_set():
                    # Copy into normal RAM before queueing so the pinned slot can
                    # be reused immediately by the decode/copy producer.
                    frame = np.array(slot.array, copy=True)
                    put_frame(
                        slot.index,
                        WorkspaceFrame(
                            bgr=frame,
                            origin_xy=(workspace.x, workspace.y),
                            full_frame_hw=(reader.height, reader.width),
                            metadata={"source_frame_index": int(slot.index), "frame_source": "live_ram_nvdec"},
                        ),
                        make_metrics(
                            "Live RAM mode: NVIDIA NVDEC decode, GPU workspace crop/BGR conversion, "
                            "pinned-memory transfer, and frame-indexed RAM delivery."
                        ),
                    )
                    emitted += 1
                slot.index = None
                slot.frame_ref = None
                slot.rgb_ref = None
                slot.bgr_ref = None
                slot.crop_done_event = None

            def copy_async_fast(frame, rgb_gpu, frame_index: int, slot: _PinnedSlot) -> None:
                x, y, w, h = workspace.xywh
                bgr_workspace_gpu = cp.ascontiguousarray(rgb_gpu[y:y + h, x:x + w, ::-1])
                try:
                    bgr_workspace_gpu.get(out=slot.array, stream=copy_stream, blocking=False)
                except TypeError:
                    with copy_stream:
                        bgr_workspace_gpu.get(out=slot.array, stream=copy_stream)
                slot.event.record(copy_stream)
                slot.index = int(frame_index)
                slot.frame_ref = frame
                slot.rgb_ref = rgb_gpu
                slot.bgr_ref = bgr_workspace_gpu
                slot.crop_done_event = None

            def copy_async_event_chained(frame, rgb_gpu, frame_index: int, slot: _PinnedSlot) -> None:
                x, y, w, h = workspace.xywh
                with compute_stream:
                    bgr_workspace_gpu = cp.ascontiguousarray(rgb_gpu[y:y + h, x:x + w, ::-1])
                    crop_done = cp.cuda.Event()
                    crop_done.record(compute_stream)
                try:
                    copy_stream.wait_event(crop_done)
                except AttributeError:
                    # Older CuPy fallback: safe, but more synchronous.
                    crop_done.synchronize()
                try:
                    bgr_workspace_gpu.get(out=slot.array, stream=copy_stream, blocking=False)
                except TypeError:
                    with copy_stream:
                        bgr_workspace_gpu.get(out=slot.array, stream=copy_stream)
                slot.event.record(copy_stream)
                slot.index = int(frame_index)
                slot.frame_ref = frame
                slot.rgb_ref = rgb_gpu
                slot.bgr_ref = bgr_workspace_gpu
                slot.crop_done_event = crop_done

            def copy_strict_sync(rgb_gpu, frame_index: int) -> None:
                nonlocal emitted
                x, y, w, h = workspace.xywh
                # Diagnostic/safest path: force all preceding GPU work for this
                # frame to become visible before copying to owned CPU memory.
                bgr_workspace_gpu = cp.ascontiguousarray(rgb_gpu[y:y + h, x:x + w, ::-1])
                cp.cuda.Device().synchronize()
                frame = np.array(cp.asnumpy(bgr_workspace_gpu), copy=True)
                put_frame(
                    frame_index,
                    WorkspaceFrame(
                        bgr=frame,
                        origin_xy=(workspace.x, workspace.y),
                        full_frame_hw=(reader.height, reader.width),
                        metadata={"source_frame_index": int(frame_index), "frame_source": "live_ram_nvdec"},
                    ),
                    make_metrics(
                        "Live RAM mode: NVIDIA NVDEC decode with strict per-frame CUDA synchronization "
                        "before CPU delivery."
                    ),
                )
                emitted += 1

            while not stop.is_set() and decoded_index < end:
                frames = decoder.get_batch_frames(DEFAULT_NVDEC_BATCH_FRAMES)
                if not frames:
                    break
                for frame in frames:
                    if stop.is_set() or decoded_index >= end:
                        break
                    if decoded_index >= start:
                        try:
                            rgb_gpu = cp.from_dlpack(frame)
                        except AttributeError:
                            rgb_gpu = cp.fromDlpack(frame)
                        padded_surface = _nvdec_surface_is_padded(
                            rgb_gpu, (reader.height, reader.width)
                        )
                        if padded_surface:
                            # Required for codec surfaces whose visible region
                            # is smaller than their coded allocation. Without
                            # this barrier, RGB stripes/warped geometry can be
                            # copied from an incompletely written decode surface.
                            cp.cuda.Device(int(gpu_id)).synchronize()
                        if transfer_mode == "strict_sync":
                            copy_strict_sync(rgb_gpu, decoded_index)
                        else:
                            slot = slots[decoded_index % len(slots)]
                            commit(slot)
                            if transfer_mode == "event_chained":
                                copy_async_event_chained(frame, rgb_gpu, decoded_index, slot)
                            else:
                                copy_async_fast(frame, rgb_gpu, decoded_index, slot)
                    decoded_index += 1

            for slot in slots:
                commit(slot)
            if copy_stream is not None:
                copy_stream.synchronize()
            if compute_stream is not None:
                compute_stream.synchronize()
            return emitted

        def producer() -> None:
            try:
                preference = str(decoder_preference or "auto").lower()
                if preference == "d3d11va":
                    try:
                        produce_d3d11va()
                    except Exception as exc:
                        if emitted_count > 0:
                            raise
                        produce_opencv(f"FFmpeg D3D11VA failed before producing frames; using OpenCV fallback. Reason: {exc}")
                elif preference == "native_d3d11":
                    try:
                        produce_native_d3d11()
                    except Exception as exc:
                        if emitted_count > 0:
                            raise
                        produce_opencv(f"Native FFmpeg D3D11VA failed before producing frames; using OpenCV fallback. Reason: {exc}")
                elif preference == "cpu":
                    produce_opencv("CPU/OpenCV selected.")
                elif prefer_nvdec:
                    try:
                        emitted = produce_nvdec()
                        if emitted <= 0 and emitted_count <= 0:
                            produce_auto_fallback("Live NVDEC produced no frames;")
                    except Exception as exc:
                        if emitted_count > 0:
                            raise
                        produce_auto_fallback(f"Live NVDEC decode failed before producing frames ({exc});")
                else:
                    produce_auto_fallback("Live NVDEC preference disabled;")
            except Exception as exc:
                live_buffer.put_error(exc)
            finally:
                live_buffer.close()

        thread = threading.Thread(target=producer, daemon=True, name="WorkspaceLiveDecoder")
        thread.start()
        try:
            # Startup prebuffer: allow the live producer to build a small indexed
            # lead before tracking begins.  Unlike the old queue, the consumer
            # never accepts "next available"; it requests each exact frame index.
            live_buffer.wait_for_prebuffer(start_prebuffer)

            for expected_index in range(start, end):
                wait_started = perf_counter()
                item = live_buffer.get(expected_index)
                if item is None:
                    break
                frame, metrics = item
                metrics = dict(metrics)
                metrics["live_queue_wait_ms"] = (perf_counter() - wait_started) * 1000.0
                source_index = int(metrics.get("source_frame_index", expected_index))
                if source_index != expected_index:
                    raise RuntimeError(
                        f"Live frame-indexed buffer mismatch: requested {expected_index}, "
                        f"slot contains source frame {source_index}."
                    )
                yield expected_index, frame, metrics
        finally:
            stop.set()
            live_buffer.close()
            thread.join(timeout=2.0)

class CacheBuildWorker(QObject):
    """Build or reuse a decoded workspace cache in a background QThread."""

    progress = pyqtSignal(int, int)          # completed, total
    status = pyqtSignal(str)
    backend_selected = pyqtSignal(str, str)  # backend, explanatory message
    finished = pyqtSignal(object)            # CacheBuildResult
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str | Path,
        cache_root: str | Path,
        workspace: WorkspaceRect,
        *,
        prefer_nvdec: bool = True,
        gpu_id: int = 0,
        decoder_preference: str = "auto",
        ram_lookahead_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path = Path(video_path)
        self._cache_root = Path(cache_root)
        self._workspace_requested = workspace
        self._prefer_nvdec = bool(prefer_nvdec)
        self._decoder_preference = str(decoder_preference or "auto").lower()
        self._gpu_id = int(gpu_id)
        self.ram_lookahead_bytes = int(ram_lookahead_bytes)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            with VideoReader(self._video_path) as reader:
                workspace = self._workspace_requested.clamped(reader.width, reader.height)
                cache_dir = cache_directory_for(
                    self._cache_root, self._video_path, reader, workspace
                )
                metadata_path = cache_dir / WorkspaceDiskCache.METADATA_FILENAME
                if metadata_path.exists():
                    try:
                        existing = WorkspaceDiskCache(cache_dir)
                        requested_backend = {
                            "nvdec": "NVDEC (PyNvVideoCodec)",
                            "d3d11va": "FFmpeg D3D11VA",
                            "native_d3d11": "Native FFmpeg D3D11VA",
                            "cpu": "OpenCV/FFmpeg CPU",
                        }.get(self._decoder_preference)
                        if requested_backend is not None and existing.backend != requested_backend:
                            raise RuntimeError("Prepared cache was created by a different explicit decoder.")
                        self.backend_selected.emit(
                            existing.backend,
                            f"Reusing prepared cache ({existing.backend}).",
                        )
                        self.progress.emit(existing.frame_count, existing.frame_count)
                        self.finished.emit(CacheBuildResult(
                            cache_dir=cache_dir,
                            backend=existing.backend,
                            backend_message=existing.backend_message,
                            reused_existing=True,
                        ))
                        return
                    except Exception:
                        shutil.rmtree(cache_dir, ignore_errors=True)

                cache_dir.mkdir(parents=True, exist_ok=True)
                frames_path = cache_dir / WorkspaceDiskCache.FRAMES_FILENAME
                expected_shape = (
                    int(reader.frame_count),
                    int(workspace.height),
                    int(workspace.width),
                    3,
                )
                mmap = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=expected_shape)

                backend = ""
                backend_message = ""
                decoded_frames = 0

                if self._decoder_preference == "native_d3d11":
                    try:
                        self.status.emit("Trying native FFmpeg D3D11VA GPU ROI decoder...")
                        backend = "Native FFmpeg D3D11VA"
                        decoded_frames = self._build_native_d3d11(mmap=mmap, reader=reader, workspace=workspace)
                        backend_message = "Using native FFmpeg D3D11VA decode with GPU ROI crop/BGR conversion."
                    except Exception as exc:
                        if self._cancelled:
                            return
                        backend = "OpenCV/FFmpeg CPU fallback"
                        backend_message = f"Native FFmpeg D3D11VA unavailable or failed; using OpenCV fallback. Reason: {exc}"
                        self.backend_selected.emit(backend, backend_message)
                        mmap[:] = 0
                        decoded_frames = self._build_opencv(mmap=mmap, reader=reader, workspace=workspace)
                elif self._decoder_preference == "d3d11va":
                    try:
                        self.status.emit("Trying FFmpeg D3D11VA...")
                        backend = "FFmpeg D3D11VA"
                        decoded_frames = self._build_d3d11va(mmap=mmap, reader=reader, workspace=workspace)
                        backend_message = "Using FFmpeg D3D11VA hardware decode with CPU workspace delivery."
                    except Exception as exc:
                        if self._cancelled:
                            return
                        backend = "OpenCV/FFmpeg CPU fallback"
                        backend_message = f"FFmpeg D3D11VA unavailable or failed; using OpenCV fallback. Reason: {exc}"
                        self.backend_selected.emit(backend, backend_message)
                        mmap[:] = 0
                        decoded_frames = self._build_opencv(mmap=mmap, reader=reader, workspace=workspace)
                elif self._decoder_preference == "cpu":
                    backend = "OpenCV/FFmpeg CPU"
                    backend_message = "CPU/OpenCV selected."
                    decoded_frames = self._build_opencv(mmap=mmap, reader=reader, workspace=workspace)
                elif self._prefer_nvdec:
                    try:
                        self.status.emit("Trying NVIDIA NVDEC / PyNvVideoCodec…")
                        backend = "NVDEC (PyNvVideoCodec)"
                        decoded_frames = self._build_nvdec(
                            mmap=mmap,
                            reader=reader,
                            workspace=workspace,
                        )
                        backend_message = (
                            "Using NVIDIA NVDEC: GPU decode, GPU workspace crop/BGR conversion, "
                            "and pinned-memory device-to-host transfer."
                        )
                    except Exception as exc:
                        if self._cancelled:
                            return
                        if self._decoder_preference == "auto":
                            try:
                                self.status.emit("NVDEC unavailable; trying native FFmpeg D3D11VA GPU ROI decoder...")
                                mmap[:] = 0
                                backend = "Native FFmpeg D3D11VA"
                                decoded_frames = self._build_native_d3d11(mmap=mmap, reader=reader, workspace=workspace)
                                backend_message = "NVDEC unavailable; using native FFmpeg D3D11VA GPU ROI decoder."
                            except Exception as native_exc:
                                backend = "OpenCV/FFmpeg CPU fallback"
                                backend_message = f"NVDEC and native D3D11 unavailable; using OpenCV fallback. NVDEC: {exc}; native: {native_exc}"
                                self.backend_selected.emit(backend, backend_message)
                                mmap[:] = 0
                                decoded_frames = self._build_opencv(mmap=mmap, reader=reader, workspace=workspace)
                        else:
                            backend = "OpenCV/FFmpeg CPU fallback"
                            backend_message = f"NVDEC unavailable or failed; using OpenCV fallback. Reason: {exc}"
                            self.backend_selected.emit(backend, backend_message)
                            mmap[:] = 0
                            decoded_frames = self._build_opencv(mmap=mmap, reader=reader, workspace=workspace)
                else:
                    backend = "OpenCV/FFmpeg CPU fallback"
                    backend_message = "NVDEC preference disabled; using OpenCV/FFmpeg CPU decoding."
                    self.backend_selected.emit(backend, backend_message)
                    decoded_frames = self._build_opencv(mmap=mmap, reader=reader, workspace=workspace)

                if self._cancelled:
                    mmap.flush()
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    return
                if decoded_frames <= 0:
                    raise RuntimeError("Decoder produced no frames.")
                mmap.flush()

                metadata = {
                    "cache_version": CACHE_VERSION,
                    "complete": True,
                    "video_path": str(self._video_path.resolve()),
                    "frame_count_reported": int(reader.frame_count),
                    "decoded_frames": int(decoded_frames),
                    "fps": float(reader.fps),
                    "full_width": int(reader.width),
                    "full_height": int(reader.height),
                    "workspace": {
                        "x": workspace.x, "y": workspace.y,
                        "width": workspace.width, "height": workspace.height,
                    },
                    "shape": [int(decoded_frames), workspace.height, workspace.width, 3],
                    "dtype": "uint8",
                    "backend": backend,
                    "backend_message": backend_message,
                    "ram_lookahead_bytes": self.ram_lookahead_bytes,
                }
                # If decoder returned fewer frames than reported, truncate the data file
                # by rebuilding an accurately shaped cache file.
                if decoded_frames != expected_shape[0]:
                    mmap.flush()
                    compact_path = cache_dir / "workspace_frames_compact.dat"
                    compact = np.memmap(
                        compact_path, dtype=np.uint8, mode="w+",
                        shape=(decoded_frames, workspace.height, workspace.width, 3)
                    )
                    compact[:] = mmap[:decoded_frames]
                    compact.flush()
                    del compact
                    del mmap
                    frames_path.unlink(missing_ok=True)
                    compact_path.replace(frames_path)
                else:
                    del mmap

                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                self.backend_selected.emit(backend, backend_message)
                self.finished.emit(CacheBuildResult(
                    cache_dir=cache_dir,
                    backend=backend,
                    backend_message=backend_message,
                    reused_existing=False,
                ))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _build_opencv(
        self,
        *,
        mmap: np.memmap,
        reader: VideoReader,
        workspace: WorkspaceRect,
    ) -> int:
        self.backend_selected.emit(
            "OpenCV/FFmpeg CPU fallback",
            "Using OpenCV/FFmpeg CPU decoding; PyNvVideoCodec/NVDEC was not selected or unavailable.",
        )
        total = int(reader.frame_count)
        decoded = 0
        for frame_index, frame_bgr in reader.read_range(0, total):
            if self._cancelled:
                break
            x, y, w, h = workspace.xywh
            mmap[frame_index] = frame_bgr[y:y + h, x:x + w]
            decoded += 1
            if decoded == 1 or decoded % 10 == 0 or decoded == total:
                self.progress.emit(decoded, total)
        return decoded

    def _build_d3d11va(
        self,
        *,
        mmap: np.memmap,
        reader: VideoReader,
        workspace: WorkspaceRect,
    ) -> int:
        self.backend_selected.emit("FFmpeg D3D11VA", "Using FFmpeg D3D11VA hardware decoder.")
        decoded = 0
        x, y, w, h = workspace.xywh
        total = int(reader.frame_count)
        for frame_index, frame_bgr in iter_d3d11va_bgr_frames(
            self._video_path, width=reader.width, height=reader.height, start=0, end=total,
            crop_xywh=workspace.xywh,
        ):
            if self._cancelled:
                break
            mmap[frame_index] = frame_bgr
            decoded += 1
            if decoded == 1 or decoded % 10 == 0 or decoded == total:
                self.progress.emit(decoded, total)
        if not self._cancelled and decoded != total:
            raise RuntimeError(f"FFmpeg D3D11VA produced {decoded} of {total} expected frames.")
        return decoded

    def _build_native_d3d11(
        self,
        *,
        mmap: np.memmap,
        reader: VideoReader,
        workspace: WorkspaceRect,
    ) -> int:
        self.backend_selected.emit("Native FFmpeg D3D11VA", "Using native FFmpeg D3D11VA GPU ROI decoder.")
        total = int(reader.frame_count)
        decoded = 0
        for frame_index, frame_bgr, _message, _decode_ms in iter_native_d3d11_workspace_frames(
            self._video_path, x=workspace.x, y=workspace.y, width=workspace.width, height=workspace.height,
            start=0, end=total,
        ):
            if self._cancelled:
                break
            mmap[frame_index] = frame_bgr
            decoded += 1
            if decoded == 1 or decoded % 10 == 0 or decoded == total:
                self.progress.emit(decoded, total)
        if not self._cancelled and decoded != total:
            raise RuntimeError(f"Native D3D11 decoder produced {decoded} of {total} expected frames.")
        return decoded

    def _build_nvdec(
        self,
        *,
        mmap: np.memmap,
        reader: VideoReader,
        workspace: WorkspaceRect,
    ) -> int:
        import cupy as cp
        try:
            import PyNvVideoCodec as nvc
        except ImportError as exc:
            raise RuntimeError(
                "PyNvVideoCodec is not installed. Install it to enable NVIDIA NVDEC."
            ) from exc

        self.backend_selected.emit(
            "NVDEC (PyNvVideoCodec)",
            "Using NVIDIA NVDEC hardware decoder; preparing GPU-cropped workspace cache.",
        )
        decoder = nvc.ThreadedDecoder(
            enc_file_path=str(self._video_path),
            buffer_size=DEFAULT_NVDEC_GPU_BUFFER_FRAMES,
            gpu_id=self._gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )
        total = int(reader.frame_count)
        shape = (workspace.height, workspace.width, 3)
        nbytes = int(np.prod(shape))
        copy_stream = cp.cuda.Stream(non_blocking=True)

        class _PinnedSlot:
            def __init__(self) -> None:
                self.memory = cp.cuda.alloc_pinned_memory(nbytes)
                self.array = np.frombuffer(self.memory, dtype=np.uint8, count=nbytes).reshape(shape)
                self.event = cp.cuda.Event()
                self.index: Optional[int] = None
                self.frame_ref = None
                self.rgb_ref = None
                self.bgr_ref = None

        slots = [_PinnedSlot() for _ in range(DEFAULT_PINNED_TRANSFER_BUFFERS)]

        def commit(slot: _PinnedSlot) -> None:
            if slot.index is None:
                return
            slot.event.synchronize()
            mmap[slot.index] = slot.array
            slot.index = None
            slot.frame_ref = None
            slot.rgb_ref = None
            slot.bgr_ref = None

        decoded = 0
        output_index = 0
        while not self._cancelled:
            frames = decoder.get_batch_frames(DEFAULT_NVDEC_BATCH_FRAMES)
            if not frames:
                break
            for frame in frames:
                if self._cancelled or output_index >= total:
                    break
                slot = slots[output_index % len(slots)]
                commit(slot)

                try:
                    rgb_gpu = cp.from_dlpack(frame)
                except AttributeError:
                    rgb_gpu = cp.fromDlpack(frame)
                padded_surface = _nvdec_surface_is_padded(
                    rgb_gpu, (reader.height, reader.width)
                )
                if padded_surface:
                    cp.cuda.Device(int(self._gpu_id)).synchronize()

                x, y, w, h = workspace.xywh
                # PyNvVideoCodec RGB output is RGB HWC; the existing tracker uses BGR.
                bgr_workspace_gpu = cp.ascontiguousarray(rgb_gpu[y:y + h, x:x + w, ::-1])
                try:
                    bgr_workspace_gpu.get(
                        out=slot.array, stream=copy_stream, blocking=False
                    )
                except TypeError:
                    # Older CuPy releases may not expose ``blocking`` on get().
                    with copy_stream:
                        bgr_workspace_gpu.get(out=slot.array, stream=copy_stream)
                slot.event.record(copy_stream)
                slot.index = output_index
                # ``rgb_gpu`` is a DLPack view. Keep its PyNvVideoCodec source
                # frame alive until the asynchronous device-to-host copy ends.
                slot.frame_ref = frame
                slot.rgb_ref = rgb_gpu
                slot.bgr_ref = bgr_workspace_gpu
                output_index += 1
                decoded += 1
            if decoded == 1 or decoded % 10 == 0 or decoded >= total:
                self.progress.emit(decoded, total)

        for slot in slots:
            commit(slot)
        copy_stream.synchronize()
        return decoded
