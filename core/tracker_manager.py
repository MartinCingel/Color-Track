"""
core/tracker_manager.py
-----------------------
Central coordinator for all active trackers.

Responsibilities
----------------
* Factory: creates the right BaseTracker subclass from a TrackerConfig
* Batch processing: runs all trackers over [start_frame, end_frame) using
  a QThread worker, emitting Qt signals for progress bars
* Result storage: per-tracker list of FrameResult, keyed by tracker UID
* Re-run: supports partial re-run from a given frame (after user re-seeds)
* Thread safety: batch runs in a worker thread; UI reads results only after
  batch completes or via copy

The manager routes the frame representation preferred by each tracker:
GPU-oriented Hessian trackers receive GPU frames, while direct colour-marker
trackers can request the cached CPU BGR frame and avoid GPU-to-CPU downloads.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.frame_buffer import FrameBuffer
from tracking.base_tracker import (
    BaseTracker, FrameResult, InitPreview,
    TrackerConfig, TrackerStatus, TrackerType,
)
from tracking.point_fast import PointFastTracker
try:
    from tracking.trackers import (
        PointAccurateTracker, BlobSimpleTracker, BlobComplexTracker,
        CurveTracker, ColorAreaTracker,
    )
except Exception:
    PointAccurateTracker = BlobSimpleTracker = BlobComplexTracker = CurveTracker = ColorAreaTracker = None
from utils.profiling import BatchProfiler, profiling_enabled


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tracker(config: TrackerConfig) -> BaseTracker:
    """Instantiate the correct BaseTracker subclass for the given config."""
    mapping = {
        TrackerType.POINT_FAST:     PointFastTracker,
        TrackerType.POINT_ACCURATE: PointAccurateTracker,
        TrackerType.BLOB_SIMPLE:    BlobSimpleTracker,
        TrackerType.BLOB_COMPLEX:   BlobComplexTracker,
        TrackerType.CURVE:          CurveTracker,
        TrackerType.COLOR_AREA:     ColorAreaTracker,
    }
    cls = mapping.get(config.tracker_type)
    if cls is None:
        raise RuntimeError(f"Tracker type {config.tracker_type.value} requires the optional CuPy GPU runtime.")
    return cls(config)


@dataclass(frozen=True)
class _InitializationRequest:
    frame_index: int
    seed_row: float
    seed_col: float
    roi: object = None


@dataclass(frozen=True)
class LossRecord:
    first_uncertain_frame: int
    lost_frame: int


@dataclass(frozen=True)
class KalmanLearningSuggestion:
    uid: str
    name: str
    trusted_frames: int
    measurement_noise: float
    process_noise: float
    gate_sigma: float
    is_average: bool = False


# ---------------------------------------------------------------------------
# Batch worker (runs in QThread)
# ---------------------------------------------------------------------------

class BatchWorker(QObject):
    """
    QObject worker for offline batch tracking.

    Signals
    -------
    frame_done(uid, frame_index, status_str)
        Emitted after each frame for each tracker → drives individual
        progress bars in the panel.
    tracker_done(uid)
        Emitted when all frames for one tracker are processed.
    batch_done()
        Emitted when the full batch is complete.
    error(uid, message)
        Emitted on unhandled exception for a tracker.
    """

    frame_done   = pyqtSignal(str, int, str)   # uid, frame_idx, status
    tracker_done = pyqtSignal(str)
    batch_done   = pyqtSignal()
    error        = pyqtSignal(str, str)
    profile_ready = pyqtSignal(str)
    debug_report_ready = pyqtSignal(str, str)   # uid, report_dir
    lost = pyqtSignal(str, int, int)             # uid, lost frame, first uncertain frame

    def __init__(
        self,
        trackers:    Dict[str, BaseTracker],
        buffer:      FrameBuffer,
        start_frame: int,
        end_frame:   int,
        results:     Dict[str, List[FrameResult]],
        initializations: Dict[str, _InitializationRequest],
        track_backwards: bool,
    ) -> None:
        super().__init__()
        self._trackers    = trackers
        self._buffer      = buffer
        self._start       = start_frame
        self._end         = end_frame
        self._results     = results   # shared dict, written here, read by UI after done
        self._initializations = initializations
        self._track_backwards = bool(track_backwards)
        self._cancelled   = False
        self._paused_on_loss: set[str] = set()

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Process one sequential decode pass and optionally record timings."""
        active = [
            (uid, tracker) for uid, tracker in self._trackers.items()
            if tracker.status != TrackerStatus.PENDING
        ]
        if not active:
            self.batch_done.emit()
            return

        needs_gpu = any(not tracker.requires_cpu_frame for _, tracker in active)
        allow_workspace_cache = all(
            (not tracker.requires_cpu_frame) or tracker.accepts_workspace_frame
            for _, tracker in active
        )
        profiler = BatchProfiler(
            enabled=profiling_enabled(),
            start_frame=self._start,
            end_frame=self._end,
            tracker_metadata=[
                {
                    "uid": uid,
                    "name": tracker.name,
                    "tracker_type": tracker.tracker_type.value,
                    "requires_cpu_frame": str(bool(tracker.requires_cpu_frame)),
                }
                for uid, tracker in active
            ],
            needs_gpu=needs_gpu,
        )

        stream = self._buffer.iter_batch_frames(
            self._start,
            self._end,
            needs_gpu=needs_gpu,
            include_timing=profiler.enabled,
            allow_workspace_cache=allow_workspace_cache,
        )

        for item in stream:
            if self._cancelled:
                break

            if profiler.enabled:
                frame_idx, frame_cpu, frame_gpu, io_metrics = item
            else:
                frame_idx, frame_cpu, frame_gpu = item
                io_metrics = {}

            frame_loop_start = perf_counter() if profiler.enabled else 0.0
            trackers_ms = 0.0
            signal_enqueue_ms = 0.0

            for uid, tracker in active:
                if self._cancelled:
                    break
                # The UI may remove a tracker while this worker is running.
                # ``active`` is intentionally a snapshot for a stable decode
                # pass, so verify the shared manager still owns this instance
                # before doing work or writing its result.
                if self._trackers.get(uid) is not tracker or uid not in self._results:
                    continue
                if uid in self._paused_on_loss:
                    continue
                request = self._initializations.get(uid)
                tracker_end = tracker.config.end_frame
                if tracker_end is not None and frame_idx > int(tracker_end):
                    continue
                if request is not None and frame_idx < request.frame_index:
                    # A tracker state is defined by its seed.  Never apply that
                    # state to frames before the seed unless the dedicated
                    # reverse pass below has been requested.
                    continue
                if (
                    self._track_backwards
                    and request is not None
                    and frame_idx <= request.frame_index
                ):
                    # The tracker is already seeded on this frame.  Let the
                    # reverse clone own earlier frames, and begin this normal
                    # forward state on the next frame.
                    continue
                frame = frame_cpu if tracker.requires_cpu_frame else frame_gpu
                try:
                    tracker_start = perf_counter() if profiler.enabled else 0.0
                    result = tracker.process_frame(frame, frame_idx)
                    process_ms = (perf_counter() - tracker_start) * 1000.0 if profiler.enabled else 0.0
                    trackers_ms += process_ms
                    results = self._results.get(uid)
                    if results is None:
                        continue
                    results.append(result)

                    if result.status == TrackerStatus.LOST and tracker.config.pause_on_loss:
                        first_uncertain = int(frame_idx)
                        for previous in reversed(results[:-1]):
                            if previous.status != TrackerStatus.UNCERTAIN:
                                break
                            first_uncertain = int(previous.frame_index)
                        self._paused_on_loss.add(uid)
                        self.lost.emit(uid, int(frame_idx), first_uncertain)

                    pop_reports = getattr(tracker, "pop_debug_reports", None)
                    if callable(pop_reports):
                        for report_dir in pop_reports():
                            self.debug_report_ready.emit(uid, str(report_dir))

                    signal_start = perf_counter() if profiler.enabled else 0.0
                    self.frame_done.emit(uid, frame_idx, result.status.value)
                    if profiler.enabled:
                        signal_enqueue_ms += (perf_counter() - signal_start) * 1000.0
                        row = {
                            "frame_index": int(frame_idx),
                            "uid": uid,
                            "tracker_name": tracker.name,
                            "tracker_type": tracker.tracker_type.value,
                            "status": result.status.value,
                            "process_ms": float(process_ms),
                        }
                        extra = getattr(tracker, "last_profile_metrics", {})
                        if isinstance(extra, dict):
                            row.update(extra)
                        profiler.add_tracker(row)
                except Exception as exc:
                    self.error.emit(uid, str(exc))

            if profiler.enabled:
                profiler.add_frame({
                    "frame_index": int(frame_idx),
                    **io_metrics,
                    "trackers_ms": float(trackers_ms),
                    "signal_enqueue_ms": float(signal_enqueue_ms),
                    "frame_loop_ms": float((perf_counter() - frame_loop_start) * 1000.0),
                    "active_tracker_count": len(active),
                })

        if self._track_backwards and not self._cancelled:
            self._run_backward_pass(active)

        for uid, _tracker in active:
            results = self._results.get(uid)
            if results is not None:
                results.sort(key=lambda result: int(result.frame_index))

        if profiler.enabled:
            report_dir = profiler.write_report(cancelled=self._cancelled)
            if report_dir is not None:
                self.profile_ready.emit(str(report_dir))

        for uid, _tracker in active:
            if uid in self._results:
                self.tracker_done.emit(uid)

        self.batch_done.emit()

    def _run_backward_pass(self, active: list[tuple[str, BaseTracker]]) -> None:
        """Track back from each seed with fresh state, then merge the results."""
        requests = {
            uid: request
            for uid, _tracker in active
            if (request := self._initializations.get(uid)) is not None
            and self._start < request.frame_index < self._end
        }
        if not requests:
            return

        clones = {
            uid: create_tracker(copy.deepcopy(tracker.config))
            for uid, tracker in active
            if uid in requests
        }
        initialized: set[str] = set()
        needs_gpu = any(not tracker.requires_cpu_frame for tracker in clones.values())
        reverse_end = max(request.frame_index for request in requests.values()) + 1

        for frame_index, frame_cpu, frame_gpu in self._buffer.iter_batch_frames_reverse(
            self._start,
            reverse_end,
            needs_gpu=needs_gpu,
        ):
            if self._cancelled:
                return
            for uid, tracker in clones.items():
                if self._trackers.get(uid) is None or uid not in self._results:
                    continue
                request = requests[uid]
                frame = frame_cpu if tracker.requires_cpu_frame else frame_gpu
                if frame is None:
                    continue
                try:
                    if frame_index == request.frame_index:
                        tracker.initialize(frame, request.seed_row, request.seed_col, request.roi)
                        initialized.add(uid)
                    elif frame_index < request.frame_index and uid in initialized:
                        result = tracker.process_frame(frame, frame_index)
                        results = self._results.get(uid)
                        if results is None:
                            continue
                        results.append(result)
                        self.frame_done.emit(uid, frame_index, result.status.value)
                except Exception as exc:
                    self.error.emit(uid, f'Backward tracking failed at frame {frame_index}: {exc}')
                    initialized.discard(uid)


# ---------------------------------------------------------------------------
# TrackerManager
# ---------------------------------------------------------------------------

class TrackerManager(QObject):
    """
    Owns all active BaseTracker instances and orchestrates batch processing.

    Parameters
    ----------
    buffer : FrameBuffer  Shared frame buffer (GPU-backed).
    """

    # Forwarded from BatchWorker
    frame_done   = pyqtSignal(str, int, str)
    tracker_done = pyqtSignal(str)
    batch_done   = pyqtSignal()
    batch_error  = pyqtSignal(str, str)
    debug_report_ready = pyqtSignal(str, str)
    tracker_lost = pyqtSignal(str, int, int)

    def __init__(self, buffer: FrameBuffer, parent=None) -> None:
        super().__init__(parent)
        self._buffer:   FrameBuffer              = buffer
        self._trackers: Dict[str, BaseTracker]   = {}   # uid → tracker
        self._results:  Dict[str, List[FrameResult]] = {}
        self._initializations: Dict[str, _InitializationRequest] = {}
        self._active_tracker_uids: set[str] = set()
        self._lost_trackers: Dict[str, LossRecord] = {}

        self._worker: Optional[BatchWorker] = None
        self._thread: Optional[QThread]     = None
        self._last_profile_report_dir: Optional[str] = None
        self._kalman_suggestions: list[KalmanLearningSuggestion] = []

    # ------------------------------------------------------------------
    # Tracker lifecycle
    # ------------------------------------------------------------------

    def add_tracker(self, config: TrackerConfig) -> BaseTracker:
        """Create and register a new tracker. Returns the instance."""
        if not config.name.strip():
            existing_names = {tracker.name for tracker in self._trackers.values()}
            next_index = 1
            while f"tracker_{next_index}" in existing_names:
                next_index += 1
            config.name = f"tracker_{next_index}"
        tracker = create_tracker(config)
        self._trackers[tracker.uid] = tracker
        self._results[tracker.uid]  = []
        self._active_tracker_uids.add(tracker.uid)
        self._lost_trackers.pop(tracker.uid, None)
        return tracker

    def remove_tracker(self, uid: str) -> None:
        self._trackers.pop(uid, None)
        self._results.pop(uid, None)
        self._initializations.pop(uid, None)
        self._active_tracker_uids.discard(uid)
        self._lost_trackers.pop(uid, None)

    def get_tracker(self, uid: str) -> Optional[BaseTracker]:
        return self._trackers.get(uid)

    def all_trackers(self) -> List[BaseTracker]:
        return list(self._trackers.values())

    def set_tracker_active(self, uid: str, active: bool) -> None:
        if uid not in self._trackers:
            return
        if active:
            self._active_tracker_uids.add(uid)
        else:
            self._active_tracker_uids.discard(uid)

    def loss_record(self, uid: str) -> Optional[LossRecord]:
        return self._lost_trackers.get(uid)

    def is_tracker_active(self, uid: str) -> bool:
        return uid in self._active_tracker_uids

    def set_tracker_end_frame(self, uid: str, end_frame: Optional[int]) -> None:
        tracker = self._trackers.get(uid)
        if tracker is not None:
            tracker.config.end_frame = None if end_frame is None else int(end_frame)

    def restart_tracker_from_scratch(
        self,
        uid: str,
        frame_index: int,
        seed_row: float,
        seed_col: float,
        roi=None,
        frame_override=None,
    ) -> Optional[InitPreview]:
        """Replace a lost tracker with a fresh instance seeded at ``frame_index``."""
        previous = self._trackers.get(uid)
        if previous is None:
            return None
        tracker = create_tracker(copy.deepcopy(previous.config))
        frame = frame_override if frame_override is not None else self._frame_for_tracker(tracker, frame_index)
        if frame is None:
            return None
        preview = tracker.initialize(frame, seed_row, seed_col, roi)
        self._trackers[uid] = tracker
        self._initializations[uid] = _InitializationRequest(
            frame_index=int(frame_index),
            seed_row=float(seed_row),
            seed_col=float(seed_col),
            roi=tuple(roi) if roi is not None else None,
        )
        self._results[uid] = [
            result for result in self._results.get(uid, [])
            if result.frame_index < int(frame_index)
        ]
        self._lost_trackers.pop(uid, None)
        self._active_tracker_uids.add(uid)
        return preview

    def _on_worker_tracker_lost(
        self, uid: str, lost_frame: int, first_uncertain_frame: int
    ) -> None:
        self._active_tracker_uids.discard(uid)
        self._lost_trackers[uid] = LossRecord(
            first_uncertain_frame=int(first_uncertain_frame),
            lost_frame=int(lost_frame),
        )
        self.tracker_lost.emit(uid, int(lost_frame), int(first_uncertain_frame))

    @property
    def last_profile_report_dir(self) -> Optional[str]:
        """Directory containing the most recent opt-in batch profile report."""
        return self._last_profile_report_dir

    def initialize_tracker(
        self,
        uid: str,
        frame_index: int,
        seed_row: float,
        seed_col: float,
        roi=None,
        frame_override=None,
    ) -> Optional[InitPreview]:
        """
        Initialize a tracker on the given frame.
        Downloads the frame from the buffer (sync) and calls tracker.initialize().
        """
        tracker = self._trackers.get(uid)
        if tracker is None:
            return None

        frame = frame_override if frame_override is not None else self._frame_for_tracker(tracker, frame_index)
        if frame is None:
            return None

        preview = tracker.initialize(frame, seed_row, seed_col, roi)
        self._initializations[uid] = _InitializationRequest(
            frame_index=int(frame_index),
            seed_row=float(seed_row),
            seed_col=float(seed_col),
            roi=tuple(roi) if roi is not None else None,
        )
        # Clear old results so re-run starts fresh
        self._results[uid] = []
        return preview

    def preview_initialization_edit(
        self,
        uid: str,
        parameters: dict,
        scissors_cuts: list,
    ) -> Optional[InitPreview]:
        tracker = self._trackers.get(uid)
        if tracker is None:
            return None
        editor = getattr(tracker, "preview_initialization_edit", None)
        if not callable(editor):
            return None
        return editor(parameters, scissors_cuts)

    def apply_initialization_edit(
        self,
        uid: str,
        parameters: dict,
        scissors_cuts: list,
    ) -> Optional[InitPreview]:
        tracker = self._trackers.get(uid)
        if tracker is None:
            return None
        editor = getattr(tracker, "apply_initialization_edit", None)
        if not callable(editor):
            return None
        preview = editor(parameters, scissors_cuts)
        self._results[uid] = []
        return preview

    def reinitialize_tracker(
        self,
        uid: str,
        frame_index: int,
        seed_row: float,
        seed_col: float,
    ) -> Optional[InitPreview]:
        """Re-seed a tracker after lock loss."""
        tracker = self._trackers.get(uid)
        if tracker is None:
            return None

        frame = self._frame_for_tracker(tracker, frame_index)
        if frame is None:
            return None

        preview = tracker.reinitialize(frame, seed_row, seed_col, frame_index)
        self._initializations[uid] = _InitializationRequest(
            frame_index=int(frame_index),
            seed_row=float(seed_row),
            seed_col=float(seed_col),
        )

        # Truncate results at re-seed frame so re-run overwrites from here
        existing = self._results.get(uid, [])
        self._results[uid] = [r for r in existing if r.frame_index < frame_index]
        return preview

    def _frame_for_tracker(self, tracker: BaseTracker, frame_index: int):
        """Return CPU/GPU/full/workspace frame according to tracker capability.

        If a prepared workspace cache is active and the tracker explicitly
        accepts WorkspaceFrame, initialization and recovery use the same
        cached/NVDEC decode path as batch tracking. If cache access fails, we
        safely fall back to the normal full CPU frame.
        """
        if tracker.requires_cpu_frame:
            if tracker.accepts_workspace_frame:
                cached = self._buffer.get_analysis_frame(frame_index)
                if cached is not None:
                    return cached
            return self._buffer.get_frame_cpu(frame_index)
        return self._buffer.get_frame(frame_index)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def start_batch(
        self,
        start_frame: int,
        end_frame: int,
        *,
        track_backwards: bool = False,
    ) -> None:
        """
        Launch batch processing in a background QThread.
        Emits batch_done when complete.
        """
        if self._thread and self._thread.isRunning():
            return   # Already running

        # Batch decoding is sequential and independent of interactive
        # scrubbing. Pause background prefetch so it does not decode the same
        # video competitively during the batch.
        self._buffer.begin_batch()
        self._last_profile_report_dir = None

        self._thread = QThread()
        active_trackers = {
            uid: tracker
            for uid, tracker in self._trackers.items()
            if uid in self._active_tracker_uids
        }
        self._worker = BatchWorker(
            trackers=active_trackers,
            buffer=self._buffer,
            start_frame=start_frame,
            end_frame=end_frame,
            results=self._results,
            initializations=dict(self._initializations),
            track_backwards=track_backwards,
        )
        self._worker.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._worker.run)
        self._worker.frame_done.connect(self.frame_done)
        self._worker.tracker_done.connect(self.tracker_done)
        self._worker.batch_done.connect(self._on_batch_done)
        self._worker.error.connect(self.batch_error)
        self._worker.profile_ready.connect(self._on_profile_ready)
        self._worker.debug_report_ready.connect(self.debug_report_ready)
        self._worker.lost.connect(self._on_worker_tracker_lost)

        self._thread.start()

    def cancel_batch(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_profile_ready(self, report_dir: str) -> None:
        self._last_profile_report_dir = report_dir

    def _on_batch_done(self) -> None:
        self._buffer.end_batch()
        self._kalman_suggestions = self._learn_kalman_suggestions()
        self.batch_done.emit()
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.isRunning())

    def kalman_learning_suggestions(self) -> list[KalmanLearningSuggestion]:
        """Return post-run global motion suggestions, including an average row."""
        return list(self._kalman_suggestions)

    def learned_preset_payload(self, uid: str) -> Optional[dict]:
        tracker = self._trackers.get(uid)
        exporter = getattr(tracker, 'export_learned_validation_profile', None)
        if tracker is None or not callable(exporter):
            return None
        suggestion = next((item for item in self._kalman_suggestions if item.uid == uid), None)
        if suggestion is None:
            return None
        return {
            'validation_samples': exporter(),
            'kalman': {
                'measurement_noise': suggestion.measurement_noise,
                'process_noise': suggestion.process_noise,
                'gate_sigma': suggestion.gate_sigma,
            },
        }

    def _learn_kalman_suggestions(self) -> list[KalmanLearningSuggestion]:
        suggestions: list[KalmanLearningSuggestion] = []
        for uid, tracker in self._trackers.items():
            seed = self._initializations.get(uid)
            rows = [
                result for result in self._results.get(uid, [])
                if result.status == TrackerStatus.LOCKED
                and result.center is not None
                and (seed is None or result.frame_index >= seed.frame_index)
            ]
            rows.sort(key=lambda result: int(result.frame_index))
            if len(rows) < 8:
                continue
            centers = np.asarray([result.center for result in rows], dtype=np.float64)
            indices = np.asarray([result.frame_index for result in rows], dtype=np.int64)
            residuals: list[float] = []
            accelerations: list[float] = []
            for index in range(1, len(rows) - 1):
                if indices[index] - indices[index - 1] != 1 or indices[index + 1] - indices[index] != 1:
                    continue
                residuals.append(float(np.linalg.norm(centers[index] - (centers[index - 1] + centers[index + 1]) * 0.5)))
                accelerations.append(float(np.linalg.norm(centers[index + 1] - 2.0 * centers[index] + centers[index - 1])))
            if len(residuals) < 5 or len(accelerations) < 5:
                continue
            # A learned setting must be a conservative operating envelope, not
            # the clean median of a successful run.  Use high quantiles and
            # never recommend a materially tighter filter than the current
            # user configuration without an explicit future "aggressive" mode.
            base_measurement = max(0.05, float(getattr(tracker.config, 'measurement_noise', 0.7)))
            base_process = max(0.001, float(getattr(tracker.config, 'process_noise', 0.5)))
            base_gate = max(1.0, float(getattr(tracker.config, 'kalman_gate_sigma', 4.0)))
            measurement = float(max(
                base_measurement * 0.75,
                float(np.quantile(np.asarray(residuals), 0.90)) / np.sqrt(2.0),
            ))
            process = float(max(
                base_process * 0.75,
                float(np.quantile(np.asarray(accelerations), 0.90)) / np.sqrt(2.0),
            ))
            innovation_q = float(np.quantile(np.asarray(residuals), 0.995))
            gate = float(np.clip(
                max(base_gate, innovation_q / max(measurement * np.sqrt(2.0), 1e-6)),
                base_gate,
                12.0,
            ))
            suggestions.append(KalmanLearningSuggestion(
                uid=uid,
                name=tracker.name or uid,
                trusted_frames=len(rows),
                measurement_noise=measurement,
                process_noise=process,
                gate_sigma=gate,
            ))
        if suggestions:
            suggestions.append(KalmanLearningSuggestion(
                uid='average',
                name='Average',
                trusted_frames=int(round(float(np.mean([item.trusted_frames for item in suggestions])))),
                measurement_noise=float(np.mean([item.measurement_noise for item in suggestions])),
                process_noise=float(np.mean([item.process_noise for item in suggestions])),
                gate_sigma=float(np.mean([item.gate_sigma for item in suggestions])),
                is_average=True,
            ))
        return suggestions

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_results(self, uid: str) -> List[FrameResult]:
        return self._results.get(uid, [])

    def get_result_at_frame(self, uid: str, frame_index: int) -> Optional[FrameResult]:
        for r in self._results.get(uid, []):
            if r.frame_index == frame_index:
                return r
        return None

    def uncertain_frames(self, uid: str) -> List[int]:
        """Return list of frame indices where status is UNCERTAIN or LOST."""
        return [
            r.frame_index for r in self._results.get(uid, [])
            if r.status in (TrackerStatus.UNCERTAIN, TrackerStatus.LOST)
        ]

    def uncertain_count(self, uid: str) -> int:
        """Return the number of UNCERTAIN/LOST rows for one tracker."""
        return len(self.uncertain_frames(uid))

    def all_uncertain_results(self):
        """Yield (uid, tracker, result) for all UNCERTAIN/LOST rows."""
        for uid, tracker in self._trackers.items():
            for result in self._results.get(uid, []):
                if result.status in (TrackerStatus.UNCERTAIN, TrackerStatus.LOST):
                    yield uid, tracker, result

    def apply_uncertain_correction(
        self,
        uid: str,
        frame_index: int,
        center_rc,
        source: str,
    ) -> bool:
        """Apply a user-selected coordinate to an uncertain/lost frame.

        The status remains UNCERTAIN/LOST so exported files still show that the
        coordinate was manually reviewed; the coordinate itself is stored in
        ``FrameResult.center`` and ``correction_source`` records where it came
        from.
        """
        import numpy as _np
        for result in self._results.get(uid, []):
            if int(result.frame_index) == int(frame_index):
                result.center = _np.asarray(center_rc, dtype=_np.float32)
                result.correction_source = str(source)
                return True
        return False

    def clear_results(self) -> None:
        for uid in self._results:
            self._results[uid] = []
