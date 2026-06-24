"""Detailed batch profiling utilities.

User-editable settings are defined directly below. Change ``ENABLE_PROFILING``
to switch report generation on or off; no command-line environment variables
are required.

Each profiled batch writes CSV files and a JSON summary into a timestamped
subdirectory.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional


# ---------------------------------------------------------------------------
# User-editable profiling settings
# ---------------------------------------------------------------------------

# Change this variable to turn detailed timing collection on or off.
ENABLE_PROFILING: bool = False

# Optional custom report folder. Example:
# PROFILE_OUTPUT_PARENT: Optional[str] = r"C:\\temp\\tracking_profiles"
# Leave as None to save to <project folder>\\tracking_profiles.
PROFILE_OUTPUT_PARENT: Optional[str] = None


def profiling_enabled() -> bool:
    """Return whether detailed tracking profiling is enabled."""
    return ENABLE_PROFILING


def profiling_parent_directory() -> Path:
    """Return the parent directory for timestamped batch reports."""
    if PROFILE_OUTPUT_PARENT:
        return Path(PROFILE_OUTPUT_PARENT).expanduser().resolve()
    from utils.app_paths import app_data_dir
    return (app_data_dir() / "tracking_profiles").resolve()


def _percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = proportion * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarise_numeric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and key != "frame_index":
                grouped[key].append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for key, values in sorted(grouped.items()):
        summary[key] = {
            "count": len(values),
            "total": float(sum(values)),
            "mean": float(sum(values) / len(values)),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "max": float(max(values)),
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        keys = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class BatchProfiler:
    """Collect per-frame and per-tracker timing rows for one offline batch."""

    enabled: bool
    start_frame: int
    end_frame: int
    tracker_metadata: list[dict[str, str]]
    needs_gpu: bool
    frame_rows: list[dict[str, Any]] = field(default_factory=list)
    tracker_rows: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=perf_counter)
    report_dir: Optional[Path] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        parent = profiling_parent_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = (parent / f"batch_{timestamp}").resolve()
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "RUNNING.txt").write_text(
            "Profiling is enabled. CSV and JSON reports will be written when the batch completes.\n",
            encoding="utf-8",
        )

    def add_frame(self, row: dict[str, Any]) -> None:
        if self.enabled:
            self.frame_rows.append(dict(row))

    def add_tracker(self, row: dict[str, Any]) -> None:
        if self.enabled:
            self.tracker_rows.append(dict(row))

    def write_report(self, *, cancelled: bool) -> Optional[Path]:
        if not self.enabled:
            return None

        if self.report_dir is None:
            parent = profiling_parent_directory()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.report_dir = (parent / f"batch_{timestamp}").resolve()
            self.report_dir.mkdir(parents=True, exist_ok=True)
        report_dir = self.report_dir

        _write_csv(report_dir / "batch_frames.csv", self.frame_rows)
        _write_csv(report_dir / "tracker_frames.csv", self.tracker_rows)

        by_tracker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.tracker_rows:
            by_tracker[str(row.get("uid", "unknown"))].append(row)

        elapsed_ms = (perf_counter() - self.started_at) * 1000.0
        summary = {
            "cancelled": bool(cancelled),
            "frame_range": {"start": self.start_frame, "end": self.end_frame},
            "requested_frames": max(0, self.end_frame - self.start_frame),
            "decoded_frames": len(self.frame_rows),
            "tracker_result_rows": len(self.tracker_rows),
            "needs_gpu_frame_upload": bool(self.needs_gpu),
            "wall_total_ms": float(elapsed_ms),
            "wall_total_s": float(elapsed_ms / 1000.0),
            "trackers": self.tracker_metadata,
            "frame_stage_statistics": _summarise_numeric_rows(self.frame_rows),
            "tracker_stage_statistics": {
                uid: _summarise_numeric_rows(rows)
                for uid, rows in by_tracker.items()
            },
        }
        with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        readme = report_dir / "README.txt"
        readme.write_text(
            "Precise tracking profile report\n"
            "===============================\n\n"
            "batch_frames.csv contains one row per decoded video frame.\n"
            "  decode_ms: sequential video decode for the frame.\n"
            "  gpu_upload_ms: CPU->GPU upload only if an active GPU tracker needs it.\n"
            "  trackers_ms: processing time for all active trackers.\n"
            "  signal_enqueue_ms: time spent queueing frame_done Qt signals; it does not\n"
            "    include later UI repaint work on the UI thread.\n"
            "  frame_loop_ms: worker time after the frame was decoded.\n\n"
            "tracker_frames.csv contains one row per tracker per video frame.\n"
            "  process_ms: total time spent in tracker.process_frame().\n"
            "  PointFastTracker additionally reports ROI area, Lab conversion, colour\n"
            "  probability, threshold/component scan, candidate geometry, and decision\n"
            "  times as well as rejection and threshold diagnostics.\n\n"
            "summary.json contains mean, p50, p95, maximum, and total timings.\n",
            encoding="utf-8",
        )
        running = report_dir / "RUNNING.txt"
        if running.exists():
            running.unlink()
        (report_dir / "COMPLETE.txt").write_text(
            "Profiling report generation completed.\n", encoding="utf-8"
        )
        return report_dir
