"""Rolling tracker-debug history and report export utilities.

Debug recording is intentionally opt-in.  While enabled, a tracker keeps a
small rolling in-memory history.  When lock is first uncertain/lost, that
history is exported as CSV/JSON plus ROI, likelihood-map, and overlay images.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

import cv2
import numpy as np


def default_debug_report_root() -> Path:
    """Return the project-local directory used for temporary debug reports."""
    from utils.app_paths import app_data_dir
    return app_data_dir() / "debug_reports"


def cleanup_debug_reports(report_root: Optional[Path] = None) -> None:
    """Remove temporary tracker-debug reports from previous runs.

    Debug reports are intentionally temporary tuning artifacts. The UI calls
    this before a new batch and when the program closes so ROI/heatmap PNGs do
    not accumulate on disk.
    """
    root = Path(report_root) if report_root is not None else default_debug_report_root()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


@dataclass
class TrackerDebugFrame:
    frame_index: int
    status: str
    point_mode: str
    failure_reasons: list[str] = field(default_factory=list)

    predicted_rc: Optional[list[float]] = None
    measured_rc: Optional[list[float]] = None
    kalman_state: Optional[list[float]] = None
    kalman_position_sigma: Optional[float] = None
    kalman_gate_radius: Optional[float] = None
    innovation_distance: Optional[float] = None

    confidence: float = 0.0
    colour_score: Optional[float] = None
    shape_score: Optional[float] = None
    motion_score: Optional[float] = None
    peak_score: Optional[float] = None
    contrast_score: Optional[float] = None
    successful_threshold: Optional[Any] = None

    search_bbox_xywh: Optional[list[int]] = None
    centroid_window_xywh: Optional[list[int]] = None
    peak_rc: Optional[list[float]] = None
    uncertain_export_rc: Optional[list[float]] = None
    uncertain_export_source: str = 'none'
    candidate_count: int = 0
    accepted: bool = False
    consecutive_misses: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    roi_bgr: Optional[np.ndarray] = None
    likelihood: Optional[np.ndarray] = None

    def to_jsonable(self) -> dict[str, Any]:
        data = {
            "frame_index": self.frame_index,
            "status": self.status,
            "point_mode": self.point_mode,
            "failure_reasons": self.failure_reasons,
            "predicted_rc": self.predicted_rc,
            "measured_rc": self.measured_rc,
            "kalman_state": self.kalman_state,
            "kalman_position_sigma": self.kalman_position_sigma,
            "kalman_gate_radius": self.kalman_gate_radius,
            "innovation_distance": self.innovation_distance,
            "confidence": self.confidence,
            "colour_score": self.colour_score,
            "shape_score": self.shape_score,
            "motion_score": self.motion_score,
            "peak_score": self.peak_score,
            "contrast_score": self.contrast_score,
            "successful_threshold": self.successful_threshold,
            "search_bbox_xywh": self.search_bbox_xywh,
            "centroid_window_xywh": self.centroid_window_xywh,
            "peak_rc": self.peak_rc,
            "uncertain_export_rc": self.uncertain_export_rc,
            "uncertain_export_source": self.uncertain_export_source,
            "candidate_count": self.candidate_count,
            "accepted": self.accepted,
            "consecutive_misses": self.consecutive_misses,
            "rejection_counts": self.rejection_counts,
            "extra": self.extra,
        }
        return data


class TrackerDebugRecorder:
    """Small rolling debug recorder for one tracker."""

    def __init__(
        self,
        *,
        enabled: bool,
        history_frames: int,
        tracker_uid: str,
        tracker_name: str,
        report_root: Optional[Path] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.history_frames = max(1, int(history_frames))
        self.tracker_uid = str(tracker_uid)
        self.tracker_name = str(tracker_name)
        self.report_root = (
            Path(report_root)
            if report_root is not None
            else default_debug_report_root()
        )
        self._history: Deque[TrackerDebugFrame] = deque(maxlen=self.history_frames)

    def append(self, record: TrackerDebugFrame) -> None:
        if self.enabled:
            self._history.append(record)

    def clear(self) -> None:
        self._history.clear()

    def export(self, *, event: str, frame_index: int) -> Optional[Path]:
        if not self.enabled or not self._history:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_name = self.tracker_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        report_dir = self.report_root / f"tracker_{self.tracker_uid}_{safe_name}_{event}_frame_{frame_index}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "roi_frames").mkdir(exist_ok=True)
        (report_dir / "likelihood_maps").mkdir(exist_ok=True)
        (report_dir / "overlay_frames").mkdir(exist_ok=True)

        records = list(self._history)
        json_records = [r.to_jsonable() for r in records]
        meta = {
            "tracker_uid": self.tracker_uid,
            "tracker_name": self.tracker_name,
            "event": event,
            "event_frame_index": int(frame_index),
            "history_frames": len(records),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with (report_dir / "debug.json").open("w", encoding="utf-8") as fh:
            json.dump({"metadata": meta, "frames": json_records}, fh, indent=2)

        self._write_csv(report_dir / "debug.csv", json_records)
        self._write_readme(report_dir / "README.txt")

        for record in records:
            self._write_frame_images(report_dir, record)
        return report_dir

    @staticmethod
    def _flatten_for_csv(data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                out[key] = json.dumps(value, separators=(",", ":"))
            else:
                out[key] = value
        return out

    def _write_csv(self, path: Path, records: list[dict[str, Any]]) -> None:
        flattened = [self._flatten_for_csv(r) for r in records]
        keys: list[str] = []
        for row in flattened:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flattened)

    @staticmethod
    def _write_readme(path: Path) -> None:
        path.write_text(
            "Tracker debug report\n"
            "====================\n\n"
            "debug.csv / debug.json contain full-frame coordinates.\n"
            "roi_frames/ contains the search ROI as seen by the tracker.\n"
            "likelihood_maps/ contains colour-likelihood heatmaps.\n"
            "overlay_frames/ contains annotated ROI images without text labels.\n\n"
            "Overlay legend:\n"
            "  green circle  = accepted measured position\n"
            "  yellow cross  = Kalman/predicted position\n"
            "  yellow circle = Kalman gate radius clipped to ROI view\n"
            "  magenta dot   = tiny-feature colour peak\n"
            "  white box     = tiny-feature centroid window\n"
            "  cyan border   = search ROI border\n",
            encoding="utf-8",
        )

    def _write_frame_images(self, report_dir: Path, record: TrackerDebugFrame) -> None:
        if record.roi_bgr is None:
            return
        frame_name = f"frame_{record.frame_index:06d}.png"
        roi = np.ascontiguousarray(record.roi_bgr)
        cv2.imwrite(str(report_dir / "roi_frames" / frame_name), roi)

        if record.likelihood is not None:
            prob = np.asarray(record.likelihood, dtype=np.float32)
            norm = np.clip(prob, 0.0, 1.0)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(str(report_dir / "likelihood_maps" / frame_name), heat)
        else:
            heat = None

        overlay = self._make_overlay(record, roi)
        cv2.imwrite(str(report_dir / "overlay_frames" / frame_name), overlay)

    @staticmethod
    def _global_rc_to_local_xy(
        rc: Optional[list[float]],
        bbox_xywh: Optional[list[int]],
    ) -> Optional[tuple[int, int]]:
        if rc is None or bbox_xywh is None:
            return None
        x0, y0, _, _ = bbox_xywh
        row, col = float(rc[0]), float(rc[1])
        return int(round(col - x0)), int(round(row - y0))

    def _make_overlay(self, record: TrackerDebugFrame, roi_bgr: np.ndarray) -> np.ndarray:
        overlay = np.ascontiguousarray(roi_bgr.copy())
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, 0), (max(0, w - 1), max(0, h - 1)), (255, 255, 0), 1)

        # Gate around prediction.
        pred_xy = self._global_rc_to_local_xy(record.predicted_rc, record.search_bbox_xywh)
        if pred_xy is not None:
            cv2.drawMarker(overlay, pred_xy, (0, 255, 255), cv2.MARKER_CROSS, 11, 1)
            if record.kalman_gate_radius is not None:
                radius = int(round(float(record.kalman_gate_radius)))
                if radius > 0:
                    cv2.circle(overlay, pred_xy, radius, (0, 255, 255), 1)

        measured_xy = self._global_rc_to_local_xy(record.measured_rc, record.search_bbox_xywh)
        if measured_xy is not None:
            cv2.circle(overlay, measured_xy, 4, (0, 255, 0), 1)
            cv2.circle(overlay, measured_xy, 1, (0, 255, 0), -1)

        peak_xy = self._global_rc_to_local_xy(record.peak_rc, record.search_bbox_xywh)
        if peak_xy is not None:
            cv2.circle(overlay, peak_xy, 3, (255, 0, 255), 1)

        if record.centroid_window_xywh is not None and record.search_bbox_xywh is not None:
            wx, wy, ww, wh = record.centroid_window_xywh
            sx, sy, _, _ = record.search_bbox_xywh
            p0 = (int(wx - sx), int(wy - sy))
            p1 = (int(wx - sx + ww - 1), int(wy - sy + wh - 1))
            cv2.rectangle(overlay, p0, p1, (255, 255, 255), 1)

        # Do not draw text into the ROI image: for tiny search windows the
        # labels cover the feature. Numeric details are shown in the replay
        # table and stored in debug.csv/debug.json instead.
        return overlay
