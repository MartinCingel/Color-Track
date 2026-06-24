"""Dialog for reviewing UNCERTAIN/LOST point-tracker frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QKeySequence, QMouseEvent, QPixmap, QShortcut, QWheelEvent
from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton,
    QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.frame_buffer import FrameBuffer
from core.tracker_manager import TrackerManager
from tracking.base_tracker import FrameResult


@dataclass
class _ReviewEntry:
    uid: str
    tracker_name: str
    frame_index: int
    result: FrameResult


class _ZoomCanvas(QLabel):
    """Zoomable image canvas that emits full-image [row, col] clicks."""

    clicked_rc = pyqtSignal(float, float)
    zoom_requested = pyqtSignal(float, float, float)  # factor, image row, image col

    def __init__(self) -> None:
        super().__init__("No frame")
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setStyleSheet("background:#111; color:#aaa;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._image_bgr: Optional[np.ndarray] = None
        self._entry: Optional[_ReviewEntry] = None
        self._zoom: float = 1.0
        self._manual_rc: Optional[tuple[float, float]] = None
        self._selected_action: str = "colour_peak"

    def set_frame(self, frame_bgr: Optional[np.ndarray], entry: Optional[_ReviewEntry]) -> None:
        self._image_bgr = frame_bgr
        self._entry = entry
        self._manual_rc = None
        self._render()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.1, min(32.0, float(zoom)))
        self._render()

    def zoom(self) -> float:
        return self._zoom

    def set_manual_point(self, row: float, col: float) -> None:
        self._manual_rc = (float(row), float(col))
        self._render()

    def set_selected_action(self, action: str) -> None:
        self._selected_action = str(action)
        self._render()

    @staticmethod
    def _draw_cross(img: np.ndarray, center_rc, color, size: int = 8, thickness: int = 1) -> None:
        if center_rc is None:
            return
        row, col = float(center_rc[0]), float(center_rc[1])
        x, y = int(round(col)), int(round(row))
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, size, thickness)

    def _draw_markers(self, img: np.ndarray, result: FrameResult) -> None:
        # Existing corrected/measured centre = green circle.
        if result.center is not None:
            row, col = float(result.center[0]), float(result.center[1])
            cv2.circle(img, (int(round(col)), int(round(row))), 6, (0, 255, 0), 2)

        def draw_selected_ring(center_rc, color, action: str) -> None:
            if center_rc is None or self._selected_action != action:
                return
            row, col = float(center_rc[0]), float(center_rc[1])
            cv2.circle(
                img,
                (int(round(col)), int(round(row))),
                14,
                color,
                2,
            )

        alt = getattr(result, 'alt_centers', {}) or {}
        kalman = alt.get('kalman_mean')
        peak = alt.get('colour_peak')
        peak_selected = self._selected_action == 'colour_peak'
        kalman_selected = self._selected_action == 'kalman_mean'
        self._draw_cross(img, kalman, (0, 255, 255), 17 if kalman_selected else 13, 2 if kalman_selected else 1)
        self._draw_cross(img, peak, (255, 0, 255), 15 if peak_selected else 11, 2 if peak_selected else 1)
        draw_selected_ring(kalman, (0, 255, 255), 'kalman_mean')
        draw_selected_ring(peak, (255, 0, 255), 'colour_peak')

        if self._manual_rc is not None:
            row, col = self._manual_rc
            manual_selected = self._selected_action == 'manual'
            cv2.drawMarker(
                img,
                (int(round(col)), int(round(row))),
                (255, 128, 0),
                cv2.MARKER_TILTED_CROSS,
                21 if manual_selected else 17,
                3 if manual_selected else 2,
            )
            cv2.circle(img, (int(round(col)), int(round(row))), 9 if manual_selected else 5, (255, 128, 0), 2 if manual_selected else 1)

    def _render(self) -> None:
        if self._image_bgr is None:
            self.setPixmap(QPixmap())
            self.setText("No frame")
            self.resize(320, 240)
            return
        bgr = np.ascontiguousarray(self._image_bgr.copy())
        if self._entry is not None:
            self._draw_markers(bgr, self._entry.result)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(image)
        if abs(self._zoom - 1.0) > 1e-6:
            pix = pix.scaled(
                max(1, int(round(w * self._zoom))),
                max(1, int(round(h * self._zoom))),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self.setPixmap(pix)
        self.resize(pix.size())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image_bgr is None or self.pixmap() is None:
            return
        self.setFocus()
        x = event.position().x()
        y = event.position().y()
        img_h, img_w = self._image_bgr.shape[:2]
        col = x / max(self._zoom, 1e-9)
        row = y / max(self._zoom, 1e-9)
        if 0 <= row < img_h and 0 <= col < img_w:
            self.clicked_rc.emit(float(row), float(col))

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image_bgr is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        col = event.position().x() / max(self._zoom, 1e-9)
        row = event.position().y() / max(self._zoom, 1e-9)
        self.zoom_requested.emit(float(factor), float(row), float(col))
        event.accept()


class _FrameView(QWidget):
    """Scrollable zoom view used for manual correction of tiny features."""

    clicked_rc = pyqtSignal(float, float)
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    previous_action_requested = pyqtSignal()
    next_action_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas = _ZoomCanvas()
        self._canvas.clicked_rc.connect(self.clicked_rc)
        self._canvas.zoom_requested.connect(self._zoom_at)
        self._scroll = QScrollArea()
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(False)
        self._scroll.setMinimumSize(640, 420)
        self._scroll.setStyleSheet("background:#111;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._scroll, stretch=1)

        controls = QHBoxLayout()
        self._zoom_label = QLabel("100%")
        self._zoom_out = QPushButton("−")
        self._zoom_out.setToolTip("Zoom out")
        self._zoom_in = QPushButton("+")
        self._zoom_in.setToolTip("Zoom in")
        self._zoom_100 = QPushButton("100%")
        self._zoom_100.setToolTip("Show true pixel size")
        self._zoom_fit = QPushButton("Fit")
        self._zoom_fit.setToolTip("Fit frame to the current view")
        self._prev_btn = QPushButton("Q Prev")
        self._prev_btn.setToolTip("Previous uncertain frame (Q)")
        self._next_btn = QPushButton("E Next")
        self._next_btn.setToolTip("Next uncertain frame (E)")
        self._zoom_out.clicked.connect(lambda: self._set_zoom(self._canvas.zoom() / 1.5))
        self._zoom_in.clicked.connect(lambda: self._set_zoom(self._canvas.zoom() * 1.5))
        self._zoom_100.clicked.connect(lambda: self._set_zoom(1.0))
        self._zoom_fit.clicked.connect(self._fit_to_view)
        self._prev_btn.clicked.connect(self.previous_requested)
        self._next_btn.clicked.connect(self.next_requested)
        controls.addWidget(QLabel("Zoom:"))
        controls.addWidget(self._zoom_out)
        controls.addWidget(self._zoom_in)
        controls.addWidget(self._zoom_100)
        controls.addWidget(self._zoom_fit)
        controls.addWidget(self._zoom_label)
        controls.addSpacing(16)
        controls.addWidget(self._prev_btn)
        controls.addWidget(self._next_btn)
        controls.addStretch(1)
        root.addLayout(controls)

    def set_frame(self, frame_bgr: Optional[np.ndarray], entry: Optional[_ReviewEntry]) -> None:
        self._canvas.set_frame(frame_bgr, entry)
        self._update_label()

    def set_manual_point(self, row: float, col: float) -> None:
        self._canvas.set_manual_point(row, col)
        # Keep selected point roughly visible when zoomed.
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        z = self._canvas.zoom()
        hbar.setValue(max(0, int(col * z - self._scroll.viewport().width() / 2)))
        vbar.setValue(max(0, int(row * z - self._scroll.viewport().height() / 2)))

    def set_selected_action(self, action: str) -> None:
        self._canvas.set_selected_action(action)

    def _zoom_at(self, factor: float, row: float, col: float) -> None:
        old_zoom = self._canvas.zoom()
        new_zoom = old_zoom * float(factor)
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        cursor_x = col * old_zoom - hbar.value()
        cursor_y = row * old_zoom - vbar.value()
        self._canvas.set_zoom(new_zoom)
        z = self._canvas.zoom()
        hbar.setValue(max(0, int(col * z - cursor_x)))
        vbar.setValue(max(0, int(row * z - cursor_y)))
        self._update_label()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        step = 50

        # Image panning deliberately uses WASD only. Arrow keys are reserved for
        # correction selection and uncertain-frame navigation.
        if key == Qt.Key.Key_A:
            hbar.setValue(hbar.value() - step)
        elif key == Qt.Key.Key_D:
            hbar.setValue(hbar.value() + step)
        elif key == Qt.Key.Key_W:
            vbar.setValue(vbar.value() - step)
        elif key == Qt.Key.Key_S:
            vbar.setValue(vbar.value() + step)
        elif key == Qt.Key.Key_Q or key == Qt.Key.Key_Up:
            self.previous_requested.emit()
        elif key == Qt.Key.Key_E or key == Qt.Key.Key_Down:
            self.next_requested.emit()
        elif key == Qt.Key.Key_Left:
            self.previous_action_requested.emit()
        elif key == Qt.Key.Key_Right:
            self.next_action_requested.emit()
        elif key == Qt.Key.Key_F:
            self._fit_to_view()
        elif key == Qt.Key.Key_1:
            self._set_zoom(1.0)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _set_zoom(self, zoom: float) -> None:
        old_zoom = self._canvas.zoom()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        cx = (hbar.value() + self._scroll.viewport().width() / 2) / max(old_zoom, 1e-9)
        cy = (vbar.value() + self._scroll.viewport().height() / 2) / max(old_zoom, 1e-9)
        self._canvas.set_zoom(zoom)
        new_zoom = self._canvas.zoom()
        hbar.setValue(max(0, int(cx * new_zoom - self._scroll.viewport().width() / 2)))
        vbar.setValue(max(0, int(cy * new_zoom - self._scroll.viewport().height() / 2)))
        self._update_label()

    def _fit_to_view(self) -> None:
        img = self._canvas._image_bgr
        if img is None:
            return
        h, w = img.shape[:2]
        vw = max(1, self._scroll.viewport().width())
        vh = max(1, self._scroll.viewport().height())
        self._set_zoom(min(vw / max(w, 1), vh / max(h, 1)))

    def _update_label(self) -> None:
        self._zoom_label.setText(f"{self._canvas.zoom() * 100:.0f}%")


class UncertainFrameReviewDialog(QDialog):
    """Review UNCERTAIN/LOST frames and optionally apply coordinates."""

    correction_applied = pyqtSignal(str, int)  # uid, frame_index

    def __init__(self, manager: TrackerManager, buffer: FrameBuffer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._buffer = buffer
        self._entries: list[_ReviewEntry] = []
        self._manual_rc: Optional[tuple[float, float]] = None
        self._selected_action: str = "colour_peak"
        self.setWindowTitle("Uncertain Frame Review")
        self.resize(1280, 820)
        self._load_entries()
        self._build_ui()
        if self._entries:
            self._list.setCurrentRow(0)

    def _load_entries(self) -> None:
        self._entries.clear()
        for uid, tracker, result in self._manager.all_uncertain_results():
            self._entries.append(_ReviewEntry(uid, tracker.name, int(result.frame_index), result))
        self._entries.sort(key=lambda e: (e.tracker_name, e.frame_index))

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self._list = QListWidget()
        for entry in self._entries:
            source = getattr(entry.result, 'correction_source', '')
            suffix = f"  [{source}]" if source else ""
            reasons = getattr(entry.result, 'failure_reasons', []) or []
            reason_text = f"  — {', '.join(str(r) for r in reasons[:3])}" if reasons else ""
            self._list.addItem(f"{entry.tracker_name}  frame {entry.frame_index}  {entry.result.status.value}{suffix}{reason_text}")
        self._list.currentRowChanged.connect(self._show_entry)
        left.addWidget(QLabel("Uncertain / lost frames"))
        left.addWidget(self._list, stretch=1)
        root.addLayout(left, stretch=0)

        right = QVBoxLayout()
        self._frame = _FrameView()
        self._frame.clicked_rc.connect(self._manual_clicked)
        self._frame.previous_requested.connect(self._select_previous)
        self._frame.next_requested.connect(self._select_next)
        self._frame.previous_action_requested.connect(lambda: self._cycle_action(-1))
        self._frame.next_action_requested.connect(lambda: self._cycle_action(1))
        right.addWidget(self._frame, stretch=3)

        buttons = QHBoxLayout()
        self._action_group = QButtonGroup(self)
        self._action_group.setExclusive(True)
        self._use_peak_btn = QPushButton("Colour peak")
        self._use_peak_btn.setCheckable(True)
        self._use_peak_btn.setToolTip("Select colour-peak correction. Press Enter to apply.")
        self._use_peak_btn.clicked.connect(lambda: self._set_selected_action('colour_peak'))
        self._action_group.addButton(self._use_peak_btn)
        buttons.addWidget(self._use_peak_btn)
        self._use_kalman_btn = QPushButton("Kalman mean")
        self._use_kalman_btn.setCheckable(True)
        self._use_kalman_btn.setToolTip("Select Kalman-mean correction. Press Enter to apply.")
        self._use_kalman_btn.clicked.connect(lambda: self._set_selected_action('kalman_mean'))
        self._action_group.addButton(self._use_kalman_btn)
        buttons.addWidget(self._use_kalman_btn)
        self._manual_btn = QPushButton("Manual click")
        self._manual_btn.setCheckable(True)
        self._manual_btn.setToolTip("Select manual-click correction. Press Enter after clicking the image.")
        self._manual_btn.clicked.connect(lambda: self._set_selected_action('manual'))
        self._action_group.addButton(self._manual_btn)
        buttons.addWidget(self._manual_btn)
        self._apply_selected_btn = QPushButton("Apply selected (Enter)")
        self._apply_selected_btn.clicked.connect(self._apply_selected)
        buttons.addWidget(self._apply_selected_btn)
        right.addLayout(buttons)

        self._action_button_style = (
            "QPushButton:checked { background:#355C9A; color:white; font-weight:bold; }"
            "QPushButton { padding:4px 8px; }"
        )
        for button in (self._use_peak_btn, self._use_kalman_btn, self._manual_btn):
            button.setStyleSheet(self._action_button_style)
        QShortcut(QKeySequence("Return"), self).activated.connect(self._apply_selected)
        QShortcut(QKeySequence("Enter"), self).activated.connect(self._apply_selected)
        QShortcut(QKeySequence("Left"), self).activated.connect(lambda: self._cycle_action(-1))
        QShortcut(QKeySequence("Right"), self).activated.connect(lambda: self._cycle_action(1))
        QShortcut(QKeySequence("Up"), self).activated.connect(self._select_previous)
        QShortcut(QKeySequence("Down"), self).activated.connect(self._select_next)
        QShortcut(QKeySequence("Q"), self).activated.connect(self._select_previous)
        QShortcut(QKeySequence("E"), self).activated.connect(self._select_next)

        help_label = QLabel(
            "Overlay: magenta cross = colour peak, yellow cross = Kalman mean, green circle = currently saved coordinate, blue/orange cross = clicked manual point. "
            "Mouse wheel zooms around cursor. WASD pans the image. Q/Up and E/Down move between uncertain frames. Left/Right changes the highlighted correction choice. 1 = 100%, F = fit. Click the image to choose a manual point. Press Enter to apply the highlighted choice. If Manual click is highlighted but no point was clicked, Enter does nothing. Status remains UNCERTAIN/LOST so correction is traceable."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color:#AAAAAA; font-size:10px;")
        right.addWidget(help_label)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Field", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self._table, stretch=2)
        root.addLayout(right, stretch=1)


    def _set_selected_action(self, action: str) -> None:
        self._selected_action = str(action)
        self._use_peak_btn.setChecked(action == 'colour_peak')
        self._use_kalman_btn.setChecked(action == 'kalman_mean')
        self._manual_btn.setChecked(action == 'manual')
        self._frame.set_selected_action(action)

    def _enabled_actions(self) -> list[str]:
        actions: list[str] = []
        if self._use_peak_btn.isEnabled():
            actions.append('colour_peak')
        if self._use_kalman_btn.isEnabled():
            actions.append('kalman_mean')
        # Manual selection is useful even before a point is clicked. Pressing
        # Enter with no manual point intentionally does nothing.
        if self._manual_btn.isEnabled():
            actions.append('manual')
        return actions or ['manual']

    def _cycle_action(self, delta: int) -> None:
        actions = self._enabled_actions()
        try:
            idx = actions.index(self._selected_action)
        except ValueError:
            idx = 0
        self._set_selected_action(actions[(idx + int(delta)) % len(actions)])

    def _apply_selected(self) -> None:
        if self._selected_action == 'colour_peak':
            self._apply_alt('colour_peak')
        elif self._selected_action == 'kalman_mean':
            self._apply_alt('kalman_mean')
        elif self._selected_action == 'manual':
            if self._manual_rc is None:
                return
            self._apply_manual()

    def _select_previous(self) -> None:
        row = self._list.currentRow()
        if row > 0:
            self._list.setCurrentRow(row - 1)

    def _select_next(self) -> None:
        row = self._list.currentRow()
        if row < self._list.count() - 1:
            self._list.setCurrentRow(row + 1)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_Q or key == Qt.Key.Key_Up:
            self._select_previous()
            event.accept()
            return
        if key == Qt.Key.Key_E or key == Qt.Key.Key_Down:
            self._select_next()
            event.accept()
            return
        if key == Qt.Key.Key_Left:
            self._cycle_action(-1)
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self._cycle_action(1)
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def _current_entry(self) -> Optional[_ReviewEntry]:
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._entries):
            return None
        return self._entries[idx]

    def _show_entry(self, index: int) -> None:
        self._manual_rc = None
        entry = self._current_entry()
        if entry is None:
            return
        frame = self._buffer.get_frame_cpu(entry.frame_index)
        self._frame.set_frame(frame, entry)
        alt = getattr(entry.result, 'alt_centers', {}) or {}
        failure_reasons = getattr(entry.result, 'failure_reasons', []) or []
        diagnostics = getattr(entry.result, 'diagnostic_values', {}) or {}
        rows = [
            ("tracker", entry.tracker_name),
            ("frame", entry.frame_index),
            ("status", entry.result.status.value),
            ("failure_reasons", ", ".join(str(r) for r in failure_reasons) or ""),
            ("current_center_xy", None if entry.result.center is None else [float(entry.result.center[1]), float(entry.result.center[0])]),
            ("correction_source", getattr(entry.result, 'correction_source', '')),
            ("colour_peak_xy", None if alt.get('colour_peak') is None else [float(alt.get('colour_peak')[1]), float(alt.get('colour_peak')[0])]),
            ("kalman_mean_xy", None if alt.get('kalman_mean') is None else [float(alt.get('kalman_mean')[1]), float(alt.get('kalman_mean')[0])]),
            ("confidence", float(entry.result.hessian_score)),
        ]
        important_keys = [
            "point_mode", "predicted_rc", "kalman_mean_rc", "colour_peak_rc",
            "search_bbox_xywh", "centroid_window_xywh",
            "kalman_gate_radius", "innovation_distance",
            "probability_max", "tiny_peak_score", "tiny_peak_threshold",
            "tiny_probability_contrast", "tiny_contrast_threshold",
            "tiny_lab_contrast", "tiny_init_contrast_threshold",
            "tiny_direction_cosine", "tiny_direction_min_cosine",
            "components_found_last_threshold", "last_threshold",
        ]
        shown = set()
        for key in important_keys:
            if key in diagnostics:
                rows.append((key, diagnostics.get(key)))
                shown.add(key)
        for key in sorted(k for k in diagnostics.keys() if k not in shown):
            if key.startswith("reject_") or key in {"threshold_passes", "components_found", "components_scored", "accepted_candidate_count", "successful_threshold"}:
                rows.append((key, diagnostics.get(key)))
        self._table.setRowCount(len(rows))
        for r, (key, value) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(str(key)))
            self._table.setItem(r, 1, QTableWidgetItem(str(value)))
        has_peak = alt.get('colour_peak') is not None
        has_kalman = alt.get('kalman_mean') is not None
        self._use_peak_btn.setEnabled(has_peak)
        self._use_kalman_btn.setEnabled(has_kalman)
        self._manual_btn.setEnabled(True)
        if self._selected_action == 'colour_peak' and not has_peak:
            self._set_selected_action('kalman_mean' if has_kalman else 'manual')
        elif self._selected_action == 'kalman_mean' and not has_kalman:
            self._set_selected_action('colour_peak' if has_peak else 'manual')
        else:
            self._set_selected_action(self._selected_action)

    def _manual_clicked(self, row: float, col: float) -> None:
        self._manual_rc = (float(row), float(col))
        self._frame.set_manual_point(float(row), float(col))
        self._set_selected_action('manual')
        entry = self._current_entry()
        if entry is not None:
            self._table.setRowCount(self._table.rowCount() + 1)
            r = self._table.rowCount() - 1
            self._table.setItem(r, 0, QTableWidgetItem("clicked_manual_rc"))
            self._table.setItem(r, 1, QTableWidgetItem(f"[{row:.3f}, {col:.3f}]"))

    def _apply_alt(self, key: str) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        alt = getattr(entry.result, 'alt_centers', {}) or {}
        center = alt.get(key)
        if center is None:
            QMessageBox.information(self, "No coordinate", f"No {key} coordinate is available for this frame.")
            return
        self._apply(center, key)

    def _apply_manual(self) -> None:
        if self._manual_rc is None:
            QMessageBox.information(self, "No manual point", "Click the zoomed image first, then press this button.")
            return
        self._apply(np.asarray(self._manual_rc, dtype=np.float32), 'manual')

    def _apply(self, center_rc, source: str) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        ok = self._manager.apply_uncertain_correction(entry.uid, entry.frame_index, center_rc, source)
        if ok:
            self.correction_applied.emit(entry.uid, entry.frame_index)
            self._show_entry(self._list.currentRow())
            idx = self._list.currentRow()
            if idx >= 0:
                reasons = getattr(entry.result, 'failure_reasons', []) or []
                reason_text = f"  — {', '.join(str(r) for r in reasons[:3])}" if reasons else ""
                self._list.item(idx).setText(f"{entry.tracker_name}  frame {entry.frame_index}  {entry.result.status.value}  [{source}]{reason_text}")
            QTimer.singleShot(500, self._select_next)
        else:
            QMessageBox.warning(self, "Correction failed", "Could not update this frame result.")
