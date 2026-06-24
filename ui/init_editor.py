"""Embedded initialization diagnostics and mask editor for point-fast initialization."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from tracking.base_tracker import InitPreview


class _ZoomImageCanvas(QLabel):
    """Scrollable image surface that keeps clicks in crop-image coordinates."""

    point_clicked = pyqtSignal(float, float)
    zoom_requested = pyqtSignal(float, float, float)  # factor, row, col

    def __init__(self) -> None:
        super().__init__('No initialization diagnostics')
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setStyleSheet('background:#111; color:#888;')
        self.setMouseTracking(True)
        self._image_rgb: Optional[np.ndarray] = None
        self._zoom = 1.0
        self._interactive = False
        self._cuts: list[tuple[float, float, float, float]] = []
        self._pending_point: Optional[tuple[float, float]] = None

    def set_image(self, image_rgb: Optional[np.ndarray]) -> None:
        self._image_rgb = None if image_rgb is None else np.ascontiguousarray(image_rgb.astype(np.uint8))
        self._render()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.1, min(64.0, float(zoom)))
        self._render()

    def zoom(self) -> float:
        return self._zoom

    def image_shape(self) -> Optional[tuple[int, int]]:
        if self._image_rgb is None:
            return None
        return self._image_rgb.shape[:2]

    def set_interaction(self, enabled: bool) -> None:
        self._interactive = bool(enabled)
        self.setCursor(
            Qt.CursorShape.CrossCursor if self._interactive else Qt.CursorShape.ArrowCursor
        )

    def set_cuts(
        self,
        cuts: list[tuple[float, float, float, float]],
        pending_point: Optional[tuple[float, float]],
    ) -> None:
        self._cuts = list(cuts)
        self._pending_point = pending_point
        self._render()

    def _render(self) -> None:
        if self._image_rgb is None:
            self.setPixmap(QPixmap())
            self.setText('No initialization diagnostics')
            self.resize(320, 240)
            return

        image = self._image_rgb.copy()
        if self._interactive:
            for x1, y1, x2, y2 in self._cuts:
                cv2.line(
                    image, (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))), (255, 72, 72), 1, cv2.LINE_AA,
                )
            if self._pending_point is not None:
                x, y = self._pending_point
                cv2.circle(image, (int(round(x)), int(round(y))), 2, (255, 220, 0), 1, cv2.LINE_AA)

        h, w = image.shape[:2]
        qimage = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        if abs(self._zoom - 1.0) > 1e-6:
            pixmap = pixmap.scaled(
                max(1, int(round(w * self._zoom))),
                max(1, int(round(h * self._zoom))),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if not self._interactive or self._image_rgb is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        h, w = self._image_rgb.shape[:2]
        col = event.position().x() / max(self._zoom, 1e-9)
        row = event.position().y() / max(self._zoom, 1e-9)
        if 0 <= row < h and 0 <= col < w:
            self.point_clicked.emit(float(col), float(row))

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image_rgb is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        col = event.position().x() / max(self._zoom, 1e-9)
        row = event.position().y() / max(self._zoom, 1e-9)
        self.zoom_requested.emit(float(factor), float(row), float(col))
        event.accept()


class InitializationEditorPanel(QWidget):
    """Main-workspace editor. The parent controls its selected image view."""

    preview_requested = pyqtSignal(object, object)
    apply_requested = pyqtSignal(object, object)

    def __init__(self, uid: str, tracker_name: str, preview: InitPreview, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._uid = uid
        self._tracker_name = tracker_name
        self._cuts: list[tuple[float, float, float, float]] = []
        self._pending_cut_point: Optional[tuple[float, float]] = None
        self._last_preview: Optional[InitPreview] = None
        self._view_mode = 'overlay'
        self._images: dict[str, np.ndarray] = {}
        diagnostics = dict(getattr(preview, 'init_diagnostics', {}) or {})
        self._editable = bool(diagnostics.get('editor_available', False))

        self.setStyleSheet(
            'QWidget { background:#1a1a1a; color:#DDDDDD; }'
            'QGroupBox { color:#CCCCCC; border:1px solid #444; margin-top:8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 3px; }'
            'QPushButton, QToolButton { background:#2d2d2d; color:#DDDDDD; border:1px solid #555; padding:5px; }'
            'QPushButton:hover, QToolButton:hover { background:#3a3a3a; }'
        )
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        image_column = QVBoxLayout()
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setStyleSheet('QScrollArea { background:#111; border:1px solid #333; }')
        self._canvas = _ZoomImageCanvas()
        self._canvas.point_clicked.connect(self._on_image_clicked)
        self._canvas.zoom_requested.connect(self._zoom_at)
        self._scroll.setWidget(self._canvas)
        image_column.addWidget(self._scroll, stretch=1)

        zoom_controls = QHBoxLayout()
        self._zoom_label = QLabel('100%')
        self._zoom_out = QToolButton()
        self._zoom_out.setText('-')
        self._zoom_out.setToolTip('Zoom out')
        self._zoom_in = QToolButton()
        self._zoom_in.setText('+')
        self._zoom_in.setToolTip('Zoom in')
        self._zoom_actual = QToolButton()
        self._zoom_actual.setText('1:1')
        self._zoom_actual.setToolTip('Show true pixel size')
        self._zoom_fit = QToolButton()
        self._zoom_fit.setText('Fit')
        self._zoom_fit.setToolTip('Fit crop to the current view')
        self._zoom_out.clicked.connect(lambda: self._set_zoom(self._canvas.zoom() / 1.5))
        self._zoom_in.clicked.connect(lambda: self._set_zoom(self._canvas.zoom() * 1.5))
        self._zoom_actual.clicked.connect(lambda: self._set_zoom(1.0))
        self._zoom_fit.clicked.connect(self._fit_to_view)
        zoom_controls.addWidget(self._zoom_out)
        zoom_controls.addWidget(self._zoom_in)
        zoom_controls.addWidget(self._zoom_actual)
        zoom_controls.addWidget(self._zoom_fit)
        zoom_controls.addWidget(self._zoom_label)
        zoom_controls.addStretch(1)
        image_column.addLayout(zoom_controls)
        root.addLayout(image_column, stretch=1)

        side = QVBoxLayout()
        if self._editable:
            side.addWidget(self._build_controls())
        self._metrics_label = QLabel()
        self._metrics_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._metrics_label.setStyleSheet('font-family: Consolas, monospace; color:#DDDDDD;')
        self._metrics_label.setWordWrap(True)
        side.addWidget(self._metrics_label)
        side.addStretch(1)
        if self._editable:
            self._reset_cuts_btn = QPushButton('Reset Cuts')
            self._reset_cuts_btn.clicked.connect(self._reset_cuts)
            self._apply_btn = QPushButton('Apply Initialization')
            self._apply_btn.clicked.connect(lambda: self.apply_requested.emit(self.parameters(), list(self._cuts)))
            side.addWidget(self._reset_cuts_btn)
            side.addWidget(self._apply_btn)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(245)
        root.addWidget(side_widget)

        self.set_preview(preview, reset_controls=True)

    def _build_controls(self) -> QGroupBox:
        group = QGroupBox('Initialization')
        form = QFormLayout(group)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.05, 0.99)
        self._threshold.setDecimals(2)
        self._threshold.setSingleStep(0.02)
        self._margin = QDoubleSpinBox()
        self._margin.setRange(0.0, 0.50)
        self._margin.setDecimals(3)
        self._margin.setSingleStep(0.01)
        self._area_ratio = QDoubleSpinBox()
        self._area_ratio.setRange(0.25, 50.0)
        self._area_ratio.setDecimals(2)
        self._area_ratio.setSingleStep(0.25)
        self._growth = QDoubleSpinBox()
        self._growth.setRange(0.0, 500.0)
        self._growth.setDecimals(1)
        self._growth.setSingleStep(2.0)
        self._growth.setSuffix(' px')
        self._close = QSpinBox()
        self._close.setRange(0, 5)
        self._cut_width = QSpinBox()
        self._cut_width.setRange(1, 15)
        self._cut_width.setValue(3)
        self._cut_width.setSuffix(' px')
        for widget in (self._threshold, self._margin, self._area_ratio, self._growth, self._close, self._cut_width):
            widget.valueChanged.connect(self._request_preview)
        form.addRow('Threshold', self._threshold)
        form.addRow('Margin', self._margin)
        form.addRow('Max area', self._area_ratio)
        form.addRow('Grow radius', self._growth)
        form.addRow('Close', self._close)
        form.addRow('Cut width', self._cut_width)
        return group

    def parameters(self) -> dict[str, object]:
        return {
            'normal_init_threshold': float(self._threshold.value()),
            'normal_init_min_probability_margin': float(self._margin.value()),
            'normal_init_max_area_ratio': float(self._area_ratio.value()),
            'normal_init_growth_radius': float(self._growth.value()),
            'normal_init_close_iterations': int(self._close.value()),
            'scissors_thickness': int(self._cut_width.value()),
        }

    def set_view_mode(self, mode: str) -> None:
        if mode not in {'original', 'probability', 'overlay'}:
            return
        self._view_mode = mode
        self._canvas.set_interaction(self._editable and mode == 'overlay')
        self._canvas.set_cuts(self._cuts, self._pending_cut_point)
        self._canvas.set_image(self._images.get(mode))

    def set_preview(self, preview: Optional[InitPreview], *, error: str = '', reset_controls: bool = False) -> None:
        if preview is None:
            if error:
                self._metrics_label.setText(f'Rejected\n{error}')
            return
        self._last_preview = preview
        diag = dict(getattr(preview, 'init_diagnostics', {}) or {})
        if reset_controls and self._editable:
            self._set_controls_from_parameters(dict(diag.get('parameters', {}) or {}))
        self._render_preview(diag, error=error)
        if reset_controls:
            QTimer.singleShot(0, self._fit_to_view)

    def _set_controls_from_parameters(self, parameters: dict) -> None:
        controls = [
            (self._threshold, 'normal_init_threshold'),
            (self._margin, 'normal_init_min_probability_margin'),
            (self._area_ratio, 'normal_init_max_area_ratio'),
            (self._growth, 'normal_init_growth_radius'),
            (self._close, 'normal_init_close_iterations'),
            (self._cut_width, 'scissors_thickness'),
        ]
        for widget, key in controls:
            if key in parameters:
                widget.blockSignals(True)
                widget.setValue(int(round(float(parameters[key]))) if isinstance(widget, QSpinBox) else float(parameters[key]))
                widget.blockSignals(False)

    def _request_preview(self) -> None:
        if not self._editable:
            return
        self.preview_requested.emit(self.parameters(), list(self._cuts))

    def _on_image_clicked(self, x: float, y: float) -> None:
        if not self._editable or self._view_mode != 'overlay':
            return
        if self._pending_cut_point is None:
            self._pending_cut_point = (x, y)
        else:
            x1, y1 = self._pending_cut_point
            self._cuts.append((x1, y1, x, y))
            self._pending_cut_point = None
            self._request_preview()
        self._canvas.set_cuts(self._cuts, self._pending_cut_point)
        self._update_metrics()

    def _reset_cuts(self) -> None:
        if not self._editable:
            return
        self._cuts.clear()
        self._pending_cut_point = None
        self._canvas.set_cuts([], None)
        self._request_preview()

    def _render_preview(self, diag: dict, *, error: str) -> None:
        roi_bgr = diag.get('roi_bgr')
        probability = diag.get('probability_roi')
        mask = diag.get('mask_roi')
        if roi_bgr is None or probability is None or mask is None:
            self._metrics_label.setText('Initialization diagnostics unavailable.')
            return
        roi_bgr = np.asarray(roi_bgr, dtype=np.uint8)
        probability = np.asarray(probability, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8)
        original_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        probability_rgb = self._probability_rgb(probability)
        self._images = {
            'original': original_rgb,
            'probability': probability_rgb,
            'overlay': self._overlay_rgb(original_rgb, probability_rgb, mask, diag),
        }
        self._last_diag = diag
        self._last_error = error
        self.set_view_mode(self._view_mode)
        self._update_metrics()

    @staticmethod
    def _probability_rgb(probability: np.ndarray) -> np.ndarray:
        scaled = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _overlay_rgb(original_rgb: np.ndarray, probability_rgb: np.ndarray, mask: np.ndarray, diag: dict) -> np.ndarray:
        overlay = cv2.addWeighted(original_rgb, 0.58, probability_rgb, 0.42, 0.0)
        hit = mask > 0
        overlay[hit] = (0.55 * overlay[hit] + np.array([70, 255, 120]) * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        ox, oy, *_ = diag.get('roi_xywh', (0, 0, 0, 0))
        center = diag.get('center_rc')
        if center is not None:
            row, col = np.asarray(center, dtype=np.float32).reshape(2)
            cv2.drawMarker(overlay, (int(round(col - ox)), int(round(row - oy))), (255, 40, 40), cv2.MARKER_CROSS, 13, 1)
        click = diag.get('click_local_rc')
        if click is not None:
            row, col = [int(v) for v in click]
            cv2.circle(overlay, (col, row), 4, (255, 230, 0), 1)
        return overlay

    def _update_metrics(self) -> None:
        diag = getattr(self, '_last_diag', {})
        metrics = dict(diag.get('metrics', {}) or {})
        lines = ['Rejected', getattr(self, '_last_error', '')] if getattr(self, '_last_error', '') else ['Accepted']
        for key, label in (
            ('confidence', 'Confidence'), ('probability_margin', 'Prob. margin'),
            ('median_inside_probability', 'Median prob.'), ('area', 'Area'),
            ('area_ratio', 'Area ratio'), ('width', 'Width'), ('height', 'Height'),
            ('circularity', 'Circularity'), ('rectangularity', 'Rectangularity'),
        ):
            if key in metrics:
                lines.append(f'{label}: {float(metrics[key]):.3f}')
        lines.extend(str(message) for message in metrics.get('cut_messages', []) or [])
        if self._pending_cut_point is not None:
            lines.append('Select the cut end point.')
        elif self._cuts:
            lines.append(f'Cuts: {len(self._cuts)}')
        self._metrics_label.setText('\n'.join(line for line in lines if line))

    def _zoom_at(self, factor: float, row: float, col: float) -> None:
        old_zoom = self._canvas.zoom()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        cursor_x = col * old_zoom - hbar.value()
        cursor_y = row * old_zoom - vbar.value()
        self._canvas.set_zoom(old_zoom * factor)
        new_zoom = self._canvas.zoom()
        hbar.setValue(max(0, int(col * new_zoom - cursor_x)))
        vbar.setValue(max(0, int(row * new_zoom - cursor_y)))
        self._update_zoom_label()

    def _set_zoom(self, zoom: float) -> None:
        old_zoom = self._canvas.zoom()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        col = (hbar.value() + self._scroll.viewport().width() / 2) / max(old_zoom, 1e-9)
        row = (vbar.value() + self._scroll.viewport().height() / 2) / max(old_zoom, 1e-9)
        self._canvas.set_zoom(zoom)
        new_zoom = self._canvas.zoom()
        hbar.setValue(max(0, int(col * new_zoom - self._scroll.viewport().width() / 2)))
        vbar.setValue(max(0, int(row * new_zoom - self._scroll.viewport().height() / 2)))
        self._update_zoom_label()

    def _fit_to_view(self) -> None:
        shape = self._canvas.image_shape()
        if shape is None:
            return
        h, w = shape
        viewport = self._scroll.viewport().size()
        self._set_zoom(max(0.1, min(viewport.width() / max(w, 1), viewport.height() / max(h, 1))))

    def _update_zoom_label(self) -> None:
        self._zoom_label.setText(f'{self._canvas.zoom() * 100:.0f}%')
