"""
ui/video_canvas.py
------------------
QOpenGLWidget subclass that:
  - Renders video frames as a GPU-uploaded OpenGL texture
  - Supports smooth zoom (scroll wheel) centred on cursor
  - Supports pan via WASD keys
  - Q/E keys step one frame back/forward
  - Draws semi-transparent overlays for all active trackers
  - Handles click/drag for seed-point and ROI selection
  - Shows the init-frame preview heatmap + shape overlay

The widget talks to TrackerManager and FrameBuffer only through signals/slots
so it can be used from the UI thread safely.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PyQt6.QtCore import (
    QPointF, Qt, QPoint, QRect, QRectF, QSize, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QImage, QKeyEvent, QMouseEvent, QPainter,
    QPen, QPolygonF, QWheelEvent, QBrush, QFont,
)
from PyQt6.QtOpenGL import (
    QOpenGLFunctions_2_0,
    QOpenGLPixelTransferOptions,
    QOpenGLTexture,
    QOpenGLTextureBlitter,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QApplication
from PyQt6 import sip

from tracking.base_tracker import (
    FrameResult, InitPreview, TrackerStatus, TrackerType,
)

# ---------------------------------------------------------------------------
# Type colours (RGB tuples)
# ---------------------------------------------------------------------------
TYPE_COLORS: Dict[TrackerType, QColor] = {
    TrackerType.POINT_FAST:     QColor(100, 149, 237),   # cornflower blue
}

STATUS_COLORS: Dict[TrackerStatus, QColor] = {
    TrackerStatus.LOCKED:    QColor(  0, 255,   0, 180),
    TrackerStatus.UNCERTAIN: QColor(255, 215,   0, 180),
    TrackerStatus.LOST:      QColor(255,  50,  50, 180),
    TrackerStatus.PENDING:   QColor(160, 160, 160, 180),
}

GL_COLOR_BUFFER_BIT = 0x00004000


class VideoCanvas(QOpenGLWidget):
    """
    Central video display widget.

    Signals
    -------
    frame_step_requested(delta)    Q/E key → step frames
    seed_clicked(row, col)         Single click in seed mode
    roi_selected(x, y, w, h)      Completed drag in ROI mode
    """

    frame_step_requested = pyqtSignal(int)
    seed_clicked         = pyqtSignal(float, float)
    roi_selected         = pyqtSignal(int, int, int, int)
    calibration_points_selected = pyqtSignal(float, float, float, float)
    coordinate_origin_selected = pyqtSignal(float, float)
    quick_point_seed_requested = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(640, 480))

        # Current frame as QImage (updated by set_frame)
        self._frame_image: Optional[QImage] = None
        self._frame_texture: Optional[QOpenGLTexture] = None
        self._frame_texture_blitter: Optional[QOpenGLTextureBlitter] = None
        self._gl_functions: Optional[QOpenGLFunctions_2_0] = None
        self._pending_texture_frame: Optional[np.ndarray] = None
        self._texture_size: Tuple[int, int] = (0, 0)
        self._texture_upload_options = QOpenGLPixelTransferOptions()
        self._texture_upload_options.setAlignment(1)
        self._texture_preview_enabled: bool = True
        self._texture_preview_failed: bool = False
        self._frame_w = 1
        self._frame_h = 1

        # Zoom / pan
        self._zoom:    float  = 1.0
        self._pan_x:   float  = 0.0
        self._pan_y:   float  = 0.0
        self._pan_step: float = 30.0

        # Drag state for ROI selection
        self._dragging:    bool          = False
        self._drag_start:  Optional[QPoint] = None
        self._drag_end:    Optional[QPoint] = None
        self._roi_mode:    bool          = False   # True = drag selects ROI
        self._seed_guide_diameter_px: Optional[float] = None
        self._seed_guide_cursor: Optional[QPointF] = None
        self._quick_seed_diameter_px: Optional[float] = None
        self._calibration_mode: Optional[str] = None
        self._calibration_first_point: Optional[tuple[float, float]] = None

        # Overlays: keyed by tracker uid
        self._results:       Dict[str, FrameResult]  = {}
        self._init_previews: Dict[str, InitPreview]  = {}
        self._init_preview_frames: Dict[str, int]     = {}
        self._tracker_types: Dict[str, TrackerType]  = {}
        self._tracker_names: Dict[str, str]          = {}

        # Heatmap overlay for init frame (uid → RGBA QImage)
        self._heatmap_images: Dict[str, QImage] = {}

        # Which frame is being shown
        self._current_frame_index: int = 0

        # Optional analysis workspace used for prepared frame caching.
        self._analysis_workspace: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Display a new video frame (BGR uint8 numpy array).
        Called from the UI thread whenever the current frame changes.
        """
        h, w, _ = frame_bgr.shape
        self._frame_w, self._frame_h = w, h
        if self._texture_preview_enabled and not self._texture_preview_failed:
            self._pending_texture_frame = np.ascontiguousarray(frame_bgr)
            self._frame_image = None
        else:
            self._set_frame_image(frame_bgr)
        self.update()

    def _set_frame_image(self, frame_bgr: np.ndarray) -> None:
        """Fallback display path using QImage when texture preview is unavailable."""
        h, w, _ = frame_bgr.shape
        rgb = frame_bgr[:, :, ::-1].copy()
        self._frame_image = QImage(
            rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
        ).copy()
        self._pending_texture_frame = None

    def set_frame_index(self, idx: int) -> None:
        self._current_frame_index = idx

    def set_analysis_workspace(self, bbox_xywh: Optional[Tuple[int, int, int, int]]) -> None:
        """Show or clear the workspace that will be decoded/cached for analysis."""
        self._analysis_workspace = bbox_xywh
        self.update()

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable drag-to-select ROI mode (for blob / color-area init)."""
        self._roi_mode = enabled
        if not enabled:
            self._drag_start = None
            self._drag_end   = None
            self._dragging   = False

    def set_seed_guide(self, diameter_px: Optional[float]) -> None:
        """Show a click-centred feature-size circle while waiting for a seed."""
        diameter = 0.0 if diameter_px is None else float(diameter_px)
        self._seed_guide_diameter_px = diameter if diameter > 0.0 else None
        self._seed_guide_cursor = None
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if self._seed_guide_diameter_px is not None else Qt.CursorShape.ArrowCursor
        )
        self.update()

    def set_quick_seed_diameter(self, diameter_px: float) -> None:
        self._quick_seed_diameter_px = max(0.0, float(diameter_px))

    def set_calibration_mode(self, mode: Optional[str]) -> None:
        self._calibration_mode = mode if mode in {'scale', 'origin'} else None
        self._calibration_first_point = None
        self.setCursor(Qt.CursorShape.CrossCursor if self._calibration_mode else Qt.CursorShape.ArrowCursor)
        self.update()

    def update_result(self, uid: str, result: FrameResult,
                      tracker_type: TrackerType, name: str) -> None:
        """Called after each frame during review to update overlay."""
        self._results[uid]       = result
        self._tracker_types[uid] = tracker_type
        self._tracker_names[uid] = name
        self.update()

    def set_init_preview(self, uid: str, preview: InitPreview,
                          tracker_type: TrackerType, name: str,
                          frame_index: Optional[int] = None) -> None:
        """Store an init-frame preview for a tracker.

        Init overlays are only valid on the frame where they were created.
        Without this guard, the normal preview can show an old seed/ROI contour
        on top of a later video frame, which looks like tracking lag even when
        the saved per-frame debug/result is correct.
        """
        self._init_previews[uid]  = preview
        self._init_preview_frames[uid] = self._current_frame_index if frame_index is None else int(frame_index)
        self._tracker_types[uid]  = tracker_type
        self._tracker_names[uid]  = name

        # Build heatmap QImage if available
        if preview.heatmap is not None:
            self._heatmap_images[uid] = self._heatmap_to_qimage(
                preview.heatmap, TYPE_COLORS.get(tracker_type, QColor(255,255,255))
            )
        elif preview.color_preview is not None:
            rgba = preview.color_preview  # already RGBA
            qi   = QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                          rgba.shape[1]*4, QImage.Format.Format_RGBA8888).copy()
            self._heatmap_images[uid] = qi

        self.update()

    def remove_tracker_overlay(self, uid: str) -> None:
        for d in (self._results, self._init_previews, self._init_preview_frames,
                  self._tracker_types, self._tracker_names, self._heatmap_images):
            d.pop(uid, None)
        self.update()

    def reset_view(self) -> None:
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Qt painting
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        self._gl_functions = QOpenGLFunctions_2_0()
        self._gl_functions.initializeOpenGLFunctions()
        self._frame_texture_blitter = QOpenGLTextureBlitter()
        if not self._frame_texture_blitter.create():
            self._texture_preview_failed = True

    def _clear_gl_background(self) -> None:
        funcs = self._gl_functions
        if funcs is None:
            return
        funcs.glClearColor(20 / 255.0, 20 / 255.0, 20 / 255.0, 1.0)
        funcs.glClear(GL_COLOR_BUFFER_BIT)

    def _release_frame_texture(self) -> None:
        if self._frame_texture is not None:
            self._frame_texture.destroy()
            self._frame_texture = None
        self._texture_size = (0, 0)

    def _ensure_frame_texture(self, width: int, height: int) -> Optional[QOpenGLTexture]:
        if self._frame_texture is not None and self._texture_size == (width, height):
            return self._frame_texture
        self._release_frame_texture()
        texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        texture.create()
        texture.setFormat(QOpenGLTexture.TextureFormat.RGB8_UNorm)
        texture.setSize(width, height)
        texture.setMipLevels(1)
        texture.setMinMagFilters(
            QOpenGLTexture.Filter.Linear,
            QOpenGLTexture.Filter.Linear,
        )
        texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        texture.allocateStorage(
            QOpenGLTexture.PixelFormat.BGR,
            QOpenGLTexture.PixelType.UInt8,
        )
        self._frame_texture = texture
        self._texture_size = (width, height)
        return texture

    def _upload_pending_texture_frame(self) -> None:
        if (
            self._pending_texture_frame is None
            or self._texture_preview_failed
            or self._frame_texture_blitter is None
            or not self._frame_texture_blitter.isCreated()
        ):
            return
        frame = self._pending_texture_frame
        try:
            h, w, _ = frame.shape
            texture = self._ensure_frame_texture(w, h)
            if texture is None:
                return
            # Storage was allocated once in _ensure_frame_texture.  Explicit
            # one-byte row alignment handles BGR rows whose byte width is not a
            # multiple of OpenGL's default four-byte alignment.
            texture.setData(
                0, 0, 0, w, h, 1,
                QOpenGLTexture.PixelFormat.BGR,
                QOpenGLTexture.PixelType.UInt8,
                sip.voidptr(frame.ctypes.data),
                self._texture_upload_options,
            )
            self._pending_texture_frame = None
        except Exception:
            self._texture_preview_failed = True
            self._set_frame_image(frame)

    def _draw_frame_texture(self, dest: QRect) -> bool:
        self._upload_pending_texture_frame()
        if (
            self._frame_texture is None
            or self._frame_texture_blitter is None
            or not self._frame_texture_blitter.isCreated()
        ):
            return False
        # QOpenGLTextureBlitter operates in framebuffer pixels while widget
        # geometry is expressed in logical Qt pixels.  Mixing the two corrupts
        # the sampled rows on fractional-DPI displays (for example 125%).
        dpr = self.devicePixelRatioF()
        target = QRectF(
            dest.x() * dpr,
            dest.y() * dpr,
            dest.width() * dpr,
            dest.height() * dpr,
        )
        viewport = QRect(
            0,
            0,
            round(self.width() * dpr),
            round(self.height() * dpr),
        )
        transform = QOpenGLTextureBlitter.targetTransform(target, viewport)
        self._frame_texture_blitter.bind()
        try:
            self._frame_texture_blitter.blit(
                self._frame_texture.textureId(),
                transform,
                QOpenGLTextureBlitter.Origin.OriginTopLeft,
            )
        finally:
            self._frame_texture_blitter.release()
        return True

    def paintGL(self) -> None:
        self._clear_gl_background()
        dest = self._frame_dest_rect()
        texture_drawn = (
            self._texture_preview_enabled
            and not self._texture_preview_failed
            and self._draw_frame_texture(dest)
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        if not texture_drawn:
            painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self._frame_image is None and not texture_drawn:
            self._draw_placeholder(painter)
            painter.end()
            return

        # Draw video frame
        if not texture_drawn and self._frame_image is not None:
            painter.drawImage(dest, self._frame_image)

        # Draw heatmap overlays (init previews)
        self._draw_heatmaps(painter, dest)

        # Draw selected analysis workspace boundary.
        self._draw_analysis_workspace(painter)

        # Draw tracker shapes
        self._draw_overlays(painter, dest)

        # Draw drag ROI rectangle
        if self._dragging and self._drag_start and self._drag_end:
            self._draw_drag_rect(painter)

        self._draw_seed_guide(painter)
        self._draw_calibration_points(painter)

        painter.end()

    def _frame_dest_rect(self) -> QRect:
        """Compute the on-screen pixel rect for the current video frame."""
        cw, ch = self.width(), self.height()
        # Fit-to-view base scale
        base_scale = min(cw / self._frame_w, ch / self._frame_h)
        scale      = base_scale * self._zoom

        dw = int(self._frame_w * scale)
        dh = int(self._frame_h * scale)

        # Centre + pan
        x = int((cw - dw) / 2 + self._pan_x)
        y = int((ch - dh) / 2 + self._pan_y)
        return QRect(x, y, dw, dh)

    def _widget_to_frame(self, wx: float, wy: float) -> Tuple[float, float]:
        """Convert widget pixel coords to frame pixel coords (row, col)."""
        dest = self._frame_dest_rect()
        if dest.width() == 0 or dest.height() == 0:
            return 0.0, 0.0
        col = (wx - dest.x()) / dest.width()  * self._frame_w
        row = (wy - dest.y()) / dest.height() * self._frame_h
        return row, col

    def _frame_to_widget(self, row: float, col: float) -> Tuple[float, float]:
        dest = self._frame_dest_rect()
        wx = dest.x() + col / self._frame_w * dest.width()
        wy = dest.y() + row / self._frame_h * dest.height()
        return wx, wy

    def _scale_factor(self) -> float:
        dest = self._frame_dest_rect()
        if self._frame_w == 0:
            return 1.0
        return dest.width() / self._frame_w

    # ------------------------------------------------------------------
    # Overlay drawing helpers
    # ------------------------------------------------------------------

    def _draw_analysis_workspace(self, painter: QPainter) -> None:
        if self._analysis_workspace is None:
            return
        x, y, width, height = self._analysis_workspace
        wx, wy = self._frame_to_widget(y, x)
        scale = self._scale_factor()
        pen = QPen(QColor(0, 220, 255), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 220, 255, 18)))
        painter.drawRect(int(wx), int(wy), int(width * scale), int(height * scale))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_heatmaps(self, painter: QPainter, dest: QRect) -> None:
        painter.setOpacity(0.35)
        for uid, qi in self._heatmap_images.items():
            if self._init_preview_frames.get(uid) != self._current_frame_index:
                continue
            painter.drawImage(dest, qi)
        painter.setOpacity(1.0)

    def _draw_overlays(self, painter: QPainter, dest: QRect) -> None:
        scale = self._scale_factor()

        # Draw init previews only on the exact frame where they were generated.
        # Otherwise a seed/ROI contour from frame 0 can be drawn over frame 200+
        # and falsely look like tracker lag.
        for uid, preview in self._init_previews.items():
            if self._init_preview_frames.get(uid) != self._current_frame_index:
                continue
            color = TYPE_COLORS.get(self._tracker_types.get(uid), QColor(255,255,255))
            self._draw_preview(painter, preview, uid, color, dest, scale)

        # Draw only per-frame results that belong to the currently displayed
        # video frame.  This prevents stale progress/result overlays from being
        # shown after scrubbing or after a batch run.
        for uid, result in self._results.items():
            if int(result.frame_index) != int(self._current_frame_index):
                continue
            color = TYPE_COLORS.get(self._tracker_types.get(uid), QColor(255,255,255))
            status_color = STATUS_COLORS.get(result.status, QColor(200,200,200,180))
            self._draw_result(painter, result, uid, color, status_color, dest, scale)

    def _draw_preview(self, painter, preview: InitPreview, uid, color, dest, scale):
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        if preview.hog_bbox:
            x, y, w, h = preview.hog_bbox
            wx, wy = self._frame_to_widget(y, x)
            painter.drawRect(int(wx), int(wy), int(w*scale), int(h*scale))

        if preview.center is not None:
            self._draw_crosshair(painter, preview.center, color, scale)

        if preview.polygon is not None:
            self._draw_polygon(painter, preview.polygon, color, scale, fill=True, alpha=60)

        if preview.mask is not None:
            self._draw_mask(painter, preview.mask, color, dest, alpha=80)

        if preview.spline_points is not None:
            self._draw_spline(painter, preview.spline_points, color, scale)

    def _draw_result(self, painter, result: FrameResult, uid, color, status_color, dest, scale):
        name  = self._tracker_names.get(uid, uid)
        label_pos = None

        if result.center is not None:
            wx, wy = self._frame_to_widget(result.center[0], result.center[1])
            self._draw_crosshair(painter, result.center, status_color, scale)
            label_pos = (wx, wy)

        if result.polygon is not None:
            self._draw_polygon(painter, result.polygon, status_color, scale)
            if result.polygon.shape[0] > 0:
                cx, cy = self._frame_to_widget(*result.polygon.mean(axis=0))
                label_pos = (cx, cy)

        if result.mask is not None:
            self._draw_mask(painter, result.mask, status_color, dest, alpha=60)

        if result.spline_points is not None:
            self._draw_spline(painter, result.spline_points, status_color, scale)
            if result.spline_points.shape[0] > 0:
                mid = result.spline_points[len(result.spline_points)//2]
                lx, ly = self._frame_to_widget(*mid)
                label_pos = (lx, ly)

        if label_pos:
            self._draw_label(painter, name, label_pos[0], label_pos[1], color)

    def _draw_crosshair(self, painter, center, color, scale):
        wx, wy = self._frame_to_widget(center[0], center[1])
        r = max(6, int(8 * scale))
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.drawLine(int(wx-r), int(wy), int(wx+r), int(wy))
        painter.drawLine(int(wx), int(wy-r), int(wx), int(wy+r))
        painter.drawEllipse(int(wx-4), int(wy-4), 8, 8)

    def _draw_polygon(self, painter, polygon, color, scale, fill=False, alpha=40):
        if polygon.shape[0] < 3:
            return
        pts = [QPoint(*map(int, self._frame_to_widget(r, c)))
               for r, c in polygon]
        pen = QPen(color, 2)
        painter.setPen(pen)
        if fill:
            fc = QColor(color)
            fc.setAlpha(alpha)
            painter.setBrush(QBrush(fc))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF([QPointF(p) for p in pts])
        painter.drawPolygon(poly)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_mask(self, painter, mask: np.ndarray, color, dest, alpha=60):
        """Draw a binary mask as a coloured semi-transparent overlay."""
        H, W = mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        hit  = mask > 0
        rgba[hit, 0] = color.red()
        rgba[hit, 1] = color.green()
        rgba[hit, 2] = color.blue()
        rgba[hit, 3] = alpha
        qi = QImage(rgba.data, W, H, W*4, QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(dest, qi)

    def _draw_spline(self, painter, spline, color, scale):
        if spline.shape[0] < 2:
            return
        pen = QPen(color, max(2, int(3*scale)))
        painter.setPen(pen)
        pts = [self._frame_to_widget(r, c) for r, c in spline]
        for i in range(len(pts)-1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
        # Draw control point dots
        for x, y in pts:
            painter.drawEllipse(int(x-3), int(y-3), 6, 6)

    def _draw_label(self, painter, text, wx, wy, color):
        painter.setPen(QPen(QColor(0,0,0,180)))
        font = QFont('Monospace', 9)
        painter.setFont(font)
        painter.drawText(int(wx+10)+1, int(wy-5)+1, text)
        painter.setPen(QPen(color))
        painter.drawText(int(wx+10), int(wy-5), text)

    def _draw_drag_rect(self, painter):
        if not self._drag_start or not self._drag_end:
            return
        r = QRect(self._drag_start, self._drag_end).normalized()
        pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 255, 0, 30)))
        painter.drawRect(r)

    def _draw_seed_guide(self, painter: QPainter) -> None:
        diameter = self._seed_guide_diameter_px
        if diameter is None and QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
            diameter = self._quick_seed_diameter_px
        if diameter is None or self._seed_guide_cursor is None:
            return
        dest = self._frame_dest_rect()
        if not dest.contains(self._seed_guide_cursor.toPoint()):
            return
        radius = max(1.0, diameter * self._scale_factor() / 2.0)
        painter.setPen(QPen(QColor(100, 149, 237, 230), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(self._seed_guide_cursor, radius, radius)

    def _draw_calibration_points(self, painter: QPainter) -> None:
        if self._calibration_first_point is None:
            return
        row, col = self._calibration_first_point
        wx, wy = self._frame_to_widget(row, col)
        painter.setPen(QPen(QColor(255, 210, 80), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(wx, wy), 5, 5)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_placeholder(self, painter):
        painter.setPen(QPen(QColor(80, 80, 80)))
        painter.setFont(QFont('Monospace', 14))
        painter.drawText(
            self.rect(), Qt.AlignmentFlag.AlignCenter,
            'Open a video file to begin\n(File → Open Video)'
        )

    # ------------------------------------------------------------------
    # Heatmap colourisation
    # ------------------------------------------------------------------

    @staticmethod
    def _heatmap_to_qimage(heatmap: np.ndarray, color: QColor) -> QImage:
        """
        Convert a [0,1] float32 heatmap to a coloured RGBA QImage
        using the tracker type colour at varying alpha.
        """
        h, w = heatmap.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        intensity = (heatmap * 255).clip(0, 255).astype(np.uint8)
        rgba[:, :, 0] = color.red()
        rgba[:, :, 1] = color.green()
        rgba[:, :, 2] = color.blue()
        rgba[:, :, 3] = (intensity * 0.7).astype(np.uint8)  # semi-transparent
        return QImage(rgba.data, w, h, w*4, QImage.Format.Format_RGBA8888).copy()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.makeCurrent()
        try:
            self._release_frame_texture()
            if self._frame_texture_blitter is not None:
                self._frame_texture_blitter.destroy()
                self._frame_texture_blitter = None
        finally:
            self.doneCurrent()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            row, col = self._widget_to_frame(event.pos().x(), event.pos().y())
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.quick_point_seed_requested.emit(row, col)
                return
            if self._calibration_mode == 'origin':
                self.coordinate_origin_selected.emit(row, col)
                self.set_calibration_mode(None)
                return
            if self._calibration_mode == 'scale':
                if self._calibration_first_point is None:
                    self._calibration_first_point = (row, col)
                    self.update()
                else:
                    first_row, first_col = self._calibration_first_point
                    self.calibration_points_selected.emit(first_row, first_col, row, col)
                    self.set_calibration_mode(None)
                return
            if self._roi_mode:
                self._dragging   = True
                self._drag_start = event.pos()
                self._drag_end   = event.pos()
            else:
                self.seed_clicked.emit(row, col)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._drag_end = event.pos()
            self.update()
        elif self._seed_guide_diameter_px is not None or self._quick_seed_diameter_px is not None:
            self._seed_guide_cursor = event.position()
            self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self._seed_guide_cursor is not None:
            self._seed_guide_cursor = None
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._drag_start and self._drag_end:
            self._dragging = False
            r = QRect(self._drag_start, self._drag_end).normalized()
            if r.width() > 5 and r.height() > 5:
                # Convert widget rect → frame rect
                r0, c0 = self._widget_to_frame(r.left(),  r.top())
                r1, c1 = self._widget_to_frame(r.right(), r.bottom())
                x  = int(min(c0, c1));  y  = int(min(r0, r1))
                bw = int(abs(c1 - c0)); bh = int(abs(r1 - r0))
                self.roi_selected.emit(x, y, bw, bh)
            self._drag_start = None
            self._drag_end   = None
            self.update()

    # ------------------------------------------------------------------
    # Wheel zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = max(0.2, min(20.0, self._zoom * factor))

        # Zoom centred on mouse cursor
        mx, my = event.position().x(), event.position().y()
        self._pan_x = mx - (mx - self._pan_x) * (new_zoom / self._zoom)
        self._pan_y = my - (my - self._pan_y) * (new_zoom / self._zoom)
        self._zoom  = new_zoom
        self.update()

    # ------------------------------------------------------------------
    # Keyboard events
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        k   = Qt.Key

        if   key == k.Key_E:                self.frame_step_requested.emit(+1)
        elif key == k.Key_Q:                self.frame_step_requested.emit(-1)
        elif key == k.Key_W:                self._pan_y += self._pan_step; self.update()
        elif key == k.Key_S:                self._pan_y -= self._pan_step; self.update()
        elif key == k.Key_A:                self._pan_x += self._pan_step; self.update()
        elif key == k.Key_D:                self._pan_x -= self._pan_step; self.update()
        elif key == k.Key_R:                self.reset_view()
        else:                               super().keyPressEvent(event)
