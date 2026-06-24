"""
ui/color_tolerance_panel.py
---------------------------
Inline panel shown below the tracker panel when a Color Area tracker is
initialised.  Provides three slider pairs (hue, saturation, value) and
updates the canvas preview in real time as sliders move.

Architecture
------------
* The panel is inserted into the TrackerPanel's scroll list beneath the
  relevant TrackerRow.
* On every slider change it calls back to MainWindow via the
  `tolerance_changed(uid, ColorTolerance)` signal, which:
    1. Updates the tracker's internal tolerance
    2. Requests a preview render from color_mask.threshold_preview()
    3. Pushes the resulting RGBA overlay to VideoCanvas.set_init_preview()

* The panel also exposes a `confirm()` slot so the user can lock-in the
  tolerance and proceed to batch.

ColorTolerance fields
---------------------
  center_h  : float  [0, 360)   Hue centre
  delta_h   : float  [0, 180]   Hue half-band
  center_s  : float  [0, 1]     Saturation centre
  delta_s   : float  [0, 1]     Saturation half-band
  center_v  : float  [0, 1]     Value centre
  delta_v   : float  [0, 1]     Value half-band
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from gpu.color_mask import ColorTolerance


# ---------------------------------------------------------------------------
# Helper: a labelled slider row that emits float values
# ---------------------------------------------------------------------------

class _LabelledSlider(QWidget):
    """
    A single slider with a left label and a right value display.
    Internal scale: integer steps mapped to float range.
    """

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label:      str,
        lo:         float,
        hi:         float,
        init:       float,
        steps:      int   = 200,
        color_hex:  str   = '#6495ED',
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lo    = lo
        self._hi    = hi
        self._steps = steps
        self._range = hi - lo

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(28)
        lbl.setStyleSheet('color:#AAAAAA; font-size:10px;')
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(steps)
        self._slider.setValue(self._to_step(init))
        self._slider.setStyleSheet(
            f'QSlider::handle:horizontal {{ background:{color_hex}; '
            f'width:10px; border-radius:5px; }}'
            f'QSlider::sub-page:horizontal {{ background:{color_hex}; }}'
        )
        self._slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self._slider, stretch=1)

        self._val_label = QLabel(f'{init:.3f}')
        self._val_label.setFixedWidth(42)
        self._val_label.setStyleSheet('color:#DDDDDD; font-size:10px;')
        self._val_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._val_label)

    def _to_step(self, val: float) -> int:
        frac = (val - self._lo) / self._range if self._range else 0.0
        return int(frac * self._steps)

    def _to_val(self, step: int) -> float:
        return self._lo + (step / self._steps) * self._range

    def _on_changed(self, step: int) -> None:
        val = self._to_val(step)
        self._val_label.setText(f'{val:.3f}')
        self.value_changed.emit(val)

    def value(self) -> float:
        return self._to_val(self._slider.value())

    def set_value(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_step(val))
        self._slider.blockSignals(False)
        self._val_label.setText(f'{val:.3f}')


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ColorTolerancePanel(QFrame):
    """
    Inline tolerance editor for Color Area trackers.

    Signals
    -------
    tolerance_changed(uid, ColorTolerance)
        Fired (debounced 80 ms) on every slider move.
    confirmed(uid)
        Fired when user presses Confirm — panel should be hidden after this.
    """

    tolerance_changed = pyqtSignal(str, object)   # (uid, ColorTolerance)
    confirmed         = pyqtSignal(str)

    _DEBOUNCE_MS = 80   # ms between slider change and preview update

    def __init__(
        self,
        uid:       str,
        tolerance: ColorTolerance,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.uid        = uid
        self._tolerance = tolerance
        self._debounce  = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._emit_tolerance)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            'ColorTolerancePanel {'
            '  background:#252535;'
            '  border:1px solid #444466;'
            '  border-radius:4px;'
            '  margin:2px;'
            '}'
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)

        # Header
        hdr = QLabel('🎨 Color Tolerance')
        hdr.setStyleSheet('color:#DA70D6; font-size:11px; font-weight:bold;')
        root.addWidget(hdr)

        hint = QLabel(
            'Adjust the hue / saturation / value bands.\n'
            'The magenta overlay shows which pixels are matched.'
        )
        hint.setStyleSheet('color:#888888; font-size:9px;')
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ---- Hue group ----
        hue_grp = QGroupBox('Hue')
        hue_grp.setStyleSheet(self._grp_style('#DA70D6'))
        hg = QVBoxLayout(hue_grp)

        self._h_center = _LabelledSlider(
            'Ctr', 0.0, 360.0, self._tolerance.center_h,
            steps=360, color_hex='#DA70D6'
        )
        self._h_delta = _LabelledSlider(
            '±', 0.0, 180.0, self._tolerance.delta_h,
            steps=180, color_hex='#B070B0'
        )
        hg.addWidget(self._h_center)
        hg.addWidget(self._h_delta)
        root.addWidget(hue_grp)

        # ---- Saturation group ----
        sat_grp = QGroupBox('Saturation')
        sat_grp.setStyleSheet(self._grp_style('#70B0DA'))
        sg = QVBoxLayout(sat_grp)

        self._s_center = _LabelledSlider(
            'Ctr', 0.0, 1.0, self._tolerance.center_s,
            steps=100, color_hex='#70B0DA'
        )
        self._s_delta = _LabelledSlider(
            '±', 0.0, 1.0, self._tolerance.delta_s,
            steps=100, color_hex='#5090B0'
        )
        sg.addWidget(self._s_center)
        sg.addWidget(self._s_delta)
        root.addWidget(sat_grp)

        # ---- Value group ----
        val_grp = QGroupBox('Value (Brightness)')
        val_grp.setStyleSheet(self._grp_style('#DAC870'))
        vg = QVBoxLayout(val_grp)

        self._v_center = _LabelledSlider(
            'Ctr', 0.0, 1.0, self._tolerance.center_v,
            steps=100, color_hex='#DAC870'
        )
        self._v_delta = _LabelledSlider(
            '±', 0.0, 1.0, self._tolerance.delta_v,
            steps=100, color_hex='#B0A050'
        )
        vg.addWidget(self._v_center)
        vg.addWidget(self._v_delta)
        root.addWidget(val_grp)

        # Wire slider changes → debounced emit
        for slider in (
            self._h_center, self._h_delta,
            self._s_center, self._s_delta,
            self._v_center, self._v_delta,
        ):
            slider.value_changed.connect(self._on_any_slider)

        # ---- Confirm button ----
        confirm_btn = QPushButton('✓  Confirm Tolerance')
        confirm_btn.setStyleSheet(
            'QPushButton {'
            '  background:#1a5c1a; color:white; padding:6px;'
            '  border-radius:4px; font-weight:bold;'
            '}'
            'QPushButton:hover { background:#226622; }'
        )
        confirm_btn.clicked.connect(lambda: self.confirmed.emit(self.uid))
        root.addWidget(confirm_btn)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _grp_style(accent: str) -> str:
        return (
            f'QGroupBox {{'
            f'  color:{accent};'
            f'  border:1px solid #3a3a4a;'
            f'  border-radius:4px;'
            f'  margin-top:6px;'
            f'  padding-top:4px;'
            f'  font-size:10px;'
            f'}}'
        )

    def _on_any_slider(self, _: float) -> None:
        """Restart debounce timer on any slider movement."""
        self._debounce.start(self._DEBOUNCE_MS)

    def _emit_tolerance(self) -> None:
        """Called after debounce expires — build tolerance and emit."""
        tol = ColorTolerance(
            center_h=self._h_center.value(),
            delta_h =self._h_delta.value(),
            center_s=self._s_center.value(),
            delta_s =self._s_delta.value(),
            center_v=self._v_center.value(),
            delta_v =self._v_delta.value(),
        )
        self._tolerance = tol
        self.tolerance_changed.emit(self.uid, tol)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def current_tolerance(self) -> ColorTolerance:
        return self._tolerance

    def update_from_tolerance(self, tol: ColorTolerance) -> None:
        """Push new values to sliders without triggering a re-emit loop."""
        for slider, val in [
            (self._h_center, tol.center_h),
            (self._h_delta,  tol.delta_h),
            (self._s_center, tol.center_s),
            (self._s_delta,  tol.delta_s),
            (self._v_center, tol.center_v),
            (self._v_delta,  tol.delta_v),
        ]:
            slider.set_value(val)
        self._tolerance = tol
