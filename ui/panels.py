"""
ui/tracker_panel.py   — Left panel: tracker list, mode selector, params
ui/timeline_widget.py — Bottom: scrubber, start/end trim, status strips
ui/settings_dialog.py — Global channel + feature size settings
ui/recovery_dialog.py — Lock-loss notification widget

All four in one file; each class stands alone and can be imported separately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    Qt, QRectF, QSize, pyqtSignal, QTimer,
)
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QIcon,
)
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from tracking.base_tracker import TrackerConfig, TrackerStatus, TrackerType
from utils.point_presets import (
    POINT_PRESET_FIELDS, default_point_preset, load_point_presets,
    restore_default_point_preset, save_point_preset,
    load_learned_presets,
)

# ---------------------------------------------------------------------------
# Colour maps (must match video_canvas.py)
# ---------------------------------------------------------------------------
TYPE_COLORS_HEX = {
    TrackerType.POINT_FAST:     '#6495ED',
}

STATUS_STYLE = {
    TrackerStatus.LOCKED:    'color: #00FF00; font-weight: bold;',
    TrackerStatus.UNCERTAIN: 'color: #FFD700; font-weight: bold;',
    TrackerStatus.LOST:      'color: #FF3232; font-weight: bold;',
    TrackerStatus.PENDING:   'color: #A0A0A0;',
}

# ============================================================
#  TrackerRow — one entry in the panel list
# ============================================================

class TrackerRow(QFrame):
    """
    Displays one tracker: colour swatch, type, name, status, progress bar,
    and batch/run controls.

    Signals
    -------
    delete_requested(uid)
    select_requested(uid)
    """

    delete_requested  = pyqtSignal(str)
    select_requested  = pyqtSignal(str)
    activity_requested = pyqtSignal(str, bool)    # uid, active
    end_frame_changed = pyqtSignal(str, object)   # uid, inclusive frame or None
    recovery_requested = pyqtSignal(str)

    def __init__(self, uid: str, config: TrackerConfig, total_frames: int, parent=None) -> None:
        super().__init__(parent)
        self.uid    = uid
        self.config = config
        self._total_frames = max(1, int(total_frames))
        self._is_active = True
        self._build_ui()

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            'TrackerRow { background: #2a2a2a; border: 1px solid #3a3a3a; '
            'border-radius: 4px; margin: 2px; }'
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)

        # ---- Top row: swatch + name + status + delete ----
        top = QHBoxLayout()

        # Colour swatch
        swatch = QLabel('  ')
        hex_   = TYPE_COLORS_HEX.get(self.config.tracker_type, '#FFFFFF')
        swatch.setStyleSheet(f'background:{hex_}; border-radius:3px;')
        swatch.setFixedSize(16, 16)
        top.addWidget(swatch)

        # Name label
        self._name_label = QLabel(self.config.name or self.config.tracker_type.value)
        self._name_label.setStyleSheet('color:#DDDDDD; font-size:11px;')
        top.addWidget(self._name_label, stretch=1)

        # Status badge
        self._status_label = QLabel('PENDING')
        self._status_label.setStyleSheet(STATUS_STYLE[TrackerStatus.PENDING])
        self._status_label.setFont(QFont('Monospace', 9))
        top.addWidget(self._status_label)

        self._move_btn = QToolButton()
        self._move_btn.setFixedSize(30, 30)
        self._move_btn.setStyleSheet(
            'QToolButton { color:#D9E8FF; background:#284969; border:1px solid #4F7DA8; '
            'border-radius:3px; font-size:18px; font-weight:bold; } '
            'QToolButton:hover { background:#3B638B; }'
        )
        self._move_btn.setToolTip('Move tracker to queue')
        self._move_btn.setText('↓')
        self._move_btn.clicked.connect(self._request_activity_toggle)
        top.addWidget(self._move_btn)

        self._recover_btn = QPushButton('Reinitialize')
        self._recover_btn.setStyleSheet(
            'QPushButton { color:white; background:#7a3d13; border:1px solid #b06c2c; '
            'border-radius:3px; padding:3px 6px; font-size:10px; } '
            'QPushButton:hover { background:#99511d; }'
        )
        self._recover_btn.setToolTip('Open the loss frame and reinitialize this tracker.')
        self._recover_btn.clicked.connect(lambda: self.recovery_requested.emit(self.uid))
        self._recover_btn.hide()
        top.addWidget(self._recover_btn)

        # Delete button
        del_btn = QPushButton('✕')
        del_btn.setFixedSize(20, 20)
        del_btn.setStyleSheet(
            'QPushButton { background:#551111; color:white; border-radius:3px; }'
            'QPushButton:hover { background:#882222; }'
        )
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self.uid))
        top.addWidget(del_btn)

        root.addLayout(top)

        # ---- Progress bar ----
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            f'QProgressBar::chunk {{ background:{TYPE_COLORS_HEX.get(self.config.tracker_type,"#6495ED")}; }}'
        )
        root.addWidget(self._progress)

        self._uncertain_label = QLabel('Uncertain: 0')
        self._uncertain_label.setStyleSheet('color:#AAAAAA; font-size:9px;')
        root.addWidget(self._uncertain_label)

        # ---- Optional per-tracker end frame ----
        end_row = QHBoxLayout()
        self._end_check = QCheckBox('End')
        self._end_check.setToolTip('Stop this tracker after the selected frame.')
        self._end_check.setChecked(self.config.end_frame is not None)
        self._end_check.toggled.connect(self._on_end_enabled_changed)
        end_row.addWidget(self._end_check)
        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, self._total_frames - 1)
        self._end_spin.setValue(
            min(
                self._total_frames - 1,
                max(0, int(self.config.end_frame if self.config.end_frame is not None else self._total_frames - 1)),
            )
        )
        self._end_spin.setEnabled(self._end_check.isChecked())
        self._end_spin.setToolTip('Inclusive final frame for this tracker.')
        self._end_spin.valueChanged.connect(self._on_end_frame_changed)
        end_row.addWidget(self._end_spin, stretch=1)
        root.addLayout(end_row)

        self._pause_on_loss_check = QCheckBox('Pause on loss')
        self._pause_on_loss_check.setChecked(self.config.pause_on_loss)
        self._pause_on_loss_check.setToolTip('Stop processing and move this tracker to Lost when it loses lock.')
        self._pause_on_loss_check.toggled.connect(
            lambda enabled: setattr(self.config, 'pause_on_loss', bool(enabled))
        )
        root.addWidget(self._pause_on_loss_check)

        # Click on row body → select
        self.mousePressEvent = lambda e: self.select_requested.emit(self.uid)

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def set_status(self, status: TrackerStatus) -> None:
        self._status_label.setText(status.value.upper())
        self._status_label.setStyleSheet(STATUS_STYLE.get(status, ''))

    def set_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(current)

    def set_uncertain_count(self, count: int) -> None:
        self._uncertain_label.setText(f'Uncertain: {int(count)}')
        self._uncertain_label.setStyleSheet(
            'color:#FFD700; font-size:9px;' if int(count) > 0 else 'color:#AAAAAA; font-size:9px;'
        )

    def set_active(self, active: bool) -> None:
        self._is_active = bool(active)
        self._move_btn.show()
        self._recover_btn.hide()
        self._move_btn.setText('↓' if self._is_active else '↑')
        self._move_btn.setToolTip(
            'Move tracker to queue' if self._is_active else 'Move tracker to active'
        )

    def set_batch_running(self, running: bool) -> None:
        self._move_btn.setEnabled(not running)
        self._end_check.setEnabled(not running)
        self._end_spin.setEnabled(not running and self._end_check.isChecked())
        self._pause_on_loss_check.setEnabled(not running)

    def set_lost(self, lost_frame: int, first_uncertain_frame: int) -> None:
        self._move_btn.hide()
        self._recover_btn.show()
        self._recover_btn.setToolTip(
            f'Open loss frame {lost_frame}; first uncertain frame was {first_uncertain_frame}.'
        )

    def _request_activity_toggle(self) -> None:
        self.activity_requested.emit(self.uid, not self._is_active)

    def _on_end_enabled_changed(self, enabled: bool) -> None:
        self._end_spin.setEnabled(enabled)
        end_frame = int(self._end_spin.value()) if enabled else None
        self.config.end_frame = end_frame
        self.end_frame_changed.emit(self.uid, end_frame)

    def _on_end_frame_changed(self, value: int) -> None:
        if self._end_check.isChecked():
            self.config.end_frame = int(value)
            self.end_frame_changed.emit(self.uid, int(value))

# ============================================================
#  TrackerPanel — scrollable left panel
# ============================================================

class TrackerPanel(QWidget):
    """
    Left panel with:
    - Mode selector (tracker type radio group)
    - Sigma (feature size) spin box
    - Search radius spin box
    - Scrollable list of TrackerRow widgets
    - Run Batch / Cancel buttons
    - Export .npz button

    Signals
    -------
    add_tracker_requested(TrackerConfig)
    remove_tracker_requested(uid)
    run_batch_requested()
    cancel_batch_requested()
    export_requested()
    export_excel_requested()
    tracker_selected(uid)
    """

    add_tracker_requested     = pyqtSignal(object)   # TrackerConfig
    remove_tracker_requested  = pyqtSignal(str)
    run_batch_requested       = pyqtSignal()
    cancel_batch_requested    = pyqtSignal()
    export_requested          = pyqtSignal()
    export_excel_requested    = pyqtSignal()
    export_csv_requested      = pyqtSignal()
    review_uncertain_requested = pyqtSignal()
    learned_kalman_requested = pyqtSignal()
    calibration_measure_requested = pyqtSignal()
    coordinate_origin_requested = pyqtSignal()
    point_feature_size_changed = pyqtSignal(float)
    calibration_changed = pyqtSignal()
    named_settings_preset_save_requested = pyqtSignal(str)
    named_settings_preset_load_requested = pyqtSignal(str)
    tracker_selected          = pyqtSignal(str)
    tracker_activity_requested = pyqtSignal(str, bool)
    tracker_end_frame_changed = pyqtSignal(str, object)
    tracker_recovery_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setStyleSheet('background:#1e1e1e;')
        self._rows: Dict[str, TrackerRow] = {}
        self._frame_counts: Dict[str, int] = {}
        self._row_location: Dict[str, str] = {}
        self._auxiliary_rows: Dict[str, QWidget] = {}
        self._point_preset_values = load_point_presets()
        self._loading_point_preset = False
        self._current_point_mode = 'normal'
        self._selected_learned_validation_samples = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 8)
        root.setSpacing(6)

        # Title
        title = QLabel('Color Track')
        title.setStyleSheet('color:#FFFFFF; font-size:14px; font-weight:bold;')
        root.addWidget(title)

        # ---- Mode selector ----
        mode_group = QGroupBox('Tracker Mode')
        mode_group.setStyleSheet(
            'QGroupBox { color:#AAAAAA; border:1px solid #3a3a3a; '
            'border-radius:4px; margin-top:6px; padding-top:6px; }'
        )
        mg_layout = QVBoxLayout(mode_group)

        self._mode_combo = QComboBox()
        self._mode_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._mode_combo.addItem('Point Fast', TrackerType.POINT_FAST)
        self._mode_combo.addItem('Other trackers (coming soon)', None)
        item = self._mode_combo.model().item(self._mode_combo.count() - 1)
        if item is not None:
            item.setEnabled(False)
        mg_layout.addWidget(self._mode_combo)
        root.addWidget(mode_group)

        # ---- Tracker naming ----
        # The old visible Feature Size / Search Radius controls belonged to
        # the Hessian/CSRT prototype and were confusing for the current colour
        # point tracker.  Keep only the name here; active point-tracker
        # parameters live under Advanced Point Settings.  Legacy sigma/search
        # values are still supplied as defaults in TrackerConfig for older
        # tracker classes and project serialization.
        param_group = QGroupBox('Tracker')
        param_group.setStyleSheet(mode_group.styleSheet())
        pg = QVBoxLayout(param_group)

        def spin_row(label, widget):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet('color:#AAAAAA; font-size:10px;')
            row.addWidget(lbl)
            row.addWidget(widget)
            return row

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText('Tracker name (optional)')
        self._name_edit.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        pg.addLayout(spin_row('Name:', self._name_edit))

        # ---- Advanced point tracker settings ----
        self._advanced_point_group = QGroupBox('Advanced Point Settings')
        self._advanced_point_group.setCheckable(True)
        self._advanced_point_group.setChecked(False)
        self._advanced_point_group.setStyleSheet(mode_group.styleSheet())
        adv = QVBoxLayout(self._advanced_point_group)

        self._point_mode_combo = QComboBox()
        self._point_mode_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._point_mode_combo.addItem('Normal marker', 'normal')
        self._point_mode_combo.addItem('Tiny feature', 'tiny')
        self._point_mode_combo.currentIndexChanged.connect(self._on_point_mode_changed)
        adv.addLayout(spin_row('Point mode:', self._point_mode_combo))

        self._sample_size_spin = QSpinBox()
        self._sample_size_spin.setRange(1, 31)
        self._sample_size_spin.setSingleStep(2)
        self._sample_size_spin.setValue(5)
        self._sample_size_spin.setSuffix(' px')
        self._sample_size_spin.setToolTip('Odd-sized click sample used to learn foreground colour.')
        adv.addLayout(spin_row('Sample size:', self._sample_size_spin))

        self._expected_diameter_spin = QDoubleSpinBox()
        self._expected_diameter_spin.setRange(1.0, 200.0)
        self._expected_diameter_spin.setValue(20.0)
        self._expected_diameter_spin.valueChanged.connect(self.point_feature_size_changed)
        self._expected_diameter_spin.setSuffix(' px')
        self._expected_diameter_spin.setToolTip('Expected visible diameter of the tracked point/feature.')
        adv.addLayout(spin_row('Feature size:', self._expected_diameter_spin))

        self._normal_init_threshold_spin = QDoubleSpinBox()
        self._normal_init_threshold_spin.setRange(0.05, 0.99)
        self._normal_init_threshold_spin.setDecimals(2)
        self._normal_init_threshold_spin.setSingleStep(0.05)
        self._normal_init_threshold_spin.setValue(0.72)
        self._normal_init_threshold_spin.setToolTip(
            'Normal-marker initialization: probability threshold used to extract the clicked component. '
            'Raise this if a marker merges with a nearby object such as a wooden rod.'
        )
        adv.addLayout(spin_row('Normal init thr.:', self._normal_init_threshold_spin))

        self._normal_init_margin_spin = QDoubleSpinBox()
        self._normal_init_margin_spin.setRange(0.0, 0.50)
        self._normal_init_margin_spin.setDecimals(3)
        self._normal_init_margin_spin.setSingleStep(0.01)
        self._normal_init_margin_spin.setValue(0.08)
        self._normal_init_margin_spin.setToolTip(
            'Normal-marker initialization: minimum foreground/background probability margin. '
            'Raise this to reject weakly separated marker/background selections.'
        )
        adv.addLayout(spin_row('Normal init margin:', self._normal_init_margin_spin))

        self._normal_init_max_area_ratio_spin = QDoubleSpinBox()
        self._normal_init_max_area_ratio_spin.setRange(0.25, 50.0)
        self._normal_init_max_area_ratio_spin.setDecimals(2)
        self._normal_init_max_area_ratio_spin.setSingleStep(0.25)
        self._normal_init_max_area_ratio_spin.setValue(6.0)
        self._normal_init_max_area_ratio_spin.setToolTip(
            'Normal-marker initialization: maximum detected component area divided by expected marker area. '
            'Lower this if initialization includes neighbouring structures.'
        )
        adv.addLayout(spin_row('Normal init max area:', self._normal_init_max_area_ratio_spin))

        self._normal_init_growth_radius_spin = QDoubleSpinBox()
        self._normal_init_growth_radius_spin.setRange(0.0, 500.0)
        self._normal_init_growth_radius_spin.setDecimals(1)
        self._normal_init_growth_radius_spin.setSingleStep(2.0)
        self._normal_init_growth_radius_spin.setValue(0.0)
        self._normal_init_growth_radius_spin.setSuffix(' px')
        self._normal_init_growth_radius_spin.setToolTip(
            'Normal-marker initialization: maximum distance from the clicked point that seeded colour growth may include. '
            '0 = automatic from expected diameter. Lower this if the marker merges into a rod or nearby dark edge.'
        )
        adv.addLayout(spin_row('Normal init grow:', self._normal_init_growth_radius_spin))

        self._normal_init_close_spin = QSpinBox()
        self._normal_init_close_spin.setRange(0, 5)
        self._normal_init_close_spin.setValue(1)
        self._normal_init_close_spin.setToolTip(
            'Normal-marker initialization: morphology closing iterations before connected components. '
            'Set to 0 if narrow bridges merge the marker with nearby material.'
        )
        adv.addLayout(spin_row('Normal init close:', self._normal_init_close_spin))

        self._tiny_init_peak_threshold_spin = QDoubleSpinBox()
        self._tiny_init_peak_threshold_spin.setRange(0.0, 0.99)
        self._tiny_init_peak_threshold_spin.setDecimals(2)
        self._tiny_init_peak_threshold_spin.setSingleStep(0.05)
        self._tiny_init_peak_threshold_spin.setValue(0.40)
        self._tiny_init_peak_threshold_spin.setToolTip(
            'Tiny-feature initialization: minimum colour-likelihood peak at the clicked sample.'
        )
        adv.addLayout(spin_row('Tiny init peak:', self._tiny_init_peak_threshold_spin))

        self._kalman_enabled_check = QCheckBox('Enable Kalman prediction')
        self._kalman_enabled_check.setChecked(False)
        self._kalman_enabled_check.setStyleSheet('color:#CCCCCC; font-size:10px;')
        adv.addWidget(self._kalman_enabled_check)

        self._kalman_model_combo = QComboBox()
        self._kalman_model_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._kalman_model_combo.addItem('Constant velocity', 'constant_velocity')
        self._kalman_model_combo.addItem('Constant acceleration', 'constant_acceleration')
        adv.addLayout(spin_row('Kalman model:', self._kalman_model_combo))

        self._measurement_noise_spin = QDoubleSpinBox()
        self._measurement_noise_spin.setRange(0.05, 20.0)
        self._measurement_noise_spin.setDecimals(2)
        self._measurement_noise_spin.setSingleStep(0.1)
        self._measurement_noise_spin.setValue(0.7)
        self._measurement_noise_spin.setSuffix(' px')
        adv.addLayout(spin_row('Measurement σ:', self._measurement_noise_spin))

        self._process_noise_spin = QDoubleSpinBox()
        self._process_noise_spin.setRange(0.001, 50.0)
        self._process_noise_spin.setDecimals(3)
        self._process_noise_spin.setSingleStep(0.1)
        self._process_noise_spin.setValue(0.5)
        self._process_noise_spin.setSuffix(' px/fr²')
        adv.addLayout(spin_row('Process σ:', self._process_noise_spin))

        self._gate_sigma_spin = QDoubleSpinBox()
        self._gate_sigma_spin.setRange(1.0, 10.0)
        self._gate_sigma_spin.setDecimals(1)
        self._gate_sigma_spin.setSingleStep(0.5)
        self._gate_sigma_spin.setValue(4.0)
        self._gate_sigma_spin.setSuffix(' σ')
        adv.addLayout(spin_row('Gate:', self._gate_sigma_spin))

        self._min_search_spin = QSpinBox()
        self._min_search_spin.setRange(1, 500)
        self._min_search_spin.setValue(12)
        self._min_search_spin.setSuffix(' px')
        adv.addLayout(spin_row('Min search:', self._min_search_spin))

        self._max_search_spin = QSpinBox()
        self._max_search_spin.setRange(4, 1000)
        self._max_search_spin.setValue(80)
        self._max_search_spin.setSuffix(' px')
        self._max_search_spin.setToolTip('Kalman/tiny-feature dynamic search cap. Normal markers with Kalman OFF use the old-style ROI controls below.')
        adv.addLayout(spin_row('Max search:', self._max_search_spin))

        self._normal_search_margin_spin = QSpinBox()
        self._normal_search_margin_spin.setRange(1, 500)
        self._normal_search_margin_spin.setValue(15)
        self._normal_search_margin_spin.setSuffix(' px')
        self._normal_search_margin_spin.setToolTip(
            'Old-style normal-marker search margin used when Kalman is OFF. '
            'Locked ROI half-size = marker half-size + this margin.'
        )
        adv.addLayout(spin_row('Normal margin:', self._normal_search_margin_spin))

        self._miss_expansion_spin = QSpinBox()
        self._miss_expansion_spin.setRange(0, 500)
        self._miss_expansion_spin.setValue(16)
        self._miss_expansion_spin.setSuffix(' px/miss')
        self._miss_expansion_spin.setToolTip(
            'Old-style normal-marker expansion used when Kalman is OFF. '
            'Each consecutive miss increases the search margin by this amount.'
        )
        adv.addLayout(spin_row('Miss expansion:', self._miss_expansion_spin))

        self._max_search_margin_spin = QSpinBox()
        self._max_search_margin_spin.setRange(1, 2000)
        self._max_search_margin_spin.setValue(150)
        self._max_search_margin_spin.setSuffix(' px')
        self._max_search_margin_spin.setToolTip(
            'Old-style normal-marker maximum search margin used when Kalman is OFF. '
            'The margin is min(max margin, normal margin + misses × expansion).'
        )
        adv.addLayout(spin_row('Max old margin:', self._max_search_margin_spin))

        self._max_prediction_error_spin = QDoubleSpinBox()
        self._max_prediction_error_spin.setRange(1.0, 2000.0)
        self._max_prediction_error_spin.setDecimals(1)
        self._max_prediction_error_spin.setSingleStep(1.0)
        self._max_prediction_error_spin.setValue(22.0)
        self._max_prediction_error_spin.setSuffix(' px')
        self._max_prediction_error_spin.setToolTip(
            'Old-style normal-marker motion limit used when Kalman is OFF. '
            'A colour candidate farther from the simple velocity prediction than this limit is rejected.'
        )
        adv.addLayout(spin_row('Motion limit:', self._max_prediction_error_spin))

        self._miss_motion_allowance_spin = QDoubleSpinBox()
        self._miss_motion_allowance_spin.setRange(0.0, 2000.0)
        self._miss_motion_allowance_spin.setDecimals(1)
        self._miss_motion_allowance_spin.setSingleStep(1.0)
        self._miss_motion_allowance_spin.setValue(14.0)
        self._miss_motion_allowance_spin.setSuffix(' px/miss')
        self._miss_motion_allowance_spin.setToolTip(
            'Old-style normal-marker motion-limit expansion used when Kalman is OFF.'
        )
        adv.addLayout(spin_row('Miss motion:', self._miss_motion_allowance_spin))

        self._accept_within_search_region_check = QCheckBox('Accept candidates anywhere in search area')
        self._accept_within_search_region_check.setChecked(False)
        self._accept_within_search_region_check.setToolTip(
            'Do not reject a candidate solely for being far from the motion prediction. '
            'The current bounded search area remains the only motion boundary.'
        )
        adv.addWidget(self._accept_within_search_region_check)

        self._normal_recenter_check = QCheckBox('Full-blob recenter near ROI edge')
        self._normal_recenter_check.setChecked(True)
        self._normal_recenter_check.setStyleSheet('color:#CCCCCC; font-size:10px;')
        self._normal_recenter_check.setToolTip(
            'Normal marker mode only: if the accepted colour blob touches or nearly touches '
            'the search ROI edge, run one larger recentered pass to estimate the full blob centre.'
        )
        adv.addWidget(self._normal_recenter_check)

        self._normal_recenter_edge_spin = QSpinBox()
        self._normal_recenter_edge_spin.setRange(0, 50)
        self._normal_recenter_edge_spin.setValue(3)
        self._normal_recenter_edge_spin.setSuffix(' px')
        self._normal_recenter_edge_spin.setToolTip(
            'Trigger full-blob recenter when the detected marker blob is this close to the current ROI edge.'
        )
        adv.addLayout(spin_row('Recenter edge:', self._normal_recenter_edge_spin))

        self._normal_recenter_extra_spin = QSpinBox()
        self._normal_recenter_extra_spin.setRange(0, 300)
        self._normal_recenter_extra_spin.setValue(20)
        self._normal_recenter_extra_spin.setSuffix(' px')
        self._normal_recenter_extra_spin.setToolTip(
            'Extra margin used for the one-pass temporary ROI when full-blob recenter is triggered.'
        )
        adv.addLayout(spin_row('Recenter extra:', self._normal_recenter_extra_spin))

        self._normal_recenter_min_area_spin = QDoubleSpinBox()
        self._normal_recenter_min_area_spin.setRange(0.0, 5.0)
        self._normal_recenter_min_area_spin.setDecimals(2)
        self._normal_recenter_min_area_spin.setSingleStep(0.05)
        self._normal_recenter_min_area_spin.setValue(0.25)
        self._normal_recenter_min_area_spin.setToolTip(
            'Minimum first-pass blob area as a fraction of expected marker area before recenter is allowed. '
            'Prevents recentering from tiny noise fragments.'
        )
        adv.addLayout(spin_row('Recenter min area:', self._normal_recenter_min_area_spin))

        self._measurement_center_combo = QComboBox()
        self._measurement_center_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._measurement_center_combo.addItem('Full blob centroid', 'full_blob')
        self._measurement_center_combo.addItem('Peak-window centroid', 'peak_window')
        self._measurement_center_combo.addItem('Auto: blob or peak-window', 'auto')
        self._measurement_center_combo.setToolTip(
            'Normal marker mode: choose the accepted measurement centre. Full blob preserves legacy behaviour; '
            'peak-window can reduce motion-blur tail bias; auto switches only when diagnostics look suspicious.'
        )
        adv.addLayout(spin_row('Centre mode:', self._measurement_center_combo))

        self._adaptive_prefilter_check = QCheckBox('Adaptive prefilter after miss')
        self._adaptive_prefilter_check.setChecked(False)
        self._adaptive_prefilter_check.setStyleSheet('color:#CCCCCC; font-size:10px;')
        self._adaptive_prefilter_check.setToolTip(
            'Normal marker mode only: use sparse LAB proposals first for large search regions, then fall back '
            'to the full scan when needed. After a miss it also searches a larger recovery region; '
            'the first recovery frame is UNCERTAIN.'
        )
        adv.addWidget(self._adaptive_prefilter_check)

        self._lab_l_weight_spin = QDoubleSpinBox()
        self._lab_l_weight_spin.setRange(0.10, 5.00)
        self._lab_l_weight_spin.setDecimals(2)
        self._lab_l_weight_spin.setSingleStep(0.10)
        self._lab_l_weight_spin.setValue(1.00)
        self._lab_l_weight_spin.setToolTip('Normal marker mode: extra weight for LAB lightness. Raise for black/dark markers to reject bright neutral false locks.')
        adv.addLayout(spin_row('Lightness:', self._lab_l_weight_spin))

        self._lab_a_weight_spin = QDoubleSpinBox()
        self._lab_a_weight_spin.setRange(0.10, 5.00)
        self._lab_a_weight_spin.setDecimals(2)
        self._lab_a_weight_spin.setSingleStep(0.10)
        self._lab_a_weight_spin.setValue(1.00)
        self._lab_a_weight_spin.setToolTip('Normal marker mode: extra weight for LAB A channel.')
        adv.addLayout(spin_row('Red vs. Green:', self._lab_a_weight_spin))

        self._lab_b_weight_spin = QDoubleSpinBox()
        self._lab_b_weight_spin.setRange(0.10, 5.00)
        self._lab_b_weight_spin.setDecimals(2)
        self._lab_b_weight_spin.setSingleStep(0.10)
        self._lab_b_weight_spin.setValue(1.00)
        self._lab_b_weight_spin.setToolTip('Normal marker mode: extra weight for LAB B channel.')
        adv.addLayout(spin_row('Yellow vs. Blue:', self._lab_b_weight_spin))

        self._luminance_polarity_check = QCheckBox('Dark/light polarity gate')
        self._luminance_polarity_check.setChecked(True)
        self._luminance_polarity_check.setStyleSheet('color:#CCCCCC; font-size:10px;')
        self._luminance_polarity_check.setToolTip(
            'Normal marker mode: if initialization learns a dark marker on lighter background, reject bright candidates; '
            'if it learns a light marker, reject dark candidates. Neutral markers are unaffected.'
        )
        adv.addWidget(self._luminance_polarity_check)

        self._luminance_tolerance_spin = QDoubleSpinBox()
        self._luminance_tolerance_spin.setRange(0.0, 120.0)
        self._luminance_tolerance_spin.setDecimals(1)
        self._luminance_tolerance_spin.setSingleStep(2.0)
        self._luminance_tolerance_spin.setValue(35.0)
        self._luminance_tolerance_spin.setToolTip('Allowed LAB-L drift from the initialized marker before polarity rejects a candidate.')
        adv.addLayout(spin_row('Polarity tol.:', self._luminance_tolerance_spin))

        self._colour_diagnostics_check = QCheckBox('Save colour diagnostics')
        self._colour_diagnostics_check.setChecked(False)
        self._colour_diagnostics_check.setStyleSheet('color:#CCCCCC; font-size:10px;')
        self._colour_diagnostics_check.setToolTip(
            'Optional: include candidate LAB-L/background/polarity diagnostics in result metadata. '
            'Useful for investigating rare false locks; leave off for lighter exports/profiles.'
        )
        adv.addWidget(self._colour_diagnostics_check)

        self._shape_validation_combo = QComboBox()
        self._shape_validation_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._shape_validation_combo.addItem('Off', 'off')
        self._shape_validation_combo.addItem('Soft confidence only', 'soft')
        self._shape_validation_combo.addItem('Strict rejection', 'strict')
        self._shape_validation_combo.setCurrentIndex(1)
        adv.addLayout(spin_row('Shape validation:', self._shape_validation_combo))

        self._tiny_peak_threshold_spin = QDoubleSpinBox()
        self._tiny_peak_threshold_spin.setRange(0.05, 0.99)
        self._tiny_peak_threshold_spin.setDecimals(2)
        self._tiny_peak_threshold_spin.setSingleStep(0.05)
        self._tiny_peak_threshold_spin.setValue(0.55)
        self._tiny_peak_threshold_spin.setToolTip(
            'Tiny-feature mode: minimum colour-likelihood peak required to accept a measurement.'
        )
        adv.addLayout(spin_row('Tiny peak:', self._tiny_peak_threshold_spin))

        self._tiny_contrast_threshold_spin = QDoubleSpinBox()
        self._tiny_contrast_threshold_spin.setRange(0.0, 0.50)
        self._tiny_contrast_threshold_spin.setDecimals(3)
        self._tiny_contrast_threshold_spin.setSingleStep(0.01)
        self._tiny_contrast_threshold_spin.setValue(0.04)
        self._tiny_contrast_threshold_spin.setToolTip(
            'Tiny-feature mode: minimum local probability contrast against the surrounding patch.'
        )
        adv.addLayout(spin_row('Tiny contrast:', self._tiny_contrast_threshold_spin))

        self._tiny_init_contrast_spin = QDoubleSpinBox()
        self._tiny_init_contrast_spin.setRange(0.0, 80.0)
        self._tiny_init_contrast_spin.setDecimals(1)
        self._tiny_init_contrast_spin.setSingleStep(1.0)
        self._tiny_init_contrast_spin.setValue(8.0)
        self._tiny_init_contrast_spin.setToolTip(
            'Tiny-feature initialization: minimum Lab distance between clicked sample and local ring. '
            'Raise this to reject weak dark spots that resemble the background.'
        )
        adv.addLayout(spin_row('Init contrast:', self._tiny_init_contrast_spin))

        self._tiny_direction_cosine_spin = QDoubleSpinBox()
        self._tiny_direction_cosine_spin.setRange(-1.0, 1.0)
        self._tiny_direction_cosine_spin.setDecimals(2)
        self._tiny_direction_cosine_spin.setSingleStep(0.05)
        self._tiny_direction_cosine_spin.setValue(0.20)
        self._tiny_direction_cosine_spin.setToolTip(
            'Tiny-feature mode: candidate centre-vs-ring colour offset must point in '
            'approximately the same Lab direction learned at initialization.'
        )
        adv.addLayout(spin_row('Direction match:', self._tiny_direction_cosine_spin))

        self._tiny_update_confidence_spin = QDoubleSpinBox()
        self._tiny_update_confidence_spin.setRange(0.0, 1.0)
        self._tiny_update_confidence_spin.setDecimals(2)
        self._tiny_update_confidence_spin.setSingleStep(0.05)
        self._tiny_update_confidence_spin.setValue(0.65)
        self._tiny_update_confidence_spin.setToolTip(
            'Tiny-feature mode: Kalman state is updated only when accepted confidence exceeds this value.'
        )
        adv.addLayout(spin_row('Update conf.:', self._tiny_update_confidence_spin))

        self._tiny_mosse_mode_combo = QComboBox()
        self._tiny_mosse_mode_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._tiny_mosse_mode_combo.addItem('Off', 'off')
        self._tiny_mosse_mode_combo.addItem('Assisted', 'assisted')
        self._tiny_mosse_mode_combo.addItem('Primary', 'primary')
        self._tiny_mosse_mode_combo.setToolTip(
            'Tiny-feature mode: run a MOSSE correlation filter on the colour-likelihood map. '
            'Assisted falls back to the colour peak when MOSSE is weak; Primary requires MOSSE gates to pass.'
        )
        adv.addLayout(spin_row('MOSSE (tiny):', self._tiny_mosse_mode_combo))

        self._normal_mosse_mode_combo = QComboBox()
        self._normal_mosse_mode_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._normal_mosse_mode_combo.addItem('Off', 'off')
        self._normal_mosse_mode_combo.addItem('Assisted', 'assisted')
        self._normal_mosse_mode_combo.addItem('Primary', 'primary')
        self._normal_mosse_mode_combo.setToolTip('Normal-marker MOSSE proposes a local verification region; weak responses fall back to full candidate scanning.')
        adv.addLayout(spin_row('MOSSE (normal):', self._normal_mosse_mode_combo))

        self._mosse_scale_adaptation_check = QCheckBox('Scale can change')
        self._mosse_scale_adaptation_check.setChecked(False)
        self._mosse_scale_adaptation_check.setToolTip(
            'When MOSSE is enabled, test nearby feature scales and update size only from strong locks. '
            'This adds CPU work and is best for perspective or zoom changes.'
        )
        adv.addWidget(self._mosse_scale_adaptation_check)

        self._tiny_mosse_window_scale_spin = QDoubleSpinBox()
        self._tiny_mosse_window_scale_spin.setRange(2.0, 10.0)
        self._tiny_mosse_window_scale_spin.setDecimals(1)
        self._tiny_mosse_window_scale_spin.setSingleStep(0.5)
        self._tiny_mosse_window_scale_spin.setValue(4.0)
        self._tiny_mosse_window_scale_spin.setToolTip(
            'MOSSE window size as a multiple of expected feature diameter.'
        )
        adv.addLayout(spin_row('MOSSE window:', self._tiny_mosse_window_scale_spin))

        self._tiny_mosse_learning_rate_spin = QDoubleSpinBox()
        self._tiny_mosse_learning_rate_spin.setRange(0.0, 0.30)
        self._tiny_mosse_learning_rate_spin.setDecimals(3)
        self._tiny_mosse_learning_rate_spin.setSingleStep(0.01)
        self._tiny_mosse_learning_rate_spin.setValue(0.05)
        self._tiny_mosse_learning_rate_spin.setToolTip(
            'MOSSE update rate. Higher adapts faster but can drift into wrong specks.'
        )
        adv.addLayout(spin_row('MOSSE learn:', self._tiny_mosse_learning_rate_spin))

        self._tiny_mosse_psr_threshold_spin = QDoubleSpinBox()
        self._tiny_mosse_psr_threshold_spin.setRange(0.0, 30.0)
        self._tiny_mosse_psr_threshold_spin.setDecimals(1)
        self._tiny_mosse_psr_threshold_spin.setSingleStep(0.5)
        self._tiny_mosse_psr_threshold_spin.setValue(6.0)
        self._tiny_mosse_psr_threshold_spin.setToolTip(
            'Minimum MOSSE peak-to-sidelobe ratio for accepting the correlation peak.'
        )
        adv.addLayout(spin_row('MOSSE PSR:', self._tiny_mosse_psr_threshold_spin))

        self._tiny_mosse_peak_margin_spin = QDoubleSpinBox()
        self._tiny_mosse_peak_margin_spin.setRange(0.0, 1.0)
        self._tiny_mosse_peak_margin_spin.setDecimals(2)
        self._tiny_mosse_peak_margin_spin.setSingleStep(0.05)
        self._tiny_mosse_peak_margin_spin.setValue(0.15)
        self._tiny_mosse_peak_margin_spin.setToolTip(
            'Minimum relative separation between the best and second-best MOSSE response peaks.'
        )
        adv.addLayout(spin_row('MOSSE margin:', self._tiny_mosse_peak_margin_spin))

        self._debug_enabled_check = QCheckBox('Enable tracker debug recorder')
        self._debug_enabled_check.setChecked(False)
        self._debug_enabled_check.setStyleSheet('color:#CCCCCC; font-size:10px;')
        self._debug_enabled_check.setToolTip(
            'When enabled, keep a rolling debug history and export it when this tracker becomes uncertain/lost.'
        )
        adv.addWidget(self._debug_enabled_check)

        self._debug_history_spin = QSpinBox()
        self._debug_history_spin.setRange(1, 2000)
        self._debug_history_spin.setValue(120)
        self._debug_history_spin.setSuffix(' frames')
        self._debug_history_spin.setToolTip('Number of previous frames kept in the rolling debug history. Set very low values only when you need tiny debug reports.')
        adv.addLayout(spin_row('Debug history:', self._debug_history_spin))

        self._debug_uncertain_policy_combo = QComboBox()
        self._debug_uncertain_policy_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD;')
        self._debug_uncertain_policy_combo.addItem('None', 'none')
        self._debug_uncertain_policy_combo.addItem('Colour peak position', 'peak')
        self._debug_uncertain_policy_combo.addItem('Kalman mean position', 'kalman')
        self._debug_uncertain_policy_combo.setToolTip(
            'Debug reports only: when a frame is UNCERTAIN/LOST, store this diagnostic coordinate. '
            'This does not change exported measured positions, which remain None when missing.'
        )
        adv.addLayout(spin_row('Uncertain coord:', self._debug_uncertain_policy_combo))

        restore_defaults_btn = QPushButton('Restore defaults for this point mode')
        restore_defaults_btn.setStyleSheet(
            'QPushButton { background:#2a2a2a; color:#DDDDDD; padding:4px; '
            'border-radius:3px; border:1px solid #444; font-size:9px; }'
            'QPushButton:hover { background:#3a3a3a; }'
        )
        restore_defaults_btn.clicked.connect(self._restore_current_point_defaults)
        adv.addWidget(restore_defaults_btn)

        # Move the high-value experiment controls out of the technical list.
        # The widgets remain the same instances, so presets and tracker config
        # construction continue to have one source of truth.
        self._simple_point_group = QGroupBox('Simple Point Settings')
        self._simple_point_group.setStyleSheet(mode_group.styleSheet())
        simple = QVBoxLayout(self._simple_point_group)

        def move_to_simple(widget: QWidget) -> None:
            for index in range(adv.count()):
                item = adv.itemAt(index)
                row = item.layout()
                if item.widget() is widget or (
                    row is not None and any(
                        row.itemAt(child_index).widget() is widget
                        for child_index in range(row.count())
                    )
                ):
                    moved = adv.takeAt(index)
                    if row is not None:
                        for child_index in range(row.count()):
                            child = row.itemAt(child_index).widget()
                            if child is not None:
                                child.setParent(self._simple_point_group)
                        simple.addLayout(row)
                    elif moved.widget() is not None:
                        moved.widget().setParent(self._simple_point_group)
                        simple.addWidget(moved.widget())
                    return

        for widget in (
            self._point_mode_combo,
            self._expected_diameter_spin,
            self._sample_size_spin,
            self._lab_l_weight_spin,
            self._lab_a_weight_spin,
            self._lab_b_weight_spin,
            self._tiny_mosse_mode_combo,
            self._normal_mosse_mode_combo,
            self._mosse_scale_adaptation_check,
        ):
            move_to_simple(widget)

        calibration_heading = QLabel('Export Calibration')
        calibration_heading.setStyleSheet('color:#72CBB8; font-size:10px; font-weight:bold; padding-top:8px;')
        simple.addWidget(calibration_heading)
        self._calibration_enabled_check = QCheckBox('Use physical scale')
        self._calibration_enabled_check.setChecked(False)
        simple.addWidget(self._calibration_enabled_check)
        def calibration_row(label: str, controls: QHBoxLayout) -> QHBoxLayout:
            row = QHBoxLayout()
            caption = QLabel(label)
            caption.setStyleSheet('color:#AAAAAA; font-size:10px;')
            row.addWidget(caption)
            row.addLayout(controls)
            return row
        self._calibration_ratio_spin = QDoubleSpinBox()
        self._calibration_ratio_spin.setRange(0.0001, 10000000.0)
        self._calibration_ratio_spin.setDecimals(4)
        self._calibration_ratio_spin.setValue(100.0)
        self._calibration_ratio_spin.setSuffix(' px')
        self._calibration_ratio_spin.valueChanged.connect(lambda _value: self._calibration_enabled_check.setChecked(True))
        self._calibration_unit_combo = QComboBox()
        self._calibration_unit_combo.addItem('m', 'm')
        self._calibration_unit_combo.addItem('cm', 'cm')
        self._calibration_unit_combo.addItem('mm', 'mm')
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self._calibration_ratio_spin)
        ratio_row.addWidget(self._calibration_unit_combo)
        simple.addLayout(calibration_row('Pixels per:', ratio_row))
        measure_button = QPushButton('Measure 2 points')
        measure_button.clicked.connect(self.calibration_measure_requested)
        origin_button = QPushButton('Set coordinate origin')
        origin_button.clicked.connect(self.coordinate_origin_requested)
        simple.addWidget(measure_button)
        simple.addWidget(origin_button)
        self._export_unit_combo = QComboBox()
        self._export_unit_combo.addItem('Pixels', 'px')
        self._export_unit_combo.addItem('Meters', 'm')
        self._export_unit_combo.addItem('Centimeters', 'cm')
        self._export_unit_combo.addItem('Millimeters', 'mm')
        simple.addLayout(spin_row('Export units:', self._export_unit_combo))
        self._named_settings_combo = QComboBox()
        self._named_settings_combo.setStyleSheet(
            'QComboBox { background:#2A2A2A; color:#FFFFFF; border:1px solid #555555; padding:3px; } '
            'QComboBox QAbstractItemView { background:#2A2A2A; color:#FFFFFF; selection-background-color:#3D5F82; }'
        )
        self._refresh_named_settings_presets()
        named_heading = QLabel('Named Settings Presets')
        named_heading.setStyleSheet('color:#AAAAAA; font-size:10px; padding-top:6px;')
        simple.addWidget(named_heading)
        simple.addWidget(self._named_settings_combo)
        self._named_settings_name = QLineEdit()
        self._named_settings_name.setPlaceholderText('Settings preset name')
        self._named_settings_name.setStyleSheet(
            'QLineEdit { background:#2A2A2A; color:#FFFFFF; border:1px solid #555555; padding:3px; } '
            'QLineEdit::placeholder { color:#B8B8B8; }'
        )
        save_named = QPushButton('Save settings')
        load_named = QPushButton('Load settings')
        save_named.clicked.connect(lambda: self.named_settings_preset_save_requested.emit(self._named_settings_name.text().strip()))
        load_named.clicked.connect(lambda: self.named_settings_preset_load_requested.emit(self._named_settings_combo.currentText()))
        simple.addWidget(self._named_settings_name)
        named_actions = QHBoxLayout()
        named_actions.addWidget(save_named)
        named_actions.addWidget(load_named)
        simple.addLayout(named_actions)
        for control in (self._calibration_enabled_check, self._calibration_ratio_spin,
                        self._calibration_unit_combo, self._export_unit_combo):
            signal = control.toggled if isinstance(control, QCheckBox) else (
                control.valueChanged if isinstance(control, QDoubleSpinBox) else control.currentIndexChanged
            )
            signal.connect(self.calibration_changed)

        # Rebuild the remaining advanced surface under lightweight headings.
        # Labels, rather than nested group boxes, keep the panel compact.
        advanced_items = []
        while adv.count():
            advanced_items.append(adv.takeAt(0))
        adv_items_remaining = list(advanced_items)

        def contains_widget(item, widget: QWidget) -> bool:
            if item.widget() is widget:
                return True
            row = item.layout()
            return row is not None and any(
                row.itemAt(child_index).widget() is widget
                for child_index in range(row.count())
            )

        def append_item(item) -> None:
            if item.widget() is not None:
                adv.addWidget(item.widget())
            elif item.layout() is not None:
                adv.addLayout(item.layout())

        def add_advanced_section(title: str, widgets: tuple[QWidget, ...], color: str) -> None:
            selected = []
            for item in list(adv_items_remaining):
                if any(contains_widget(item, widget) for widget in widgets):
                    adv_items_remaining.remove(item)
                    selected.append(item)
            if not selected:
                return
            heading = QLabel(title)
            heading.setStyleSheet(
                f'color:{color}; font-size:10px; font-weight:bold; padding:8px 0 2px 5px; '
                f'border-left:3px solid {color};'
            )
            adv.addWidget(heading)
            for item in selected:
                append_item(item)

        add_advanced_section('Normal Mode Only - Initialization', (
            self._normal_init_threshold_spin, self._normal_init_margin_spin,
            self._normal_init_max_area_ratio_spin, self._normal_init_growth_radius_spin,
            self._normal_init_close_spin,
        ), '#78B7FF')
        add_advanced_section('Tiny Mode Only - Initialization', (
            self._tiny_init_peak_threshold_spin,
        ), '#F0B36E')
        add_advanced_section('Both Modes - General Motion', (
            self._normal_search_margin_spin, self._miss_expansion_spin,
            self._max_search_margin_spin, self._max_prediction_error_spin,
            self._miss_motion_allowance_spin, self._accept_within_search_region_check,
        ), '#72CBB8')
        add_advanced_section('Kalman Only', (
            self._kalman_enabled_check, self._kalman_model_combo,
            self._measurement_noise_spin, self._process_noise_spin, self._gate_sigma_spin,
            self._min_search_spin, self._max_search_spin,
        ), '#8ED39A')
        add_advanced_section('Normal Mode Only - Search and Validation', (
            self._normal_recenter_check, self._normal_recenter_edge_spin,
            self._normal_recenter_extra_spin, self._normal_recenter_min_area_spin,
            self._measurement_center_combo, self._adaptive_prefilter_check,
            self._luminance_polarity_check, self._luminance_tolerance_spin,
            self._colour_diagnostics_check, self._shape_validation_combo,
        ), '#78B7FF')
        add_advanced_section('Tiny Mode Only - Validation', (
            self._tiny_peak_threshold_spin, self._tiny_contrast_threshold_spin,
            self._tiny_init_contrast_spin, self._tiny_direction_cosine_spin,
            self._tiny_update_confidence_spin,
        ), '#F0B36E')
        add_advanced_section('Both Modes - MOSSE Tuning', (
            self._tiny_mosse_window_scale_spin, self._tiny_mosse_learning_rate_spin,
            self._tiny_mosse_psr_threshold_spin, self._tiny_mosse_peak_margin_spin,
        ), '#72CBB8')
        add_advanced_section('Both Modes - Diagnostics', (
            self._debug_enabled_check, self._debug_history_spin,
            self._debug_uncertain_policy_combo,
        ), '#72CBB8')
        if adv_items_remaining:
            heading = QLabel('Other Advanced Settings')
            heading.setStyleSheet('color:#8FB9D8; font-size:10px; font-weight:bold; padding-top:6px;')
            adv.addWidget(heading)
            for item in adv_items_remaining:
                append_item(item)

        self._connect_point_preset_autosave()
        self._apply_point_preset('normal')

        def _toggle_advanced(enabled: bool) -> None:
            for i in range(adv.count()):
                item = adv.itemAt(i)
                if item.widget() is not None:
                    item.widget().setVisible(enabled)
                elif item.layout() is not None:
                    layout = item.layout()
                    for j in range(layout.count()):
                        child = layout.itemAt(j).widget()
                        if child is not None:
                            child.setVisible(enabled)

        self._advanced_point_group.toggled.connect(_toggle_advanced)
        _toggle_advanced(False)
        self._tiny_mosse_mode_combo.currentIndexChanged.connect(self._update_scale_adaptation_availability)
        self._normal_mosse_mode_combo.currentIndexChanged.connect(self._update_scale_adaptation_availability)
        self._update_scale_adaptation_availability()

        root.addWidget(param_group)
        root.addWidget(self._simple_point_group)
        root.addWidget(self._advanced_point_group)

        # ---- Add tracker button ----
        add_btn = QPushButton('+ Add Tracker')
        add_btn.setStyleSheet(
            'QPushButton { background:#1a5c1a; color:white; padding:6px; '
            'border-radius:4px; font-weight:bold; }'
            'QPushButton:hover { background:#226622; }'
        )
        add_btn.clicked.connect(self._on_add_clicked)
        root.addWidget(add_btn)

        # ---- Active / queued tracker lists ----
        self._tracker_tabs = QTabWidget()
        self._tracker_tabs.setStyleSheet('QTabWidget::pane { border:none; }')
        self._tracker_tabs.setMinimumHeight(380)
        self._active_list_widget, self._active_list_layout = self._make_tracker_list()
        self._queue_list_widget, self._queue_list_layout = self._make_tracker_list()
        self._lost_list_widget, self._lost_list_layout = self._make_tracker_list()
        self._tracker_tabs.addTab(self._active_list_widget, 'Active (0)')
        self._tracker_tabs.addTab(self._queue_list_widget, 'Queue (0)')
        self._tracker_tabs.addTab(self._lost_list_widget, 'Lost (0)')
        # Compatibility alias for lightweight auxiliary widgets owned by the
        # main window. New code should use ``add_auxiliary_row``.
        self._list_layout = self._active_list_layout
        root.addWidget(self._tracker_tabs, stretch=1)

        # ---- Batch controls ----
        self._track_backwards_check = QCheckBox('Track backward from initialization')
        self._track_backwards_check.setToolTip(
            'Also track from each tracker\'s initialization frame back to the start of the selected range.'
        )
        root.addWidget(self._track_backwards_check)

        self._run_btn = QPushButton('▶ Run Batch')
        self._run_btn.setStyleSheet(
            'QPushButton { background:#1a3a6c; color:white; padding:8px; '
            'border-radius:4px; font-weight:bold; }'
            'QPushButton:hover { background:#1a4a8c; }'
        )
        self._run_btn.clicked.connect(self.run_batch_requested)
        root.addWidget(self._run_btn)

        self._cancel_btn = QPushButton('■ Cancel')
        self._cancel_btn.setStyleSheet(
            'QPushButton { background:#4a1a1a; color:white; padding:4px; '
            'border-radius:4px; }'
            'QPushButton:hover { background:#6a2a2a; }'
        )
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_batch_requested)
        root.addWidget(self._cancel_btn)

        self._export_btn = QPushButton('💾 Export .npz')
        self._export_btn.setStyleSheet(
            'QPushButton { background:#2a2a2a; color:#DDDDDD; padding:6px; '
            'border-radius:4px; border:1px solid #444; }'
            'QPushButton:hover { background:#3a3a3a; }'
        )
        self._export_btn.clicked.connect(self.export_requested)
        root.addWidget(self._export_btn)

        self._export_excel_btn = QPushButton('📊 Export Excel')
        self._export_excel_btn.setStyleSheet(
            'QPushButton { background:#2a2a2a; color:#DDDDDD; padding:6px; '
            'border-radius:4px; border:1px solid #444; }'
            'QPushButton:hover { background:#3a3a3a; }'
        )
        self._export_excel_btn.clicked.connect(self.export_excel_requested)
        root.addWidget(self._export_excel_btn)

        self._export_csv_btn = QPushButton('Export CSV')
        self._export_csv_btn.setStyleSheet(
            'QPushButton { background:#2a2a2a; color:#DDDDDD; padding:6px; '
            'border-radius:4px; border:1px solid #444; }'
            'QPushButton:hover { background:#3a3a3a; }'
        )
        self._export_csv_btn.clicked.connect(self.export_csv_requested)
        root.addWidget(self._export_csv_btn)

        self._review_uncertain_btn = QPushButton('⚠ View Uncertain Frames')
        self._review_uncertain_btn.setStyleSheet(
            'QPushButton { background:#3a2f1a; color:#FFE08A; padding:6px; '
            'border-radius:4px; border:1px solid #665522; }'
            'QPushButton:hover { background:#4a3a22; }'
        )
        self._review_uncertain_btn.clicked.connect(self.review_uncertain_requested)
        root.addWidget(self._review_uncertain_btn)

        self._learned_kalman_btn = QPushButton('Learned Kalman Values')
        self._learned_kalman_btn.setToolTip('Show motion settings learned from the most recent completed batch.')
        self._learned_kalman_btn.clicked.connect(self.learned_kalman_requested)
        root.addWidget(self._learned_kalman_btn)

        self._learned_preset_combo = QComboBox()
        self._learned_preset_combo.setToolTip('Load a named learned validation and Kalman preset.')
        self._refresh_learned_preset_names()
        root.addWidget(self._learned_preset_combo)
        load_learned = QPushButton('Load')
        load_learned.clicked.connect(self._load_selected_learned_preset)
        root.addWidget(load_learned)

    def _point_setting_values(self) -> dict:
        sample_size = int(self._sample_size_spin.value())
        if sample_size % 2 == 0:
            sample_size += 1
        return {
            "sample_size": sample_size,
            "expected_diameter": float(self._expected_diameter_spin.value()),
            "normal_init_threshold": float(self._normal_init_threshold_spin.value()),
            "normal_init_min_probability_margin": float(self._normal_init_margin_spin.value()),
            "normal_init_max_area_ratio": float(self._normal_init_max_area_ratio_spin.value()),
            "normal_init_growth_radius": float(self._normal_init_growth_radius_spin.value()),
            "normal_init_close_iterations": int(self._normal_init_close_spin.value()),
            "tiny_init_peak_threshold": float(self._tiny_init_peak_threshold_spin.value()),
            "kalman_enabled": bool(self._kalman_enabled_check.isChecked()),
            "kalman_model": self._kalman_model_combo.currentData(),
            "measurement_noise": float(self._measurement_noise_spin.value()),
            "process_noise": float(self._process_noise_spin.value()),
            "kalman_gate_sigma": float(self._gate_sigma_spin.value()),
            "min_search_radius": int(self._min_search_spin.value()),
            "max_search_radius": int(self._max_search_spin.value()),
            "normal_search_margin": int(self._normal_search_margin_spin.value()),
            "miss_expansion": int(self._miss_expansion_spin.value()),
            "max_search_margin": int(self._max_search_margin_spin.value()),
            "max_prediction_error": float(self._max_prediction_error_spin.value()),
            "miss_motion_allowance": float(self._miss_motion_allowance_spin.value()),
            "accept_within_search_region": self._accept_within_search_region_check.isChecked(),
            "normal_recenter_on_edge": bool(self._normal_recenter_check.isChecked()),
            "normal_recenter_edge_px": int(self._normal_recenter_edge_spin.value()),
            "normal_recenter_extra_margin": int(self._normal_recenter_extra_spin.value()),
            "normal_recenter_min_area_ratio": float(self._normal_recenter_min_area_spin.value()),
            "measurement_center_mode": self._measurement_center_combo.currentData(),
            "adaptive_prefilter_enabled": bool(self._adaptive_prefilter_check.isChecked()),
            "lab_l_weight": float(self._lab_l_weight_spin.value()),
            "lab_a_weight": float(self._lab_a_weight_spin.value()),
            "lab_b_weight": float(self._lab_b_weight_spin.value()),
            "luminance_polarity_enabled": bool(self._luminance_polarity_check.isChecked()),
            "luminance_polarity_min_delta": 8.0,
            "luminance_polarity_tolerance": float(self._luminance_tolerance_spin.value()),
            "colour_diagnostics_enabled": bool(self._colour_diagnostics_check.isChecked()),
            "shape_validation": self._shape_validation_combo.currentData(),
            "tiny_peak_threshold": float(self._tiny_peak_threshold_spin.value()),
            "tiny_contrast_threshold": float(self._tiny_contrast_threshold_spin.value()),
            "tiny_init_contrast_threshold": float(self._tiny_init_contrast_spin.value()),
            "tiny_direction_min_cosine": float(self._tiny_direction_cosine_spin.value()),
            "tiny_update_confidence_threshold": float(self._tiny_update_confidence_spin.value()),
            "tiny_mosse_mode": self._tiny_mosse_mode_combo.currentData(),
            "normal_mosse_mode": self._normal_mosse_mode_combo.currentData(),
            "mosse_scale_adaptation": self._mosse_scale_adaptation_check.isChecked(),
            "tiny_mosse_window_scale": float(self._tiny_mosse_window_scale_spin.value()),
            "tiny_mosse_learning_rate": float(self._tiny_mosse_learning_rate_spin.value()),
            "tiny_mosse_psr_threshold": float(self._tiny_mosse_psr_threshold_spin.value()),
            "tiny_mosse_peak_margin_threshold": float(self._tiny_mosse_peak_margin_spin.value()),
            "debug_uncertain_center_policy": self._debug_uncertain_policy_combo.currentData(),
        }

    def current_point_fast_config(self) -> TrackerConfig:
        values = self._point_setting_values()
        return TrackerConfig(
            tracker_type=TrackerType.POINT_FAST,
            name='',
            point_mode=str(self._point_mode_combo.currentData() or 'normal'),
            **values,
        )

    def current_named_settings_payload(self) -> dict[str, object]:
        return {
            'point_mode': str(self._point_mode_combo.currentData() or 'normal'),
            'point_settings': self._point_setting_values(),
            'calibration': self.export_calibration(),
        }

    def apply_named_settings_payload(self, payload: dict[str, object]) -> None:
        mode = str(payload.get('point_mode', 'normal'))
        index = self._point_mode_combo.findData(mode)
        if index >= 0:
            self._point_mode_combo.setCurrentIndex(index)
        values = payload.get('point_settings', {})
        if isinstance(values, dict):
            self._point_preset_values[mode] = dict(values)
            self._apply_point_preset(mode)
        calibration = payload.get('calibration', {})
        if isinstance(calibration, dict):
            self.apply_export_calibration(calibration)

    def _refresh_named_settings_presets(self) -> None:
        from utils.point_presets import load_named_settings_presets
        selected = self._named_settings_combo.currentText() if hasattr(self, '_named_settings_combo') else ''
        self._named_settings_combo.clear()
        for name in sorted(load_named_settings_presets()):
            self._named_settings_combo.addItem(name)
        index = self._named_settings_combo.findText(selected)
        if index >= 0:
            self._named_settings_combo.setCurrentIndex(index)

    def named_settings_preset_saved(self, name: str) -> None:
        self._refresh_named_settings_presets()
        index = self._named_settings_combo.findText(name)
        if index >= 0:
            self._named_settings_combo.setCurrentIndex(index)

    def _set_combo_data(self, combo: QComboBox, value: object) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    def _apply_point_preset(self, mode: str) -> None:
        mode = 'tiny' if str(mode) == 'tiny' else 'normal'
        values = dict(default_point_preset(mode))
        values.update(self._point_preset_values.get(mode, {}))
        self._loading_point_preset = True
        try:
            self._sample_size_spin.setValue(int(values.get('sample_size', 5)))
            self._expected_diameter_spin.setValue(float(values.get('expected_diameter', 20.0)))
            self._normal_init_threshold_spin.setValue(float(values.get('normal_init_threshold', 0.72)))
            self._normal_init_margin_spin.setValue(float(values.get('normal_init_min_probability_margin', 0.08)))
            self._normal_init_max_area_ratio_spin.setValue(float(values.get('normal_init_max_area_ratio', 6.0)))
            self._normal_init_growth_radius_spin.setValue(float(values.get('normal_init_growth_radius', 0.0)))
            self._normal_init_close_spin.setValue(int(values.get('normal_init_close_iterations', 1)))
            self._tiny_init_peak_threshold_spin.setValue(float(values.get('tiny_init_peak_threshold', 0.40)))
            self._kalman_enabled_check.setChecked(bool(values.get('kalman_enabled', False)))
            self._set_combo_data(self._kalman_model_combo, values.get('kalman_model', 'constant_velocity'))
            self._measurement_noise_spin.setValue(float(values.get('measurement_noise', 0.7)))
            self._process_noise_spin.setValue(float(values.get('process_noise', 0.5)))
            self._gate_sigma_spin.setValue(float(values.get('kalman_gate_sigma', 4.0)))
            self._min_search_spin.setValue(int(values.get('min_search_radius', 12)))
            self._max_search_spin.setValue(int(values.get('max_search_radius', 80)))
            self._normal_search_margin_spin.setValue(int(values.get('normal_search_margin', 15)))
            self._miss_expansion_spin.setValue(int(values.get('miss_expansion', 16)))
            self._max_search_margin_spin.setValue(int(values.get('max_search_margin', 150)))
            self._max_prediction_error_spin.setValue(float(values.get('max_prediction_error', 22.0)))
            self._miss_motion_allowance_spin.setValue(float(values.get('miss_motion_allowance', 14.0)))
            self._accept_within_search_region_check.setChecked(bool(values.get('accept_within_search_region', False)))
            self._normal_recenter_check.setChecked(bool(values.get('normal_recenter_on_edge', True)))
            self._normal_recenter_edge_spin.setValue(int(values.get('normal_recenter_edge_px', 3)))
            self._normal_recenter_extra_spin.setValue(int(values.get('normal_recenter_extra_margin', 20)))
            self._normal_recenter_min_area_spin.setValue(float(values.get('normal_recenter_min_area_ratio', 0.25)))
            self._set_combo_data(self._measurement_center_combo, values.get('measurement_center_mode', 'full_blob'))
            self._adaptive_prefilter_check.setChecked(bool(values.get('adaptive_prefilter_enabled', False)))
            self._lab_l_weight_spin.setValue(float(values.get('lab_l_weight', 1.0)))
            self._lab_a_weight_spin.setValue(float(values.get('lab_a_weight', 1.0)))
            self._lab_b_weight_spin.setValue(float(values.get('lab_b_weight', 1.0)))
            self._luminance_polarity_check.setChecked(bool(values.get('luminance_polarity_enabled', True)))
            self._luminance_tolerance_spin.setValue(float(values.get('luminance_polarity_tolerance', 35.0)))
            self._colour_diagnostics_check.setChecked(bool(values.get('colour_diagnostics_enabled', False)))
            self._set_combo_data(self._shape_validation_combo, values.get('shape_validation', 'soft'))
            self._tiny_peak_threshold_spin.setValue(float(values.get('tiny_peak_threshold', 0.55)))
            self._tiny_contrast_threshold_spin.setValue(float(values.get('tiny_contrast_threshold', 0.04)))
            self._tiny_init_contrast_spin.setValue(float(values.get('tiny_init_contrast_threshold', 8.0)))
            self._tiny_direction_cosine_spin.setValue(float(values.get('tiny_direction_min_cosine', 0.20)))
            self._tiny_update_confidence_spin.setValue(float(values.get('tiny_update_confidence_threshold', 0.65)))
            self._set_combo_data(self._tiny_mosse_mode_combo, values.get('tiny_mosse_mode', 'off'))
            self._set_combo_data(self._normal_mosse_mode_combo, values.get('normal_mosse_mode', 'off'))
            self._mosse_scale_adaptation_check.setChecked(bool(values.get('mosse_scale_adaptation', False)))
            self._tiny_mosse_window_scale_spin.setValue(float(values.get('tiny_mosse_window_scale', 4.0)))
            self._tiny_mosse_learning_rate_spin.setValue(float(values.get('tiny_mosse_learning_rate', 0.05)))
            self._tiny_mosse_psr_threshold_spin.setValue(float(values.get('tiny_mosse_psr_threshold', 6.0)))
            self._tiny_mosse_peak_margin_spin.setValue(float(values.get('tiny_mosse_peak_margin_threshold', 0.15)))
            self._set_combo_data(self._debug_uncertain_policy_combo, values.get('debug_uncertain_center_policy', 'none'))
        finally:
            self._loading_point_preset = False

    def _save_current_point_preset(self) -> None:
        if self._loading_point_preset:
            return
        mode = self._point_mode_combo.currentData() or 'normal'
        values = self._point_setting_values()
        self._point_preset_values[str(mode)] = values
        save_point_preset(str(mode), values)

    def _connect_point_preset_autosave(self) -> None:
        controls = [
            self._sample_size_spin, self._expected_diameter_spin,
            self._normal_init_threshold_spin, self._normal_init_margin_spin,
            self._normal_init_max_area_ratio_spin, self._normal_init_growth_radius_spin, self._normal_init_close_spin,
            self._tiny_init_peak_threshold_spin,
            self._measurement_noise_spin, self._process_noise_spin,
            self._gate_sigma_spin, self._min_search_spin, self._max_search_spin,
            self._normal_search_margin_spin, self._miss_expansion_spin,
            self._max_search_margin_spin, self._max_prediction_error_spin,
            self._miss_motion_allowance_spin,
            self._normal_recenter_edge_spin, self._normal_recenter_extra_spin,
            self._normal_recenter_min_area_spin,
            self._lab_l_weight_spin, self._lab_a_weight_spin, self._lab_b_weight_spin,
            self._luminance_tolerance_spin,
            self._tiny_peak_threshold_spin, self._tiny_contrast_threshold_spin,
            self._tiny_init_contrast_spin, self._tiny_direction_cosine_spin,
            self._tiny_update_confidence_spin,
            self._tiny_mosse_window_scale_spin, self._tiny_mosse_learning_rate_spin,
            self._tiny_mosse_psr_threshold_spin, self._tiny_mosse_peak_margin_spin,
        ]
        for widget in controls:
            widget.valueChanged.connect(lambda *_: self._save_current_point_preset())
        self._kalman_enabled_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._accept_within_search_region_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._normal_recenter_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._kalman_model_combo.currentIndexChanged.connect(lambda *_: self._save_current_point_preset())
        self._measurement_center_combo.currentIndexChanged.connect(lambda *_: self._save_current_point_preset())
        self._adaptive_prefilter_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._luminance_polarity_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._colour_diagnostics_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._shape_validation_combo.currentIndexChanged.connect(lambda *_: self._save_current_point_preset())
        self._tiny_mosse_mode_combo.currentIndexChanged.connect(lambda *_: self._save_current_point_preset())
        self._normal_mosse_mode_combo.currentIndexChanged.connect(lambda *_: self._save_current_point_preset())
        self._mosse_scale_adaptation_check.toggled.connect(lambda *_: self._save_current_point_preset())
        self._debug_uncertain_policy_combo.currentIndexChanged.connect(lambda *_: self._save_current_point_preset())

    def apply_point_threshold_adjustments(self, adjustments: list[dict]) -> None:
        if not adjustments:
            return
        widgets = {
            "normal_init_threshold": self._normal_init_threshold_spin,
            "normal_init_min_probability_margin": self._normal_init_margin_spin,
            "normal_init_max_area_ratio": self._normal_init_max_area_ratio_spin,
            "normal_init_growth_radius": self._normal_init_growth_radius_spin,
            "normal_init_close_iterations": self._normal_init_close_spin,
            "tiny_init_peak_threshold": self._tiny_init_peak_threshold_spin,
            "tiny_peak_threshold": self._tiny_peak_threshold_spin,
            "tiny_contrast_threshold": self._tiny_contrast_threshold_spin,
            "tiny_init_contrast_threshold": self._tiny_init_contrast_spin,
            "tiny_mosse_psr_threshold": self._tiny_mosse_psr_threshold_spin,
            "tiny_mosse_peak_margin_threshold": self._tiny_mosse_peak_margin_spin,
        }
        for item in adjustments:
            key = str(item.get("field", ""))
            if key in widgets:
                widget = widgets[key]
                value = float(item.get("new", widget.value()))
                if isinstance(widget, QSpinBox):
                    widget.setValue(int(round(value)))
                else:
                    widget.setValue(value)

    def apply_initialization_editor_values(self, values: dict) -> None:
        widgets = {
            "normal_init_threshold": self._normal_init_threshold_spin,
            "normal_init_min_probability_margin": self._normal_init_margin_spin,
            "normal_init_max_area_ratio": self._normal_init_max_area_ratio_spin,
            "normal_init_growth_radius": self._normal_init_growth_radius_spin,
            "normal_init_close_iterations": self._normal_init_close_spin,
        }
        for key, widget in widgets.items():
            if key not in values:
                continue
            if isinstance(widget, QSpinBox):
                widget.setValue(int(round(float(values[key]))))
            else:
                widget.setValue(float(values[key]))

    def _restore_current_point_defaults(self) -> None:
        mode = self._point_mode_combo.currentData() or 'normal'
        restored = restore_default_point_preset(str(mode))
        self._point_preset_values[str(mode)] = restored
        self._apply_point_preset(str(mode))

    def _on_point_mode_changed(self, _idx: int) -> None:
        """Save old mode values, then load autosaved values for the new mode."""
        if not getattr(self, '_loading_point_preset', False):
            old_mode = getattr(self, '_current_point_mode', 'normal')
            self._point_preset_values[str(old_mode)] = self._point_setting_values()
            save_point_preset(str(old_mode), self._point_preset_values[str(old_mode)])
        mode = self._point_mode_combo.currentData() or 'normal'
        self._current_point_mode = str(mode)
        self._apply_point_preset(str(mode))
        self._update_scale_adaptation_availability()

    def _update_scale_adaptation_availability(self, *_args) -> None:
        mode = self._point_mode_combo.currentData() or 'normal'
        mosse_mode = (
            self._tiny_mosse_mode_combo.currentData()
            if mode == 'tiny' else self._normal_mosse_mode_combo.currentData()
        )
        enabled = mosse_mode != 'off'
        self._mosse_scale_adaptation_check.setEnabled(enabled)
        if not enabled:
            self._mosse_scale_adaptation_check.setChecked(False)

    # ------------------------------------------------------------------
    # Slot: add button clicked
    # ------------------------------------------------------------------

    def _on_add_clicked(self) -> None:
        ttype = self._mode_combo.currentData()
        sample_size = int(self._sample_size_spin.value())
        if sample_size % 2 == 0:
            sample_size += 1

        cfg   = TrackerConfig(
            tracker_type=ttype,
            name=self._name_edit.text().strip(),
            # Legacy defaults retained for tracker types that still read them.
            # Current PointFastTracker uses the advanced point settings below.
            sigma=5.0,
            search_radius=60,
            point_mode=self._point_mode_combo.currentData(),
            sample_size=sample_size,
            expected_diameter=self._expected_diameter_spin.value(),
            normal_init_threshold=self._normal_init_threshold_spin.value(),
            normal_init_min_probability_margin=self._normal_init_margin_spin.value(),
            normal_init_max_area_ratio=self._normal_init_max_area_ratio_spin.value(),
            normal_init_growth_radius=self._normal_init_growth_radius_spin.value(),
            normal_init_close_iterations=self._normal_init_close_spin.value(),
            tiny_init_peak_threshold=self._tiny_init_peak_threshold_spin.value(),
            kalman_enabled=self._kalman_enabled_check.isChecked(),
            kalman_model=self._kalman_model_combo.currentData(),
            measurement_noise=self._measurement_noise_spin.value(),
            process_noise=self._process_noise_spin.value(),
            kalman_gate_sigma=self._gate_sigma_spin.value(),
            min_search_radius=self._min_search_spin.value(),
            max_search_radius=self._max_search_spin.value(),
            normal_search_margin=self._normal_search_margin_spin.value(),
            miss_expansion=self._miss_expansion_spin.value(),
            max_search_margin=self._max_search_margin_spin.value(),
            max_prediction_error=self._max_prediction_error_spin.value(),
            miss_motion_allowance=self._miss_motion_allowance_spin.value(),
            accept_within_search_region=self._accept_within_search_region_check.isChecked(),
            normal_recenter_on_edge=self._normal_recenter_check.isChecked(),
            normal_recenter_edge_px=self._normal_recenter_edge_spin.value(),
            normal_recenter_extra_margin=self._normal_recenter_extra_spin.value(),
            normal_recenter_min_area_ratio=self._normal_recenter_min_area_spin.value(),
            measurement_center_mode=self._measurement_center_combo.currentData(),
            adaptive_prefilter_enabled=self._adaptive_prefilter_check.isChecked(),
            lab_l_weight=self._lab_l_weight_spin.value(),
            lab_a_weight=self._lab_a_weight_spin.value(),
            lab_b_weight=self._lab_b_weight_spin.value(),
            luminance_polarity_enabled=self._luminance_polarity_check.isChecked(),
            luminance_polarity_min_delta=8.0,
            luminance_polarity_tolerance=self._luminance_tolerance_spin.value(),
            colour_diagnostics_enabled=self._colour_diagnostics_check.isChecked(),
            shape_validation=self._shape_validation_combo.currentData(),
            tiny_peak_threshold=self._tiny_peak_threshold_spin.value(),
            tiny_contrast_threshold=self._tiny_contrast_threshold_spin.value(),
            tiny_init_contrast_threshold=self._tiny_init_contrast_spin.value(),
            tiny_direction_min_cosine=self._tiny_direction_cosine_spin.value(),
            tiny_update_confidence_threshold=self._tiny_update_confidence_spin.value(),
            tiny_mosse_mode=self._tiny_mosse_mode_combo.currentData(),
            normal_mosse_mode=self._normal_mosse_mode_combo.currentData(),
            mosse_scale_adaptation=self._mosse_scale_adaptation_check.isChecked(),
            tiny_mosse_window_scale=self._tiny_mosse_window_scale_spin.value(),
            tiny_mosse_learning_rate=self._tiny_mosse_learning_rate_spin.value(),
            tiny_mosse_psr_threshold=self._tiny_mosse_psr_threshold_spin.value(),
            tiny_mosse_peak_margin_threshold=self._tiny_mosse_peak_margin_spin.value(),
            debug_enabled=self._debug_enabled_check.isChecked(),
            debug_history_frames=self._debug_history_spin.value(),
            debug_uncertain_center_policy=self._debug_uncertain_policy_combo.currentData(),
            learned_validation_samples=self._selected_learned_validation_samples,
        )
        self.add_tracker_requested.emit(cfg)
        self._name_edit.clear()

    def _refresh_learned_preset_names(self) -> None:
        selected = self._learned_preset_combo.currentText() if hasattr(self, '_learned_preset_combo') else ''
        presets = load_learned_presets()
        self._learned_preset_combo.clear()
        self._learned_preset_combo.addItem('Learned preset: none', None)
        for name in sorted(presets):
            self._learned_preset_combo.addItem(name, presets[name])
        index = self._learned_preset_combo.findText(selected)
        if index >= 0:
            self._learned_preset_combo.setCurrentIndex(index)

    def _load_selected_learned_preset(self) -> None:
        payload = self._learned_preset_combo.currentData()
        if not isinstance(payload, dict):
            self._selected_learned_validation_samples = None
            return
        kalman = dict(payload.get('kalman', {}) or {})
        self._kalman_enabled_check.setChecked(True)
        self._measurement_noise_spin.setValue(float(kalman.get('measurement_noise', self._measurement_noise_spin.value())))
        self._process_noise_spin.setValue(float(kalman.get('process_noise', self._process_noise_spin.value())))
        self._gate_sigma_spin.setValue(float(kalman.get('gate_sigma', self._gate_sigma_spin.value())))
        self._selected_learned_validation_samples = list(payload.get('validation_samples', []) or [])

    def learned_preset_saved(self, name: str) -> None:
        self._refresh_learned_preset_names()
        index = self._learned_preset_combo.findText(name)
        if index >= 0:
            self._learned_preset_combo.setCurrentIndex(index)

    def export_calibration(self) -> dict[str, object]:
        unit = str(self._calibration_unit_combo.currentData())
        meters_per_unit = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[unit]
        return {
            'pixels_per_meter': (
                float(self._calibration_ratio_spin.value()) / meters_per_unit
                if self._calibration_enabled_check.isChecked() else None
            ),
            'export_unit': str(self._export_unit_combo.currentData()),
        }

    def set_calibration_pixels_per_meter(self, pixels_per_meter: float) -> None:
        unit = str(self._calibration_unit_combo.currentData())
        meters_per_unit = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[unit]
        self._calibration_ratio_spin.setValue(float(pixels_per_meter) * meters_per_unit)
        self._calibration_enabled_check.setChecked(True)

    def apply_export_calibration(self, values: dict[str, object]) -> None:
        unit = str(values.get('ratio_unit', 'm'))
        index = self._calibration_unit_combo.findData(unit)
        if index >= 0:
            self._calibration_unit_combo.setCurrentIndex(index)
        ppm = values.get('pixels_per_meter')
        if ppm is not None:
            meters_per_unit = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[unit]
            self._calibration_ratio_spin.setValue(float(ppm) * meters_per_unit)
        self._calibration_enabled_check.setChecked(bool(values.get('enabled', ppm is not None)))
        export_unit = str(values.get('export_unit', 'px'))
        index = self._export_unit_combo.findData(export_unit)
        if index >= 0:
            self._export_unit_combo.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Public: add / remove rows
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tracker_list() -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('border:none;')
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(2)
        scroll.setWidget(content)
        return scroll, layout

    def _layout_for(self, location: str) -> QVBoxLayout:
        return {
            'active': self._active_list_layout,
            'queue': self._queue_list_layout,
            'lost': self._lost_list_layout,
        }[location]

    def _refresh_tracker_tab_labels(self) -> None:
        active_count = sum(location == 'active' for location in self._row_location.values())
        queue_count = sum(location == 'queue' for location in self._row_location.values())
        lost_count = sum(location == 'lost' for location in self._row_location.values())
        self._tracker_tabs.setTabText(0, f'Active ({active_count})')
        self._tracker_tabs.setTabText(1, f'Queue ({queue_count})')
        self._tracker_tabs.setTabText(2, f'Lost ({lost_count})')

    def add_row(self, uid: str, config: TrackerConfig, total_frames: int) -> None:
        row = TrackerRow(uid, config, total_frames)
        row.set_progress(0, total_frames)
        row.delete_requested.connect(self.remove_tracker_requested)
        row.select_requested.connect(self.tracker_selected)
        row.activity_requested.connect(self.tracker_activity_requested)
        row.end_frame_changed.connect(self.tracker_end_frame_changed)
        row.recovery_requested.connect(self.tracker_recovery_requested)
        self._rows[uid] = row
        self._frame_counts[uid] = total_frames
        self._row_location[uid] = 'active'
        self._active_list_layout.addWidget(row)
        self._refresh_tracker_tab_labels()

    def remove_row(self, uid: str) -> None:
        row = self._rows.pop(uid, None)
        if row:
            self._active_list_layout.removeWidget(row)
            self._queue_list_layout.removeWidget(row)
            self._lost_list_layout.removeWidget(row)
            row.deleteLater()
        auxiliary = self._auxiliary_rows.pop(uid, None)
        if auxiliary is not None:
            self._active_list_layout.removeWidget(auxiliary)
            self._queue_list_layout.removeWidget(auxiliary)
            self._lost_list_layout.removeWidget(auxiliary)
            auxiliary.deleteLater()
        self._frame_counts.pop(uid, None)
        self._row_location.pop(uid, None)
        self._refresh_tracker_tab_labels()

    def set_tracker_active(self, uid: str, active: bool) -> None:
        self.set_tracker_location(uid, 'active' if active else 'queue')

    def set_tracker_lost(
        self, uid: str, lost_frame: int, first_uncertain_frame: int
    ) -> None:
        self.set_tracker_location(uid, 'lost')
        row = self._rows.get(uid)
        if row is not None:
            row.set_lost(lost_frame, first_uncertain_frame)
        self._tracker_tabs.setCurrentIndex(2)

    def set_tracker_location(self, uid: str, location: str) -> None:
        row = self._rows.get(uid)
        if row is None or location not in {'active', 'queue', 'lost'}:
            return
        if self._row_location.get(uid) == location:
            return
        self._active_list_layout.removeWidget(row)
        self._queue_list_layout.removeWidget(row)
        self._lost_list_layout.removeWidget(row)
        auxiliary = self._auxiliary_rows.get(uid)
        if auxiliary is not None:
            self._active_list_layout.removeWidget(auxiliary)
            self._queue_list_layout.removeWidget(auxiliary)
            self._lost_list_layout.removeWidget(auxiliary)
        target = self._layout_for(location)
        target.addWidget(row)
        if auxiliary is not None:
            target.addWidget(auxiliary)
        self._row_location[uid] = location
        if location != 'lost':
            row.set_active(location == 'active')
        self._refresh_tracker_tab_labels()

    def add_auxiliary_row(self, uid: str, widget: QWidget) -> None:
        previous = self._auxiliary_rows.get(uid)
        if previous is not None and previous is not widget:
            self.remove_auxiliary_row(uid)
        self._auxiliary_rows[uid] = widget
        self._layout_for(self._row_location.get(uid, 'active')).addWidget(widget)

    def remove_auxiliary_row(self, uid: str) -> Optional[QWidget]:
        widget = self._auxiliary_rows.pop(uid, None)
        if widget is not None:
            self._active_list_layout.removeWidget(widget)
            self._queue_list_layout.removeWidget(widget)
            self._lost_list_layout.removeWidget(widget)
        return widget

    def update_status(self, uid: str, status: TrackerStatus) -> None:
        if uid in self._rows:
            self._rows[uid].set_status(status)

    def update_progress(self, uid: str, frame_index: int) -> None:
        if uid in self._rows:
            total = self._frame_counts.get(uid, 1)
            self._rows[uid].set_progress(frame_index, total)

    def update_uncertain_count(self, uid: str, count: int) -> None:
        if uid in self._rows:
            self._rows[uid].set_uncertain_count(count)

    def set_batch_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._track_backwards_check.setEnabled(not running)
        for row in self._rows.values():
            row.set_batch_running(running)

    def track_backwards_enabled(self) -> bool:
        return self._track_backwards_check.isChecked()


# ============================================================
#  TimelineWidget — bottom scrubber with per-tracker strips
# ============================================================

class TimelineWidget(QWidget):
    """
    Horizontal timeline showing:
    - Video scrubber (click to jump)
    - Start / end trim handles
    - Per-tracker coloured status strips (green=locked, yellow=uncertain, red=lost)
    - Current frame counter

    Signals
    -------
    frame_changed(index)
    range_changed(start, end)
    """

    frame_changed = pyqtSignal(int)
    range_changed = pyqtSignal(int, int)
    playback_toggled = pyqtSignal(bool)

    _STRIP_H = 8    # height of each tracker status strip in px
    _HEAD_H  = 28   # height of the main scrubber row

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self._HEAD_H + 6)
        self.setMinimumWidth(400)

        self._total_frames  = 1
        self._current_frame = 0
        self._start_frame   = 0
        self._end_frame     = 0

        # uid → list of (frame_idx, TrackerStatus)
        self._tracker_data: Dict[str, List[Tuple[int, TrackerStatus]]] = {}
        self._tracker_types: Dict[str, TrackerType] = {}

        self._dragging_head   = False
        self._dragging_start  = False
        self._dragging_end    = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._play_button = QToolButton()
        self._play_button.setText('▶')
        self._play_button.setCheckable(True)
        self._play_button.setFixedSize(28, 24)
        self._play_button.setToolTip('Play video (Space)')
        self._play_button.setStyleSheet(
            'QToolButton { color:white; background:#333; border:1px solid #555; }'
            'QToolButton:checked { background:#3b5870; }'
        )
        self._play_button.toggled.connect(self._on_play_toggled)
        layout.addWidget(self._play_button)

        self._frame_label = QLabel('Frame: 0 / 0')
        self._frame_label.setStyleSheet('color:#AAAAAA; font-size:10px; min-width:100px;')
        layout.addWidget(self._frame_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        self._time_label = QLabel('0.00s')
        self._time_label.setStyleSheet('color:#AAAAAA; font-size:10px; min-width:50px;')
        layout.addWidget(self._time_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_video(self, frame_count: int, fps: float) -> None:
        self._total_frames = max(1, frame_count)
        self._fps          = fps
        self._end_frame    = self._total_frames - 1
        self._slider.setMaximum(self._total_frames - 1)
        self._frame_label.setText(f'Frame: 0 / {self._total_frames}')

    def set_frame(self, index: int) -> None:
        self._current_frame = index
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._frame_label.setText(f'Frame: {index} / {self._total_frames}')
        fps = getattr(self, '_fps', 25.0)
        self._time_label.setText(f'{index/fps:.2f}s')

    def set_playing(self, playing: bool) -> None:
        self._play_button.blockSignals(True)
        self._play_button.setChecked(bool(playing))
        self._play_button.setText('Ⅱ' if playing else '▶')
        self._play_button.setToolTip('Pause video (Space)' if playing else 'Play video (Space)')
        self._play_button.blockSignals(False)

    def get_range(self) -> Tuple[int, int]:
        return self._start_frame, self._end_frame

    def set_tracker_data(self, uid: str, data: List[Tuple[int, TrackerStatus]],
                         ttype: TrackerType) -> None:
        self._tracker_data[uid]  = data
        self._tracker_types[uid] = ttype

    def remove_tracker(self, uid: str) -> None:
        self._tracker_data.pop(uid, None)
        self._tracker_types.pop(uid, None)

    def _on_slider_changed(self, value: int) -> None:
        self._current_frame = value
        self._frame_label.setText(f'Frame: {value} / {self._total_frames}')
        fps = getattr(self, '_fps', 25.0)
        self._time_label.setText(f'{value/fps:.2f}s')
        self.frame_changed.emit(value)

    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText('Ⅱ' if playing else '▶')
        self._play_button.setToolTip('Pause video (Space)' if playing else 'Play video (Space)')
        self.playback_toggled.emit(bool(playing))


# ============================================================
#  SettingsDialog — global channel + Hessian defaults
# ============================================================

class SettingsDialog(QDialog):
    """
    Global settings for channel extraction and Hessian parameters.

    Emits settings_applied(ChannelConfig) on OK.
    """

    settings_applied = pyqtSignal(object)   # ChannelConfig

    def __init__(self, current: ChannelConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setFixedWidth(380)
        self._config = current
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        grp = QGroupBox('Default Channel Extraction')
        gl  = QVBoxLayout(grp)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem('Greyscale (BT.601)', ChannelMode.GREY)
        self._mode_combo.addItem('Red channel',        ChannelMode.RED)
        self._mode_combo.addItem('Green channel',      ChannelMode.GREEN)
        self._mode_combo.addItem('Blue channel',       ChannelMode.BLUE)
        self._mode_combo.addItem('Custom weights',     ChannelMode.CUSTOM)
        # Set current
        for i in range(self._mode_combo.count()):
            if self._mode_combo.itemData(i) == self._config.mode:
                self._mode_combo.setCurrentIndex(i)
                break
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        gl.addWidget(self._mode_combo)

        # Custom weight fields
        self._custom_group = QGroupBox('Custom RGB Weights')
        cg = QHBoxLayout(self._custom_group)
        self._r_spin = QDoubleSpinBox(); self._r_spin.setRange(0, 10); self._r_spin.setValue(self._config.custom_weights[0]); self._r_spin.setPrefix('R ')
        self._g_spin = QDoubleSpinBox(); self._g_spin.setRange(0, 10); self._g_spin.setValue(self._config.custom_weights[1]); self._g_spin.setPrefix('G ')
        self._b_spin = QDoubleSpinBox(); self._b_spin.setRange(0, 10); self._b_spin.setValue(self._config.custom_weights[2]); self._b_spin.setPrefix('B ')
        for sp in (self._r_spin, self._g_spin, self._b_spin):
            sp.setSingleStep(0.05)
            sp.valueChanged.connect(self._update_preview_label)
            cg.addWidget(sp)
        gl.addWidget(self._custom_group)

        self._preview_label = QLabel()
        self._preview_label.setStyleSheet('color:#888888; font-size:10px;')
        gl.addWidget(self._preview_label)

        root.addWidget(grp)
        self._on_mode_changed(self._mode_combo.currentIndex())
        self._update_preview_label()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_mode_changed(self, idx: int) -> None:
        mode = self._mode_combo.itemData(idx)
        self._custom_group.setVisible(mode == ChannelMode.CUSTOM)

    def _update_preview_label(self) -> None:
        cfg = self._build_config()
        if cfg.mode == ChannelMode.CUSTOM:
            r, g, b = cfg.normalised_weights()
            self._preview_label.setText(
                f'Effective: {r:.2f}·R + {g:.2f}·G + {b:.2f}·B'
            )
        else:
            self._preview_label.setText('')

    def _build_config(self) -> ChannelConfig:
        mode = self._mode_combo.currentData()
        return ChannelConfig(
            mode=mode,
            custom_weights=(
                self._r_spin.value(),
                self._g_spin.value(),
                self._b_spin.value(),
            )
        )

    def _apply(self) -> None:
        cfg = self._build_config()
        self.settings_applied.emit(cfg)
        self.accept()


# ============================================================
#  RecoveryWidget — inline notification in tracker row when uncertain/lost
# ============================================================

class RecoveryWidget(QWidget):
    """
    Small inline widget shown below a tracker row when it's UNCERTAIN or LOST.
    Buttons:
      • Re-init: user will click on canvas to place a new seed
      • Dismiss: hide this widget, keep last known position
    """

    reinit_requested = pyqtSignal(str)    # uid
    dismissed        = pyqtSignal(str)    # uid

    def __init__(self, uid: str, status: TrackerStatus, parent=None) -> None:
        super().__init__(parent)
        self.uid = uid
        self._build_ui(status)

    def _build_ui(self, status: TrackerStatus) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        color = '#FFD700' if status == TrackerStatus.UNCERTAIN else '#FF3232'
        icon  = '⚠' if status == TrackerStatus.UNCERTAIN else '✖'
        lbl   = QLabel(f'{icon} {status.value.upper()}')
        lbl.setStyleSheet(f'color:{color}; font-size:10px;')
        layout.addWidget(lbl, stretch=1)

        reinit_btn = QPushButton('Re-seed')
        reinit_btn.setFixedHeight(22)
        reinit_btn.setStyleSheet(
            'QPushButton { background:#1a4a1a; color:white; font-size:9px; '
            'border-radius:3px; }'
            'QPushButton:hover { background:#226622; }'
        )
        reinit_btn.clicked.connect(lambda: self.reinit_requested.emit(self.uid))
        layout.addWidget(reinit_btn)

        dismiss_btn = QPushButton('Dismiss')
        dismiss_btn.setFixedHeight(22)
        dismiss_btn.setStyleSheet(
            'QPushButton { background:#2a2a2a; color:#888; font-size:9px; '
            'border-radius:3px; }'
        )
        dismiss_btn.clicked.connect(lambda: self.dismissed.emit(self.uid))
        layout.addWidget(dismiss_btn)


# ============================================================
#  AnalysisCachePanel — decoded workspace/NVDEC preparation controls
# ============================================================

class AnalysisCachePanel(QGroupBox):
    """Controls for analysis-workspace frame sourcing."""

    select_workspace_requested = pyqtSignal()
    full_frame_requested = pyqtSignal()
    prepare_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    source_mode_changed = pyqtSignal(str)  # "live" or "disk"
    transfer_mode_changed = pyqtSignal(str)  # fast_async | event_chained | strict_sync

    def __init__(self, parent=None) -> None:
        super().__init__('Analysis Workspace', parent)
        self.setStyleSheet(
            'QGroupBox { color:#AAAAAA; border:1px solid #3a3a3a; '
            'border-radius:4px; margin-top:6px; padding-top:8px; }'
        )
        self._workspace_text = 'Not selected'
        self._source_mode = 'live'
        self._transfer_mode = 'event_chained'
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 6)
        layout.setSpacing(4)

        self._workspace_label = QLabel('Workspace: Not selected')
        self._workspace_label.setWordWrap(True)
        self._workspace_label.setStyleSheet('color:#BBBBBB; font-size:9px;')
        layout.addWidget(self._workspace_label)

        row = QHBoxLayout()
        self._select_btn = QPushButton('Select')
        self._full_btn = QPushButton('Full frame')
        for button in (self._select_btn, self._full_btn):
            button.setStyleSheet(
                'QPushButton { background:#2a2a2a; color:#DDDDDD; padding:4px; '
                'border-radius:3px; border:1px solid #444; font-size:9px; }'
                'QPushButton:hover { background:#3a3a3a; }'
            )
            row.addWidget(button)
        self._select_btn.clicked.connect(self.select_workspace_requested)
        self._full_btn.clicked.connect(self.full_frame_requested)
        layout.addLayout(row)

        mode_row = QHBoxLayout()
        mode_lbl = QLabel('Frame source:')
        mode_lbl.setStyleSheet('color:#AAAAAA; font-size:9px;')
        mode_row.addWidget(mode_lbl)
        self._source_combo = QComboBox()
        self._source_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD; font-size:9px;')
        self._source_combo.addItem('Live RAM buffer', 'live')
        self._source_combo.addItem('Reusable disk cache', 'disk')
        self._source_combo.setToolTip(
            'Live RAM buffer decodes/crops ahead without SSD writes. '\
            'Reusable disk cache writes a workspace cache for repeated reruns.'
        )
        self._source_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        mode_row.addWidget(self._source_combo, stretch=1)
        layout.addLayout(mode_row)

        transfer_row = QHBoxLayout()
        transfer_lbl = QLabel('NVDEC transfer:')
        transfer_lbl.setStyleSheet('color:#AAAAAA; font-size:9px;')
        transfer_row.addWidget(transfer_lbl)
        self._transfer_combo = QComboBox()
        self._transfer_combo.setStyleSheet('background:#2a2a2a; color:#DDDDDD; font-size:9px;')
        self._transfer_combo.addItem('Fast async', 'fast_async')
        self._transfer_combo.addItem('Event chained', 'event_chained')
        self._transfer_combo.addItem('Strict sync', 'strict_sync')
        self._transfer_combo.setToolTip(
            'Fast async is the original fastest path. '
            'Event chained uses CUDA events to order crop and copy. '
            'Strict sync is safest/diagnostic and may be slower.'
        )
        self._transfer_combo.setCurrentIndex(1)
        self._transfer_combo.currentIndexChanged.connect(self._on_transfer_mode_changed)
        transfer_row.addWidget(self._transfer_combo, stretch=1)
        layout.addLayout(transfer_row)

        row2 = QHBoxLayout()
        self._prepare_btn = QPushButton('Prepare Disk Cache')
        self._clear_btn = QPushButton('Clear')
        self._prepare_btn.setStyleSheet(
            'QPushButton { background:#1a4a6c; color:white; padding:5px; '
            'border-radius:3px; font-size:9px; font-weight:bold; }'
            'QPushButton:hover { background:#205d87; }'
        )
        self._clear_btn.setStyleSheet(
            'QPushButton { background:#2a2a2a; color:#AAAAAA; padding:5px; '
            'border-radius:3px; font-size:9px; }'
        )
        self._prepare_btn.clicked.connect(self.prepare_requested)
        self._clear_btn.clicked.connect(self.clear_requested)
        row2.addWidget(self._prepare_btn)
        row2.addWidget(self._clear_btn)
        layout.addLayout(row2)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(14)
        self._progress.setStyleSheet(
            'QProgressBar { border:1px solid #444; color:#AAAAAA; font-size:8px; '
            'background:#202020; text-align:center; }'
            'QProgressBar::chunk { background:#1a5c1a; }'
        )
        layout.addWidget(self._progress)

        self._decoder_label = QLabel('Decoder: not prepared')
        self._decoder_label.setWordWrap(True)
        self._decoder_label.setStyleSheet('color:#AAAAAA; font-size:9px;')
        layout.addWidget(self._decoder_label)

        self._cache_label = QLabel('Cache: idle')
        self._cache_label.setWordWrap(True)
        self._cache_label.setStyleSheet('color:#888888; font-size:9px;')
        layout.addWidget(self._cache_label)

        self._sync_prepare_button()

    def reset(self) -> None:
        self.set_workspace_text('Not selected')
        self._progress.setValue(0)
        self.set_decoder_status('not prepared')
        self.set_cache_status('idle')
        self.set_preparing(False)
        self.set_source_mode('live')
        self.set_transfer_mode('event_chained')

    def set_workspace_text(self, text: str) -> None:
        self._workspace_text = text
        self._workspace_label.setText(f'Workspace: {text}')

    def set_preparing(self, preparing: bool) -> None:
        self._select_btn.setEnabled(not preparing)
        self._full_btn.setEnabled(not preparing)
        self._source_combo.setEnabled(not preparing)
        self._transfer_combo.setEnabled(not preparing)
        self._prepare_btn.setEnabled((not preparing) and self._source_mode == 'disk')
        self._clear_btn.setEnabled(not preparing)

    def _sync_prepare_button(self) -> None:
        self._prepare_btn.setEnabled(self._source_mode == 'disk')

    def _on_source_mode_changed(self, _idx: int) -> None:
        mode = self._source_combo.currentData() or 'live'
        self._source_mode = str(mode)
        self._sync_prepare_button()
        self.source_mode_changed.emit(self._source_mode)

    def _on_transfer_mode_changed(self, _idx: int) -> None:
        mode = self._transfer_combo.currentData() or 'event_chained'
        self._transfer_mode = str(mode)
        self.transfer_mode_changed.emit(self._transfer_mode)

    def transfer_mode(self) -> str:
        return self._transfer_mode

    def set_transfer_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in {'fast_async', 'event_chained', 'strict_sync'}:
            mode = 'event_chained'
        for index in range(self._transfer_combo.count()):
            if self._transfer_combo.itemData(index) == mode:
                blocked = self._transfer_combo.blockSignals(True)
                self._transfer_combo.setCurrentIndex(index)
                self._transfer_combo.blockSignals(blocked)
                break
        self._transfer_mode = mode

    def source_mode(self) -> str:
        return self._source_mode

    def set_source_mode(self, mode: str) -> None:
        mode = 'disk' if str(mode) == 'disk' else 'live'
        for index in range(self._source_combo.count()):
            if self._source_combo.itemData(index) == mode:
                blocked = self._source_combo.blockSignals(True)
                self._source_combo.setCurrentIndex(index)
                self._source_combo.blockSignals(blocked)
                break
        self._source_mode = mode
        self._sync_prepare_button()

    def set_progress(self, completed: int, total: int) -> None:
        percentage = 0 if total <= 0 else int(round(100.0 * completed / total))
        self._progress.setValue(max(0, min(100, percentage)))

    def set_decoder_status(self, text: str) -> None:
        self._decoder_label.setText(f'Decoder: {text}')

    def set_cache_status(self, text: str) -> None:
        self._cache_label.setText(f'Cache: {text}')
