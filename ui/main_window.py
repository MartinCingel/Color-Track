"""
ui/main_window.py
-----------------
Top-level application window. Wires together:
  VideoCanvas, TrackerPanel, TimelineWidget, TrackerManager, FrameBuffer.

Layout
------
  ┌─────────────────────────────────────────┐
  │  Menu bar                               │
  ├────────────┬────────────────────────────┤
  │ TrackerPanel│    VideoCanvas             │
  ├────────────┴────────────────────────────┤
  │  TimelineWidget                         │
  └─────────────────────────────────────────┘

Interaction flow
----------------
1. File → Open Video  →  VideoReader + FrameBuffer created
2. User selects mode in panel, clicks Add Tracker
3. Tracker registered; canvas enters seed-click mode
4. User clicks / drags on canvas → manager.initialize_tracker()
5. InitPreview rendered on canvas
6. Run Batch → worker thread processes all frames
7. User scrubs timeline → canvas shows results per frame
8. Lock-loss → RecoveryWidget appears; user re-seeds
9. File → Export .npz
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QDialog, QDialogButtonBox, QFrame, QHeaderView, QLineEdit,
    QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QStatusBar, QDoubleSpinBox, QComboBox,
    QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from core.export import estimate_export_size_mb, export_csv, export_npz, export_xlsx
from core.frame_buffer import FrameBuffer
from core.tracker_manager import TrackerManager
from core.video_reader import VideoReader
from core.workspace_cache import (
    CacheBuildResult,
    CacheBuildWorker,
    DEFAULT_RAM_LOOKAHEAD_BYTES,
    WorkspaceDiskCache,
    WorkspaceRect,
    decode_single_workspace_frame,
)
from core.ffmpeg_decoder import d3d11va_availability_message
from gpu.color_mask import ColorTolerance, sample_pixel, threshold_preview
import numpy as np

from tracking.base_tracker import (
    InitPreview, TrackerConfig, TrackerStatus, TrackerType,
)
from ui.color_tolerance_panel import ColorTolerancePanel
from ui.panels import (
    AnalysisCachePanel, RecoveryWidget, TimelineWidget, TrackerPanel,
)
from ui.video_canvas import VideoCanvas
from ui.debug_replay import DebugReplayDialog
from ui.init_editor import InitializationEditorPanel
from ui.uncertain_review import UncertainFrameReviewDialog
from utils.point_presets import (
    load_export_calibration, load_named_settings_presets, save_export_calibration,
    save_learned_preset, save_named_settings_preset,
)
from utils import state as app_state
from utils.tracker_debug import cleanup_debug_reports


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Color Track')
        self.resize(1280, 780)
        self.setStyleSheet(
            'QMainWindow { background:#1a1a1a; } '
            'QToolTip { color:#FFFFFF; background:#4A4A4A; '
            'border:1px solid #777777; padding:4px; }'
        )

        # Core objects (created on video open)
        self._reader:  Optional[VideoReader]   = None
        self._buffer:  Optional[FrameBuffer]   = None
        self._manager: Optional[TrackerManager] = None

        # UI state
        self._current_frame:    int             = 0
        self._pending_uid:      Optional[str]   = None   # tracker awaiting seed
        self._recovery_pending_uid: Optional[str] = None
        self._pending_roi:      Optional[Tuple] = None
        self._needs_roi:        bool            = False
        self._recovery_widgets: Dict[str, RecoveryWidget]      = {}
        self._color_panels:     Dict[str, 'ColorTolerancePanel'] = {}
        self._batch_start_frame: int = 0
        self._timeline_update_interval: int = 300
        self._debug_replay_windows: list[DebugReplayDialog] = []
        self._uncertain_review_windows: list[UncertainFrameReviewDialog] = []
        self._init_editor: Optional[InitializationEditorPanel] = None
        self._init_editor_uid: Optional[str] = None
        self._selected_tracker_uid: Optional[str] = None
        self._export_origin_rc: Optional[tuple[float, float]] = None
        self._pending_calibration_distance_m: Optional[float] = None

        # Prepared analysis-workspace cache state.
        self._workspace: Optional[WorkspaceRect] = None
        self._selecting_workspace: bool = False
        self._workspace_cache: Optional[WorkspaceDiskCache] = None
        self._cache_worker: Optional[CacheBuildWorker] = None
        self._cache_thread: Optional[QThread] = None
        self._cache_ram_lookahead_bytes: int = DEFAULT_RAM_LOOKAHEAD_BYTES
        self._workspace_source_mode: str = 'live'  # live RAM buffer by default; disk cache optional
        self._live_nvdec_transfer_mode: str = 'event_chained'
        self._decoder_preference: str = 'auto'
        self._playback_running = False
        self._playback_next_due = 0.0
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(16)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._build_ui()
        self._build_menus()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Main content row
        content = QHBoxLayout()
        content.setSpacing(0)

        self._panel   = TrackerPanel()
        self._cache_panel = AnalysisCachePanel()
        self._canvas  = VideoCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._canvas)
        self._view_mode_buttons: Dict[str, QToolButton] = {}
        self._view_mode_group = QButtonGroup(self)
        self._view_mode_group.setExclusive(True)

        canvas_area = QWidget()
        canvas_layout = QVBoxLayout(canvas_area)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        view_bar = QWidget()
        view_bar.setStyleSheet('QWidget { background:#202020; } QToolButton { color:#CCCCCC; background:#2a2a2a; border:1px solid #444; padding:5px 10px; } QToolButton:checked { background:#3b5870; color:white; }')
        view_layout = QHBoxLayout(view_bar)
        view_layout.setContentsMargins(6, 4, 6, 4)
        view_layout.setSpacing(4)
        for mode, label in (
            ('video', 'Video'),
            ('original', 'Original'),
            ('probability', 'Probability'),
            ('overlay', 'Overlay'),
        ):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolTip(f'Show {label.lower()} view')
            button.clicked.connect(lambda _checked=False, selected=mode: self._set_main_view(selected))
            self._view_mode_group.addButton(button)
            self._view_mode_buttons[mode] = button
            view_layout.addWidget(button)
        view_layout.addStretch(1)
        self._view_mode_buttons['video'].setChecked(True)
        canvas_layout.addWidget(view_bar)
        canvas_layout.addWidget(self._view_stack, stretch=1)

        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addWidget(self._panel, stretch=1)
        sidebar_layout.addWidget(self._cache_panel)

        # Keep the bottom TimelineWidget visible even when Advanced Point
        # Settings is expanded.  Without this scroll wrapper, the left
        # sidebar can impose a very large minimum height on the main layout
        # and push the video-position bar out of view on smaller screens.
        self._sidebar_scroll = QScrollArea()
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setFixedWidth(270)
        self._sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored)
        self._sidebar_scroll.setWidget(sidebar)

        content.addWidget(self._sidebar_scroll)
        content.addWidget(canvas_area)
        root.addLayout(content, stretch=1)

        # Timeline
        self._timeline = TimelineWidget()
        root.addWidget(self._timeline)

        # Status bar
        self._status = QStatusBar()
        self._status.setStyleSheet('color:#AAAAAA; font-size:10px;')
        self.setStatusBar(self._status)
        self._status.showMessage('Ready. Open a video file to begin.')

        # Wire signals
        self._panel.add_tracker_requested.connect(self._on_add_tracker)
        self._panel.remove_tracker_requested.connect(self._on_remove_tracker)
        self._panel.tracker_activity_requested.connect(self._on_tracker_activity_requested)
        self._panel.tracker_end_frame_changed.connect(self._on_tracker_end_frame_changed)
        self._panel.tracker_recovery_requested.connect(self._on_tracker_recovery_requested)
        self._panel.run_batch_requested.connect(self._on_run_batch)
        self._panel.cancel_batch_requested.connect(self._on_cancel_batch)
        self._panel.export_requested.connect(self._on_export)
        self._panel.export_excel_requested.connect(self._on_export_excel)
        self._panel.export_csv_requested.connect(self._on_export_csv)
        self._panel.review_uncertain_requested.connect(self._on_review_uncertain_frames)
        self._panel.learned_kalman_requested.connect(self._on_show_learned_kalman)
        self._panel.tracker_selected.connect(self._on_tracker_selected)

        self._cache_panel.select_workspace_requested.connect(self._on_select_workspace)
        self._cache_panel.full_frame_requested.connect(self._on_use_full_frame_workspace)
        self._cache_panel.prepare_requested.connect(self._on_prepare_cache)
        self._cache_panel.clear_requested.connect(self._on_clear_cache)
        self._cache_panel.source_mode_changed.connect(self._on_workspace_source_mode_changed)
        self._cache_panel.transfer_mode_changed.connect(self._on_live_nvdec_transfer_mode_changed)

        self._canvas.frame_step_requested.connect(self._on_frame_step)
        self._canvas.seed_clicked.connect(self._on_seed_clicked)
        self._canvas.roi_selected.connect(self._on_roi_selected)
        self._canvas.calibration_points_selected.connect(self._on_calibration_points_selected)
        self._canvas.coordinate_origin_selected.connect(self._on_coordinate_origin_selected)
        self._canvas.quick_point_seed_requested.connect(self._on_quick_point_seed)
        self._panel.point_feature_size_changed.connect(self._canvas.set_quick_seed_diameter)
        self._canvas.set_quick_seed_diameter(self._panel.current_point_fast_config().expected_diameter)
        saved_calibration = load_export_calibration()
        self._panel.apply_export_calibration(saved_calibration)
        origin = saved_calibration.get('origin_rc')
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            self._export_origin_rc = (float(origin[0]), float(origin[1]))
        self._panel.calibration_changed.connect(self._save_export_calibration)
        self._panel.named_settings_preset_save_requested.connect(self._save_named_settings_preset)
        self._panel.named_settings_preset_load_requested.connect(self._load_named_settings_preset)
        self._panel.calibration_measure_requested.connect(self._on_calibration_measure_requested)
        self._panel.coordinate_origin_requested.connect(lambda: self._canvas.set_calibration_mode('origin'))

        self._timeline.frame_changed.connect(self._on_timeline_seek)
        self._timeline.playback_toggled.connect(self._set_playback_running)
        self._set_main_view('video')

    def _set_main_view(self, mode: str) -> None:
        """Select the normal video or one embedded initialization diagnostic view."""
        if mode == 'video' or self._init_editor is None:
            mode = 'video'
            self._view_stack.setCurrentWidget(self._canvas)
            self._canvas.setFocus()
        else:
            self._init_editor.set_view_mode(mode)
            self._view_stack.setCurrentWidget(self._init_editor)
        button = self._view_mode_buttons.get(mode)
        if button is not None:
            button.setChecked(True)
        for diagnostic_mode in ('original', 'probability', 'overlay'):
            self._view_mode_buttons[diagnostic_mode].setEnabled(self._init_editor is not None)

    def _clear_initialization_editor(self) -> None:
        if self._init_editor is not None:
            self._view_stack.removeWidget(self._init_editor)
            self._init_editor.deleteLater()
        self._init_editor = None
        self._init_editor_uid = None
        self._set_main_view('video')

    def _build_menus(self) -> None:
        from PyQt6.QtGui import QAction, QKeySequence
        bar = self.menuBar()
        bar.setStyleSheet('QMenuBar { background:#252525; color:#DDDDDD; }')
        self._decoder_combo = QComboBox(bar)
        self._decoder_combo.addItem('Decoder: Auto', 'auto')
        self._decoder_combo.addItem('Decoder: NVIDIA NVDEC', 'nvdec')
        self._decoder_combo.addItem('Decoder: FFmpeg D3D11VA', 'd3d11va')
        self._decoder_combo.addItem('Decoder: Native D3D11 GPU ROI', 'native_d3d11')
        self._decoder_combo.addItem('Decoder: CPU/OpenCV', 'cpu')
        self._decoder_combo.setToolTip('Choose the decoder for live workspace, disk cache, and playback prebuffer.')
        self._decoder_combo.setStyleSheet('QComboBox { color:white; background:#333; border:1px solid #555; padding:2px 8px; } QComboBox QAbstractItemView { color:white; background:#333; }')
        self._decoder_combo.currentIndexChanged.connect(self._on_decoder_preference_changed)
        bar.setCornerWidget(self._decoder_combo, Qt.Corner.TopRightCorner)

        def action(text, slot, shortcut=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            return a

        # File menu
        file_m = bar.addMenu('File')
        file_m.addAction(action('Open Video…',  self._on_open_video,  'Ctrl+O'))
        file_m.addAction(action('Export .npz…', self._on_export,      'Ctrl+S'))
        file_m.addAction(action('Export Excel…', self._on_export_excel))
        file_m.addSeparator()
        file_m.addAction(action('Quit', self.close, 'Ctrl+Q'))


        # View menu
        view_m = bar.addMenu('View')
        view_m.addAction(action('Reset View (R)', self._canvas.reset_view))

    @pyqtSlot(int)
    def _on_decoder_preference_changed(self, index: int) -> None:
        if not hasattr(self, '_decoder_combo'):
            return
        preference = str(self._decoder_combo.itemData(index) or 'auto')
        self._decoder_preference = preference
        if self._workspace_source_mode == 'live':
            self._activate_workspace_source()
        if self._buffer:
            self._buffer.set_playback_prebuffer_enabled(
                False, decoder_preference=preference,
            )
        labels = {
            'auto': 'Auto: NVIDIA NVDEC, then native D3D11 GPU ROI, then CPU fallback',
            'nvdec': 'NVIDIA NVDEC selected; CPU fallback when unavailable',
            'd3d11va': f'FFmpeg D3D11VA selected; {d3d11va_availability_message()} CPU fallback when unavailable',
            'native_d3d11': 'Native FFmpeg D3D11VA GPU ROI selected; CPU fallback when unavailable',
            'cpu': 'CPU/OpenCV selected',
        }
        message = labels[preference]
        self._cache_panel.set_decoder_status(message)
        self._status.showMessage(message)

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Video', '',
            'Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)'
        )
        if not path:
            return

        # Clean up existing cache preparation and video state.
        self._stop_cache_preparation()
        if self._buffer:
            self._buffer.close()
        if self._reader:
            self._reader.close()
        self._workspace = None
        self._workspace_cache = None
        self._workspace_source_mode = 'live'
        self._selecting_workspace = False
        self._cache_panel.reset()
        self._canvas.set_analysis_workspace(None)
        self._clear_initialization_editor()

        try:
            self._reader  = VideoReader(path)
            self._buffer  = FrameBuffer(self._reader)
            self._manager = TrackerManager(self._buffer)

            # Wire manager signals
            self._manager.frame_done.connect(self._on_batch_frame_done)
            self._manager.batch_done.connect(self._on_batch_done)
            self._manager.batch_error.connect(self._on_batch_error)
            self._manager.debug_report_ready.connect(self._on_debug_report_ready)
            self._manager.tracker_lost.connect(self._on_tracker_lost)

        except Exception as exc:
            QMessageBox.critical(self, 'Open Failed', str(exc))
            return

        # Initialise timeline
        self._timeline.set_video(self._reader.frame_count, self._reader.fps)
        self._current_frame = 0
        self._set_playback_running(False)

        # Reset app state
        app_state.reset()
        s = app_state.get()
        s.video_path   = path
        s.frame_count  = self._reader.frame_count
        s.fps          = self._reader.fps
        s.video_width  = self._reader.width
        s.video_height = self._reader.height
        s.end_frame    = self._reader.frame_count - 1

        self._show_frame(0)
        self._status.showMessage(
            f'Opened: {Path(path).name}  |  '
            f'{self._reader.width}×{self._reader.height}  |  '
            f'{self._reader.fps:.2f} fps  |  '
            f'{self._reader.frame_count} frames'
        )

    @pyqtSlot()
    def _on_export(self) -> None:
        if not self._manager or not self._reader:
            return
        est = estimate_export_size_mb(self._manager, self._reader)
        if est > 1000:
            ans = QMessageBox.question(
                self, 'Large Export',
                f'Estimated export size: {est/1024:.1f} GB. Continue?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Results', 'tracking_results.npz',
            'NumPy Archive (*.npz)'
        )
        if not path:
            return
        try:
            calibration = self._export_calibration_or_warn()
            if calibration is None:
                return
            summary = export_npz(path, self._manager, self._reader, calibration=calibration)
            QMessageBox.information(
                self, 'Export Complete',
                f'Exported {len(summary)} tracker(s) → {path}'
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Export Failed', str(exc))

    @pyqtSlot(float, float, float, float)
    def _on_calibration_points_selected(self, row1: float, col1: float, row2: float, col2: float) -> None:
        pixels = float(np.hypot(row2 - row1, col2 - col1))
        distance_m = self._pending_calibration_distance_m
        self._pending_calibration_distance_m = None
        if distance_m is None or pixels <= 1e-6 or distance_m <= 0.0:
            self._status.showMessage('Calibration points must be separated and known distance must be positive.')
            return
        self._panel.set_calibration_pixels_per_meter(pixels / distance_m)
        self._status.showMessage(f'Calibration set from {pixels:.2f} px.')

    @pyqtSlot()
    def _on_calibration_measure_requested(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle('Known Distance')
        layout = QVBoxLayout(dlg)
        row = QHBoxLayout()
        distance = QDoubleSpinBox(dlg)
        distance.setRange(0.0001, 10000000.0)
        distance.setDecimals(4)
        distance.setValue(1.0)
        unit = QComboBox(dlg)
        for value in ('m', 'cm', 'mm'):
            unit.addItem(value, value)
        row.addWidget(distance)
        row.addWidget(unit)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, parent=dlg)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        meters_per_unit = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}
        self._pending_calibration_distance_m = float(distance.value()) * meters_per_unit[str(unit.currentData())]
        self._canvas.set_calibration_mode('scale')
        self._status.showMessage('Click the two endpoints of the known distance on the video.')

    @pyqtSlot(float, float)
    def _on_coordinate_origin_selected(self, row: float, col: float) -> None:
        self._export_origin_rc = (float(row), float(col))
        self._save_export_calibration()
        self._status.showMessage(f'Coordinate origin set to ({col:.1f}, {row:.1f}) px.')

    def _save_export_calibration(self) -> None:
        values = self._panel.export_calibration()
        values['enabled'] = values['pixels_per_meter'] is not None
        values['ratio_unit'] = str(self._panel._calibration_unit_combo.currentData())
        values['origin_rc'] = self._export_origin_rc
        save_export_calibration(values)

    @pyqtSlot(str)
    def _save_named_settings_preset(self, name: str) -> None:
        if not name:
            self._status.showMessage('Enter a settings preset name first.')
            return
        payload = self._panel.current_named_settings_payload()
        calibration = dict(payload.get('calibration', {}) or {})
        calibration['origin_rc'] = self._export_origin_rc
        calibration['enabled'] = calibration.get('pixels_per_meter') is not None
        calibration['ratio_unit'] = str(self._panel._calibration_unit_combo.currentData())
        payload['calibration'] = calibration
        try:
            save_named_settings_preset(name, payload)
        except Exception as exc:
            QMessageBox.warning(self, 'Save Settings Preset Failed', str(exc))
            return
        self._panel.named_settings_preset_saved(name)
        self._status.showMessage(f'Saved settings preset "{name}".')

    @pyqtSlot(str)
    def _load_named_settings_preset(self, name: str) -> None:
        payload = load_named_settings_presets().get(str(name))
        if not isinstance(payload, dict):
            self._status.showMessage('Select a settings preset to load.')
            return
        self._panel.apply_named_settings_payload(payload)
        calibration = dict(payload.get('calibration', {}) or {})
        origin = calibration.get('origin_rc')
        self._export_origin_rc = (
            (float(origin[0]), float(origin[1]))
            if isinstance(origin, (list, tuple)) and len(origin) == 2 else None
        )
        self._save_export_calibration()
        self._status.showMessage(f'Loaded settings preset "{name}".')

    def _export_calibration_or_warn(self) -> Optional[dict[str, object]]:
        calibration = self._panel.export_calibration()
        unit = str(calibration['export_unit'])
        if unit != 'px' and calibration['pixels_per_meter'] is None:
            QMessageBox.information(self, 'Physical Scale Required', 'Set a physical scale before exporting meters, centimeters, or millimeters.')
            return None
        calibration['origin_rc'] = self._export_origin_rc
        return calibration

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_export_excel(self) -> None:
        if not self._manager or not self._reader:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Excel', '', 'Excel workbook (*.xlsx)'
        )
        if not path:
            return
        try:
            calibration = self._export_calibration_or_warn()
            if calibration is None:
                return
            summary = export_xlsx(path, self._manager, self._reader, calibration=calibration)
            self._status.showMessage(
                f'Exported Excel workbook with {sum(summary.values())} tracker-frame rows.'
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Excel Export Failed', str(exc))

    @pyqtSlot()
    def _on_export_csv(self) -> None:
        if not self._manager or not self._reader:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export CSV Results', 'tracking_results.csv', 'CSV Files (*.csv)'
        )
        if not path:
            return
        calibration = self._export_calibration_or_warn()
        if calibration is None:
            return
        try:
            summary = export_csv(path, self._manager, self._reader, calibration=calibration)
            QMessageBox.information(self, 'CSV Export Complete', f'Exported {len(summary)} tracker(s) to {path}')
        except Exception as exc:
            QMessageBox.critical(self, 'CSV Export Failed', str(exc))

    # ------------------------------------------------------------------
    # Analysis workspace cache
    # ------------------------------------------------------------------

    def _activate_workspace_source(self) -> None:
        """Apply the selected workspace source mode to FrameBuffer."""
        if not self._buffer:
            return
        if self._workspace is None:
            self._buffer.clear_analysis_cache()
            return
        if self._workspace_source_mode == 'live':
            self._workspace_cache = None
            self._buffer.set_live_analysis_workspace(
                self._workspace,
                ram_lookahead_bytes=self._cache_ram_lookahead_bytes,
                prefer_nvdec=True,
                gpu_id=0,
                nvdec_transfer_mode=self._live_nvdec_transfer_mode,
                decoder_preference=self._decoder_preference,
            )
        elif self._workspace_cache is not None:
            self._buffer.set_analysis_cache(
                self._workspace_cache,
                ram_lookahead_bytes=self._cache_ram_lookahead_bytes,
            )
        else:
            self._buffer.clear_analysis_cache()

    @pyqtSlot(str)
    def _on_workspace_source_mode_changed(self, mode: str) -> None:
        self._workspace_source_mode = 'disk' if mode == 'disk' else 'live'
        if self._workspace_source_mode == 'live':
            self._workspace_cache = None
            if self._buffer and self._workspace is not None:
                self._buffer.set_live_analysis_workspace(
                    self._workspace,
                    ram_lookahead_bytes=self._cache_ram_lookahead_bytes,
                    prefer_nvdec=True,
                    gpu_id=0,
                    nvdec_transfer_mode=self._live_nvdec_transfer_mode,
                    decoder_preference=self._decoder_preference,
                )
            self._cache_panel.set_cache_status('live RAM buffer selected; no SSD cache writes')
            self._cache_panel.set_decoder_status('live: NVDEC preferred, OpenCV fallback')
            self._status.showMessage('Live RAM buffer selected. Batch runs decode ahead without writing a disk cache.')
        else:
            if self._buffer:
                self._buffer.clear_analysis_cache()
            self._cache_panel.set_cache_status('disk cache mode selected; press Prepare Disk Cache')
            self._cache_panel.set_decoder_status('not prepared')
            self._status.showMessage('Reusable disk cache selected. Press Prepare Disk Cache before running for cached reruns.')


    @pyqtSlot(str)
    def _on_live_nvdec_transfer_mode_changed(self, mode: str) -> None:
        mode = str(mode or 'event_chained')
        if mode not in {'fast_async', 'event_chained', 'strict_sync'}:
            mode = 'event_chained'
        self._live_nvdec_transfer_mode = mode
        if self._buffer and self._workspace is not None and self._workspace_source_mode == 'live':
            self._buffer.set_live_analysis_workspace(
                self._workspace,
                ram_lookahead_bytes=self._cache_ram_lookahead_bytes,
                prefer_nvdec=True,
                gpu_id=0,
                nvdec_transfer_mode=self._live_nvdec_transfer_mode,
                decoder_preference=self._decoder_preference,
            )
        label = {
            'fast_async': 'fast async',
            'event_chained': 'event chained',
            'strict_sync': 'strict sync',
        }.get(mode, mode)
        self._cache_panel.set_decoder_status(f'live: NVDEC preferred, transfer {label}, OpenCV fallback')
        self._status.showMessage(f'Live NVDEC transfer mode set to {label}.')

    @pyqtSlot()
    def _on_select_workspace(self) -> None:
        if not self._reader:
            self._status.showMessage('Open a video before selecting an analysis workspace.')
            return
        if self._pending_uid:
            self._status.showMessage('Finish initializing the pending tracker before selecting a workspace.')
            return
        self._selecting_workspace = True
        self._canvas.set_roi_mode(True)
        self._status.showMessage(
            'Drag a generous analysis workspace containing all expected marker movement.'
        )

    @pyqtSlot()
    def _on_use_full_frame_workspace(self) -> None:
        if not self._reader:
            self._status.showMessage('Open a video before selecting an analysis workspace.')
            return
        self._selecting_workspace = False
        self._canvas.set_roi_mode(False)
        self._workspace = WorkspaceRect.full_frame(self._reader.width, self._reader.height)
        self._workspace_cache = None
        self._activate_workspace_source()
        self._canvas.set_analysis_workspace(self._workspace.xywh)
        estimated_gib = (
            self._reader.frame_count * self._workspace.width * self._workspace.height * 3
            / (1024 ** 3)
        )
        self._cache_panel.set_workspace_text(
            f'Full frame {self._workspace.width}×{self._workspace.height} '
            f'(~{estimated_gib:.2f} GiB cache)'
        )
        if self._workspace_source_mode == 'live':
            self._cache_panel.set_cache_status('full-frame live RAM buffer selected; no SSD cache writes')
            self._cache_panel.set_decoder_status('live: NVDEC preferred, OpenCV fallback')
        else:
            self._cache_panel.set_cache_status('workspace selected; press Prepare Disk Cache')
        self._status.showMessage(
            f'Full-frame workspace selected. Disk-cache size would be about {estimated_gib:.2f} GiB.'
        )

    @pyqtSlot()
    def _on_prepare_cache(self) -> None:
        if not self._reader or not self._buffer or self._workspace is None:
            self._status.showMessage('Select an analysis workspace or Full frame first.')
            return
        if self._workspace_source_mode != 'disk':
            self._status.showMessage('Switch Frame source to Reusable disk cache before preparing a disk cache.')
            return
        if self._cache_thread and self._cache_thread.isRunning():
            self._status.showMessage('Cache preparation is already running.')
            return

        estimated_gib = (
            self._reader.frame_count * self._workspace.width * self._workspace.height * 3
            / (1024 ** 3)
        )
        if estimated_gib > 4.0:
            answer = QMessageBox.question(
                self,
                'Large Analysis Cache',
                f'This workspace will require approximately {estimated_gib:.2f} GiB of disk space. '
                'Continue preparing the reusable cache?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        from utils.app_paths import app_data_dir
        cache_root = app_data_dir() / 'analysis_cache'
        self._cache_worker = CacheBuildWorker(
            video_path=self._reader.path,
            cache_root=cache_root,
            workspace=self._workspace,
            prefer_nvdec=True,
            gpu_id=0,
            decoder_preference=self._decoder_preference,
            ram_lookahead_bytes=self._cache_ram_lookahead_bytes,
        )
        self._cache_thread = QThread(self)
        self._cache_worker.moveToThread(self._cache_thread)
        self._cache_thread.started.connect(self._cache_worker.run)
        self._cache_worker.progress.connect(self._on_cache_progress)
        self._cache_worker.status.connect(self._on_cache_status)
        self._cache_worker.backend_selected.connect(self._on_cache_backend)
        self._cache_worker.finished.connect(self._on_cache_finished)
        self._cache_worker.failed.connect(self._on_cache_failed)
        self._cache_worker.finished.connect(self._cache_thread.quit)
        self._cache_worker.failed.connect(self._cache_thread.quit)
        self._cache_thread.finished.connect(self._on_cache_thread_finished)

        self._cache_panel.set_preparing(True)
        self._cache_panel.set_progress(0, max(1, self._reader.frame_count))
        self._cache_panel.set_cache_status('preparing…')
        self._cache_panel.set_decoder_status('attempting NVIDIA NVDEC…')
        self._status.showMessage(
            'Preparing analysis workspace cache in background. You can initialize trackers now.'
        )
        self._cache_thread.start()

    @pyqtSlot(int, int)
    def _on_cache_progress(self, completed: int, total: int) -> None:
        self._cache_panel.set_progress(completed, total)
        self._cache_panel.set_cache_status(f'preparing {completed} / {total} frames')

    @pyqtSlot(str)
    def _on_cache_status(self, message: str) -> None:
        self._cache_panel.set_cache_status(message)

    @pyqtSlot(str, str)
    def _on_cache_backend(self, backend: str, message: str) -> None:
        self._cache_panel.set_decoder_status(backend)
        self._status.showMessage(message)

    @pyqtSlot(object)
    def _on_cache_finished(self, result: CacheBuildResult) -> None:
        try:
            cache = WorkspaceDiskCache(result.cache_dir)
            self._workspace_cache = cache
            if self._buffer:
                self._buffer.set_analysis_cache(
                    cache, ram_lookahead_bytes=self._cache_ram_lookahead_bytes
                )
            self._workspace_source_mode = 'disk'
            self._cache_panel.set_source_mode('disk')
            source = 'reused' if result.reused_existing else 'ready'
            self._cache_panel.set_progress(cache.frame_count, cache.frame_count)
            self._cache_panel.set_cache_status(
                f'{source}: {cache.frame_count} frames; {cache.cache_dir.name}'
            )
            self._cache_panel.set_decoder_status(cache.backend)
            self._status.showMessage(
                f'Analysis cache {source}. Decoder backend: {cache.backend}.'
            )
        except Exception as exc:
            self._on_cache_failed(str(exc))

    @pyqtSlot(str)
    def _on_cache_failed(self, message: str) -> None:
        self._workspace_cache = None
        if self._buffer:
            self._buffer.clear_analysis_cache()
        self._cache_panel.set_cache_status('failed')
        self._cache_panel.set_decoder_status('unavailable')
        self._status.showMessage(f'Cache preparation failed: {message}')
        QMessageBox.warning(self, 'Cache Preparation Failed', message)

    @pyqtSlot()
    def _on_clear_cache(self) -> None:
        self._stop_cache_preparation()
        self._workspace_cache = None
        if self._buffer:
            if self._workspace_source_mode == 'live' and self._workspace is not None:
                self._buffer.set_live_analysis_workspace(
                    self._workspace,
                    ram_lookahead_bytes=self._cache_ram_lookahead_bytes,
                    prefer_nvdec=True,
                    gpu_id=0,
                    nvdec_transfer_mode=self._live_nvdec_transfer_mode,
                    decoder_preference=self._decoder_preference,
                )
            else:
                self._buffer.clear_analysis_cache()
        self._cache_panel.set_progress(0, 1)
        if self._workspace_source_mode == 'live':
            self._cache_panel.set_cache_status('live RAM buffer selected; no SSD cache writes')
            self._cache_panel.set_decoder_status('live: NVDEC preferred, OpenCV fallback')
            self._status.showMessage('Reusable disk cache cleared; live RAM buffer remains active.')
        else:
            self._cache_panel.set_cache_status('cache disabled; prepared files remain on disk')
            self._cache_panel.set_decoder_status('not active')
            self._status.showMessage('Prepared cache disabled for this session. Files remain in analysis_cache.')

    def _on_cache_thread_finished(self) -> None:
        self._cache_panel.set_preparing(False)
        if self._cache_worker:
            self._cache_worker.deleteLater()
        if self._cache_thread:
            self._cache_thread.deleteLater()
        self._cache_worker = None
        self._cache_thread = None

    def _stop_cache_preparation(self) -> None:
        if self._cache_worker:
            self._cache_worker.cancel()
        if self._cache_thread and self._cache_thread.isRunning():
            self._cache_thread.quit()
            self._cache_thread.wait(2000)
        self._cache_worker = None
        self._cache_thread = None

    # ------------------------------------------------------------------
    # Tracker management
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_add_tracker(self, config: TrackerConfig) -> None:
        if not self._manager:
            self._status.showMessage('Open a video first.')
            return
        if self._buffer:
            self._buffer.set_playback_prebuffer_enabled(False)
        tracker = self._manager.add_tracker(config)
        self._panel.add_row(
            tracker.uid, config,
            total_frames=self._reader.frame_count if self._reader else 1
        )
        self._pending_uid = tracker.uid
        # Determine if ROI drag is needed
        self._needs_roi = config.tracker_type in (
            TrackerType.BLOB_SIMPLE,
            TrackerType.BLOB_COMPLEX,
            TrackerType.COLOR_AREA,
        )
        self._canvas.set_roi_mode(self._needs_roi)
        self._set_seed_guide_for_config(config, enabled=not self._needs_roi)
        self._status.showMessage(
            f'Tracker "{tracker.name}" added. '
            + ('Drag a region on the video.' if self._needs_roi
               else 'Click on the object to track.')
        )

    @pyqtSlot(float, float)
    def _on_quick_point_seed(self, row: float, col: float) -> None:
        if not self._manager or self._pending_uid or self._canvas._calibration_mode is not None:
            return
        self._on_add_tracker(self._panel.current_point_fast_config())
        self._on_seed_clicked(row, col)

    @pyqtSlot(str)
    def _on_remove_tracker(self, uid: str) -> None:
        if self._manager:
            self._manager.remove_tracker(uid)
        self._remove_tolerance_panel(uid)
        self._panel.remove_row(uid)
        self._canvas.remove_tracker_overlay(uid)
        self._timeline.remove_tracker(uid)
        if uid in self._recovery_widgets:
            self._recovery_widgets.pop(uid).deleteLater()
        if uid == self._init_editor_uid:
            self._clear_initialization_editor()
        if uid == self._pending_uid:
            self._pending_uid = None
            self._canvas.set_seed_guide(None)

    @pyqtSlot(str, bool)
    def _on_tracker_activity_requested(self, uid: str, active: bool) -> None:
        if not self._manager:
            return
        self._manager.set_tracker_active(uid, active)
        self._panel.set_tracker_active(uid, active)
        self._status.showMessage(
            f'Tracker moved to {"Active" if active else "Queue"}: {uid}'
        )

    @pyqtSlot(str, object)
    def _on_tracker_end_frame_changed(self, uid: str, end_frame: object) -> None:
        if not self._manager:
            return
        self._manager.set_tracker_end_frame(uid, end_frame)

    @pyqtSlot(str, int, int)
    def _on_tracker_lost(
        self, uid: str, lost_frame: int, first_uncertain_frame: int
    ) -> None:
        self._panel.set_tracker_lost(uid, lost_frame, first_uncertain_frame)
        self._status.showMessage(
            f'Tracker "{uid}" paused at loss frame {lost_frame}. '
            f'First uncertain frame: {first_uncertain_frame}.'
        )

    @pyqtSlot(str)
    def _on_tracker_recovery_requested(self, uid: str) -> None:
        if not self._manager:
            return
        loss = self._manager.loss_record(uid)
        if loss is None:
            return
        tracker = self._manager.get_tracker(uid)
        if tracker is None:
            return
        self._recovery_pending_uid = uid
        self._pending_uid = uid
        self._needs_roi = tracker.config.tracker_type in (
            TrackerType.BLOB_SIMPLE,
            TrackerType.BLOB_COMPLEX,
            TrackerType.COLOR_AREA,
        )
        self._canvas.set_roi_mode(self._needs_roi)
        self._set_seed_guide_for_config(tracker.config, enabled=not self._needs_roi)
        self._show_frame(loss.lost_frame)
        self._timeline.set_frame(loss.lost_frame)
        self._status.showMessage(
            f'Reinitialize "{tracker.name}" from scratch. Choose any frame, then click the feature.'
        )

    @pyqtSlot(str)
    def _on_tracker_selected(self, uid: str) -> None:
        self._selected_tracker_uid = uid
        self._status.showMessage(f'Selected tracker: {uid}')

    # ------------------------------------------------------------------
    # Seed / ROI from canvas
    # ------------------------------------------------------------------

    def _initialization_frame_override_for_tracker(self, tracker):
        """Return a workspace-cache-style frame for initialization when possible.

        If a prepared cache is already active, TrackerManager will fetch the
        cached WorkspaceFrame itself.  When cache preparation is still running
        or no complete cache exists yet, this decodes just the current workspace
        frame through the same preferred NVDEC path, with OpenCV fallback.
        This keeps point-marker colour initialization consistent with the later
        prepared-cache tracking path.
        """
        if (
            not self._reader
            or self._workspace is None
            or self._workspace_cache is not None
            or not getattr(tracker, 'accepts_workspace_frame', False)
        ):
            return None
        try:
            decoded = decode_single_workspace_frame(
                video_path=self._reader.path,
                workspace=self._workspace,
                frame_index=self._current_frame,
                prefer_nvdec=True,
                gpu_id=0,
                decoder_preference=self._decoder_preference,
            )
            self._status.showMessage(
                f'{decoded.backend_message} Initializing from analysis workspace frame.'
            )
            return decoded.frame
        except Exception as exc:
            self._status.showMessage(
                'Workspace initialization-frame decode failed; falling back to normal display frame. '
                f'Reason: {exc}'
            )
            return None

    def _set_seed_guide_for_config(self, config: TrackerConfig, *, enabled: bool = True) -> None:
        diameter = (
            float(config.expected_diameter)
            if enabled and config.tracker_type == TrackerType.POINT_FAST
            else None
        )
        self._canvas.set_seed_guide(diameter)

    def _show_tracker_initialization_error(self, uid: str, exc: Exception) -> None:
        """Show initialization errors without letting them crash the UI.

        The tracker remains pending so the user can adjust settings and click/drag
        again. This is important for strict colour/area initialization thresholds.
        """
        tracker = self._manager.get_tracker(uid) if self._manager else None
        tracker_name = tracker.name if tracker is not None else uid
        message = str(exc) or exc.__class__.__name__
        self._pending_uid = uid
        self._canvas.set_roi_mode(bool(self._needs_roi))
        if tracker is not None:
            self._set_seed_guide_for_config(tracker.config, enabled=not self._needs_roi)
        self._status.showMessage(f'Tracker initialization failed for "{tracker_name}".')
        QMessageBox.warning(
            self,
            'Tracker Initialization Failed',
            (
                f'Could not initialize tracker "{tracker_name}".\n\n'
                f'{message}\n\n'
                'Adjust the initialization/threshold settings or click a more distinct feature, then try again.'
            ),
        )

    def _show_threshold_adjustments(self, preview: InitPreview) -> None:
        adjustments = list(getattr(preview, 'threshold_adjustments', []) or [])
        if not adjustments:
            return
        self._panel.apply_point_threshold_adjustments(adjustments)
        lines = []
        for item in adjustments:
            label = str(item.get('label', item.get('field', 'threshold')))
            old = float(item.get('old', 0.0))
            new = float(item.get('new', 0.0))
            metric = float(item.get('metric', 0.0))
            lines.append(f'{label}: {old:.3f} -> {new:.3f}  (init metric {metric:.3f})')
        QMessageBox.information(
            self,
            'Initialization Thresholds Adjusted',
            (
                'Current thresholds would reject the initialization frame.\n\n'
                'The tracker relaxed these thresholds to match the initialized feature:\n\n'
                + '\n'.join(lines)
                + '\n\nYou may lower them slightly for tolerance, or choose a stronger feature.'
            ),
        )

    def _show_initialization_editor(self, uid: str, preview: InitPreview) -> None:
        if not self._manager:
            return
        diagnostics = dict(getattr(preview, 'init_diagnostics', {}) or {})
        if not diagnostics.get('viewer_available'):
            return
        tracker = self._manager.get_tracker(uid)
        if tracker is None:
            return
        self._clear_initialization_editor()
        editor = InitializationEditorPanel(uid, tracker.name, preview, self)
        editor.preview_requested.connect(
            lambda parameters, cuts, tracker_uid=uid, panel=editor:
                self._on_initialization_editor_preview(tracker_uid, panel, parameters, cuts)
        )
        editor.apply_requested.connect(
            lambda parameters, cuts, tracker_uid=uid, panel=editor:
                self._on_initialization_editor_apply(tracker_uid, panel, parameters, cuts)
        )
        self._init_editor = editor
        self._init_editor_uid = uid
        self._view_stack.addWidget(editor)
        self._set_main_view('video')

    def _on_initialization_editor_preview(
        self,
        uid: str,
        dialog: InitializationEditorPanel,
        parameters: dict,
        cuts: list,
    ) -> None:
        if not self._manager:
            return
        try:
            preview = self._manager.preview_initialization_edit(uid, parameters, cuts)
        except Exception as exc:
            dialog.set_preview(None, error=str(exc) or exc.__class__.__name__)
            return
        if preview is None:
            return
        dialog.set_preview(preview)
        tracker = self._manager.get_tracker(uid)
        if tracker is not None:
            self._canvas.set_init_preview(
                uid, preview, tracker.config.tracker_type, tracker.name, self._current_frame
            )

    def _on_initialization_editor_apply(
        self,
        uid: str,
        dialog: InitializationEditorPanel,
        parameters: dict,
        cuts: list,
    ) -> None:
        if not self._manager:
            return
        try:
            preview = self._manager.apply_initialization_edit(uid, parameters, cuts)
        except Exception as exc:
            QMessageBox.warning(
                self,
                'Initialization Edit Rejected',
                str(exc) or exc.__class__.__name__,
            )
            return
        if preview is None:
            return
        tracker = self._manager.get_tracker(uid)
        if tracker is not None:
            self._canvas.set_init_preview(
                uid, preview, tracker.config.tracker_type, tracker.name, self._current_frame
            )
            self._panel.apply_initialization_editor_values(parameters)
            dialog.set_preview(preview)
            self._status.showMessage(f'Updated initialization for "{tracker.name}".')

    @pyqtSlot(float, float)
    def _on_seed_clicked(self, row: float, col: float) -> None:
        if not self._pending_uid or not self._manager:
            return
        if self._needs_roi:
            return   # waiting for ROI drag

        uid = self._pending_uid
        self._pending_uid = None
        self._canvas.set_roi_mode(False)
        self._canvas.set_seed_guide(None)
        restarting_lost_tracker = self._recovery_pending_uid == uid

        tracker = self._manager.get_tracker(uid)
        frame_override = (
            self._initialization_frame_override_for_tracker(tracker)
            if tracker is not None else None
        )
        try:
            if restarting_lost_tracker:
                preview = self._manager.restart_tracker_from_scratch(
                    uid, self._current_frame, row, col, roi=None, frame_override=frame_override
                )
            else:
                preview = self._manager.initialize_tracker(
                    uid, self._current_frame, row, col, roi=None, frame_override=frame_override
                )
        except Exception as exc:
            self._show_tracker_initialization_error(uid, exc)
            return

        if preview:
            t = self._manager.get_tracker(uid)
            if restarting_lost_tracker:
                self._recovery_pending_uid = None
                self._panel.set_tracker_active(uid, True)
                self._refresh_tracker_timeline(uid)
                self._panel.update_uncertain_count(uid, self._manager.uncertain_count(uid))
            self._canvas.set_init_preview(uid, preview, t.config.tracker_type, t.name, self._current_frame)
            self._show_threshold_adjustments(preview)
            self._show_initialization_editor(uid, preview)
            self._status.showMessage(f'Tracker "{t.name}" initialised on frame {self._current_frame}.')

        # If Color Area: show tolerance sliders
        if (t := self._manager.get_tracker(uid)) and \
           t.config.tracker_type == TrackerType.COLOR_AREA:
            self._show_color_tolerance_ui(uid)

    @pyqtSlot(int, int, int, int)
    def _on_roi_selected(self, x: int, y: int, w: int, h: int) -> None:
        if self._selecting_workspace:
            if not self._reader:
                return
            self._selecting_workspace = False
            self._canvas.set_roi_mode(False)
            self._workspace = WorkspaceRect(x, y, w, h).clamped(
                self._reader.width, self._reader.height
            )
            self._workspace_cache = None
            self._activate_workspace_source()
            self._canvas.set_analysis_workspace(self._workspace.xywh)
            estimated_gib = (
                self._reader.frame_count * self._workspace.width * self._workspace.height * 3
                / (1024 ** 3)
            )
            self._cache_panel.set_workspace_text(
                f'{self._workspace.width}×{self._workspace.height} at '
                f'({self._workspace.x}, {self._workspace.y}); ~{estimated_gib:.2f} GiB'
            )
            if self._workspace_source_mode == 'live':
                self._cache_panel.set_cache_status('live RAM buffer selected; no SSD cache writes')
                self._cache_panel.set_decoder_status('live: NVDEC preferred, OpenCV fallback')
                self._status.showMessage(
                    'Workspace selected. Live RAM buffer will decode ahead during batch; initialize trackers now.'
                )
            else:
                self._cache_panel.set_cache_status('workspace selected; press Prepare Disk Cache')
                self._status.showMessage(
                    'Workspace selected. Press Prepare Disk Cache, then initialize trackers while it runs.'
                )
            return

        if not self._pending_uid or not self._manager:
            return
        uid = self._pending_uid
        self._pending_uid = None
        self._canvas.set_roi_mode(False)
        self._canvas.set_seed_guide(None)
        self._pending_roi = (x, y, w, h)
        restarting_lost_tracker = self._recovery_pending_uid == uid

        seed_row = y + h / 2.0
        seed_col = x + w / 2.0
        tracker = self._manager.get_tracker(uid)
        frame_override = (
            self._initialization_frame_override_for_tracker(tracker)
            if tracker is not None else None
        )
        try:
            if restarting_lost_tracker:
                preview = self._manager.restart_tracker_from_scratch(
                    uid, self._current_frame, seed_row, seed_col,
                    roi=(x, y, w, h), frame_override=frame_override
                )
            else:
                preview = self._manager.initialize_tracker(
                    uid, self._current_frame, seed_row, seed_col,
                    roi=(x, y, w, h), frame_override=frame_override
                )
        except Exception as exc:
            self._show_tracker_initialization_error(uid, exc)
            return

        if preview:
            t = self._manager.get_tracker(uid)
            if restarting_lost_tracker:
                self._recovery_pending_uid = None
                self._panel.set_tracker_active(uid, True)
                self._refresh_tracker_timeline(uid)
                self._panel.update_uncertain_count(uid, self._manager.uncertain_count(uid))
            self._canvas.set_init_preview(uid, preview, t.config.tracker_type, t.name, self._current_frame)
            self._show_threshold_adjustments(preview)
            self._show_initialization_editor(uid, preview)
            self._status.showMessage(f'Tracker "{t.name}" initialised.')

        if (t := self._manager.get_tracker(uid)) and \
           t.config.tracker_type == TrackerType.COLOR_AREA:
            self._show_color_tolerance_ui(uid)

    def _show_color_tolerance_ui(self, uid: str) -> None:
        """
        Insert a ColorTolerancePanel into the tracker panel list beneath
        the relevant tracker row.  Wire slider changes to live preview updates.
        """
        if not self._manager:
            return

        tracker = self._manager.get_tracker(uid)
        if tracker is None:
            return

        # Get the current tolerance from the tracker
        from tracking.trackers import ColorAreaTracker
        if not isinstance(tracker, ColorAreaTracker):
            return

        tol = tracker._tolerance
        if tol is None:
            return

        # Remove any existing tolerance panel for this uid
        self._remove_tolerance_panel(uid)

        panel = ColorTolerancePanel(uid, tol)
        panel.tolerance_changed.connect(self._on_tolerance_changed)
        panel.confirmed.connect(self._on_tolerance_confirmed)

        # Insert into the scroll list after the tracker row
        self._color_panels[uid] = panel
        self._panel.add_auxiliary_row(uid, panel)

        self._status.showMessage(
            f'Adjust color tolerance — the magenta overlay shows matched pixels.'
        )

    def _remove_tolerance_panel(self, uid: str) -> None:
        panel = self._color_panels.pop(uid, None)
        if panel:
            self._panel.remove_auxiliary_row(uid)
            panel.deleteLater()

    @pyqtSlot(str, object)
    def _on_tolerance_changed(self, uid: str, tol: ColorTolerance) -> None:
        """
        Live update: push new tolerance to tracker, regenerate preview mask,
        push RGBA overlay to canvas.
        """
        if not self._manager or not self._buffer:
            return

        from tracking.trackers import ColorAreaTracker
        tracker = self._manager.get_tracker(uid)
        if not isinstance(tracker, ColorAreaTracker):
            return

        # Update tracker's tolerance
        tracker.update_tolerance(tol)

        # Regenerate preview on current frame
        frame_gpu = self._buffer.get_frame(self._current_frame)
        if frame_gpu is None:
            return

        from gpu.color_mask import threshold_preview
        preview_rgba = threshold_preview(frame_gpu, tol)

        # Build a synthetic InitPreview carrying just the color_preview
        preview = InitPreview(color_preview=preview_rgba)
        self._canvas.set_init_preview(
            uid, preview, TrackerType.COLOR_AREA,
            tracker.name, self._current_frame
        )

    @pyqtSlot(str)
    def _on_tolerance_confirmed(self, uid: str) -> None:
        """User clicked Confirm — hide tolerance panel, status ready."""
        self._remove_tolerance_panel(uid)
        self._status.showMessage(
            f'Color tolerance confirmed. Click "Run Batch" to start tracking.'
        )

    # ------------------------------------------------------------------
    # Re-seed (lock loss recovery)
    # ------------------------------------------------------------------

    def _enter_reinit_mode(self, uid: str) -> None:
        self._pending_uid = uid
        self._canvas.set_roi_mode(False)
        tracker = self._manager.get_tracker(uid) if self._manager else None
        if tracker is not None:
            self._set_seed_guide_for_config(tracker.config)
        self._status.showMessage(f'Click on the video to re-seed tracker "{uid}".')

    # ------------------------------------------------------------------
    # Temporary debug artifacts
    # ------------------------------------------------------------------

    def _cleanup_debug_artifacts(self) -> None:
        """Close debug replay windows and remove temporary debug PNG reports."""
        for dlg in list(self._debug_replay_windows):
            try:
                dlg.close()
            except Exception:
                pass
        self._debug_replay_windows.clear()
        try:
            cleanup_debug_reports()
        except Exception:
            # Debug cleanup must never block normal tracking.
            pass

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_run_batch(self) -> None:
        if not self._manager or not self._reader:
            return
        self._set_playback_running(False)
        if self._buffer:
            self._buffer.set_playback_prebuffer_enabled(False)
        # Debug reports are temporary tuning artifacts. Clear previous ROI /
        # likelihood / overlay images whenever the user starts a new run.
        self._cleanup_debug_artifacts()
        if self._workspace is not None and self._workspace_source_mode == 'live':
            self._activate_workspace_source()
        if self._workspace is not None and self._workspace_source_mode == 'disk' and self._workspace_cache is None:
            answer = QMessageBox.question(
                self,
                'Prepared Cache Not Ready',
                'Reusable disk cache mode is selected, but no prepared cache is active. '
                'Run with the live RAM buffer instead for this batch?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._workspace_source_mode = 'live'
            self._cache_panel.set_source_mode('live')
            self._activate_workspace_source()
        start, end = self._timeline.get_range()
        if end <= start:
            end = self._reader.frame_count

        self._batch_start_frame = start
        self._panel.set_batch_running(True)
        if self._workspace_source_mode == 'live' and self._workspace is not None:
            source_note = 'live RAM workspace buffer (NVDEC preferred, OpenCV fallback)'
        elif self._workspace_cache is not None:
            source_note = f'prepared cache ({self._workspace_cache.backend})'
        else:
            source_note = 'direct OpenCV decoding'
        backward_note = (
            ' plus backward tracking from each initialization frame'
            if self._panel.track_backwards_enabled() else ''
        )
        self._status.showMessage(
            f'Batch running: frames {start}–{end} using {source_note}{backward_note}…'
        )
        self._manager.start_batch(
            start,
            end,
            track_backwards=self._panel.track_backwards_enabled(),
        )

    @pyqtSlot()
    def _on_cancel_batch(self) -> None:
        if self._manager:
            self._manager.cancel_batch()

    def _refresh_tracker_timeline(self, uid: str) -> None:
        """Rebuild one tracker strip; called infrequently during batch runs."""
        if not self._manager:
            return
        data = [(r.frame_index, r.status) for r in self._manager.get_results(uid)]
        tracker = self._manager.get_tracker(uid)
        if tracker:
            self._timeline.set_tracker_data(uid, data, tracker.config.tracker_type)

    @pyqtSlot(str, int, str)
    def _on_batch_frame_done(self, uid: str, frame_idx: int, status_str: str) -> None:
        status = TrackerStatus(status_str)
        self._panel.update_status(uid, status)
        self._panel.update_progress(uid, frame_idx)

        # Timeline reconstruction is deliberately throttled. With several
        # trackers, rebuilding a growing strip for every frame creates a large
        # amount of UI work without improving the batch result.
        completed = frame_idx - self._batch_start_frame + 1
        if completed > 0 and completed % self._timeline_update_interval == 0:
            self._refresh_tracker_timeline(uid)

    @pyqtSlot()
    def _on_batch_done(self) -> None:
        self._panel.set_batch_running(False)
        if self._manager:
            for tracker in self._manager.all_trackers():
                self._refresh_tracker_timeline(tracker.uid)
        if self._buffer:
            self._buffer.set_playback_prebuffer_enabled(
                True,
                lookahead_frames=96,
                nvdec_transfer_mode='event_chained',
                decoder_preference=self._decoder_preference,
            )
        if self._manager:
            for tracker in self._manager.all_trackers():
                self._panel.update_uncertain_count(
                    tracker.uid, self._manager.uncertain_count(tracker.uid)
                )
        total = sum(
            len(self._manager.get_results(t.uid))
            for t in self._manager.all_trackers()
        ) if self._manager else 0
        profile_note = ""
        if self._manager and self._manager.last_profile_report_dir:
            profile_note = f" Profile report: {self._manager.last_profile_report_dir}"
        self._status.showMessage(f'Batch complete. {total} frames processed.{profile_note}')

    @pyqtSlot(str, str)
    def _on_debug_report_ready(self, uid: str, report_dir: str) -> None:
        self._status.showMessage(f'Debug report for tracker {uid}: {report_dir}')
        try:
            dlg = DebugReplayDialog(report_dir, parent=self)
            dlg.show()
            self._debug_replay_windows.append(dlg)
            # Drop closed windows from the reference list so they can be garbage-collected.
            dlg.finished.connect(lambda _code, d=dlg: self._debug_replay_windows.remove(d) if d in self._debug_replay_windows else None)
        except Exception as exc:
            self._status.showMessage(f'Debug report created but replay window failed: {exc}. Path: {report_dir}')

    @pyqtSlot()
    def _on_review_uncertain_frames(self) -> None:
        if not self._manager or not self._buffer:
            return
        total = sum(self._manager.uncertain_count(t.uid) for t in self._manager.all_trackers())
        if total <= 0:
            QMessageBox.information(self, 'No Uncertain Frames', 'There are no UNCERTAIN or LOST frames to review.')
            return
        dlg = UncertainFrameReviewDialog(self._manager, self._buffer, parent=self)
        dlg.correction_applied.connect(self._on_uncertain_correction_applied)
        dlg.show()
        self._uncertain_review_windows.append(dlg)
        dlg.finished.connect(lambda _code, d=dlg: self._uncertain_review_windows.remove(d) if d in self._uncertain_review_windows else None)

    @pyqtSlot()
    def _on_show_learned_kalman(self) -> None:
        if not self._manager:
            return
        suggestions = self._manager.kalman_learning_suggestions()
        if not suggestions:
            QMessageBox.information(
                self,
                'Learned Kalman Values',
                'No reliable global motion estimate is available yet. Complete a batch with more locked frames.',
            )
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Learned Kalman Values')
        dlg.setMinimumWidth(690)
        layout = QVBoxLayout(dlg)

        table = QTableWidget(len(suggestions), 5, dlg)
        table.setHorizontalHeaderLabels((
            'Tracker', 'Trusted frames', 'Measurement sigma', 'Process sigma', 'Gate sigma',
        ))
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        for row, item in enumerate(suggestions):
            values = (
                item.name,
                str(item.trusted_frames),
                f'{item.measurement_noise:.3f} px',
                f'{item.process_noise:.3f} px/fr²',
                f'{item.gate_sigma:.2f}',
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, column, cell)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(table)

        preset_row = QHBoxLayout()
        preset_name = QLineEdit(dlg)
        preset_name.setPlaceholderText('Preset name')
        preset_name.setStyleSheet(
            'QLineEdit { color:#FFFFFF; background:#303030; border:1px solid #666666; padding:4px; } '
            'QLineEdit::placeholder { color:#B8B8B8; }'
        )
        save_button = QPushButton('Save preset', dlg)
        preset_row.addWidget(preset_name)
        preset_row.addWidget(save_button)
        layout.addLayout(preset_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dlg)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        selected_index = next(
            (index for index, item in enumerate(suggestions) if item.uid == self._selected_tracker_uid),
            0,
        )
        table.selectRow(selected_index)

        def save_selected_preset() -> None:
            name = preset_name.text().strip()
            if not name:
                QMessageBox.information(dlg, 'Preset Name Required', 'Enter a name for the learned preset.')
                return
            row = table.currentRow()
            if not 0 <= row < len(suggestions):
                QMessageBox.information(dlg, 'Select a Profile', 'Select the learned profile to save.')
                return
            source = suggestions[row]
            if source.is_average:
                payload = {
                    'validation_samples': [],
                    'kalman': {
                        'measurement_noise': source.measurement_noise,
                        'process_noise': source.process_noise,
                        'gate_sigma': source.gate_sigma,
                    },
                }
            else:
                payload = self._manager.learned_preset_payload(source.uid)
            if payload is None:
                QMessageBox.warning(dlg, 'Save Learned Preset Failed', 'The selected tracker has no learned profile to save.')
                return
            try:
                save_learned_preset(name, payload)
            except Exception as exc:
                QMessageBox.warning(dlg, 'Save Learned Preset Failed', str(exc))
                return
            self._panel.learned_preset_saved(name)
            self._status.showMessage(f'Saved learned preset "{name}".')
            dlg.accept()

        save_button.clicked.connect(save_selected_preset)
        dlg.exec()

    @pyqtSlot(str, int)
    def _on_uncertain_correction_applied(self, uid: str, frame_index: int) -> None:
        if self._manager:
            self._panel.update_uncertain_count(uid, self._manager.uncertain_count(uid))
            self._refresh_tracker_timeline(uid)
        self._status.showMessage(f'Applied manual review coordinate for tracker {uid}, frame {frame_index}.')

    @pyqtSlot(str, str)
    def _on_batch_error(self, uid: str, msg: str) -> None:
        self._status.showMessage(f'Error in tracker {uid}: {msg}')

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_frame_step(self, delta: int) -> None:
        if not self._reader:
            return
        new_idx = max(0, min(self._reader.frame_count - 1,
                              self._current_frame + delta))
        self._show_frame(new_idx)
        self._timeline.set_frame(new_idx)

    @pyqtSlot(int)
    def _on_timeline_seek(self, frame_index: int) -> None:
        if self._playback_running:
            self._set_playback_running(False)
        self._show_frame(frame_index)

    @pyqtSlot(bool)
    def _set_playback_running(self, playing: bool) -> None:
        if playing and (not self._reader or not self._buffer):
            self._timeline.set_playing(False)
            return
        self._playback_running = bool(playing)
        self._timeline.set_playing(self._playback_running)
        if self._playback_running:
            self._playback_next_due = perf_counter()
            self._buffer.set_playback_prebuffer_enabled(
                True,
                lookahead_frames=96,
                decoder_preference=self._decoder_preference,
            )
            self._buffer.start_playback_prebuffer(self._current_frame)
            self._playback_timer.start()
        else:
            self._playback_timer.stop()

    @pyqtSlot()
    def _on_playback_tick(self) -> None:
        if not self._playback_running or not self._reader:
            self._set_playback_running(False)
            return
        now = perf_counter()
        if now < self._playback_next_due:
            return
        frame_index = min(self._reader.frame_count - 1, self._current_frame + 1)
        frame_cpu = self._buffer.get_playback_frame(frame_index) if self._buffer else None
        if frame_cpu is None:
            # Do not synchronously decode from the UI thread on a miss. Keep
            # the current image visible and try again as soon as the producer
            # has had a chance to fill the next frame.
            self._playback_next_due = now + 0.01
            return
        if frame_index != self._current_frame:
            self._show_frame(frame_index, frame_cpu=frame_cpu)
            self._timeline.set_frame(frame_index)
        self._playback_next_due = perf_counter() + 1.0 / max(0.001, float(self._reader.fps))
        if frame_index >= self._reader.frame_count - 1:
            self._set_playback_running(False)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            focus = self.focusWidget()
            if not isinstance(focus, (QLineEdit, QDoubleSpinBox, QComboBox)):
                self._set_playback_running(not self._playback_running)
                event.accept()
                return
        super().keyPressEvent(event)

    def _show_frame(self, index: int, frame_cpu=None) -> None:
        if not self._buffer:
            return
        self._current_frame = index
        self._canvas.set_frame_index(index)
        if frame_cpu is None:
            frame_cpu = self._buffer.get_frame_cpu(index)
        if frame_cpu is not None:
            self._canvas.set_frame(frame_cpu)

        # Update canvas overlays from results
        if self._manager:
            for tracker in self._manager.all_trackers():
                result = self._manager.get_result_at_frame(tracker.uid, index)
                if result:
                    self._canvas.update_result(
                        tracker.uid, result,
                        tracker.config.tracker_type, tracker.name
                    )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._set_playback_running(False)
        self._cleanup_debug_artifacts()
        super().closeEvent(event)
