"""Small debug replay dialog for tracker failure reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class DebugReplayDialog(QDialog):
    """Display overlay/likelihood frames and per-frame diagnostics."""

    def __init__(self, report_dir: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report_dir = Path(report_dir)
        self._rows: list[dict[str, Any]] = []
        self.setWindowTitle(f"Tracker Debug Replay — {self._report_dir.name}")
        self.resize(1250, 740)
        self._load()
        self._build_ui()
        if self._rows:
            self._list.setCurrentRow(len(self._rows) - 1)

    def _load(self) -> None:
        csv_path = self._report_dir / "debug.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                self._rows = list(csv.DictReader(fh))

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        path_label = QLabel(str(self._report_dir))
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(path_label)

        self._list = QListWidget()
        for row in self._rows:
            frame = row.get("frame_index", "?")
            status = row.get("status", "")
            reasons = row.get("failure_reasons", "")
            self._list.addItem(f"{frame}  {status}  {reasons}")
        self._list.currentRowChanged.connect(self._show_row)
        left.addWidget(self._list, stretch=1)

        open_folder = QPushButton("Open Report Folder")
        open_folder.clicked.connect(self._open_folder)
        left.addWidget(open_folder)
        root.addLayout(left, stretch=0)

        right = QVBoxLayout()

        image_row = QHBoxLayout()
        overlay_col = QVBoxLayout()
        overlay_col.addWidget(QLabel("Overlay"))
        self._overlay_image = QLabel("No overlay image")
        self._overlay_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_image.setMinimumSize(420, 300)
        self._overlay_image.setStyleSheet("background:#111; color:#aaa;")
        overlay_col.addWidget(self._overlay_image, stretch=1)
        image_row.addLayout(overlay_col, stretch=1)

        likelihood_col = QVBoxLayout()
        likelihood_col.addWidget(QLabel("Colour likelihood map"))
        self._likelihood_image = QLabel("No likelihood map")
        self._likelihood_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._likelihood_image.setMinimumSize(420, 300)
        self._likelihood_image.setStyleSheet("background:#111; color:#aaa;")
        likelihood_col.addWidget(self._likelihood_image, stretch=1)
        image_row.addLayout(likelihood_col, stretch=1)

        right.addLayout(image_row, stretch=3)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Field", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self._table, stretch=2)
        root.addLayout(right, stretch=1)

    def _set_scaled_pixmap(self, label: QLabel, path: Path, missing_text: str) -> None:
        if path.exists():
            pix = QPixmap(str(path))
            label.setPixmap(
                pix.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            label.setPixmap(QPixmap())
            label.setText(missing_text)

    def _show_row(self, index: int) -> None:
        if index < 0 or index >= len(self._rows):
            return
        row = self._rows[index]
        frame = row.get("frame_index", "")
        if str(frame).isdigit():
            frame_name = f"frame_{int(frame):06d}.png"
            overlay_path = self._report_dir / "overlay_frames" / frame_name
            likelihood_path = self._report_dir / "likelihood_maps" / frame_name
            self._set_scaled_pixmap(self._overlay_image, overlay_path, "No overlay image")
            self._set_scaled_pixmap(self._likelihood_image, likelihood_path, "No likelihood map")
        else:
            self._overlay_image.setText("No overlay image")
            self._likelihood_image.setText("No likelihood map")

        interesting = [
            "frame_index", "status", "failure_reasons", "confidence",
            "peak_score", "contrast_score", "innovation_distance",
            "kalman_gate_radius", "predicted_rc", "measured_rc",
            "peak_rc", "uncertain_export_rc", "uncertain_export_source",
            "search_bbox_xywh", "centroid_window_xywh",
            "candidate_count", "rejection_counts", "extra",
        ]
        self._table.setRowCount(len(interesting))
        for r, key in enumerate(interesting):
            self._table.setItem(r, 0, QTableWidgetItem(key))
            self._table.setItem(r, 1, QTableWidgetItem(str(row.get(key, ""))))
        self._table.resizeRowsToContents()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "_list"):
            index = self._list.currentRow()
            if index >= 0:
                self._show_row(index)

    def _open_folder(self) -> None:
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._report_dir)))
