"""
main.py
-------
Application entry point for Color Track.

Usage
-----
    python main.py

Requirements
------------
See requirements.txt for the full dependency list.
CUDA must be available; the application will raise on import of CuPy
if no compatible GPU is found.
"""

import sys
from pathlib import Path


def report_gpu_capability() -> None:
    """Report optional CUDA support without blocking CPU Point Fast startup."""
    from gpu.runtime import cupy_runtime_available
    if cupy_runtime_available():
        print('Optional CuPy GPU acceleration is available.')
    else:
        print('Starting in CPU/D3D11 mode. CUDA-only legacy tracker modes are unavailable.')


def main() -> None:
    report_gpu_capability()

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    app = QApplication(sys.argv)
    # High-DPI is always enabled in PyQt6 — no setAttribute needed
    app.setApplicationName('Color Track')
    app.setOrganizationName('ColorTrack')

    # Dark palette
    app.setStyle('Fusion')
    from PyQt6.QtGui import QIcon, QPalette, QColor
    icon_path = Path(__file__).resolve().parent / 'assets' / 'ColorTrack.ico'
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base,            QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button,          QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
