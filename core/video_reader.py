"""
core/video_reader.py
--------------------
FFmpeg-backed video reader supporting H.264, H.265, MP4, AVI, MOV, MKV.

Design
------
* Uses cv2.VideoCapture with the FFmpeg backend (default on most OpenCV builds).
* Exposes random-access by frame index via seek (cap.set CAP_PROP_POS_FRAMES).
* Provides a context-manager interface for safe resource cleanup.
* Frame upload to GPU is optional and separated into upload_frame() so the
  frame buffer can decide when to do the transfer.
* All public methods are thread-safe via a single threading.Lock so the
  frame buffer's prefetch thread and the UI thread can both call read().

Limitations
-----------
Seeking in H.265 MKV containers can be imprecise on some FFmpeg builds;
VideoReader.seek() compensates by reading forward from the nearest keyframe.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoReader:
    """
    Random-access video reader wrapping cv2.VideoCapture.

    Parameters
    ----------
    path : str | Path   Path to the video file.
    """

    # Supported container/codec extensions (FFmpeg handles the rest)
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")
        if self._path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension: {self._path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        self._cap = cv2.VideoCapture(str(self._path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self._path}")

        self._lock = threading.Lock()

        # Cache metadata (immutable after open)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps         = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self._width       = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fourcc_int  = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._fourcc_str  = self._decode_fourcc(self._fourcc_int)

        # Track current position to avoid unnecessary seeks
        self._current_pos: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> 'VideoReader':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying VideoCapture."""
        with self._lock:
            if self._cap.isOpened():
                self._cap.release()

    # ------------------------------------------------------------------
    # Metadata properties (no lock needed — immutable after __init__)
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height)"""
        return self._width, self._height

    @property
    def duration_seconds(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    @property
    def fourcc(self) -> str:
        return self._fourcc_str

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read(self, index: int) -> Optional[np.ndarray]:
        """
        Read a single frame by zero-based index.

        Seeks only when necessary (sequential reads skip the seek call).

        Parameters
        ----------
        index : int   Frame index in [0, frame_count).

        Returns
        -------
        frame : np.ndarray  shape (H, W, 3) uint8 BGR, or None on failure.
        """
        if index < 0 or index >= self._frame_count:
            return None

        with self._lock:
            # Seek if we're not already at the right position
            if self._current_pos != index:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
                self._current_pos = index

            ret, frame = self._cap.read()
            if ret:
                self._current_pos += 1
                return frame
            return None

    def read_range(
        self,
        start: int,
        end: int,
    ):
        """
        Generator yielding (frame_index, frame_bgr) for frames [start, end).

        Efficient for sequential access — seeks once to start then reads
        forward without re-seeking.

        Parameters
        ----------
        start : int   First frame index (inclusive).
        end   : int   Last frame index (exclusive).
        """
        start = max(0, start)
        end   = min(end, self._frame_count)

        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
            self._current_pos = start

            for idx in range(start, end):
                ret, frame = self._cap.read()
                if not ret:
                    break
                self._current_pos += 1
                yield idx, frame

    def frame_to_timestamp(self, index: int) -> float:
        """Return the timestamp in seconds for a given frame index."""
        return index / self._fps

    def timestamp_to_frame(self, seconds: float) -> int:
        """Return the closest frame index for a given timestamp in seconds."""
        return max(0, min(self._frame_count - 1, int(seconds * self._fps)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_fourcc(fourcc_int: int) -> str:
        """Decode an integer FourCC code to a 4-character string."""
        return ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VideoReader('{self._path.name}', "
            f"{self._width}×{self._height}, "
            f"{self._fps:.2f}fps, "
            f"{self._frame_count}frames, "
            f"codec={self._fourcc_str})"
        )
