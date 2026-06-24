"""Small FFmpeg D3D11VA bridge for CPU-facing workspace frames.

The tracker currently consumes BGR NumPy arrays.  FFmpeg therefore keeps the
decode surface on D3D11VA as long as possible, then downloads/converts only the
selected workspace to BGR for the existing CPU tracking pipeline.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    # WinGet updates PATH only for processes started after installation.  Locate
    # its standard package directory so the running desktop app can use FFmpeg
    # immediately as well.
    packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(packages.glob("Gyan.FFmpeg.Shared_*/*/bin/ffmpeg.exe"))
    return str(matches[-1]) if matches else None


def ffmpeg_available() -> bool:
    return ffmpeg_executable() is not None


def d3d11va_availability_message() -> str:
    if not ffmpeg_available():
        return "FFmpeg executable was not found on PATH."
    return "FFmpeg D3D11VA will be tested when decoding starts."


def iter_d3d11va_bgr_frames(
    video_path: str | Path,
    *,
    width: int,
    height: int,
    start: int,
    end: int,
    crop_xywh: tuple[int, int, int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield frame-accurate sequential BGR frames from FFmpeg D3D11VA.

    Decoding begins at frame zero intentionally.  Seeking with ``-ss`` is not
    reliably frame accurate for all H.264/H.265 GOP layouts, whereas batch and
    playback prebuffer correctness matter more than a small startup cost.
    """
    executable = ffmpeg_executable()
    if executable is None:
        raise RuntimeError(d3d11va_availability_message())
    if end <= start:
        return

    output_width, output_height = width, height
    trim_x = trim_y = 0
    requested_width, requested_height = width, height
    filter_parts = ["hwdownload", "format=nv12"]
    if crop_xywh is not None:
        x, y, requested_width, requested_height = (int(value) for value in crop_xywh)
        if x < 0 or y < 0 or requested_width <= 0 or requested_height <= 0:
            raise ValueError("D3D11VA crop must have non-negative origin and positive dimensions.")
        if x + requested_width > width or y + requested_height > height:
            raise ValueError("D3D11VA crop exceeds the decoded frame bounds.")
        # FFmpeg's D3D11VA crop filter currently reports success while yielding
        # full-sized surfaces. Crop after hwdownload instead: this still copies
        # NV12 once, but avoids full-frame BGR conversion, pipe traffic, and
        # NumPy allocation/copy for a small tracking workspace.
        # NV12/YUV420 chroma samples are 2x2. Cropping at an odd position before
        # conversion changes chroma phase and produces different colours than a
        # full-frame conversion. Expand to even boundaries, then trim in BGR.
        aligned_x = x & ~1
        aligned_y = y & ~1
        aligned_right = min(width, (x + requested_width + 1) & ~1)
        aligned_bottom = min(height, (y + requested_height + 1) & ~1)
        output_width = aligned_right - aligned_x
        output_height = aligned_bottom - aligned_y
        trim_x = x - aligned_x
        trim_y = y - aligned_y
        filter_parts.append(f"crop={output_width}:{output_height}:{aligned_x}:{aligned_y}:exact=1")
    filter_parts.append("format=bgr24")
    command = [
        executable, "-hide_banner", "-loglevel", "error",
        "-hwaccel", "d3d11va",
        "-hwaccel_output_format", "d3d11",
        "-i", str(video_path),
        "-an", "-sn", "-dn",
        "-vf", ",".join(filter_parts),
        "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    frame_bytes = int(output_width * output_height * 3)
    try:
        for frame_index in range(end):
            assert process.stdout is not None
            payload = process.stdout.read(frame_bytes)
            if len(payload) != frame_bytes:
                break
            if frame_index >= start:
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(output_height, output_width, 3)
                if crop_xywh is not None:
                    frame = frame[
                        trim_y:trim_y + requested_height,
                        trim_x:trim_x + requested_width,
                    ]
                frame = np.ascontiguousarray(frame)
                yield frame_index, frame
        return_code = process.wait(timeout=10)
        if return_code != 0:
            assert process.stderr is not None
            message = process.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(message or f"FFmpeg exited with status {return_code}.")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
