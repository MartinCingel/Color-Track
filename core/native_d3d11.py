"""Python bridge for the bundled Windows D3D11 workspace decoder."""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np


_DLL_NAME = "NativeD3DDecoder.dll"
_DLL_DIRECTORY_HANDLES: list[object] = []


def native_d3d11_dll_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "bin" / _DLL_NAME


def _configure_runtime_dll_directories() -> None:
    if os.name != "nt" or _DLL_DIRECTORY_HANDLES:
        return
    directories = [native_d3d11_dll_path().parent]
    configured = os.environ.get("FFMPEG_SHARED_ROOT")
    if configured:
        directories.append(Path(configured) / "bin")
    # Local development convenience after installing Gyan's shared FFmpeg build
    # with WinGet. Export builds copy their selected LGPL runtime DLLs beside the
    # native decoder and do not rely on this directory.
    packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    directories.extend(packages.glob("Gyan.FFmpeg.Shared_*/*/bin"))
    for directory in directories:
        if directory.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))


@lru_cache(maxsize=1)
def _load_library():
    if os.name != "nt":
        raise RuntimeError("Native D3D11 decoding is available only on Windows.")
    path = native_d3d11_dll_path()
    if not path.exists():
        raise RuntimeError(f"Native D3D11 decoder is not built: {path}")
    _configure_runtime_dll_directories()
    library = ctypes.WinDLL(str(path))
    library.ct_d3d11_abi_version.restype = ctypes.c_uint
    library.ct_d3d11_probe.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
    library.ct_d3d11_probe.restype = ctypes.c_int
    library.ct_d3d11_decode_probe.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
    library.ct_d3d11_decode_probe.restype = ctypes.c_int
    library.ct_ffmpeg_d3d11_decode_probe.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
    library.ct_ffmpeg_d3d11_decode_probe.restype = ctypes.c_int
    library.ct_ffmpeg_d3d11_decoder_open.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char), ctypes.c_uint,
    ]
    library.ct_ffmpeg_d3d11_decoder_open.restype = ctypes.c_int
    library.ct_ffmpeg_d3d11_decoder_next.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char), ctypes.c_uint,
    ]
    library.ct_ffmpeg_d3d11_decoder_next.restype = ctypes.c_int
    library.ct_ffmpeg_d3d11_decoder_close.argtypes = [ctypes.c_void_p]
    library.ct_ffmpeg_d3d11_decoder_close.restype = None
    if library.ct_d3d11_abi_version() != 1:
        raise RuntimeError("Native D3D11 decoder ABI version is incompatible.")
    return library


def native_d3d11_probe() -> tuple[bool, str]:
    """Return whether the bundled D3D11/Media Foundation runtime is usable."""
    try:
        library = _load_library()
        message = ctypes.create_string_buffer(512)
        status = library.ct_d3d11_probe(message, len(message))
        text = message.value.decode("utf-8", "replace") or "No diagnostic message returned."
        return status == 0, text
    except Exception as exc:
        return False, str(exc)


def native_d3d11_decode_probe(video_path: str | Path) -> tuple[bool, str]:
    """Verify that Media Foundation exposes a hardware D3D11 decode texture."""
    try:
        library = _load_library()
        message = ctypes.create_string_buffer(512)
        status = library.ct_d3d11_decode_probe(str(Path(video_path)), message, len(message))
        text = message.value.decode("utf-8", "replace") or "No diagnostic message returned."
        return status == 0, text
    except Exception as exc:
        return False, str(exc)


def native_ffmpeg_d3d11_decode_probe(video_path: str | Path) -> tuple[bool, str]:
    """Verify FFmpeg exposes a D3D11VA hardware texture for this codec."""
    try:
        library = _load_library()
        message = ctypes.create_string_buffer(512)
        status = library.ct_ffmpeg_d3d11_decode_probe(str(Path(video_path)), message, len(message))
        text = message.value.decode("utf-8", "replace") or "No diagnostic message returned."
        return status == 0, text
    except Exception as exc:
        return False, str(exc)


class NativeD3D11WorkspaceDecoder:
    """Sequential FFmpeg D3D11VA decoder returning GPU-cropped BGR workspaces."""

    def __init__(self, video_path: str | Path, *, x: int, y: int, width: int, height: int) -> None:
        self._library = _load_library()
        self._handle = ctypes.c_void_p()
        self._shape = (int(height), int(width), 3)
        self._frame_bytes = int(np.prod(self._shape))
        self.last_decode_ms = 0.0
        message = ctypes.create_string_buffer(512)
        result = self._library.ct_ffmpeg_d3d11_decoder_open(
            str(Path(video_path)), int(x), int(y), int(width), int(height),
            ctypes.byref(self._handle), message, len(message),
        )
        if result < 0 or not self._handle.value:
            text = message.value.decode("utf-8", "replace") or f"Native decoder failed ({result})."
            raise RuntimeError(text)
        self.backend_message = message.value.decode("utf-8", "replace")

    def read(self) -> tuple[int, np.ndarray] | None:
        if not self._handle.value:
            return None
        frame = np.empty(self._shape, dtype=np.uint8)
        frame_index = ctypes.c_int(-1)
        message = ctypes.create_string_buffer(512)
        started = perf_counter()
        result = self._library.ct_ffmpeg_d3d11_decoder_next(
            self._handle,
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), self._frame_bytes,
            ctypes.byref(frame_index), message, len(message),
        )
        self.last_decode_ms = (perf_counter() - started) * 1000.0
        if result == 1:
            return None
        if result < 0:
            text = message.value.decode("utf-8", "replace") or f"Native decoder failed ({result})."
            raise RuntimeError(text)
        return int(frame_index.value), frame

    def close(self) -> None:
        if self._handle.value:
            self._library.ct_ffmpeg_d3d11_decoder_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "NativeD3D11WorkspaceDecoder":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def iter_native_d3d11_workspace_frames(
    video_path: str | Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    start: int,
    end: int,
):
    """Yield sequential, GPU-cropped BGR workspace frames from frame zero."""
    with NativeD3D11WorkspaceDecoder(video_path, x=x, y=y, width=width, height=height) as decoder:
        while True:
            item = decoder.read()
            if item is None:
                return
            frame_index, frame = item
            if frame_index >= end:
                return
            if frame_index >= start:
                yield frame_index, frame, decoder.backend_message, decoder.last_decode_ms
