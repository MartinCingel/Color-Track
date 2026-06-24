"""Verify the portable native D3D11 runtime against a video file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.native_d3d11 import native_d3d11_dll_path, native_d3d11_probe, native_ffmpeg_d3d11_decode_probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path, help='H.264 or HEVC video to probe')
    args = parser.parse_args()
    print(f'Native DLL: {native_d3d11_dll_path()}')
    ok, message = native_d3d11_probe()
    print(f'D3D11 device: {ok} - {message}')
    if not ok:
        return 1
    ok, message = native_ffmpeg_d3d11_decode_probe(args.video)
    print(f'FFmpeg D3D11 decode: {ok} - {message}')
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
