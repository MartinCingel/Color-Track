# Native FFmpeg Runtime

`NativeD3DDecoder.dll` dynamically links to these FFmpeg runtime DLLs:

- `avformat-62.dll`
- `avcodec-62.dll`
- `avutil-60.dll`
- `swresample-6.dll`

Use `build_release_ffmpeg.ps1` to build and stage the release runtime beside
the native DLL. It creates an explicitly LGPL-compatible shared FFmpeg build
from a pinned upstream source commit. Ship its license notices, build
configuration, script, and source link as described in
`../FFMPEG_PROVENANCE.md`.

`stage_ffmpeg_runtime.ps1` remains useful for temporary local experiments,
but an arbitrary local FFmpeg build is not a release redistribution decision.
