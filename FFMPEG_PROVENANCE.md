# FFmpeg Provenance

## Current Release Runtime

The FFmpeg runtime DLLs staged in `native/bin` are built locally by
`native/build_release_ffmpeg.ps1` from the exact upstream FFmpeg commit:

- Commit: `239f2c733de417201d7ad3b3b8b0d9b63285b2b1`
- Source: https://github.com/FFmpeg/FFmpeg/tree/239f2c733de417201d7ad3b3b8b0d9b63285b2b1
- Build configuration: `native/FFMPEG_BUILD_CONFIGURATION.txt`
- License: LGPL-2.1-or-later

The release build deliberately excludes GPL-only external libraries such as
`libx264` and `libx265`. Its only additional runtime dependency is the
MIT/BSD-licensed MinGW-w64 `libwinpthread-1.dll`, whose notice is in
`native/licenses/libwinpthread-COPYING.txt`.

## Historical Development Runtime

Earlier local testing used Gyan.FFmpeg.Shared `8.1.1-full_build-shared`, a GPL
build containing x264/x265 and many unrelated external libraries. Those DLLs
are no longer the runtime staged in `native/bin` and must not be included in a
Color Track release.

## Local Modifications

Color Track does not modify the FFmpeg source code or its DLL binaries.
`native/d3d11_workspace_decoder.cpp` is separate Color Track code that links
to FFmpeg's public shared-library interfaces. It must not be described as an
FFmpeg source modification.

## Release Requirement

Before sharing a Color Track build that contains the DLLs, include the two
files in `native/licenses`, this provenance record, the exact build
configuration, the release build script, and the source URL above. Recipients
must be able to obtain the exact corresponding source for the shipped FFmpeg
runtime.
