# Minimal FFmpeg Release Runtime

Color Track's release FFmpeg runtime is built from the exact upstream commit:

- Source: https://github.com/FFmpeg/FFmpeg/tree/239f2c733de417201d7ad3b3b8b0d9b63285b2b1
- Commit: `239f2c733de417201d7ad3b3b8b0d9b63285b2b1`
- Build script: `native/build_release_ffmpeg.ps1`

The script enables only the container readers, decoders, and both D3D11VA
hardware-surface variants needed by Color Track. It deliberately does not
enable GPL-only external libraries such as x264 or x265. The resulting FFmpeg
libraries are LGPL-2.1-or-later unless the build configuration is changed.

The MinGW-w64 build also stages `libwinpthread-1.dll`. Its installed package
declares MIT/BSD licensing; ship its `COPYING` notice with any installer that
contains this DLL.

For each binary release, include this file, the FFmpeg LGPL license text, the
exact build configuration emitted by `ffmpeg -buildconf`, and this source URL.
