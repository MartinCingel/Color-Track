# Third-Party Notices

Color Track source code is licensed under GNU GPLv3 in `LICENSE`.
Dependencies and distributed binaries remain subject to their own licenses.
This notice is an inventory and release guide, not a replacement for the full
license texts that must accompany a packaged release.

## Python Runtime and Packages

| Component | Use | License | Project |
| --- | --- | --- | --- |
| Python | Application runtime | PSF License | https://www.python.org/psf/license/ |
| PyQt6 | GUI bindings | GPLv3 or commercial | https://www.riverbankcomputing.com/software/pyqt/ |
| Qt 6 (`PyQt6-Qt6`) | GUI runtime | LGPLv3 | https://www.qt.io/licensing/open-source-lgpl-obligations |
| PyQt6-sip | PyQt binding support | BSD-2-Clause | https://www.riverbankcomputing.com/software/sip/ |
| OpenCV / opencv-contrib-python | Image processing, MOSSE | Apache-2.0 | https://opencv.org/license/ |
| NumPy | Array operations | BSD-3-Clause | https://numpy.org/doc/stable/license.html |
| SciPy | Scientific routines | BSD-3-Clause | https://scipy.org/ |
| openpyxl | XLSX export | MIT | https://openpyxl.readthedocs.io/ |
| CuPy (optional) | NVIDIA CUDA acceleration | MIT | https://github.com/cupy/cupy/blob/main/LICENSE |
| PyNvVideoCodec (optional) | NVIDIA video decoding | MIT | https://docs.nvidia.com/video-technologies/pynvvideocodec/ |
| MinGW-w64 winpthreads (release FFmpeg runtime) | FFmpeg thread runtime | MIT/BSD | https://www.mingw-w64.org/ |

NVIDIA GPU drivers, CUDA runtime/toolkit components, and Video Codec SDK
components are not licensed by this project. Package only the components that
their NVIDIA terms permit, and retain their accompanying notices.

## Native Video Decoder

`native/bin/NativeD3DDecoder.dll` uses Windows D3D11 and Media Foundation
system APIs and dynamically links to FFmpeg runtime DLLs when they are staged:

- `avcodec-62.dll`
- `avformat-62.dll`
- `avutil-60.dll`
- `swresample-6.dll`

The currently staged FFmpeg runtime is a minimal LGPL-2.1-or-later shared
build created by `native/build_release_ffmpeg.ps1` from the exact upstream
commit recorded in `FFMPEG_PROVENANCE.md`. Include its license files, build
configuration, script, and matching source link with every binary release.

FFmpeg licensing information: https://ffmpeg.org/legal.html

## Packaging Requirements

Before publishing a binary installer, include the exact license and notice
files from every wheel/DLL actually bundled by the installer. In particular,
NumPy, SciPy, OpenCV, and Qt wheels may contain further bundled binary
components with additional notices.

### PyQt6 GPLv3 Requirement

The PyQt6 package installed for this project includes GPLv3 licensing. Color
Track is therefore distributed under GPLv3, which is compatible with this
PyQt6 use. A different distribution model would require one of these routes:

1. Distribute the combined application under GPLv3 and satisfy its source and
   notice obligations.
2. Obtain and use a commercial PyQt6 license.
3. Replace PyQt6 with an alternative whose licensing fits the intended release.

Do not claim that an installer is "MIT licensed." Any packaged release must
still complete the FFmpeg redistribution decision and include the exact
third-party notices for its bundled files.
