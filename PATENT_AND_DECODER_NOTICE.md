# Decoder and Patent Notice

Color Track supports video decoding through Windows system APIs, graphics
drivers installed by the user, and optional FFmpeg runtime DLLs. Hardware
paths such as NVDEC, AMD hardware decoding, and Intel Quick Sync use the
driver and operating-system components already present on the user's system;
Color Track does not distribute graphics drivers.

Color Track does not grant any license under third-party patents that may
apply to a video codec, including H.264/AVC, HEVC/H.265, or AV1. This notice
does not restrict the rights granted by the GNU GPLv3 for Color Track's own
copyrighted code.

The FFmpeg DLLs currently present in `native/bin` are distributed components,
not merely source-code references. They remain subject to their own license
and notice obligations as described in `THIRD_PARTY_NOTICES.md` and
`native/THIRD_PARTY_RUNTIME.md`.

This document records the architecture and licensing boundary. It is not a
transfer of legal responsibility and is not legal advice.
