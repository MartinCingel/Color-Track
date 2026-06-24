# Color Track

Color Track is a Windows desktop application for tracking coloured markers in
video. It is designed for practical motion-analysis workflows: place one or
more trackers, run a batch, review uncertain frames when needed, and export
the measurements.

## Install and Run

**The recommended way to use Color Track is the Windows installer.**

1. Open the project's [Releases](../../releases) page.
2. Download `ColorTrack-0.1.0-beta-Windows-x64-Setup.exe` from the latest
   release.
3. Run the installer, then launch **Color Track** from the Start menu.

The installer is per-user: it does not need administrator rights and stores
your settings, calibration, and optional diagnostics in your local AppData
folder. The first launch is ready to use; Python, CUDA, and a separate FFmpeg
installation are not required.

Color Track runs on Windows 10/11, 64-bit. CPU tracking works without a
dedicated GPU. When compatible drivers are present, the application can use
hardware-accelerated video decoding through NVIDIA, AMD, or Intel graphics.

> This is a beta release. Keep the original video and inspect uncertain or
> lost sections before relying on exported measurements.

## What It Does

- Tracks multiple coloured markers in Normal or Tiny Feature mode.
- Uses colour likelihood, local contrast, motion gating, and optional MOSSE
  assistance to reject poor candidates.
- Supports a tracker queue, per-tracker start/end frames, forward and optional
  backward tracking, and lost-tracker reinitialisation.
- Lets you inspect and correct uncertain frames without hiding the correction
  source in the export.
- Provides video, probability, overlay, and original-frame views.
- Supports GPU preview/playback and hardware decoding where available.
- Exports CSV, Excel, and NPZ data. Excel point-tracker sheets begin with
  `frame`, `time`, `x`, `y`, `status`, and `correction_source`.
- Includes pixel-to-distance calibration and a configurable coordinate origin
  for physical-unit exports.

## Basic Workflow

1. Open a video.
2. Choose Normal or Tiny Feature mode and set the approximate feature size.
3. Click the marker to initialise a tracker. Hold `Shift` and left-click to
   add another tracker quickly.
4. Use the Active list for trackers to include in the next batch; move
   prepared trackers to the Queue when they should not run yet.
5. Set the video range and optional tracker end frame, then run the batch.
6. Review uncertain frames or reinitialise a lost tracker when appropriate.
7. Export results from the Export section or File menu.

`Space` toggles playback. The timeline controls the displayed frame and batch
range.

## Decoder Choices

The decoder selector lets you choose the preferred path for the current video.
The application falls back safely when a requested GPU path is unavailable.

- **Native D3D11 / FFmpeg**: the cross-vendor Windows path for compatible
  NVIDIA, AMD, and Intel hardware.
- **NVIDIA NVDEC**: available on supported NVIDIA configurations.
- **CPU / OpenCV fallback**: available when hardware decoding cannot be used.

Hardware decoding can improve responsiveness and batch preparation, but it
does not change the measured tracker data.

## Export Columns

CSV and Excel exports preserve tracker status and manual-correction history.
For point trackers, the core measurement columns are:

| Column | Meaning |
| --- | --- |
| `frame` | Zero-based video frame index. |
| `time_s` | Time in seconds from the video frame rate. |
| `col_x_<unit>` | Horizontal coordinate. |
| `row_y_<unit>` | Vertical coordinate. |
| `status` | `locked`, `uncertain`, `lost`, or `pending`. |
| `correction_source` | Records an accepted correction, when present. |

Coordinates are in pixels unless calibration is configured. Physical exports
can use metres, centimetres, or millimetres.

## Source and Development

The source is included for people who want to inspect, modify, or build Color
Track themselves. The installer remains the supported route for ordinary use.

For a basic development run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The optional Windows installer build is described in
[installer/README.md](installer/README.md). Rebuilding the native decoder
runtime additionally requires the build scripts and provenance files in
`native/`.
Generated build output, downloaded FFmpeg source, and runtime DLLs are
intentionally excluded from version control.

## License and Notices

Color Track is licensed under the [GNU GPLv3](LICENSE). Copyright 2026 Martin
Cingel. Third-party notices and decoder information are available in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[PATENT_AND_DECODER_NOTICE.md](PATENT_AND_DECODER_NOTICE.md), and
[FFMPEG_PROVENANCE.md](FFMPEG_PROVENANCE.md).
