# Analysis Workspace Sources / NVDEC

## What this build adds

The application now supports two workspace frame sources:

```text
Live RAM buffer      default, no SSD cache writes
Reusable disk cache  optional, for repeated reruns
```

Both modes use the same preferred decode path when available:

```text
NVIDIA NVDEC GPU frame
  -> GPU crop and RGB-to-BGR conversion
  -> asynchronous transfer into rotating pinned RAM buffers
  -> CPU BGR WorkspaceFrame for PointFastTracker
```

When `PyNvVideoCodec` is not installed or cannot decode the particular video,
the system visibly falls back to sequential OpenCV/FFmpeg decoding.

## Install optional NVDEC backend

```bat
pip install PyNvVideoCodec
```

Keep the existing CuPy installation that matches your CUDA environment.

## Recommended workflow: Live RAM buffer

1. Open a video.
2. In **Analysis Workspace**, choose:
   - **Select** and drag a workspace containing all expected marker movement; or
   - **Full frame** when the source video is already cropped.
3. Keep **Frame source** set to **Live RAM buffer**.
4. Initialize trackers.
5. Run the batch.

In this mode, the decoder runs ahead into a bounded RAM queue and frames are
discarded after tracking. No `workspace_frames.dat` file is written, so it
avoids unnecessary SSD wear for one-off runs.

## Optional workflow: Reusable disk cache

Use **Reusable disk cache** only when you expect to rerun the same video and
workspace multiple times.

1. Select the workspace or choose full frame.
2. Change **Frame source** to **Reusable disk cache**.
3. Press **Prepare Disk Cache**.
4. Initialize trackers while preparation runs.
5. Run the batch after the cache is ready.

Prepared disk caches are stored in:

```text
<project folder>\analysis_cache\
```

A completed cache is reused automatically when the same video file and identical
workspace are prepared again.

## Important measurement constraint

The selected workspace must contain the complete search area throughout the
experiment. If a point marker approaches/leaves the workspace, the point tracker
will reject the measurement rather than reporting a coordinate outside cached
image data. Enlarge the workspace and rerun in that case.

## Memory use

Live RAM mode and disk-cache reads use a bounded look-ahead queue. The default
budget is 1 GiB, defined in `core/workspace_cache.py` as
`DEFAULT_RAM_LOOKAHEAD_BYTES`.

## Decoder-consistent point initialization

For direct colour point tracking, initialization should use the same pixel path
as batch tracking. When a workspace is active, the UI decodes the current
initialization frame through the same preferred path:

```text
NVDEC / PyNvVideoCodec if available
  ↓
GPU crop + RGB→BGR conversion
  ↓
WorkspaceFrame passed to PointFastTracker.initialize()
```

If that one-frame NVDEC decode fails, the app falls back to OpenCV for the
initialization frame and reports the fallback reason in the status bar. This
prevents the mismatch where the marker colour model is learned from the normal
OpenCV display frame but tracking later uses NVDEC-prepared workspace frames.
