# Color Track

GPU-accelerated multi-object video tracker with Hessian-based shape detection.
Built with Python, PyQt6, CuPy (CUDA), and OpenCV.

---

## Requirements

- NVIDIA GPU with CUDA 11.x or 12.x
- Python 3.10+
- CUDA Toolkit installed (matching your GPU driver)

---

## Licensing

Color Track's own source code is released under the [GNU GPLv3](LICENSE).
Third-party dependencies, GPU components, and native FFmpeg runtime DLLs have
their own terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md). A binary
installer needs an approved FFmpeg redistribution bundle before publication;
the release checklist is in [EXPORT_LICENSING_CHECKLIST.md](EXPORT_LICENSING_CHECKLIST.md).
Decoder architecture and third-party codec-patent boundaries are described in
[PATENT_AND_DECODER_NOTICE.md](PATENT_AND_DECODER_NOTICE.md).
The FFmpeg runtime provenance and source-release requirement are recorded in
[FFMPEG_PROVENANCE.md](FFMPEG_PROVENANCE.md).

---

## Installation

```bash
# 1. Clone / unzip the project
cd shape_tracker

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install CuPy for your CUDA version (REQUIRED — not in requirements.txt)
pip install cupy-cuda12x    # CUDA 12.x
# OR
pip install cupy-cuda11x    # CUDA 11.x

# 4. Run
python main.py
```

---

## Project Structure

```
shape_tracker/
├── main.py                    Entry point
├── gpu/
│   ├── fft_utils.py           FFT-based derivative helpers (shared)
│   ├── hessian.py             DoH blob detection + ridge detection
│   ├── color_mask.py          GPU HSV thresholding
│   └── channel.py             Channel extraction (Grey/R/G/B/Custom)
├── core/
│   ├── video_reader.py        FFmpeg-backed video reader
│   ├── frame_buffer.py        Async GPU prefetch ring buffer
│   ├── tracker_manager.py     Tracker factory + batch QThread worker
│   └── export.py              .npz export
├── tracking/
│   ├── base_tracker.py        Abstract base + data classes
│   ├── point_fast.py          CSRT + HOG + Hessian re-lock
│   └── trackers.py            PointAccurate, BlobSimple, BlobComplex,
│                               CurveTracker, ColorAreaTracker
└── ui/
    ├── main_window.py         Top-level window + wiring
    ├── video_canvas.py        QOpenGLWidget — video + overlays
    └── panels.py              TrackerPanel, TimelineWidget,
                                SettingsDialog, RecoveryWidget
```

---

## Controls

| Input         | Action                          |
|---------------|---------------------------------|
| Scroll wheel  | Zoom in / out (centred on cursor) |
| W / A / S / D | Pan up / left / down / right    |
| E             | Next frame                      |
| Q             | Previous frame                  |
| R             | Reset view                      |
| Left click    | Place seed point                |
| Left drag     | Draw ROI (blob / color-area)    |

---

## Tracker Types

### Point Fast
CSRT tracker (OpenCV) with HOG appearance model.
Hessian DoH score validates each prediction; drops to UNCERTAIN if score
falls below `relock_threshold × init_score`.

### Point Accurate
Hessian blob detection every frame → sub-pixel centre of mass.
More expensive than Point Fast but does not drift.

### Blob Simple
Hessian blob → convex hull polygon stored per frame.

### Blob Complex
Hessian blob → full binary mask (uint8) stored per frame.
Large exports (~1 GB for 4K, 1000 frames) — warned before export.

### Curve
Hessian eigenvalue ridge detection (λ₁ >> λ₂) every frame.
Output: B-spline fitted to ridge skeleton, stored as K control points.
Good for ropes, cables, thin edges.

### Color Area
GPU HSV thresholding.  Click a pixel → adjust hue/sat/val tolerance sliders
with live preview → binary mask per frame.

---

## FFT-Based Hessian (Performance)

The Hessian second derivatives are computed entirely in frequency space:

```
H_xx = IFFT( -u²  · FFT(Gaussian(σ) · I) )
H_yy = IFFT( -v²  · FFT(Gaussian(σ) · I) )
H_xy = IFFT( -u·v · FFT(Gaussian(σ) · I) )
```

The FFT of the pre-smoothed image is computed **once per frame** and cached.
All three derivatives reuse it — cost = 1 forward FFT + 3 multiplications
+ 3 inverse FFTs, all on GPU via CuPy / cuFFT.

---

## Channel Extraction

All Hessian-based trackers (everything except Color Area) apply a channel
extraction step before detection:

- **Greyscale** (BT.601): `0.299·R + 0.587·G + 0.114·B`
- **Red / Green / Blue**: single channel
- **Custom**: user-supplied weights, auto-normalised to sum = 1

Set the global default via **Edit → Settings**.
Override per-tracker in the tracker panel row.

---

## Lock-Loss Recovery

When a tracker's Hessian score drops below the threshold:
1. Status changes to **UNCERTAIN** (yellow in panel)
2. Last known position is kept
3. A **Re-seed** button appears — click it, then click the object on the canvas
4. Batch re-runs from that frame to the original end frame
5. If auto re-lock still fails → repeat as needed

---

## Export Format (.npz)

```python
import numpy as np
data = np.load('results.npz', allow_pickle=True)

# Metadata
import json
meta = json.loads(str(data['meta']))

# Point tracker
uid = 'abc12345'
frames  = data[f'{uid}_frames']    # int32 (F,)
status  = data[f'{uid}_status']    # str   (F,)
centers = data[f'{uid}_center']    # float32 (F, 2)  [row, col]

# Blob simple
polygon = data[f'{uid}_polygon']   # float32 (F, P, 2)

# Blob complex / Color area
masks = data[f'{uid}_mask']        # uint8 (F, H, W)

# Curve
spline = data[f'{uid}_spline']     # float32 (F, K, 2)
```
