"""
gpu/color_mask.py
-----------------
GPU-accelerated HSV colour thresholding for the Color-Area tracker.

Workflow
--------
1. BGR frame (uint8, GPU) is converted to HSV float32 on the GPU via a
   custom CuPy ElementwiseKernel (avoids CPU round-trip).
2. A tolerance band [center ± delta] is applied per channel independently.
3. Hue wrapping is handled correctly (hue is circular, 0–360°).
4. The resulting binary mask (uint8) stays on GPU for downstream use.

The live-preview path (`threshold_preview`) returns a coloured RGBA overlay
(R=255, G=0, B=255, A=128) that PyQt's QImage can display semi-transparently
over the video canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
    import cupy as cp
except Exception:  # Keep shared ColorTolerance importable without CUDA.
    cp = None
import numpy as np

# ---------------------------------------------------------------------------
# CuPy kernel: BGR uint8 → HSV float32 (H∈[0,360), S∈[0,1], V∈[0,1])
# ---------------------------------------------------------------------------
_BGR_TO_HSV_KERNEL = cp.ElementwiseKernel(
    # inputs: flat R, G, B (already split by caller)
    'float32 b, float32 g, float32 r',
    'float32 h, float32 s, float32 v',
    r'''
    float maxc = max(r, max(g, b));
    float minc = min(r, min(g, b));
    float delta = maxc - minc;

    v = maxc;
    s = (maxc > 0.0f) ? (delta / maxc) : 0.0f;

    if (delta < 1e-6f) {
        h = 0.0f;
    } else if (maxc == r) {
        h = 60.0f * fmodf((g - b) / delta, 6.0f);
    } else if (maxc == g) {
        h = 60.0f * ((b - r) / delta + 2.0f);
    } else {
        h = 60.0f * ((r - g) / delta + 4.0f);
    }
    if (h < 0.0f) h += 360.0f;
    ''',
    name='bgr_to_hsv'
) if cp is not None else None


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class ColorTolerance:
    """
    Defines the HSV acceptance band around a sampled pixel colour.

    All values are in the same units as OpenCV HSV (H: 0–360, S/V: 0–1).
    """
    center_h: float      # Sampled hue        [0, 360)
    center_s: float      # Sampled saturation  [0, 1]
    center_v: float      # Sampled value       [0, 1]
    delta_h:  float      # Hue tolerance       [0, 180]
    delta_s:  float      # Saturation tolerance [0, 1]
    delta_v:  float      # Value tolerance      [0, 1]

    @staticmethod
    def from_bgr_pixel(bgr: Tuple[int, int, int]) -> 'ColorTolerance':
        """
        Construct a ColorTolerance by sampling a single BGR pixel.
        Default tolerances are moderate starting points for the UI sliders.
        """
        b, g, r = [x / 255.0 for x in bgr]
        # Quick Python HSV conversion for the seed point
        import colorsys
        h_norm, s, v = colorsys.rgb_to_hsv(r, g, b)
        return ColorTolerance(
            center_h=h_norm * 360.0,
            center_s=s,
            center_v=v,
            delta_h=15.0,
            delta_s=0.25,
            delta_v=0.25,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_to_hsv(frame_gpu: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Convert a (H, W, 3) uint8 BGR GPU frame to three float32 HSV channel maps.

    Returns (H_map, S_map, V_map) each shape (H, W), float32, on GPU.
    """
    if cp is None or _BGR_TO_HSV_KERNEL is None:
        raise RuntimeError('GPU HSV processing requires CuPy.')
    frame_f = frame_gpu.astype(cp.float32) / 255.0
    b = cp.ascontiguousarray(frame_f[:, :, 0])
    g = cp.ascontiguousarray(frame_f[:, :, 1])
    r = cp.ascontiguousarray(frame_f[:, :, 2])

    H_map = cp.empty_like(r)
    S_map = cp.empty_like(r)
    V_map = cp.empty_like(r)

    _BGR_TO_HSV_KERNEL(b, g, r, H_map, S_map, V_map)
    return H_map, S_map, V_map


def _hue_distance(H_map: cp.ndarray, center_h: float) -> cp.ndarray:
    """
    Circular hue distance on [0, 360), returned as float32 in [0, 180].
    """
    diff = cp.abs(H_map - center_h)
    return cp.minimum(diff, 360.0 - diff)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_mask(
    frame_gpu: cp.ndarray,
    tol: ColorTolerance,
) -> cp.ndarray:
    """
    Compute a binary mask for pixels matching the colour tolerance.

    Parameters
    ----------
    frame_gpu : cp.ndarray  shape (H, W, 3), uint8 BGR, on GPU.
    tol       : ColorTolerance  defining the acceptance band.

    Returns
    -------
    mask : cp.ndarray  shape (H, W), uint8 (0 or 255), on GPU.
    """
    H_map, S_map, V_map = _frame_to_hsv(frame_gpu)

    in_h = _hue_distance(H_map, tol.center_h) <= tol.delta_h
    in_s = cp.abs(S_map - tol.center_s) <= tol.delta_s
    in_v = cp.abs(V_map - tol.center_v) <= tol.delta_v

    mask = (in_h & in_s & in_v).astype(cp.uint8) * 255
    return mask


def threshold_preview(
    frame_gpu: cp.ndarray,
    tol: ColorTolerance,
) -> np.ndarray:
    """
    Generate a CPU-side RGBA overlay image for the live preview.

    Matched pixels → magenta semi-transparent (R=255, G=0, B=255, A=128).
    Unmatched pixels → fully transparent (A=0).

    Parameters
    ----------
    frame_gpu : cp.ndarray  shape (H, W, 3) uint8 BGR on GPU.
    tol       : ColorTolerance.

    Returns
    -------
    overlay : np.ndarray  shape (H, W, 4) uint8 RGBA on CPU,
              ready to be wrapped in a QImage for Qt rendering.
    """
    mask_gpu = compute_mask(frame_gpu, tol)
    mask_np  = mask_gpu.get()          # (H, W), 0 or 255

    H, W = mask_np.shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    hit = mask_np == 255
    overlay[hit, 0] = 255   # R
    overlay[hit, 1] = 0     # G
    overlay[hit, 2] = 255   # B
    overlay[hit, 3] = 128   # A (semi-transparent)
    return overlay


def sample_pixel(
    frame_gpu: cp.ndarray,
    row: int,
    col: int,
) -> ColorTolerance:
    """
    Sample the BGR colour at (row, col) from a GPU frame and return a
    ColorTolerance with default delta values.

    Parameters
    ----------
    frame_gpu : cp.ndarray  shape (H, W, 3) uint8 BGR on GPU.
    row, col  : Pixel coordinates.

    Returns
    -------
    ColorTolerance  centred on the sampled colour.
    """
    pixel_gpu = frame_gpu[row, col]          # shape (3,), uint8
    pixel_np  = pixel_gpu.get()              # to CPU
    b, g, r   = int(pixel_np[0]), int(pixel_np[1]), int(pixel_np[2])
    return ColorTolerance.from_bgr_pixel((b, g, r))
