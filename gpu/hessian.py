"""
gpu/hessian.py
--------------
Hessian-based feature detection on the GPU.

Two detection modes are exposed:

1. **Blob / Point Accurate**
   Uses the Determinant of Hessian (DoH) response:
       det(H) = Ixx·Iyy - Ixy²
   Optionally normalised by scale (sigma⁴) so responses are comparable
   across different user-supplied feature sizes.
   Local maxima of det(H) are blob centres; centre-of-mass weighting gives
   sub-pixel accuracy for Point Accurate mode.

2. **Curve / Ridge**
   Uses Hessian eigenvalues.  For a ridge (e.g. white rope on dark BG):
       λ₁ ≥ λ₂  (both eigenvalues of H at each pixel)
   Ridge condition: λ₁ is large and negative (bright ridge on dark) while
   λ₂ ≈ 0 (no curvature along the ridge).
   Ridgeness measure R = |λ₁| * (1 - |λ₂| / (|λ₁| + ε))
   which peaks along the ridge centreline and falls off laterally.

   Dark ridge on bright BG: set dark_ridge=True → uses λ₁ large positive.

All heavy work stays on GPU (CuPy).  Results are returned as CuPy arrays;
callers that need NumPy can call `.get()` themselves.

Dependencies
------------
    cupy, gpu.fft_utils
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cupy as cp
import numpy as np

from gpu.fft_utils import (
    fft2_image,
    gaussian_kernel_fft,
    smooth_fft,
    compute_second_derivatives,
)

# ---------------------------------------------------------------------------
# Data classes returned to callers
# ---------------------------------------------------------------------------

@dataclass
class BlobResult:
    """Output of detect_blobs()."""
    response_map: cp.ndarray        # DoH map, shape (H, W), float64
    centers: np.ndarray             # (N, 2) float32, (row, col) sub-pixel
    scores: np.ndarray              # (N,)   float32, DoH score at each centre


@dataclass
class RidgeResult:
    """Output of detect_ridges()."""
    ridgeness_map: cp.ndarray       # R map, shape (H, W), float64
    lambda1_map: cp.ndarray         # λ₁ map for visualisation
    lambda2_map: cp.ndarray         # λ₂ map for visualisation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hessian_maps(
    image_gpu: cp.ndarray,
    sigma: float,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, int, int]:
    """
    Core pipeline: image → (Ixx, Iyy, Ixy).

    Returns Ixx, Iyy, Ixy (all float64, GPU) and H, W.
    """
    H, W = image_gpu.shape
    F = fft2_image(image_gpu)
    if sigma > 0.0:
        G = gaussian_kernel_fft(H, W, sigma)
        F = smooth_fft(F, G)
    Ixx, Iyy, Ixy = compute_second_derivatives(F, H, W)
    return Ixx, Iyy, Ixy, H, W


def _non_maximum_suppression_2d(
    response: cp.ndarray,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
) -> cp.ndarray:
    """
    GPU non-maximum suppression via max-pooling comparison.

    Returns a boolean mask (H, W) marking local maxima.

    Parameters
    ----------
    response      : DoH map, positive values only (negatives already zeroed).
    min_distance  : Minimum pixel separation between peaks.
    threshold_rel : Peaks below this fraction of global max are rejected.
    """
    from cupyx.scipy.ndimage import maximum_filter
    r = min_distance
    footprint = cp.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    local_max = maximum_filter(response, footprint=footprint)
    is_max = (response == local_max)
    thresh = float(response.max()) * threshold_rel
    is_max &= (response > thresh)
    return is_max


def _subpixel_centres(
    response: cp.ndarray,
    mask: cp.ndarray,
    patch: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine pixel-level peak positions to sub-pixel via weighted centre-of-mass
    within a (2*patch+1) × (2*patch+1) neighbourhood.

    Returns (centres, scores) as NumPy arrays for easy downstream use.
    centres : (N, 2) float32  [row, col]
    scores  : (N,)   float32
    """
    rows, cols = cp.where(mask)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    H, W = response.shape
    rows_np = rows.get().astype(np.int32)
    cols_np = cols.get().astype(np.int32)
    resp_np = response.get()

    refined_rows = []
    refined_cols = []
    scores_out   = []

    for r, c in zip(rows_np, cols_np):
        r0, r1 = max(0, r - patch), min(H, r + patch + 1)
        c0, c1 = max(0, c - patch), min(W, c + patch + 1)
        patch_data = resp_np[r0:r1, c0:c1]
        total = patch_data.sum()
        if total == 0:
            refined_rows.append(float(r))
            refined_cols.append(float(c))
            scores_out.append(resp_np[r, c])
        else:
            pr = np.arange(r0, r1, dtype=np.float64)
            pc = np.arange(c0, c1, dtype=np.float64)
            pr_grid, pc_grid = np.meshgrid(pr, pc, indexing='ij')
            refined_rows.append(float((pr_grid * patch_data).sum() / total))
            refined_cols.append(float((pc_grid * patch_data).sum() / total))
            scores_out.append(float(resp_np[r, c]))

    centres = np.array(
        list(zip(refined_rows, refined_cols)), dtype=np.float32
    ).reshape(-1, 2)
    scores = np.array(scores_out, dtype=np.float32)
    return centres, scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_blobs(
    image_gpu: cp.ndarray,
    sigma: float,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
    normalise_by_scale: bool = True,
) -> BlobResult:
    """
    Detect blob-like structures using the Determinant of Hessian.

    Parameters
    ----------
    image_gpu        : Single-channel float image on GPU, shape (H, W).
    sigma            : Feature scale in pixels (user-supplied).
    min_distance     : Minimum pixel separation between detected blobs.
    threshold_rel    : Relative threshold (fraction of max response).
    normalise_by_scale: Multiply det(H) by sigma⁴ for scale-normalised response.

    Returns
    -------
    BlobResult with response_map, centers (sub-pixel), scores.
    """
    Ixx, Iyy, Ixy, H, W = _hessian_maps(image_gpu, sigma)

    det_H = Ixx * Iyy - Ixy ** 2
    if normalise_by_scale:
        det_H = det_H * (sigma ** 4)

    # Keep only positive responses (blobs have positive det(H))
    det_H_pos = cp.maximum(det_H, 0.0)

    if float(det_H_pos.max()) < 1e-12:
        return BlobResult(
            response_map=det_H_pos,
            centers=np.empty((0, 2), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )

    mask = _non_maximum_suppression_2d(det_H_pos, min_distance, threshold_rel)
    centers, scores = _subpixel_centres(det_H_pos, mask)

    return BlobResult(
        response_map=det_H_pos,
        centers=centers,
        scores=scores,
    )


def detect_ridges(
    image_gpu: cp.ndarray,
    sigma: float,
    dark_ridge: bool = False,
    threshold_rel: float = 0.05,
) -> RidgeResult:
    """
    Detect ridge-like structures (curves, ropes, edges) using Hessian
    eigenvalues.

    The eigenvalues of the 2×2 Hessian at each pixel are:
        λ₁ = (Ixx+Iyy)/2 + sqrt(((Ixx-Iyy)/2)² + Ixy²)
        λ₂ = (Ixx+Iyy)/2 - sqrt(((Ixx-Iyy)/2)² + Ixy²)

    Ridge condition (bright ridge, dark_ridge=False):
        λ₁ < 0  (concave in cross-section)
        |λ₂| << |λ₁|  (nearly flat along ridge)

    Ridgeness measure:
        R = |λ₁| * (1 - |λ₂| / (|λ₁| + |λ₂| + ε))
          ≈ |λ₁| when λ₂ ≈ 0  (pure ridge)
          ≈  0    when |λ₁| ≈ |λ₂|  (corner / blob, not a ridge)

    Parameters
    ----------
    image_gpu   : Single-channel float image on GPU, shape (H, W).
    sigma       : Feature scale in pixels.
    dark_ridge  : If True, detects dark ridges on bright background.
    threshold_rel: Zero-out ridgeness below this fraction of max.

    Returns
    -------
    RidgeResult with ridgeness_map, lambda1_map, lambda2_map.
    """
    Ixx, Iyy, Ixy, H, W = _hessian_maps(image_gpu, sigma)

    trace_half = (Ixx + Iyy) * 0.5
    disc       = cp.sqrt(((Ixx - Iyy) * 0.5) ** 2 + Ixy ** 2)

    lam1 = trace_half + disc   # larger eigenvalue
    lam2 = trace_half - disc   # smaller eigenvalue

    eps = 1e-10

    if dark_ridge:
        # Dark ridge: λ₁ > 0, λ₂ ≈ 0
        valid = lam1 > 0
        abs1  = cp.abs(lam1)
        abs2  = cp.abs(lam2)
    else:
        # Bright ridge: λ₁ < 0, λ₂ ≈ 0
        valid = lam1 < 0
        abs1  = cp.abs(lam1)
        abs2  = cp.abs(lam2)

    ridgeness = abs1 * (1.0 - abs2 / (abs1 + abs2 + eps))
    ridgeness = cp.where(valid, ridgeness, 0.0)

    # Threshold
    max_r = float(ridgeness.max())
    if max_r > eps:
        ridgeness = cp.where(ridgeness >= threshold_rel * max_r, ridgeness, 0.0)

    return RidgeResult(
        ridgeness_map=ridgeness,
        lambda1_map=lam1,
        lambda2_map=lam2,
    )


def hessian_score_at_bbox(
    image_gpu: cp.ndarray,
    sigma: float,
    bbox: Tuple[int, int, int, int],
) -> float:
    """
    Compute the mean DoH response inside a bounding box.

    Used by the Point Fast tracker to validate CSRT predictions: if the
    score is below a caller-supplied threshold the tracker is marked UNCERTAIN.

    Parameters
    ----------
    image_gpu : Single-channel float image on GPU.
    sigma     : Feature scale for Hessian computation.
    bbox      : (x, y, w, h) in pixel coordinates (OpenCV convention).

    Returns
    -------
    score : float   Mean scale-normalised DoH inside the bbox.
    """
    x, y, w, h = bbox
    H_img, W_img = image_gpu.shape
    x  = max(0, x);  y  = max(0, y)
    x2 = min(W_img, x + w);  y2 = min(H_img, y + h)
    if x2 <= x or y2 <= y:
        return 0.0

    roi = image_gpu[y:y2, x:x2]
    result = detect_blobs(roi, sigma, min_distance=1, threshold_rel=0.0,
                          normalise_by_scale=True)
    if result.response_map.size == 0:
        return 0.0
    return float(result.response_map.mean())


def blob_polygon(
    image_gpu: cp.ndarray,
    sigma: float,
    center: Tuple[float, float],
    search_radius: int = 50,
    threshold_rel: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Given a rough centre (row, col), extract the convex hull polygon of the
    nearest blob in a search window.

    Returns
    -------
    polygon : np.ndarray  shape (M, 2) float32 [row, col], or None if not found.
    """
    import cv2  # local import — only needed here

    H_img, W_img = image_gpu.shape
    cr, cc = int(center[0]), int(center[1])
    r0 = max(0, cr - search_radius);  r1 = min(H_img, cr + search_radius)
    c0 = max(0, cc - search_radius);  c1 = min(W_img, cc + search_radius)
    roi_gpu = image_gpu[r0:r1, c0:c1]

    result = detect_blobs(roi_gpu, sigma, min_distance=3,
                          threshold_rel=threshold_rel, normalise_by_scale=True)
    resp = result.response_map

    # Threshold the response map to create a binary blob mask
    max_val = float(resp.max())
    if max_val < 1e-10:
        return None

    binary = (resp >= threshold_rel * max_val).astype(cp.uint8)
    binary_np = binary.get()

    # Find contours on CPU (OpenCV)
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the contour closest to the click centre (relative to ROI)
    rel_cr, rel_cc = cr - r0, cc - c0
    best = None
    best_dist = float('inf')
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        d  = (cx - rel_cc) ** 2 + (cy - rel_cr) ** 2
        if d < best_dist:
            best_dist = d
            best      = cnt

    if best is None:
        return None

    hull = cv2.convexHull(best).squeeze()  # (M, 2) in (col, row)
    if hull.ndim < 2:
        return None

    # Convert back to full-image coordinates in (row, col) order
    polygon = hull[:, ::-1].astype(np.float32)
    polygon[:, 0] += r0
    polygon[:, 1] += c0
    return polygon


def blob_mask(
    image_gpu: cp.ndarray,
    sigma: float,
    center: Tuple[float, float],
    search_radius: int = 50,
    threshold_rel: float = 0.3,
) -> Optional[cp.ndarray]:
    """
    Like blob_polygon but returns a full-image binary mask (uint8 GPU array)
    instead of a polygon.  Used by Blob Complex mode.

    Returns None if no blob found.
    """
    H_img, W_img = image_gpu.shape
    cr, cc = int(center[0]), int(center[1])
    r0 = max(0, cr - search_radius);  r1 = min(H_img, cr + search_radius)
    c0 = max(0, cc - search_radius);  c1 = min(W_img, cc + search_radius)
    roi_gpu = image_gpu[r0:r1, c0:c1]

    result = detect_blobs(roi_gpu, sigma, min_distance=3,
                          threshold_rel=threshold_rel, normalise_by_scale=True)
    resp = result.response_map
    max_val = float(resp.max())
    if max_val < 1e-10:
        return None

    roi_mask = (resp >= threshold_rel * max_val).astype(cp.uint8)

    # Embed ROI mask back into full-image mask
    full_mask = cp.zeros((H_img, W_img), dtype=cp.uint8)
    full_mask[r0:r1, c0:c1] = roi_mask
    return full_mask
