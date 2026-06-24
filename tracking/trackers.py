"""
tracking/point_accurate.py  — Hessian blob → sub-pixel centre of mass
tracking/blob_simple.py     — Hessian blob → convex polygon
tracking/blob_complex.py    — Hessian blob → binary mask
tracking/curve_tracker.py   — Hessian ridgeness → B-spline
tracking/color_area.py      — HSV threshold  → binary mask

All five are in this file to keep imports concise; each class is also
re-exported from its own stub module for the build-order clarity described
in the architecture plan.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cupy as cp
import numpy as np

from gpu.channel import extract_channel, resolve
from gpu.color_mask import ColorTolerance, compute_mask, threshold_preview
from gpu.hessian import (
    detect_blobs, detect_ridges,
    blob_polygon, blob_mask,
    hessian_score_at_bbox,
)
from tracking.base_tracker import (
    BaseTracker, FrameResult, InitPreview,
    TrackerConfig, TrackerStatus,
)


# ============================================================
#  Shared helper: normalise a GPU float map to [0,1] CPU array
# ============================================================

def _norm_heatmap(arr: cp.ndarray) -> np.ndarray:
    a = arr.get().astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


# ============================================================
#  Point Accurate
# ============================================================

class PointAccurateTracker(BaseTracker):
    """
    Hessian DoH blob detection → sub-pixel centre of mass.

    Each frame the DoH map is computed inside a search window centred on
    the last known position.  The highest-scoring detected blob centre
    becomes the new position.
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._last_center: Optional[np.ndarray] = None   # [row, col] float32
        self._channel = resolve(config.channel)

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int,int,int,int]] = None,
    ) -> InitPreview:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        result = detect_blobs(ch_gpu, self.config.sigma, min_distance=3,
                               threshold_rel=0.1)

        # Pick blob closest to seed
        center = self._closest_blob(result, seed_row, seed_col)
        if center is None:
            center = np.array([seed_row, seed_col], dtype=np.float32)

        self._last_center = center
        self._init_score  = float(result.response_map[
            int(center[0]), int(center[1])
        ]) if result.response_map.size > 0 else 1.0
        self._reset_to_locked()

        return InitPreview(
            heatmap=_norm_heatmap(result.response_map),
            center=center,
        )

    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        sr     = self.config.search_radius
        cr, cc = int(self._last_center[0]), int(self._last_center[1])
        H, W   = ch_gpu.shape

        r0, r1 = max(0, cr - sr), min(H, cr + sr)
        c0, c1 = max(0, cc - sr), min(W, cc + sr)
        roi_gpu = ch_gpu[r0:r1, c0:c1]

        result = detect_blobs(roi_gpu, self.config.sigma, min_distance=3,
                               threshold_rel=0.1)

        # Closest blob in ROI → full-image coords
        best = self._closest_blob(result,
                                   cr - r0, cc - c0)
        if best is not None:
            full_center = np.array([best[0] + r0, best[1] + c0], dtype=np.float32)
            score = float(result.response_map[int(best[0]), int(best[1])])
        else:
            full_center = self._last_center.copy()
            score = 0.0

        self._last_center = full_center
        self._update_status(score)

        return FrameResult(
            frame_index, self.status,
            center=full_center, hessian_score=score,
        )

    def reinitialize(self, frame_gpu, seed_row, seed_col, frame_index):
        return self.initialize(frame_gpu, seed_row, seed_col)

    @staticmethod
    def _closest_blob(result, seed_row, seed_col):
        if result.centers.shape[0] == 0:
            return None
        dists = np.linalg.norm(result.centers - [seed_row, seed_col], axis=1)
        idx   = int(np.argmin(dists))
        return result.centers[idx]


# ============================================================
#  Blob Simple  (Hessian blob → convex polygon)
# ============================================================

class BlobSimpleTracker(BaseTracker):
    """
    Tracks a blob as a convex polygon hull derived from the Hessian DoH map.
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._last_center: Optional[np.ndarray] = None
        self._last_polygon: Optional[np.ndarray] = None
        self._channel = resolve(config.channel)

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int,int,int,int]] = None,
    ) -> InitPreview:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        result = detect_blobs(ch_gpu, self.config.sigma, min_distance=3,
                               threshold_rel=0.15)

        polygon = blob_polygon(ch_gpu, self.config.sigma,
                                (seed_row, seed_col),
                                self.config.search_radius)

        center = np.array([seed_row, seed_col], dtype=np.float32)
        if polygon is not None:
            center = polygon.mean(axis=0)   # centroid of hull

        self._last_center  = center
        self._last_polygon = polygon
        self._init_score   = float(result.response_map.mean()) + 1e-10
        self._reset_to_locked()

        return InitPreview(
            heatmap=_norm_heatmap(result.response_map),
            polygon=polygon,
            center=center,
        )

    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        cr, cc = float(self._last_center[0]), float(self._last_center[1])

        polygon = blob_polygon(ch_gpu, self.config.sigma,
                                (cr, cc), self.config.search_radius)

        result = detect_blobs(ch_gpu, self.config.sigma, min_distance=3,
                               threshold_rel=0.1)
        score  = float(result.response_map.mean())
        self._update_status(score)

        if polygon is not None:
            self._last_center  = polygon.mean(axis=0)
            self._last_polygon = polygon
        # else keep last known

        return FrameResult(
            frame_index, self.status,
            polygon=self._last_polygon,
            center=self._last_center,
            hessian_score=score,
        )

    def reinitialize(self, frame_gpu, seed_row, seed_col, frame_index):
        return self.initialize(frame_gpu, seed_row, seed_col)


# ============================================================
#  Blob Complex  (Hessian blob → binary mask)
# ============================================================

class BlobComplexTracker(BaseTracker):
    """
    Tracks a blob as a full binary mask (uint8, same size as frame).
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._last_center: Optional[np.ndarray] = None
        self._last_mask:   Optional[np.ndarray] = None   # CPU uint8
        self._channel = resolve(config.channel)

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int,int,int,int]] = None,
    ) -> InitPreview:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        result = detect_blobs(ch_gpu, self.config.sigma, min_distance=3,
                               threshold_rel=0.15)

        mask_gpu = blob_mask(ch_gpu, self.config.sigma,
                              (seed_row, seed_col), self.config.search_radius)
        mask_cpu = mask_gpu.get() if mask_gpu is not None else None

        center = np.array([seed_row, seed_col], dtype=np.float32)
        if mask_cpu is not None:
            ys, xs = np.where(mask_cpu > 0)
            if len(ys):
                center = np.array([ys.mean(), xs.mean()], dtype=np.float32)

        self._last_center = center
        self._last_mask   = mask_cpu
        self._init_score  = float(result.response_map.mean()) + 1e-10
        self._reset_to_locked()

        return InitPreview(
            heatmap=_norm_heatmap(result.response_map),
            mask=mask_cpu,
        )

    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        cr, cc = float(self._last_center[0]), float(self._last_center[1])

        mask_gpu = blob_mask(ch_gpu, self.config.sigma,
                              (cr, cc), self.config.search_radius)

        result = detect_blobs(ch_gpu, self.config.sigma,
                               min_distance=3, threshold_rel=0.1)
        score = float(result.response_map.mean())
        self._update_status(score)

        if mask_gpu is not None:
            mask_cpu = mask_gpu.get()
            self._last_mask = mask_cpu
            ys, xs = np.where(mask_cpu > 0)
            if len(ys):
                self._last_center = np.array([ys.mean(), xs.mean()],
                                              dtype=np.float32)

        return FrameResult(
            frame_index, self.status,
            mask=self._last_mask,
            center=self._last_center,
            hessian_score=score,
        )

    def reinitialize(self, frame_gpu, seed_row, seed_col, frame_index):
        return self.initialize(frame_gpu, seed_row, seed_col)


# ============================================================
#  Curve Tracker  (Hessian ridgeness → B-spline)
# ============================================================

class CurveTracker(BaseTracker):
    """
    Tracks a curve-like structure (rope, edge) using Hessian ridge eigenvalues.

    Every frame:
    1. Compute ridgeness map inside a band around last spline
    2. Extract ridge skeleton pixels (ridgeness > threshold)
    3. Sort skeleton pixels by arc-distance along last spline
    4. Fit a B-spline through the ordered skeleton pixels
    5. Sample K evenly-spaced control points from the fitted spline
    """

    N_CONTROL_PTS = 20    # Spline control points stored per frame
    BAND_WIDTH    = 40    # px either side of last spline to search

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._last_spline: Optional[np.ndarray] = None   # (K,2) float32
        self._dark_ridge:  bool = False
        self._channel = resolve(config.channel)

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi=None,
    ) -> InitPreview:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        result = detect_ridges(ch_gpu, self.config.sigma,
                                dark_ridge=self._dark_ridge)

        # Auto-detect contour near click using ridgeness map
        ridge_cpu = result.ridgeness_map.get().astype(np.float32)
        spline    = self._fit_spline_near_seed(ridge_cpu, seed_row, seed_col)

        if spline is None:
            # Fallback: straight line through seed (will correct on next frame)
            spline = np.array([[seed_row, seed_col]] * self.N_CONTROL_PTS,
                               dtype=np.float32)

        self._last_spline = spline
        self._init_score  = float(result.ridgeness_map.mean()) + 1e-10
        self._reset_to_locked()

        return InitPreview(
            heatmap=_norm_heatmap(result.ridgeness_map),
            spline_points=spline,
        )

    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        ch_gpu = extract_channel(frame_gpu, self._channel)
        result = detect_ridges(ch_gpu, self.config.sigma,
                                dark_ridge=self._dark_ridge)

        ridge_cpu = result.ridgeness_map.get().astype(np.float32)
        score     = float(result.ridgeness_map.mean())
        self._update_status(score)

        # Restrict search to band around last spline
        mask  = self._spline_band_mask(ridge_cpu.shape, self._last_spline)
        local = ridge_cpu * mask

        spline = self._fit_spline_from_ridge(local)
        if spline is not None:
            self._last_spline = spline

        return FrameResult(
            frame_index, self.status,
            spline_points=self._last_spline,
            hessian_score=score,
        )

    def reinitialize(self, frame_gpu, seed_row, seed_col, frame_index):
        return self.initialize(frame_gpu, seed_row, seed_col)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_spline_near_seed(
        self,
        ridge_map: np.ndarray,
        seed_row: float,
        seed_col: float,
    ) -> Optional[np.ndarray]:
        """
        Extract ridge pixels near the click, order them by connectivity,
        fit a B-spline, return K sampled control points.
        """
        from scipy.interpolate import splprep, splev

        sr    = self.config.search_radius
        H, W  = ridge_map.shape
        r0    = max(0, int(seed_row) - sr)
        r1    = min(H, int(seed_row) + sr)
        c0    = max(0, int(seed_col) - sr)
        c1    = min(W, int(seed_col) + sr)

        local = ridge_map[r0:r1, c0:c1]
        thresh = local.max() * 0.3
        ys, xs = np.where(local > thresh)
        if len(ys) < 4:
            return None

        pts = np.column_stack([ys + r0, xs + c0]).astype(np.float32)
        return self._order_and_fit(pts)

    def _fit_spline_from_ridge(
        self,
        ridge_map: np.ndarray,
    ) -> Optional[np.ndarray]:
        thresh = ridge_map.max() * 0.3
        ys, xs = np.where(ridge_map > thresh)
        if len(ys) < 4:
            return None
        pts = np.column_stack([ys, xs]).astype(np.float32)
        return self._order_and_fit(pts)

    def _order_and_fit(self, pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Greedy nearest-neighbour ordering of ridge pixels, then B-spline fit.
        Returns (N_CONTROL_PTS, 2) float32 or None.
        """
        from scipy.interpolate import splprep, splev
        from scipy.spatial import KDTree

        if len(pts) < 4:
            return None

        # Greedy NN chain starting from topmost point
        tree     = KDTree(pts)
        visited  = np.zeros(len(pts), dtype=bool)
        start    = int(np.argmin(pts[:, 0]))   # topmost row
        ordered  = [start]
        visited[start] = True

        for _ in range(len(pts) - 1):
            cur  = pts[ordered[-1]]
            dists, idxs = tree.query(cur, k=min(10, len(pts)))
            found = False
            for d, idx in zip(dists, idxs):
                if not visited[idx]:
                    ordered.append(idx)
                    visited[idx] = True
                    found = True
                    break
            if not found:
                break

        chain = pts[ordered]
        if len(chain) < 4:
            return None

        try:
            tck, u = splprep([chain[:, 0], chain[:, 1]], s=len(chain) * 2, k=3)
            u_new  = np.linspace(0, 1, self.N_CONTROL_PTS)
            r_new, c_new = splev(u_new, tck)
            return np.column_stack([r_new, c_new]).astype(np.float32)
        except Exception:
            return None

    def _spline_band_mask(
        self,
        shape: Tuple[int,int],
        spline: np.ndarray,
    ) -> np.ndarray:
        """
        Create a float32 mask (1 inside band, 0 outside) around the spline.
        """
        H, W = shape
        mask = np.zeros((H, W), dtype=np.float32)
        bw   = self.BAND_WIDTH
        for r, c in spline.astype(int):
            r0 = max(0, r - bw);  r1 = min(H, r + bw)
            c0 = max(0, c - bw);  c1 = min(W, c + bw)
            mask[r0:r1, c0:c1] = 1.0
        return mask


# ============================================================
#  Color Area Tracker  (HSV threshold → binary mask)
# ============================================================

class ColorAreaTracker(BaseTracker):
    """
    Tracks regions of a target colour using GPU HSV thresholding.

    The channel selector is intentionally bypassed — HSV needs all three
    BGR channels.  The channel field in config is ignored.
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._tolerance: Optional[ColorTolerance] = config.color_tolerance
        self._last_mask: Optional[np.ndarray]     = None   # CPU uint8

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi=None,
    ) -> InitPreview:
        if self._tolerance is None:
            from gpu.color_mask import sample_pixel
            self._tolerance = sample_pixel(frame_gpu,
                                            int(seed_row), int(seed_col))

        mask_gpu = compute_mask(frame_gpu, self._tolerance)
        mask_cpu = mask_gpu.get()
        preview  = threshold_preview(frame_gpu, self._tolerance)

        self._last_mask  = mask_cpu
        self._init_score = 1.0   # Color area doesn't use Hessian for status
        self._reset_to_locked()

        return InitPreview(
            mask=mask_cpu,
            color_preview=preview,
        )

    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        if self._tolerance is None:
            return FrameResult(frame_index, TrackerStatus.LOST)

        mask_gpu  = compute_mask(frame_gpu, self._tolerance)
        mask_cpu  = mask_gpu.get()
        self._last_mask = mask_cpu

        # Status: LOCKED as long as any pixels match
        pixel_count = int((mask_cpu > 0).sum())
        score = float(pixel_count) / (mask_cpu.size + 1)
        self._update_status(score)

        return FrameResult(
            frame_index, self.status,
            mask=mask_cpu, hessian_score=score,
        )

    def reinitialize(self, frame_gpu, seed_row, seed_col, frame_index):
        return self.initialize(frame_gpu, seed_row, seed_col)

    def update_tolerance(self, tol: ColorTolerance) -> None:
        """Called by the UI tolerance slider in real time."""
        self._tolerance = tol
