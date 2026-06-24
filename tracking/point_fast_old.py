"""
tracking/point_fast.py
----------------------
Point Fast tracker: CSRT + HOG with Hessian re-lock validation.

Pipeline per frame
------------------
1. Extract channel (GPU) → float64 single-channel
2. Download ROI to CPU for CSRT (OpenCV CSRT is CPU-only)
3. CSRT predicts new bbox
4. Compute Hessian DoH score at predicted location (GPU)
5. Compare score to initialisation score * relock_threshold
   → LOCKED or UNCERTAIN
6. If UNCERTAIN for N frames → LOST

Init preview
------------
Shows the HOG patch bounding box around the seed point.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional, Tuple

import cv2
import cupy as cp
import numpy as np

from gpu.channel import ChannelConfig, extract_channel, resolve
from gpu.hessian import hessian_score_at_bbox
from tracking.base_tracker import (
    BaseTracker, FrameResult, InitPreview, TrackerConfig, TrackerStatus
)

from time import perf_counter
class PointFastTracker(BaseTracker):
    """
    Fast point tracker using OpenCV CSRT + Hessian confidence gating.
    """

    # HOG patch size around seed (full side = 2 * _PATCH_HALF + 1)
    _PATCH_HALF = 32

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._csrt:    Optional[cv2.Tracker] = None
        self._bbox:    Optional[Tuple[int,int,int,int]] = None  # (x,y,w,h)
        self._channel: ChannelConfig = resolve(config.channel)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int,int,int,int]] = None,
    ) -> InitPreview:
        H, W = frame_gpu.shape[:2]
        r, c = int(seed_row), int(seed_col)

        # Build initial bbox centred on seed (or use dragged ROI)
        if roi is not None:
            bbox = roi
        else:
            ph = self._PATCH_HALF
            x0 = max(0, c - ph);  y0 = max(0, r - ph)
            x1 = min(W, c + ph);  y1 = min(H, r + ph)
            bbox = (x0, y0, x1 - x0, y1 - y0)

        self._bbox = bbox

        # Extract single channel for CSRT init
        ch_gpu  = extract_channel(frame_gpu, self._channel)
        ch_cpu  = (ch_gpu.get() * 255).astype(np.uint8)

        # CSRT tracker
        self._csrt = cv2.TrackerCSRT_create()
        self._csrt.init(ch_cpu, bbox)

        # Hessian score at init bbox (used as reference)
        self._init_score = hessian_score_at_bbox(
            ch_gpu, self.config.sigma, bbox
        )
        self._reset_to_locked()

        x, y, bw, bh = bbox
        return InitPreview(hog_bbox=(x, y, bw, bh))

    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        if self._csrt is None:
            return FrameResult(frame_index, TrackerStatus.LOST)

        stream = cp.cuda.get_current_stream()

        # 1. Channel extraction only
        t0 = perf_counter()
        ch_gpu = extract_channel(frame_gpu, self._channel)
        stream.synchronize()
        t1 = perf_counter()

        # 2. Host image preparation and transfer
        t2 = perf_counter()
        ch_cpu = (ch_gpu.get() * 255).astype(np.uint8)
        t3 = perf_counter()

        # 3. CSRT
        success, bbox = self._csrt.update(ch_cpu)
        t4 = perf_counter()

        if not success:
            self._update_status(0.0)
            center = (
                np.array(
                    [
                        self._bbox[1] + self._bbox[3] / 2,
                        self._bbox[0] + self._bbox[2] / 2,
                    ],
                    dtype=np.float32,
                )
                if self._bbox else None
            )
            print(
                f"channel={(t1-t0)*1000:.2f} ms, "
                f"download+cpu_convert={(t3-t2)*1000:.2f} ms, "
                f"CSRT={(t4-t3)*1000:.2f} ms"
            )
            return FrameResult(
                frame_index, self.status,
                center=center, hessian_score=0.0
            )

        bbox = tuple(map(int, bbox))
        self._bbox = bbox

        # 4. Hessian
        t5 = perf_counter()
        score = hessian_score_at_bbox(ch_gpu, self.config.sigma, bbox)
        self._update_status(score)
        stream.synchronize()
        t6 = perf_counter()

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        center = np.array([cy, cx], dtype=np.float32)

        print(
            f"channel={(t1-t0)*1000:.2f} ms, "
            f"download+cpu_convert={(t3-t2)*1000:.2f} ms, "
            f"CSRT={(t4-t3)*1000:.2f} ms, "
            f"Hessian={(t6-t5)*1000:.2f} ms"
        )

        return FrameResult(
            frame_index, self.status,
            center=center, hessian_score=score
        )
    def reinitialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        frame_index: int,
    ) -> InitPreview:
        preview = self.initialize(frame_gpu, seed_row, seed_col, roi=None)
        return preview
