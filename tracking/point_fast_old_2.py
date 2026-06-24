"""Direct colour-marker point tracker with per-frame measurement.

This module is the tracking counterpart of ``point_color_initializer.py``.
It removes CSRT/Hessian from the normal point-marker path and instead performs:

    predicted local ROI -> Lab foreground probability -> connected candidates
    -> colour/geometry/motion validation -> weighted subpixel centroid

The motion model only chooses where to search and rejects implausible candidates.
It never substitutes a predicted point for a missing measurement.

Expected input: OpenCV-style BGR ``uint8`` CPU frames.  This is deliberate:
small CPU ROIs avoid the full-frame GPU extraction/download bottleneck of the
old CSRT implementation.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cupy as cp



from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from tracking.point_color_initializer import (
    GaussianLabModel,
    MarkerInitialization,
    PointInitConfig,
    PointMarkerInitializer,
)


class PointTrackStatus(str, Enum):
    """Measurement status for one processed frame."""

    LOCKED = "LOCKED"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"


@dataclass(frozen=True)
class PointTrackConfig:
    """Per-frame localization and validity parameters.

    Defaults are deliberately permissive enough for early tests.  Tighten them
    only from validation recordings with known motion, blur and lighting.
    """

    initializer: PointInitConfig = PointInitConfig()

    # Search box: marker half-size is added automatically.
    normal_search_margin_px: int = 22
    miss_expansion_px: int = 16
    max_search_margin_px: int = 150

    # Multi-threshold candidate formation.  The first threshold producing an
    # accepted candidate set is used, avoiding duplicate copies of one blob.
    probability_thresholds: tuple[float, ...] = (0.72, 0.62, 0.52, 0.42)
    morphology_close_iterations: int = 1
    min_component_area_px: int = 10

    # Colour validity gates.
    min_inside_median_probability: float = 0.48
    min_inside_strong_fraction: float = 0.28
    min_probability_margin: float = 0.08
    strong_probability_threshold: float = 0.70

    # General size validity gates relative to the current expected model.
    min_area_ratio: float = 0.55
    max_area_ratio: float = 1.70
    min_axis_size_ratio: float = 0.55
    max_axis_size_ratio: float = 1.75

    # Shape-specific validity gates.  These are softer than initialization
    # because blur/compression may reduce contour quality during motion.
    min_circle_circularity: float = 0.48
    min_rectangle_rectangularity: float = 0.60
    max_geometric_center_difference_px: float = 3.0

    # Candidate must remain plausible relative to the motion prediction.
    max_prediction_error_px: float = 22.0
    miss_motion_allowance_px: float = 14.0
    ambiguity_score_margin: float = 0.08

    # State and adaptation.
    max_uncertain_frames: int = 2
    high_confidence_threshold: float = 0.78
    geometry_adaptation_alpha: float = 0.05
    adapt_geometry: bool = True

    # Optional boundary-gradient refinement inherited from the initializer.
    refine_boundary_with_gradient: bool = True


@dataclass(frozen=True)
class PointFrameMeasurement:
    """Result of processing one frame.

    ``center_rc`` is present only for accepted, visible measurements.  No
    predicted point is exported as measured data when ``valid`` is false.
    """

    frame_index: int
    status: PointTrackStatus
    valid: bool
    center_rc: Optional[np.ndarray]
    predicted_center_rc: Optional[np.ndarray]
    confidence: float

    search_bbox_xywh: Optional[tuple[int, int, int, int]] = None
    tight_bbox_xywh: Optional[tuple[int, int, int, int]] = None
    contour_xy_global: Optional[np.ndarray] = None

    colour_score: Optional[float] = None
    shape_score: Optional[float] = None
    motion_score: Optional[float] = None
    selected_threshold: Optional[float] = None

    measured_area: Optional[float] = None
    circularity: Optional[float] = None
    rectangularity: Optional[float] = None
    centre_difference: Optional[float] = None
    miss_count: int = 0
    reason: str = ""


@dataclass
class _TrackingState:
    """Mutable internal state. The learned colour model remains fixed here."""

    initialization: MarkerInitialization
    expected_area: float
    expected_width: float
    expected_height: float
    last_valid_center_rc: np.ndarray
    previous_valid_center_rc: Optional[np.ndarray]
    last_valid_frame_index: int
    consecutive_misses: int = 0
    had_lock: bool = True


@dataclass(frozen=True)
class _Candidate:
    center_rc: np.ndarray
    tight_bbox_xywh: tuple[int, int, int, int]
    contour_xy_global: np.ndarray
    mask_roi: np.ndarray

    confidence: float
    colour_score: float
    shape_score: float
    motion_score: float
    threshold: float

    area: float
    width: float
    height: float
    circularity: float
    rectangularity: float
    centre_difference: float
from tracking.base_tracker import (
    BaseTracker, FrameResult, InitPreview, TrackerConfig, TrackerStatus
)
from .point_color_initializer import PointInitConfig
# from .point_color_tracker import PointColorTracker, PointTrackConfig
class PointFastTracker(BaseTracker):
    """One-click direct colour marker tracker for experimental positions."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)

        self._engine = PointColorTracker(
            PointTrackConfig(
                initializer=PointInitConfig(
                    init_half_size=50,
                    search_margin_px=22,
                ),
                normal_search_margin_px=22,
                max_prediction_error_px=float(config.search_radius),
                max_uncertain_frames=config.lost_after_frames,
            )
        )

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def initialization(self) -> Optional[MarkerInitialization]:
        return self._state.initialization if self._state is not None else None

    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> InitPreview:
        # Colour point tracker initializes only from the click.
        # ROI is accepted for compatibility with BaseTracker/TrackerManager.
        _ = roi

        frame_bgr = cp.asnumpy(frame_gpu)

        initialization = self._engine.initialize(
            frame_bgr,
            click_row=seed_row,
            click_col=seed_col,
            frame_index=0,
        )

        self._reset_to_locked()

        return InitPreview(
            center=initialization.center_rc,
            hog_bbox=initialization.tight_bbox_xywh,
        )
    def reinitialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        frame_index: int,
    ) -> InitPreview:
        frame_bgr = cp.asnumpy(frame_gpu)

        initialization = self._engine.initialize(
            frame_bgr,
            click_row=seed_row,
            click_col=seed_col,
            frame_index=frame_index,
        )

        self._reset_to_locked()

        return InitPreview(
            center=initialization.center_rc.astype(np.float32),
            hog_bbox=initialization.tight_bbox_xywh,
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int) -> PointFrameMeasurement:
        """Measure the marker in one new frame.

        The returned centre is a current-frame colour/shape measurement only.
        When no candidate passes validation, ``center_rc`` is ``None`` and the
        internal prediction is retained only for subsequent reacquisition.
        """
        self._validate_frame(frame_bgr)
        if self._state is None:
            return PointFrameMeasurement(
                frame_index=frame_index,
                status=PointTrackStatus.LOST,
                valid=False,
                center_rc=None,
                predicted_center_rc=None,
                confidence=0.0,
                reason="Tracker has not been initialized.",
            )

        predicted_center = self._predict_center(frame_index)
        search_bbox = self._search_bbox(frame_bgr.shape[:2], predicted_center)
        x0, y0, width, height = search_bbox
        roi_bgr = frame_bgr[y0 : y0 + height, x0 : x0 + width]
        roi_lab = self._to_lab(roi_bgr)
        probability = self._foreground_probability(
            roi_lab,
            self._state.initialization.foreground_model,
            self._state.initialization.background_model,
        )

        candidates, threshold = self._accepted_candidates(
            probability=probability,
            roi_origin_xy=(x0, y0),
            predicted_center_rc=predicted_center,
        )
        if not candidates:
            reason = (
                "No candidate passed colour, shape and motion validation."
                if threshold is not None
                else "No plausible colour-connected candidate was detected."
            )
            return self._measurement_failure(frame_index, predicted_center, search_bbox, reason)

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        best = candidates[0]
        if (
            len(candidates) > 1
            and best.confidence - candidates[1].confidence < self.config.ambiguity_score_margin
        ):
            return self._measurement_failure(
                frame_index,
                predicted_center,
                search_bbox,
                "Multiple similarly plausible candidates were detected.",
            )

        was_missing = self._state.consecutive_misses > 0
        was_lost = self._state.consecutive_misses > self.config.max_uncertain_frames
        self._accept_candidate(best, frame_index)
        status = PointTrackStatus.REACQUIRED if (was_missing or was_lost) else PointTrackStatus.LOCKED

        return PointFrameMeasurement(
            frame_index=frame_index,
            status=status,
            valid=True,
            center_rc=best.center_rc.copy(),
            predicted_center_rc=predicted_center.copy(),
            confidence=best.confidence,
            search_bbox_xywh=search_bbox,
            tight_bbox_xywh=best.tight_bbox_xywh,
            contour_xy_global=best.contour_xy_global.copy(),
            colour_score=best.colour_score,
            shape_score=best.shape_score,
            motion_score=best.motion_score,
            selected_threshold=best.threshold,
            measured_area=best.area,
            circularity=best.circularity,
            rectangularity=best.rectangularity,
            centre_difference=best.centre_difference,
            miss_count=0,
            reason="Accepted visible marker measurement.",
        )

    def render_measurement(
        self,
        frame_bgr: np.ndarray,
        result: PointFrameMeasurement,
    ) -> np.ndarray:
        """Draw search area, accepted geometry and measured/predicted centres."""
        preview = frame_bgr.copy()
        if result.search_bbox_xywh is not None:
            x, y, width, height = result.search_bbox_xywh
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 255), 1)
        if result.contour_xy_global is not None:
            contour = np.rint(result.contour_xy_global).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(preview, [contour], True, (0, 255, 0), 1)
        if result.tight_bbox_xywh is not None:
            x, y, width, height = result.tight_bbox_xywh
            cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 0, 0), 1)
        if result.predicted_center_rc is not None:
            row, col = result.predicted_center_rc
            cv2.drawMarker(
                preview,
                (int(round(col)), int(round(row))),
                (255, 255, 0),
                cv2.MARKER_TILTED_CROSS,
                9,
                1,
            )
        if result.center_rc is not None:
            row, col = result.center_rc
            cv2.drawMarker(
                preview,
                (int(round(col)), int(round(row))),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                11,
                1,
            )
        cv2.putText(
            preview,
            f"{result.status.value} conf={result.confidence:.2f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255) if result.valid else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return preview

    # ------------------------------------------------------------------
    # Prediction/search
    # ------------------------------------------------------------------

    def _predict_center(self, frame_index: int) -> np.ndarray:
        assert self._state is not None
        state = self._state
        if state.previous_valid_center_rc is None:
            return state.last_valid_center_rc.copy()
        velocity = state.last_valid_center_rc - state.previous_valid_center_rc
        frames_since_measurement = max(1, int(frame_index) - state.last_valid_frame_index)
        return (state.last_valid_center_rc + velocity * frames_since_measurement).astype(np.float32)

    def _search_bbox(
        self,
        frame_hw: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> tuple[int, int, int, int]:
        assert self._state is not None
        expected_half_size = int(
            np.ceil(max(self._state.expected_width, self._state.expected_height) / 2.0)
        )
        margin = min(
            self.config.max_search_margin_px,
            self.config.normal_search_margin_px
            + self._state.consecutive_misses * self.config.miss_expansion_px,
        )
        half_size = expected_half_size + margin
        return self._box_around_point(
            row=int(round(float(predicted_center_rc[0]))),
            col=int(round(float(predicted_center_rc[1]))),
            half_size=half_size,
            frame_hw=frame_hw,
        )

    # ------------------------------------------------------------------
    # Candidate extraction/scoring
    # ------------------------------------------------------------------

    def _accepted_candidates(
        self,
        *,
        probability: np.ndarray,
        roi_origin_xy: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> tuple[list[_Candidate], Optional[float]]:
        kernel = np.ones((3, 3), dtype=np.uint8)
        any_components = False
        for threshold in self.config.probability_thresholds:
            binary = (probability >= threshold).astype(np.uint8)
            if self.config.morphology_close_iterations > 0:
                binary = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=self.config.morphology_close_iterations,
                )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            accepted: list[_Candidate] = []
            for label in range(1, count):
                area_px = int(stats[label, cv2.CC_STAT_AREA])
                if area_px < self.config.min_component_area_px:
                    continue
                any_components = True
                component_mask = (labels == label).astype(np.uint8)
                candidate = self._score_candidate(
                    probability=probability,
                    component_mask=component_mask,
                    roi_origin_xy=roi_origin_xy,
                    predicted_center_rc=predicted_center_rc,
                    threshold=float(threshold),
                )
                if candidate is not None:
                    accepted.append(candidate)
            if accepted:
                return accepted, float(threshold)
        return [], (self.config.probability_thresholds[-1] if any_components else None)

    def _score_candidate(
        self,
        *,
        probability: np.ndarray,
        component_mask: np.ndarray,
        roi_origin_xy: tuple[int, int],
        predicted_center_rc: np.ndarray,
        threshold: float,
    ) -> Optional[_Candidate]:
        assert self._state is not None
        x0, y0 = roi_origin_xy
        contour = self._outer_contour(component_mask)
        if contour is None:
            return None
        contour_xy = contour.reshape(-1, 2).astype(np.float32)
        if self.config.refine_boundary_with_gradient:
            contour_xy = self._initializer._refine_contour_by_probability_gradient(
                contour, probability
            )

        center_local_rc = self._weighted_centroid(probability, component_mask)
        center_rc = np.array([y0 + center_local_rc[0], x0 + center_local_rc[1]], dtype=np.float32)
        geometry = self._geometry(contour_xy, center_local_rc)
        ring_mask = self._initializer._background_ring(component_mask)
        quality = self._colour_quality(probability, component_mask, ring_mask)

        area_ratio = geometry["area"] / max(self._state.expected_area, 1e-9)
        expected_axes = sorted([self._state.expected_width, self._state.expected_height])
        measured_axes = sorted([geometry["width"], geometry["height"]])
        axis_ratios = [
            measured_axes[index] / max(expected_axes[index], 1e-9)
            for index in range(2)
        ]
        prediction_error = float(np.linalg.norm(center_rc - predicted_center_rc))
        max_motion_error = (
            self.config.max_prediction_error_px
            + self._state.consecutive_misses * self.config.miss_motion_allowance_px
        )

        if not self._passes_colour_gates(quality):
            return None
        if not (self.config.min_area_ratio <= area_ratio <= self.config.max_area_ratio):
            return None
        if not all(
            self.config.min_axis_size_ratio <= ratio <= self.config.max_axis_size_ratio
            for ratio in axis_ratios
        ):
            return None
        if prediction_error > max_motion_error:
            return None
        if not self._passes_shape_gates(geometry):
            return None

        colour_score = self._colour_score(quality)
        shape_score = self._shape_score(geometry, area_ratio, axis_ratios)
        motion_score = float(np.exp(-0.5 * (prediction_error / max(max_motion_error / 2.0, 1.0)) ** 2))
        confidence = float(np.clip(0.48 * colour_score + 0.32 * shape_score + 0.20 * motion_score, 0.0, 1.0))

        tight_bbox_local = self._mask_bbox(component_mask)
        tight_bbox_global = (
            x0 + tight_bbox_local[0],
            y0 + tight_bbox_local[1],
            tight_bbox_local[2],
            tight_bbox_local[3],
        )
        contour_global = contour_xy.copy()
        contour_global[:, 0] += x0
        contour_global[:, 1] += y0

        return _Candidate(
            center_rc=center_rc,
            tight_bbox_xywh=tight_bbox_global,
            contour_xy_global=contour_global,
            mask_roi=component_mask,
            confidence=confidence,
            colour_score=colour_score,
            shape_score=shape_score,
            motion_score=motion_score,
            threshold=threshold,
            area=float(geometry["area"]),
            width=float(geometry["width"]),
            height=float(geometry["height"]),
            circularity=float(geometry["circularity"]),
            rectangularity=float(geometry["rectangularity"]),
            centre_difference=float(geometry["center_difference"]),
        )

    def _passes_colour_gates(self, quality: dict[str, float]) -> bool:
        return bool(
            quality["median_inside_probability"] >= self.config.min_inside_median_probability
            and quality["strong_inside_fraction"] >= self.config.min_inside_strong_fraction
            and quality["probability_margin"] >= self.config.min_probability_margin
        )

    def _passes_shape_gates(self, geometry: dict[str, float]) -> bool:
        assert self._state is not None
        shape_model = self._state.initialization.shape_model
        if shape_model == "circle_like":
            if geometry["circularity"] < self.config.min_circle_circularity:
                return False
        elif shape_model == "rectangle":
            if geometry["rectangularity"] < self.config.min_rectangle_rectangularity:
                return False
        centre_difference = geometry["center_difference"]
        if np.isfinite(centre_difference) and centre_difference > self.config.max_geometric_center_difference_px:
            return False
        return True

    def _colour_quality(
        self,
        probability: np.ndarray,
        component_mask: np.ndarray,
        ring_mask: np.ndarray,
    ) -> dict[str, float]:
        inside = probability[component_mask > 0]
        ring = probability[ring_mask > 0]
        if inside.size == 0 or ring.size == 0:
            return {
                "median_inside_probability": 0.0,
                "strong_inside_fraction": 0.0,
                "mean_inside_probability": 0.0,
                "mean_ring_probability": 1.0,
                "probability_margin": -1.0,
            }
        return {
            "median_inside_probability": float(np.median(inside)),
            "strong_inside_fraction": float(np.mean(inside > self.config.strong_probability_threshold)),
            "mean_inside_probability": float(np.mean(inside)),
            "mean_ring_probability": float(np.mean(ring)),
            "probability_margin": float(np.mean(inside) - np.mean(ring)),
        }

    @staticmethod
    def _colour_score(quality: dict[str, float]) -> float:
        margin = float(np.clip(quality["probability_margin"] / 0.60, 0.0, 1.0))
        return float(
            np.clip(
                0.45 * quality["median_inside_probability"]
                + 0.30 * quality["strong_inside_fraction"]
                + 0.25 * margin,
                0.0,
                1.0,
            )
        )

    def _shape_score(
        self,
        geometry: dict[str, float],
        area_ratio: float,
        axis_ratios: list[float],
    ) -> float:
        assert self._state is not None
        area_score = float(np.exp(-abs(np.log(max(area_ratio, 1e-9))) / 0.40))
        axes_score = float(
            np.mean([np.exp(-abs(np.log(max(ratio, 1e-9))) / 0.40) for ratio in axis_ratios])
        )
        shape_model = self._state.initialization.shape_model
        if shape_model == "circle_like":
            model_score = float(np.clip(geometry["circularity"] / max(self._state.initialization.circularity, 0.5), 0.0, 1.0))
        elif shape_model == "rectangle":
            model_score = float(np.clip(geometry["rectangularity"] / max(self._state.initialization.rectangularity, 0.5), 0.0, 1.0))
        else:
            model_score = 1.0
        centre_difference = geometry["center_difference"]
        centre_score = (
            float(np.exp(-0.5 * (centre_difference / 1.5) ** 2))
            if np.isfinite(centre_difference)
            else 1.0
        )
        return float(np.clip(0.38 * area_score + 0.27 * axes_score + 0.22 * model_score + 0.13 * centre_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # State update and failure handling
    # ------------------------------------------------------------------

    def _accept_candidate(self, candidate: _Candidate, frame_index: int) -> None:
        assert self._state is not None
        state = self._state
        state.previous_valid_center_rc = state.last_valid_center_rc.copy()
        state.last_valid_center_rc = candidate.center_rc.copy()
        state.last_valid_frame_index = int(frame_index)
        state.consecutive_misses = 0
        state.had_lock = True

        if self.config.adapt_geometry and candidate.confidence >= self.config.high_confidence_threshold:
            alpha = float(self.config.geometry_adaptation_alpha)
            state.expected_area = (1.0 - alpha) * state.expected_area + alpha * candidate.area
            state.expected_width = (1.0 - alpha) * state.expected_width + alpha * candidate.width
            state.expected_height = (1.0 - alpha) * state.expected_height + alpha * candidate.height

    def _measurement_failure(
        self,
        frame_index: int,
        predicted_center_rc: np.ndarray,
        search_bbox_xywh: tuple[int, int, int, int],
        reason: str,
    ) -> PointFrameMeasurement:
        assert self._state is not None
        self._state.consecutive_misses += 1
        status = (
            PointTrackStatus.UNCERTAIN
            if self._state.consecutive_misses <= self.config.max_uncertain_frames
            else PointTrackStatus.LOST
        )
        return PointFrameMeasurement(
            frame_index=frame_index,
            status=status,
            valid=False,
            center_rc=None,
            predicted_center_rc=predicted_center_rc.copy(),
            confidence=0.0,
            search_bbox_xywh=search_bbox_xywh,
            miss_count=self._state.consecutive_misses,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Low-level image and geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> None:
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a NumPy array.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3).")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr must be uint8 BGR data.")

    @staticmethod
    def _to_lab(roi_bgr: np.ndarray) -> np.ndarray:
        roi_float = roi_bgr.astype(np.float32) / 255.0
        return cv2.cvtColor(roi_float, cv2.COLOR_BGR2Lab).astype(np.float32)

    @staticmethod
    def _foreground_probability(
        roi_lab: np.ndarray,
        foreground: GaussianLabModel,
        background: GaussianLabModel,
    ) -> np.ndarray:
        score = foreground.log_likelihood(roi_lab) - background.log_likelihood(roi_lab)
        score = np.clip(score, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)

    @staticmethod
    def _outer_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(contours, key=cv2.contourArea) if contours else None

    @staticmethod
    def _weighted_centroid(probability: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = probability * mask.astype(np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            raise RuntimeError("Candidate component has zero probability mass.")
        rows, cols = np.indices(weights.shape, dtype=np.float32)
        return np.array(
            [float((weights * rows).sum() / total), float((weights * cols).sum() / total)],
            dtype=np.float32,
        )

    @staticmethod
    def _geometry(contour_xy: np.ndarray, weighted_center_local_rc: np.ndarray) -> dict[str, float]:
        contour = contour_xy.reshape(-1, 1, 2).astype(np.float32)
        area = float(abs(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            return {
                "area": 0.0,
                "width": 0.0,
                "height": 0.0,
                "circularity": 0.0,
                "rectangularity": 0.0,
                "center_difference": float("inf"),
            }
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-9))
        rect = cv2.minAreaRect(contour)
        (rect_cx, rect_cy), (rect_width, rect_height), _ = rect
        rect_area = max(float(rect_width * rect_height), 1e-9)
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
        weighted_xy = np.array([weighted_center_local_rc[1], weighted_center_local_rc[0]], dtype=np.float32)

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        geometric_center: Optional[np.ndarray] = None
        if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.float32)) and rectangularity >= 0.60:
            geometric_center = np.array([rect_cx, rect_cy], dtype=np.float32)
        elif contour.shape[0] >= 5 and circularity >= 0.48:
            geometric_center = np.array(cv2.fitEllipse(contour)[0], dtype=np.float32)
        centre_difference = (
            float(np.linalg.norm(weighted_xy - geometric_center))
            if geometric_center is not None
            else float("nan")
        )
        return {
            "area": area,
            "width": float(rect_width),
            "height": float(rect_height),
            "circularity": circularity,
            "rectangularity": rectangularity,
            "center_difference": centre_difference,
        }

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        rows, cols = np.nonzero(mask)
        if cols.size == 0:
            raise RuntimeError("Cannot compute bounding box for empty component.")
        x0, x1 = int(cols.min()), int(cols.max())
        y0, y1 = int(rows.min()), int(rows.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    @staticmethod
    def _box_around_point(
        row: int,
        col: int,
        half_size: int,
        frame_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = frame_hw
        x0 = max(0, col - half_size)
        y0 = max(0, row - half_size)
        x1 = min(width, col + half_size + 1)
        y1 = min(height, row + half_size + 1)
        return (x0, y0, x1 - x0, y1 - y0)
