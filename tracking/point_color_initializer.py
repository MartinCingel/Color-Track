"""One-click initialization for a direct colour-marker point tracker.

This module implements initialization only. It intentionally contains no CSRT
or Hessian logic. It learns a foreground/background Lab colour model from one
click, segments the connected clicked feature, optionally refines its contour
using probability-gradient peaks, estimates initial geometry, and produces a
larger search ROI for later per-frame localization.

Expected input: an OpenCV-style BGR uint8 CPU frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np


ShapeModel = Literal["generic_compact", "circle_like", "rectangle"]


class InitializationError(RuntimeError):
    """Raised when a marker cannot be initialized reliably from the click."""


@dataclass(frozen=True)
class GaussianLabModel:
    """Regularized three-channel Lab Gaussian colour model."""

    mean: np.ndarray
    inv_cov: np.ndarray
    log_det_cov: float

    @classmethod
    def fit(
        cls,
        pixels_lab: np.ndarray,
        *,
        sigma_floor: tuple[float, float, float],
        lightness_variance_scale: float = 4.0,
    ) -> "GaussianLabModel":
        pixels = np.asarray(pixels_lab, dtype=np.float32).reshape(-1, 3)
        if pixels.shape[0] < 3:
            raise InitializationError("Too few pixels to learn a Lab colour model.")

        mean = np.median(pixels, axis=0).astype(np.float32)
        centered = pixels - mean
        cov = (
            np.cov(centered, rowvar=False).astype(np.float32)
            if pixels.shape[0] > 3
            else np.zeros((3, 3), dtype=np.float32)
        )

        floors = np.square(np.asarray(sigma_floor, dtype=np.float32))
        for index in range(3):
            cov[index, index] = max(float(cov[index, index]), float(floors[index]))
        cov[0, 0] *= lightness_variance_scale
        cov += np.eye(3, dtype=np.float32) * 1e-4

        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            raise InitializationError("Failed to construct a positive colour covariance matrix.")

        return cls(
            mean=mean,
            inv_cov=np.linalg.inv(cov).astype(np.float32),
            log_det_cov=float(log_det),
        )

    def log_likelihood(self, lab_image: np.ndarray) -> np.ndarray:
        delta = np.asarray(lab_image, dtype=np.float32) - self.mean
        distance2 = np.einsum("...i,ij,...j->...", delta, self.inv_cov, delta)
        return -0.5 * (distance2 + self.log_det_cov)


@dataclass(frozen=True)
class PointInitConfig:
    """Parameters for one-click marker initialization."""

    init_half_size: int = 50          # 101 x 101 local initialization window
    seed_radius: int = 2              # 5 x 5 clicked-colour seed
    border_width: int = 8             # preliminary background sampled at window border

    probability_thresholds: tuple[float, ...] = (0.72, 0.62, 0.52, 0.42)
    min_component_area: int = 20
    max_component_fraction: float = 0.65

    # Preliminary models are tolerant; final models use clean interior/ring pixels.
    preliminary_fg_sigma_floor: tuple[float, float, float] = (8.0, 8.0, 8.0)
    preliminary_bg_sigma_floor: tuple[float, float, float] = (8.0, 6.0, 6.0)
    final_fg_sigma_floor: tuple[float, float, float] = (5.0, 4.0, 4.0)
    final_bg_sigma_floor: tuple[float, float, float] = (6.0, 5.0, 5.0)
    lightness_variance_scale: float = 4.0

    erode_iterations: int = 1
    background_ring_inner_dilate: int = 2
    background_ring_outer_dilate: int = 7
    min_model_pixels: int = 12

    refine_boundary_with_gradient: bool = True
    probability_blur_sigma: float = 0.8
    gradient_normal_radius: float = 2.5
    gradient_profile_samples: int = 17

    search_margin_px: int = 22

    min_inside_median_probability: float = 0.52
    min_inside_strong_fraction: float = 0.40
    min_probability_margin: float = 0.12

    rectangle_rectangularity_threshold: float = 0.78
    circle_circularity_threshold: float = 0.73


@dataclass(frozen=True)
class MarkerInitialization:
    """Model and geometry obtained from one successful click."""

    center_rc: np.ndarray
    geometric_center_rc: Optional[np.ndarray]
    shape_model: ShapeModel

    foreground_model: GaussianLabModel
    background_model: GaussianLabModel

    tight_bbox_xywh: tuple[int, int, int, int]
    search_bbox_xywh: tuple[int, int, int, int]
    init_roi_xywh: tuple[int, int, int, int]

    marker_mask_roi: np.ndarray
    probability_roi: np.ndarray
    refined_contour_xy_global: np.ndarray

    expected_area: float
    expected_width: float
    expected_height: float
    circularity: float
    rectangularity: float
    confidence: float
    quality: dict[str, float]


class PointMarkerInitializer:
    """One-click direct colour marker initializer without Hessian."""

    def __init__(self, config: PointInitConfig | None = None) -> None:
        self.config = config or PointInitConfig()

    def initialize(
        self,
        frame_bgr: np.ndarray,
        click_row: float,
        click_col: float,
    ) -> MarkerInitialization:
        """Initialize a marker from one click inside it.

        Args:
            frame_bgr: CPU BGR image, uint8, shape (H, W, 3).
            click_row: Clicked y/row coordinate in full-frame coordinates.
            click_col: Clicked x/column coordinate in full-frame coordinates.
        """
        self._validate_frame(frame_bgr)
        height, width = frame_bgr.shape[:2]
        row = int(round(click_row))
        col = int(round(click_col))
        if not (0 <= row < height and 0 <= col < width):
            raise InitializationError("Clicked point lies outside the frame.")

        roi_xywh = self._box_around_point(
            row, col, self.config.init_half_size, frame_bgr.shape[:2]
        )
        x0, y0, roi_width, roi_height = roi_xywh
        roi_bgr = frame_bgr[y0 : y0 + roi_height, x0 : x0 + roi_width]
        roi_lab = self._to_lab(roi_bgr)
        click_local_rc = (row - y0, col - x0)

        seed_pixels = self._seed_pixels(roi_lab, click_local_rc)
        border_pixels = self._border_pixels(roi_lab)
        preliminary_fg = GaussianLabModel.fit(
            seed_pixels,
            sigma_floor=self.config.preliminary_fg_sigma_floor,
            lightness_variance_scale=self.config.lightness_variance_scale,
        )
        preliminary_bg = GaussianLabModel.fit(
            border_pixels,
            sigma_floor=self.config.preliminary_bg_sigma_floor,
            lightness_variance_scale=self.config.lightness_variance_scale,
        )

        provisional_probability = self._foreground_probability(
            roi_lab, preliminary_fg, preliminary_bg
        )
        provisional_mask = self._component_containing_click(
            provisional_probability, click_local_rc
        )

        interior_mask = self._interior_mask(provisional_mask)
        ring_mask = self._background_ring(provisional_mask)
        if int(np.count_nonzero(interior_mask)) < self.config.min_model_pixels:
            raise InitializationError(
                "Marker interior is too small after boundary exclusion; "
                "choose a larger or clearer marker."
            )
        if int(np.count_nonzero(ring_mask)) < self.config.min_model_pixels:
            raise InitializationError("Not enough local background around marker.")

        foreground_model = GaussianLabModel.fit(
            roi_lab[interior_mask > 0],
            sigma_floor=self.config.final_fg_sigma_floor,
            lightness_variance_scale=self.config.lightness_variance_scale,
        )
        background_model = GaussianLabModel.fit(
            roi_lab[ring_mask > 0],
            sigma_floor=self.config.final_bg_sigma_floor,
            lightness_variance_scale=self.config.lightness_variance_scale,
        )

        probability = self._foreground_probability(
            roi_lab, foreground_model, background_model
        )
        final_mask = self._component_containing_click(probability, click_local_rc)
        provisional_area = float(np.count_nonzero(provisional_mask))
        final_area = float(np.count_nonzero(final_mask))
        if not (0.50 * provisional_area <= final_area <= 2.0 * provisional_area):
            raise InitializationError(
                "Refined colour model changed marker area implausibly; initialization is ambiguous."
            )

        contour_local = self._outer_contour(final_mask)
        refined_contour_local = (
            self._refine_contour_by_probability_gradient(contour_local, probability)
            if self.config.refine_boundary_with_gradient
            else contour_local.reshape(-1, 2).astype(np.float32)
        )
        weighted_center_local_rc = self._weighted_centroid(probability, final_mask)
        geometry = self._geometry(refined_contour_local, weighted_center_local_rc)

        tight_bbox_local = self._mask_bbox(final_mask)
        tight_bbox_global = (
            x0 + tight_bbox_local[0],
            y0 + tight_bbox_local[1],
            tight_bbox_local[2],
            tight_bbox_local[3],
        )
        search_bbox = self._expand_bbox(
            tight_bbox_global, self.config.search_margin_px, frame_bgr.shape[:2]
        )

        center_global_rc = np.array(
            [y0 + weighted_center_local_rc[0], x0 + weighted_center_local_rc[1]],
            dtype=np.float32,
        )
        geometric_center_rc: Optional[np.ndarray] = None
        if geometry["geometric_center_xy"] is not None:
            geometric_center_xy = geometry["geometric_center_xy"]
            geometric_center_rc = np.array(
                [y0 + geometric_center_xy[1], x0 + geometric_center_xy[0]],
                dtype=np.float32,
            )

        refined_contour_global = refined_contour_local.copy()
        refined_contour_global[:, 0] += x0
        refined_contour_global[:, 1] += y0

        quality = self._quality(
            probability=probability,
            component_mask=final_mask,
            ring_mask=self._background_ring(final_mask),
            geometry=geometry,
        )
        self._assert_quality(quality)
        confidence = self._confidence(quality)

        return MarkerInitialization(
            center_rc=center_global_rc,
            geometric_center_rc=geometric_center_rc,
            shape_model=geometry["shape_model"],
            foreground_model=foreground_model,
            background_model=background_model,
            tight_bbox_xywh=tight_bbox_global,
            search_bbox_xywh=search_bbox,
            init_roi_xywh=roi_xywh,
            marker_mask_roi=final_mask,
            probability_roi=probability.astype(np.float32),
            refined_contour_xy_global=refined_contour_global.astype(np.float32),
            expected_area=geometry["area"],
            expected_width=geometry["width"],
            expected_height=geometry["height"],
            circularity=geometry["circularity"],
            rectangularity=geometry["rectangularity"],
            confidence=confidence,
            quality=quality,
        )

    def render_preview(
        self,
        frame_bgr: np.ndarray,
        result: MarkerInitialization,
    ) -> np.ndarray:
        """Draw contour, measured centre, tight bbox and future search ROI."""
        preview = frame_bgr.copy()
        contour = np.rint(result.refined_contour_xy_global).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [contour], isClosed=True, color=(0, 255, 0), thickness=1)

        row, col = result.center_rc
        cv2.drawMarker(
            preview,
            (int(round(col)), int(round(row))),
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=11,
            thickness=1,
        )

        x, y, box_width, box_height = result.tight_bbox_xywh
        cv2.rectangle(preview, (x, y), (x + box_width, y + box_height), (255, 0, 0), 1)
        x, y, box_width, box_height = result.search_bbox_xywh
        cv2.rectangle(preview, (x, y), (x + box_width, y + box_height), (0, 255, 255), 1)
        cv2.putText(
            preview,
            f"{result.shape_model} conf={result.confidence:.2f}",
            (max(0, x), max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return preview

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

    def _seed_pixels(self, roi_lab: np.ndarray, click_local_rc: tuple[int, int]) -> np.ndarray:
        row, col = click_local_rc
        radius = self.config.seed_radius
        y0, y1 = max(0, row - radius), min(roi_lab.shape[0], row + radius + 1)
        x0, x1 = max(0, col - radius), min(roi_lab.shape[1], col + radius + 1)
        patch = roi_lab[y0:y1, x0:x1].reshape(-1, 3)
        if patch.shape[0] < 4:
            raise InitializationError("Clicked point is too close to frame border.")
        return patch

    def _border_pixels(self, roi_lab: np.ndarray) -> np.ndarray:
        border = min(self.config.border_width, roi_lab.shape[0] // 4, roi_lab.shape[1] // 4)
        if border < 1:
            raise InitializationError("Initialization ROI is too small.")
        mask = np.zeros(roi_lab.shape[:2], dtype=np.uint8)
        mask[:border, :] = 1
        mask[-border:, :] = 1
        mask[:, :border] = 1
        mask[:, -border:] = 1
        return roi_lab[mask > 0]

    def _component_containing_click(
        self,
        probability: np.ndarray,
        click_local_rc: tuple[int, int],
    ) -> np.ndarray:
        click_row, click_col = click_local_rc
        kernel = np.ones((3, 3), dtype=np.uint8)
        border_failure = False
        for threshold in self.config.probability_thresholds:
            binary = (probability >= threshold).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            label = int(labels[click_row, click_col])
            if label == 0:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not (self.config.min_component_area <= area <= probability.size * self.config.max_component_fraction):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            touches_border = (
                x <= 0 or y <= 0 or x + width >= probability.shape[1] or y + height >= probability.shape[0]
            )
            if touches_border:
                border_failure = True
                continue
            return (labels == label).astype(np.uint8)
        if border_failure:
            raise InitializationError(
                "Detected marker reaches initialization-window border; increase init_half_size."
            )
        raise InitializationError("No reliable colour-connected region containing the click was found.")

    def _interior_mask(self, marker_mask: np.ndarray) -> np.ndarray:
        if self.config.erode_iterations <= 0:
            return marker_mask.copy()
        return cv2.erode(
            marker_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=self.config.erode_iterations,
        )

    def _background_ring(self, marker_mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=np.uint8)
        inner = cv2.dilate(marker_mask, kernel, iterations=self.config.background_ring_inner_dilate)
        outer = cv2.dilate(marker_mask, kernel, iterations=self.config.background_ring_outer_dilate)
        return ((outer > 0) & (inner == 0)).astype(np.uint8)

    @staticmethod
    def _outer_contour(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise InitializationError("No contour found for initialized marker.")
        return max(contours, key=cv2.contourArea)

    def _refine_contour_by_probability_gradient(
        self,
        contour: np.ndarray,
        probability: np.ndarray,
    ) -> np.ndarray:
        """Move contour samples to the strongest nearby probability-gradient boundary."""
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] < 5:
            return points
        smooth = cv2.GaussianBlur(
            probability.astype(np.float32),
            (0, 0),
            sigmaX=self.config.probability_blur_sigma,
            sigmaY=self.config.probability_blur_sigma,
        )
        offsets = np.linspace(
            -self.config.gradient_normal_radius,
            self.config.gradient_normal_radius,
            self.config.gradient_profile_samples,
            dtype=np.float32,
        )
        refined = points.copy()
        for index, point in enumerate(points):
            tangent = points[(index + 2) % len(points)] - points[(index - 2) % len(points)]
            magnitude = float(np.linalg.norm(tangent))
            if magnitude < 1e-6:
                continue
            tangent /= magnitude
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            samples_xy = point[None, :] + offsets[:, None] * normal[None, :]
            profile = cv2.remap(
                smooth,
                samples_xy[:, 0].reshape(-1, 1).astype(np.float32),
                samples_xy[:, 1].reshape(-1, 1).astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1)
            gradient = np.abs(np.gradient(profile, offsets))
            peak_index = int(np.argmax(gradient))
            peak_offset = float(offsets[peak_index])
            if 0 < peak_index < len(offsets) - 1:
                before, peak, after = gradient[peak_index - 1], gradient[peak_index], gradient[peak_index + 1]
                denominator = float(before - 2.0 * peak + after)
                if abs(denominator) > 1e-8:
                    fraction = 0.5 * float(before - after) / denominator
                    step = float(offsets[1] - offsets[0])
                    peak_offset += float(np.clip(fraction, -1.0, 1.0)) * step
            refined[index] = point + peak_offset * normal
        return refined

    @staticmethod
    def _weighted_centroid(probability: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
        weights = probability * marker_mask.astype(np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            raise InitializationError("Marker has zero probability mass.")
        rows, cols = np.indices(weights.shape, dtype=np.float32)
        return np.array(
            [float((weights * rows).sum() / total), float((weights * cols).sum() / total)],
            dtype=np.float32,
        )

    def _geometry(self, refined_contour_xy: np.ndarray, weighted_center_local_rc: np.ndarray) -> dict[str, object]:
        contour = refined_contour_xy.reshape(-1, 1, 2).astype(np.float32)
        area = float(abs(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            raise InitializationError("Initialized contour has invalid geometry.")
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-9))
        rect = cv2.minAreaRect(contour)
        (rect_cx, rect_cy), (rect_width, rect_height), _ = rect
        rectangle_area = max(float(rect_width * rect_height), 1e-8)
        rectangularity = float(np.clip(area / rectangle_area, 0.0, 1.0))
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        is_rectangle = (
            len(approx) == 4
            and cv2.isContourConvex(approx.astype(np.float32))
            and rectangularity >= self.config.rectangle_rectangularity_threshold
        )
        geometric_center_xy: Optional[np.ndarray] = None
        if is_rectangle:
            shape_model: ShapeModel = "rectangle"
            geometric_center_xy = np.array([rect_cx, rect_cy], dtype=np.float32)
        elif circularity >= self.config.circle_circularity_threshold:
            shape_model = "circle_like"
            if contour.shape[0] >= 5:
                geometric_center_xy = np.array(cv2.fitEllipse(contour)[0], dtype=np.float32)
        else:
            shape_model = "generic_compact"
        centre_xy = np.array([weighted_center_local_rc[1], weighted_center_local_rc[0]], dtype=np.float32)
        center_difference = (
            float(np.linalg.norm(centre_xy - geometric_center_xy))
            if geometric_center_xy is not None
            else float("nan")
        )
        return {
            "shape_model": shape_model,
            "geometric_center_xy": geometric_center_xy,
            "area": area,
            "width": float(rect_width),
            "height": float(rect_height),
            "circularity": circularity,
            "rectangularity": rectangularity,
            "center_difference": center_difference,
        }

    def _quality(
        self,
        *,
        probability: np.ndarray,
        component_mask: np.ndarray,
        ring_mask: np.ndarray,
        geometry: dict[str, object],
    ) -> dict[str, float]:
        inside = probability[component_mask > 0]
        ring = probability[ring_mask > 0]
        if inside.size == 0 or ring.size == 0:
            raise InitializationError("Cannot evaluate marker/background colour quality.")
        return {
            "median_inside_probability": float(np.median(inside)),
            "strong_inside_fraction": float(np.mean(inside > 0.70)),
            "mean_inside_probability": float(np.mean(inside)),
            "mean_ring_probability": float(np.mean(ring)),
            "probability_margin": float(np.mean(inside) - np.mean(ring)),
            "area": float(geometry["area"]),
            "circularity": float(geometry["circularity"]),
            "rectangularity": float(geometry["rectangularity"]),
            "center_difference": float(geometry["center_difference"]),
        }

    def _assert_quality(self, quality: dict[str, float]) -> None:
        if quality["median_inside_probability"] < self.config.min_inside_median_probability:
            raise InitializationError("Detected region is not strongly marker-coloured.")
        if quality["strong_inside_fraction"] < self.config.min_inside_strong_fraction:
            raise InitializationError("Marker interior colour is too heterogeneous.")
        if quality["probability_margin"] < self.config.min_probability_margin:
            raise InitializationError("Marker colour is insufficiently separated from local background.")

    @staticmethod
    def _confidence(quality: dict[str, float]) -> float:
        margin_score = float(np.clip(quality["probability_margin"] / 0.60, 0.0, 1.0))
        return float(
            np.clip(
                0.45 * quality["median_inside_probability"]
                + 0.30 * quality["strong_inside_fraction"]
                + 0.25 * margin_score,
                0.0,
                1.0,
            )
        )

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

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        rows, cols = np.nonzero(mask)
        if cols.size == 0:
            raise InitializationError("Cannot compute bounding box of empty marker mask.")
        x0, x1 = int(cols.min()), int(cols.max())
        y0, y1 = int(rows.min()), int(rows.max())
        return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    @staticmethod
    def _expand_bbox(
        bbox_xywh: tuple[int, int, int, int],
        margin: int,
        frame_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = bbox_xywh
        frame_height, frame_width = frame_hw
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(frame_width, x + width + margin)
        y1 = min(frame_height, y + height + margin)
        return (x0, y0, x1 - x0, y1 - y0)
