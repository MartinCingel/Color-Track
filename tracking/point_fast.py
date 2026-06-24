"""One-click direct colour-marker point tracker.

Drop-in replacement for ``tracking/point_fast.py``.

The tracker keeps the existing BaseTracker interface used by TrackerManager,
but replaces CSRT/Hessian tracking with direct measurement of a coloured compact
marker:

    click initialization -> learned Lab foreground/background model
    -> predicted small ROI -> connected colour candidate
    -> geometry/motion validation -> weighted subpixel centroid

Prediction is used only to choose the search ROI and reject impossible
candidates.  A predicted point is never exported as a measured position.

Performance design
------------------
This tracker explicitly requests cached CPU BGR frames from ``TrackerManager``.
The frame buffer retains the original decoded CPU frame alongside the GPU copy,
so direct colour localization no longer performs a GPU-to-CPU download.  Only
small predicted ROIs are converted to Lab and analysed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Literal, Optional, Tuple

import cv2
import numpy as np

from tracking.base_tracker import (
    BaseTracker,
    FrameResult,
    InitPreview,
    TrackerConfig,
    TrackerStatus,
)
from utils.profiling import profiling_enabled
from utils.tracker_debug import TrackerDebugFrame, TrackerDebugRecorder
from core.workspace_types import WorkspaceFrame


ShapeModel = Literal["generic_compact", "circle_like", "rectangle"]
PointMode = Literal["normal", "tiny"]
KalmanModel = Literal["constant_velocity", "constant_acceleration"]
ShapeValidation = Literal["off", "soft", "strict"]
MeasurementCenterMode = Literal["full_blob", "peak_window", "auto"]
TinyMosseMode = Literal["off", "assisted", "primary"]


class InitializationError(RuntimeError):
    """Raised when a marker cannot be reliably initialized from the click."""


@dataclass(frozen=True)
class _GaussianLabModel:
    mean: np.ndarray
    inv_cov: np.ndarray
    log_det_cov: float

    @classmethod
    def fit(
        cls,
        pixels: np.ndarray,
        *,
        sigma_floor: tuple[float, float, float],
        lightness_variance_scale: float = 4.0,
    ) -> "_GaussianLabModel":
        values = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
        if values.shape[0] == 0:
            raise InitializationError("Cannot learn a colour model from no pixels.")

        mean = np.median(values, axis=0).astype(np.float32)
        if values.shape[0] > 1:
            cov = np.cov(values.T).astype(np.float32)
        else:
            cov = np.zeros((3, 3), dtype=np.float32)

        floors = np.square(np.asarray(sigma_floor, dtype=np.float32))
        cov += np.diag(floors)
        cov[0, 0] *= float(lightness_variance_scale)
        cov += np.eye(3, dtype=np.float32) * 1e-4

        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            raise InitializationError("Colour covariance matrix is not positive definite.")
        return cls(
            mean=mean,
            inv_cov=np.linalg.inv(cov).astype(np.float32),
            log_det_cov=float(log_det),
        )

    def log_likelihood(self, lab: np.ndarray) -> np.ndarray:
        delta = lab.astype(np.float32) - self.mean
        d2 = np.einsum("...i,ij,...j->...", delta, self.inv_cov, delta)
        return (-0.5 * (d2 + self.log_det_cov)).astype(np.float32)

    def log_likelihood_weighted(self, lab: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # User LAB weights are an extra reliability control on top of the
        # learned covariance.  Constants cancel in the foreground-vs-background
        # likelihood ratio, so the determinant term is intentionally unchanged.
        delta = (lab.astype(np.float32) - self.mean) * weights.reshape(1, 1, 3)
        d2 = np.einsum("...i,ij,...j->...", delta, self.inv_cov, delta)
        return (-0.5 * (d2 + self.log_det_cov)).astype(np.float32)


@dataclass(frozen=True)
class _TinyLabDistanceModel:
    """Fixed-tolerance Lab distance model for very small colour features.

    It intentionally avoids covariance estimation because a 3-5 px feature may
    provide too few clean foreground pixels for a stable Gaussian covariance.
    """

    mean: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_pixels(
        cls,
        pixels: np.ndarray,
        *,
        weights: tuple[float, float, float] = (0.35, 1.0, 1.0),
    ) -> "_TinyLabDistanceModel":
        values = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
        if values.shape[0] == 0:
            raise InitializationError("Cannot learn tiny-feature colour from no pixels.")
        return cls(
            mean=np.median(values, axis=0).astype(np.float32),
            weights=np.asarray(weights, dtype=np.float32),
        )

    def log_likelihood(self, lab: np.ndarray) -> np.ndarray:
        delta = (lab.astype(np.float32) - self.mean) * self.weights
        d2 = np.sum(delta * delta, axis=-1)
        # Lab units are large; scale to keep the sigmoid useful.
        return (-0.5 * d2 / (14.0 * 14.0)).astype(np.float32)


@dataclass(frozen=True)
class _Settings:
    # Initialization
    init_half_size: int = 55
    seed_radius: int = 2
    border_width: int = 8
    init_probability_thresholds: tuple[float, ...] = (0.72,)
    init_min_component_area: int = 12
    init_max_component_fraction: float = 0.70
    init_max_area_ratio: float = 6.0
    init_growth_radius_px: float = 0.0  # 0 = auto from expected diameter
    init_morphology_close_iterations: int = 1
    normal_init_min_probability_margin: float = 0.08
    preliminary_fg_sigma_floor: tuple[float, float, float] = (8.0, 8.0, 8.0)
    preliminary_bg_sigma_floor: tuple[float, float, float] = (8.0, 6.0, 6.0)
    final_fg_sigma_floor: tuple[float, float, float] = (5.0, 4.0, 4.0)
    final_bg_sigma_floor: tuple[float, float, float] = (6.0, 5.0, 5.0)
    lightness_variance_scale: float = 4.0
    erode_iterations: int = 1
    background_ring_inner_dilate: int = 2
    background_ring_outer_dilate: int = 7

    # Boundary and initial geometry
    refine_boundary_on_initialize: bool = True
    refine_boundary_on_tracking: bool = False
    probability_blur_sigma: float = 0.8
    gradient_normal_radius: float = 2.5
    gradient_profile_samples: int = 17
    rectangle_rectangularity_threshold: float = 0.78
    circle_circularity_threshold: float = 0.73

    # Per-frame candidate extraction
    probability_thresholds: tuple[float, ...] = (0.72, 0.62, 0.52, 0.42)
    morphology_close_iterations: int = 1
    min_component_area_px: int = 8
    # Tracking gates are deliberately tolerant: colour/motion provide the
    # reliable lock, while exact boundary geometry may vary under blur/video
    # compression. Initialization remains stricter than normal tracking.
    min_inside_median_probability: float = 0.40
    min_inside_strong_fraction: float = 0.12
    min_probability_margin: float = 0.035
    strong_probability_threshold: float = 0.70

    # Geometry gates
    min_area_ratio: float = 0.40
    max_area_ratio: float = 2.25
    min_axis_size_ratio: float = 0.35
    max_axis_size_ratio: float = 2.75
    min_circle_circularity: float = 0.48
    min_rectangle_rectangularity: float = 0.60
    max_geometric_center_difference_px: float = 3.0

    # Search/motion
    normal_search_margin_px: int = 15
    miss_expansion_px: int = 16
    max_search_margin_px: int = 150
    max_prediction_error_px: float = 22.0
    miss_motion_allowance_px: float = 14.0
    accept_within_search_region: bool = False
    normal_recenter_on_edge: bool = True
    normal_recenter_edge_px: int = 3
    normal_recenter_extra_margin_px: int = 20
    normal_recenter_min_area_ratio: float = 0.25
    measurement_center_mode: MeasurementCenterMode = "full_blob"
    adaptive_prefilter_enabled: bool = False
    adaptive_prefilter_max_candidates: int = 3
    adaptive_prefilter_min_probability: float = 0.62
    adaptive_prefilter_support_threshold: float = 0.52
    lab_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    luminance_polarity_enabled: bool = True
    luminance_polarity_min_delta: float = 8.0
    luminance_polarity_tolerance: float = 35.0
    colour_diagnostics_enabled: bool = False
    ambiguity_score_margin: float = 0.04

    # Adaptation
    high_confidence_threshold: float = 0.78
    geometry_adaptation_alpha: float = 0.05
    adapt_geometry: bool = True

    # User-configurable reliability options
    point_mode: PointMode = "normal"
    sample_size: int = 5
    expected_diameter_px: float = 20.0
    kalman_enabled: bool = False
    kalman_model: KalmanModel = "constant_velocity"
    measurement_noise_px: float = 0.7
    process_noise_px: float = 0.5
    kalman_gate_sigma: float = 4.0
    min_search_radius_px: int = 12
    max_dynamic_search_radius_px: int = 80
    shape_validation: ShapeValidation = "soft"

    # Tiny-feature mode
    tiny_peak_threshold: float = 0.55
    tiny_min_contrast: float = 0.04
    tiny_init_peak_threshold: float = 0.40
    tiny_init_contrast_threshold: float = 8.0
    tiny_direction_min_cosine: float = 0.20
    tiny_update_confidence_threshold: float = 0.65
    tiny_mosse_mode: TinyMosseMode = "off"
    tiny_mosse_window_scale: float = 4.0
    tiny_mosse_learning_rate: float = 0.05
    tiny_mosse_psr_threshold: float = 6.0
    tiny_mosse_peak_margin_threshold: float = 0.15
    normal_mosse_mode: str = "off"
    mosse_scale_adaptation: bool = False
    mosse_scale_levels: int = 7
    mosse_scale_step: float = 1.02
    mosse_scale_max_step: float = 0.05

    # Debug/reporting only: for UNCERTAIN/LOST frames, optionally store a
    # diagnostic fallback coordinate in debug reports. It never changes the
    # exported FrameResult.center, which remains None when not measured.
    debug_uncertain_center_policy: str = 'none'  # 'none', 'peak', 'kalman'


@dataclass(frozen=True)
class _Initialization:
    center_rc: np.ndarray
    geometric_center_rc: Optional[np.ndarray]
    shape_model: ShapeModel
    foreground_model: _GaussianLabModel | _TinyLabDistanceModel
    background_model: _GaussianLabModel | _TinyLabDistanceModel
    tight_bbox_xywh: tuple[int, int, int, int]
    search_bbox_xywh: tuple[int, int, int, int]
    contour_xy_global: np.ndarray
    expected_area: float
    expected_width: float
    expected_height: float
    circularity: float
    rectangularity: float
    confidence: float
    tiny_reference_delta_lab: Optional[np.ndarray] = None
    tiny_init_contrast: float = 0.0
    tiny_seed_bbox_xywh: Optional[tuple[int, int, int, int]] = None
    tiny_ring_bbox_xywh: Optional[tuple[int, int, int, int]] = None
    preview_rgba: Optional[np.ndarray] = None
    foreground_l_mean: float = 0.0
    background_l_mean: float = 0.0
    luminance_polarity: str = "neutral"
    threshold_adjustments: tuple[dict[str, object], ...] = ()
    init_roi_xywh: Optional[tuple[int, int, int, int]] = None
    init_bgr_roi: Optional[np.ndarray] = None
    init_probability_roi: Optional[np.ndarray] = None
    init_mask_roi: Optional[np.ndarray] = None
    init_click_local_rc: Optional[tuple[int, int]] = None
    init_metrics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _InitEditContext:
    roi_lab: np.ndarray
    roi_bgr: np.ndarray
    roi_origin_xy: tuple[int, int]
    click_local_rc: tuple[int, int]
    frame_index: Optional[int]
    full_frame_hw: tuple[int, int]
    point_mode: PointMode


@dataclass
class _MosseState:
    A: np.ndarray
    B: np.ndarray
    desired_fft: np.ndarray
    hann: np.ndarray
    window_size: int
    last_psr: float = 0.0
    last_peak_margin: float = 0.0
    last_peak_response: float = 0.0
    scale_factor: float = 1.0


@dataclass
class _TrackingState:
    initialization: _Initialization
    expected_area: float
    expected_width: float
    expected_height: float
    last_valid_center_rc: np.ndarray
    previous_valid_center_rc: Optional[np.ndarray]
    last_valid_frame_index: Optional[int]
    kalman_x: Optional[np.ndarray] = None
    kalman_P: Optional[np.ndarray] = None
    kalman_frame_index: Optional[int] = None
    consecutive_misses: int = 0
    geometry_history: deque = field(default_factory=lambda: deque(maxlen=5))
    center_history: deque = field(default_factory=lambda: deque(maxlen=5))
    mosse: Optional[_MosseState] = None
    validation_samples: list[np.ndarray] = field(default_factory=list)
    validation_seen: int = 0
    validation_median: Optional[np.ndarray] = None
    validation_scale: Optional[np.ndarray] = None
    validation_dirty: bool = True


@dataclass(frozen=True)
class _Candidate:
    center_rc: np.ndarray
    tight_bbox_xywh: tuple[int, int, int, int]
    contour_xy_global: np.ndarray
    confidence: float
    colour_score: float
    shape_score: float
    motion_score: float
    area: float
    width: float
    height: float
    circularity: float
    rectangularity: float
    centre_difference: float
    angle_deg: float = 0.0
    center_mode_used: str = "full_blob"
    full_blob_center_rc: Optional[np.ndarray] = None
    peak_window_center_rc: Optional[np.ndarray] = None
    peak_rc: Optional[np.ndarray] = None
    peak_score: float = 0.0
    centroid_peak_distance_px: float = 0.0
    blob_elongation: float = 1.0
    centroid_window_xywh: Optional[tuple[int, int, int, int]] = None
    adaptive_source: str = "classical"
    candidate_l_mean: float = 0.0
    candidate_l_median: float = 0.0
    local_background_l_mean: float = 0.0
    luminance_polarity: str = "neutral"
    luminance_polarity_pass: bool = True
    luminance_error: float = 0.0
    scale_factor: float = 1.0


class PointFastTracker(BaseTracker):
    """Direct Lab-colour point marker tracker compatible with TrackerManager."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)

        point_mode = str(getattr(config, "point_mode", "normal"))
        if point_mode not in ("normal", "tiny"):
            point_mode = "normal"

        expected_diameter = max(1.0, float(getattr(config, "expected_diameter", config.sigma)))
        sample_size = int(getattr(config, "sample_size", 5))
        sample_size = max(1, min(31, sample_size))
        if sample_size % 2 == 0:
            sample_size += 1

        kalman_model = str(getattr(config, "kalman_model", "constant_velocity"))
        if kalman_model not in ("constant_velocity", "constant_acceleration"):
            kalman_model = "constant_velocity"

        shape_validation = str(getattr(config, "shape_validation", "soft"))
        if shape_validation not in ("off", "soft", "strict"):
            shape_validation = "soft"
        if point_mode == "tiny":
            shape_validation = "off"

        measurement_center_mode = str(getattr(config, "measurement_center_mode", "full_blob"))
        if measurement_center_mode not in ("full_blob", "peak_window", "auto"):
            measurement_center_mode = "full_blob"
        adaptive_prefilter_enabled = bool(getattr(config, "adaptive_prefilter_enabled", False))
        lab_weights = (
            max(0.10, min(5.0, float(getattr(config, "lab_l_weight", 1.0)))),
            max(0.10, min(5.0, float(getattr(config, "lab_a_weight", 1.0)))),
            max(0.10, min(5.0, float(getattr(config, "lab_b_weight", 1.0)))),
        )
        luminance_polarity_enabled = bool(getattr(config, "luminance_polarity_enabled", True))
        luminance_polarity_min_delta = max(0.0, float(getattr(config, "luminance_polarity_min_delta", 8.0)))
        luminance_polarity_tolerance = max(0.0, float(getattr(config, "luminance_polarity_tolerance", 35.0)))
        colour_diagnostics_enabled = bool(getattr(config, "colour_diagnostics_enabled", False))

        # Old-style normal-marker ROI controls.  When Kalman is disabled,
        # normal mode uses the reliable pre-Kalman search rule exactly:
        #   margin = min(max_search_margin, normal_search_margin + misses * miss_expansion)
        # These are exposed in the UI so normal-marker tracking can be tuned
        # without accidentally invoking Kalman/covariance ROI logic.
        normal_margin = max(1, int(getattr(config, "normal_search_margin", 15)))
        miss_expansion = max(0, int(getattr(config, "miss_expansion", 16)))
        max_search_margin = max(normal_margin, int(getattr(config, "max_search_margin", 150)))
        max_motion = max(1.0, float(getattr(config, "max_prediction_error", 22.0)))
        miss_motion_allowance = max(0.0, float(getattr(config, "miss_motion_allowance", 14.0)))
        recenter_enabled = bool(getattr(config, "normal_recenter_on_edge", True))
        recenter_edge_px = max(0, int(getattr(config, "normal_recenter_edge_px", 3)))
        recenter_extra_margin = max(0, int(getattr(config, "normal_recenter_extra_margin", 20)))
        recenter_min_area_ratio = max(0.0, float(getattr(config, "normal_recenter_min_area_ratio", 0.25)))
        uncertain_policy = str(getattr(config, 'debug_uncertain_center_policy', 'none'))
        if uncertain_policy not in ('none', 'peak', 'kalman'):
            uncertain_policy = 'none'
        tiny_mosse_mode = str(getattr(config, "tiny_mosse_mode", "off"))
        if tiny_mosse_mode not in ("off", "assisted", "primary"):
            tiny_mosse_mode = "off"
        normal_mosse_mode = str(getattr(config, "normal_mosse_mode", "off"))
        if normal_mosse_mode not in ("off", "assisted", "primary"):
            normal_mosse_mode = "off"

        self._settings = _Settings(
            init_half_size=max(30, int(round(expected_diameter * 6.0))),
            seed_radius=max(0, sample_size // 2),
            init_probability_thresholds=(float(getattr(config, "normal_init_threshold", 0.72)),),
            normal_init_min_probability_margin=float(getattr(config, "normal_init_min_probability_margin", 0.08)),
            init_max_area_ratio=max(0.25, float(getattr(config, "normal_init_max_area_ratio", 6.0))),
            init_growth_radius_px=max(0.0, float(getattr(config, "normal_init_growth_radius", 0.0))),
            init_morphology_close_iterations=max(0, int(getattr(config, "normal_init_close_iterations", 1))),
            normal_search_margin_px=normal_margin,
            miss_expansion_px=miss_expansion,
            max_search_margin_px=max_search_margin,
            max_prediction_error_px=max_motion,
            miss_motion_allowance_px=miss_motion_allowance,
            accept_within_search_region=bool(getattr(config, "accept_within_search_region", False)),
            normal_recenter_on_edge=recenter_enabled,
            normal_recenter_edge_px=recenter_edge_px,
            normal_recenter_extra_margin_px=recenter_extra_margin,
            normal_recenter_min_area_ratio=recenter_min_area_ratio,
            measurement_center_mode=measurement_center_mode,  # type: ignore[arg-type]
            adaptive_prefilter_enabled=adaptive_prefilter_enabled,
            lab_weights=lab_weights,
            luminance_polarity_enabled=luminance_polarity_enabled,
            luminance_polarity_min_delta=luminance_polarity_min_delta,
            luminance_polarity_tolerance=luminance_polarity_tolerance,
            colour_diagnostics_enabled=colour_diagnostics_enabled,
            refine_boundary_on_tracking=False,
            point_mode=point_mode,  # type: ignore[arg-type]
            sample_size=sample_size,
            expected_diameter_px=expected_diameter,
            kalman_enabled=bool(getattr(config, "kalman_enabled", False)),
            kalman_model=kalman_model,  # type: ignore[arg-type]
            measurement_noise_px=max(0.05, float(getattr(config, "measurement_noise", 0.7))),
            process_noise_px=max(0.001, float(getattr(config, "process_noise", 0.5))),
            kalman_gate_sigma=max(1.0, float(getattr(config, "kalman_gate_sigma", 4.0))),
            min_search_radius_px=max(1, int(getattr(config, "min_search_radius", 12))),
            max_dynamic_search_radius_px=max(4, int(getattr(config, "max_search_radius", 80))),
            shape_validation=shape_validation,  # type: ignore[arg-type]
            tiny_peak_threshold=float(getattr(config, "tiny_peak_threshold", 0.55)),
            tiny_min_contrast=float(getattr(config, "tiny_contrast_threshold", 0.04)),
            tiny_init_peak_threshold=float(getattr(config, "tiny_init_peak_threshold", 0.40)),
            tiny_init_contrast_threshold=float(getattr(config, "tiny_init_contrast_threshold", 8.0)),
            tiny_direction_min_cosine=float(getattr(config, "tiny_direction_min_cosine", 0.20)),
            tiny_update_confidence_threshold=float(getattr(config, "tiny_update_confidence_threshold", 0.65)),
            tiny_mosse_mode=tiny_mosse_mode,  # type: ignore[arg-type]
            tiny_mosse_window_scale=float(np.clip(float(getattr(config, "tiny_mosse_window_scale", 4.0)), 2.0, 10.0)),
            tiny_mosse_learning_rate=float(np.clip(float(getattr(config, "tiny_mosse_learning_rate", 0.05)), 0.0, 0.30)),
            tiny_mosse_psr_threshold=float(np.clip(float(getattr(config, "tiny_mosse_psr_threshold", 6.0)), 0.0, 30.0)),
            tiny_mosse_peak_margin_threshold=float(np.clip(float(getattr(config, "tiny_mosse_peak_margin_threshold", 0.15)), 0.0, 1.0)),
            normal_mosse_mode=normal_mosse_mode,
            mosse_scale_adaptation=bool(getattr(config, "mosse_scale_adaptation", False)),
            debug_uncertain_center_policy=uncertain_policy,
            min_component_area_px=1 if point_mode == "tiny" else 8,
            morphology_close_iterations=0 if point_mode == "tiny" else 1,
        )
        self._state: Optional[_TrackingState] = None
        self._last_rejection_counts: dict[str, int] = {}
        self._last_failure_reasons: list[str] = []
        self._last_peak_center_rc: Optional[np.ndarray] = None
        self._last_predicted_center_rc: Optional[np.ndarray] = None
        self._last_threshold_adjustments: list[dict[str, object]] = []
        self._pending_mosse_update: Optional[tuple[np.ndarray, tuple[int, int], np.ndarray]] = None
        self._last_init_edit_context: Optional[_InitEditContext] = None
        self._preset_validation_samples = list(getattr(config, 'learned_validation_samples', None) or [])

        # Debug recording is separate from profiling and is also opt-in. It
        # keeps only a rolling history and writes reports when lock degrades.
        self._debug_recorder = TrackerDebugRecorder(
            enabled=bool(getattr(config, "debug_enabled", False)),
            history_frames=int(getattr(config, "debug_history_frames", 120)),
            tracker_uid=self.uid,
            tracker_name=self.name,
        )
        self._debug_current: dict[str, Any] = {}
        self._debug_reports_pending: list[str] = []
        self._debug_reported_uncertain = False
        self._debug_reported_lost = False

        # Detailed stage timings are opt-in because recording one metrics row
        # per tracker-frame adds overhead and memory use. Enable with
        # TRACKING_PROFILE=1 before starting the application.
        self._profiling_enabled = profiling_enabled()
        self._last_profile_metrics: dict[str, Any] = {}
        self._profile_accumulators: dict[str, float] = {}
        self._last_candidate_scan_metrics: dict[str, Any] = {}
        self._last_failure_diagnostics: dict[str, Any] = {}

    @property
    def requires_cpu_frame(self) -> bool:
        """Request the cached decoded CPU BGR frame from TrackerManager."""
        return True

    @property
    def accepts_workspace_frame(self) -> bool:
        """Accept cropped prepared-cache frames with full-frame coordinate mapping."""
        return True

    @property
    def last_rejection_counts(self) -> dict[str, int]:
        """Diagnostic counts from the most recent candidate scan."""
        return dict(self._last_rejection_counts)

    @property
    def last_profile_metrics(self) -> dict[str, Any]:
        """Most recent opt-in per-frame metrics consumed by BatchWorker."""
        return dict(self._last_profile_metrics)

    @property
    def latest_debug_report_dir(self) -> Optional[str]:
        """Most recently generated debug report path, if any."""
        return self._debug_reports_pending[-1] if self._debug_reports_pending else None

    def pop_debug_reports(self) -> list[str]:
        """Return and clear newly generated debug report directories."""
        reports = list(self._debug_reports_pending)
        self._debug_reports_pending.clear()
        return reports

    def _set_failure_reasons(self, *reasons: str) -> None:
        """Set unique human-readable debug failure reasons for this frame."""
        unique: list[str] = []
        for reason in reasons:
            if reason and reason not in unique:
                unique.append(str(reason))
        self._last_failure_reasons = unique

    def _add_failure_reason(self, reason: str) -> None:
        if reason and reason not in self._last_failure_reasons:
            self._last_failure_reasons.append(str(reason))

    def _failure_reasons_from_counts(self) -> list[str]:
        if self._last_failure_reasons:
            return list(self._last_failure_reasons)
        reasons: list[str] = []
        motion_reason = (
            "outside_kalman_gate" if self._settings.kalman_enabled else "outside_motion_limit"
        )
        mapping = {
            "workspace_boundary": "outside_workspace",
            "adaptive_workspace": "adaptive_prefilter_outside_workspace",
            "adaptive_no_seed": "adaptive_prefilter_no_seed",
            "adaptive_no_supported_seed": "adaptive_prefilter_no_supported_seed",
            "adaptive_no_full_candidate": "adaptive_prefilter_no_full_candidate",
            "small_component": "no_peak_found",
            "colour": "colour_rejected",
            "local_contrast": "contrast_below_threshold",
            "direction": "contrast_direction_mismatch",
            "weak_update": "weak_measurement_not_used_for_kalman",
            "size": "size_rejected",
            "motion": motion_reason,
            "ambiguity": "ambiguous_candidate",
        }
        for key, value in self._last_rejection_counts.items():
            if int(value) > 0:
                reason = mapping.get(key, key)
                if reason not in reasons:
                    reasons.append(reason)
        return reasons or ["no_candidate"]

    @staticmethod
    def _jsonable_debug_value(value: Any) -> Any:
        """Convert numpy/scalar values to review-table friendly Python values."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [PointFastTracker._jsonable_debug_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): PointFastTracker._jsonable_debug_value(v) for k, v in value.items()}
        return value

    def _set_failure_diagnostic(self, key: str, value: Any) -> None:
        self._last_failure_diagnostics[str(key)] = self._jsonable_debug_value(value)

    def _failure_diagnostics(self) -> dict[str, Any]:
        values = dict(self._last_failure_diagnostics)
        values.update({
            f"reject_{key}": int(value)
            for key, value in self._last_rejection_counts.items()
            if int(value) > 0
        })
        for key, value in self._last_candidate_scan_metrics.items():
            if key not in values:
                values[key] = self._jsonable_debug_value(value)
        if self._last_peak_center_rc is not None and "colour_peak_rc" not in values:
            values["colour_peak_rc"] = self._last_peak_center_rc.astype(float).tolist()
        if self._last_predicted_center_rc is not None and "kalman_mean_rc" not in values:
            values["kalman_mean_rc"] = self._last_predicted_center_rc.astype(float).tolist()
        return values

    def _profile_add(self, key: str, elapsed_seconds: float) -> None:
        if self._profiling_enabled:
            self._profile_accumulators[key] = (
                self._profile_accumulators.get(key, 0.0)
                + elapsed_seconds * 1000.0
            )

    def _debug_begin_frame(
        self,
        *,
        frame_index: int,
        roi_bgr: Optional[np.ndarray],
        probability: Optional[np.ndarray],
        search_bbox: Optional[tuple[int, int, int, int]],
        predicted_center: Optional[np.ndarray],
    ) -> None:
        """Start a per-frame debug record when debug mode is enabled."""
        if not self._debug_recorder.enabled:
            return
        self._debug_current = {
            "frame_index": int(frame_index),
            "roi_bgr": None if roi_bgr is None else np.ascontiguousarray(roi_bgr.copy()),
            "likelihood": None if probability is None else np.ascontiguousarray(probability.copy()),
            "search_bbox_xywh": None if search_bbox is None else [int(v) for v in search_bbox],
            "predicted_rc": None if predicted_center is None else [float(predicted_center[0]), float(predicted_center[1])],
            "kalman_gate_radius": float(self._kalman_gate_radius()) if self._state is not None else None,
        }

    def _debug_finalize_frame(
        self,
        *,
        result: FrameResult,
        candidate: Optional[_Candidate] = None,
    ) -> None:
        """Append one debug record and export if this frame degraded lock."""
        if not self._debug_recorder.enabled or self._state is None:
            return
        current = dict(self._debug_current)
        measured_rc = None
        if result.center is not None:
            measured_rc = [float(result.center[0]), float(result.center[1])]
        elif candidate is not None:
            measured_rc = [float(candidate.center_rc[0]), float(candidate.center_rc[1])]

        predicted_rc = current.get("predicted_rc")
        innovation = None
        if predicted_rc is not None and measured_rc is not None:
            innovation = float(np.linalg.norm(np.asarray(measured_rc) - np.asarray(predicted_rc)))

        kalman_state = None
        kalman_sigma = None
        if self._state.kalman_x is not None:
            kalman_state = [float(v) for v in self._state.kalman_x.ravel().tolist()]
        if self._state.kalman_P is not None:
            kalman_sigma = float(np.sqrt(max(self._state.kalman_P[0, 0], self._state.kalman_P[1, 1], 1e-9)))

        scan = dict(self._last_candidate_scan_metrics)
        reasons = [] if result.status == TrackerStatus.LOCKED else self._failure_reasons_from_counts()
        if result.status == TrackerStatus.LOCKED:
            self._debug_reported_uncertain = False
            self._debug_reported_lost = False

        uncertain_export_rc = None
        uncertain_export_source = "none"
        if result.status != TrackerStatus.LOCKED:
            policy = self._settings.debug_uncertain_center_policy
            if policy == "peak" and current.get("peak_rc") is not None:
                uncertain_export_rc = current.get("peak_rc")
                uncertain_export_source = "colour_peak"
            elif policy == "kalman" and predicted_rc is not None:
                uncertain_export_rc = predicted_rc
                uncertain_export_source = "kalman_mean"

        record = TrackerDebugFrame(
            frame_index=int(result.frame_index),
            status=result.status.value,
            point_mode=self._settings.point_mode,
            failure_reasons=reasons,
            predicted_rc=predicted_rc,
            measured_rc=measured_rc,
            kalman_state=kalman_state,
            kalman_position_sigma=kalman_sigma,
            kalman_gate_radius=current.get("kalman_gate_radius"),
            innovation_distance=innovation,
            confidence=float(candidate.confidence if candidate is not None else result.hessian_score),
            colour_score=None if candidate is None else float(candidate.colour_score),
            shape_score=None if candidate is None else float(candidate.shape_score),
            motion_score=None if candidate is None else float(candidate.motion_score),
            peak_score=current.get("peak_score"),
            contrast_score=current.get("contrast_score"),
            successful_threshold=scan.get("successful_threshold"),
            search_bbox_xywh=current.get("search_bbox_xywh"),
            centroid_window_xywh=current.get("centroid_window_xywh"),
            peak_rc=current.get("peak_rc"),
            uncertain_export_rc=uncertain_export_rc,
            uncertain_export_source=uncertain_export_source,
            candidate_count=int(scan.get("components_scored", scan.get("accepted_candidate_count", 0)) or 0),
            accepted=bool(result.status == TrackerStatus.LOCKED and result.center is not None),
            consecutive_misses=int(self._state.consecutive_misses),
            rejection_counts=dict(self._last_rejection_counts),
            extra={
                **{k: v for k, v in scan.items() if k not in {"successful_threshold"}},
                **{
                    k: current.get(k)
                    for k in ("lab_contrast", "direction_cosine")
                    if current.get(k) is not None
                },
            },
            roi_bgr=current.get("roi_bgr"),
            likelihood=current.get("likelihood"),
        )
        self._debug_recorder.append(record)

        report_dir = None
        if result.status == TrackerStatus.UNCERTAIN and not self._debug_reported_uncertain:
            self._debug_reported_uncertain = True
            report_dir = self._debug_recorder.export(
                event="first_uncertain", frame_index=int(result.frame_index)
            )
        elif result.status == TrackerStatus.LOST and not self._debug_reported_lost:
            self._debug_reported_lost = True
            report_dir = self._debug_recorder.export(
                event="lost", frame_index=int(result.frame_index)
            )
        if report_dir is not None:
            self._debug_reports_pending.append(str(report_dir))

    def _finish_frame_profile(
        self,
        *,
        frame_index: int,
        status: TrackerStatus,
        search_bbox: Optional[tuple[int, int, int, int]],
        confidence: float,
        started_at: float,
    ) -> None:
        if not self._profiling_enabled:
            return
        metrics: dict[str, Any] = dict(self._profile_accumulators)
        metrics.update(self._last_candidate_scan_metrics)
        metrics.update({f"reject_{key}": int(value) for key, value in self._last_rejection_counts.items()})
        metrics.update({
            "point_fast_frame_index": int(frame_index),
            "point_fast_status": status.value,
            "confidence": float(confidence),
            "misses_after_frame": int(self._state.consecutive_misses if self._state is not None else 0),
            "point_fast_total_ms": float((perf_counter() - started_at) * 1000.0),
        })
        if search_bbox is not None:
            metrics.update({
                "roi_width": int(search_bbox[2]),
                "roi_height": int(search_bbox[3]),
                "roi_pixels": int(search_bbox[2] * search_bbox[3]),
            })
        self._last_profile_metrics = metrics

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        frame_bgr: np.ndarray | WorkspaceFrame,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> InitPreview:
        """Initialize from one click; ``roi`` is accepted but intentionally unused."""
        _ = roi
        initialization = self._initialize_from_click(
            frame_bgr, seed_row, seed_col, frame_index=None
        )
        self._reset_to_locked()
        return self._preview(initialization)

    def reinitialize(
        self,
        frame_bgr: np.ndarray | WorkspaceFrame,
        seed_row: float,
        seed_col: float,
        frame_index: int,
    ) -> InitPreview:
        initialization = self._initialize_from_click(
            frame_bgr, seed_row, seed_col, frame_index=int(frame_index)
        )
        self._reset_to_locked()
        return self._preview(initialization)

    def preview_initialization_edit(
        self,
        parameters: dict[str, object],
        scissors_cuts: list[tuple[float, ...]],
    ) -> InitPreview:
        initialization = self._normal_initialization_from_edit_context(
            parameters=parameters,
            scissors_cuts=scissors_cuts,
            commit=False,
        )
        return self._preview(initialization)

    def apply_initialization_edit(
        self,
        parameters: dict[str, object],
        scissors_cuts: list[tuple[float, ...]],
    ) -> InitPreview:
        initialization = self._normal_initialization_from_edit_context(
            parameters=parameters,
            scissors_cuts=scissors_cuts,
            commit=True,
        )
        self._reset_to_locked()
        return self._preview(initialization)

    def process_frame(
        self,
        frame_bgr: np.ndarray | WorkspaceFrame,
        frame_index: int,
    ) -> FrameResult:
        """Measure one frame, accepting either a full BGR image or cached workspace view."""
        profiling = self._profiling_enabled
        total_start = perf_counter() if profiling else 0.0
        if profiling:
            self._profile_accumulators = {}
            self._last_candidate_scan_metrics = {}

        if self._state is None:
            self.status = TrackerStatus.LOST
            result = FrameResult(frame_index, self.status, center=None, hessian_score=0.0)
            self._finish_frame_profile(
                frame_index=frame_index, status=result.status, search_bbox=None,
                confidence=0.0, started_at=total_start,
            )
            return result

        image_bgr, frame_origin_xy, full_frame_hw = self._frame_view_parts(frame_bgr)

        stage_start = perf_counter() if profiling else 0.0
        predicted_center = self._predict_center(frame_index)
        self._last_predicted_center_rc = predicted_center.astype(np.float32).copy()
        self._last_peak_center_rc = None
        self._last_failure_diagnostics = {
            "predicted_rc": predicted_center.astype(float).tolist(),
            "kalman_gate_radius": float(self._kalman_gate_radius()),
            "point_mode": self._settings.point_mode,
            "kalman_model": self._settings.kalman_model,
            "measurement_noise_sigma": float(self._settings.measurement_noise_px),
            "process_noise_sigma": float(self._settings.process_noise_px),
        }
        requested_search_bbox = self._search_bbox(full_frame_hw, predicted_center)
        self._set_failure_diagnostic("requested_search_bbox_xywh", requested_search_bbox)
        mapped_bbox = self._global_bbox_to_view_bbox_clipped(
            requested_search_bbox,
            frame_origin_xy,
            image_bgr.shape[:2],
            anchor_center_rc=predicted_center,
        )
        if mapped_bbox is None:
            # Prepared workspace does not contain the predicted centre. If only
            # the ROI edge is outside, the clipped mapper would allow tracking.
            self._last_rejection_counts = {"workspace_boundary": 1}
            self._set_failure_reasons("outside_workspace")
            self._set_failure_diagnostic("workspace_origin_xy", frame_origin_xy)
            self._set_failure_diagnostic("workspace_size_hw", image_bgr.shape[:2])
            result = self._measurement_failure(frame_index)
            self._debug_begin_frame(
                frame_index=frame_index,
                roi_bgr=None,
                probability=None,
                search_bbox=requested_search_bbox,
                predicted_center=predicted_center,
            )
            self._debug_finalize_frame(result=result, candidate=None)
            self._finish_frame_profile(
                frame_index=frame_index, status=result.status, search_bbox=requested_search_bbox,
                confidence=0.0, started_at=total_start,
            )
            return result

        local_bbox, search_bbox = mapped_bbox
        self._set_failure_diagnostic("search_bbox_xywh", search_bbox)
        self._set_failure_diagnostic("search_bbox_was_clipped", bool(search_bbox != requested_search_bbox))

        roi_bgr = self._crop_bgr_roi(image_bgr, local_bbox)
        if profiling:
            self._profile_add("predict_crop_ms", perf_counter() - stage_start)

        stage_start = perf_counter() if profiling else 0.0
        roi_lab = self._to_lab(roi_bgr)
        if profiling:
            self._profile_add("lab_conversion_ms", perf_counter() - stage_start)

        stage_start = perf_counter() if profiling else 0.0
        probability = self._foreground_probability(
            roi_lab,
            self._state.initialization.foreground_model,
            self._state.initialization.background_model,
        )
        if probability.size and (self._debug_recorder.enabled or profiling):
            self._set_failure_diagnostic("probability_max", float(np.max(probability)))
            self._set_failure_diagnostic("probability_mean", float(np.mean(probability)))
        if profiling:
            self._profile_add("probability_map_ms", perf_counter() - stage_start)

        self._debug_begin_frame(
            frame_index=frame_index,
            roi_bgr=roi_bgr,
            probability=probability,
            search_bbox=search_bbox,
            predicted_center=predicted_center,
        )

        stage_start = perf_counter() if profiling else 0.0
        adaptive_uncertain_first = False
        if self._settings.point_mode == "tiny":
            candidates = self._tiny_candidates(
                probability=probability,
                roi_lab=roi_lab,
                roi_origin_xy=(search_bbox[0], search_bbox[1]),
                predicted_center_rc=predicted_center,
            )
        else:
            candidates = self._normal_candidates_with_mosse(
                probability=probability,
                roi_lab=roi_lab,
                # Candidate coordinates must remain in original full-frame space.
                roi_origin_xy=(search_bbox[0], search_bbox[1]),
                predicted_center_rc=predicted_center,
            )
        if profiling:
            self._profile_add("candidate_scan_total_ms", perf_counter() - stage_start)

        decision_start = perf_counter() if profiling else 0.0
        if not candidates:
            adaptive = self._adaptive_prefilter_candidates(
                image_bgr=image_bgr,
                frame_origin_xy=frame_origin_xy,
                full_frame_hw=full_frame_hw,
                predicted_center=predicted_center,
                normal_search_bbox=search_bbox,
            )
            if adaptive is not None:
                candidates, adaptive_uncertain_first = adaptive

        if not candidates:
            result = self._measurement_failure(frame_index)
            self._debug_finalize_frame(result=result, candidate=None)
            if profiling:
                self._profile_add("decision_state_ms", perf_counter() - decision_start)
            self._finish_frame_profile(
                frame_index=frame_index, status=result.status, search_bbox=search_bbox,
                confidence=0.0, started_at=total_start,
            )
            return result

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        best = candidates[0]

        recentered = None
        if best.adaptive_source == "classical":
            recentered = self._try_normal_blob_recenter(
                image_bgr=image_bgr,
                full_frame_hw=full_frame_hw,
                frame_origin_xy=frame_origin_xy,
                current_search_bbox=search_bbox,
                candidate=best,
            )
        if recentered:
            candidates = recentered
            candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
            best = candidates[0]

        if (
            len(candidates) > 1
            and best.confidence - candidates[1].confidence
            < self._settings.ambiguity_score_margin
        ):
            self._last_rejection_counts["ambiguity"] = (
                self._last_rejection_counts.get("ambiguity", 0) + 1
            )
            self._add_failure_reason("ambiguous_candidate")
            result = self._measurement_failure(frame_index)
            self._debug_finalize_frame(result=result, candidate=None)
            if profiling:
                self._profile_add("decision_state_ms", perf_counter() - decision_start)
            self._finish_frame_profile(
                frame_index=frame_index, status=result.status, search_bbox=search_bbox,
                confidence=0.0, started_at=total_start,
            )
            return result

        if not self._passes_learned_validation(best):
            self._last_rejection_counts["learned_validation"] = (
                self._last_rejection_counts.get("learned_validation", 0) + 1
            )
            self._add_failure_reason("learned_feature_profile_rejected")
            result = self._measurement_failure(frame_index)
            self._debug_finalize_frame(result=result, candidate=None)
            if profiling:
                self._profile_add("decision_state_ms", perf_counter() - decision_start)
            self._finish_frame_profile(
                frame_index=frame_index, status=result.status, search_bbox=search_bbox,
                confidence=0.0, started_at=total_start,
            )
            return result

        self._accept_candidate(best, frame_index)
        if adaptive_uncertain_first:
            # Do not let a one-frame reacquisition jump create a huge velocity
            # extrapolation.  Use the recovered centre as a tentative anchor and
            # require the next frame to confirm before exporting LOCKED data.
            if self._state is not None:
                self._state.previous_valid_center_rc = None
            self.status = TrackerStatus.UNCERTAIN
            self._uncertain_count = max(1, self._uncertain_count)
            self._add_failure_reason("adaptive_reacquisition_pending_confirmation")
        else:
            self.status = TrackerStatus.LOCKED
            self._uncertain_count = 0

        # ``FrameResult`` has a legacy field named hessian_score. In this new
        # tracker it carries accepted measurement confidence [0, 1].
        locked_diagnostics = {
            "confidence": float(best.confidence),
            "colour_score": float(best.colour_score),
            "motion_score": float(best.motion_score),
            "adaptive_source": str(best.adaptive_source),
            "failure_reasons": list(self._last_failure_reasons),
        }
        if self._settings.point_mode == "tiny":
            locked_diagnostics.update({
                "mosse_mode": str(self._settings.tiny_mosse_mode),
                "mosse_used": bool(best.adaptive_source == "mosse_probability"),
                "mosse_psr": self._last_failure_diagnostics.get("mosse_psr", ""),
                "mosse_peak_margin": self._last_failure_diagnostics.get("mosse_peak_margin", ""),
                "mosse_update_applied": self._last_failure_diagnostics.get("mosse_update_applied", False),
            })
        if self._settings.colour_diagnostics_enabled:
            locked_diagnostics.update({
                "candidate_l_mean": float(best.candidate_l_mean),
                "candidate_l_median": float(best.candidate_l_median),
                "local_background_l_mean": float(best.local_background_l_mean),
                "init_foreground_l_mean": float(self._state.initialization.foreground_l_mean),
                "init_background_l_mean": float(self._state.initialization.background_l_mean),
                "luminance_polarity": str(best.luminance_polarity),
                "luminance_polarity_pass": bool(best.luminance_polarity_pass),
                "luminance_error": float(best.luminance_error),
                "lab_l_weight": float(self._settings.lab_weights[0]),
                "lab_a_weight": float(self._settings.lab_weights[1]),
                "lab_b_weight": float(self._settings.lab_weights[2]),
            })
        result = FrameResult(
            frame_index=frame_index,
            status=self.status,
            center=None if adaptive_uncertain_first else best.center_rc.astype(np.float32),
            hessian_score=float(best.confidence),
            alt_centers={"adaptive_recovery": best.center_rc.astype(np.float32)} if adaptive_uncertain_first else {},
            failure_reasons=list(self._last_failure_reasons) if adaptive_uncertain_first else [],
            diagnostic_values=locked_diagnostics,
        )
        self._debug_finalize_frame(result=result, candidate=best)
        if profiling:
            self._profile_add("decision_state_ms", perf_counter() - decision_start)
        self._finish_frame_profile(
            frame_index=frame_index, status=result.status, search_bbox=search_bbox,
            confidence=best.confidence, started_at=total_start,
        )
        return result

    @staticmethod
    def _frame_view_parts(
        frame: np.ndarray | WorkspaceFrame,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Return BGR data, global origin (x,y), and full-frame size (h,w)."""
        if isinstance(frame, WorkspaceFrame):
            return frame.bgr, frame.origin_xy, frame.full_frame_hw
        return frame, (0, 0), frame.shape[:2]

    @staticmethod
    def _global_bbox_to_view_bbox(
        global_bbox_xywh: tuple[int, int, int, int],
        view_origin_xy: tuple[int, int],
        view_hw: tuple[int, int],
    ) -> Optional[tuple[int, int, int, int]]:
        """Map a required global ROI into a workspace; reject if not fully covered."""
        gx, gy, width, height = global_bbox_xywh
        ox, oy = view_origin_xy
        lx, ly = gx - ox, gy - oy
        view_height, view_width = view_hw
        if lx < 0 or ly < 0 or lx + width > view_width or ly + height > view_height:
            return None
        return (lx, ly, width, height)

    @staticmethod
    def _global_bbox_to_view_bbox_clipped(
        global_bbox_xywh: tuple[int, int, int, int],
        view_origin_xy: tuple[int, int],
        view_hw: tuple[int, int],
        *,
        anchor_center_rc: Optional[np.ndarray | tuple[float, float]] = None,
    ) -> Optional[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
        """Map a global ROI into the workspace, clipping ROI edges.

        If an anchor centre is supplied it must be inside the prepared workspace.
        This lets normal/tiny searches continue when only the ROI edge extends
        outside the workspace, while still rejecting genuinely out-of-workspace
        predicted/clicked centres.
        """
        gx, gy, width, height = [int(v) for v in global_bbox_xywh]
        ox, oy = [int(v) for v in view_origin_xy]
        view_height, view_width = [int(v) for v in view_hw]
        if width <= 0 or height <= 0 or view_width <= 0 or view_height <= 0:
            return None
        if anchor_center_rc is not None:
            anchor = np.asarray(anchor_center_rc, dtype=np.float32).reshape(2)
            ar, ac = float(anchor[0]), float(anchor[1])
            if not (oy <= ar < oy + view_height and ox <= ac < ox + view_width):
                return None
        bx0, by0 = gx, gy
        bx1, by1 = gx + width, gy + height
        vx0, vy0 = ox, oy
        vx1, vy1 = ox + view_width, oy + view_height
        cx0, cy0 = max(bx0, vx0), max(by0, vy0)
        cx1, cy1 = min(bx1, vx1), min(by1, vy1)
        if cx1 <= cx0 or cy1 <= cy0:
            return None
        local_bbox = (cx0 - ox, cy0 - oy, cx1 - cx0, cy1 - cy0)
        clipped_global = (cx0, cy0, cx1 - cx0, cy1 - cy0)
        return local_bbox, clipped_global

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_from_click(
        self,
        frame_bgr: np.ndarray | WorkspaceFrame,
        seed_row: float,
        seed_col: float,
        frame_index: Optional[int],
    ) -> _Initialization:
        image_bgr, frame_origin_xy, full_frame_hw = self._frame_view_parts(frame_bgr)
        height, width = full_frame_hw
        row = int(round(float(seed_row)))
        col = int(round(float(seed_col)))
        if not (0 <= row < height and 0 <= col < width):
            raise InitializationError("Clicked point lies outside the frame.")

        init_bbox = self._box_around_point(
            row, col, self._settings.init_half_size, (height, width)
        )
        mapped_init_bbox = self._global_bbox_to_view_bbox_clipped(
            init_bbox,
            frame_origin_xy,
            image_bgr.shape[:2],
            anchor_center_rc=(row, col),
        )
        if mapped_init_bbox is None:
            raise InitializationError(
                "Clicked point lies outside the prepared analysis workspace. "
                "Move/enlarge the workspace or initialize without the prepared cache."
            )

        local_init_bbox, init_bbox = mapped_init_bbox
        x0, y0, _, _ = init_bbox
        roi_bgr = self._crop_bgr_roi(image_bgr, local_init_bbox)
        roi_lab = self._to_lab(roi_bgr)
        click_local_rc = (row - y0, col - x0)

        seed_pixels = self._seed_pixels(roi_lab, click_local_rc)
        if self._settings.point_mode == "tiny":
            self._last_init_edit_context = None
            return self._initialize_tiny_from_click(
                roi_bgr=roi_bgr,
                roi_lab=roi_lab,
                roi_origin_xy=(x0, y0),
                click_local_rc=click_local_rc,
                frame_index=frame_index,
                full_frame_hw=(height, width),
            )

        border_pixels = self._border_pixels(roi_lab)
        prelim_fg = _GaussianLabModel.fit(
            seed_pixels,
            sigma_floor=self._settings.preliminary_fg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        prelim_bg = _GaussianLabModel.fit(
            border_pixels,
            sigma_floor=self._settings.preliminary_bg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        prelim_probability = self._foreground_probability(roi_lab, prelim_fg, prelim_bg)
        prelim_mask = self._clicked_component(
            prelim_probability, click_local_rc, initialization=True
        )

        interior_mask = self._interior_mask(prelim_mask)
        ring_mask = self._background_ring(prelim_mask)
        fg_pixels = roi_lab[interior_mask > 0]
        bg_pixels = roi_lab[ring_mask > 0]
        if fg_pixels.shape[0] < 8 or bg_pixels.shape[0] < 8:
            raise InitializationError(
                "Insufficient clean foreground/background pixels after segmentation."
            )

        foreground = _GaussianLabModel.fit(
            fg_pixels,
            sigma_floor=self._settings.final_fg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        background = _GaussianLabModel.fit(
            bg_pixels,
            sigma_floor=self._settings.final_bg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        probability = self._foreground_probability(roi_lab, foreground, background)
        marker_mask = self._clicked_component(
            probability, click_local_rc, initialization=True
        )

        contour = self._outer_contour(marker_mask)
        contour_xy = contour.reshape(-1, 2).astype(np.float32)
        if self._settings.refine_boundary_on_initialize:
            contour_xy = self._refine_contour_by_probability_gradient(contour, probability)

        center_local_rc = self._weighted_centroid(probability, marker_mask)
        geometry = self._geometry(contour_xy, center_local_rc)
        if geometry["area"] <= 0:
            raise InitializationError("Detected marker boundary has no measurable area.")

        expected_init_area = float(np.pi * (max(1.0, self._settings.expected_diameter_px) / 2.0) ** 2)
        init_area_ratio = float(geometry["area"] / max(expected_init_area, 1e-6))
        if init_area_ratio > self._settings.init_max_area_ratio:
            raise InitializationError(
                "Clicked initialization component is too large for the expected marker: "
                f"area ratio {init_area_ratio:.2f} exceeds threshold "
                f"{self._settings.init_max_area_ratio:.2f}. Increase the init threshold, "
                "reduce init close iterations, or increase expected diameter if appropriate."
            )

        quality = self._colour_quality(probability, marker_mask, self._background_ring(marker_mask))
        confidence = self._colour_score(quality)
        if quality["median_inside_probability"] < 0.48:
            raise InitializationError("Clicked region has insufficient marker colour confidence.")
        if quality["probability_margin"] < self._settings.normal_init_min_probability_margin:
            raise InitializationError(
                "Clicked marker is insufficiently separated from background: "
                f"probability margin {quality['probability_margin']:.3f} is below "
                f"threshold {self._settings.normal_init_min_probability_margin:.3f}."
            )

        shape_model: ShapeModel = "generic_compact"
        if geometry["is_rectangle"] and geometry["rectangularity"] >= self._settings.rectangle_rectangularity_threshold:
            shape_model = "rectangle"
        elif geometry["circularity"] >= self._settings.circle_circularity_threshold:
            shape_model = "circle_like"

        center_global = np.array(
            [y0 + center_local_rc[0], x0 + center_local_rc[1]], dtype=np.float32
        )
        geometric_center_global: Optional[np.ndarray] = None
        if geometry["geometric_center_xy"] is not None:
            gx, gy = geometry["geometric_center_xy"]
            geometric_center_global = np.array([y0 + gy, x0 + gx], dtype=np.float32)

        local_bbox = self._mask_bbox(marker_mask)
        tight_bbox = (
            x0 + local_bbox[0],
            y0 + local_bbox[1],
            local_bbox[2],
            local_bbox[3],
        )
        search_bbox = self._bbox_from_feature_and_margin(
            tight_bbox, self._settings.normal_search_margin_px, (height, width)
        )
        contour_global = contour_xy.copy()
        contour_global[:, 0] += x0
        contour_global[:, 1] += y0

        polarity, foreground_l_mean, background_l_mean = self._luminance_polarity_from_models(
            foreground, background
        )

        seed_bbox_global = self._sample_bbox_global((x0, y0), click_local_rc, self._settings.seed_radius)
        growth_r = int(round(self._init_growth_radius_px()))
        growth_bbox_global = self._ring_bbox_global((x0, y0), click_local_rc, growth_r)
        preview_rgba = self._initialization_overlay(
            frame_hw=(height, width),
            seed_bbox_xywh=seed_bbox_global,
            ring_bbox_xywh=growth_bbox_global,
        )

        initialization = _Initialization(
            center_rc=center_global,
            geometric_center_rc=geometric_center_global,
            shape_model=shape_model,
            foreground_model=foreground,
            background_model=background,
            tight_bbox_xywh=tight_bbox,
            search_bbox_xywh=search_bbox,
            contour_xy_global=contour_global,
            expected_area=float(geometry["area"]),
            expected_width=float(geometry["width"]),
            expected_height=float(geometry["height"]),
            circularity=float(geometry["circularity"]),
            rectangularity=float(geometry["rectangularity"]),
            confidence=float(confidence),
            tiny_seed_bbox_xywh=seed_bbox_global,
            tiny_ring_bbox_xywh=growth_bbox_global,
            preview_rgba=preview_rgba,
            foreground_l_mean=float(foreground_l_mean),
            background_l_mean=float(background_l_mean),
            luminance_polarity=str(polarity),
            init_roi_xywh=init_bbox,
            init_probability_roi=probability.astype(np.float32),
            init_mask_roi=marker_mask.astype(np.uint8),
            init_click_local_rc=(int(click_local_rc[0]), int(click_local_rc[1])),
            init_metrics={
                "median_inside_probability": float(quality["median_inside_probability"]),
                "strong_inside_fraction": float(quality["strong_inside_fraction"]),
                "probability_margin": float(quality["probability_margin"]),
                "confidence": float(confidence),
                "area": float(geometry["area"]),
                "area_ratio": float(init_area_ratio),
                "width": float(geometry["width"]),
                "height": float(geometry["height"]),
                "circularity": float(geometry["circularity"]),
                "rectangularity": float(geometry["rectangularity"]),
            },
        )
        self._last_init_edit_context = _InitEditContext(
            roi_lab=np.ascontiguousarray(roi_lab.copy()),
            roi_bgr=np.ascontiguousarray(roi_bgr.copy()),
            roi_origin_xy=(int(x0), int(y0)),
            click_local_rc=(int(click_local_rc[0]), int(click_local_rc[1])),
            frame_index=frame_index,
            full_frame_hw=(int(height), int(width)),
            point_mode="normal",
        )
        self._state = _TrackingState(
            initialization=initialization,
            expected_area=initialization.expected_area,
            expected_width=initialization.expected_width,
            expected_height=initialization.expected_height,
            last_valid_center_rc=center_global.copy(),
            previous_valid_center_rc=None,
            last_valid_frame_index=frame_index,
            consecutive_misses=0,
        )
        self._state.geometry_history.append({
            "width": float(initialization.expected_width),
            "height": float(initialization.expected_height),
            "area": float(initialization.expected_area),
            "angle_deg": 0.0,
        })
        self._state.center_history.append(center_global.astype(np.float32).copy())
        self._seed_validation_profile(self._state)
        self._kalman_reset(center_global, frame_index)
        return initialization

    def _initialize_tiny_from_click(
        self,
        *,
        roi_bgr: np.ndarray,
        roi_lab: np.ndarray,
        roi_origin_xy: tuple[int, int],
        click_local_rc: tuple[int, int],
        frame_index: Optional[int],
        full_frame_hw: tuple[int, int],
    ) -> _Initialization:
        """Initialize a tiny colour feature from a local centre-vs-ring sample.

        Tiny features do not have reliable contours.  The important reliability
        test is whether the clicked centre is locally distinctive from its
        surrounding ring.  We store the signed Lab contrast vector
        (centre_mean - ring_mean) and require later candidates to show a
        similar centre-vs-ring direction, so a generic dark pixel on a wooden
        desk is not treated as equivalent to the clicked dark spot.
        """
        x0, y0 = roi_origin_xy
        click_r, click_c = click_local_rc
        self._last_threshold_adjustments = []
        seed_pixels = self._seed_pixels(roi_lab, click_local_rc)

        rows, cols = np.indices(roi_lab.shape[:2])
        dist = np.sqrt((rows - click_r) ** 2 + (cols - click_c) ** 2)
        inner = max(self._settings.seed_radius + 1, int(round(self._settings.expected_diameter_px)))
        outer = max(inner + 4, int(round(self._settings.expected_diameter_px * 4.0)))
        ring = (dist >= inner) & (dist <= outer)
        bg_pixels = roi_lab[ring]
        if bg_pixels.shape[0] < 8:
            bg_pixels = self._border_pixels(roi_lab)
            ring_bbox_global = self._ring_bbox_global((x0, y0), click_local_rc, self._settings.init_half_size)
        else:
            ring_bbox_global = self._ring_bbox_global((x0, y0), click_local_rc, outer)

        foreground = _TinyLabDistanceModel.from_pixels(seed_pixels)
        background = _TinyLabDistanceModel.from_pixels(bg_pixels)

        init_delta = (foreground.mean - background.mean).astype(np.float32)
        init_contrast = self._weighted_lab_norm(init_delta)

        probability = self._foreground_probability(roi_lab, foreground, background)

        radius = max(1, int(round(self._settings.expected_diameter_px / 2.0)))
        window_mask = (dist <= max(radius, self._settings.seed_radius + 1)).astype(np.uint8)
        center_local_rc = self._weighted_centroid(probability, window_mask)

        center_global = np.array(
            [y0 + center_local_rc[0], x0 + center_local_rc[1]], dtype=np.float32
        )
        diameter = float(max(1.0, self._settings.expected_diameter_px))
        expected_area = float(np.pi * (diameter / 2.0) ** 2)
        tight_bbox = self._box_around_point(
            int(round(float(center_global[0]))),
            int(round(float(center_global[1]))),
            max(1, int(round(diameter / 2.0))),
            full_frame_hw,
        )
        search_bbox = self._bbox_from_feature_and_margin(
            tight_bbox, self._settings.normal_search_margin_px, full_frame_hw
        )
        x, y, w, h = tight_bbox
        contour = np.array(
            [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
            dtype=np.float32,
        )
        local_peak = float(probability[int(np.clip(click_r, 0, probability.shape[0]-1)), int(np.clip(click_c, 0, probability.shape[1]-1))])
        init_probability_contrast = self._tiny_probability_contrast_at(probability, click_r, click_c, radius)
        mosse_state = None
        init_mosse_psr = 0.0
        init_mosse_peak_margin = 0.0
        if self._settings.tiny_mosse_mode != "off":
            mosse_state = self._mosse_init_or_replace(probability, center_local_rc)
            temp_state = _TrackingState(
                initialization=_Initialization(
                    center_rc=center_global,
                    geometric_center_rc=None,
                    shape_model="generic_compact",
                    foreground_model=foreground,
                    background_model=background,
                    tight_bbox_xywh=tight_bbox,
                    search_bbox_xywh=search_bbox,
                    contour_xy_global=contour,
                    expected_area=expected_area,
                    expected_width=diameter,
                    expected_height=diameter,
                    circularity=0.0,
                    rectangularity=0.0,
                    confidence=0.0,
                ),
                expected_area=expected_area,
                expected_width=diameter,
                expected_height=diameter,
                last_valid_center_rc=center_global.copy(),
                previous_valid_center_rc=None,
                last_valid_frame_index=frame_index,
                mosse=mosse_state,
            )
            old_state = self._state
            self._state = temp_state
            try:
                metrics = self._mosse_locate(probability, center_local_rc)
                if metrics:
                    init_mosse_psr = float(metrics["psr"])
                    init_mosse_peak_margin = float(metrics["peak_margin"])
            finally:
                self._state = old_state

        self._relax_threshold_if_needed(
            field="tiny_init_peak_threshold",
            label="Tiny init peak",
            metric=local_peak,
            safety_factor=0.90,
        )
        self._relax_threshold_if_needed(
            field="tiny_peak_threshold",
            label="Tiny peak",
            metric=local_peak,
            safety_factor=0.90,
        )
        self._relax_threshold_if_needed(
            field="tiny_contrast_threshold",
            label="Tiny contrast",
            metric=init_probability_contrast,
            safety_factor=0.80,
        )
        self._relax_threshold_if_needed(
            field="tiny_init_contrast_threshold",
            label="Init contrast",
            metric=init_contrast,
            safety_factor=0.90,
        )
        if self._settings.tiny_mosse_mode != "off":
            self._relax_threshold_if_needed(
                field="tiny_mosse_psr_threshold",
                label="MOSSE PSR",
                metric=init_mosse_psr,
                safety_factor=0.75,
            )
            self._relax_threshold_if_needed(
                field="tiny_mosse_peak_margin_threshold",
                label="MOSSE margin",
                metric=init_mosse_peak_margin,
                safety_factor=0.80,
            )
        seed_bbox_global = self._sample_bbox_global((x0, y0), click_local_rc, self._settings.seed_radius)
        preview_rgba = self._initialization_overlay(
            frame_hw=full_frame_hw,
            seed_bbox_xywh=seed_bbox_global,
            ring_bbox_xywh=ring_bbox_global,
        )

        initialization = _Initialization(
            center_rc=center_global,
            geometric_center_rc=None,
            shape_model="generic_compact",
            foreground_model=foreground,
            background_model=background,
            tight_bbox_xywh=tight_bbox,
            search_bbox_xywh=search_bbox,
            contour_xy_global=contour,
            expected_area=expected_area,
            expected_width=diameter,
            expected_height=diameter,
            circularity=0.0,
            rectangularity=0.0,
            confidence=float(np.clip(0.55 * local_peak + 0.45 * np.clip(init_contrast / 30.0, 0.0, 1.0), 0.0, 1.0)),
            tiny_reference_delta_lab=init_delta,
            tiny_init_contrast=float(init_contrast),
            tiny_seed_bbox_xywh=seed_bbox_global,
            tiny_ring_bbox_xywh=ring_bbox_global,
            preview_rgba=preview_rgba,
            threshold_adjustments=tuple(self._last_threshold_adjustments),
            init_roi_xywh=(int(x0), int(y0), int(roi_lab.shape[1]), int(roi_lab.shape[0])),
            init_bgr_roi=np.ascontiguousarray(roi_bgr.copy()),
            init_probability_roi=probability.astype(np.float32),
            init_mask_roi=window_mask.astype(np.uint8),
            init_click_local_rc=(int(click_local_rc[0]), int(click_local_rc[1])),
            init_metrics={
                "confidence": float(np.clip(0.55 * local_peak + 0.45 * np.clip(init_contrast / 30.0, 0.0, 1.0), 0.0, 1.0)),
                "peak_probability": local_peak,
                "probability_contrast": float(init_probability_contrast),
                "lab_contrast": float(init_contrast),
                "mosse_psr": init_mosse_psr,
                "mosse_peak_margin": init_mosse_peak_margin,
            },
        )
        self._state = _TrackingState(
            initialization=initialization,
            expected_area=expected_area,
            expected_width=diameter,
            expected_height=diameter,
            last_valid_center_rc=center_global.copy(),
            previous_valid_center_rc=None,
            last_valid_frame_index=frame_index,
            consecutive_misses=0,
            mosse=mosse_state,
        )
        self._kalman_reset(center_global, frame_index)
        self._seed_validation_profile(self._state)
        return initialization

    def _normal_initialization_from_edit_context(
        self,
        *,
        parameters: dict[str, object],
        scissors_cuts: list[tuple[float, float, float, float]],
        commit: bool,
    ) -> _Initialization:
        ctx = self._last_init_edit_context
        if ctx is None or ctx.point_mode != "normal":
            raise InitializationError("Initialization editor is available only after normal point initialization.")

        old_settings = self._settings
        old_state = self._state
        overrides = self._settings_overrides_from_init_editor(parameters)
        self._settings = replace(self._settings, **overrides)
        committed = False
        try:
            initialization = self._build_normal_initialization_from_context(
                ctx,
                parameters=parameters,
                scissors_cuts=scissors_cuts,
            )
            if commit:
                self._commit_initialization(initialization, ctx.frame_index)
                self._write_init_editor_parameters_to_config(parameters)
                self._last_init_edit_context = ctx
                committed = True
            else:
                self._state = old_state
            return initialization
        finally:
            if not committed:
                self._settings = old_settings
                self._state = old_state

    def _settings_overrides_from_init_editor(self, parameters: dict[str, object]) -> dict[str, object]:
        overrides: dict[str, object] = {}

        def get_float(key: str, current: float, low: float, high: float) -> float:
            try:
                value = float(parameters.get(key, current))
            except (TypeError, ValueError):
                value = current
            return float(np.clip(value, low, high))

        def get_int(key: str, current: int, low: int, high: int) -> int:
            try:
                value = int(round(float(parameters.get(key, current))))
            except (TypeError, ValueError):
                value = current
            return int(np.clip(value, low, high))

        overrides["init_probability_thresholds"] = (
            get_float(
                "normal_init_threshold",
                float(self._settings.init_probability_thresholds[0]),
                0.05,
                0.99,
            ),
        )
        overrides["normal_init_min_probability_margin"] = get_float(
            "normal_init_min_probability_margin",
            float(self._settings.normal_init_min_probability_margin),
            0.0,
            0.50,
        )
        overrides["init_max_area_ratio"] = get_float(
            "normal_init_max_area_ratio",
            float(self._settings.init_max_area_ratio),
            0.25,
            50.0,
        )
        overrides["init_growth_radius_px"] = get_float(
            "normal_init_growth_radius",
            float(self._settings.init_growth_radius_px),
            0.0,
            500.0,
        )
        overrides["init_morphology_close_iterations"] = get_int(
            "normal_init_close_iterations",
            int(self._settings.init_morphology_close_iterations),
            0,
            5,
        )
        return overrides

    def _write_init_editor_parameters_to_config(self, parameters: dict[str, object]) -> None:
        mapping = {
            "normal_init_threshold": float,
            "normal_init_min_probability_margin": float,
            "normal_init_max_area_ratio": float,
            "normal_init_growth_radius": float,
            "normal_init_close_iterations": int,
        }
        for key, caster in mapping.items():
            if key not in parameters or not hasattr(self.config, key):
                continue
            try:
                setattr(self.config, key, caster(parameters[key]))
            except (TypeError, ValueError):
                pass

    def _build_normal_initialization_from_context(
        self,
        ctx: _InitEditContext,
        *,
        parameters: dict[str, object],
        scissors_cuts: list[tuple[float, ...]],
    ) -> _Initialization:
        roi_lab = ctx.roi_lab
        x0, y0 = ctx.roi_origin_xy
        height, width = ctx.full_frame_hw
        click_local_rc = ctx.click_local_rc

        seed_pixels = self._seed_pixels(roi_lab, click_local_rc)
        border_pixels = self._border_pixels(roi_lab)
        prelim_fg = _GaussianLabModel.fit(
            seed_pixels,
            sigma_floor=self._settings.preliminary_fg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        prelim_bg = _GaussianLabModel.fit(
            border_pixels,
            sigma_floor=self._settings.preliminary_bg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        prelim_probability = self._foreground_probability(roi_lab, prelim_fg, prelim_bg)
        prelim_mask = self._clicked_component(
            prelim_probability, click_local_rc, initialization=True
        )

        interior_mask = self._interior_mask(prelim_mask)
        ring_mask = self._background_ring(prelim_mask)
        fg_pixels = roi_lab[interior_mask > 0]
        bg_pixels = roi_lab[ring_mask > 0]
        if fg_pixels.shape[0] < 8 or bg_pixels.shape[0] < 8:
            raise InitializationError(
                "Insufficient clean foreground/background pixels after edited segmentation."
            )

        foreground = _GaussianLabModel.fit(
            fg_pixels,
            sigma_floor=self._settings.final_fg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        background = _GaussianLabModel.fit(
            bg_pixels,
            sigma_floor=self._settings.final_bg_sigma_floor,
            lightness_variance_scale=self._settings.lightness_variance_scale,
        )
        probability = self._foreground_probability(roi_lab, foreground, background)
        marker_mask = self._clicked_component(
            probability, click_local_rc, initialization=True
        )

        cut_messages: list[str] = []
        if scissors_cuts:
            try:
                cut_thickness = int(round(float(parameters.get("scissors_thickness", 3))))
            except (TypeError, ValueError):
                cut_thickness = 3
            marker_mask, cut_messages = self._apply_scissors_cuts_to_mask(
                marker_mask,
                click_local_rc,
                scissors_cuts,
                thickness=max(1, min(15, cut_thickness)),
            )
            # Refit on the corrected mask so the saved foreground/background
            # model follows the user-approved component, not the discarded tail.
            edited_interior = self._interior_mask(marker_mask)
            edited_ring = self._background_ring(marker_mask)
            edited_fg_pixels = roi_lab[edited_interior > 0]
            edited_bg_pixels = roi_lab[edited_ring > 0]
            if edited_fg_pixels.shape[0] >= 8 and edited_bg_pixels.shape[0] >= 8:
                foreground = _GaussianLabModel.fit(
                    edited_fg_pixels,
                    sigma_floor=self._settings.final_fg_sigma_floor,
                    lightness_variance_scale=self._settings.lightness_variance_scale,
                )
                background = _GaussianLabModel.fit(
                    edited_bg_pixels,
                    sigma_floor=self._settings.final_bg_sigma_floor,
                    lightness_variance_scale=self._settings.lightness_variance_scale,
                )
                probability = self._foreground_probability(roi_lab, foreground, background)

        contour = self._outer_contour(marker_mask)
        contour_xy = contour.reshape(-1, 2).astype(np.float32)
        if self._settings.refine_boundary_on_initialize:
            contour_xy = self._refine_contour_by_probability_gradient(contour, probability)

        center_local_rc = self._weighted_centroid(probability, marker_mask)
        geometry = self._geometry(contour_xy, center_local_rc)
        if geometry["area"] <= 0:
            raise InitializationError("Edited marker boundary has no measurable area.")

        expected_init_area = float(np.pi * (max(1.0, self._settings.expected_diameter_px) / 2.0) ** 2)
        init_area_ratio = float(geometry["area"] / max(expected_init_area, 1e-6))
        if init_area_ratio > self._settings.init_max_area_ratio:
            raise InitializationError(
                "Edited initialization component is too large for the expected marker: "
                f"area ratio {init_area_ratio:.2f} exceeds threshold "
                f"{self._settings.init_max_area_ratio:.2f}."
            )

        quality = self._colour_quality(probability, marker_mask, self._background_ring(marker_mask))
        confidence = self._colour_score(quality)
        if quality["median_inside_probability"] < 0.48:
            raise InitializationError("Edited region has insufficient marker colour confidence.")
        if quality["probability_margin"] < self._settings.normal_init_min_probability_margin:
            raise InitializationError(
                "Edited marker is insufficiently separated from background: "
                f"probability margin {quality['probability_margin']:.3f} is below "
                f"threshold {self._settings.normal_init_min_probability_margin:.3f}."
            )

        shape_model: ShapeModel = "generic_compact"
        if geometry["is_rectangle"] and geometry["rectangularity"] >= self._settings.rectangle_rectangularity_threshold:
            shape_model = "rectangle"
        elif geometry["circularity"] >= self._settings.circle_circularity_threshold:
            shape_model = "circle_like"

        center_global = np.array(
            [y0 + center_local_rc[0], x0 + center_local_rc[1]], dtype=np.float32
        )
        geometric_center_global: Optional[np.ndarray] = None
        if geometry["geometric_center_xy"] is not None:
            gx, gy = geometry["geometric_center_xy"]
            geometric_center_global = np.array([y0 + gy, x0 + gx], dtype=np.float32)

        local_bbox = self._mask_bbox(marker_mask)
        tight_bbox = (
            x0 + local_bbox[0],
            y0 + local_bbox[1],
            local_bbox[2],
            local_bbox[3],
        )
        search_bbox = self._bbox_from_feature_and_margin(
            tight_bbox, self._settings.normal_search_margin_px, (height, width)
        )
        contour_global = contour_xy.copy()
        contour_global[:, 0] += x0
        contour_global[:, 1] += y0

        polarity, foreground_l_mean, background_l_mean = self._luminance_polarity_from_models(
            foreground, background
        )
        seed_bbox_global = self._sample_bbox_global((x0, y0), click_local_rc, self._settings.seed_radius)
        growth_r = int(round(self._init_growth_radius_px()))
        growth_bbox_global = self._ring_bbox_global((x0, y0), click_local_rc, growth_r)
        preview_rgba = self._initialization_overlay(
            frame_hw=(height, width),
            seed_bbox_xywh=seed_bbox_global,
            ring_bbox_xywh=growth_bbox_global,
        )
        metrics: dict[str, object] = {
            "median_inside_probability": float(quality["median_inside_probability"]),
            "strong_inside_fraction": float(quality["strong_inside_fraction"]),
            "probability_margin": float(quality["probability_margin"]),
            "confidence": float(confidence),
            "area": float(geometry["area"]),
            "area_ratio": float(init_area_ratio),
            "width": float(geometry["width"]),
            "height": float(geometry["height"]),
            "circularity": float(geometry["circularity"]),
            "rectangularity": float(geometry["rectangularity"]),
            "cut_messages": cut_messages,
        }

        return _Initialization(
            center_rc=center_global,
            geometric_center_rc=geometric_center_global,
            shape_model=shape_model,
            foreground_model=foreground,
            background_model=background,
            tight_bbox_xywh=tight_bbox,
            search_bbox_xywh=search_bbox,
            contour_xy_global=contour_global,
            expected_area=float(geometry["area"]),
            expected_width=float(geometry["width"]),
            expected_height=float(geometry["height"]),
            circularity=float(geometry["circularity"]),
            rectangularity=float(geometry["rectangularity"]),
            confidence=float(confidence),
            tiny_seed_bbox_xywh=seed_bbox_global,
            tiny_ring_bbox_xywh=growth_bbox_global,
            preview_rgba=preview_rgba,
            foreground_l_mean=float(foreground_l_mean),
            background_l_mean=float(background_l_mean),
            luminance_polarity=str(polarity),
            init_roi_xywh=(int(x0), int(y0), int(roi_lab.shape[1]), int(roi_lab.shape[0])),
            init_probability_roi=probability.astype(np.float32),
            init_mask_roi=marker_mask.astype(np.uint8),
            init_click_local_rc=(int(click_local_rc[0]), int(click_local_rc[1])),
            init_metrics=metrics,
        )

    def _apply_scissors_cuts_to_mask(
        self,
        marker_mask: np.ndarray,
        click_local_rc: tuple[int, int],
        scissors_cuts: list[tuple[float, ...]],
        *,
        thickness: int,
    ) -> tuple[np.ndarray, list[str]]:
        edited = marker_mask.astype(np.uint8).copy()
        before_count, _, _, _ = cv2.connectedComponentsWithStats(edited, 8)
        for cut in scissors_cuts:
            if len(cut) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in cut[:4]]
            cv2.line(edited, (x1, y1), (x2, y2), 0, max(1, int(thickness)), cv2.LINE_8)

        row, col = click_local_rc
        count, labels, stats, _ = cv2.connectedComponentsWithStats(edited, 8)
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
            raise InitializationError("Edited click point lies outside the initialization crop.")
        keep_label = int(labels[row, col])
        if keep_label == 0:
            raise InitializationError(
                "Scissors cut removed the clicked side of the marker. Reset cuts or draw the cut farther from the click."
            )
        kept = (labels == keep_label).astype(np.uint8)
        messages: list[str] = []
        if count <= before_count:
            messages.append("Cut did not split the selected region.")
        removed_px = int(np.count_nonzero(marker_mask) - np.count_nonzero(kept))
        if removed_px <= 0:
            messages.append("Cut kept the original region unchanged.")
        else:
            messages.append(f"Cut removed {removed_px} px from initialization mask.")
        return kept, messages

    def _commit_initialization(
        self,
        initialization: _Initialization,
        frame_index: Optional[int],
    ) -> None:
        self._state = _TrackingState(
            initialization=initialization,
            expected_area=initialization.expected_area,
            expected_width=initialization.expected_width,
            expected_height=initialization.expected_height,
            last_valid_center_rc=initialization.center_rc.copy(),
            previous_valid_center_rc=None,
            last_valid_frame_index=frame_index,
            consecutive_misses=0,
        )
        self._state.geometry_history.append({
            "width": float(initialization.expected_width),
            "height": float(initialization.expected_height),
            "area": float(initialization.expected_area),
            "angle_deg": 0.0,
        })
        self._state.center_history.append(initialization.center_rc.astype(np.float32).copy())
        self._seed_validation_profile(self._state)
        self._kalman_reset(initialization.center_rc, frame_index)

    # ------------------------------------------------------------------
    # Initialization preview / tiny contrast helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_lab_norm(delta_lab: np.ndarray) -> float:
        weights = np.asarray((0.35, 1.0, 1.0), dtype=np.float32)
        delta = np.asarray(delta_lab, dtype=np.float32) * weights
        return float(np.linalg.norm(delta))

    @staticmethod
    def _weighted_lab_cosine(a_lab: np.ndarray, b_lab: np.ndarray) -> float:
        weights = np.asarray((0.35, 1.0, 1.0), dtype=np.float32)
        a = np.asarray(a_lab, dtype=np.float32) * weights
        b = np.asarray(b_lab, dtype=np.float32) * weights
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-9:
            return -1.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _sample_bbox_global(
        roi_origin_xy: tuple[int, int],
        click_local_rc: tuple[int, int],
        radius: int,
    ) -> tuple[int, int, int, int]:
        x0, y0 = roi_origin_xy
        row, col = click_local_rc
        r = max(0, int(radius))
        return (int(x0 + col - r), int(y0 + row - r), int(2 * r + 1), int(2 * r + 1))

    @staticmethod
    def _ring_bbox_global(
        roi_origin_xy: tuple[int, int],
        click_local_rc: tuple[int, int],
        radius: float,
    ) -> tuple[int, int, int, int]:
        x0, y0 = roi_origin_xy
        row, col = click_local_rc
        r = max(1, int(round(float(radius))))
        return (int(x0 + col - r), int(y0 + row - r), int(2 * r + 1), int(2 * r + 1))

    @staticmethod
    def _initialization_overlay(
        *,
        frame_hw: tuple[int, int],
        seed_bbox_xywh: Optional[tuple[int, int, int, int]],
        ring_bbox_xywh: Optional[tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        """Return a sparse RGBA overlay showing init sample and local ring.

        Green box = foreground sample around the click.
        Orange box = local background/contrast ring extent.
        """
        if seed_bbox_xywh is None and ring_bbox_xywh is None:
            return None
        height, width = frame_hw
        overlay = np.zeros((int(height), int(width), 4), dtype=np.uint8)

        def draw_rect(bbox: tuple[int, int, int, int], rgba: tuple[int, int, int, int], thickness: int) -> None:
            x, y, w, h = [int(v) for v in bbox]
            x0 = max(0, min(width - 1, x))
            y0 = max(0, min(height - 1, y))
            x1 = max(0, min(width - 1, x + max(1, w) - 1))
            y1 = max(0, min(height - 1, y + max(1, h) - 1))
            if x1 < x0 or y1 < y0:
                return
            for t in range(max(1, thickness)):
                yy0 = min(height - 1, y0 + t)
                yy1 = max(0, y1 - t)
                xx0 = min(width - 1, x0 + t)
                xx1 = max(0, x1 - t)
                overlay[yy0, xx0:xx1 + 1] = rgba
                overlay[yy1, xx0:xx1 + 1] = rgba
                overlay[yy0:yy1 + 1, xx0] = rgba
                overlay[yy0:yy1 + 1, xx1] = rgba

        if ring_bbox_xywh is not None:
            draw_rect(ring_bbox_xywh, (255, 165, 0, 190), 2)
        if seed_bbox_xywh is not None:
            draw_rect(seed_bbox_xywh, (0, 255, 0, 230), 2)
        return overlay

    def _preview(self, initialization: _Initialization) -> InitPreview:
        # BaseTracker uses [row, col] for geometry payloads.
        polygon_rc = initialization.contour_xy_global[:, [1, 0]].astype(np.float32)
        diagnostics: dict[str, object] = {}
        if (
            initialization.init_roi_xywh is not None
            and initialization.init_probability_roi is not None
            and initialization.init_mask_roi is not None
            and initialization.init_click_local_rc is not None
        ):
            diagnostics = {
                "viewer_available": True,
                "editor_available": self._settings.point_mode == "normal",
                "point_mode": self._settings.point_mode,
                "roi_xywh": tuple(int(v) for v in initialization.init_roi_xywh),
                "roi_bgr": (
                    initialization.init_bgr_roi.copy()
                    if initialization.init_bgr_roi is not None
                    else None if self._last_init_edit_context is None
                    else self._last_init_edit_context.roi_bgr.copy()
                ),
                "probability_roi": initialization.init_probability_roi.astype(np.float32).copy(),
                "mask_roi": initialization.init_mask_roi.astype(np.uint8).copy(),
                "click_local_rc": tuple(int(v) for v in initialization.init_click_local_rc),
                "center_rc": initialization.center_rc.astype(np.float32).copy(),
                "metrics": dict(initialization.init_metrics),
                "parameters": {
                    "normal_init_threshold": float(self._settings.init_probability_thresholds[0]),
                    "normal_init_min_probability_margin": float(self._settings.normal_init_min_probability_margin),
                    "normal_init_max_area_ratio": float(self._settings.init_max_area_ratio),
                    "normal_init_growth_radius": float(self._settings.init_growth_radius_px),
                    "normal_init_close_iterations": int(self._settings.init_morphology_close_iterations),
                    "scissors_thickness": 3,
                },
            }
        return InitPreview(
            center=initialization.center_rc.astype(np.float32),
            polygon=polygon_rc,
            hog_bbox=initialization.tight_bbox_xywh,
            color_preview=initialization.preview_rgba,
            threshold_adjustments=list(initialization.threshold_adjustments),
            init_diagnostics=diagnostics,
        )

    def _relax_threshold_if_needed(
        self,
        *,
        field: str,
        label: str,
        metric: float,
        safety_factor: float,
        minimum: float = 0.0,
    ) -> None:
        metric = float(metric)
        if not np.isfinite(metric) or metric <= minimum:
            return
        settings_field = "tiny_min_contrast" if field == "tiny_contrast_threshold" else field
        current = float(getattr(self._settings, settings_field))
        allowed = max(float(minimum), metric * float(safety_factor))
        if current <= allowed:
            return
        object.__setattr__(self._settings, settings_field, allowed)
        if hasattr(self.config, field):
            setattr(self.config, field, allowed)
        self._last_threshold_adjustments.append({
            "field": field,
            "label": label,
            "old": current,
            "new": allowed,
            "metric": metric,
        })

    @staticmethod
    def _tiny_probability_contrast_at(
        probability: np.ndarray,
        row: int,
        col: int,
        radius: int,
    ) -> float:
        if probability.size == 0:
            return 0.0
        row = int(np.clip(row, 0, probability.shape[0] - 1))
        col = int(np.clip(col, 0, probability.shape[1] - 1))
        radius = max(1, int(radius))
        r0 = max(0, row - radius)
        r1 = min(probability.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(probability.shape[1], col + radius + 1)
        rr0 = max(0, row - radius * 2)
        rr1 = min(probability.shape[0], row + radius * 2 + 1)
        cc0 = max(0, col - radius * 2)
        cc1 = min(probability.shape[1], col + radius * 2 + 1)
        inner = probability[r0:r1, c0:c1]
        outer = probability[rr0:rr1, cc0:cc1]
        if inner.size == 0 or outer.size == 0:
            return 0.0
        ring_mask = np.ones(outer.shape, dtype=bool)
        iy0, iy1 = r0 - rr0, r1 - rr0
        ix0, ix1 = c0 - cc0, c1 - cc0
        ring_mask[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
        ring = outer[ring_mask]
        ring_mean = float(np.mean(ring)) if ring.size else float(np.mean(outer))
        return float(np.mean(inner) - ring_mean)

    def _mosse_window_size(self) -> int:
        raw = int(round(max(1.0, self._settings.expected_diameter_px) * self._settings.tiny_mosse_window_scale))
        size = int(np.clip(raw, 16, 96))
        return size + 1 if size % 2 == 0 else size

    @staticmethod
    def _extract_probability_patch(probability: np.ndarray, center_rc: np.ndarray, size: int) -> np.ndarray:
        size = max(3, int(size))
        half = size // 2
        center = np.asarray(center_rc, dtype=np.float32).reshape(2)
        row = int(round(float(center[0])))
        col = int(round(float(center[1])))
        padded = np.pad(probability.astype(np.float32), ((half, half), (half, half)), mode="edge")
        pr = row + half
        pc = col + half
        return np.ascontiguousarray(padded[pr - half:pr + half + 1, pc - half:pc + half + 1], dtype=np.float32)

    def _extract_scale_normalized_patch(
        self,
        probability: np.ndarray,
        center_rc: np.ndarray,
        size: int,
        scale_factor: float,
    ) -> np.ndarray:
        """Sample a physical scale, then normalize it to the MOSSE template size."""
        source_size = max(3, int(round(float(size) * float(scale_factor))))
        if source_size % 2 == 0:
            source_size += 1
        patch = self._extract_probability_patch(probability, center_rc, source_size)
        if source_size == size:
            return patch
        return cv2.resize(patch, (int(size), int(size)), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    @staticmethod
    def _mosse_preprocess(patch: np.ndarray, hann: np.ndarray) -> np.ndarray:
        x = np.asarray(patch, dtype=np.float32)
        x = np.log1p(np.clip(x, 0.0, 1.0) * 4.0)
        x = x - float(np.mean(x))
        std = float(np.std(x))
        if std > 1e-6:
            x = x / std
        return (x * hann).astype(np.float32)

    @staticmethod
    def _mosse_desired_response(size: int, sigma: float) -> np.ndarray:
        center = size // 2
        rows, cols = np.indices((size, size), dtype=np.float32)
        dist2 = (rows - center) ** 2 + (cols - center) ** 2
        response = np.exp(-0.5 * dist2 / max(float(sigma) ** 2, 1e-6))
        return response.astype(np.float32)

    def _mosse_init_or_replace(self, probability: np.ndarray, center_local_rc: np.ndarray) -> Optional[_MosseState]:
        size = self._mosse_window_size()
        hann_1d = np.hanning(size).astype(np.float32)
        hann = np.outer(hann_1d, hann_1d).astype(np.float32)
        desired = self._mosse_desired_response(size, max(1.0, self._settings.expected_diameter_px / 2.0))
        desired_fft = np.fft.fft2(desired).astype(np.complex64)
        patch = self._extract_probability_patch(probability, center_local_rc, size)
        F = np.fft.fft2(self._mosse_preprocess(patch, hann)).astype(np.complex64)
        A = (desired_fft * np.conj(F)).astype(np.complex64)
        B = (F * np.conj(F)).astype(np.complex64)
        return _MosseState(A=A, B=B, desired_fft=desired_fft, hann=hann, window_size=size)

    @staticmethod
    def _mosse_response_stats(response: np.ndarray, peak_rc: tuple[int, int], exclude_radius: int) -> tuple[float, float, float]:
        peak_r, peak_c = peak_rc
        peak = float(response[peak_r, peak_c])
        mask = np.ones(response.shape, dtype=bool)
        r0 = max(0, peak_r - exclude_radius)
        r1 = min(response.shape[0], peak_r + exclude_radius + 1)
        c0 = max(0, peak_c - exclude_radius)
        c1 = min(response.shape[1], peak_c + exclude_radius + 1)
        mask[r0:r1, c0:c1] = False
        sidelobe = response[mask]
        if sidelobe.size == 0:
            return peak, 0.0, 1.0
        mean = float(np.mean(sidelobe))
        std = float(np.std(sidelobe))
        psr = (peak - mean) / max(std, 1e-6)
        second = float(np.max(sidelobe))
        margin = (peak - second) / max(abs(peak), 1e-6)
        return peak, float(psr), float(margin)

    def _mosse_locate(
        self,
        probability: np.ndarray,
        predicted_local_rc: np.ndarray,
    ) -> Optional[dict[str, object]]:
        assert self._state is not None
        mosse = self._state.mosse
        if mosse is None:
            return None
        H = mosse.A / (mosse.B + 1e-5)
        if self._settings.mosse_scale_adaptation:
            half_levels = self._settings.mosse_scale_levels // 2
            relative_scales = [self._settings.mosse_scale_step ** offset for offset in range(-half_levels, half_levels + 1)]
        else:
            relative_scales = [1.0]
        best_scale_match: Optional[tuple[float, float, float, float, int, int]] = None
        for relative_scale in relative_scales:
            proposed = float(mosse.scale_factor * relative_scale)
            proposed = float(np.clip(
                proposed,
                mosse.scale_factor * (1.0 - self._settings.mosse_scale_max_step),
                mosse.scale_factor * (1.0 + self._settings.mosse_scale_max_step),
            ))
            patch = self._extract_scale_normalized_patch(
                probability, predicted_local_rc, mosse.window_size, proposed
            )
            F = np.fft.fft2(self._mosse_preprocess(patch, mosse.hann)).astype(np.complex64)
            response = np.fft.ifft2(H * F).real.astype(np.float32)
            peak_index = int(np.argmax(response))
            peak_r, peak_c = np.unravel_index(peak_index, response.shape)
            response_peak, psr, margin = self._mosse_response_stats(
                response, (int(peak_r), int(peak_c)),
                max(2, int(round(self._state.expected_width))),
            )
            candidate = (float(psr), float(margin), float(response_peak), proposed, int(peak_r), int(peak_c))
            if best_scale_match is None or candidate[:3] > best_scale_match[:3]:
                best_scale_match = candidate
        if best_scale_match is None:
            return None
        psr, margin, response_peak, scale_factor, peak_r, peak_c = best_scale_match
        center = mosse.window_size // 2
        dr = int(peak_r) - center
        dc = int(peak_c) - center
        if dr > center:
            dr -= mosse.window_size
        if dc > center:
            dc -= mosse.window_size
        peak_local_rc = np.asarray([
            predicted_local_rc[0] + dr * scale_factor,
            predicted_local_rc[1] + dc * scale_factor,
        ], dtype=np.float32)
        mosse.last_psr = psr
        mosse.last_peak_margin = margin
        mosse.last_peak_response = response_peak
        return {
            "peak_local_rc": peak_local_rc,
            "psr": psr,
            "peak_margin": margin,
            "response_peak": response_peak,
            "scale_factor": scale_factor,
        }

    def _mosse_update(self, probability: np.ndarray, roi_origin_xy: tuple[int, int], center_rc: np.ndarray) -> None:
        assert self._state is not None
        mosse = self._state.mosse
        if mosse is None or self._settings.tiny_mosse_learning_rate <= 0.0:
            return
        local = np.asarray([center_rc[0] - roi_origin_xy[1], center_rc[1] - roi_origin_xy[0]], dtype=np.float32)
        patch = self._extract_scale_normalized_patch(
            probability, local, mosse.window_size, mosse.scale_factor
        )
        F = np.fft.fft2(self._mosse_preprocess(patch, mosse.hann)).astype(np.complex64)
        A_new = (mosse.desired_fft * np.conj(F)).astype(np.complex64)
        B_new = (F * np.conj(F)).astype(np.complex64)
        alpha = float(self._settings.tiny_mosse_learning_rate)
        mosse.A = ((1.0 - alpha) * mosse.A + alpha * A_new).astype(np.complex64)
        mosse.B = ((1.0 - alpha) * mosse.B + alpha * B_new).astype(np.complex64)

    # ------------------------------------------------------------------
    # Prediction/search
    # ------------------------------------------------------------------

    def _predict_center(self, frame_index: int) -> np.ndarray:
        assert self._state is not None
        if self._settings.kalman_enabled:
            self._kalman_predict_to(int(frame_index))
            assert self._state.kalman_x is not None
            return self._state.kalman_x[:2].astype(np.float32).copy()

        state = self._state
        if state.previous_valid_center_rc is None:
            return state.last_valid_center_rc.copy()

        velocity = state.last_valid_center_rc - state.previous_valid_center_rc
        if state.last_valid_frame_index is None:
            frames_since_measurement = 1
        else:
            frames_since_measurement = max(1, int(frame_index) - state.last_valid_frame_index)
        return (state.last_valid_center_rc + velocity * frames_since_measurement).astype(np.float32)

    def _search_bbox(
        self,
        frame_hw: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> tuple[int, int, int, int]:
        assert self._state is not None
        if self._settings.point_mode == "tiny":
            expected_half = int(np.ceil(max(1.0, self._state.expected_width) / 2.0))
        else:
            expected_half = int(np.ceil(max(self._state.expected_width, self._state.expected_height) / 2.0))

        if self._settings.kalman_enabled and self._state.kalman_P is not None:
            sigma = float(np.sqrt(max(self._state.kalman_P[0, 0], self._state.kalman_P[1, 1], 1e-9)))
            # The ROI half-extent is marker half-size plus permitted centre
            # displacement.  Keeping these separate avoids adding the marker
            # size twice when Kalman uncertainty is low.
            center_margin = int(np.ceil(max(
                self._settings.normal_search_margin_px,
                self._settings.kalman_gate_sigma * sigma,
            )))
            half_extent = max(
                self._settings.min_search_radius_px,
                expected_half + center_margin,
            )
            half_extent = min(self._settings.max_dynamic_search_radius_px, half_extent)
        else:
            margin = min(
                self._settings.max_search_margin_px,
                self._settings.normal_search_margin_px
                + self._state.consecutive_misses * self._settings.miss_expansion_px,
            )
            half_extent = expected_half + margin
        return self._box_around_point(
            int(round(float(predicted_center_rc[0]))),
            int(round(float(predicted_center_rc[1]))),
            half_extent,
            frame_hw,
        )

    def _candidate_touches_search_edge(
        self,
        candidate: _Candidate,
        search_bbox_xywh: tuple[int, int, int, int],
    ) -> bool:
        """Return whether a normal-marker candidate looks clipped by the ROI edge."""
        edge = int(self._settings.normal_recenter_edge_px)
        if edge <= 0:
            return False
        sx, sy, sw, sh = [int(v) for v in search_bbox_xywh]
        bx, by, bw, bh = [int(v) for v in candidate.tight_bbox_xywh]
        left = bx - sx
        top = by - sy
        right = (sx + sw) - (bx + bw)
        bottom = (sy + sh) - (by + bh)
        return min(left, top, right, bottom) <= edge

    def _try_normal_blob_recenter(
        self,
        *,
        image_bgr: np.ndarray,
        full_frame_hw: tuple[int, int],
        frame_origin_xy: tuple[int, int],
        current_search_bbox: tuple[int, int, int, int],
        candidate: _Candidate,
    ) -> Optional[list[_Candidate]]:
        """One-pass normal-marker recentering for clipped colour blobs.

        If a valid normal-marker blob touches the search ROI border, the first
        centroid may represent only the visible part of the marker.  This pass
        recentres a slightly larger ROI around that accepted blob, recomputes
        the whole colour segmentation, and returns candidates from the updated
        ROI.  Tiny feature mode intentionally does not use this path.
        """
        assert self._state is not None
        if self._settings.point_mode != "normal":
            return None
        if not self._settings.normal_recenter_on_edge:
            return None
        if not self._candidate_touches_search_edge(candidate, current_search_bbox):
            return None
        if candidate.area < self._state.expected_area * self._settings.normal_recenter_min_area_ratio:
            return None

        expected_half = int(np.ceil(max(self._state.expected_width, self._state.expected_height) / 2.0))
        current_half = int(np.ceil(max(current_search_bbox[2], current_search_bbox[3]) / 2.0))
        current_margin = max(0, current_half - expected_half)
        target_margin = max(
            current_margin,
            self._settings.normal_search_margin_px + self._settings.normal_recenter_extra_margin_px,
        )
        target_margin = min(self._settings.max_search_margin_px, int(target_margin))
        new_half = expected_half + target_margin
        new_bbox = self._box_around_point(
            int(round(float(candidate.center_rc[0]))),
            int(round(float(candidate.center_rc[1]))),
            new_half,
            full_frame_hw,
        )
        if new_bbox == current_search_bbox:
            return None

        mapped_bbox = self._global_bbox_to_view_bbox_clipped(
            new_bbox,
            frame_origin_xy,
            image_bgr.shape[:2],
            anchor_center_rc=candidate.center_rc,
        )
        if mapped_bbox is None:
            return None
        local_bbox, new_bbox = mapped_bbox

        saved_counts = dict(self._last_rejection_counts)
        saved_reasons = list(self._last_failure_reasons)
        saved_scan = dict(self._last_candidate_scan_metrics)
        saved_diag = dict(self._last_failure_diagnostics)
        saved_peak = None if self._last_peak_center_rc is None else self._last_peak_center_rc.copy()

        try:
            stage_start = perf_counter() if self._profiling_enabled else 0.0
            roi_bgr = self._crop_bgr_roi(image_bgr, local_bbox)
            roi_lab = self._to_lab(roi_bgr)
            probability = self._foreground_probability(
                roi_lab,
                self._state.initialization.foreground_model,
                self._state.initialization.background_model,
            )
            # Use the accepted first-pass blob as the anchor for the second pass.
            # The original prediction may be lagging under acceleration; the point
            # of this pass is to recover the whole marker without permanently
            # enlarging the locked-frame ROI.
            recentered_candidates = self._accepted_candidates(
                probability=probability,
                roi_lab=roi_lab,
                roi_origin_xy=(new_bbox[0], new_bbox[1]),
                predicted_center_rc=candidate.center_rc,
            )
            if self._profiling_enabled:
                self._profile_add("normal_recenter_ms", perf_counter() - stage_start)
            if not recentered_candidates:
                self._last_rejection_counts = saved_counts
                self._last_failure_reasons = saved_reasons
                self._last_candidate_scan_metrics = saved_scan
                self._last_failure_diagnostics = saved_diag
                self._last_peak_center_rc = saved_peak
                return None
            self._set_failure_diagnostic("normal_recentered", True)
            self._set_failure_diagnostic("normal_recenter_bbox_xywh", new_bbox)
            return recentered_candidates
        except Exception:
            self._last_rejection_counts = saved_counts
            self._last_failure_reasons = saved_reasons
            self._last_candidate_scan_metrics = saved_scan
            self._last_failure_diagnostics = saved_diag
            self._last_peak_center_rc = saved_peak
            return None


    def _smoothed_blob_geometry(self) -> tuple[float, float, float, float]:
        """Return smoothed width, height, area, angle from recent accepted normal blobs."""
        assert self._state is not None
        history = list(getattr(self._state, "geometry_history", []))
        if not history:
            return (
                float(max(1.0, self._state.expected_width)),
                float(max(1.0, self._state.expected_height)),
                float(max(1.0, self._state.expected_area)),
                0.0,
            )
        widths = np.asarray([max(1.0, float(g.get("width", self._state.expected_width))) for g in history], dtype=np.float32)
        heights = np.asarray([max(1.0, float(g.get("height", self._state.expected_height))) for g in history], dtype=np.float32)
        areas = np.asarray([max(1.0, float(g.get("area", self._state.expected_area))) for g in history], dtype=np.float32)
        angles = np.asarray([float(g.get("angle_deg", 0.0)) for g in history], dtype=np.float32)
        # Circular mean over 180-degree orientation because rectangle axes are bidirectional.
        radians = np.deg2rad(angles * 2.0)
        mean_angle = 0.5 * np.rad2deg(np.arctan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))
        return float(np.median(widths)), float(np.median(heights)), float(np.median(areas)), float(mean_angle)

    def _adaptive_grid_axes(self, width: float, height: float, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Return two row/col unit vectors for adaptive sparse probing."""
        assert self._state is not None
        centers = list(getattr(self._state, "center_history", []))
        if len(centers) >= 2:
            delta = np.asarray(centers[-1], dtype=np.float32) - np.asarray(centers[0], dtype=np.float32)
            norm = float(np.linalg.norm(delta))
            if norm >= 1.0:
                u = delta / norm  # row/col motion direction
                v = np.array([-u[1], u[0]], dtype=np.float32)
                return u.astype(np.float32), v.astype(np.float32)
        if max(width, height) / max(min(width, height), 1e-6) >= 1.20:
            theta = np.deg2rad(float(angle_deg))
            # Convert image x/y angle to row/col axis vector.
            u = np.array([np.sin(theta), np.cos(theta)], dtype=np.float32)
            v = np.array([-u[1], u[0]], dtype=np.float32)
            return u, v
        return np.array([0.0, 1.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)

    def _adaptive_grid_points(
        self,
        shape_hw: tuple[int, int],
        center_local_rc: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        spacing_u: float,
        spacing_v: float,
        *,
        shifted: bool,
    ) -> list[tuple[int, int]]:
        height, width = [int(v) for v in shape_hw]
        diag = float(np.hypot(height, width))
        nu = int(np.ceil(diag / max(spacing_u, 1.0))) + 2
        nv = int(np.ceil(diag / max(spacing_v, 1.0))) + 2
        offsets = [(0.0, 0.0)] if not shifted else [(0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
        points: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for ou, ov in offsets:
            for i in range(-nu, nu + 1):
                for j in range(-nv, nv + 1):
                    rc = center_local_rc + (i + ou) * spacing_u * axis_u + (j + ov) * spacing_v * axis_v
                    r = int(round(float(rc[0])))
                    c = int(round(float(rc[1])))
                    if 0 <= r < height and 0 <= c < width and (r, c) not in seen:
                        seen.add((r, c))
                        points.append((r, c))
        return points

    def _adaptive_prefilter_candidates(
        self,
        *,
        image_bgr: np.ndarray,
        frame_origin_xy: tuple[int, int],
        full_frame_hw: tuple[int, int],
        predicted_center: np.ndarray,
        normal_search_bbox: tuple[int, int, int, int],
    ) -> Optional[tuple[list[_Candidate], bool]]:
        """Coarse-to-fine normal-marker recovery search.

        Phase 2A: after at least one miss, use a sparse LAB prefilter in a 2x
        recovery ROI (3x after repeated misses).  Grid density is based on the
        smoothed previous blob geometry, assuming the current visible feature is
        at least about 90% of that size.  Expensive PointFast scoring is run only
        on the best small candidate patches.
        """
        assert self._state is not None
        if not self._settings.adaptive_prefilter_enabled:
            return None
        if self._settings.point_mode != "normal":
            return None
        if self._state.consecutive_misses < 1:
            return None

        started = perf_counter() if self._profiling_enabled else 0.0
        width_s, height_s, area_s, angle_s = self._smoothed_blob_geometry()
        scale = 2.0 if self._state.consecutive_misses == 1 else 3.0
        normal_half = max(int(np.ceil(max(normal_search_bbox[2], normal_search_bbox[3]) / 2.0)), 1)
        recovery_half = int(np.ceil(normal_half * scale))
        requested_recovery_bbox = self._box_around_point(
            int(round(float(predicted_center[0]))),
            int(round(float(predicted_center[1]))),
            recovery_half,
            full_frame_hw,
        )
        mapped = self._global_bbox_to_view_bbox_clipped(
            requested_recovery_bbox,
            frame_origin_xy,
            image_bgr.shape[:2],
            anchor_center_rc=predicted_center,
        )
        if mapped is None:
            self._last_rejection_counts["adaptive_workspace"] = 1
            self._add_failure_reason("adaptive_prefilter_outside_workspace")
            return None
        local_bbox, recovery_bbox = mapped
        roi_bgr = self._crop_bgr_roi(image_bgr, local_bbox)
        if roi_bgr.size == 0:
            return None
        roi_lab = self._to_lab(roi_bgr)
        probability = self._foreground_probability(
            roi_lab,
            self._state.initialization.foreground_model,
            self._state.initialization.background_model,
        )

        axis_u, axis_v = self._adaptive_grid_axes(width_s, height_s, angle_s)
        # Assume current feature may be 90% of recent geometry; spacing remains
        # safely below that to avoid stepping over the feature.
        long_s = max(width_s, height_s, 1.0)
        short_s = max(min(width_s, height_s), 1.0)
        spacing_u = float(np.clip(0.72 * long_s, 3.0, 40.0))
        spacing_v = float(np.clip(0.72 * short_s, 3.0, 40.0))
        center_local = np.asarray([predicted_center[0] - recovery_bbox[1], predicted_center[1] - recovery_bbox[0]], dtype=np.float32)

        seeds: list[tuple[float, int, int]] = []
        for shifted in (False, True):
            points = self._adaptive_grid_points(
                probability.shape,
                center_local,
                axis_u,
                axis_v,
                spacing_u,
                spacing_v,
                shifted=shifted,
            )
            for r, c in points:
                score = float(probability[r, c])
                if score >= self._settings.adaptive_prefilter_min_probability:
                    seeds.append((score, r, c))
            if seeds:
                break
        if not seeds:
            self._last_rejection_counts["adaptive_no_seed"] = 1
            self._add_failure_reason("adaptive_prefilter_no_seed")
            self._set_failure_diagnostic("adaptive_recovery_bbox_xywh", recovery_bbox)
            return None

        patch_half = max(4, int(round(max(width_s, height_s) * 0.85)))
        expected_area = max(1.0, float(area_s))
        proposals: list[dict[str, Any]] = []
        seeds.sort(reverse=True, key=lambda item: item[0])
        for seed_score, r, c in seeds[:200]:
            r0 = max(0, r - patch_half)
            r1 = min(probability.shape[0], r + patch_half + 1)
            c0 = max(0, c - patch_half)
            c1 = min(probability.shape[1], c + patch_half + 1)
            local_prob = probability[r0:r1, c0:c1]
            mask = local_prob >= self._settings.adaptive_prefilter_support_threshold
            support_area = int(mask.sum())
            if support_area < max(3, int(0.12 * expected_area)):
                continue
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                continue
            by0, by1 = int(rows.min()), int(rows.max()) + 1
            bx0, bx1 = int(cols.min()), int(cols.max()) + 1
            comp_w = float(bx1 - bx0)
            comp_h = float(by1 - by0)
            approx_area = float(support_area)
            if approx_area > expected_area * 4.0:
                continue
            axis_ratio = min(comp_w / max(width_s, 1.0), comp_h / max(height_s, 1.0))
            area_ratio = approx_area / expected_area
            size_score = float(np.exp(-abs(np.log(max(area_ratio, 1e-6))) / 0.80))
            center_rc = np.array([
                recovery_bbox[1] + r0 + float((rows.mean() if rows.size else r - r0)),
                recovery_bbox[0] + c0 + float((cols.mean() if cols.size else c - c0)),
            ], dtype=np.float32)
            distance = float(np.linalg.norm(center_rc - predicted_center))
            motion_scale = max(float(self._settings.max_prediction_error_px + self._state.consecutive_misses * self._settings.miss_motion_allowance_px), 1.0)
            motion_score = float(np.exp(-0.5 * (distance / max(motion_scale, 1.0)) ** 2))
            support_density = float(support_area / max(local_prob.size, 1))
            priority = float(0.42 * seed_score + 0.24 * size_score + 0.20 * motion_score + 0.14 * min(1.0, support_density * 4.0))
            gx0 = recovery_bbox[0] + c0 + bx0
            gy0 = recovery_bbox[1] + r0 + by0
            gw = int(bx1 - bx0)
            gh = int(by1 - by0)
            margin = max(4, int(round(max(width_s, height_s) * 0.40)))
            patch_bbox = self._bbox_from_feature_and_margin((gx0, gy0, gw, gh), margin, full_frame_hw)
            duplicate = False
            for existing in proposals:
                if float(np.linalg.norm(existing["center_rc"] - center_rc)) < max(3.0, 0.45 * short_s):
                    duplicate = True
                    if priority > existing["priority"]:
                        existing.update({"priority": priority, "patch_bbox": patch_bbox, "center_rc": center_rc, "seed_score": seed_score})
                    break
            if not duplicate:
                proposals.append({"priority": priority, "patch_bbox": patch_bbox, "center_rc": center_rc, "seed_score": seed_score})
        if not proposals:
            self._last_rejection_counts["adaptive_no_supported_seed"] = 1
            self._add_failure_reason("adaptive_prefilter_no_supported_seed")
            return None

        proposals.sort(key=lambda item: float(item["priority"]), reverse=True)
        accepted: list[_Candidate] = []
        for proposal in proposals[: max(1, int(self._settings.adaptive_prefilter_max_candidates))]:
            mapped_patch = self._global_bbox_to_view_bbox_clipped(
                proposal["patch_bbox"],
                frame_origin_xy,
                image_bgr.shape[:2],
                anchor_center_rc=proposal["center_rc"],
            )
            if mapped_patch is None:
                continue
            local_patch, patch_bbox = mapped_patch
            patch_bgr = self._crop_bgr_roi(image_bgr, local_patch)
            if patch_bgr.size == 0:
                continue
            patch_lab = self._to_lab(patch_bgr)
            patch_probability = self._foreground_probability(
                patch_lab,
                self._state.initialization.foreground_model,
                self._state.initialization.background_model,
            )
            candidates = self._accepted_candidates(
                probability=patch_probability,
                roi_lab=patch_lab,
                roi_origin_xy=(patch_bbox[0], patch_bbox[1]),
                predicted_center_rc=np.asarray(proposal["center_rc"], dtype=np.float32),
            )
            if candidates:
                candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
                best = candidates[0]
                object.__setattr__(best, "adaptive_source", "adaptive_prefilter")
                accepted.append(best)
                # Candidate list is already priority-sorted; stop at first full
                # candidate that passes expensive colour/geometry scoring.
                break
        if self._profiling_enabled:
            self._profile_add("adaptive_prefilter_ms", perf_counter() - started)
        self._set_failure_diagnostic("adaptive_prefilter_enabled", True)
        self._set_failure_diagnostic("adaptive_recovery_bbox_xywh", recovery_bbox)
        self._set_failure_diagnostic("adaptive_seed_count", len(seeds))
        self._set_failure_diagnostic("adaptive_proposal_count", len(proposals))
        self._set_failure_diagnostic("adaptive_scale", scale)
        self._set_failure_diagnostic("adaptive_spacing_u", spacing_u)
        self._set_failure_diagnostic("adaptive_spacing_v", spacing_v)
        if not accepted:
            self._last_rejection_counts["adaptive_no_full_candidate"] = 1
            self._add_failure_reason("adaptive_prefilter_no_full_candidate")
            return None
        self._add_failure_reason("adaptive_reacquisition_pending_confirmation")
        return accepted, True

    def _kalman_dimension(self) -> int:
        return 6 if self._settings.kalman_model == "constant_acceleration" else 4

    def _kalman_transition_and_q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        q = float(self._settings.process_noise_px) ** 2
        if self._settings.kalman_model == "constant_acceleration":
            F = np.eye(6, dtype=np.float32)
            for offset in (0, 1):
                F[offset, offset + 2] = dt
                F[offset, offset + 4] = 0.5 * dt * dt
                F[offset + 2, offset + 4] = dt
            # White-jerk-style approximation. Tuned for robustness rather than
            # exact physics; process_noise is user-adjustable in px/frame^2-ish.
            Q = np.eye(6, dtype=np.float32) * q * 1e-3
            for offset in (0, 1):
                Q[offset, offset] = q * (dt ** 4) / 4.0
                Q[offset + 2, offset + 2] = q * (dt ** 2)
                Q[offset + 4, offset + 4] = q
            return F, Q

        F = np.eye(4, dtype=np.float32)
        F[0, 2] = dt
        F[1, 3] = dt
        Q = np.zeros((4, 4), dtype=np.float32)
        for offset in (0, 1):
            Q[offset, offset] = q * (dt ** 4) / 4.0
            Q[offset, offset + 2] = q * (dt ** 3) / 2.0
            Q[offset + 2, offset] = q * (dt ** 3) / 2.0
            Q[offset + 2, offset + 2] = q * (dt ** 2)
        Q += np.eye(4, dtype=np.float32) * 1e-6
        return F, Q

    def _kalman_reset(self, center_rc: np.ndarray, frame_index: Optional[int]) -> None:
        assert self._state is not None
        dim = self._kalman_dimension()
        x = np.zeros(dim, dtype=np.float32)
        x[0:2] = np.asarray(center_rc, dtype=np.float32)
        p = np.eye(dim, dtype=np.float32)
        meas_var = float(self._settings.measurement_noise_px) ** 2
        p[0, 0] = p[1, 1] = max(meas_var, 1e-3)
        if dim >= 4:
            p[2, 2] = p[3, 3] = max(25.0 * meas_var, 4.0)
        if dim == 6:
            p[4, 4] = p[5, 5] = max(float(self._settings.process_noise_px) ** 2, 1.0)
        self._state.kalman_x = x
        self._state.kalman_P = p
        self._state.kalman_frame_index = frame_index

    def _kalman_predict_to(self, frame_index: int) -> None:
        assert self._state is not None
        if self._state.kalman_x is None or self._state.kalman_P is None:
            self._kalman_reset(self._state.last_valid_center_rc, self._state.last_valid_frame_index)
        last = self._state.kalman_frame_index
        # Reverse tracking learns velocity in reverse time, so the Kalman
        # transition uses elapsed-frame magnitude rather than its sign.
        dt = 1.0 if last is None else abs(float(int(frame_index) - int(last)))
        if dt <= 0.0:
            return
        F, Q = self._kalman_transition_and_q(dt)
        self._state.kalman_x = (F @ self._state.kalman_x).astype(np.float32)
        self._state.kalman_P = (F @ self._state.kalman_P @ F.T + Q).astype(np.float32)
        self._state.kalman_frame_index = int(frame_index)

    def _kalman_update(self, measurement_rc: np.ndarray) -> None:
        assert self._state is not None
        if self._state.kalman_x is None or self._state.kalman_P is None:
            self._kalman_reset(measurement_rc, self._state.last_valid_frame_index)
            return
        dim = self._state.kalman_x.shape[0]
        H = np.zeros((2, dim), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        r = float(self._settings.measurement_noise_px) ** 2
        R = np.eye(2, dtype=np.float32) * max(r, 1e-4)
        z = np.asarray(measurement_rc, dtype=np.float32)
        y = z - (H @ self._state.kalman_x)
        S = H @ self._state.kalman_P @ H.T + R
        try:
            K = self._state.kalman_P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self._state.kalman_P @ H.T @ np.linalg.pinv(S)
        self._state.kalman_x = (self._state.kalman_x + K @ y).astype(np.float32)
        I = np.eye(dim, dtype=np.float32)
        self._state.kalman_P = ((I - K @ H) @ self._state.kalman_P).astype(np.float32)

    def _kalman_gate_radius(self) -> float:
        if not self._settings.kalman_enabled or self._state is None or self._state.kalman_P is None:
            return self._settings.max_prediction_error_px
        sigma = float(np.sqrt(max(self._state.kalman_P[0, 0], self._state.kalman_P[1, 1], 1e-9)))
        return float(np.clip(
            self._settings.kalman_gate_sigma * sigma + self._settings.expected_diameter_px,
            self._settings.min_search_radius_px,
            self._settings.max_dynamic_search_radius_px,
        ))

    # ------------------------------------------------------------------
    # Candidate extraction/scoring
    # ------------------------------------------------------------------

    def _tiny_candidates(
        self,
        *,
        probability: np.ndarray,
        roi_lab: np.ndarray,
        roi_origin_xy: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> list[_Candidate]:
        """Tiny-feature measurement with local colour contrast validation.

        The peak must be colour-likely, locally distinctive, and its signed
        centre-vs-ring Lab offset must match the direction learned during
        initialization.  This prevents a clicked dark spot from being confused
        with arbitrary dark wood-grain pixels.
        """
        assert self._state is not None
        self._last_rejection_counts = {
            "small_component": 0,
            "colour": 0,
            "local_contrast": 0,
            "direction": 0,
            "size": 0,
            "motion": 0,
            "ambiguity": 0,
            "mosse": 0,
        }
        self._last_failure_reasons = []
        if self._profiling_enabled:
            self._last_candidate_scan_metrics = {
                "threshold_passes": 1,
                "components_found": 1,
                "components_scored": 1,
                "accepted_candidate_count": 0,
                "successful_threshold": "peak",
            }

        x0, y0 = roi_origin_xy
        if probability.size == 0:
            self._last_rejection_counts["colour"] += 1
            self._add_failure_reason("no_peak_found")
            return []

        # Motion-prior-weighted peak: colour likelihood still dominates, but a
        # far dark speck should not beat a nearby plausible feature.
        rows_all, cols_all = np.indices(probability.shape, dtype=np.float32)
        pred_r = float(predicted_center_rc[0] - y0)
        pred_c = float(predicted_center_rc[1] - x0)
        gate = self._kalman_gate_radius()
        dist2 = (rows_all - pred_r) ** 2 + (cols_all - pred_c) ** 2
        motion_prior = np.exp(-0.5 * dist2 / max((gate / 2.0) ** 2, 1.0)).astype(np.float32)
        search_score = probability.astype(np.float32) * (0.65 + 0.35 * motion_prior)

        mosse_used = False
        mosse_metrics: dict[str, object] = {}
        if self._settings.tiny_mosse_mode != "off" and self._state.mosse is not None:
            metrics = self._mosse_locate(probability, np.asarray([pred_r, pred_c], dtype=np.float32))
            if metrics is not None:
                mosse_metrics = metrics
                mosse_peak_local = np.asarray(metrics["peak_local_rc"], dtype=np.float32)
                mr = int(np.clip(round(float(mosse_peak_local[0])), 0, probability.shape[0] - 1))
                mc = int(np.clip(round(float(mosse_peak_local[1])), 0, probability.shape[1] - 1))
                mosse_psr = float(metrics["psr"])
                mosse_margin = float(metrics["peak_margin"])
                mosse_scale_factor = float(metrics.get("scale_factor", 1.0))
                self._set_failure_diagnostic("mosse_enabled", True)
                self._set_failure_diagnostic("mosse_mode", self._settings.tiny_mosse_mode)
                self._set_failure_diagnostic("mosse_psr", mosse_psr)
                self._set_failure_diagnostic("mosse_psr_threshold", float(self._settings.tiny_mosse_psr_threshold))
                self._set_failure_diagnostic("mosse_peak_margin", mosse_margin)
                self._set_failure_diagnostic("mosse_peak_margin_threshold", float(self._settings.tiny_mosse_peak_margin_threshold))
                self._set_failure_diagnostic("mosse_peak_rc", [float(y0 + mr), float(x0 + mc)])
                self._set_failure_diagnostic("mosse_scale_factor", mosse_scale_factor)
                if (
                    mosse_psr >= self._settings.tiny_mosse_psr_threshold
                    and mosse_margin >= self._settings.tiny_mosse_peak_margin_threshold
                ):
                    peak_r, peak_c = mr, mc
                    mosse_used = True
                elif self._settings.tiny_mosse_mode == "primary":
                    self._last_rejection_counts["mosse"] += 1
                    self._add_failure_reason("mosse_response_rejected")
                    return []
            elif self._settings.tiny_mosse_mode == "primary":
                self._last_rejection_counts["mosse"] += 1
                self._add_failure_reason("mosse_unavailable")
                return []

        if not mosse_used:
            peak_index = int(np.argmax(search_score))
            peak_r, peak_c = np.unravel_index(peak_index, search_score.shape)
        peak_value = float(probability[peak_r, peak_c])
        self._set_failure_diagnostic("tiny_peak_score", peak_value)
        self._set_failure_diagnostic("tiny_peak_threshold", float(self._settings.tiny_peak_threshold))
        self._set_failure_diagnostic("tiny_peak_rc", [float(y0 + peak_r), float(x0 + peak_c)])
        self._set_failure_diagnostic("tiny_motion_prior_at_peak", float(motion_prior[peak_r, peak_c]))
        self._last_peak_center_rc = np.array([float(y0 + peak_r), float(x0 + peak_c)], dtype=np.float32)
        if self._debug_recorder.enabled:
            self._debug_current["peak_rc"] = [float(y0 + peak_r), float(x0 + peak_c)]
            self._debug_current["peak_score"] = peak_value
        if peak_value < self._settings.tiny_peak_threshold:
            self._last_rejection_counts["colour"] += 1
            self._add_failure_reason("peak_below_threshold")
            return []

        radius = max(2, int(round(self._state.expected_width / 2.0)))
        r0 = max(0, peak_r - radius)
        r1 = min(probability.shape[0], peak_r + radius + 1)
        c0 = max(0, peak_c - radius)
        c1 = min(probability.shape[1], peak_c + radius + 1)
        self._set_failure_diagnostic("centroid_window_xywh", [
            int(x0 + c0), int(y0 + r0), int(c1 - c0), int(r1 - r0)
        ])
        if self._debug_recorder.enabled:
            self._debug_current["centroid_window_xywh"] = [
                int(x0 + c0), int(y0 + r0), int(c1 - c0), int(r1 - r0)
            ]
        window = probability[r0:r1, c0:c1]
        if window.size == 0:
            self._last_rejection_counts["small_component"] += 1
            self._add_failure_reason("no_peak_found")
            return []

        rows, cols = np.indices(window.shape, dtype=np.float32)
        weights = window.astype(np.float32)
        weights = np.clip(weights - max(0.0, peak_value * 0.25), 0.0, 1.0)
        total = float(weights.sum())
        if total <= 1e-8:
            self._last_rejection_counts["colour"] += 1
            self._add_failure_reason("peak_below_threshold")
            return []

        center_local_r = r0 + float((weights * rows).sum() / total)
        center_local_c = c0 + float((weights * cols).sum() / total)
        center_rc = np.array([y0 + center_local_r, x0 + center_local_c], dtype=np.float32)

        # Candidate local centre-vs-ring Lab contrast.
        rr0 = max(0, peak_r - radius * 2)
        rr1 = min(probability.shape[0], peak_r + radius * 2 + 1)
        cc0 = max(0, peak_c - radius * 2)
        cc1 = min(probability.shape[1], peak_c + radius * 2 + 1)
        ring_prob_patch = probability[rr0:rr1, cc0:cc1]
        ring_prob_mean = float(np.mean(ring_prob_patch)) if ring_prob_patch.size else 0.0
        window_mean = float(np.mean(window))
        probability_contrast = window_mean - ring_prob_mean

        candidate_lab_window = roi_lab[r0:r1, c0:c1].reshape(-1, 3)
        outer_lab_patch = roi_lab[rr0:rr1, cc0:cc1]
        ring_mask = np.ones(outer_lab_patch.shape[:2], dtype=bool)
        inner_y0, inner_y1 = r0 - rr0, r1 - rr0
        inner_x0, inner_x1 = c0 - cc0, c1 - cc0
        ring_mask[max(0, inner_y0):max(0, inner_y1), max(0, inner_x0):max(0, inner_x1)] = False
        candidate_ring_lab = outer_lab_patch[ring_mask]
        if candidate_lab_window.size == 0 or candidate_ring_lab.size == 0:
            self._last_rejection_counts["local_contrast"] += 1
            self._add_failure_reason("contrast_below_threshold")
            return []
        center_mean_lab = np.median(candidate_lab_window, axis=0).astype(np.float32)
        ring_mean_lab = np.median(candidate_ring_lab.reshape(-1, 3), axis=0).astype(np.float32)
        candidate_delta_lab = center_mean_lab - ring_mean_lab
        lab_contrast = self._weighted_lab_norm(candidate_delta_lab)

        reference_delta = self._state.initialization.tiny_reference_delta_lab
        direction_cosine = 1.0
        if reference_delta is not None:
            direction_cosine = self._weighted_lab_cosine(candidate_delta_lab, reference_delta)

        self._set_failure_diagnostic("tiny_probability_contrast", float(probability_contrast))
        self._set_failure_diagnostic("tiny_contrast_threshold", float(self._settings.tiny_min_contrast))
        self._set_failure_diagnostic("tiny_lab_contrast", float(lab_contrast))
        self._set_failure_diagnostic("tiny_init_contrast_threshold", float(self._settings.tiny_init_contrast_threshold))
        self._set_failure_diagnostic("tiny_direction_cosine", float(direction_cosine))
        self._set_failure_diagnostic("tiny_direction_min_cosine", float(self._settings.tiny_direction_min_cosine))
        if self._debug_recorder.enabled:
            self._debug_current["contrast_score"] = float(probability_contrast)
            self._debug_current["lab_contrast"] = float(lab_contrast)
            self._debug_current["direction_cosine"] = float(direction_cosine)

        if probability_contrast < self._settings.tiny_min_contrast and peak_value < 0.85:
            self._last_rejection_counts["local_contrast"] += 1
            self._add_failure_reason("contrast_below_threshold")
            return []
        # Require actual Lab distinctiveness as well, not only probability-map contrast.
        if lab_contrast < 0.50 * self._settings.tiny_init_contrast_threshold and peak_value < 0.90:
            self._last_rejection_counts["local_contrast"] += 1
            self._add_failure_reason("contrast_below_threshold")
            return []
        if direction_cosine < self._settings.tiny_direction_min_cosine:
            self._last_rejection_counts["direction"] += 1
            self._add_failure_reason("contrast_direction_mismatch")
            return []

        prediction_error = float(np.linalg.norm(center_rc - predicted_center_rc))
        self._set_failure_diagnostic("innovation_distance", prediction_error)
        self._set_failure_diagnostic("kalman_gate_radius", float(gate))
        self._set_failure_diagnostic("measured_candidate_rc", center_rc.astype(float).tolist())
        if prediction_error > gate and not self._settings.accept_within_search_region:
            self._last_rejection_counts["motion"] += 1
            self._add_failure_reason("outside_kalman_gate")
            return []

        motion_score = float(np.exp(-0.5 * (prediction_error / max(gate / 2.0, 1.0)) ** 2))
        probability_score = np.clip(probability_contrast / max(self._settings.tiny_min_contrast * 4.0, 1e-6), 0.0, 1.0)
        lab_score = np.clip(lab_contrast / max(self._state.initialization.tiny_init_contrast, self._settings.tiny_init_contrast_threshold, 1e-6), 0.0, 1.0)
        direction_score = np.clip((direction_cosine + 1.0) / 2.0, 0.0, 1.0)
        colour_score = float(np.clip(
            0.45 * peak_value + 0.20 * probability_score + 0.20 * lab_score + 0.15 * direction_score,
            0.0, 1.0,
        ))
        confidence = float(np.clip(0.72 * colour_score + 0.28 * motion_score, 0.0, 1.0))

        scale_factor = float(
            mosse_metrics.get("scale_factor", self._state.mosse.scale_factor if self._state.mosse else 1.0)
            if mosse_used else (self._state.mosse.scale_factor if self._state.mosse else 1.0)
        )
        diameter = max(1.0, float(self._state.initialization.expected_width) * scale_factor)
        half = max(1, int(round(diameter / 2.0)))
        bbox = (
            int(round(center_rc[1])) - half,
            int(round(center_rc[0])) - half,
            2 * half + 1,
            2 * half + 1,
        )
        contour = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0] + bbox[2] - 1, bbox[1]],
                [bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1],
                [bbox[0], bbox[1] + bbox[3] - 1],
            ],
            dtype=np.float32,
        )
        candidate = _Candidate(
            center_rc=center_rc,
            tight_bbox_xywh=bbox,
            contour_xy_global=contour,
            confidence=confidence,
            colour_score=colour_score,
            shape_score=1.0,
            motion_score=motion_score,
            area=float(np.pi * (diameter / 2.0) ** 2),
            width=diameter,
            height=diameter,
            circularity=0.0,
            rectangularity=0.0,
            centre_difference=float("nan"),
            adaptive_source="mosse_probability" if mosse_used else "classical",
            scale_factor=scale_factor,
        )
        self._set_failure_diagnostic("mosse_used", bool(mosse_used))
        if mosse_metrics:
            self._set_failure_diagnostic("mosse_response_peak", float(mosse_metrics.get("response_peak", 0.0)))
        self._pending_mosse_update = (
            probability,
            roi_origin_xy,
            center_rc.astype(np.float32).copy(),
        ) if mosse_used else None
        if self._profiling_enabled:
            self._last_candidate_scan_metrics["accepted_candidate_count"] = 1
            self._last_candidate_scan_metrics["tiny_lab_contrast"] = float(lab_contrast)
            self._last_candidate_scan_metrics["tiny_direction_cosine"] = float(direction_cosine)
            self._last_candidate_scan_metrics["mosse_used"] = bool(mosse_used)
        return [candidate]

    def _normal_candidates_with_mosse(
        self,
        *,
        probability: np.ndarray,
        roi_lab: np.ndarray,
        roi_origin_xy: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> list[_Candidate]:
        """Use a reliable normal MOSSE peak to verify only a local component."""
        mode = self._settings.normal_mosse_mode
        if mode == "off":
            return self._accepted_candidates(
                probability=probability, roi_lab=roi_lab,
                roi_origin_xy=roi_origin_xy, predicted_center_rc=predicted_center_rc,
            )
        assert self._state is not None
        x0, y0 = roi_origin_xy
        predicted_local = np.asarray([predicted_center_rc[0] - y0, predicted_center_rc[1] - x0], dtype=np.float32)
        if self._state.mosse is None:
            self._state.mosse = self._mosse_init_or_replace(probability, predicted_local)
            if mode == "primary":
                self._last_rejection_counts = {"mosse": 1}
                self._add_failure_reason("mosse_warming_up")
                return []
            return self._accepted_candidates(
                probability=probability, roi_lab=roi_lab,
                roi_origin_xy=roi_origin_xy, predicted_center_rc=predicted_center_rc,
            )
        started = perf_counter() if self._profiling_enabled else 0.0
        metrics = self._mosse_locate(probability, predicted_local)
        if self._profiling_enabled:
            self._profile_add("normal_mosse_locate_ms", perf_counter() - started)
        if metrics is not None and float(metrics["psr"]) >= self._settings.tiny_mosse_psr_threshold and float(metrics["peak_margin"]) >= self._settings.tiny_mosse_peak_margin_threshold:
            peak = np.asarray(metrics["peak_local_rc"], dtype=np.float32)
            radius = max(8, int(round(self._settings.expected_diameter_px * 1.5)))
            r0, r1 = max(0, int(round(peak[0])) - radius), min(probability.shape[0], int(round(peak[0])) + radius + 1)
            c0, c1 = max(0, int(round(peak[1])) - radius), min(probability.shape[1], int(round(peak[1])) + radius + 1)
            local = self._accepted_candidates(
                probability=probability[r0:r1, c0:c1], roi_lab=roi_lab[r0:r1, c0:c1],
                roi_origin_xy=(x0 + c0, y0 + r0), predicted_center_rc=predicted_center_rc,
            )
            if local:
                scale_factor = float(metrics.get("scale_factor", 1.0))
                if self._settings.mosse_scale_adaptation:
                    local = [replace(candidate, scale_factor=scale_factor, adaptive_source="mosse_normal") for candidate in local]
                    self._set_failure_diagnostic("mosse_scale_factor", scale_factor)
                self._pending_mosse_update = (probability, roi_origin_xy, local[0].center_rc)
                self._last_candidate_scan_metrics["normal_mosse_used"] = True
                self._last_candidate_scan_metrics["normal_mosse_primary_used"] = mode == "primary"
                return local
            if mode == "primary":
                self._last_rejection_counts["mosse"] = self._last_rejection_counts.get("mosse", 0) + 1
                self._add_failure_reason("mosse_local_verification_rejected")
                return []
        elif mode == "primary":
            self._last_rejection_counts = {"mosse": 1}
            self._add_failure_reason("mosse_response_rejected")
            return []
        self._last_candidate_scan_metrics["normal_mosse_used"] = False
        return self._accepted_candidates(
            probability=probability, roi_lab=roi_lab,
            roi_origin_xy=roi_origin_xy, predicted_center_rc=predicted_center_rc,
        )

    def _accepted_candidates(
        self,
        *,
        probability: np.ndarray,
        roi_lab: np.ndarray,
        roi_origin_xy: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> list[_Candidate]:
        self._last_rejection_counts = {
            "small_component": 0,
            "colour": 0,
            "size": 0,
            "motion": 0,
            "ambiguity": 0,
        }
        self._last_failure_reasons = []
        if self._profiling_enabled:
            self._last_candidate_scan_metrics = {
                "threshold_passes": 0,
                "components_found": 0,
                "components_scored": 0,
                "accepted_candidate_count": 0,
                "successful_threshold": "",
            }

        for threshold in self._settings.probability_thresholds:
            if self._profiling_enabled:
                self._last_candidate_scan_metrics["threshold_passes"] += 1

            stage_start = perf_counter() if self._profiling_enabled else 0.0
            binary = (probability >= threshold).astype(np.uint8)
            if self._settings.morphology_close_iterations:
                binary = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    np.ones((3, 3), np.uint8),
                    iterations=self._settings.morphology_close_iterations,
                )
            if self._profiling_enabled:
                self._profile_add("threshold_morphology_ms", perf_counter() - stage_start)

            stage_start = perf_counter() if self._profiling_enabled else 0.0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            if self._profiling_enabled:
                self._profile_add("connected_components_ms", perf_counter() - stage_start)
                self._last_candidate_scan_metrics["components_found"] += max(0, int(count) - 1)
            self._set_failure_diagnostic("last_threshold", float(threshold))
            self._set_failure_diagnostic("components_found_last_threshold", max(0, int(count) - 1))

            accepted: list[_Candidate] = []
            for label in range(1, count):
                component_area = int(stats[label, cv2.CC_STAT_AREA])
                if component_area < self._settings.min_component_area_px:
                    self._last_rejection_counts["small_component"] += 1
                    self._add_failure_reason("no_peak_found")
                    continue
                if self._profiling_enabled:
                    self._last_candidate_scan_metrics["components_scored"] += 1
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                width = int(stats[label, cv2.CC_STAT_WIDTH])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                candidate = self._score_candidate(
                    probability=probability,
                    roi_lab=roi_lab,
                    component_mask=(labels == label).astype(np.uint8),
                    roi_origin_xy=roi_origin_xy,
                    predicted_center_rc=predicted_center_rc,
                )
                if candidate is not None:
                    accepted.append(candidate)
            if accepted:
                if self._profiling_enabled:
                    self._last_candidate_scan_metrics["accepted_candidate_count"] = len(accepted)
                    self._last_candidate_scan_metrics["successful_threshold"] = float(threshold)
                return accepted
        return []

    def _score_candidate(
        self,
        *,
        probability: np.ndarray,
        roi_lab: np.ndarray,
        component_mask: np.ndarray,
        roi_origin_xy: tuple[int, int],
        predicted_center_rc: np.ndarray,
    ) -> Optional[_Candidate]:
        assert self._state is not None
        candidate_started = perf_counter() if self._profiling_enabled else 0.0
        x0, y0 = roi_origin_xy

        stage_start = perf_counter() if self._profiling_enabled else 0.0
        contour = self._outer_contour(component_mask)
        contour_xy = contour.reshape(-1, 2).astype(np.float32)
        if self._settings.refine_boundary_on_tracking:
            contour_xy = self._refine_contour_by_probability_gradient(contour, probability)
        if self._profiling_enabled:
            self._profile_add("candidate_contour_ms", perf_counter() - stage_start)

        stage_start = perf_counter() if self._profiling_enabled else 0.0
        center_local_rc = self._weighted_centroid(probability, component_mask)
        full_blob_center_rc = np.array(
            [y0 + center_local_rc[0], x0 + center_local_rc[1]], dtype=np.float32
        )
        peak_window_center_rc, peak_rc, peak_score, centroid_window_xywh = self._normal_peak_window_center(
            probability=probability,
            component_mask=component_mask,
            roi_origin_xy=roi_origin_xy,
            fallback_center_local_rc=center_local_rc,
        )
        center_rc = full_blob_center_rc.copy()
        center_mode_used = "full_blob"
        centroid_peak_distance = float(np.linalg.norm(full_blob_center_rc - peak_window_center_rc))
        if self._profiling_enabled:
            self._profile_add("candidate_centroid_ms", perf_counter() - stage_start)

        stage_start = perf_counter() if self._profiling_enabled else 0.0
        geometry = self._geometry(contour_xy, center_local_rc)
        if self._profiling_enabled:
            self._profile_add("candidate_geometry_ms", perf_counter() - stage_start)

        stage_start = perf_counter() if self._profiling_enabled else 0.0
        ring_mask = self._background_ring(component_mask)
        quality = self._colour_quality(probability, component_mask, ring_mask)
        luminance_quality = self._luminance_polarity_quality(roi_lab, component_mask, ring_mask)
        if self._profiling_enabled:
            self._profile_add("candidate_colour_quality_ms", perf_counter() - stage_start)

        if self._settings.colour_diagnostics_enabled:
            self._set_failure_diagnostic("candidate_l_mean", float(luminance_quality["candidate_l_mean"]))
            self._set_failure_diagnostic("candidate_l_median", float(luminance_quality["candidate_l_median"]))
            self._set_failure_diagnostic("local_background_l_mean", float(luminance_quality["local_background_l_mean"]))
            self._set_failure_diagnostic("init_foreground_l_mean", float(self._state.initialization.foreground_l_mean))
            self._set_failure_diagnostic("init_background_l_mean", float(self._state.initialization.background_l_mean))
            self._set_failure_diagnostic("luminance_polarity", str(luminance_quality["luminance_polarity"]))
            self._set_failure_diagnostic("luminance_error", float(luminance_quality["luminance_error"]))
            self._set_failure_diagnostic("lab_l_weight", float(self._settings.lab_weights[0]))
            self._set_failure_diagnostic("lab_a_weight", float(self._settings.lab_weights[1]))
            self._set_failure_diagnostic("lab_b_weight", float(self._settings.lab_weights[2]))

        if not bool(luminance_quality["luminance_polarity_pass"]):
            self._last_rejection_counts["colour"] += 1
            self._add_failure_reason("luminance_polarity_rejected")
            if self._profiling_enabled:
                self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
            return None

        if (
            quality["median_inside_probability"] < self._settings.min_inside_median_probability
            or quality["strong_inside_fraction"] < self._settings.min_inside_strong_fraction
            or quality["probability_margin"] < self._settings.min_probability_margin
        ):
            self._last_rejection_counts["colour"] += 1
            self._add_failure_reason("colour_rejected")
            if self._profiling_enabled:
                self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
            return None

        area_ratio = geometry["area"] / max(self._state.expected_area, 1e-9)
        expected_axes = sorted([self._state.expected_width, self._state.expected_height])
        measured_axes = sorted([geometry["width"], geometry["height"]])
        axis_ratios = [
            measured_axes[i] / max(expected_axes[i], 1e-9) for i in range(2)
        ]
        if not (self._settings.min_area_ratio <= area_ratio <= self._settings.max_area_ratio):
            self._last_rejection_counts["size"] += 1
            self._add_failure_reason("size_rejected")
            if self._profiling_enabled:
                self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
            return None
        if not all(
            self._settings.min_axis_size_ratio <= ratio <= self._settings.max_axis_size_ratio
            for ratio in axis_ratios
        ):
            self._last_rejection_counts["size"] += 1
            self._add_failure_reason("size_rejected")
            if self._profiling_enabled:
                self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
            return None

        long_axis = max(float(geometry["width"]), float(geometry["height"]), 1e-6)
        short_axis = max(min(float(geometry["width"]), float(geometry["height"])), 1e-6)
        blob_elongation = float(long_axis / short_axis)
        requested_center_mode = self._settings.measurement_center_mode
        auto_peak_due_to_elongation = blob_elongation >= 1.70 and centroid_peak_distance >= 2.0
        auto_peak_due_to_offset = centroid_peak_distance >= max(3.0, 0.30 * self._settings.expected_diameter_px)
        if requested_center_mode == "peak_window":
            center_rc = peak_window_center_rc.astype(np.float32)
            center_mode_used = "peak_window"
        elif requested_center_mode == "auto" and (auto_peak_due_to_elongation or auto_peak_due_to_offset):
            center_rc = peak_window_center_rc.astype(np.float32)
            center_mode_used = "auto_peak_window"
        self._last_peak_center_rc = peak_rc.astype(np.float32)
        self._set_failure_diagnostic("measurement_center_mode", requested_center_mode)
        self._set_failure_diagnostic("measurement_center_mode_used", center_mode_used)
        self._set_failure_diagnostic("centroid_peak_distance_px", float(centroid_peak_distance))
        self._set_failure_diagnostic("blob_elongation", float(blob_elongation))

        # Shape variation from motion blur/compression contributes to confidence,
        # but does not alone reject an otherwise colour/motion-valid lock.
        prediction_error = float(np.linalg.norm(center_rc - predicted_center_rc))
        max_motion_error = self._kalman_gate_radius() if self._settings.kalman_enabled else (
            self._settings.max_prediction_error_px
            + self._state.consecutive_misses * self._settings.miss_motion_allowance_px
        )
        if prediction_error > max_motion_error and not self._settings.accept_within_search_region:
            self._last_rejection_counts["motion"] += 1
            self._add_failure_reason(
                "outside_kalman_gate" if self._settings.kalman_enabled else "outside_motion_limit"
            )
            if self._profiling_enabled:
                self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
            return None

        if self._settings.shape_validation == "strict":
            if self._state.initialization.shape_model == "circle_like" and geometry["circularity"] < self._settings.min_circle_circularity:
                self._last_rejection_counts["size"] += 1
                self._add_failure_reason("size_rejected")
                if self._profiling_enabled:
                    self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
                return None
            if self._state.initialization.shape_model == "rectangle" and geometry["rectangularity"] < self._settings.min_rectangle_rectangularity:
                self._last_rejection_counts["size"] += 1
                self._add_failure_reason("size_rejected")
                if self._profiling_enabled:
                    self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
                return None
            if np.isfinite(geometry["center_difference"]) and geometry["center_difference"] > self._settings.max_geometric_center_difference_px:
                self._last_rejection_counts["size"] += 1
                self._add_failure_reason("size_rejected")
                if self._profiling_enabled:
                    self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
                return None

        colour_score = self._colour_score(quality)
        shape_score = 1.0 if self._settings.shape_validation == "off" else self._shape_score(geometry, area_ratio, axis_ratios)
        motion_score = float(
            np.exp(-0.5 * (prediction_error / max(max_motion_error / 2.0, 1.0)) ** 2)
        )
        if self._settings.shape_validation == "off":
            confidence = float(np.clip(0.68 * colour_score + 0.32 * motion_score, 0.0, 1.0))
        else:
            confidence = float(np.clip(
                0.48 * colour_score + 0.32 * shape_score + 0.20 * motion_score,
                0.0, 1.0,
            ))

        local_bbox = self._mask_bbox(component_mask)
        global_bbox = (x0 + local_bbox[0], y0 + local_bbox[1], local_bbox[2], local_bbox[3])
        contour_global = contour_xy.copy()
        contour_global[:, 0] += x0
        contour_global[:, 1] += y0
        candidate = _Candidate(
            center_rc=center_rc,
            tight_bbox_xywh=global_bbox,
            contour_xy_global=contour_global,
            confidence=confidence,
            colour_score=colour_score,
            shape_score=shape_score,
            motion_score=motion_score,
            area=float(geometry["area"]),
            width=float(geometry["width"]),
            height=float(geometry["height"]),
            circularity=float(geometry["circularity"]),
            rectangularity=float(geometry["rectangularity"]),
            centre_difference=float(geometry["center_difference"]),
            angle_deg=float(geometry.get("angle_deg", 0.0)),
            center_mode_used=center_mode_used,
            full_blob_center_rc=full_blob_center_rc.astype(np.float32),
            peak_window_center_rc=peak_window_center_rc.astype(np.float32),
            peak_rc=peak_rc.astype(np.float32),
            peak_score=float(peak_score),
            centroid_peak_distance_px=float(centroid_peak_distance),
            blob_elongation=float(blob_elongation),
            centroid_window_xywh=centroid_window_xywh,
            candidate_l_mean=float(luminance_quality["candidate_l_mean"]),
            candidate_l_median=float(luminance_quality["candidate_l_median"]),
            local_background_l_mean=float(luminance_quality["local_background_l_mean"]),
            luminance_polarity=str(luminance_quality["luminance_polarity"]),
            luminance_polarity_pass=bool(luminance_quality["luminance_polarity_pass"]),
            luminance_error=float(luminance_quality["luminance_error"]),
        )
        if self._profiling_enabled:
            self._profile_add("candidate_scoring_total_ms", perf_counter() - candidate_started)
        return candidate

    def _accept_candidate(self, candidate: _Candidate, frame_index: int) -> None:
        assert self._state is not None
        state = self._state
        state.previous_valid_center_rc = state.last_valid_center_rc.copy()
        state.last_valid_center_rc = candidate.center_rc.copy()
        state.last_valid_frame_index = int(frame_index)
        state.consecutive_misses = 0
        if (
            self._settings.mosse_scale_adaptation
            and state.mosse is not None
            and candidate.adaptive_source in {"mosse_probability", "mosse_normal"}
            and candidate.confidence >= self._settings.high_confidence_threshold
        ):
            old_scale = float(state.mosse.scale_factor)
            new_scale = float(np.clip(
                candidate.scale_factor,
                old_scale * (1.0 - self._settings.mosse_scale_max_step),
                old_scale * (1.0 + self._settings.mosse_scale_max_step),
            ))
            state.mosse.scale_factor = new_scale
            state.expected_width = float(state.initialization.expected_width * new_scale)
            state.expected_height = float(state.initialization.expected_height * new_scale)
            state.expected_area = float(state.initialization.expected_area * new_scale * new_scale)
            self._set_failure_diagnostic("mosse_scale_factor", new_scale)
        if self._settings.kalman_enabled:
            if (
                self._settings.point_mode == "tiny"
                and candidate.confidence < self._settings.tiny_update_confidence_threshold
            ):
                # Keep the measured point as the exported data, but do not let
                # a borderline tiny-feature match pull the motion model toward
                # a wrong dark speck.  Future search remains based on the last
                # stronger Kalman state until a confident measurement arrives.
                self._last_rejection_counts["weak_update"] = (
                    self._last_rejection_counts.get("weak_update", 0) + 1
                )
            else:
                self._kalman_update(candidate.center_rc)

        if (
            (
                (self._settings.point_mode == "tiny" and self._settings.tiny_mosse_mode != "off")
                or (self._settings.point_mode == "normal" and self._settings.normal_mosse_mode == "assisted")
            )
            and candidate.confidence >= self._settings.tiny_update_confidence_threshold
            and self._pending_mosse_update is not None
        ):
            probability, roi_origin_xy, center_rc = self._pending_mosse_update
            self._mosse_update(probability, roi_origin_xy, center_rc)
            self._set_failure_diagnostic("mosse_update_applied", True)
        else:
            self._set_failure_diagnostic("mosse_update_applied", False)
        self._pending_mosse_update = None

        if (
            self._settings.adapt_geometry
            and candidate.confidence >= self._settings.high_confidence_threshold
        ):
            alpha = self._settings.geometry_adaptation_alpha
            state.expected_area = (1.0 - alpha) * state.expected_area + alpha * candidate.area
            state.expected_width = (1.0 - alpha) * state.expected_width + alpha * candidate.width
            state.expected_height = (1.0 - alpha) * state.expected_height + alpha * candidate.height
        state.geometry_history.append({
            "width": float(candidate.width),
            "height": float(candidate.height),
            "area": float(candidate.area),
            "angle_deg": float(candidate.angle_deg),
        })
        state.center_history.append(candidate.center_rc.astype(np.float32).copy())
        if candidate.confidence >= self._settings.high_confidence_threshold:
            self._learn_feature_profile(candidate)

    @staticmethod
    def _validation_vector(candidate: _Candidate) -> np.ndarray:
        """Stable, scale-friendly descriptor used for learned candidate rejection."""
        return np.asarray([
            np.log(max(candidate.area, 1e-3)),
            np.log(max(candidate.width, 1e-3)),
            np.log(max(candidate.height, 1e-3)),
            float(candidate.circularity),
            float(candidate.rectangularity),
            float(candidate.colour_score),
            float(candidate.confidence),
        ], dtype=np.float32)

    def _passes_learned_validation(self, candidate: _Candidate) -> bool:
        """Reject candidates that disagree with the accumulated trusted feature profile."""
        if self._state is None:
            return True
        state = self._state
        samples = state.validation_samples
        # The existing explicit gates own the warm-up period.  A full profile
        # needs enough examples to distinguish genuine variability from noise.
        if len(samples) < 32:
            self._set_failure_diagnostic("learned_validation_state", "warming_up")
            self._set_failure_diagnostic("learned_validation_samples", len(samples))
            return True
        if state.validation_dirty or state.validation_median is None or state.validation_scale is None:
            data = np.asarray(samples, dtype=np.float32)
            state.validation_median = np.median(data, axis=0)
            mad = np.median(np.abs(data - state.validation_median), axis=0)
            state.validation_scale = np.maximum(2.5 * 1.4826 * mad, np.asarray([
                0.22, 0.22, 0.22, 0.09, 0.09, 0.10, 0.10,
            ], dtype=np.float32))
            state.validation_dirty = False
        median = state.validation_median
        scale = state.validation_scale
        z = np.abs(self._validation_vector(candidate) - median) / scale
        if self._settings.mosse_scale_adaptation:
            # Area and axes are deliberately allowed to evolve under the
            # scale filter; colour, shape, and confidence still guard identity.
            z[:3] = 0.0
        # One mildly unusual property is normal under compression and blur.
        # Multiple independent deviations, or one extreme deviation, indicates
        # a different feature rather than ordinary measurement noise.
        rejected = bool(np.max(z) > 14.0 or np.count_nonzero(z > 6.0) >= 3)
        self._set_failure_diagnostic("learned_validation_state", "active")
        self._set_failure_diagnostic("learned_validation_samples", len(samples))
        self._set_failure_diagnostic("learned_validation_max_z", float(np.max(z)))
        return not rejected

    def _learn_feature_profile(self, candidate: _Candidate) -> None:
        assert self._state is not None
        state = self._state
        value = self._validation_vector(candidate)
        state.validation_seen += 1
        # Fixed-size deterministic reservoir: representative of the complete
        # run without retaining a Python object for every video frame.
        capacity = 512
        if len(state.validation_samples) < capacity:
            state.validation_samples.append(value)
            state.validation_dirty = len(state.validation_samples) % 16 == 0
            return
        seen = state.validation_seen
        slot = ((seen * 1103515245 + 12345) & 0x7FFFFFFF) % seen
        if slot < capacity:
            state.validation_samples[int(slot)] = value
            state.validation_dirty = True

    def _seed_validation_profile(self, state: _TrackingState) -> None:
        samples = []
        for value in self._preset_validation_samples[:512]:
            array = np.asarray(value, dtype=np.float32)
            if array.shape == (7,) and np.all(np.isfinite(array)):
                samples.append(array)
        state.validation_samples = samples
        state.validation_seen = len(samples)
        state.validation_dirty = True

    def export_learned_validation_profile(self) -> list[list[float]]:
        if self._state is None:
            return []
        return [np.asarray(value, dtype=np.float32).astype(float).tolist() for value in self._state.validation_samples]

    def _measurement_failure(self, frame_index: int) -> FrameResult:
        assert self._state is not None
        self._pending_mosse_update = None
        self._state.consecutive_misses += 1
        self._uncertain_count = self._state.consecutive_misses
        self.status = (
            TrackerStatus.LOST
            if self._state.consecutive_misses >= self.config.lost_after_frames
            else TrackerStatus.UNCERTAIN
        )
        alternatives: dict[str, np.ndarray] = {}
        if self._last_peak_center_rc is not None:
            alternatives["colour_peak"] = self._last_peak_center_rc.astype(np.float32).copy()
        if self._last_predicted_center_rc is not None:
            alternatives["kalman_mean"] = self._last_predicted_center_rc.astype(np.float32).copy()
        return FrameResult(
            frame_index, self.status, center=None, hessian_score=0.0,
            alt_centers=alternatives,
            failure_reasons=self._failure_reasons_from_counts(),
            diagnostic_values=self._failure_diagnostics(),
        )

    # ------------------------------------------------------------------
    # Colour/geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_lab(roi_bgr: np.ndarray) -> np.ndarray:
        if roi_bgr.ndim != 3 or roi_bgr.shape[2] != 3:
            raise ValueError("PointFastTracker requires BGR colour frames with shape (H, W, 3).")
        if roi_bgr.dtype != np.uint8:
            raise ValueError("PointFastTracker requires uint8 BGR frame data.")
        roi_float = roi_bgr.astype(np.float32) / 255.0
        return cv2.cvtColor(roi_float, cv2.COLOR_BGR2Lab).astype(np.float32)

    @staticmethod
    def _crop_bgr_roi(
        frame_bgr: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return a CPU ROI view; FrameBuffer already retained decoded BGR data."""
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(
                "PointFastTracker requires CPU frames. Ensure TrackerManager routes "
                "trackers with requires_cpu_frame=True through get_frame_cpu()."
            )
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("PointFastTracker requires uint8 BGR CPU frames.")
        x, y, width, height = bbox_xywh
        return frame_bgr[y : y + height, x : x + width]

    def _foreground_probability(
        self,
        roi_lab: np.ndarray,
        foreground: _GaussianLabModel | _TinyLabDistanceModel,
        background: _GaussianLabModel | _TinyLabDistanceModel,
    ) -> np.ndarray:
        if isinstance(foreground, _GaussianLabModel) and isinstance(background, _GaussianLabModel):
            weights = np.asarray(self._settings.lab_weights, dtype=np.float32)
            score = (
                foreground.log_likelihood_weighted(roi_lab, weights)
                - background.log_likelihood_weighted(roi_lab, weights)
            )
        else:
            score = foreground.log_likelihood(roi_lab) - background.log_likelihood(roi_lab)
        score = np.clip(score, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)

    def _init_growth_radius_px(self) -> float:
        """Spatial radius used to keep normal-marker initialization compact.

        A black marker on a thin wooden rod can be colour-connected to rod
        shadows/edges.  During initialization, we therefore do seeded growth
        only inside a circle around the click.  The user may override this;
        otherwise it is derived from expected diameter and sample radius.
        """
        explicit = float(self._settings.init_growth_radius_px)
        if explicit > 0.0:
            return explicit
        diameter = max(1.0, float(self._settings.expected_diameter_px))
        return max(diameter * 1.35, float(self._settings.seed_radius + 4))

    @staticmethod
    def _spatial_radius_mask(
        shape_hw: tuple[int, int],
        click_local_rc: tuple[int, int],
        radius_px: float,
    ) -> np.ndarray:
        height, width = shape_hw
        row, col = click_local_rc
        yy, xx = np.ogrid[:height, :width]
        radius = max(1.0, float(radius_px))
        return ((yy - float(row)) ** 2 + (xx - float(col)) ** 2 <= radius * radius)

    def _clicked_component(
        self,
        probability: np.ndarray,
        click_local_rc: tuple[int, int],
        *,
        initialization: bool,
    ) -> np.ndarray:
        row, col = click_local_rc
        if not (0 <= row < probability.shape[0] and 0 <= col < probability.shape[1]):
            raise InitializationError("Clicked point lies outside the initialization probability map.")

        kernel = np.ones((3, 3), dtype=np.uint8)
        thresholds = (
            self._settings.init_probability_thresholds
            if initialization else self._settings.probability_thresholds
        )
        click_probability = float(probability[row, col])
        local_r = max(1, int(round(max(self._settings.seed_radius + 1, self._settings.expected_diameter_px / 4.0))))
        y0, y1 = max(0, row - local_r), min(probability.shape[0], row + local_r + 1)
        x0, x1 = max(0, col - local_r), min(probability.shape[1], col + local_r + 1)
        local_max = float(np.max(probability[y0:y1, x0:x1])) if y1 > y0 and x1 > x0 else click_probability
        original_init_threshold = float(thresholds[0]) if thresholds else 0.0
        # A threshold above the clicked likelihood previously prevented the
        # auto-adjustment path from ever reaching a candidate.  Try one
        # bounded seed-compatible threshold, then persist it only on success.
        relaxed_seed_threshold: Optional[float] = None
        if initialization and self._settings.point_mode != "tiny" and click_probability > 0.05:
            candidate = float(np.clip(click_probability * 0.98, 0.05, 0.99))
            if candidate < original_init_threshold:
                relaxed_seed_threshold = candidate
                thresholds = tuple(thresholds) + (candidate,)

        expected_init_area = float(
            np.pi * (max(1.0, self._settings.expected_diameter_px) / 2.0) ** 2
        )
        area_limit = expected_init_area * self._settings.init_max_area_ratio
        max_frame_fraction_area = probability.size * self._settings.init_max_component_fraction
        spatial_mask: Optional[np.ndarray] = None
        growth_radius = 0.0
        if initialization and self._settings.point_mode != "tiny":
            growth_radius = self._init_growth_radius_px()
            spatial_mask = self._spatial_radius_mask(probability.shape, click_local_rc, growth_radius).astype(np.uint8)

        diagnostics: list[str] = []
        for threshold in thresholds:
            binary = (probability >= threshold).astype(np.uint8)
            if spatial_mask is not None:
                binary = binary * spatial_mask
            iterations = (
                self._settings.init_morphology_close_iterations
                if initialization else self._settings.morphology_close_iterations
            )
            if iterations > 0:
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                if spatial_mask is not None:
                    binary = binary * spatial_mask
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            components = max(0, int(count) - 1)
            label = int(labels[row, col])
            if label == 0:
                diagnostics.append(
                    f"thr {threshold:.3f}: click probability {click_probability:.3f} is below threshold "
                    f"or disconnected after compact growth; components={components}, local max near click={local_max:.3f}"
                )
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self._settings.init_min_component_area:
                diagnostics.append(
                    f"thr {threshold:.3f}: clicked component area {area} px is below minimum "
                    f"{self._settings.init_min_component_area} px"
                )
                continue
            if area > max_frame_fraction_area:
                diagnostics.append(
                    f"thr {threshold:.3f}: clicked component area {area} px is too large for init ROI "
                    f"fraction limit {max_frame_fraction_area:.0f} px"
                )
                continue
            if initialization and self._settings.point_mode != "tiny":
                if area > area_limit:
                    diagnostics.append(
                        f"thr {threshold:.3f}: clicked component area {area} px is too large; "
                        f"expected marker area {expected_init_area:.0f} px, max allowed {area_limit:.0f} px "
                        f"(ratio limit {self._settings.init_max_area_ratio:.2f})"
                    )
                    continue
            if relaxed_seed_threshold is not None and abs(float(threshold) - relaxed_seed_threshold) < 1e-6:
                self._commit_seed_threshold_relaxation(
                    old=original_init_threshold,
                    new=relaxed_seed_threshold,
                    metric=click_probability,
                )
            return (labels == label).astype(np.uint8)

        if initialization and self._settings.point_mode != "tiny":
            radius_msg = (
                f"compact init growth radius {growth_radius:.1f} px; "
                "increase Normal init grow if your click is far from marker centre"
                if growth_radius > 0.0 else "compact init growth disabled"
            )
            detail = "\n".join(diagnostics[-5:]) if diagnostics else "No accepted component candidates were produced."
            raise InitializationError(
                "No accepted colour-connected marker region containing the clicked point was found.\n\n"
                f"Click probability: {click_probability:.3f}; local max near click: {local_max:.3f}.\n"
                f"Thresholds tested: {', '.join(f'{t:.3f}' for t in thresholds)}.\n"
                f"Expected marker diameter: {self._settings.expected_diameter_px:.1f} px; "
                f"expected area: {expected_init_area:.0f} px; area ratio limit: {self._settings.init_max_area_ratio:.2f}.\n"
                f"{radius_msg}.\n\n"
                f"Details:\n{detail}\n\n"
                "Try clicking nearer the marker centre, lowering Normal init thr. if the click is below threshold, "
                "raising Normal init thr. if the marker merges into the rod, lowering Normal init max area, "
                "or adjusting Normal init grow."
            )
        raise InitializationError(
            "No colour-connected marker region containing the clicked point was found."
        )

    def _commit_seed_threshold_relaxation(self, *, old: float, new: float, metric: float) -> None:
        if new >= old:
            return
        object.__setattr__(self._settings, "init_probability_thresholds", (float(new),))
        self.config.normal_init_threshold = float(new)
        self._last_threshold_adjustments.append({
            "field": "normal_init_threshold",
            "label": "Normal init threshold",
            "old": float(old),
            "new": float(new),
            "metric": float(metric),
        })

    def _seed_pixels(
        self,
        roi_lab: np.ndarray,
        click_local_rc: tuple[int, int],
    ) -> np.ndarray:
        row, col = click_local_rc
        r = self._settings.seed_radius
        y0, y1 = max(0, row - r), min(roi_lab.shape[0], row + r + 1)
        x0, x1 = max(0, col - r), min(roi_lab.shape[1], col + r + 1)
        return roi_lab[y0:y1, x0:x1].reshape(-1, 3)

    def _border_pixels(self, roi_lab: np.ndarray) -> np.ndarray:
        b = min(self._settings.border_width, roi_lab.shape[0] // 3, roi_lab.shape[1] // 3)
        if b < 1:
            raise InitializationError("Initialization window is too small for background sampling.")
        mask = np.zeros(roi_lab.shape[:2], dtype=bool)
        mask[:b, :] = True
        mask[-b:, :] = True
        mask[:, :b] = True
        mask[:, -b:] = True
        return roi_lab[mask].reshape(-1, 3)

    def _interior_mask(self, marker_mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=np.uint8)
        interior = cv2.erode(
            marker_mask.astype(np.uint8),
            kernel,
            iterations=self._settings.erode_iterations,
        )
        return interior if int(interior.sum()) >= 8 else marker_mask.astype(np.uint8)

    def _background_ring(self, marker_mask: np.ndarray) -> np.ndarray:
        inner = cv2.dilate(
            marker_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=self._settings.background_ring_inner_dilate,
        )
        outer = cv2.dilate(
            marker_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=self._settings.background_ring_outer_dilate,
        )
        return ((outer > 0) & (inner == 0)).astype(np.uint8)

    @staticmethod
    def _outer_contour(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            raise InitializationError("No contour found for marker candidate.")
        return max(contours, key=cv2.contourArea)

    def _refine_contour_by_probability_gradient(
        self,
        contour: np.ndarray,
        probability: np.ndarray,
    ) -> np.ndarray:
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] < 5:
            return points

        smooth = cv2.GaussianBlur(
            probability.astype(np.float32),
            (0, 0),
            sigmaX=self._settings.probability_blur_sigma,
            sigmaY=self._settings.probability_blur_sigma,
        )
        offsets = np.linspace(
            -self._settings.gradient_normal_radius,
            self._settings.gradient_normal_radius,
            self._settings.gradient_profile_samples,
            dtype=np.float32,
        )
        refined = points.copy()
        for index, point in enumerate(points):
            tangent = points[(index + 2) % len(points)] - points[(index - 2) % len(points)]
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                continue
            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            sample_xy = point[None, :] + offsets[:, None] * normal[None, :]
            profile = cv2.remap(
                smooth,
                sample_xy[:, 0].reshape(-1, 1),
                sample_xy[:, 1].reshape(-1, 1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).ravel()
            gradient = np.abs(np.gradient(profile))
            peak = int(np.argmax(gradient))
            refined[index] = sample_xy[peak]
        return refined

    @staticmethod
    def _weighted_centroid(probability: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = probability * mask.astype(np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            raise InitializationError("Marker component has zero probability mass.")
        rows, cols = np.indices(weights.shape, dtype=np.float32)
        return np.array(
            [(weights * rows).sum() / total, (weights * cols).sum() / total],
            dtype=np.float32,
        )

    def _normal_peak_window_center(
        self,
        probability: np.ndarray,
        component_mask: np.ndarray,
        roi_origin_xy: tuple[int, int],
        fallback_center_local_rc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int, int, int]]:
        x0, y0 = roi_origin_xy
        masked = probability.astype(np.float32) * component_mask.astype(np.float32)
        if masked.size == 0 or float(np.max(masked)) <= 1e-8:
            center_global = np.array([y0 + fallback_center_local_rc[0], x0 + fallback_center_local_rc[1]], dtype=np.float32)
            bbox = (int(round(center_global[1])), int(round(center_global[0])), 1, 1)
            return center_global, center_global.copy(), 0.0, bbox
        peak_index = int(np.argmax(masked))
        peak_r, peak_c = np.unravel_index(peak_index, masked.shape)
        peak_value = float(probability[peak_r, peak_c])
        plateau_threshold = max(0.0, min(peak_value * 0.98, peak_value - 0.02))
        plateau_mask = ((component_mask > 0) & (probability >= plateau_threshold)).astype(np.uint8)
        if int(plateau_mask.sum()) > 1:
            plateau_center = self._weighted_centroid(probability, plateau_mask)
            peak_r = int(np.clip(round(float(plateau_center[0])), 0, probability.shape[0] - 1))
            peak_c = int(np.clip(round(float(plateau_center[1])), 0, probability.shape[1] - 1))
        expected_core = min(
            max(1.0, float(self._state.expected_width if self._state is not None else self._settings.expected_diameter_px)),
            max(1.0, float(self._state.expected_height if self._state is not None else self._settings.expected_diameter_px)),
        )
        half = max(2, int(round(expected_core * 0.35)))
        r0 = max(0, peak_r - half)
        r1 = min(probability.shape[0], peak_r + half + 1)
        c0 = max(0, peak_c - half)
        c1 = min(probability.shape[1], peak_c + half + 1)
        weights = probability[r0:r1, c0:c1].astype(np.float32) * component_mask[r0:r1, c0:c1].astype(np.float32)
        weights = np.clip(weights - max(0.0, peak_value * 0.25), 0.0, 1.0)
        total = float(weights.sum())
        if total <= 1e-8:
            center_local = np.asarray(fallback_center_local_rc, dtype=np.float32)
        else:
            rows, cols = np.indices(weights.shape, dtype=np.float32)
            center_local = np.array(
                [r0 + float((weights * rows).sum() / total), c0 + float((weights * cols).sum() / total)],
                dtype=np.float32,
            )
        center_global = np.array([y0 + center_local[0], x0 + center_local[1]], dtype=np.float32)
        peak_global = np.array([float(y0 + peak_r), float(x0 + peak_c)], dtype=np.float32)
        bbox_global = (int(x0 + c0), int(y0 + r0), int(c1 - c0), int(r1 - r0))
        return center_global, peak_global, peak_value, bbox_global

    @staticmethod
    def _geometry(contour_xy: np.ndarray, weighted_center_local_rc: np.ndarray) -> dict:
        contour = contour_xy.reshape(-1, 1, 2).astype(np.float32)
        area = float(abs(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            return {
                "area": 0.0, "width": 0.0, "height": 0.0,
                "circularity": 0.0, "rectangularity": 0.0,
                "center_difference": float("inf"), "geometric_center_xy": None,
                "is_rectangle": False,
            }

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-9))
        rect = cv2.minAreaRect(contour)
        (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect
        rect_area = max(float(rect_w * rect_h), 1e-9)
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        is_rectangle = bool(len(approx) == 4 and cv2.isContourConvex(approx))

        geometric_center_xy: Optional[np.ndarray] = None
        if is_rectangle and rectangularity >= 0.60:
            geometric_center_xy = np.array([rect_cx, rect_cy], dtype=np.float32)
        elif contour.shape[0] >= 5 and circularity >= 0.48:
            geometric_center_xy = np.array(cv2.fitEllipse(contour)[0], dtype=np.float32)

        weighted_xy = np.array(
            [weighted_center_local_rc[1], weighted_center_local_rc[0]], dtype=np.float32
        )
        center_difference = (
            float(np.linalg.norm(weighted_xy - geometric_center_xy))
            if geometric_center_xy is not None else float("nan")
        )
        return {
            "area": area,
            "width": float(rect_w),
            "height": float(rect_h),
            "circularity": circularity,
            "rectangularity": rectangularity,
            "center_difference": center_difference,
            "geometric_center_xy": geometric_center_xy,
            "is_rectangle": is_rectangle,
            "angle_deg": float(rect_angle),
        }

    def _colour_quality(
        self,
        probability: np.ndarray,
        marker_mask: np.ndarray,
        ring_mask: np.ndarray,
    ) -> dict[str, float]:
        inside = probability[marker_mask > 0]
        ring = probability[ring_mask > 0]
        if inside.size == 0 or ring.size == 0:
            return {
                "median_inside_probability": 0.0,
                "strong_inside_fraction": 0.0,
                "probability_margin": -1.0,
            }
        return {
            "median_inside_probability": float(np.median(inside)),
            "strong_inside_fraction": float(
                np.mean(inside > self._settings.strong_probability_threshold)
            ),
            "probability_margin": float(np.mean(inside) - np.mean(ring)),
        }

    def _luminance_polarity_from_models(
        self,
        foreground: _GaussianLabModel | _TinyLabDistanceModel,
        background: _GaussianLabModel | _TinyLabDistanceModel,
    ) -> tuple[str, float, float]:
        fg_l = float(np.asarray(foreground.mean, dtype=np.float32)[0])
        bg_l = float(np.asarray(background.mean, dtype=np.float32)[0])
        delta = fg_l - bg_l
        threshold = float(self._settings.luminance_polarity_min_delta)
        if delta <= -threshold:
            return "dark", fg_l, bg_l
        if delta >= threshold:
            return "light", fg_l, bg_l
        return "neutral", fg_l, bg_l

    def _luminance_polarity_quality(
        self,
        roi_lab: np.ndarray,
        component_mask: np.ndarray,
        ring_mask: np.ndarray,
    ) -> dict[str, float | str | bool]:
        assert self._state is not None
        init = self._state.initialization
        inside_l = roi_lab[..., 0][component_mask > 0].astype(np.float32)
        ring_l = roi_lab[..., 0][ring_mask > 0].astype(np.float32)
        if inside_l.size == 0:
            return {
                "candidate_l_mean": 0.0,
                "candidate_l_median": 0.0,
                "local_background_l_mean": 0.0,
                "luminance_polarity": init.luminance_polarity,
                "luminance_polarity_pass": True,
                "luminance_error": 0.0,
            }
        cand_mean = float(np.mean(inside_l))
        cand_median = float(np.median(inside_l))
        bg_mean = float(np.mean(ring_l)) if ring_l.size else float(init.background_l_mean)
        polarity = str(init.luminance_polarity)
        passed = True
        error = 0.0
        if self._settings.luminance_polarity_enabled and polarity == "dark":
            too_bright_vs_init = cand_mean - (float(init.foreground_l_mean) + self._settings.luminance_polarity_tolerance)
            too_bright_vs_local_bg = cand_mean - (bg_mean - self._settings.luminance_polarity_min_delta)
            error = float(max(too_bright_vs_init, too_bright_vs_local_bg, 0.0))
            passed = error <= 0.0
        elif self._settings.luminance_polarity_enabled and polarity == "light":
            too_dark_vs_init = (float(init.foreground_l_mean) - self._settings.luminance_polarity_tolerance) - cand_mean
            too_dark_vs_local_bg = (bg_mean + self._settings.luminance_polarity_min_delta) - cand_mean
            error = float(max(too_dark_vs_init, too_dark_vs_local_bg, 0.0))
            passed = error <= 0.0
        return {
            "candidate_l_mean": cand_mean,
            "candidate_l_median": cand_median,
            "local_background_l_mean": bg_mean,
            "luminance_polarity": polarity,
            "luminance_polarity_pass": bool(passed),
            "luminance_error": float(error),
        }

    @staticmethod
    def _colour_score(quality: dict[str, float]) -> float:
        margin = float(np.clip(quality["probability_margin"] / 0.60, 0.0, 1.0))
        return float(np.clip(
            0.45 * quality["median_inside_probability"]
            + 0.30 * quality["strong_inside_fraction"]
            + 0.25 * margin,
            0.0, 1.0,
        ))

    def _shape_score(
        self,
        geometry: dict,
        area_ratio: float,
        axis_ratios: list[float],
    ) -> float:
        assert self._state is not None
        area_score = float(np.exp(-abs(np.log(max(area_ratio, 1e-9))) / 0.40))
        axes_score = float(np.mean([
            np.exp(-abs(np.log(max(ratio, 1e-9))) / 0.40) for ratio in axis_ratios
        ]))
        if self._state.initialization.shape_model == "circle_like":
            model_score = float(np.clip(
                geometry["circularity"] /
                max(self._state.initialization.circularity, 0.5),
                0.0, 1.0,
            ))
        elif self._state.initialization.shape_model == "rectangle":
            model_score = float(np.clip(
                geometry["rectangularity"] /
                max(self._state.initialization.rectangularity, 0.5),
                0.0, 1.0,
            ))
        else:
            model_score = 1.0

        difference = geometry["center_difference"]
        centre_score = (
            float(np.exp(-0.5 * (difference / 1.5) ** 2))
            if np.isfinite(difference) else 1.0
        )
        return float(np.clip(
            0.38 * area_score + 0.27 * axes_score
            + 0.22 * model_score + 0.13 * centre_score,
            0.0, 1.0,
        ))

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        rows, cols = np.nonzero(mask)
        if cols.size == 0:
            raise InitializationError("Cannot compute bbox for an empty marker component.")
        return (
            int(cols.min()), int(rows.min()),
            int(cols.max() - cols.min() + 1),
            int(rows.max() - rows.min() + 1),
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
    def _bbox_from_feature_and_margin(
        bbox: tuple[int, int, int, int],
        margin: int,
        frame_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = bbox
        frame_h, frame_w = frame_hw
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(frame_w, x + width + margin)
        y1 = min(frame_h, y + height + margin)
        return (x0, y0, x1 - x0, y1 - y0)
