"""
tracking/base_tracker.py
------------------------
Abstract base class for all tracker types, plus shared data structures.

Every tracker follows the same lifecycle:
  1. __init__(config)           — constructed with a TrackerConfig
  2. initialize(frame_gpu, ...)  — called once on the start frame; returns
                                   InitPreview for the canvas overlay
  3. process_frame(frame_gpu, frame_idx) — called per-frame during batch;
                                           returns FrameResult
  4. reinitialize(frame_gpu, seed, frame_idx) — called after lock loss when
                                                  user provides a new seed

Status transitions:
  PENDING → LOCKED  (after successful initialize)
  LOCKED  → UNCERTAIN  (Hessian score drops below threshold)
  UNCERTAIN → LOCKED   (auto re-lock succeeds)
  UNCERTAIN → LOST     (N consecutive uncertain frames)
  LOST / UNCERTAIN → LOCKED  (user manually re-seeds)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gpu.channel import ChannelConfig


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackerType(Enum):
    POINT_FAST     = 'point_fast'
    POINT_ACCURATE = 'point_accurate'
    BLOB_SIMPLE    = 'blob_simple'
    BLOB_COMPLEX   = 'blob_complex'
    CURVE          = 'curve'
    COLOR_AREA     = 'color_area'


class TrackerStatus(Enum):
    PENDING   = 'pending'    # Not yet initialised
    LOCKED    = 'locked'     # Tracking confidently
    UNCERTAIN = 'uncertain'  # Hessian score low, last known pos stored
    LOST      = 'lost'       # Too many consecutive uncertain frames


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    """
    All user-editable parameters for one tracker instance.
    Serialised to/from the .npz meta JSON.
    """
    tracker_type:   TrackerType
    name:           str                      = ''
    sigma:          float                    = 5.0   # Feature size in pixels
    channel:        Optional[ChannelConfig]  = None  # None = use global default
    # Hessian re-lock threshold (fraction of initialisation score)
    relock_threshold: float = 0.20
    # Frames of UNCERTAIN before declaring LOST
    lost_after_frames: int = 10
    # ROI search radius for blobs / curves (pixels)
    search_radius:  int   = 60
    # Optional inclusive frame at which this tracker stops. ``None`` follows
    # the batch range through its final frame.
    end_frame: Optional[int] = None
    pause_on_loss: bool = True

    # Point-fast colour tracker settings
    point_mode: str = 'normal'              # 'normal' or 'tiny'
    sample_size: int = 5                    # odd pixel width used for colour sampling
    expected_diameter: float = 20.0         # point/feature diameter in pixels

    # Point-fast initialization gates.  These affect the one-click segmentation
    # step before normal tracking begins.  They are intentionally separate from
    # per-frame tracking thresholds because initialization errors poison the
    # learned model.
    normal_init_threshold: float = 0.72     # probability threshold for clicked-component init
    normal_init_min_probability_margin: float = 0.08
    normal_init_max_area_ratio: float = 6.0 # max init component area / expected marker area
    normal_init_growth_radius: float = 0.0  # 0 = auto; limits seeded init growth around click
    normal_init_close_iterations: int = 1   # set 0 to avoid merging narrow bridges
    tiny_init_peak_threshold: float = 0.40  # click likelihood must be at least this high
    kalman_enabled: bool = False
    kalman_model: str = 'constant_velocity' # 'constant_velocity' or 'constant_acceleration'
    measurement_noise: float = 0.7          # measurement sigma in pixels
    process_noise: float = 0.5              # acceleration/jerk sigma in px/frame^2-ish
    kalman_gate_sigma: float = 4.0
    min_search_radius: int = 12
    max_search_radius: int = 80
    # Old-style normal-marker search controls used when Kalman is disabled.
    # These reproduce the reliable pre-Kalman ROI rule:
    #   margin = min(max_search_margin, normal_search_margin + misses * miss_expansion)
    normal_search_margin: int = 15
    miss_expansion: int = 16
    max_search_margin: int = 150
    max_prediction_error: float = 22.0
    miss_motion_allowance: float = 14.0
    accept_within_search_region: bool = False
    # Optional normal-marker second pass. If a valid blob looks clipped by the
    # ROI edge, recenter/expand once around that blob and recompute the full
    # segmentation so the centroid is taken from the whole marker, not a partial edge.
    normal_recenter_on_edge: bool = True
    normal_recenter_edge_px: int = 3
    normal_recenter_extra_margin: int = 20
    normal_recenter_min_area_ratio: float = 0.25
    measurement_center_mode: str = 'full_blob'  # 'full_blob', 'peak_window', or 'auto'
    adaptive_prefilter_enabled: bool = False
    # Normal marker colour reliability controls.  LAB weights multiply the
    # normal-marker LAB distance before probability conversion; L > 1 makes
    # dark-vs-light differences stricter.  Polarity gate uses the initialization
    # foreground/background luminance relation to reject bright candidates for a
    # dark marker and dark candidates for a light marker.
    lab_l_weight: float = 1.0
    lab_a_weight: float = 1.0
    lab_b_weight: float = 1.0
    luminance_polarity_enabled: bool = True
    luminance_polarity_min_delta: float = 8.0
    luminance_polarity_tolerance: float = 35.0
    colour_diagnostics_enabled: bool = False
    shape_validation: str = 'soft'          # 'off', 'soft', or 'strict'
    tiny_peak_threshold: float = 0.55       # minimum tiny-feature likelihood peak
    tiny_contrast_threshold: float = 0.04   # minimum tiny-feature probability contrast
    tiny_init_contrast_threshold: float = 8.0  # minimum Lab centre-vs-ring contrast at init
    tiny_direction_min_cosine: float = 0.20    # required match to init contrast direction
    tiny_update_confidence_threshold: float = 0.65  # do not update Kalman below this
    tiny_mosse_mode: str = 'off'               # 'off', 'assisted', or 'primary'
    tiny_mosse_window_scale: float = 4.0        # MOSSE window size = expected diameter * scale
    tiny_mosse_learning_rate: float = 0.05
    tiny_mosse_psr_threshold: float = 6.0
    tiny_mosse_peak_margin_threshold: float = 0.15
    normal_mosse_mode: str = 'off'
    mosse_scale_adaptation: bool = False

    # Point tracker debug recorder; off by default to avoid memory/I/O overhead.
    debug_enabled: bool = False
    debug_history_frames: int = 120
    # In debug reports, when a frame is UNCERTAIN/LOST, optionally record
    # a diagnostic fallback coordinate for tuning only. This never changes
    # FrameResult.center, which remains None for missing measurements.
    debug_uncertain_center_policy: str = 'none'  # 'none', 'peak', or 'kalman'
    learned_validation_samples: Any = None

    # Color area specific
    color_tolerance: Any  = None   # gpu.color_mask.ColorTolerance | None

    # Auto-assigned unique ID
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        d = {
            'uid':              self.uid,
            'name':             self.name,
            'tracker_type':     self.tracker_type.value,
            'sigma':            self.sigma,
            'relock_threshold': self.relock_threshold,
            'lost_after_frames':self.lost_after_frames,
            'search_radius':    self.search_radius,
            'end_frame':        self.end_frame,
            'pause_on_loss':    self.pause_on_loss,
            'point_mode':       self.point_mode,
            'sample_size':      self.sample_size,
            'expected_diameter':self.expected_diameter,
            'normal_init_threshold': self.normal_init_threshold,
            'normal_init_min_probability_margin': self.normal_init_min_probability_margin,
            'normal_init_max_area_ratio': self.normal_init_max_area_ratio,
            'normal_init_growth_radius': self.normal_init_growth_radius,
            'normal_init_close_iterations': self.normal_init_close_iterations,
            'tiny_init_peak_threshold': self.tiny_init_peak_threshold,
            'kalman_enabled':   self.kalman_enabled,
            'kalman_model':     self.kalman_model,
            'measurement_noise':self.measurement_noise,
            'process_noise':    self.process_noise,
            'kalman_gate_sigma':self.kalman_gate_sigma,
            'min_search_radius':self.min_search_radius,
            'max_search_radius':self.max_search_radius,
            'normal_search_margin': self.normal_search_margin,
            'miss_expansion': self.miss_expansion,
            'max_search_margin': self.max_search_margin,
            'max_prediction_error': self.max_prediction_error,
            'miss_motion_allowance': self.miss_motion_allowance,
            'accept_within_search_region': self.accept_within_search_region,
            'normal_recenter_on_edge': self.normal_recenter_on_edge,
            'normal_recenter_edge_px': self.normal_recenter_edge_px,
            'normal_recenter_extra_margin': self.normal_recenter_extra_margin,
            'normal_recenter_min_area_ratio': self.normal_recenter_min_area_ratio,
            'measurement_center_mode': self.measurement_center_mode,
            'adaptive_prefilter_enabled': self.adaptive_prefilter_enabled,
            'lab_l_weight': self.lab_l_weight,
            'lab_a_weight': self.lab_a_weight,
            'lab_b_weight': self.lab_b_weight,
            'luminance_polarity_enabled': self.luminance_polarity_enabled,
            'luminance_polarity_min_delta': self.luminance_polarity_min_delta,
            'luminance_polarity_tolerance': self.luminance_polarity_tolerance,
            'colour_diagnostics_enabled': self.colour_diagnostics_enabled,
            'shape_validation': self.shape_validation,
            'tiny_peak_threshold': self.tiny_peak_threshold,
            'tiny_contrast_threshold': self.tiny_contrast_threshold,
            'tiny_init_contrast_threshold': self.tiny_init_contrast_threshold,
            'tiny_direction_min_cosine': self.tiny_direction_min_cosine,
            'tiny_update_confidence_threshold': self.tiny_update_confidence_threshold,
            'tiny_mosse_mode': self.tiny_mosse_mode,
            'normal_mosse_mode': self.normal_mosse_mode,
            'mosse_scale_adaptation': self.mosse_scale_adaptation,
            'tiny_mosse_window_scale': self.tiny_mosse_window_scale,
            'tiny_mosse_learning_rate': self.tiny_mosse_learning_rate,
            'tiny_mosse_psr_threshold': self.tiny_mosse_psr_threshold,
            'tiny_mosse_peak_margin_threshold': self.tiny_mosse_peak_margin_threshold,
            'debug_enabled':   self.debug_enabled,
            'debug_history_frames': self.debug_history_frames,
            'debug_uncertain_center_policy': self.debug_uncertain_center_policy,
        }
        if self.channel:
            d['channel'] = self.channel.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> 'TrackerConfig':
        from gpu.channel import ChannelConfig
        cfg = TrackerConfig(
            tracker_type=TrackerType(d['tracker_type']),
            name=d.get('name', ''),
            sigma=d.get('sigma', 5.0),
            relock_threshold=d.get('relock_threshold', 0.20),
            lost_after_frames=d.get('lost_after_frames', 10),
            search_radius=d.get('search_radius', 60),
            end_frame=d.get('end_frame'),
            pause_on_loss=d.get('pause_on_loss', True),
            point_mode=d.get('point_mode', 'normal'),
            sample_size=d.get('sample_size', 5),
            expected_diameter=d.get('expected_diameter', 20.0),
            normal_init_threshold=d.get('normal_init_threshold', 0.72),
            normal_init_min_probability_margin=d.get('normal_init_min_probability_margin', 0.08),
            normal_init_max_area_ratio=d.get('normal_init_max_area_ratio', 6.0),
            normal_init_growth_radius=d.get('normal_init_growth_radius', 0.0),
            normal_init_close_iterations=d.get('normal_init_close_iterations', 1),
            tiny_init_peak_threshold=d.get('tiny_init_peak_threshold', 0.40),
            kalman_enabled=d.get('kalman_enabled', False),
            kalman_model=d.get('kalman_model', 'constant_velocity'),
            measurement_noise=d.get('measurement_noise', 0.7),
            process_noise=d.get('process_noise', 0.5),
            kalman_gate_sigma=d.get('kalman_gate_sigma', 4.0),
            min_search_radius=d.get('min_search_radius', 12),
            max_search_radius=d.get('max_search_radius', 80),
            normal_search_margin=d.get('normal_search_margin', 15),
            miss_expansion=d.get('miss_expansion', 16),
            max_search_margin=d.get('max_search_margin', 150),
            max_prediction_error=d.get('max_prediction_error', 22.0),
            miss_motion_allowance=d.get('miss_motion_allowance', 14.0),
            accept_within_search_region=d.get('accept_within_search_region', False),
            normal_recenter_on_edge=d.get('normal_recenter_on_edge', True),
            normal_recenter_edge_px=d.get('normal_recenter_edge_px', 3),
            normal_recenter_extra_margin=d.get('normal_recenter_extra_margin', 20),
            normal_recenter_min_area_ratio=d.get('normal_recenter_min_area_ratio', 0.25),
            measurement_center_mode=d.get('measurement_center_mode', 'full_blob'),
            adaptive_prefilter_enabled=d.get('adaptive_prefilter_enabled', False),
            lab_l_weight=d.get('lab_l_weight', 1.0),
            lab_a_weight=d.get('lab_a_weight', 1.0),
            lab_b_weight=d.get('lab_b_weight', 1.0),
            luminance_polarity_enabled=d.get('luminance_polarity_enabled', True),
            luminance_polarity_min_delta=d.get('luminance_polarity_min_delta', 8.0),
            luminance_polarity_tolerance=d.get('luminance_polarity_tolerance', 35.0),
            colour_diagnostics_enabled=d.get('colour_diagnostics_enabled', False),
            shape_validation=d.get('shape_validation', 'soft'),
            tiny_peak_threshold=d.get('tiny_peak_threshold', 0.55),
            tiny_contrast_threshold=d.get('tiny_contrast_threshold', 0.04),
            tiny_init_contrast_threshold=d.get('tiny_init_contrast_threshold', 8.0),
            tiny_direction_min_cosine=d.get('tiny_direction_min_cosine', 0.20),
            tiny_update_confidence_threshold=d.get('tiny_update_confidence_threshold', 0.65),
            tiny_mosse_mode=d.get('tiny_mosse_mode', 'off'),
            normal_mosse_mode=d.get('normal_mosse_mode', 'off'),
            mosse_scale_adaptation=d.get('mosse_scale_adaptation', False),
            tiny_mosse_window_scale=d.get('tiny_mosse_window_scale', 4.0),
            tiny_mosse_learning_rate=d.get('tiny_mosse_learning_rate', 0.05),
            tiny_mosse_psr_threshold=d.get('tiny_mosse_psr_threshold', 6.0),
            tiny_mosse_peak_margin_threshold=d.get('tiny_mosse_peak_margin_threshold', 0.15),
            debug_enabled=d.get('debug_enabled', False),
            debug_history_frames=d.get('debug_history_frames', 120),
            debug_uncertain_center_policy=d.get('debug_uncertain_center_policy', 'none'),
            uid=d.get('uid', str(uuid.uuid4())[:8]),
        )
        if 'channel' in d:
            cfg.channel = ChannelConfig.from_dict(d['channel'])
        return cfg


# ---------------------------------------------------------------------------
# Per-frame result
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """
    Result produced by a tracker for one video frame.

    Exactly one of the payload fields is populated depending on tracker type:
      point_fast / point_accurate → center
      blob_simple                 → polygon
      blob_complex / color_area   → mask
      curve                       → spline_points
    """
    frame_index: int
    status:      TrackerStatus

    # Payloads (only one populated per tracker type)
    center:        Optional[np.ndarray] = None   # (2,) float32 [row, col]
    polygon:       Optional[np.ndarray] = None   # (N,2) float32 [row, col]
    mask:          Optional[np.ndarray] = None   # (H,W) uint8 CPU
    spline_points: Optional[np.ndarray] = None   # (K,2) float32 [row, col]

    # Diagnostic
    hessian_score: float = 0.0

    # Optional alternative positions produced by the tracker for manual review
    # of UNCERTAIN/LOST frames. Coordinates are full-frame [row, col].
    # They are not exported as measurements unless the user explicitly applies
    # one as a correction in the uncertain-frame review tool.
    alt_centers: Dict[str, np.ndarray] = field(default_factory=dict)
    correction_source: str = ''

    # Human-readable rejection/failure reasons and compact numeric diagnostics
    # for UNCERTAIN/LOST review. These are intentionally lightweight and are
    # available even when the heavier debug recorder is disabled.
    failure_reasons: list[str] = field(default_factory=list)
    diagnostic_values: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Init-frame preview (for canvas overlay on first frame)
# ---------------------------------------------------------------------------

@dataclass
class InitPreview:
    """
    Data returned by initialize() for rendering the semi-transparent overlay
    on the start frame so the user can confirm the tracker found the right object.
    """
    # Hessian response map or ridgeness map — normalised to [0,1], float32 CPU
    heatmap:       Optional[np.ndarray] = None   # (H,W) float32

    # Overlay shape (same semantics as FrameResult)
    center:        Optional[np.ndarray] = None
    polygon:       Optional[np.ndarray] = None
    mask:          Optional[np.ndarray] = None   # (H,W) uint8 CPU
    spline_points: Optional[np.ndarray] = None
    color_preview: Optional[np.ndarray] = None   # (H,W,4) RGBA uint8 CPU

    # HOG patch bbox for Point Fast: (x, y, w, h)
    hog_bbox:      Optional[Tuple[int,int,int,int]] = None
    threshold_adjustments: list[dict[str, Any]] = field(default_factory=list)
    init_diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseTracker(ABC):
    """
    Abstract base for all tracker types.

    Subclasses implement initialize(), process_frame(), and reinitialize().
    The base class handles status bookkeeping and uncertain-frame counting.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config  = config
        self.status  = TrackerStatus.PENDING
        self._uncertain_count = 0
        self._init_score:  float = 0.0   # Hessian score at init frame

    # ------------------------------------------------------------------
    # Frame-source preference
    # ------------------------------------------------------------------

    @property
    def requires_cpu_frame(self) -> bool:
        """Return True when this tracker should receive CPU BGR frames.

        Most existing Hessian trackers operate on GPU frames and keep the
        default False value.  Direct colour-marker localization works on small
        CPU ROIs and overrides this property to avoid GPU-to-CPU downloads.
        """
        return False

    @property
    def accepts_workspace_frame(self) -> bool:
        """Return True when this tracker can receive a cropped WorkspaceFrame.

        A WorkspaceFrame contains a BGR crop plus its full-frame origin. Most
        existing trackers expect full-frame images, so the default is False.
        PointFastTracker overrides this to keep initialization and tracking on
        the same prepared-cache/NVDEC decode path.
        """
        return False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        roi: Optional[Tuple[int,int,int,int]] = None,
    ) -> InitPreview:
        """
        Initialise the tracker on the start frame.

        Parameters
        ----------
        frame_gpu : cp.ndarray | np.ndarray
                    (H,W,3) uint8 BGR; GPU by default, or CPU when the
                    subclass overrides ``requires_cpu_frame``.
        seed_row  : float       Click row (full-image coordinates).
        seed_col  : float       Click column.
        roi       : (x,y,w,h)  Optional bounding-box drag from user.

        Returns
        -------
        InitPreview  — data for the canvas overlay.
        """
        ...

    @abstractmethod
    def process_frame(
        self,
        frame_gpu: cp.ndarray,
        frame_index: int,
    ) -> FrameResult:
        """
        Track the object in one frame during batch processing.

        Parameters
        ----------
        frame_gpu   : cp.ndarray | np.ndarray
                      (H,W,3) uint8 BGR; GPU by default, or CPU when the
                      subclass overrides ``requires_cpu_frame``.
        frame_index : int

        Returns
        -------
        FrameResult
        """
        ...

    @abstractmethod
    def reinitialize(
        self,
        frame_gpu: cp.ndarray,
        seed_row: float,
        seed_col: float,
        frame_index: int,
    ) -> InitPreview:
        """
        Re-seed the tracker after manual recovery.
        Resets status to LOCKED and clears uncertain count.
        """
        ...

    # ------------------------------------------------------------------
    # Shared status helpers (used by subclasses)
    # ------------------------------------------------------------------

    def _update_status(self, hessian_score: float) -> TrackerStatus:
        """
        Update status based on current Hessian score vs initialisation score.

        Returns the new TrackerStatus.
        """
        if self._init_score < 1e-10:
            # Not initialised properly — stay LOCKED without score gating
            self.status = TrackerStatus.LOCKED
            return self.status

        ratio = hessian_score / self._init_score
        if ratio >= self.config.relock_threshold:
            self.status = TrackerStatus.LOCKED
            self._uncertain_count = 0
        else:
            self._uncertain_count += 1
            if self._uncertain_count >= self.config.lost_after_frames:
                self.status = TrackerStatus.LOST
            else:
                self.status = TrackerStatus.UNCERTAIN

        return self.status

    def _reset_to_locked(self) -> None:
        """Call after successful re-initialization."""
        self.status = TrackerStatus.LOCKED
        self._uncertain_count = 0

    @property
    def tracker_type(self) -> TrackerType:
        return self.config.tracker_type

    @property
    def uid(self) -> str:
        return self.config.uid

    @property
    def name(self) -> str:
        return self.config.name or f"{self.config.tracker_type.value}_{self.config.uid}"
