"""
utils/colors.py
---------------
Centralised colour palette for tracker overlays.

Single source of truth — imported by both video_canvas.py and panels.py
so the swatch colours in the panel always match the canvas overlays.

All colours are defined as (R, G, B, A) tuples where A is the default
alpha for overlay rendering (0–255).  Qt QColor objects are built on demand
via `qcolor()` and cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from tracking.base_tracker import TrackerStatus, TrackerType

# ---------------------------------------------------------------------------
# Palette definition
# ---------------------------------------------------------------------------

# (R, G, B, A_overlay)
_TYPE_PALETTE: dict[TrackerType, Tuple[int, int, int, int]] = {
    TrackerType.POINT_FAST:     (100, 149, 237, 200),   # cornflower blue
    TrackerType.POINT_ACCURATE: ( 70, 130, 180, 200),   # steel blue
    TrackerType.BLOB_SIMPLE:    (255, 165,   0, 200),   # orange
    TrackerType.BLOB_COMPLEX:   (255, 128,   0, 200),   # dark orange
    TrackerType.CURVE:          ( 50, 205,  50, 200),   # lime green
    TrackerType.COLOR_AREA:     (218, 112, 214, 200),   # orchid / magenta
}

_STATUS_PALETTE: dict[TrackerStatus, Tuple[int, int, int, int]] = {
    TrackerStatus.LOCKED:    (  0, 230,   0, 200),   # green
    TrackerStatus.UNCERTAIN: (255, 210,   0, 200),   # amber
    TrackerStatus.LOST:      (230,  40,  40, 200),   # red
    TrackerStatus.PENDING:   (160, 160, 160, 160),   # grey
}

# Hex strings for Qt stylesheets (no alpha channel in hex for CSS)
_TYPE_HEX: dict[TrackerType, str] = {
    t: '#{:02X}{:02X}{:02X}'.format(*rgba[:3])
    for t, rgba in _TYPE_PALETTE.items()
}

_STATUS_HEX: dict[TrackerStatus, str] = {
    s: '#{:02X}{:02X}{:02X}'.format(*rgba[:3])
    for s, rgba in _STATUS_PALETTE.items()
}

# ---------------------------------------------------------------------------
# RGBA accessors
# ---------------------------------------------------------------------------

def type_rgba(ttype: TrackerType) -> Tuple[int, int, int, int]:
    """Return (R, G, B, A) for a tracker type's primary colour."""
    return _TYPE_PALETTE.get(ttype, (200, 200, 200, 200))


def status_rgba(status: TrackerStatus) -> Tuple[int, int, int, int]:
    """Return (R, G, B, A) for a tracker status indicator colour."""
    return _STATUS_PALETTE.get(status, (200, 200, 200, 200))


def type_hex(ttype: TrackerType) -> str:
    """Return CSS hex colour string for stylesheet use."""
    return _TYPE_HEX.get(ttype, '#C8C8C8')


def status_hex(status: TrackerStatus) -> str:
    return _STATUS_HEX.get(status, '#C8C8C8')


# ---------------------------------------------------------------------------
# Qt QColor builders (lazy import — avoids importing Qt at module load time
# in non-UI contexts such as the batch worker thread)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def type_qcolor(ttype: TrackerType):
    """Return a cached QColor for a tracker type."""
    from PyQt6.QtGui import QColor
    r, g, b, a = type_rgba(ttype)
    return QColor(r, g, b, a)


@lru_cache(maxsize=None)
def status_qcolor(status: TrackerStatus):
    """Return a cached QColor for a tracker status."""
    from PyQt6.QtGui import QColor
    r, g, b, a = status_rgba(status)
    return QColor(r, g, b, a)


# ---------------------------------------------------------------------------
# Heatmap colourisation helper (used by video_canvas for overlay images)
# ---------------------------------------------------------------------------

def colorise_heatmap(heatmap, ttype: TrackerType, alpha_scale: float = 0.7):
    """
    Convert a [0, 1] float32 numpy heatmap to an RGBA uint8 array,
    tinted with the tracker type colour.

    Parameters
    ----------
    heatmap     : np.ndarray  shape (H, W), float32, values in [0, 1]
    ttype       : TrackerType
    alpha_scale : float       Max alpha = alpha_scale * 255

    Returns
    -------
    np.ndarray  shape (H, W, 4), uint8
    """
    import numpy as np
    r, g, b, _ = type_rgba(ttype)
    H, W = heatmap.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    intensity = heatmap.clip(0, 1)
    rgba[:, :, 0] = r
    rgba[:, :, 1] = g
    rgba[:, :, 2] = b
    rgba[:, :, 3] = (intensity * alpha_scale * 255).astype(np.uint8)
    return rgba


def colorise_mask(mask, ttype: TrackerType, alpha: int = 80):
    """
    Convert a binary uint8 mask (0 / 255) to a coloured RGBA overlay.

    Parameters
    ----------
    mask  : np.ndarray  shape (H, W), uint8
    ttype : TrackerType
    alpha : int         Alpha for matched pixels (0–255)

    Returns
    -------
    np.ndarray  shape (H, W, 4), uint8
    """
    import numpy as np
    r, g, b, _ = type_rgba(ttype)
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    hit = mask > 0
    rgba[hit, 0] = r
    rgba[hit, 1] = g
    rgba[hit, 2] = b
    rgba[hit, 3] = alpha
    return rgba
