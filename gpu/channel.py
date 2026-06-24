"""
gpu/channel.py
--------------
Channel extraction layer — sits at the top of every tracker's pipeline.

Converts a raw BGR uint8 GPU frame (H, W, 3) into a normalised float64
single-channel GPU array (H, W) ready for the Hessian / HOG pipeline.

Supported modes
---------------
GREY    Standard perceptual greyscale  0.299·R + 0.587·G + 0.114·B  (BT.601)
RED     Red channel only
GREEN   Green channel only
BLUE    Blue channel only
CUSTOM  User-supplied (r_w, g_w, b_w) weights, auto-normalised to sum = 1.

The ColorTolerance / Color-Area tracker bypasses this module entirely and
works on the raw BGR frame — channel selection is not applicable there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

try:
    import cupy as cp
except Exception:  # CPU-only Point Fast startup must not require CuPy.
    cp = None


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class ChannelMode(Enum):
    GREY   = auto()
    RED    = auto()
    GREEN  = auto()
    BLUE   = auto()
    CUSTOM = auto()


# ---------------------------------------------------------------------------
# Config dataclass (serialisable → stored per-tracker in TrackerConfig)
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """
    Full channel-extraction configuration for one tracker (or the global default).

    custom_weights is only used when mode == CUSTOM.
    Weights are normalised on first use so the user doesn't have to ensure
    they sum to 1.0.
    """
    mode: ChannelMode = ChannelMode.GREY
    # (r_weight, g_weight, b_weight) — raw, un-normalised
    custom_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def normalised_weights(self) -> Tuple[float, float, float]:
        """Return custom weights normalised so they sum to 1.0."""
        r, g, b = self.custom_weights
        total = r + g + b
        if total < 1e-9:
            return (1.0 / 3, 1.0 / 3, 1.0 / 3)
        return (r / total, g / total, b / total)

    def display_label(self) -> str:
        """Human-readable label for the UI dropdown."""
        labels = {
            ChannelMode.GREY:   'Greyscale (BT.601)',
            ChannelMode.RED:    'Red channel',
            ChannelMode.GREEN:  'Green channel',
            ChannelMode.BLUE:   'Blue channel',
            ChannelMode.CUSTOM: self._custom_label(),
        }
        return labels[self.mode]

    def _custom_label(self) -> str:
        r, g, b = self.normalised_weights()
        return f'Custom ({r:.2f}R + {g:.2f}G + {b:.2f}B)'

    def to_dict(self) -> dict:
        return {
            'mode': self.mode.name,
            'custom_weights': list(self.custom_weights),
        }

    @staticmethod
    def from_dict(d: dict) -> 'ChannelConfig':
        return ChannelConfig(
            mode=ChannelMode[d['mode']],
            custom_weights=tuple(d.get('custom_weights', [1.0, 1.0, 1.0])),
        )


# ---------------------------------------------------------------------------
# BT.601 weights (constant)
# ---------------------------------------------------------------------------
_GREY_W_R = 0.299
_GREY_W_G = 0.587
_GREY_W_B = 0.114


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_channel(
    frame_gpu: cp.ndarray,
    config: ChannelConfig,
) -> cp.ndarray:
    """
    Extract a single-channel float64 image from a BGR uint8 GPU frame.

    Parameters
    ----------
    frame_gpu : cp.ndarray  shape (H, W, 3), dtype uint8, BGR, on GPU.
    config    : ChannelConfig  specifying which channel / weights to use.

    Returns
    -------
    out : cp.ndarray  shape (H, W), dtype float64, values in [0, 1], on GPU.
    """
    # Normalise to [0, 1] in float64
    if cp is None:
        raise RuntimeError('GPU channel extraction requires CuPy.')
    frame_f = frame_gpu.astype(cp.float64) / 255.0  # (H, W, 3)

    b = frame_f[:, :, 0]
    g = frame_f[:, :, 1]
    r = frame_f[:, :, 2]

    if config.mode == ChannelMode.GREY:
        return _GREY_W_R * r + _GREY_W_G * g + _GREY_W_B * b

    elif config.mode == ChannelMode.RED:
        return r.copy()

    elif config.mode == ChannelMode.GREEN:
        return g.copy()

    elif config.mode == ChannelMode.BLUE:
        return b.copy()

    elif config.mode == ChannelMode.CUSTOM:
        rw, gw, bw = config.normalised_weights()
        return rw * r + gw * g + bw * b

    else:
        raise ValueError(f"Unknown ChannelMode: {config.mode}")


def upload_frame(frame_bgr_cpu: 'np.ndarray') -> cp.ndarray:
    """
    Upload a CPU BGR uint8 frame (from OpenCV) to GPU memory.

    Parameters
    ----------
    frame_bgr_cpu : np.ndarray  shape (H, W, 3) uint8.

    Returns
    -------
    cp.ndarray  shape (H, W, 3) uint8, on GPU.
    """
    if cp is None:
        raise RuntimeError('GPU frame upload requires CuPy.')
    return cp.asarray(frame_bgr_cpu)


# ---------------------------------------------------------------------------
# Global default (module-level singleton, mutated by Settings dialog)
# ---------------------------------------------------------------------------

_global_default: ChannelConfig = ChannelConfig(mode=ChannelMode.GREY)


def get_global_default() -> ChannelConfig:
    """Return the current application-wide default ChannelConfig."""
    return _global_default


def set_global_default(config: ChannelConfig) -> None:
    """Update the application-wide default (called from Settings dialog)."""
    global _global_default
    _global_default = config


def resolve(config: 'ChannelConfig | None') -> ChannelConfig:
    """
    If *config* is None (tracker uses global default), return the global
    default.  Otherwise return *config* as-is.

    Convenience used by every tracker's pipeline entry point.
    """
    return _global_default if config is None else config
