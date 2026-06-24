"""
utils/state.py
--------------
Lightweight application-state singleton.

Rather than threading a "state" argument through every widget, components
can read/write well-defined fields here.  The MainWindow is the only writer
for most fields; UI widgets read them when they need context.

Not thread-safe by design — all state mutations happen on the Qt UI thread.
The batch worker thread reads only frame data from FrameBuffer and never
touches this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class InteractionMode(Enum):
    """What happens when the user clicks on the video canvas."""
    IDLE         = auto()   # No pending action
    SEED_CLICK   = auto()   # Next click places a tracker seed point
    ROI_DRAG     = auto()   # Next drag selects a bounding-box ROI
    REINIT_CLICK = auto()   # Next click re-seeds a specific (uid) tracker


@dataclass
class AppState:
    """
    Central mutable state for the running application.

    All fields have sensible defaults so the object is valid before
    a video is opened.
    """

    # ---- Video ----
    video_path:    Optional[str] = None
    frame_count:   int           = 0
    fps:           float         = 25.0
    video_width:   int           = 1
    video_height:  int           = 1

    # ---- Playback ----
    current_frame: int  = 0
    start_frame:   int  = 0
    end_frame:     int  = 0      # 0 = use frame_count

    # ---- Interaction ----
    mode:           InteractionMode = InteractionMode.IDLE
    pending_uid:    Optional[str]   = None   # tracker UID awaiting seed
    reinit_uid:     Optional[str]   = None   # tracker UID being re-seeded

    # ---- Batch ----
    batch_running:  bool = False

    # ---- Display ----
    show_heatmaps:  bool = True    # toggle Hessian heatmap overlays
    show_labels:    bool = True    # toggle tracker name labels on canvas

    def reset_interaction(self) -> None:
        """Return to IDLE after a seed or ROI has been received."""
        self.mode        = InteractionMode.IDLE
        self.pending_uid = None
        self.reinit_uid  = None

    def effective_end_frame(self) -> int:
        """Return the actual end frame for batch processing."""
        return self.end_frame if self.end_frame > 0 else max(0, self.frame_count - 1)

    def is_video_open(self) -> bool:
        return self.video_path is not None and self.frame_count > 0


# Module-level singleton
_state: AppState = AppState()


def get() -> AppState:
    """Return the global AppState instance."""
    return _state


def reset() -> None:
    """Reset state to defaults (called when a new video is opened)."""
    global _state
    _state = AppState()
