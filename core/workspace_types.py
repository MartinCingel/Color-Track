"""Pure data structures shared by prepared-workspace caching and CPU trackers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class WorkspaceRect:
    """Rectangle in original full-frame coordinates."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def full_frame(cls, width: int, height: int) -> "WorkspaceRect":
        return cls(0, 0, int(width), int(height))

    def clamped(self, full_width: int, full_height: int) -> "WorkspaceRect":
        x0 = max(0, min(int(self.x), int(full_width) - 1))
        y0 = max(0, min(int(self.y), int(full_height) - 1))
        x1 = max(x0 + 1, min(int(full_width), int(self.x + self.width)))
        y1 = max(y0 + 1, min(int(full_height), int(self.y + self.height)))
        return WorkspaceRect(x0, y0, x1 - x0, y1 - y0)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def is_full_frame(self, full_width: int, full_height: int) -> bool:
        return self == WorkspaceRect.full_frame(full_width, full_height)


@dataclass(frozen=True)
class WorkspaceFrame:
    """One cached BGR workspace frame with original coordinate mapping."""

    bgr: np.ndarray
    origin_xy: tuple[int, int]
    full_frame_hw: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)
