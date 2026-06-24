"""Writable per-user locations for installed Color Track builds."""

from __future__ import annotations

import os
from pathlib import Path


def app_data_dir() -> Path:
    """Return Color Track's user-writable data directory."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return Path(base) / "Color Track" if base else Path.home() / ".colortrack"


def app_data_path(*parts: str) -> Path:
    path = app_data_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
