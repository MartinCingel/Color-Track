"""Optional CUDA/CuPy capability detection used by CPU-first application paths."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def cupy_runtime_available() -> bool:
    try:
        import cupy as cp
        cp.asarray([0], dtype=cp.uint8)
        return True
    except Exception:
        return False


def require_cupy() -> None:
    if not cupy_runtime_available():
        raise RuntimeError("This tracker mode requires a working CuPy GPU runtime.")
