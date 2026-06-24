"""
tests/smoke_test.py
-------------------
Validates module structure, imports, and CPU-side logic without requiring
a real GPU, video file, or Qt display.

Run with:
    python -m pytest tests/smoke_test.py -v
OR:
    python tests/smoke_test.py

GPU-dependent code paths are monkey-patched with numpy stubs so the tests
run on any machine (CI, CPU-only laptops, etc.).
"""

from __future__ import annotations

import sys
import os
import types
import numpy as np

# ---------------------------------------------------------------------------
# 0. Add project root to path
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# 1. Stub CuPy with a numpy-backed shim so gpu/ modules can be imported
# ---------------------------------------------------------------------------

def _make_cupy_stub():
    """
    Build a minimal cupy-compatible module backed by numpy.
    Only the symbols actually used by gpu/ modules are needed.
    """
    cp = types.ModuleType('cupy')

    # Array operations → delegate to numpy
    for name in [
        'array', 'zeros', 'ones', 'empty', 'empty_like',
        'asarray', 'ascontiguousarray',
        'where', 'abs', 'maximum', 'minimum', 'sqrt', 'exp',
        'meshgrid', 'arange', 'linspace',
    ]:
        setattr(cp, name, getattr(np, name))

    cp.ndarray  = np.ndarray
    cp.float32  = np.float32
    cp.float64  = np.float64
    cp.uint8    = np.uint8
    cp.complex128 = np.complex128
    cp.bool_    = np.bool_
    cp.pi       = np.pi

    # cp.fft → numpy.fft
    cp.fft = types.ModuleType('cupy.fft')
    cp.fft.fft2    = np.fft.fft2
    cp.fft.ifft2   = np.fft.ifft2
    cp.fft.fftfreq = np.fft.fftfreq

    # cp.cuda (stubs)
    cuda = types.ModuleType('cupy.cuda')
    stream_cls = type('Stream', (), {
        '__init__': lambda self, **kw: None,
        '__enter__': lambda self: self,
        '__exit__':  lambda self, *a: None,
    })
    cuda.Stream = stream_cls
    cp.cuda = cuda

    # ElementwiseKernel stub
    class _EKStub:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass
    cp.ElementwiseKernel = _EKStub

    return cp


_cp_stub = _make_cupy_stub()
sys.modules['cupy'] = _cp_stub


# Stub cupyx.scipy.ndimage
_cupyx = types.ModuleType('cupyx')
_cupyx_scipy = types.ModuleType('cupyx.scipy')
_cupyx_scipy_ndimage = types.ModuleType('cupyx.scipy.ndimage')
_cupyx_scipy_ndimage.maximum_filter = lambda arr, **kw: arr
_cupyx.scipy = _cupyx_scipy
_cupyx_scipy.ndimage = _cupyx_scipy_ndimage
sys.modules['cupyx'] = _cupyx
sys.modules['cupyx.scipy'] = _cupyx_scipy
sys.modules['cupyx.scipy.ndimage'] = _cupyx_scipy_ndimage


# ---------------------------------------------------------------------------
# 2. Import all modules under test
# ---------------------------------------------------------------------------

def test_import_gpu_fft_utils():
    from gpu import fft_utils
    assert hasattr(fft_utils, 'get_freq_grids')
    assert hasattr(fft_utils, 'gaussian_kernel_fft')
    assert hasattr(fft_utils, 'compute_second_derivatives')
    print('  ✓ gpu.fft_utils imports OK')


def test_import_gpu_channel():
    from gpu.channel import ChannelConfig, ChannelMode, extract_channel
    cfg = ChannelConfig(mode=ChannelMode.GREY)
    assert cfg.display_label() == 'Greyscale (BT.601)'
    assert cfg.normalised_weights() == (1.0/3, 1.0/3, 1.0/3)  # default custom

    cfg2 = ChannelConfig(mode=ChannelMode.CUSTOM, custom_weights=(2.0, 1.0, 1.0))
    r, g, b = cfg2.normalised_weights()
    assert abs(r - 0.5) < 1e-6
    assert abs(g - 0.25) < 1e-6

    d = cfg2.to_dict()
    cfg3 = ChannelConfig.from_dict(d)
    assert cfg3.mode == ChannelMode.CUSTOM
    print('  ✓ gpu.channel imports and config round-trips OK')


def test_import_gpu_color_mask():
    from gpu.color_mask import ColorTolerance, sample_pixel, compute_mask
    tol = ColorTolerance.from_bgr_pixel((128, 64, 200))
    assert 0 <= tol.center_h < 360
    assert 0 <= tol.center_s <= 1
    assert 0 <= tol.center_v <= 1
    print('  ✓ gpu.color_mask imports OK')


def test_import_gpu_hessian():
    from gpu import hessian
    assert hasattr(hessian, 'detect_blobs')
    assert hasattr(hessian, 'detect_ridges')
    assert hasattr(hessian, 'hessian_score_at_bbox')
    print('  ✓ gpu.hessian imports OK')


def test_import_tracking_base():
    from tracking.base_tracker import (
        TrackerConfig, TrackerType, TrackerStatus,
        FrameResult, InitPreview,
    )
    cfg = TrackerConfig(tracker_type=TrackerType.POINT_FAST, name='test', sigma=5.0)
    assert cfg.uid  # auto-generated
    d = cfg.to_dict()
    cfg2 = TrackerConfig.from_dict(d)
    assert cfg2.tracker_type == TrackerType.POINT_FAST
    assert cfg2.sigma == 5.0
    print('  ✓ tracking.base_tracker imports and config round-trips OK')


def test_import_tracking_trackers():
    from tracking.trackers import (
        PointAccurateTracker, BlobSimpleTracker, BlobComplexTracker,
        CurveTracker, ColorAreaTracker,
    )
    from tracking.point_fast import PointFastTracker
    from tracking.base_tracker import TrackerConfig, TrackerType
    for cls, ttype in [
        (PointFastTracker,     TrackerType.POINT_FAST),
        (PointAccurateTracker, TrackerType.POINT_ACCURATE),
        (BlobSimpleTracker,    TrackerType.BLOB_SIMPLE),
        (BlobComplexTracker,   TrackerType.BLOB_COMPLEX),
        (CurveTracker,         TrackerType.CURVE),
        (ColorAreaTracker,     TrackerType.COLOR_AREA),
    ]:
        cfg = TrackerConfig(tracker_type=ttype, sigma=5.0)
        t   = cls(cfg)
        assert t.uid == cfg.uid
        assert t.tracker_type == ttype
    print('  ✓ All 6 tracker classes instantiate OK')


def test_import_utils():
    import utils.state as state
    import utils.colors as colors
    from tracking.base_tracker import TrackerType, TrackerStatus

    state.reset()
    s = state.get()
    assert s.current_frame == 0
    s.current_frame = 42
    assert state.get().current_frame == 42

    r, g, b, a = colors.type_rgba(TrackerType.CURVE)
    assert 0 <= r <= 255
    hex_ = colors.type_hex(TrackerType.BLOB_SIMPLE)
    assert hex_.startswith('#')

    rgba_map = colors.colorise_heatmap(np.zeros((4, 4), dtype=np.float32),
                                        TrackerType.CURVE)
    assert rgba_map.shape == (4, 4, 4)
    print('  ✓ utils.state and utils.colors OK')


def test_export_schema():
    """Verify export module imports and estimate function works (Qt stubbed)."""
    # Stub PyQt6 minimally so tracker_manager can be imported
    if 'PyQt6' not in sys.modules:
        qt = types.ModuleType('PyQt6')
        qtcore = types.ModuleType('PyQt6.QtCore')

        class _FakeSignal:
            def __init__(self, *a): pass
            def connect(self, *a): pass
            def emit(self, *a): pass

        class _QObjectMeta(type):
            pass

        class QObject:
            def __init__(self, parent=None): pass
            def moveToThread(self, t): pass

        class QThread:
            def __init__(self): pass
            def start(self): pass
            def quit(self): pass
            def wait(self): pass
            def isRunning(self): return False
            started = _FakeSignal()

        qtcore.QObject    = QObject
        qtcore.QThread    = QThread
        qtcore.pyqtSignal = _FakeSignal
        qt.QtCore         = qtcore
        sys.modules['PyQt6']       = qt
        sys.modules['PyQt6.QtCore'] = qtcore

    from core import export
    assert hasattr(export, 'export_npz')
    assert hasattr(export, 'estimate_export_size_mb')
    print('  ✓ core.export imports OK')


def test_color_tolerance_round_trip():
    from gpu.color_mask import ColorTolerance
    tol = ColorTolerance(
        center_h=120.0, delta_h=15.0,
        center_s=0.7,   delta_s=0.2,
        center_v=0.8,   delta_v=0.25,
    )
    # Simulate what the panel does: read fields, re-build
    tol2 = ColorTolerance(
        center_h=tol.center_h, delta_h=tol.delta_h,
        center_s=tol.center_s, delta_s=tol.delta_s,
        center_v=tol.center_v, delta_v=tol.delta_v,
    )
    assert tol.center_h == tol2.center_h
    assert tol.delta_v  == tol2.delta_v
    print('  ✓ ColorTolerance round-trip OK')


def test_channel_extract_cpu():
    """Run extract_channel on a fake CPU (numpy-backed) frame."""
    from gpu.channel import ChannelConfig, ChannelMode, extract_channel
    import numpy as np

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    for mode in ChannelMode:
        cfg = ChannelConfig(mode=mode, custom_weights=(0.5, 0.3, 0.2))
        out = extract_channel(frame, cfg)
        assert out.shape == (64, 64), f"shape mismatch for {mode}"
        assert out.dtype == np.float64
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    print('  ✓ extract_channel works for all modes on CPU-backed arrays')


# ---------------------------------------------------------------------------
# 3. Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_import_gpu_fft_utils,
        test_import_gpu_channel,
        test_import_gpu_color_mask,
        test_import_gpu_hessian,
        test_import_tracking_base,
        test_import_tracking_trackers,
        test_import_utils,
        test_export_schema,
        test_color_tolerance_round_trip,
        test_channel_extract_cpu,
    ]

    print('\n=== Color Track Smoke Tests ===\n')
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f'  ✗ {test.__name__}: {exc}')
            import traceback; traceback.print_exc()
            failed += 1

    print(f'\n{"="*34}')
    print(f'  {passed} passed, {failed} failed')
    if failed:
        sys.exit(1)
    else:
        print('  All smoke tests passed ✓')
