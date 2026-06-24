"""
gpu/fft_utils.py
----------------
Frequency-domain utility helpers shared across the Hessian pipeline.

Key idea
--------
For a 2-D image I of shape (H, W), the second partial derivatives can be
obtained purely in the frequency domain:

    FFT(∂²I/∂x²) = -u²  · FFT(I)
    FFT(∂²I/∂y²) = -v²  · FFT(I)
    FFT(∂²I/∂xy) = -u·v · FFT(I)

where u, v are the normalised angular frequencies produced by cp.fft.fftfreq.

Because FFT(I) is computed once and reused, we get all three second-derivative
maps for the cost of a single forward FFT + three element-wise multiplications
+ three inverse FFTs.

GPU notes
---------
* All arrays live on the GPU as CuPy ndarrays.
* cuFFT (via CuPy) is used automatically when cp.fft is called.
* Frequency grids are built once per unique (H, W) shape and cached in a
  module-level dict so repeated calls on same-sized frames pay zero overhead.
* Thread-safety: the cache is written only from the main processing thread
  (TrackerManager serialises frame dispatch), so no lock is needed.
"""

from __future__ import annotations

import cupy as cp
from typing import Tuple

# ---------------------------------------------------------------------------
# Module-level cache:  (H, W) -> (U2, V2, UV)  frequency-grid tensors
# ---------------------------------------------------------------------------
_FREQ_CACHE: dict[Tuple[int, int], Tuple[cp.ndarray, cp.ndarray, cp.ndarray]] = {}


def get_freq_grids(H: int, W: int) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Return cached (U², V², U·V) frequency grids for a frame of size (H, W).

    All returned arrays are complex128-compatible float64 on the GPU and have
    shape (H, W).  They are multiplied directly against FFT(I) to obtain the
    spectra of the three second derivatives.

    Parameters
    ----------
    H, W : int
        Frame height and width in pixels.

    Returns
    -------
    u2 : cp.ndarray  shape (H, W)   -u²  kernel
    v2 : cp.ndarray  shape (H, W)   -v²  kernel
    uv : cp.ndarray  shape (H, W)   -u·v kernel
    """
    key = (H, W)
    if key not in _FREQ_CACHE:
        # Angular frequency axes  (cycles per pixel, range [-π, π])
        u = cp.fft.fftfreq(W).astype(cp.float64) * (2.0 * cp.pi)   # (W,)
        v = cp.fft.fftfreq(H).astype(cp.float64) * (2.0 * cp.pi)   # (H,)

        # Build 2-D grids:  V varies along axis-0, U along axis-1
        V, U = cp.meshgrid(v, u, indexing='ij')   # both (H, W)

        _FREQ_CACHE[key] = (
            -(U ** 2),   # u2 : multiplied → FFT(I_xx)
            -(V ** 2),   # v2 : multiplied → FFT(I_yy)
            -(U * V),    # uv : multiplied → FFT(I_xy)
        )
    return _FREQ_CACHE[key]


def gaussian_kernel_fft(H: int, W: int, sigma: float) -> cp.ndarray:
    """
    Return the FFT of a 2-D Gaussian with std-dev *sigma* in frequency space,
    ready to be multiplied against FFT(I) for pre-smoothing.

    The Gaussian in frequency space is:
        G(u, v) = exp( -sigma² (u² + v²) / 2 )

    Parameters
    ----------
    H, W  : int    Frame dimensions.
    sigma : float  Smoothing scale in pixels (user-supplied feature size).

    Returns
    -------
    G_fft : cp.ndarray  shape (H, W), dtype complex128  (real-valued spectrum)
    """
    u = cp.fft.fftfreq(W).astype(cp.float64) * (2.0 * cp.pi)
    v = cp.fft.fftfreq(H).astype(cp.float64) * (2.0 * cp.pi)
    V, U = cp.meshgrid(v, u, indexing='ij')
    G = cp.exp(-0.5 * (sigma ** 2) * (U ** 2 + V ** 2))
    # Return as complex so it can be directly multiplied against FFT(I)
    return G.astype(cp.complex128)


def fft2_image(image: cp.ndarray) -> cp.ndarray:
    """
    Compute the 2-D FFT of a single-channel float image on the GPU.

    Parameters
    ----------
    image : cp.ndarray  shape (H, W), any real dtype
        Single-channel image already on GPU.

    Returns
    -------
    F : cp.ndarray  shape (H, W), dtype complex128
    """
    return cp.fft.fft2(image.astype(cp.float64))


def ifft2_real(spectrum: cp.ndarray) -> cp.ndarray:
    """
    Inverse 2-D FFT, returning only the real part as float64.

    Imaginary residuals from floating-point rounding are discarded.

    Parameters
    ----------
    spectrum : cp.ndarray  shape (H, W), complex128

    Returns
    -------
    out : cp.ndarray  shape (H, W), float64
    """
    return cp.fft.ifft2(spectrum).real


def smooth_fft(F: cp.ndarray, G: cp.ndarray) -> cp.ndarray:
    """
    Apply Gaussian smoothing in frequency domain.

    Parameters
    ----------
    F : cp.ndarray  FFT of image, shape (H, W), complex128
    G : cp.ndarray  Gaussian kernel FFT from gaussian_kernel_fft(), same shape

    Returns
    -------
    F_smooth : cp.ndarray  shape (H, W), complex128
    """
    return F * G


def compute_second_derivatives(
    F_smooth: cp.ndarray,
    H: int,
    W: int,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Compute all three second-derivative maps from a (smoothed) FFT.

    Uses cached frequency grids — no grid recomputation on repeated calls.

    Parameters
    ----------
    F_smooth : cp.ndarray  shape (H, W), complex128
        FFT of the (optionally smoothed) image.
    H, W : int
        Frame dimensions (must match F_smooth.shape).

    Returns
    -------
    Ixx : cp.ndarray  shape (H, W), float64   ∂²I/∂x²
    Iyy : cp.ndarray  shape (H, W), float64   ∂²I/∂y²
    Ixy : cp.ndarray  shape (H, W), float64   ∂²I/∂x∂y
    """
    u2, v2, uv = get_freq_grids(H, W)
    Ixx = ifft2_real(F_smooth * u2)
    Iyy = ifft2_real(F_smooth * v2)
    Ixy = ifft2_real(F_smooth * uv)
    return Ixx, Iyy, Ixy


def clear_cache() -> None:
    """
    Free all cached frequency grids from GPU memory.
    Call when the video changes resolution or on application shutdown.
    """
    _FREQ_CACHE.clear()
