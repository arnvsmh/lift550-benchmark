"""
metrics.py
==========
LIFT-550 evaluation metrics (Section 9 of protocol).

Metrics:
  - RMSE  : Root Mean Squared Error (physical units, post-denormalization)
  - NRMSE : Normalized RMSE = RMSE / std(DNS target)
  - SSIM  : Structural Similarity Index
  - E(k)  : Turbulent kinetic energy spectrum error (spectral)
"""

import numpy as np
import torch


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error over full field."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized RMSE = RMSE / std(target)."""
    return rmse(pred, target) / (float(np.std(target)) + 1e-8)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = None) -> float:
    """
    Structural Similarity Index.
    Computed per channel then averaged.
    pred, target: [C, H, W] or [H, W]
    """
    if pred.ndim == 2:
        pred   = pred[None]
        target = target[None]

    if data_range is None:
        data_range = float(target.max() - target.min())

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_vals = []
    for c in range(pred.shape[0]):
        p = pred[c].astype(np.float64)
        t = target[c].astype(np.float64)

        mu_p  = _gaussian_filter(p)
        mu_t  = _gaussian_filter(t)
        mu_pp = _gaussian_filter(p * p)
        mu_tt = _gaussian_filter(t * t)
        mu_pt = _gaussian_filter(p * t)

        sigma_p  = mu_pp - mu_p ** 2
        sigma_t  = mu_tt - mu_t ** 2
        sigma_pt = mu_pt - mu_p * mu_t

        num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        den = (mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2)
        ssim_vals.append(float(np.mean(num / (den + 1e-12))))

    return float(np.mean(ssim_vals))


def _gaussian_filter(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Simple Gaussian blur for SSIM computation."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=sigma)


def energy_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D turbulent kinetic energy spectrum E(k).
    field: [H, W] single velocity component
    Returns: (k, E) wavenumber bins and energy
    """
    H, W   = field.shape
    fft2   = np.fft.fft2(field)
    power  = (np.abs(fft2) ** 2) / (H * W) ** 2

    # Wavenumber grid
    kx = np.fft.fftfreq(W, d=1.0/W)
    ky = np.fft.fftfreq(H, d=1.0/H)
    KX, KY = np.meshgrid(kx, ky)
    K      = np.sqrt(KX**2 + KY**2)

    k_max  = int(min(H, W) // 2)
    k_bins = np.arange(0, k_max + 1)
    E      = np.zeros(k_max + 1)

    for ki in range(k_max + 1):
        mask    = (K >= ki - 0.5) & (K < ki + 0.5)
        E[ki]   = np.sum(power[mask])

    return k_bins, E


def spectral_error(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Normalized spectral energy error.
    Uses globally normalized absolute error:
        sum(|E_pred - E_target|) / sum(E_target)
    averaged over channels. Avoids division by near-zero
    high-wavenumber bins which causes explosive variance.
    pred, target: [C, H, W]
    """
    errors = []
    for c in range(pred.shape[0]):
        _, E_pred   = energy_spectrum(pred[c])
        _, E_target = energy_spectrum(target[c])
        denom = np.sum(E_target)
        if denom < 1e-12:
            continue
        err = float(np.sum(np.abs(E_pred - E_target)) / denom)
        errors.append(err)
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def compute_all_metrics(
    pred:   np.ndarray,
    target: np.ndarray,
) -> dict:
    """
    Compute all LIFT-550 metrics for a single sample.
    pred, target: [C, H, W] float32, denormalized (physical units)
    """
    return {
        "rmse":           rmse(pred, target),
        "nrmse":          nrmse(pred, target),
        "ssim":           ssim(pred, target),
        "spectral_error": spectral_error(pred, target),
    }
