"""
degradation.py
==============
LIFT-550 degradation pipeline: Tiers 1, 2, and 3.

All functions operate on torch tensors of shape [C, H, W] or [B, C, H, W].
Degradation is applied BEFORE normalization in the data pipeline.

Tier 1 - Filtered DNS    : spectral truncation -> downsample -> bicubic upsample
Tier 2 - LES proxy       : T1 + Gaussian smoothing (sigma = 0.5 * r)
Tier 3 - RANS proxy      : aggressive truncation + heavy smoothing + noise

All tiers use r = 8 (64x64 low-res intermediate).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

# ── Protocol constants ────────────────────────────────────────────────────────
R        = 8          # downscaling factor
NX       = 512
NZ       = 512
LR_SIZE  = NX // R    # 64

# Tier 1/2 spectral cutoff: keep |k| <= N/(2r)
K_CUTOFF_T1 = NX // (2 * R)   # 32

# Tier 3 spectral cutoff: keep |k| <= N/(4r)  -- 2x more aggressive
K_CUTOFF_T3 = NX // (4 * R)   # 16

# Gaussian sigma values
SIGMA_T2 = 0.5
SIGMA_T3 = 0.3

# Tier 3 noise std
NOISE_STD = 0.05


# ── Spectral truncation ───────────────────────────────────────────────────────

def spectral_truncate(plane: torch.Tensor, k_cutoff: int) -> torch.Tensor:
    """
    Apply a 2D ideal low-pass filter in spectral space.
    Zeros out all wavenumbers |kx| > k_cutoff or |kz| > k_cutoff.

    Parameters
    ----------
    plane    : [..., H, W] float tensor
    k_cutoff : maximum retained wavenumber (inclusive)

    Returns
    -------
    [..., H, W] float tensor, real-valued
    """
    # FFT
    f = torch.fft.rfft2(plane)   # [..., H, W//2+1] complex

    H, W_spec = f.shape[-2], f.shape[-1]

    # Build wavenumber mask
    # kz (rows): frequencies 0..H//2, then aliased -(H//2)..−1
    kz = torch.fft.fftfreq(H, d=1.0) * H   # cycles over H points
    # kx (cols): 0..W//2 (rfft, only positive freqs)
    kx = torch.arange(W_spec, dtype=torch.float32)

    kz = kz.to(plane.device)
    kx = kx.to(plane.device)

    mask_kz = (torch.abs(kz) <= k_cutoff)   # (H,)
    mask_kx = (kx <= k_cutoff)              # (W_spec,)

    mask = (mask_kz[:, None] & mask_kx[None, :])  # (H, W_spec)

    # Broadcast mask to batch/channel dims
    for _ in range(f.dim() - 2):
        mask = mask.unsqueeze(0)

    f = f * mask.float()

    return torch.fft.irfft2(f, s=(H, plane.shape[-1]))


# ── Gaussian smoothing kernel ─────────────────────────────────────────────────

def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """
    Build a 2D Gaussian kernel for use with F.conv2d.
    Kernel size = 2 * ceil(3*sigma) + 1 (covers ±3 sigma).
    """
    radius = int(np.ceil(3 * sigma))
    size   = 2 * radius + 1
    x      = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d    = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g1d    = g1d / g1d.sum()
    kernel = g1d[:, None] * g1d[None, :]   # (size, size)
    return kernel


def gaussian_smooth(plane: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing with periodic (circular) padding.

    Parameters
    ----------
    plane : [C, H, W] float tensor
    sigma : Gaussian standard deviation in grid units

    Returns
    -------
    [C, H, W] smoothed tensor
    """
    kernel = _gaussian_kernel(sigma, plane.device)
    radius = kernel.shape[0] // 2
    C      = plane.shape[0]

    # Periodic padding
    padded = F.pad(
        plane.unsqueeze(0),                       # [1, C, H, W]
        (radius, radius, radius, radius),
        mode="circular"
    )

    # Apply depthwise conv (each channel independently)
    k = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)  # [C, 1, ks, ks]
    out = F.conv2d(padded, k, groups=C)                         # [1, C, H, W]
    return out.squeeze(0)                                        # [C, H, W]


# ── Downsample + upsample ─────────────────────────────────────────────────────

def downsample(plane: torch.Tensor, factor: int) -> torch.Tensor:
    """Spatial downsampling by taking every factor-th grid point."""
    return plane[..., ::factor, ::factor]   # [C, H//r, W//r]


def bicubic_upsample(plane: torch.Tensor, size: tuple) -> torch.Tensor:
    """Bicubic upsampling to target spatial size."""
    return F.interpolate(
        plane.unsqueeze(0),          # [1, C, H, W]
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).squeeze(0)                     # [C, H, W]


# ── Tier implementations ──────────────────────────────────────────────────────

def degrade_t1(plane: torch.Tensor) -> torch.Tensor:
    """
    Tier 1 — Filtered DNS (best-case).

    Steps:
      1. Spectral truncation at k_cutoff = Nx/(2r) = 32
      2. Spatial downsample to 64x64
      3. Bicubic upsample to 512x512
    """
    x = spectral_truncate(plane, K_CUTOFF_T1)
    x = downsample(x, R)
    x = bicubic_upsample(x, (NZ, NX))
    return x


def degrade_t2(plane: torch.Tensor) -> torch.Tensor:
    """
    Tier 2 — LES proxy (realistic).

    Steps:
      1. Spectral truncation at k_cutoff = 32
      2. Spatial downsample to 64x64
      3. Gaussian smoothing (sigma = 4.0 grid units at 64x64)
      4. Bicubic upsample to 512x512
    """
    x = spectral_truncate(plane, K_CUTOFF_T1)
    x = downsample(x, R)
    x = gaussian_smooth(x, SIGMA_T2)
    x = bicubic_upsample(x, (NZ, NX))
    return x


def degrade_t3(
    plane: torch.Tensor,
    noise_seed: Optional[int] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    Tier 3 — RANS proxy (worst-case).

    Steps:
      1. Aggressive spectral truncation at k_cutoff = Nx/(4r) = 16
      2. Spatial downsample to 64x64
      3. Heavy Gaussian smoothing (sigma = 8.0 grid units at 64x64)
      4. Additive Gaussian noise ~ N(0, 0.05^2), independent per channel
      5. Bicubic upsample to 512x512

    Parameters
    ----------
    plane      : [C, H, W] tensor
    noise_seed : fixed seed for evaluation (reproducible noise);
                 if None and training=True, noise is freshly sampled each call
    training   : if True and noise_seed is None, noise is random (augmentation)
    """
    x = spectral_truncate(plane, K_CUTOFF_T3)
    x = downsample(x, R)
    x = gaussian_smooth(x, SIGMA_T3)

    # Noise
    if noise_seed is not None:
        rng   = torch.Generator(device=plane.device)
        rng.manual_seed(noise_seed)
        noise = torch.randn(*x.shape, generator=rng, device=plane.device) * NOISE_STD
    elif training:
        noise = torch.randn_like(x) * NOISE_STD
    else:
        # Evaluation without a seed: deterministic zero noise (conservative)
        noise = torch.zeros_like(x)

    x = x + noise
    x = bicubic_upsample(x, (NZ, NX))
    return x


# ── Dispatch by tier ──────────────────────────────────────────────────────────

TIER_FN = {1: degrade_t1, 2: degrade_t2, 3: degrade_t3}

def degrade(
    plane: torch.Tensor,
    tier: int,
    noise_seed: Optional[int] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    Apply degradation for the given tier.

    Parameters
    ----------
    plane      : [C, H, W] float32 tensor (NOT yet normalized)
    tier       : 1, 2, or 3
    noise_seed : fixed seed for Tier 3 evaluation noise
    training   : controls Tier 3 noise sampling

    Returns
    -------
    [C, H, W] float32 tensor, same shape as input
    """
    if tier not in TIER_FN:
        raise ValueError(f"tier must be 1, 2, or 3 — got {tier}")
    if tier == 3:
        return degrade_t3(plane, noise_seed=noise_seed, training=training)
    return TIER_FN[tier](plane)


# ── Sanity checks (run directly to test) ─────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("Running degradation sanity checks...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Synthetic turbulence-like field: sum of Fourier modes
    torch.manual_seed(0)
    C = 3
    field = torch.randn(C, NZ, NX, device=device)

    for tier in [1, 2, 3]:
        out = degrade(field, tier=tier, training=True)
        assert out.shape == (C, NZ, NX), f"Shape mismatch tier {tier}: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in tier {tier} output"
        assert not torch.isinf(out).any(), f"Inf in tier {tier} output"

        # Energy should be reduced vs input
        e_in  = (field ** 2).mean().item()
        e_out = (out   ** 2).mean().item()

        print(f"  Tier {tier}: output shape={tuple(out.shape)}  "
              f"input_energy={e_in:.4f}  output_energy={e_out:.4f}  "
              f"ratio={e_out/e_in:.3f}")

    # Check Tier 3 noise reproducibility
    out_a = degrade(field, tier=3, noise_seed=42, training=False)
    out_b = degrade(field, tier=3, noise_seed=42, training=False)
    assert torch.allclose(out_a, out_b), "Tier 3 noise not reproducible with fixed seed"
    print("  Tier 3 noise reproducibility: OK")

    # Check Tier 3 training noise is different each call
    out_c = degrade(field, tier=3, noise_seed=None, training=True)
    out_d = degrade(field, tier=3, noise_seed=None, training=True)
    assert not torch.allclose(out_c, out_d), "Tier 3 training noise is not random"
    print("  Tier 3 training noise randomness: OK")

    print("\nAll checks passed.")
