"""
trainer_yasunet_v4.py
=====================
Trainer for YASU-Net v4. Architecture is IDENTICAL to v3 (yasunet_v3.py,
modes=10, n_refine=6) — every v4 change lives in the training objective so the
improvement is cleanly attributable to the loss, not to added capacity.

GOAL (per project lead): slightly beat the Kim CycleGAN baseline on NRMSE while
DOMINATING on spectral error. v4 keeps the model deterministic (clean comparison
against the deterministic baselines; avoids the perception–distortion NRMSE risk
of a GAN) and pushes the spectral objective hard.

Four evidence-backed changes over v3, agreed by a three-way review
(Claude / ChatGPT / Gemini):

  1. NEAR-WALL WEIGHTED CHARBONNIER  — the highest-ROI NRMSE lever.
     The model's error is concentrated near the wall: per-y+ NRMSE runs
     ~0.166 at yp15 vs ~0.077 at yp100 (>2x). We weight each sample's pixel
     loss by a smooth, physically motivated function of wall distance,

         w(y+) = min(CAP, C / sqrt(y+)),   normalized so E[w] = 1 over the
                                            protocol y+ bands {15,30,50,100}.

     Smooth 1/sqrt(y+) (not hand-tuned per-band weights) is defensible
     ("near-wall regions have steeper gradients and are physically harder"),
     generalizes to arbitrary y+, and naturally pushes optimization toward the
     worst-performing region. The CAP prevents yp15 from dominating and
     starving the mid-field. Mean-1 normalization keeps the loss scale
     comparable to v3 so val numbers stay interpretable.

  2. STRONG 2-D ANISOTROPIC, LOG, HIGH-k-WEIGHTED SPECTRAL LOSS — the spectral
     domination engine. v3's spectral loss was too weak (lambda=0.05) and used
     1-D radial averaging, which discards the streamwise/spanwise anisotropy of
     wall-bounded channel flow and lets energetic low-k bands drown the high-k
     bands where deterministic models systematically lose energy
     (Cheng et al., Phys. Fluids 2025). v4 instead:
        - works on the full 2-D power spectrum (preserves anisotropy),
        - matches the LOG spectrum (so low-energy high-k bands are not
          dominated by the energetic large scales),
        - applies a high-wavenumber weight w(k) proportional to |k| (explicitly
          punishes high-k underestimation),
        - uses a RAMPED lambda schedule rather than a large fixed weight, so the
          network learns coarse flow structure first and the spectral objective
          is introduced progressively:
              epochs  1-15: lambda = 0.05
              epochs 16-35: lambda = 0.10
              epochs 36-50: lambda = 0.20

  3. DIVERGENCE (continuity) PHYSICS LOSS — for incompressible flow div(u)=0.
     The single highest-value physics term in the turbulence-SR literature
     (Bode et al., PIESRGAN). Penalizes ||div(u_pred)||, computed with periodic
     finite differences in the wall-parallel (x,z) plane.

  4. GRADIENT PHYSICS LOSS — matches spatial velocity gradients of prediction
     and target (sharpens fine structure; Yousif et al. MS-ESRGAN). Computed
     as Charbonnier on the finite-difference gradients.

  Total objective:
     L = L_charb_weighted
       + lambda_spec(epoch) * L_spectral_2d
       + LAMBDA_DIV   * L_divergence
       + LAMBDA_GRAD  * L_gradient

Everything else — optimizer (Adam), cosine LR schedule, batch size (32),
weight decay (1e-4), early stopping (patience 20), MIN/MAX epochs (10/50),
AMP float16, gradient clipping (1.0) — is IDENTICAL to v2/v3/baselines for a
fair comparison. Supports resume from latest.pt.

Usage:
    python trainer_yasunet_v4.py \
        --cache_dir  cache_512/cache_512 \
        --stats_path norm_stats_train.json \
        --output_dir runs_final \
        --tier       2 \
        --seed       42
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset
from yasunet_v3   import YASUNetV2   # v4 reuses the v3 architecture unchanged

# ── Hyperparameters — identical to v2/v3/baselines for fair comparison ────────
BATCH_SIZE    = 32
NUM_WORKERS   = 16
MAX_EPOCHS    = 50
MIN_EPOCHS    = 10
LR_INIT       = 1e-3
LR_MIN        = 1e-5
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 20

# ── v4 loss settings ──────────────────────────────────────────────────────────
CHARBONNIER_EPS = 1e-3      # standard LapSRN value (unchanged from v3)

# Near-wall weighting: w(yp) = min(CAP, C / sqrt(yp)), normalized to mean 1
NEARWALL_CAP    = 2.0       # ceiling so yp15 cannot starve the mid-field

# Spectral loss: ramped lambda schedule (epoch -> weight)
def lambda_spectral_for_epoch(epoch):
    """Progressive introduction of the spectral objective."""
    if epoch <= 15:
        return 0.10
    elif epoch <= 35:
        return 0.30
    else:
        return 0.60

SPEC_HIGHK_POWER = 1.0      # w(k) ~ |k|^power weighting on the spectral error
SPEC_LOG_EPS     = 1e-8     # floor inside log10 of the power spectrum

# Physics loss weights (small — they regularize, pixel loss leads)
LAMBDA_DIV      = 0.10      # divergence / continuity
LAMBDA_GRAD     = 0.05      # velocity-gradient matching

# y+ band scalars present in the protocol (must match dataset_a100.YP_SCALARS)
PROTOCOL_YP_VALUES = [15.0, 30.0, 50.0, 100.0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Near-wall weight ──────────────────────────────────────────────────────────
def _nearwall_norm_const():
    """C such that E[min(CAP, C/sqrt(yp))] = 1 over the protocol y+ bands."""
    yps = np.array(PROTOCOL_YP_VALUES, dtype=np.float64)
    raw = 1.0 / np.sqrt(yps)
    C   = 1.0 / raw.mean()            # makes uncapped mean exactly 1
    w   = np.minimum(C * raw, NEARWALL_CAP)
    # If the cap bound, renormalize so the capped mean is exactly 1.
    return C / w.mean()

_NEARWALL_C = _nearwall_norm_const()


def nearwall_weight(yp):
    """
    yp: [B] tensor of wall-distance scalars (15/30/50/100).
    Returns [B] per-sample loss weights, smooth in y+, mean ~1 over bands.
    """
    w = _NEARWALL_C / torch.sqrt(yp.float())
    return torch.clamp(w, max=NEARWALL_CAP)


def weighted_charbonnier_loss(pred, target, yp, eps=CHARBONNIER_EPS):
    """
    Charbonnier with a per-sample near-wall weight.
    pred, target: [B, C, H, W]; yp: [B].
    """
    per_elem = torch.sqrt((pred - target) ** 2 + eps ** 2)   # [B,C,H,W]
    per_samp = per_elem.mean(dim=(1, 2, 3))                   # [B]
    w        = nearwall_weight(yp).to(per_samp.device)        # [B]
    return (w * per_samp).mean()


# ── 2-D anisotropic, log, high-k-weighted spectral loss ───────────────────────
def _highk_weight(H, W, device, power=SPEC_HIGHK_POWER):
    """
    Precompute |k| weight on the rfft2 grid, normalized to mean 1 so the
    spectral-loss magnitude stays comparable across resolutions.
    Returns [H, W//2+1].
    """
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=device)
    kx = torch.fft.rfftfreq(W, d=1.0 / W, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K = torch.sqrt(KX ** 2 + KY ** 2)
    w = K ** power
    w = w / (w.mean() + 1e-12)
    return w


def spectral_loss_2d(pred, target, highk_w):
    """
    2-D anisotropic, log-spectrum, high-k-weighted spectral error.
    Operates per channel on the full 2-D power spectrum (no radial averaging),
    so streamwise/spanwise anisotropy is preserved. Differentiable.

    pred, target: [B, C, H, W];  highk_w: [H, W//2+1] precomputed weight.
    """
    B, C, H, W = pred.shape
    # rfft2 over the spatial dims for all channels at once
    Fp = torch.fft.rfft2(pred.float(),   dim=(-2, -1))
    Ft = torch.fft.rfft2(target.float(), dim=(-2, -1))
    Pp = (Fp.real ** 2 + Fp.imag ** 2) / (H * W) ** 2       # power, [B,C,H,W//2+1]
    Pt = (Ft.real ** 2 + Ft.imag ** 2) / (H * W) ** 2
    # log spectrum so low-energy high-k bands are not dominated by large scales
    Lp = torch.log10(Pp + SPEC_LOG_EPS)
    Lt = torch.log10(Pt + SPEC_LOG_EPS)
    diff = torch.abs(Lp - Lt)                               # [B,C,H,W//2+1]
    # high-k weighting (broadcast over batch & channel)
    diff = diff * highk_w.view(1, 1, *highk_w.shape)
    return diff.mean()


# ── Physics losses ─────────────────────────────────────────────────────────────
def _periodic_grad(field, dim):
    """Centered periodic finite difference along a spatial dim. field: [...,H,W]."""
    return (torch.roll(field, shifts=-1, dims=dim)
            - torch.roll(field, shifts=1, dims=dim)) * 0.5


def divergence_loss(pred):
    """
    Continuity: for incompressible flow div(u)=0. Channels are (u,v,w) =
    (streamwise ux, wall-normal vy, spanwise wz) on an x-z plane at fixed y+.
    We penalize the in-plane divergence du/dx + dw/dz (the resolvable part on a
    single wall-parallel plane), using periodic finite differences in x and z.
    pred: [B, 3, H, W] with dims (..., z(H), x(W)).
    """
    u = pred[:, 0]          # streamwise, varies along x (W, dim=-1)
    w = pred[:, 2]          # spanwise,   varies along z (H, dim=-2)
    du_dx = _periodic_grad(u, dim=-1)
    dw_dz = _periodic_grad(w, dim=-2)
    div   = du_dx + dw_dz
    return torch.mean(div ** 2)


def gradient_loss(pred, target, eps=CHARBONNIER_EPS):
    """
    Charbonnier on spatial gradients (x and z) of every channel — sharpens
    fine structure and aligns the gradient field with the DNS target.
    """
    gx_p = _periodic_grad(pred,   dim=-1)
    gx_t = _periodic_grad(target, dim=-1)
    gz_p = _periodic_grad(pred,   dim=-2)
    gz_t = _periodic_grad(target, dim=-2)
    lx = torch.mean(torch.sqrt((gx_p - gx_t) ** 2 + eps ** 2))
    lz = torch.mean(torch.sqrt((gz_p - gz_t) ** 2 + eps ** 2))
    return 0.5 * (lx + lz)


def combined_loss(pred, target, yp, highk_w, lambda_spec):
    lc = weighted_charbonnier_loss(pred, target, yp)
    ls = spectral_loss_2d(pred, target, highk_w)
    ld = divergence_loss(pred)
    lg = gradient_loss(pred, target)
    total = lc + lambda_spec * ls + LAMBDA_DIV * ld + LAMBDA_GRAD * lg
    return total, lc, ls, ld, lg


def train_epoch(model, loader, optimizer, scaler, device, highk_w, lambda_spec):
    model.train()
    total = 0.0; n = 0
    for step, batch in enumerate(loader):
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        yp     = batch["yp"].to(device,     non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp, yp)
            loss, _, _, _, _ = combined_loss(pred, target, yp, highk_w,
                                             lambda_spec)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item(); n += 1
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{len(loader)}  avg_loss={total/n:.4f}",
                  flush=True)
    return total / n


@torch.no_grad()
def val_epoch(model, loader, device, highk_w, lambda_spec):
    """
    Validation reported as the same combined objective used in training (with
    the current epoch's spectral lambda) so early stopping is consistent with
    what we optimize.
    """
    model.eval()
    total = 0.0; n = 0
    for batch in loader:
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        yp     = batch["yp"].to(device,     non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp, yp)
            loss, _, _, _, _ = combined_loss(pred, target, yp, highk_w,
                                             lambda_spec)
        total += loss.item(); n += 1
    return total / n


def train(cache_dir, stats_path, output_dir, tier, seed):
    set_seed(seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device}  GPU: {gpu_name}")

    print("Building datasets...")
    train_ds = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="train")
    val_ds   = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    model = YASUNetV2(
        in_channels=3, out_channels=3,
        base_ch=32, embed_dim=16, hidden_dim=128,
        modes_h=10, modes_w=10, n_refine=6,
    ).to(device)
    n_params = model.count_parameters()
    print(f"YASU-Net v4  params: {n_params:,}")
    print(f"  Near-wall weights (yp->w): "
          + ", ".join(f"{int(y)}:{float(nearwall_weight(torch.tensor([y]))):.3f}"
                      for y in PROTOCOL_YP_VALUES))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Precompute the high-k spectral weight once (fields are 256x256 here).
    # Inferred from the first batch to be safe about spatial dims.
    sample = next(iter(val_loader))
    _, _, H, W = sample["target"].shape
    highk_w = _highk_weight(H, W, device)
    print(f"  Spectral loss: 2D log high-k (|k|^{SPEC_HIGHK_POWER}), "
          f"grid {H}x{W}, ramped lambda 0.05->0.10->0.20")

    run_name = f"yasunet_v4b_tier{tier}_seed{seed}"
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "yasunet_v4b", "tier": tier, "seed": seed,
        "base_ch": 32, "modes": 10, "n_refine": 6, "n_params": n_params,
        "batch_size": BATCH_SIZE, "lr_init": LR_INIT,
        "loss": "nearwall_charbonnier+2d_log_highk_spectral+divergence+gradient",
        "charbonnier_eps": CHARBONNIER_EPS,
        "nearwall": "min(2.0, C/sqrt(yp)), mean-1 normalized",
        "lambda_spectral_schedule": {"1-15": 0.05, "16-35": 0.10, "36-50": 0.20},
        "spec_highk_power": SPEC_HIGHK_POWER,
        "lambda_div": LAMBDA_DIV, "lambda_grad": LAMBDA_GRAD,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Resume support
    start_epoch    = 1
    best_val       = float("inf")
    patience_count = 0
    history        = []
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        hist_path = run_dir / "history.json"
        if hist_path.exists():
            history  = json.load(open(hist_path))
            best_val = min(x["val_loss"] for x in history)
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val={best_val:.6f}")

    print(f"\nStarting training — {run_name}  (epochs {start_epoch}->{MAX_EPOCHS})")
    print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'lam_s':>6} {'LR':>12} {'Time':>8}")
    print("-" * 66)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        lambda_spec = lambda_spectral_for_epoch(epoch)
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device,
                                 highk_w, lambda_spec)
        val_loss   = val_epoch(model, val_loader, device, highk_w, lambda_spec)
        scheduler.step()
        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(f"  {epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} "
              f"{lambda_spec:>6.2f} {lr:>12.2e} {elapsed:>6.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr, "lambda_spectral": lambda_spec,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch, "val_loss": val_loss,
                "model_state": model.state_dict(), "config": config,
            }, run_dir / "best.pt")
            print(f"    New best: {val_loss:.6f}")
            patience_count = 0
        else:
            patience_count += 1

        torch.save({
            "epoch": epoch, "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        }, run_dir / "latest.pt")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if epoch >= MIN_EPOCHS and patience_count >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved: {run_dir}")
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",  required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tier",  type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    train(
        cache_dir   = args.cache_dir,
        stats_path  = args.stats_path,
        output_dir  = args.output_dir,
        tier        = args.tier,
        seed        = args.seed,
    )
