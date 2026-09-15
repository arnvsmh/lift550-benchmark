"""
trainer_yasunet_v3.py
=====================
Trainer for YASU-Net v3. Same protocol as v2 for fair comparison, with two
loss-function upgrades that target reconstruction error (NRMSE) and spectral
fidelity directly:

  1. Charbonnier loss replaces MSE.
     Charbonnier = sqrt((pred - target)^2 + eps^2) is a smooth, differentiable
     L1 variant. The SR literature consistently finds L1/Charbonnier produces
     lower reconstruction error than MSE *even when measured on MSE-derived
     metrics* (PSNR/NRMSE), because MSE over-penalizes large errors and
     over-smooths fine structure. (Zhao et al. 2017, "Is L2 a Good Loss
     Function for Image Processing"; LapSRN; EDSR.)

  2. Spectral loss term aligns the radially-averaged energy spectrum of the
     prediction with the target. Training objective (pixel-only MSE) was
     previously misaligned with evaluation (which includes spectral error).
     L = L_charbonnier + lambda_spec * L_spectral.

Everything else — optimizer (Adam), cosine LR schedule, batch size (32),
early stopping (patience 20), max 50 epochs, AMP — is identical to v2/baselines
for a fair comparison. Supports resume from latest.pt.

Usage:
    python trainer_yasunet_v3.py \
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
from yasunet_v3   import YASUNetV2   # v3 architecture (modes=10, n_refine=6)

# ── Hyperparameters — identical to v2/baselines for fair comparison ───────────
BATCH_SIZE    = 32
NUM_WORKERS   = 16
MAX_EPOCHS    = 50
MIN_EPOCHS    = 10
LR_INIT       = 1e-3
LR_MIN        = 1e-5
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 20

# ── v3 loss settings ──────────────────────────────────────────────────────────
CHARBONNIER_EPS = 1e-3      # standard LapSRN value
LAMBDA_SPECTRAL = 0.05      # small — refine spectrum without dominating pixel loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def charbonnier_loss(pred, target, eps=CHARBONNIER_EPS):
    """Smooth L1 (Charbonnier). Mean over all elements."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def radial_spectrum(field):
    """
    Radially-averaged power spectrum of a batch of single-channel fields.
    field: [B, H, W] (float32)
    Returns: [B, k_max+1] energy per radial wavenumber bin.
    Differentiable — uses torch.fft and a precomputed bin mask via bincount.
    """
    B, H, W = field.shape
    fft = torch.fft.rfft2(field.float())                 # [B, H, W//2+1]
    power = (fft.real ** 2 + fft.imag ** 2) / (H * W) ** 2

    # wavenumber grid for rfft layout
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=field.device)
    kx = torch.fft.rfftfreq(W, d=1.0 / W, device=field.device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K = torch.sqrt(KX ** 2 + KY ** 2)
    k_int = K.round().long()                              # [H, W//2+1]
    k_max = int(min(H, W) // 2)
    k_int = k_int.clamp(max=k_max)

    flat_k = k_int.reshape(-1)                            # [H*(W//2+1)]
    E = torch.zeros(B, k_max + 1, device=field.device, dtype=power.dtype)
    p_flat = power.reshape(B, -1)
    # scatter-add energies into radial bins, per batch element
    E.index_add_(1, flat_k, p_flat)
    return E


def spectral_loss(pred, target):
    """
    Normalized L1 difference between radially-averaged energy spectra,
    averaged over channels. Differentiable. Operates on normalized fields
    (same scale as the pixel loss).
    pred, target: [B, C, H, W]
    """
    B, C, H, W = pred.shape
    total = 0.0
    for c in range(C):
        Ep = radial_spectrum(pred[:, c])     # [B, k_max+1]
        Et = radial_spectrum(target[:, c])
        denom = Et.sum(dim=1, keepdim=True) + 1e-8
        total = total + torch.mean(torch.abs(Ep - Et) / denom)
    return total / C


def combined_loss(pred, target):
    lc = charbonnier_loss(pred, target)
    ls = spectral_loss(pred, target)
    return lc + LAMBDA_SPECTRAL * ls, lc, ls


def train_epoch(model, loader, optimizer, scaler, device):
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
            loss, _, _ = combined_loss(pred, target)

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
def val_epoch(model, loader, device):
    """
    Validation reported as Charbonnier-equivalent for early stopping.
    We track the combined loss (same objective as training) so early
    stopping is consistent with what we optimize.
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
            loss, _, _ = combined_loss(pred, target)
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

    model    = YASUNetV2(
        in_channels=3, out_channels=3,
        base_ch=32, embed_dim=16, hidden_dim=128,
        modes_h=10, modes_w=10, n_refine=6,
    ).to(device)
    n_params = model.count_parameters()
    print(f"YASU-Net v3  params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    run_name = f"yasunet_v3_tier{tier}_seed{seed}"
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "yasunet_v3", "tier": tier, "seed": seed,
        "base_ch": 32, "modes": 10, "n_refine": 6, "n_params": n_params,
        "batch_size": BATCH_SIZE, "lr_init": LR_INIT,
        "loss": "charbonnier+spectral", "lambda_spectral": LAMBDA_SPECTRAL,
        "charbonnier_eps": CHARBONNIER_EPS,
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
        epoch_num = ckpt["epoch"]
        print(f"  Resumed from epoch {epoch_num}  best_val={best_val:.6f}")

    print(f"\nStarting training — {run_name}  (epochs {start_epoch}->{MAX_EPOCHS})")
    print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'LR':>12} {'Time':>8}")
    print("-" * 58)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(f"  {epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} "
              f"{lr:>12.2e} {elapsed:>6.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr,
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
