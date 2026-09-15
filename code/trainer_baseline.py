"""
trainer_baseline.py
===================
Unified trainer for all 5 LIFT-550 baseline models.

Uses the same hyperparameters and data pipeline as trainer_a100.py to
ensure fair comparison with YASU-Net. Key design decisions:

- Same optimizer (Adam), same LR schedule (cosine annealing)
- Same batch size (32), same early stopping (patience=20)
- Same MSE loss for all baselines (including Yousif GAN-generator and
  Kim CycleGAN-generator — running generator-only with MSE for fair comparison)
- Same data pipeline via dataset_a100.py (pre-normalized cache)
- Mixed precision (float16) for training speed

Usage:
    python trainer_baseline.py \\
        --model      fukami_cnn \\
        --cache_dir  /workspace/lift550/cache/tier1 \\
        --stats_path /workspace/lift550/norm_stats_train.json \\
        --output_dir /workspace/lift550/runs_baselines \\
        --tier       1 \\
        --seed       42

Model names: fukami_cnn | fukami_dscms | guastoni | yousif | kim
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset
from baselines_pkg import build_baseline

# ── Hyperparameters ──────────────────────────────────────────────────────────
# These match trainer_a100.py exactly for fair YASU-Net comparison
BATCH_SIZE   = 32
NUM_WORKERS  = 16
MAX_EPOCHS   = 50
MIN_EPOCHS   = 10
LR_INIT      = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 20   # patience


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    criterion  = nn.MSELoss()

    for step, batch in enumerate(loader):
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp)
            loss = criterion(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{len(loader)}  "
                  f"avg_loss={total_loss/n_batches:.4f}", flush=True)

    return total_loss / n_batches


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    criterion  = nn.MSELoss()

    for batch in loader:
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp)
            loss = criterion(pred, target)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def train(model_name, cache_dir, stats_path, output_dir, tier, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device}  GPU: {gpu_name}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="train")
    val_ds   = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = build_baseline(model_name, in_channels=3, out_channels=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}  params: {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # ── Output directory ──────────────────────────────────────────────────────
    # Sanitize model name for directory (replace / with _)
    safe_name = model_name.replace('/', '_').replace(' ', '_')
    run_name  = f"{safe_name}_tier{tier}_seed{seed}"
    run_dir   = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model":        model_name,
        "tier":         tier,
        "seed":         seed,
        "batch_size":   BATCH_SIZE,
        "lr_init":      LR_INIT,
        "lr_min":       LR_MIN,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs":   MAX_EPOCHS,
        "n_params":     n_params,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val       = float("inf")
    patience_count = 0
    history        = []

    print(f"\nStarting — {run_name}")
    print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'LR':>12} {'Time':>8}")
    print("-" * 58)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(f"  {epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} "
              f"{lr:>12.2e} {elapsed:>6.1f}s")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         lr,
        })

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":       epoch,
                "val_loss":    val_loss,
                "model_state": model.state_dict(),
                "config":      config,
            }, run_dir / "best.pt")
            print(f"    ✓ New best: {val_loss:.6f}")
            patience_count = 0
        else:
            patience_count += 1

        # Save latest (for resume)
        torch.save({
            "epoch":       epoch,
            "val_loss":    val_loss,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "config":      config,
        }, run_dir / "latest.pt")

        # Save history
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epoch >= MIN_EPOCHS and patience_count >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP} epochs)")
            break

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved to: {run_dir}")
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LIFT-550 Baseline Trainer"
    )
    parser.add_argument("--model", required=True,
                        choices=["fukami_cnn", "fukami_dscms",
                                 "guastoni", "yousif", "kim"],
                        help="Baseline model to train")
    parser.add_argument("--cache_dir",  required=True,
                        help="Path to pre-normalized cache for this tier")
    parser.add_argument("--stats_path", required=True,
                        help="Path to norm_stats_train.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--tier",  type=int, required=True, choices=[1, 2, 3],
                        help="Degradation tier")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    train(
        model_name  = args.model,
        cache_dir   = args.cache_dir,
        stats_path  = args.stats_path,
        output_dir  = args.output_dir,
        tier        = args.tier,
        seed        = args.seed,
    )
