"""
trainer_yasunet_v2.py
=====================
Trainer for YASU-Net v2. Same protocol as trainer_a100.py for fair comparison.

Usage:
    python trainer_yasunet_v2.py \
        --cache_dir  /workspace/lift550/cache/tier1 \
        --stats_path /workspace/lift550/norm_stats_train.json \
        --output_dir /workspace/lift550/runs_yasunet_v2 \
        --tier       1 \
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset
from yasunet_v2   import YASUNetV2

# ── Hyperparameters — identical to trainer_a100.py for fair comparison ────────
BATCH_SIZE   = 32
NUM_WORKERS  = 16
MAX_EPOCHS   = 100
MIN_EPOCHS   = 10
LR_INIT      = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    n     = 0
    crit  = nn.MSELoss()

    for step, batch in enumerate(loader):
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        yp     = batch["yp"].to(device,     non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp, yp)
            loss = crit(pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        n     += 1

        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{len(loader)}  "
                  f"avg_loss={total/n:.4f}", flush=True)

    return total / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    n     = 0
    crit  = nn.MSELoss()

    for batch in loader:
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        yp     = batch["yp"].to(device,     non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp, yp)
            loss = crit(pred, target)
        total += loss.item()
        n     += 1

    return total / n


def train(cache_dir, stats_path, output_dir, tier, seed):
    set_seed(seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device}  GPU: {gpu_name}")

    # Dataset
    print("Building datasets...")
    train_ds = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="train")
    val_ds   = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True,
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # Model
    model    = YASUNetV2(
        in_channels=3, out_channels=3,
        base_ch=32, embed_dim=16, hidden_dim=128,
        modes_h=8, modes_w=8, n_refine=3,
    ).to(device)
    n_params = model.count_parameters()
    print(f"YASU-Net v2  params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Output
    run_name = f"yasunet_v2_tier{tier}_seed{seed}"
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "yasunet_v2", "tier": tier, "seed": seed,
        "base_ch": 32, "modes": 8, "n_params": n_params,
        "batch_size": BATCH_SIZE, "lr_init": LR_INIT,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val       = float("inf")
    patience_count = 0
    history        = []
    start_epoch    = 1

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

    print(f"\nStarting training — {run_name}")
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
            print(f"    ✓ New best: {val_loss:.6f}")
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
