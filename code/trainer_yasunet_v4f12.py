"""
trainer_yasunet_v4c.py — root-cause-corrected v4.
Same architecture as v3/v4 (yasunet_v3.py, modes=10, n_refine=6).
Changes vs v4, each evidence-driven:
  1. Near-wall weighting REMOVED (plain Charbonnier). Measured yp15/yp100 error
     ratio was identical with and without weighting (2.13-2.15): near-wall error
     is information-limited. Weighting only inflated loss display + variance.
  2. Spectral loss = differentiable replica of the LOCKED eval metric
     (radially-binned energy spectra, sum|dE|/sumE, exact metrics.py binning).
     Verified match to 1e-8. Pressure lands mid-k where phase info exists ->
     lowest NRMSE tax per spectral point.
  3. Validation always scored at FINAL lambda; early stopping disabled.
Loss: L = Charb + lambda(ep)*L_spec_eval + 0.02*L_div + 0.05*L_grad
Ramp: 1-15: 0.05 | 16-35: 0.15 | 36-50: 0.30 (val fixed at 0.30)
"""
import sys, json, time, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset
from yasunet_v3   import YASUNetV2

BATCH_SIZE    = 32
NUM_WORKERS   = 16
MAX_EPOCHS    = 50
LR_INIT       = 1e-3
LR_MIN        = 1e-5
WEIGHT_DECAY  = 1e-4
CHARBONNIER_EPS = 1e-3

def lambda_spectral_for_epoch(epoch):
    if epoch <= 15:
        return 0.05
    else:
        return 0.12   # threshold ablation: probing inside the (0.10, 0.15) gap

LAMBDA_DIV  = 0.02
LAMBDA_GRAD = 0.05

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def charbonnier_loss(pred, target, eps=CHARBONNIER_EPS):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

class RadialSpectralLoss(nn.Module):
    """Differentiable replica of the locked eval metric (metrics.py):
    full fft2 power, ring bins round(K) for k=0..min(H,W)//2 (cells beyond
    excluded), error = sum|E_pred-E_target| / sum(E_target), mean over B,C."""
    def __init__(self, H, W, device):
        super().__init__()
        ky = torch.fft.fftfreq(H, d=1.0 / H, device=device)
        kx = torch.fft.fftfreq(W, d=1.0 / W, device=device)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K = torch.sqrt(KX ** 2 + KY ** 2)
        self.k_max = int(min(H, W) // 2)
        ring = torch.round(K).long()
        ring = torch.where(ring <= self.k_max, ring,
                           torch.full_like(ring, self.k_max + 1))
        self.register_buffer("ring_idx", ring.flatten())
        self.n_bins = self.k_max + 2
        self.H, self.W = H, W

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        Fp = torch.fft.fft2(pred.float(),   dim=(-2, -1))
        Ft = torch.fft.fft2(target.float(), dim=(-2, -1))
        Pp = (Fp.real ** 2 + Fp.imag ** 2) / float(H * W) ** 2
        Pt = (Ft.real ** 2 + Ft.imag ** 2) / float(H * W) ** 2
        Pp = Pp.reshape(B * C, H * W)
        Pt = Pt.reshape(B * C, H * W)
        Ep = Pp.new_zeros(B * C, self.n_bins)
        Et = Pt.new_zeros(B * C, self.n_bins)
        Ep.index_add_(1, self.ring_idx, Pp)
        Et.index_add_(1, self.ring_idx, Pt)
        Ep = Ep[:, : self.k_max + 1]
        Et = Et[:, : self.k_max + 1]
        err = torch.sum(torch.abs(Ep - Et), dim=1) / (torch.sum(Et, dim=1) + 1e-12)
        return err.mean()

def _periodic_grad(field, dim):
    return (torch.roll(field, shifts=-1, dims=dim)
            - torch.roll(field, shifts=1, dims=dim)) * 0.5

def divergence_loss(pred):
    u = pred[:, 0]; w = pred[:, 2]
    du_dx = _periodic_grad(u, dim=-1)
    dw_dz = _periodic_grad(w, dim=-2)
    return torch.mean((du_dx + dw_dz) ** 2)

def gradient_loss(pred, target, eps=CHARBONNIER_EPS):
    gx_p = _periodic_grad(pred,   dim=-1)
    gx_t = _periodic_grad(target, dim=-1)
    gz_p = _periodic_grad(pred,   dim=-2)
    gz_t = _periodic_grad(target, dim=-2)
    lx = torch.mean(torch.sqrt((gx_p - gx_t) ** 2 + eps ** 2))
    lz = torch.mean(torch.sqrt((gz_p - gz_t) ** 2 + eps ** 2))
    return 0.5 * (lx + lz)

def combined_loss(pred, target, spec_loss_fn, lambda_spec):
    lc = charbonnier_loss(pred, target)
    ls = spec_loss_fn(pred, target)
    ld = divergence_loss(pred)
    lg = gradient_loss(pred, target)
    total = lc + lambda_spec * ls + LAMBDA_DIV * ld + LAMBDA_GRAD * lg
    return total, lc, ls, ld, lg

def train_epoch(model, loader, optimizer, scaler, device, spec_loss_fn, lambda_spec):
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
            loss, _, _, _, _ = combined_loss(pred, target, spec_loss_fn, lambda_spec)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item(); n += 1
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{len(loader)}  avg_loss={total/n:.4f}", flush=True)
    return total / n

@torch.no_grad()
def val_epoch(model, loader, device, spec_loss_fn, lambda_spec_final):
    model.eval()
    total = 0.0; tot_c = 0.0; tot_s = 0.0; n = 0
    for batch in loader:
        inp    = batch["input"].to(device,  non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        yp     = batch["yp"].to(device,     non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = model(inp, yp)
            loss, lc, ls, _, _ = combined_loss(pred, target, spec_loss_fn, lambda_spec_final)
        total += loss.item(); tot_c += lc.item(); tot_s += ls.item(); n += 1
    return total / n, tot_c / n, tot_s / n

def train(cache_dir, stats_path, output_dir, tier, seed):
    set_seed(seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device}  GPU: {gpu_name}")

    print("Building datasets...")
    train_ds = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="train")
    val_ds   = LIFT550Dataset(cache_dir, stats_path, tier=tier, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16,
                      hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
    n_params = model.count_parameters()
    print(f"YASU-Net v4d  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    sample = next(iter(val_loader))
    _, _, H, W = sample["target"].shape
    spec_loss_fn = RadialSpectralLoss(H, W, device).to(device)
    lam_final    = lambda_spectral_for_epoch(MAX_EPOCHS)
    print(f"  Spectral loss: EVAL-ALIGNED radial (k_max={spec_loss_fn.k_max}), "
          f"ramp 0.05->0.10 (v4d), validation fixed at lambda={lam_final}")
    print(f"  Charbonnier: PLAIN (near-wall weighting removed)")

    run_name = f"yasunet_v4f12_tier{tier}_seed{seed}"
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "yasunet_v4f12", "tier": tier, "seed": seed,
        "base_ch": 32, "modes": 10, "n_refine": 6, "n_params": n_params,
        "batch_size": BATCH_SIZE, "lr_init": LR_INIT,
        "loss": "plain_charbonnier + eval_aligned_radial_spectral + div + grad",
        "charbonnier_eps": CHARBONNIER_EPS,
        "lambda_spectral_schedule": {"1-15": 0.05, "16-50": 0.12},
        "validation_lambda": lam_final,
        "lambda_div": LAMBDA_DIV, "lambda_grad": LAMBDA_GRAD,
        "early_stopping": "disabled (full 50-epoch budget)",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    start_epoch = 1
    best_val    = float("inf")
    best_charb  = float("inf")
    history     = []
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
            history    = json.load(open(hist_path))
            best_val   = min(x["val_loss"] for x in history)
            charbs     = [x["val_charb"] for x in history if "val_charb" in x]
            best_charb = min(charbs) if charbs else float("inf")
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val={best_val:.6f}")

    print(f"\nStarting training — {run_name}  (epochs {start_epoch}->{MAX_EPOCHS})")
    print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'ValCharb':>10} "
          f"{'ValSpec':>9} {'lam_s':>6} {'LR':>12} {'Time':>8}")
    print("-" * 86)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        lambda_spec = lambda_spectral_for_epoch(epoch)
        t0          = time.time()
        train_loss  = train_epoch(model, train_loader, optimizer, scaler,
                                  device, spec_loss_fn, lambda_spec)
        val_loss, val_charb, val_spec = val_epoch(model, val_loader, device, spec_loss_fn, lam_final)
        scheduler.step()
        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print(f"  {epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} "
              f"{val_charb:>10.6f} {val_spec:>9.5f} "
              f"{lambda_spec:>6.2f} {lr:>12.2e} {elapsed:>6.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_charb": val_charb,
                        "val_spec": val_spec, "lr": lr,
                        "lambda_spectral": lambda_spec})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "val_loss": val_loss,
                        "val_charb": val_charb, "val_spec": val_spec,
                        "model_state": model.state_dict(), "config": config},
                       run_dir / "best.pt")
            print(f"    New best (combined): {val_loss:.6f}")

        # NRMSE-guard checkpoint (diagnostic/fallback only; selection rule
        # must be declared before test eval, based on val components only).
        if val_charb < best_charb:
            best_charb = val_charb
            torch.save({"epoch": epoch, "val_loss": val_loss,
                        "val_charb": val_charb, "val_spec": val_spec,
                        "model_state": model.state_dict(), "config": config},
                       run_dir / "best_charb.pt")
            print(f"    New best (charb):    {val_charb:.6f}")

        torch.save({"epoch": epoch, "val_loss": val_loss,
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config}, run_dir / "latest.pt")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val loss (combined): {best_val:.6f}")
    print(f"Best val charb (NRMSE proxy): {best_charb:.6f}")
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
    train(cache_dir=args.cache_dir, stats_path=args.stats_path,
          output_dir=args.output_dir, tier=args.tier, seed=args.seed)
