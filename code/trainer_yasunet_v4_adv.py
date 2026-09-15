"""
trainer_yasunet_v4_adv.py
=========================
Adversarial variant of YASU-Net v4 ("v4.1"). Same generator architecture
(yasunet_v3.py, modes=10, n_refine=6) and the SAME full v4 content objective
(near-wall weighted Charbonnier + 2-D log high-k spectral + divergence +
gradient), imported directly from trainer_yasunet_v4.py so the generator loss is
identical. On top of that it adds a relativistic-average GAN (RaGAN) term via a
spectral-normalized PatchGAN discriminator, to break the conditional-mean
ceiling that caps deterministic models' spectral/SSIM fidelity.

Design choices (chosen to protect the goal — slightly beat Kim on NRMSE while
dominating spectral error — and to keep adversarial training stable):

  • FINE-TUNING STAGE, NOT FROM SCRATCH. The generator is initialized from a
    fully-trained deterministic v4 checkpoint (best.pt). This is the ESRGAN
    recipe: far more stable than adversarial-from-scratch and keeps NRMSE close
    to the deterministic model (the adversarial term refines high-frequency
    detail rather than relearning the field).

  • SMALL ADVERSARIAL WEIGHT (LAMBDA_ADV) so the perception–distortion tradeoff
    does not blow up NRMSE. The content objective still leads.

  • RELATIVISTIC AVERAGE DISCRIMINATOR (RaGAN) — sharper / more stable than
    vanilla GAN for SR (ESRGAN, Yousif MS-ESRGAN).

  • SPECTRAL NORMALIZATION on every discriminator conv for Lipschitz stability.

  • Discriminator forward in float32 (autocast disabled) to avoid float16
    instabilities.

Fair-comparison generator hyperparameters (Adam, weight decay, batch size, AMP,
grad clipping) match the deterministic trainers. Only adversarial-specific knobs
are new. This is the PROPOSED FULL METHOD (not unified-MSE) — it belongs in the
proposed-method table, compared honestly against the unified-MSE benchmark.

Usage:
    python trainer_yasunet_v4_adv.py \
        --cache_dir  cache_512/cache_512 \
        --stats_path norm_stats_train.json \
        --output_dir runs_final \
        --tier 2 --seed 42 \
        --init_from  runs_final/yasunet_v4_tier2_seed42/best.pt

If --init_from is omitted the trainer auto-locates the matching deterministic v4
best.pt; if none exists it errors (adversarial-from-scratch is not the default).
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset
from yasunet_v3 import YASUNetV2          # generator: identical to v3/v4

# Reuse the EXACT v4 content-loss components so the generator objective matches
from trainer_yasunet_v4 import (
    set_seed,
    nearwall_weight,
    weighted_charbonnier_loss,
    _highk_weight,
    spectral_loss_2d,
    divergence_loss,
    gradient_loss,
    lambda_spectral_for_epoch,
    CHARBONNIER_EPS,
    LAMBDA_DIV,
    LAMBDA_GRAD,
    PROTOCOL_YP_VALUES,
)

# ── Fair-comparison generator hyperparameters (match deterministic trainers) ──
BATCH_SIZE    = 32
NUM_WORKERS   = 16
LR_INIT       = 1e-3        # used only if training generator from scratch
LR_MIN        = 1e-5
WEIGHT_DECAY  = 1e-4

# ── Adversarial fine-tuning settings ──────────────────────────────────────────
ADV_EPOCHS    = 20          # short refinement schedule on top of pretrained v4
G_LR          = 1e-4        # generator LR during adversarial fine-tuning (low)
D_LR          = 1e-4        # discriminator LR
ADAM_BETAS    = (0.9, 0.999)
LAMBDA_ADV    = 5e-3        # small: content loss must keep leading (protect NRMSE)
LAMBDA_SPEC_FT= 0.20        # hold spectral weight at its final (strong) value
D_STEPS       = 1           # discriminator updates per generator update


# ── Spectral-normalized PatchGAN discriminator ────────────────────────────────
class PatchDiscriminator(nn.Module):
    """
    70x70-style PatchGAN with spectral normalization on every conv. Operates on
    the 3-channel (u,v,w) field. Outputs a patch map of real-vs-fake logits;
    RaGAN consumes the mean logit.
    """
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        sn = nn.utils.spectral_norm

        def block(cin, cout, stride):
            return nn.Sequential(
                sn(nn.Conv2d(cin, cout, 4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            sn(nn.Conv2d(in_channels, base, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            block(base,     base * 2, 2),
            block(base * 2, base * 4, 2),
            block(base * 4, base * 8, 1),
            sn(nn.Conv2d(base * 8, 1, 4, stride=1, padding=1)),  # patch logits
        )

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ── Relativistic-average GAN losses ───────────────────────────────────────────
def _bce(logits, target_is_real):
    target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def d_ragan_loss(d_real, d_fake):
    """
    Relativistic average discriminator loss:
      real should be 'more real than average fake'  -> label 1
      fake should be 'less real than average real'  -> label 0
    """
    real_rel = d_real - d_fake.mean()
    fake_rel = d_fake - d_real.mean()
    return 0.5 * (_bce(real_rel, True) + _bce(fake_rel, False))


def g_ragan_loss(d_real, d_fake):
    """Generator's relativistic adversarial loss (symmetric to D)."""
    real_rel = d_real - d_fake.mean()
    fake_rel = d_fake - d_real.mean()
    return 0.5 * (_bce(real_rel, False) + _bce(fake_rel, True))


def generator_content_loss(pred, target, yp, highk_w, lambda_spec):
    """The full v4 content objective (identical to the deterministic trainer)."""
    lc = weighted_charbonnier_loss(pred, target, yp)
    ls = spectral_loss_2d(pred, target, highk_w)
    ld = divergence_loss(pred)
    lg = gradient_loss(pred, target)
    content = lc + lambda_spec * ls + LAMBDA_DIV * ld + LAMBDA_GRAD * lg
    return content, lc, ls, ld, lg


# ── NRMSE on a batch (true monitoring metric, objective-independent) ──────────
@torch.no_grad()
def batch_nrmse(pred, target):
    num = torch.sqrt(((pred - target) ** 2).mean(dim=(1, 2, 3)))
    den = torch.sqrt((target ** 2).mean(dim=(1, 2, 3))) + 1e-12
    return (num / den).mean().item()


def locate_init_checkpoint(init_from, output_dir, tier, seed):
    if init_from:
        p = Path(init_from)
        if not p.exists():
            raise FileNotFoundError(f"--init_from not found: {p}")
        return p
    auto = Path(output_dir) / f"yasunet_v4_tier{tier}_seed{seed}" / "best.pt"
    if auto.exists():
        return auto
    raise FileNotFoundError(
        "No deterministic v4 checkpoint found to fine-tune from. "
        "Train trainer_yasunet_v4.py first, or pass --init_from. "
        "Adversarial-from-scratch is intentionally not the default.")


def train(cache_dir, stats_path, output_dir, tier, seed, init_from):
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

    # Generator — identical architecture, initialized from deterministic v4
    G = YASUNetV2(
        in_channels=3, out_channels=3,
        base_ch=32, embed_dim=16, hidden_dim=128,
        modes_h=10, modes_w=10, n_refine=6,
    ).to(device)
    init_path = locate_init_checkpoint(init_from, output_dir, tier, seed)
    ckpt = torch.load(init_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["model_state"])
    print(f"Generator params: {G.count_parameters():,}")
    print(f"  Initialized from deterministic v4: {init_path} "
          f"(epoch {ckpt.get('epoch','?')}, val {ckpt.get('val_loss', float('nan')):.6f})")

    # Discriminator
    D = PatchDiscriminator(in_channels=3, base=64).to(device)
    print(f"Discriminator params: {D.count_parameters():,}")

    optG = torch.optim.Adam(G.parameters(), lr=G_LR, betas=ADAM_BETAS,
                            weight_decay=WEIGHT_DECAY)
    optD = torch.optim.Adam(D.parameters(), lr=D_LR, betas=ADAM_BETAS)
    schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=ADV_EPOCHS, eta_min=LR_MIN)
    schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=ADV_EPOCHS, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    sample = next(iter(val_loader))
    _, _, H, W = sample["target"].shape
    highk_w = _highk_weight(H, W, device)

    run_name = f"yasunet_v4adv_tier{tier}_seed{seed}"
    run_dir  = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "yasunet_v4_adv", "tier": tier, "seed": seed,
        "init_from": str(init_path),
        "generator": "yasunet_v3 arch (modes=10, n_refine=6)",
        "discriminator": "spectral-norm PatchGAN, base=64",
        "adv_loss": "relativistic average GAN (RaGAN)",
        "lambda_adv": LAMBDA_ADV, "lambda_spec_ft": LAMBDA_SPEC_FT,
        "g_lr": G_LR, "d_lr": D_LR, "adv_epochs": ADV_EPOCHS,
        "content_loss": "nearwall_charbonnier+2d_log_highk_spectral+div+grad",
        "note": "PROPOSED METHOD (not unified-MSE). Fine-tuned from deterministic v4.",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Track the best by NRMSE (the protected metric) so the adversarial term
    # cannot silently trade away pointwise accuracy without us seeing it.
    best_nrmse = float("inf")
    history = []

    print(f"\nAdversarial fine-tuning — {run_name}  ({ADV_EPOCHS} epochs)")
    print(f"  lambda_adv={LAMBDA_ADV}  lambda_spec={LAMBDA_SPEC_FT}  G_lr={G_LR}")
    print(f"  {'Epoch':>5} {'G_total':>9} {'content':>9} {'G_adv':>8} "
          f"{'D_loss':>8} {'val_NRMSE':>10} {'Time':>7}")
    print("-" * 70)

    for epoch in range(1, ADV_EPOCHS + 1):
        G.train(); D.train()
        t0 = time.time()
        run_g = run_c = run_a = run_d = 0.0; n = 0

        for batch in train_loader:
            inp    = batch["input"].to(device,  non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            yp     = batch["yp"].to(device,     non_blocking=True)

            # ---- Generator forward (AMP) ----
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                pred = G(inp, yp)

            # ---- Discriminator update (float32 for stability) ----
            for _ in range(D_STEPS):
                optD.zero_grad(set_to_none=True)
                d_real = D(target.float())
                d_fake = D(pred.detach().float())
                d_loss = d_ragan_loss(d_real, d_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                optD.step()

            # ---- Generator update ----
            optG.zero_grad(set_to_none=True)
            content, lc, ls, ld, lg = generator_content_loss(
                pred, target, yp, highk_w, LAMBDA_SPEC_FT)
            d_real = D(target.float())
            d_fake = D(pred.float())
            g_adv  = g_ragan_loss(d_real, d_fake)
            g_total = content + LAMBDA_ADV * g_adv

            scaler.scale(g_total).backward()
            scaler.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler.step(optG)
            scaler.update()

            run_g += g_total.item(); run_c += content.item()
            run_a += g_adv.item();   run_d += d_loss.item(); n += 1

        schG.step(); schD.step()

        # ---- Validation: monitor NRMSE (objective-independent) ----
        G.eval()
        vnrmse = 0.0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                inp    = batch["input"].to(device,  non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                yp     = batch["yp"].to(device,     non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    pred = G(inp, yp)
                vnrmse += batch_nrmse(pred.float(), target.float()); vn += 1
        vnrmse /= vn
        elapsed = time.time() - t0

        print(f"  {epoch:>5} {run_g/n:>9.5f} {run_c/n:>9.5f} {run_a/n:>8.4f} "
              f"{run_d/n:>8.4f} {vnrmse:>10.5f} {elapsed:>6.1f}s")

        history.append({
            "epoch": epoch, "g_total": run_g/n, "content": run_c/n,
            "g_adv": run_a/n, "d_loss": run_d/n, "val_nrmse": vnrmse,
        })

        # Save best-by-NRMSE generator (this is what we evaluate)
        if vnrmse < best_nrmse:
            best_nrmse = vnrmse
            torch.save({
                "epoch": epoch, "val_nrmse": vnrmse,
                "model_state": G.state_dict(), "config": config,
            }, run_dir / "best.pt")
            print(f"    New best NRMSE: {vnrmse:.5f}")

        # latest (G+D) for resume
        torch.save({
            "epoch": epoch, "val_nrmse": vnrmse,
            "model_state": G.state_dict(),
            "discriminator": D.state_dict(),
            "optG": optG.state_dict(), "optD": optD.state_dict(),
            "schG": schG.state_dict(), "schD": schD.state_dict(),
            "config": config,
        }, run_dir / "latest.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val NRMSE: {best_nrmse:.5f}")
    print(f"Saved: {run_dir}")
    return best_nrmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",  required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tier", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Deterministic v4 best.pt to fine-tune from. "
                             "If omitted, auto-locates the matching v4 run.")
    args = parser.parse_args()

    train(
        cache_dir   = args.cache_dir,
        stats_path  = args.stats_path,
        output_dir  = args.output_dir,
        tier        = args.tier,
        seed        = args.seed,
        init_from   = args.init_from,
    )
