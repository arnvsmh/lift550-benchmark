"""
coh_lib.py — shared pipeline for the multi-sample spectral-coherence run.

Read-only with respect to everything under E:/lift550_paper except this
directory. Rebuilds Tier-1 degraded inputs from the raw DNS (the Tier-1
validation cache is gone) and reproduces the original cache's numerics,
including the float16 quantisation of normalised planes that the cache used.

Validated against figure_data/arrays_ALL_tier1.npz (snapshot: train seed 0,
snap 708 = first validation snapshot, y+ = 15).
"""

import sys, json, math
from pathlib import Path

import numpy as np
import torch

CODE = r"E:/lift550_paper/code"
sys.path.insert(0, CODE)

from reader import build_file_index, get_complete_seeds, read_snapshot   # noqa: E402
from degradation import degrade_t1                                        # noqa: E402

TRAIN_DIR   = r"D:/Research/Misc/data/train"
STATS_PATH  = r"E:/lift550_paper/code/norm_stats_train.json"
RUNS        = r"E:/lift550_paper/runs_final"
TRAIN_FRAC  = 0.85          # from normalization.py; verified to reproduce
                            # norm_stats_train.json's 17702 train snapshots
COMPONENTS  = ["u", "v", "w"]
VAR_OF      = {"u": "uxz", "v": "vxz", "w": "wxz"}

# ── the seven Tier-1 seed-42 models ───────────────────────────────────────────
# (label, run directory, lambda_spectral, key in arrays_ALL_tier1.npz)
MODELS = [
    ("Kim",   "kim_tier1_seed42",           None, "field_Kim"),
    ("v3",    "yasunet_v3_tier1_seed42",    0.05, "field_v3"),
    ("v4e",   "yasunet_v4e_tier1_seed42",   0.07, "field_v4e"),
    ("v4d",   "yasunet_v4d_tier1_seed42",   0.10, "field_v4d"),
    ("v4f12", "yasunet_v4f12_tier1_seed42", 0.12, "field_f12"),
    ("v4clo", "yasunet_v4clo_tier1_seed42", 0.15, "field_basinLo"),
    ("v4c",   "yasunet_v4c_tier1_seed42",   0.30, "field_basin"),
]

# ── radial band masks, identical binning to metrics.py energy_spectrum ────────
# metrics.py bins with (K >= ki-0.5) & (K < ki+0.5) on K = sqrt(KX^2+KY^2)
# where kx = fftfreq(W, d=1/W) (integer cycles per domain). A band [a,b]
# is therefore the union of those rings: (K >= a-0.5) & (K < b+0.5).
H = W = 512
_kx = np.fft.fftfreq(W, d=1.0 / W)
_ky = np.fft.fftfreq(H, d=1.0 / H)
_KX, _KY = np.meshgrid(_kx, _ky)
K_RADIAL = np.sqrt(_KX ** 2 + _KY ** 2)

BANDS = [("33-64", 33, 64), ("65-128", 65, 128), ("129-256", 129, 256)]
BAND_MASKS = {
    name: ((K_RADIAL >= lo - 0.5) & (K_RADIAL < hi + 0.5))
    for name, lo, hi in BANDS
}


def load_stats():
    with open(STATS_PATH) as f:
        return json.load(f)


# ── evaluation-sample enumeration ─────────────────────────────────────────────
def val_snapshot_list(yp="yp15"):
    """
    Reconstruct the Tier-1 validation split.

    The cache's index_val.json is gone. normalization.py defines the train
    range as range(0, floor(0.85 * n)) per file; validation is the remainder.
    Verified: sum of floor(0.85*n) over the 20 complete seeds == 17702, the
    n_snapshots recorded in norm_stats_train.json, for all of u, v and w.

    Returns list of (seed, snap) in the cache's own order (seed-major).
    """
    idx = build_file_index(TRAIN_DIR)
    seeds = get_complete_seeds(idx)
    out = []
    for s in seeds:
        n = idx["uxz"][yp][s]["n_snapshots"]
        cut = math.floor(TRAIN_FRAC * n)
        for sn in range(cut, n):
            out.append((s, sn))
    return out, idx


def _f16(a):
    return a.astype(np.float16).astype(np.float32)


def load_sample(idx, seed, snap, yp, stats, quantize_f16=True):
    """
    Build one evaluation sample from the raw DNS.

    Returns (inp_norm[3,H,W] float32, dns_phys[3,H,W] float64).

    quantize_f16=True reproduces the original Tier-1 cache's numerics, which
    were recovered by matching arrays_ALL_tier1.npz (seed 0 / snap 708):

      DNS target      raw -> normalise(float32) -> float16
                      reproduced to 4.5e-08 (float32 round-trip; exact)
      degraded input  degrade_t1(float32) -> float16 PHYSICAL
                      -> normalise -> float16
                      reproduced to rms 1.6e-05 normalised (0.0016% of sigma);
                      0.068% of pixels differ by one float16 ulp

    The intermediate float16 in physical units is what the preprocessing's
    degraded cache stored; dropping it raises the input error 26x (rms
    4.2e-04, 45% of pixels), so it is part of the original pipeline.
    """
    raw = np.stack([
        read_snapshot(idx[VAR_OF[c]][yp][seed]["path"], snap) for c in COMPONENTS
    ])                                                    # [3,H,W] float64

    # degradation happens BEFORE normalisation (degradation.py docstring)
    deg = degrade_t1(torch.from_numpy(raw).float()).numpy()

    mean = np.array([stats[f"{VAR_OF[c]}_{yp}"]["mean"] for c in COMPONENTS], np.float32)
    std  = np.array([stats[f"{VAR_OF[c]}_{yp}"]["std"]  for c in COMPONENTS], np.float32)
    m = mean[:, None, None]; s = std[:, None, None]

    if quantize_f16:
        inp_norm = _f16((_f16(deg.astype(np.float32)) - m) / s)
        dns_norm = _f16((raw.astype(np.float32) - m) / s)
    else:
        inp_norm = (deg.astype(np.float32) - m) / s
        dns_norm = (raw.astype(np.float32) - m) / s

    # the published analysis scored physical (denormalised) fields
    dns_phys = (dns_norm.astype(np.float64) * s.astype(np.float64)
                + m.astype(np.float64))
    return inp_norm, dns_phys


# ── models ────────────────────────────────────────────────────────────────────
def build_model(run_dir, device):
    """Instantiate and load one checkpoint. Mirrors evaluate_cache.py."""
    ckpt = torch.load(Path(RUNS) / run_dir / "best.pt",
                      map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    name = cfg["model"]
    if name == "kim":
        from baselines_pkg import build_baseline
        model = build_baseline("kim", in_channels=3, out_channels=3).to(device)
        takes_yp = False
    else:
        from yasunet_v3 import YASUNetV2
        model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32,
                          embed_dim=16, hidden_dim=128,
                          modes_h=10, modes_w=10, n_refine=6).to(device)
        takes_yp = True
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, takes_yp, cfg, ckpt.get("epoch")


@torch.no_grad()
def predict(model, takes_yp, inp_norm, yp_value, stats, yp, device, use_amp=True):
    """
    Run one sample through a model and return the prediction in physical units.

    autocast(float16) matches evaluate_cache.py, the path that produced every
    other number in the paper.
    """
    x = torch.from_numpy(inp_norm).unsqueeze(0).to(device)
    ypt = torch.tensor([yp_value], dtype=torch.float32, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(use_amp and device.type == "cuda")):
        out = model(x, ypt) if takes_yp else model(x)
    out = out.float().squeeze(0).cpu().numpy().astype(np.float64)
    mean = np.array([stats[f"{VAR_OF[c]}_{yp}"]["mean"] for c in COMPONENTS], np.float64)
    std  = np.array([stats[f"{VAR_OF[c]}_{yp}"]["std"]  for c in COMPONENTS], np.float64)
    return out * std[:, None, None] + mean[:, None, None]


# ── the statistics under test ─────────────────────────────────────────────────
def coherence(pred_plane, dns_plane):
    """
    C(B) = |sum_B F_pred conj(F_DNS)| / sqrt(sum_B |F_pred|^2 * sum_B |F_DNS|^2)

    Returns dict band -> C. Scale-invariant in each field separately.
    """
    Fp = np.fft.fft2(pred_plane.astype(np.float64))
    Fd = np.fft.fft2(dns_plane.astype(np.float64))
    cross = Fp * np.conj(Fd)
    pp = (Fp.real ** 2 + Fp.imag ** 2)
    dd = (Fd.real ** 2 + Fd.imag ** 2)
    out = {}
    for name, m in BAND_MASKS.items():
        num = abs(cross[m].sum())
        den = math.sqrt(pp[m].sum() * dd[m].sum())
        out[name] = num / den if den > 0 else float("nan")
    return out


def flag_mask(pred_plane, dns_plane, n_sigma=3.0):
    """
    The paper's amplitude rule: flag pixels more than n_sigma beyond the DNS
    range for that plane and component.

    Verified to reproduce the published single-snapshot flagged counts exactly
    for all seven models and all three components:
      Kim/v3/v4e/v4d (0,0,0)  v4f12 (0,31,0)
      v4clo (46,29,54)        v4c  (70,29,66)
    """
    s = dns_plane.std()
    lo = dns_plane.min() - n_sigma * s
    hi = dns_plane.max() + n_sigma * s
    return (pred_plane > hi) | (pred_plane < lo)


def dilate(mask, width=2):
    """Widen flagged regions by `width` pixels (square structuring element).

    Chebyshev dilation: an isolated pixel becomes (2*width+1)^2. Periodic, to
    match the periodic x-z plane.
    """
    out = mask.copy()
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dy == 0 and dx == 0:
                continue
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out
