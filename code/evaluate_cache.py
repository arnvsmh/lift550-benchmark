"""
evaluate_cache.py
=================
Evaluates trained checkpoints using the preprocessed cache
instead of raw .pl files. Much faster - no need to download 608GB.

Usage:
    python evaluate_cache.py \
        --checkpoint /workspace/lift550/runs_yasunet/yasunet_tier1_seed42/best.pt \
        --cache_dir  /workspace/lift550/cache/tier1 \
        --stats_path /workspace/lift550/norm_stats_train.json \
        --output_dir /workspace/lift550/results \
        --max_samples 5000
"""

import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from normalization import load_stats
from metrics import compute_all_metrics

PROTOCOL_VARS = ["uxz", "vxz", "wxz"]
PROTOCOL_YP   = ["yp15", "yp30", "yp50", "yp100"]
YP_SCALARS    = {"yp15": 15.0, "yp30": 30.0, "yp50": 50.0, "yp100": 100.0}


class CacheEvalDataset(Dataset):

    def __init__(self, cache_dir, stats_path, tier, split="val", max_samples=None):
        self.cache_dir = Path(cache_dir)
        self.stats     = load_stats(stats_path)
        self.tier      = tier
        self.entries   = []

        index_file = self.cache_dir / f"index_{split}.json"
        index      = json.load(open(index_file))
        n_snaps    = index["total_snapshots"]

        for yp in PROTOCOL_YP:
            for i in range(n_snaps):
                self.entries.append({"yp": yp, "snap": i})

        if max_samples:
            step = max(1, len(self.entries) // max_samples)
            self.entries = self.entries[::step][:max_samples]

        self.arrays = {}
        for yp in PROTOCOL_YP:
            for var in PROTOCOL_VARS:
                key      = f"{var}_{yp}"
                deg_path = self.cache_dir / f"{key}_{split}_t{tier}_norm.npy"
                dns_path = self.cache_dir / f"{key}_{split}_dns_norm.npy"
                raw_path = self.cache_dir / f"{key}_{split}.npy"
                self.arrays[f"{key}_deg"] = np.load(str(deg_path), mmap_mode='r')
                self.arrays[f"{key}_dns"] = np.load(str(dns_path), mmap_mode='r')
                self.arrays[f"{key}_raw"] = np.load(str(raw_path), mmap_mode='r')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        yp    = entry["yp"]
        snap  = entry["snap"]

        deg_planes = []
        dns_planes = []
        raw_planes = []

        for var in PROTOCOL_VARS:
            key = f"{var}_{yp}"
            deg_planes.append(self.arrays[f"{key}_deg"][snap].astype(np.float32))
            dns_planes.append(self.arrays[f"{key}_dns"][snap].astype(np.float32))
            raw_planes.append(self.arrays[f"{key}_raw"][snap].astype(np.float32))

        return {
            "input":   torch.from_numpy(np.stack(deg_planes)),
            "target":  torch.from_numpy(np.stack(dns_planes)),
            "dns_raw": torch.from_numpy(np.stack(raw_planes)),
            "yp":      YP_SCALARS[yp],
            "yp_name": yp,
        }


def denormalize(tensor, yp_name, stats):
    out = torch.empty_like(tensor)
    for c, var in enumerate(PROTOCOL_VARS):
        key     = f"{var}_{yp_name}"
        mean    = stats[key]["mean"]
        std     = stats[key]["std"]
        out[c]  = tensor[c] * std + mean
    return out


def evaluate(checkpoint_path, cache_dir, stats_path, output_dir, max_samples=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config     = ckpt["config"]
    tier       = config["tier"]
    seed       = config["seed"]
    model_name = config["model"]
    print(f"Model: {model_name}  Tier: {tier}  Seed: {seed}")
    print(f"Checkpoint epoch: {ckpt['epoch']}  Val loss: {ckpt['val_loss']:.6f}")

    if model_name == "yasunet":
        from yasunet_a100 import YASUNet
        base_ch = config.get("base_ch", 64)
        modes   = config.get("modes", 16)
        model   = YASUNet(
            in_channels=3, out_channels=3,
            base_ch=base_ch, embed_dim=16, hidden_dim=128,
            modes_h=modes, modes_w=modes,
        ).to(device)
    elif model_name in ("yasunet_v2", "yasunet_v3", "yasunet_v3fskip", "yasunet_v4c", "yasunet_v4clo", "yasunet_v4d", "yasunet_v4e", "yasunet_v4f12"):
        if model_name == "yasunet_v2":
            from yasunet_v2 import YASUNetV2
            model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=8, modes_w=8, n_refine=3).to(device)
        elif model_name == "yasunet_v3":
            from yasunet_v3 import YASUNetV2
            model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
        elif model_name == "yasunet_v3fskip":
            from yasunet_v3_fskip import YASUNetV2
            model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
        elif model_name in ("yasunet_v4c", "yasunet_v4clo", "yasunet_v4d", "yasunet_v4e", "yasunet_v4f12"):
            from yasunet_v3 import YASUNetV2
            model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
    elif model_name in ("fukami_cnn", "fukami_dscms", "guastoni", "yousif", "kim"):
        from baselines_pkg import build_baseline
        model = build_baseline(model_name, in_channels=3, out_channels=3).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = load_stats(stats_path)

    print(f"Loading cache: {cache_dir}")
    ds = CacheEvalDataset(
        cache_dir, stats_path, tier,
        split="val", max_samples=max_samples
    )
    print(f"Eval samples: {len(ds)}")

    loader = DataLoader(
        ds, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )

    all_metrics = defaultdict(list)
    yp_metrics  = defaultdict(lambda: defaultdict(list))
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inp  = batch["input"].to(device)
            yp_t = batch["yp"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                if model_name in ("yasunet", "yasunet_v2", "yasunet_v3", "yasunet_v3fskip", "yasunet_v4c", "yasunet_v4clo", "yasunet_v4d", "yasunet_v4e", "yasunet_v4f12"):
                    pred = model(inp, yp_t)
                else:
                    pred = model(inp)

            for i in range(pred.shape[0]):
                yp_name  = batch["yp_name"][i]
                pred_raw = denormalize(pred[i].float().cpu(), yp_name, stats)
                dns_raw  = batch["dns_raw"][i]
                m = compute_all_metrics(
                    pred_raw.numpy(), dns_raw.numpy()
                )
                for k, v in m.items():
                    all_metrics[k].append(v)
                    yp_metrics[yp_name][k].append(v)

            if (batch_idx + 1) % 50 == 0:
                print(f"  {batch_idx+1}/{len(loader)} batches")

    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s")

    results     = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    results_std = {k: float(np.std(v))  for k, v in all_metrics.items()}

    print(f"\n{'='*55}")
    print(f"Results — {model_name} Tier {tier} Seed {seed}")
    print(f"{'='*55}")
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 45)
    for k in ["rmse", "nrmse", "ssim", "spectral_error"]:
        if k in results:
            print(f"  {k:<23} {results[k]:>10.4f} {results_std[k]:>10.4f}")

    print(f"\nPer y+ breakdown:")
    print(f"{'yp':<10} {'NRMSE mean':>12} {'SSIM mean':>12}")
    print("-" * 35)
    for yp in PROTOCOL_YP:
        if yp in yp_metrics:
            n = float(np.mean(yp_metrics[yp].get("nrmse", [0])))
            s = float(np.mean(yp_metrics[yp].get("ssim",  [0])))
            print(f"  {yp:<8} {n:>12.4f} {s:>12.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_name}_tier{tier}_seed{seed}"

    summary = {
        "model":        model_name,
        "tier":         tier,
        "seed":         seed,
        "epoch":        ckpt["epoch"],
        "val_loss":     ckpt["val_loss"],
        "metrics_mean": results,
        "metrics_std":  results_std,
        "per_yp": {
            yp: {k: float(np.mean(v)) for k, v in yp_metrics[yp].items()}
            for yp in PROTOCOL_YP if yp in yp_metrics
        }
    }

    with open(output_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_dir}/{run_name}_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--cache_dir",   required=True)
    parser.add_argument("--stats_path",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        cache_dir       = args.cache_dir,
        stats_path      = args.stats_path,
        output_dir      = args.output_dir,
        max_samples     = args.max_samples,
    )
