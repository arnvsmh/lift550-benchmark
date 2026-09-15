"""
normalization.py
================
Compute and save z-score normalization statistics from the training set.

Run once before any training:
    python normalization.py --train_dir "D:/Research/misc/data/train" --output norm_stats_train.json
"""

import sys
import json
import argparse
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reader import read_snapshot, build_file_index, get_complete_seeds

PROTOCOL_VARS = ["uxz", "vxz", "wxz"]
PROTOCOL_YP   = ["yp15", "yp30", "yp50", "yp100"]
TRAIN_FRAC    = 0.85


def get_train_snapshot_range(n_snapshots: int) -> range:
    cutoff = math.floor(TRAIN_FRAC * n_snapshots)
    return range(0, cutoff)


def compute_all_stats(train_dir: str, output_path: str):
    train_dir   = Path(train_dir)
    output_path = Path(output_path)

    print(f"Building file index: {train_dir}")
    file_index     = build_file_index(train_dir)
    complete_seeds = get_complete_seeds(file_index)
    print(f"Complete seeds: {complete_seeds}")
    print(f"Total seeds: {len(complete_seeds)}\n")

    stats = {}

    for var in PROTOCOL_VARS:
        for yp in PROTOCOL_YP:
            key = f"{var}_{yp}"
            print(f"Computing {key} ...", flush=True)

            # Two-pass vectorized: accumulate sum and sum-of-squares
            # then derive mean and std. Memory cost = one plane at a time.
            total_sum  = 0.0
            total_sum2 = 0.0
            total_n    = 0
            n_snaps    = 0

            for seed in complete_seeds:
                entry      = file_index[var][yp][seed]
                fpath      = entry["path"]
                n          = entry["n_snapshots"]
                train_idxs = get_train_snapshot_range(n)

                for snap_idx in train_idxs:
                    plane       = read_snapshot(fpath, snap_idx).astype(np.float64)
                    total_sum  += plane.sum()
                    total_sum2 += (plane ** 2).sum()
                    total_n    += plane.size
                    n_snaps    += 1

            mean = total_sum / total_n
            std  = math.sqrt(total_sum2 / total_n - mean ** 2)

            stats[key] = {
                "mean":        round(mean,    8),
                "std":         round(std,     8),
                "n_snapshots": n_snaps,
                "n_pixels":    total_n,
            }

            print(f"  mean={mean:.6f}  std={std:.6f}  "
                  f"n_snapshots={n_snaps}  n_pixels={total_n:,}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {output_path}")
    return stats


def load_stats(stats_path: str) -> dict:
    with open(stats_path) as f:
        return json.load(f)


def normalize(plane: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (plane - mean) / std


def denormalize(plane: np.ndarray, mean: float, std: float) -> np.ndarray:
    return plane * std + mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output", default="norm_stats_train.json")
    args = parser.parse_args()
    compute_all_stats(args.train_dir, args.output)
