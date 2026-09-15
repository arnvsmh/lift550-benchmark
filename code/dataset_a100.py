"""
dataset_a100.py
===============
LIFT-550 dataset for Linux A100 training.
Reads from pre-normalized numpy cache (zero CPU math per sample).
Uses multiprocessing-safe file handles (no Windows mmap issues).
"""

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from normalization import load_stats

PROTOCOL_VARS = ["uxz", "vxz", "wxz"]
PROTOCOL_YP   = ["yp15", "yp30", "yp50", "yp100"]
YP_SCALARS    = {"yp15": 15.0, "yp30": 30.0, "yp50": 50.0, "yp100": 100.0}


class LIFT550Dataset(Dataset):
    """
    Reads pre-normalized degraded inputs and DNS targets from .npy cache.
    __getitem__ does zero arithmetic — just memmap read + float32 cast.
    """

    def __init__(
        self,
        cache_dir:  str,
        stats_path: str,
        tier:       int,
        split:      Literal["train", "val", "test"] = "train",
        yp_filter:  list[str] | None = None,
    ):
        super().__init__()
        cache_dir      = Path(cache_dir)
        self.yp_groups = yp_filter if yp_filter else PROTOCOL_YP
        self.tier      = tier

        # Load index
        index_path = cache_dir / f"index_{split}.json"
        with open(index_path) as f:
            idx_data = json.load(f)
        self.snap_index = idx_data["snap_index"]
        total_snaps     = idx_data["total_snapshots"]

        # Verify normalized cache exists
        self.inp_paths = {}
        self.tgt_paths = {}
        for var in PROTOCOL_VARS:
            for yp in self.yp_groups:
                key = f"{var}_{yp}"
                inp_path = cache_dir / f"{key}_{split}_t{tier}_norm.npy"
                tgt_path = cache_dir / f"{key}_{split}_dns_norm.npy"
                if not inp_path.exists():
                    raise FileNotFoundError(
                        f"Normalized cache not found: {inp_path}\n"
                        f"Run preprocess.py --normalize first."
                    )
                self.inp_paths[key] = str(inp_path)
                self.tgt_paths[key] = str(tgt_path)

        # Open memmaps lazily (set in _open_arrays, called on first access)
        # This is multiprocessing-safe: each worker opens its own file handles
        self._inp_arrays = None
        self._tgt_arrays = None
        self.cache_dir   = str(cache_dir)
        self.split       = split
        self.total_snaps = total_snaps

        # Build flat index
        self._index = [(i, yp)
                       for i in range(total_snaps)
                       for yp in self.yp_groups]

    def _open_arrays(self):
        """Open memory-mapped arrays lazily (called once per worker process)."""
        self._inp_arrays = {
            k: np.load(p, mmap_mode="r") for k, p in self.inp_paths.items()
        }
        self._tgt_arrays = {
            k: np.load(p, mmap_mode="r") for k, p in self.tgt_paths.items()
        }

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        # Lazy open — each worker process opens its own handles
        if self._inp_arrays is None:
            self._open_arrays()

        snap_idx, yp = self._index[idx]
        meta         = self.snap_index[snap_idx]

        inp_planes = []
        tgt_planes = []
        for var in PROTOCOL_VARS:
            key = f"{var}_{yp}"
            inp_planes.append(self._inp_arrays[key][snap_idx].astype(np.float32))
            tgt_planes.append(self._tgt_arrays[key][snap_idx].astype(np.float32))

        return {
            "input":   torch.from_numpy(np.stack(inp_planes, axis=0)),
            "target":  torch.from_numpy(np.stack(tgt_planes, axis=0)),
            "yp":      YP_SCALARS[yp],
            "yp_name": yp,
            "seed":    meta["seed"],
            "snap":    meta["snap"],
        }
