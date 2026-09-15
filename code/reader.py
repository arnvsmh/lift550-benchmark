"""
data/reader.py
==============
Low-level reader for KTH-FlowAI .pl files.

File format (verified from binary inspection):
  - Fortran unformatted sequential, little-endian float64
  - 136-byte header (4 Fortran records)
  - Snapshots 0..N-2: (4+16+4) metadata record + (4+512*512*8+4) data record
  - Snapshot N-1 (last): data record only, no leading metadata record
  - Small files (~1.63 GB): 833 or 834 snapshots
  - Large files (~3.26 GB): 1667 or 1668 snapshots
"""

import os
import struct
import numpy as np
from pathlib import Path
from functools import lru_cache

# ── Format constants ──────────────────────────────────────────────────────────
NX = 512
NZ = 512
NY = 193

FILE_HEADER_BYTES   = 136
META_RECORD_BYTES   = 4 + 16 + 4      # 24
DATA_RECORD_BYTES   = 4 + NX * NZ * 8 + 4  # 2_097_160
FULL_SNAPSHOT_BYTES = META_RECORD_BYTES + DATA_RECORD_BYTES  # 2_097_184
DATA_ONLY_BYTES     = DATA_RECORD_BYTES


def get_snapshot_count(filepath: str | Path) -> int:
    """
    Return number of snapshots in a .pl file.
    Returns 0 for empty files, -1 for unrecognised format.
    """
    size = os.path.getsize(filepath)
    if size == 0:
        return 0
    remainder = size - FILE_HEADER_BYTES - DATA_ONLY_BYTES
    if remainder < 0:
        return -1
    if remainder % FULL_SNAPSHOT_BYTES != 0:
        return -1
    return remainder // FULL_SNAPSHOT_BYTES + 1


def read_snapshot(filepath: str | Path, snapshot_idx: int = 0) -> np.ndarray:
    """
    Read one 512x512 snapshot from a .pl file.

    Returns
    -------
    np.ndarray, shape (NZ, NX) = (512, 512), dtype float64
    """
    filepath = Path(filepath)
    n = get_snapshot_count(filepath)

    if n <= 0:
        raise ValueError(f"Cannot read {filepath.name}: {n} snapshots")
    if not (0 <= snapshot_idx < n):
        raise IndexError(
            f"snapshot_idx {snapshot_idx} out of range [0, {n}) for {filepath.name}"
        )

    # Offset to the start of the data payload (after the 4-byte opening marker)
    if snapshot_idx < n - 1:
        data_offset = (
            FILE_HEADER_BYTES
            + snapshot_idx * FULL_SNAPSHOT_BYTES
            + META_RECORD_BYTES
            + 4  # opening marker of data record
        )
    else:
        # Last snapshot has no preceding metadata record
        data_offset = (
            FILE_HEADER_BYTES
            + (n - 1) * FULL_SNAPSHOT_BYTES
            + 4
        )

    plane_bytes = NX * NZ * 8  # float64

    with open(filepath, "rb") as f:
        f.seek(data_offset)
        raw = f.read(plane_bytes)

    arr = np.frombuffer(raw, dtype="<f8").copy()  # copy: frombuffer is read-only
    arr = arr.reshape(NZ, NX)   # (512, 512)

    # ── Metadata bleed fix ────────────────────────────────────────────────────
    # Positions (511, 510) and (511, 511) contain Fortran record markers
    # misread as float64: Re_tau value and closing-marker zero respectively.
    # Replace both with the spatial mean of the 2x2 patch immediately above-left.
    arr[511, 509] = float(np.mean(arr[509:511, 507:510]))
    arr[511, 510] = float(np.mean(arr[509:511, 508:511]))
    arr[511, 511] = float(np.mean(arr[509:511, 509:511]))

    return arr


def read_header(filepath: str | Path) -> dict:
    """
    Parse the 136-byte file header and return domain metadata.
    """
    with open(filepath, "rb") as f:
        data = f.read(136)

    return {
        "t_plus":  struct.unpack("<d", data[4:12])[0],
        "Lx":      struct.unpack("<d", data[16:24])[0],
        "Lz":      struct.unpack("<d", data[24:32])[0],
        "Re_tau":  struct.unpack("<d", data[32:40])[0],
        "nz":      struct.unpack("<i", data[56:60])[0],
        "ny":      struct.unpack("<i", data[60:64])[0],
        "nx":      struct.unpack("<i", data[64:68])[0],
    }


def build_file_index(directory: str | Path) -> dict:
    """
    Scan a directory and return a nested dict:
        index[var][yp][seed_idx] = {"path": Path, "n_snapshots": int}

    Only includes files with n_snapshots > 0.
    Only includes protocol variables (uxz, vxz, wxz) and
    protocol y+ groups (yp15, yp30, yp50, yp100).

    Parameters
    ----------
    directory : path to train or test folder

    Returns
    -------
    dict with keys: var -> yp -> seed_idx -> {"path", "n_snapshots"}
    """
    PROTOCOL_VARS = {"uxz", "vxz", "wxz"}
    PROTOCOL_YP   = {"yp15", "yp30", "yp50", "yp100"}

    directory = Path(directory)
    index = {}

    for fpath in sorted(directory.glob("*.pl")):
        stem  = fpath.stem              # e.g. "uxz_yp15_7"
        parts = stem.split("_")
        if len(parts) != 3:
            continue

        var, yp, idx_str = parts
        if var not in PROTOCOL_VARS or yp not in PROTOCOL_YP:
            continue

        try:
            idx = int(idx_str)
        except ValueError:
            continue

        n = get_snapshot_count(fpath)
        if n <= 0:
            continue

        index.setdefault(var, {}).setdefault(yp, {})[idx] = {
            "path":        fpath,
            "n_snapshots": n,
        }

    return index


def get_complete_seeds(file_index: dict) -> list[int]:
    """
    Return sorted list of seed indices where u, v, w are ALL present
    at ALL four protocol y+ locations.
    """
    VARS = ["uxz", "vxz", "wxz"]
    YPS  = ["yp15", "yp30", "yp50", "yp100"]

    # gather all candidate seeds
    all_seeds = set()
    for var in VARS:
        for yp in YPS:
            all_seeds.update(file_index.get(var, {}).get(yp, {}).keys())

    complete = []
    for seed in sorted(all_seeds):
        ok = all(
            seed in file_index.get(var, {}).get(yp, {})
            for var in VARS for yp in YPS
        )
        if ok:
            complete.append(seed)

    return complete


# ── Quick sanity check (run this file directly to test) ──────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python reader.py <path_to_train_or_test_dir>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    print(f"Scanning: {directory}")

    idx = build_file_index(directory)
    seeds = get_complete_seeds(idx)
    print(f"Complete seeds: {seeds}")

    if seeds:
        seed0 = seeds[0]
        entry = idx["uxz"]["yp15"][seed0]
        print(f"\nReading uxz_yp15 seed={seed0}: {entry['n_snapshots']} snapshots")

        plane = read_snapshot(entry["path"], snapshot_idx=0)
        print(f"  Shape:  {plane.shape}")
        print(f"  dtype:  {plane.dtype}")
        print(f"  min:    {plane.min():.4f}")
        print(f"  max:    {plane.max():.4f}")
        print(f"  mean:   {plane.mean():.4f}")
        print(f"  std:    {plane.std():.4f}")
        print(f"  any NaN: {np.isnan(plane).any()}")
        print(f"  any Inf: {np.isinf(plane).any()}")

        # read same snapshot for v and w
        for var in ["vxz", "wxz"]:
            e = idx[var]["yp15"][seed0]
            p = read_snapshot(e["path"], snapshot_idx=0)
            print(f"  {var}_yp15 snapshot 0: mean={p.mean():.4f}  std={p.std():.4f}")

        hdr = read_header(entry["path"])
        print(f"\nHeader: {hdr}")
