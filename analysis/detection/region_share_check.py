# region_share_check.py - read-only. Does "the region covers at most 0.16% of the plane" (Section III-H) hold beyond Tier 1?
# Lists every records*.npz under the scratchpad and, where a region_px column exists, the spike-region share of the plane.
import glob
import os
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
N = 512 * 512
files = sorted(set(glob.glob(os.path.join(SP, "**", "records*.npz"), recursive=True)))
print("files:", [os.path.relpath(f, SP) for f in files])
for f in files:
    R = np.load(f, allow_pickle=True)
    keys = [k for k in R.files if k != "cols"]
    cols = [str(c) for c in R["cols"]] if "cols" in R.files else None
    print("\n==", os.path.relpath(f, SP), "| keys:", len(keys), "| cols:", cols)
    if cols is None or "region_px" not in cols:
        print("   keys:", keys[:60])
        continue
    ci = {c: i for i, c in enumerate(cols)}
    for k in keys:
        a = R[k]
        if a.ndim != 2 or a.shape[1] != len(cols):
            print(f"   {k:<30} shape {a.shape}")
            continue
        a = a.astype(float)
        share = 100.0 * a[:, ci["region_px"]] / N
        flagged = int((share > 0).sum())
        line = f"   {k:<30} n={len(a):5d} with-region={flagged:5d}  region% mean {share.mean():.4f} max {share.max():.4f}"
        if "plane" in ci:
            by = []
            for p in np.unique(a[:, ci["plane"]]):
                s = share[a[:, ci["plane"]] == p]
                by.append(f"p{int(p)}:{s.mean():.3f}/{s.max():.3f}")
            line += "  mean/max by plane " + " ".join(by)
        print(line)
