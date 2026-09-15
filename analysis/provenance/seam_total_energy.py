"""
seam_total_energy.py - how much the cached-array seam changes the total spectral energy of a DNS plane.
Source of the Section V-D sentence "changes the total by less than 1e-5 relative when pooled over samples and by at
most about 2e-4 in a single sample".

Cached plane: the project reader's offset (every non-last snapshot read 24 B late) plus its three-value patch of
row 511, x = 509..511 (reader.py). Clean plane: the correct offset. Total energy E = sum over all wavevectors of
|fft2|^2 (all k, including k = 0). Per sample: |E_cached - E_clean| / E_clean. Pooled: |sum E_cached - sum E_clean| /
sum E_clean over the samples of one plane and component.

Design A reproduces the earlier check: 16 + 16 validation snapshots from files 21 and 23 (the same picks).
Design B widens it: 5 evenly spaced snapshots from the validation segment of each of the 20 files (the file's last
snapshot excluded, since it is read correctly and carries no seam). Read-only. Writes seam_total_energy.json.
"""
import os
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TR = r"D:/Research/misc/data/train"
NX = NZ = 512
HDR, META = 136, 24
DATA = 4 + NX * NZ * 8 + 4
FULL = META + DATA
FILES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 23]


def nsnap(f):
    r = os.path.getsize(f) - HDR - DATA
    return r // FULL + 1


def raw(f, i, buggy):
    n = nsnap(f)
    off = (HDR + i * FULL + META + 4) if (buggy and i < n - 1) else (HDR + i * FULL + 4)
    with open(f, "rb") as fh:
        fh.seek(off)
        b = fh.read(NX * NZ * 8)
    return np.frombuffer(b, "<f8").copy().reshape(NZ, NX)


def patch(a):
    a[511, 509] = float(np.mean(a[509:511, 507:510]))
    a[511, 510] = float(np.mean(a[509:511, 508:511]))
    a[511, 511] = float(np.mean(a[509:511, 509:511]))
    return a


def energy(a):
    return float((np.abs(np.fft.fft2(a)) ** 2).sum())


def run(picks_for):
    res = {}
    for comp in ("u", "v", "w"):
        for yp in (15, 30, 50, 100):
            per, pc, pk = [], 0.0, 0.0
            for fi, si in picks_for(comp, yp):
                f = os.path.join(TR, f"{comp}xz_yp{yp}_{fi}.pl")
                k = energy(raw(f, int(si), False))
                c = energy(patch(raw(f, int(si), True)))
                per.append(abs(c - k) / k)
                pc += c
                pk += k
            res[f"{comp}_yp{yp}"] = {"n": len(per), "per_sample_max": max(per), "per_sample_median": float(np.median(per)),
                                     "pooled": abs(pc - pk) / pk}
    return res


def picks_A(comp, yp):
    return [(21, s) for s in np.linspace(196, 1660, 16).astype(int)] + [(23, s) for s in np.linspace(3, 1660, 16).astype(int)]


def picks_B(comp, yp):
    out = []
    for fi in FILES:
        n = nsnap(os.path.join(TR, f"{comp}xz_yp{yp}_{fi}.pl"))
        lo, hi = math.floor(0.85 * n), n - 2
        out += [(fi, s) for s in np.linspace(lo, hi, 5).astype(int)]
    return out


out = {}
for name, fn in (("A_32_snapshots_files_21_23", picks_A), ("B_5_per_validation_segment_all_20_files", picks_B)):
    r = run(fn)
    out[name] = {"by_plane_component": r,
                 "max_per_sample": max(v["per_sample_max"] for v in r.values()),
                 "max_pooled": max(v["pooled"] for v in r.values())}
    print(f"{name}: max single-sample {out[name]['max_per_sample']:.3e}   max pooled {out[name]['max_pooled']:.3e}")
json.dump(out, open(os.path.join(HERE, "seam_total_energy.json"), "w"), indent=2)
