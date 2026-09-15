"""
matched_time_correlation.py - correlation of u at y+ = 15 between snapshots of different simulations at matched times.
Source of the Section III-B sentence "|r| ~ 0.02 on average and at most 0.08 over 360 matched-time pairs".

Simulations are chained by seam continuity: file B follows file A when the last snapshot of A correlates with the
first snapshot of B at r > 0.8. Snapshots are read at the correct offset (header 136 B; each data record followed
by a 24-B metadata record; payload at 140 + i*2,097,184). Snapshot time = header start time (byte 32) + 0.15 i.
For every pair of different simulations, k times are spaced evenly over their common interval; each is matched to
the nearest snapshot of the other simulation within half an output interval. Reported: |Pearson r| per pair.
Read-only. Writes matched_time_correlation.json next to this script.
"""
import os
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NX = NZ = 512
HDR, META = 136, 24
DATA = 4 + NX * NZ * 8 + 4
FULL = META + DATA
DT = 0.15


def nsnap(f):
    r = os.path.getsize(f) - HDR - DATA
    return r // FULL + 1 if r >= 0 and r % FULL == 0 else 0


def f64(f, off):
    with open(f, "rb") as fh:
        fh.seek(off)
        return float(np.frombuffer(fh.read(8), "<f8")[0])


def read_u(f, i):
    with open(f, "rb") as fh:
        fh.seek(HDR + i * FULL + 4)
        return np.frombuffer(fh.read(NX * NZ * 8), "<f8").reshape(NZ, NX)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


sims = []
for part in ("train", "test"):
    fs = [f for f in sorted(glob.glob(f"D:/Research/misc/data/{part}/uxz_yp15_*.pl"),
                            key=lambda s: int(s.rsplit("_", 1)[1][:-3])) if nsnap(f)]
    first = {f: read_u(f, 0) for f in fs}
    last = {f: read_u(f, nsnap(f) - 1) for f in fs}
    succ, seam_r = {}, {}
    for a in fs:
        r, b = max((corr(last[a], first[b]), b) for b in fs if b != a)
        if r > 0.8:
            succ[a], seam_r[a] = b, r
    for h in [f for f in fs if f not in succ.values()]:
        chain = [h]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        sims.append({"part": part, "files": chain, "seam_r": [seam_r[x] for x in chain[:-1]]})

for s in sims:
    s["t"] = np.concatenate([f64(f, 32) + DT * np.arange(nsnap(f)) for f in s["files"]])
    s["loc"] = [(f, i) for f in s["files"] for i in range(nsnap(f))]


def matched(sa, sb, k):
    lo, hi = max(sa["t"][0], sb["t"][0]), min(sa["t"][-1], sb["t"][-1])
    if hi - lo < DT:
        return []
    out = []
    for tt in np.linspace(lo, hi, k):
        ia = int(np.argmin(np.abs(sa["t"] - tt)))
        ib = int(np.argmin(np.abs(sb["t"] - sa["t"][ia])))
        if abs(sb["t"][ib] - sa["t"][ia]) < DT / 2:
            out.append(abs(corr(read_u(*sa["loc"][ia]), read_u(*sb["loc"][ib]))))
    return out


result = {"simulations": [{"part": s["part"], "files": [os.path.basename(f) for f in s["files"]],
                           "seam_r": [round(x, 4) for x in s["seam_r"]]} for s in sims]}
for k in (12, 60):
    pairs, allr = {}, []
    for a in range(len(sims)):
        for b in range(a + 1, len(sims)):
            r = matched(sims[a], sims[b], k)
            if r:
                pairs[f"{a}-{b} ({sims[a]['part']}/{sims[b]['part']})"] = {"n": len(r), "mean_abs_r": float(np.mean(r)), "max_abs_r": float(np.max(r))}
                allr += r
    allr = np.array(allr)
    result[f"k{k}"] = {"pairs": pairs, "n_total": int(len(allr)), "mean_abs_r": float(allr.mean()),
                       "max_abs_r": float(allr.max()), "share_above_0.05": float(np.mean(allr > 0.05))}

json.dump(result, open(os.path.join(HERE, "matched_time_correlation.json"), "w"), indent=2)
print(json.dumps(result, indent=2))
