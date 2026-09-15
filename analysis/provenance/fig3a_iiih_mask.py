"""
fig3a_iiih_mask.py - Figure 3a's masked-spectrum ratios under the rule the figure draws and under Section III-H.

Snapshot: E:/lift550_paper/figure_data/arrays_ALL_tier2.npz (Tier 2, y+ = 15, seed 42, validation index 0),
u component; v4c (field_basin) masked with the v4e prediction (field_v4e). Spectrum: metrics.py binning,
|fft2|^2 / (512*512)^2, bins (K >= k-0.5) & (K < k+0.5).

Rules compared:
  drawn   fig3_spectra_profiles.py: u only; this snapshot's DNS min/max +- 3 x this snapshot's DNS std; 7x7 dilation
  III-H   manuscript Section III-H: DNS envelope over the 1,233-sample audit subset +- 3 sigma_train;
          flags of any component merged; 5x5 dilation (two periodic 3x3 max filters)
  hybrid  this snapshot's DNS min/max +- 3 sigma_train; merged; 5x5  (the construction behind the printed 0.105%)

Gates: the binning must reproduce the stored DNS spectrum, and the drawn rule must reproduce the drawn curve.
Read-only. Writes fig3a_iiih_mask.json next to this script.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(r"E:/lift550_paper/figure_data/arrays_ALL_tier2.npz")
STATS = json.load(open(r"E:/lift550_paper/code/norm_stats_train.json"))
ENV = json.load(open(r"E:/lift550_paper/analysis_coherence_threeseed_20260912/envelope_iiih_yp15.json"))
N = 512
dns = Z["dns"].astype(np.float64)
v4c = Z["field_basin"].astype(np.float64)
v4e = Z["field_v4e"].astype(np.float64)

kx = np.fft.fftfreq(N, d=1.0 / N)
KX, KY = np.meshgrid(kx, kx)
K = np.sqrt(KX ** 2 + KY ** 2)
BINS = [(K >= k - 0.5) & (K < k + 0.5) for k in range(257)]
QS = (32, 64, 128)


def spectrum(f):
    P = np.abs(np.fft.fft2(f)) ** 2 / float(N * N) ** 2
    return np.array([P[m].sum() for m in BINS])


def dilate(m, r):
    o = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


kst = Z["spec_yp15_k"]
idx = {q: int(np.where(np.isclose(kst, q))[0][0]) for q in QS}
D = Z["spec_yp15_DNS"]
Ed = spectrum(dns[0])
gate_binning = float(max(abs(Ed[q] - D[idx[q]]) / abs(D[idx[q]]) for q in QS))
v4e_ratio = {q: float(Z["spec_yp15_v4e"][idx[q]] / D[idx[q]]) for q in QS}

sig = np.array([STATS[f"{v}_yp15"]["std"] for v in ("uxz", "vxz", "wxz")], np.float64)
emin, emax = np.array(ENV["emin"], np.float64), np.array(ENV["emax"], np.float64)


def evaluate(flag, region):
    mu = v4c[0].copy()
    mu[region] = v4e[0][region]
    Em = spectrum(mu)
    return {"flagged_px": int(flag.sum()), "region_px": int(region.sum()),
            "region_pct_of_plane": 100.0 * float(region.sum()) / (N * N),
            "ratio_to_dns": {str(q): float(Em[q] / D[idx[q]]) for q in QS}}


out = {"gate_binning_max_rel_err_at_32_64_128": gate_binning, "v4e_ratio_to_dns": {str(q): v for q, v in v4e_ratio.items()}}

sd_snap = dns[0].std()
f_drawn = (v4c[0] < dns[0].min() - 3 * sd_snap) | (v4c[0] > dns[0].max() + 3 * sd_snap)
out["drawn"] = evaluate(f_drawn, dilate(f_drawn, 3))

f_iiih = np.zeros((N, N), bool)
for c in range(3):
    f_iiih |= (v4c[c] < emin[c] - 3 * sig[c]) | (v4c[c] > emax[c] + 3 * sig[c])
out["section_III_H"] = evaluate(f_iiih, dilate(f_iiih, 2))

f_hyb = np.zeros((N, N), bool)
for c in range(3):
    f_hyb |= (v4c[c] < dns[c].min() - 3 * sig[c]) | (v4c[c] > dns[c].max() + 3 * sig[c])
out["hybrid_snapshot_range"] = evaluate(f_hyb, dilate(f_hyb, 2))

p1 = r"C:\Users\ARNAVS~1\AppData\Local\Temp\claude\C--Users-Arnav-Simha-AppData-Roaming-Claude-scratch-workspaces-1ca040d2-beed-4004-87e5-7e25f8e84e2c-ad963e29-37f9-4d31-9c6d-d2ba770099b0-scratch-2026-09-11-e1f53e\e83fac3e-0e3d-4103-901a-6b105f49369a\scratchpad\spike_out\records_p1.npz"
if os.path.exists(p1):
    R = np.load(p1, allow_pickle=True)
    ci = {str(c): i for i, c in enumerate(R["cols"])}
    r = R["T2|v4c_s42"]
    row = r[(r[:, ci["plane"]] == 0) & (r[:, ci["vi"]] == 0)]
    out["crosscheck_records_p1_region_px"] = int(row[0, ci["region_px"]]) if len(row) else None

json.dump(out, open(os.path.join(HERE, "fig3a_iiih_mask.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
