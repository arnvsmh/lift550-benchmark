"""
validate.py — pipeline validation gates. Must all pass before the full run.

Gate 1  radial binning reproduces the stored spectra in arrays_ALL_tier1.npz
Gate 2  degradation + normalisation reproduces the stored degraded input
Gate 3  end-to-end inference reproduces the stored model fields
Gate 4  coherence reproduces the published single-snapshot table
Gate 5  amplitude flagging reproduces the published flagged counts
"""
import json, time
import numpy as np
import torch

import coh_lib as L

NPZ = r"E:/lift550_paper/figure_data/arrays_ALL_tier1.npz"
SEED, SNAP, YP, YPV = 0, 708, "yp15", 15.0

# published single-snapshot coherence table (component -> model -> 3 bands)
PUBLISHED = {
    "v": {"Kim": (0.6699, 0.2861, 0.0436), "v3": (0.6616, 0.2723, 0.0664),
          "v4e": (0.6462, 0.2562, 0.0425), "v4d": (0.6380, 0.2405, 0.0435),
          "v4f12": (0.4209, 0.0466, 0.0070), "v4clo": (0.4197, 0.0718, 0.0020),
          "v4c": (0.4075, 0.0712, 0.0003)},
    "w": {"v4f12": (0.6383, 0.2009, 0.0247), "v4clo": (0.4433, 0.0390, 0.0082),
          "v4c": (0.4439, 0.0654, 0.0135)},
}
PUB_FLAGS = {"Kim": (0, 0, 0), "v3": (0, 0, 0), "v4e": (0, 0, 0),
             "v4d": (0, 0, 0), "v4f12": (0, 31, 0), "v4clo": (46, 29, 54),
             "v4c": (70, 29, 66)}

report = {}
ok_all = True
z = np.load(NPZ, allow_pickle=True)
stats = L.load_stats()


def spectrum(field):
    """metrics.py energy_spectrum, vectorised."""
    P = (np.abs(np.fft.fft2(field.astype(np.float64))) ** 2) / float(H * W) ** 2
    E = np.zeros(257)
    for ki in range(257):
        E[ki] = P[(L.K_RADIAL >= ki - 0.5) & (L.K_RADIAL < ki + 0.5)].sum()
    return E


H = W = 512
print("=" * 72)
print("GATE 1 — radial binning vs stored spectra (arrays_ALL_tier1.npz)")
print("=" * 72)
worst = 0.0
for nm, key in [("DNS", "dns"), ("Kim", "field_Kim"), ("v3", "field_v3"),
                ("v4e", "field_v4e"), ("v4d", "field_v4d"),
                ("f12", "field_f12"), ("basinLo", "field_basinLo"),
                ("basin", "field_basin")]:
    E = spectrum(z[key][0])
    stored = z[f"spec_{YP}_{nm}"]
    rel = np.abs(E - stored) / np.maximum(np.abs(stored), 1e-300)
    worst = max(worst, float(np.nanmax(rel[1:])))
    print(f"  spec_{YP}_{nm:8s} max rel err = {np.nanmax(rel[1:]):.2e}")
g1 = worst < 1e-5
ok_all &= g1
print(f"  -> worst {worst:.2e}   (earlier session reported 5.4e-06)  "
      f"{'PASS' if g1 else 'FAIL'}")
report["gate1_binning_max_rel_err"] = worst

print()
print("=" * 72)
print("GATE 2 — degradation + normalisation vs stored degraded input")
print("=" * 72)
_, idx = L.val_snapshot_list(YP)
t0 = time.time()
inp_norm, dns_phys = L.load_sample(idx, SEED, SNAP, YP, stats, quantize_f16=True)
t_sample = time.time() - t0
d_inp = np.abs(inp_norm - z["input"]).max()
r_inp = float(np.std(inp_norm - z["input"]))
f_inp = float((np.abs(inp_norm - z["input"]) > 0).mean())
d_dns = np.abs(dns_phys - z["dns"].astype(np.float64)).max()
# bit-exactness is not attainable: preprocess.py is not on disk and the stored
# cache was float16-quantised twice. Require the input to agree to well inside
# one float16 ulp in rms, and the DNS to agree to float32 round-trip.
g2 = bool(r_inp < 5e-5 and f_inp < 0.01 and d_dns < 1e-7)
ok_all &= g2
print(f"  |my inp_norm - stored input|  max={d_inp:.3e}  rms={r_inp:.3e} "
      f"({r_inp/1.0*100:.4f}% of sigma)  pixels differing={f_inp*100:.3f}%")
print(f"  |my dns_phys - stored dns|max   = {d_dns:.3e}")
print(f"  sample build time = {t_sample*1000:.0f} ms   "
      f"(September audit: ~72 ms/sample)")
print(f"  -> {'PASS' if g2 else 'FAIL'}")
report["gate2_input_maxdiff"] = float(d_inp)
report["gate2_input_rms"] = r_inp
report["gate2_input_frac_differing"] = f_inp
report["gate2_dns_maxdiff"] = float(d_dns)
report["sample_build_ms"] = t_sample * 1000

print()
print("=" * 72)
print("GATE 3/4/5 — end-to-end inference, coherence, flagging")
print("=" * 72)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}  "
      f"{torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")
print()
hdr = (f"  {'model':7s} {'|pred-stored|max':>16s} {'rmsrel':>8s} "
       f"{'flags(u,v,w)':>14s} {'pub':>14s}  "
       f"{'v:33-64':>8s} {'pub':>8s} {'v:65-128':>9s} {'pub':>8s} "
       f"{'v:129-256':>10s} {'pub':>8s}")
print(hdr)
g3_worst, g4_worst, g5 = 0.0, 0.0, True
per_model = {}
for label, run_dir, lam, npz_key in L.MODELS:
    model, takes_yp, cfg, ep = L.build_model(run_dir, device)
    pred = L.predict(model, takes_yp, inp_norm, YPV, stats, YP, device)
    stored_field = z[npz_key].astype(np.float64)

    dmax = float(np.abs(pred - stored_field).max())
    rel = float(np.std(pred - stored_field) / stored_field.std())   # RMS-relative
    g3_worst = max(g3_worst, rel)

    flags = tuple(int(L.flag_mask(pred[c], dns_phys[c]).sum()) for c in range(3))
    flags_stored = tuple(
        int(L.flag_mask(stored_field[c], dns_phys[c]).sum()) for c in range(3))
    g5 &= (flags_stored == PUB_FLAGS[label])

    cv = L.coherence(pred[1], dns_phys[1])
    pub = PUBLISHED["v"][label]
    diffs = [abs(cv[b] - p) for b, p in zip(["33-64", "65-128", "129-256"], pub)]
    g4_worst = max(g4_worst, max(diffs))

    print(f"  {label:7s} {dmax:16.3e} {rel:8.1e} {str(flags):>14s} "
          f"{str(PUB_FLAGS[label]):>14s}  "
          f"{cv['33-64']:8.4f} {pub[0]:8.4f} {cv['65-128']:9.4f} {pub[1]:8.4f} "
          f"{cv['129-256']:10.4f} {pub[2]:8.4f}")
    per_model[label] = {
        "epoch": ep, "lambda": lam,
        "pred_vs_stored_maxdiff": dmax, "pred_vs_stored_rel": rel,
        "flags_from_my_pred": flags, "flags_from_stored_field": flags_stored,
        "coh_v_from_my_pred": cv,
        "coh_v_published": list(pub),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# Gate 4 strictly: coherence recomputed from the STORED fields must match the
# published table to 4dp (this isolates the statistic from the inference path).
print()
print("  Gate 4 strict — coherence from STORED fields vs published table:")
g4s_worst = 0.0
for comp_i, comp in [(1, "v"), (2, "w")]:
    for label, _, _, npz_key in L.MODELS:
        if label not in PUBLISHED[comp]:
            continue
        c = L.coherence(z[npz_key][comp_i].astype(np.float64), dns_phys[comp_i])
        pub = PUBLISHED[comp][label]
        d = max(abs(c[b] - p)
                for b, p in zip(["33-64", "65-128", "129-256"], pub))
        g4s_worst = max(g4s_worst, d)
print(f"    worst abs deviation over all 30 published values = {g4s_worst:.2e}")

g3 = g3_worst < 0.01     # RMS-relative; fp16 autocast on a different GPU
g4 = g4s_worst < 5e-5
g6 = g4_worst < 2e-3      # end-to-end: coherence from MY inference vs published
ok_all &= g3 and g4 and g5 and g6
print()
print(f"  GATE 3 (inference reproduces stored fields): worst RMS-rel "
      f"{g3_worst:.2e}  {'PASS' if g3 else 'FAIL'}")
print(f"  GATE 4 (coherence statistic exact to 4dp):   {g4s_worst:.2e}  "
      f"{'PASS' if g4 else 'FAIL'}")
print(f"  GATE 5 (flag counts reproduce Table VI):     "
      f"{'PASS' if g5 else 'FAIL'}")
print(f"  GATE 6 (end-to-end coherence vs published):  max dev "
      f"{g4_worst:.5f}  {'PASS' if g6 else 'FAIL'}")

report.update({
    "gate3_worst_rel": g3_worst, "gate4_strict_worst_abs": g4s_worst,
    "gate5_flags_exact": bool(g5),
    "coh_drift_my_inference_vs_published": g4_worst,
    "per_model": per_model, "all_gates_pass": bool(ok_all),
    "snapshot": {"seed": SEED, "snap": SNAP, "yp": YP},
})
with open("validation_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print()
print("=" * 72)
print(f"ALL GATES {'PASS' if ok_all else 'FAIL'} — wrote validation_report.json")
print("=" * 72)
