# analyze_spike.py - summaries of spike_out/records_full.npz. Read-only; prints tables and writes spike_out/summary.json.
import os, sys, json
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(SP, "spike_out")
ZD, META = {}, {}
for fn in ("records_full_A.npz", "records_full_B.npz", "records_full.npz"):
    if not os.path.exists(os.path.join(OUT, fn)):
        continue
    z = np.load(os.path.join(OUT, fn), allow_pickle=True)
    COLS = list(z["cols"]); DCOLS = list(z["dcols"]); ENV = json.loads(str(z["env"])); mz = json.loads(str(z["meta"]))
    for k in z.files:
        if k.startswith("T") and k not in ZD and len(z[k]):
            ZD[k] = z[k]; META[k] = mz[k]
    if "DNS" not in ZD:
        ZD["DNS"] = z["DNS"]
    print("loaded", fn)
YP = ["yp15", "yp30", "yp50", "yp100"]; PX = 512 * 512
AUD = r"C:\Users\Arnav Simha\AppData\Local\Temp\claude\C--Users-Arnav-Simha-AppData-Roaming-Claude-scratch-workspaces-1ca040d2-beed-4004-87e5-7e25f8e84e2c-ad963e29-37f9-4d31-9c6d-d2ba770099b0-scratch-2026-09-11-8d73ad\fdd0e3d7-14b8-410a-91ce-f0d1d67a823f\scratchpad\part1_out\records.npz"


def c(a, n):
    return a[:, COLS.index(n)]


def rec(key, subset=False):
    a = ZD[key]
    return a[c(a, "insub") > 0] if subset else a


def pm(a, f):
    """paper convention: mean over the four planes of per-plane means (planes missing are skipped)"""
    v = [f(a[c(a, "plane") == pi]) for pi in range(4) if np.any(c(a, "plane") == pi)]
    return float(np.mean(v))


def eall(a, rep=False):
    s = "_rep" if rep else ""
    return (c(a, "eu" + s) + c(a, "ev" + s) + c(a, "ew" + s)) / 3


keys = [k for k in ZD if k.startswith("T")]
S = {}
print("=" * 160)
print("1. VALIDATION - full 5,000-sample evaluation set vs summary JSON (NRMSE and E_err, per plane and plane-mean)")
print("=" * 160)
for k in keys:
    m = META[k]; a = rec(k)
    if not m["js"] or len(a) == 0 or m["sub_only"]:
        continue
    js = json.load(open(m["js"]))
    dn = []; de = []; cells = []
    for pi, yp in enumerate(YP):
        s = a[c(a, "plane") == pi]
        n1, e1 = c(s, "nrmse").mean(), eall(s).mean()
        n0, e0 = js["per_yp"][yp]["nrmse"], js["per_yp"][yp]["spectral_error"]
        dn.append(n1 - n0); de.append(e1 - e0)
        cells.append(f"{yp}:{len(s)}")
    N1 = pm(a, lambda s: c(s, "nrmse").mean()); E1 = pm(a, lambda s: eall(s).mean())
    N0 = np.mean([js["per_yp"][yp]["nrmse"] for yp in YP]); E0 = np.mean([js["per_yp"][yp]["spectral_error"] for yp in YP])
    ok = max(np.abs(dn + de + [N1 - N0, E1 - E0])) < 5e-5
    S.setdefault(k, {})["validation"] = dict(N=N1, N_json=N0, E=E1, E_json=E0, max_abs_diff=float(max(np.abs(dn + de))), epoch=m["epoch"], json_epoch=m["json_epoch"], ok=bool(ok))
    print(f" {k:<24} ep {m['epoch']}/{m['json_epoch']}  n[{', '.join(cells)}]  NRMSE {N1:.5f} vs {N0:.5f}  E_err {E1:.5f} vs {E0:.5f}  "
          f"max|per-plane diff| N {max(np.abs(dn)):.1e} E {max(np.abs(de)):.1e}  -> {'REPRODUCES (4 dp)' if ok else 'MISMATCH'}")

print("\n" + "=" * 160)
print("2. SPIKE STATISTICS (outlier = beyond DNS envelope +/- 3 sigma; region = outliers dilated 2 px). A = audit subset (1,233); F = full set (5,000)")
print("=" * 160)
env_min = np.array(ENV["emin"]); env_max = np.array(ENV["emax"])
hdr = (f" {'checkpoint':<24}{'set':<4}{'n':>5} {'%smp spiked':>11} {'%smp u/v/w':>18} {'outlier px%':>11} {'region px%':>10} {'blobs/smp':>9} "
       f"{'max|z| u/v/w (sigma)':>26} | {'NRMSE as-is -> masked':<22} | {'E_err as-is -> masked':<22}")
print(hdr)
for k in keys:
    for tag, subset in (("A", True), ("F", False)):
        a = rec(k, subset)
        if len(a) == 0 or (not subset and META[k]["sub_only"]):
            continue
        sp = 100 * np.mean(c(a, "vo_any") > 0)
        comp = [100 * np.mean(c(a, f"vo_{x}") > 0) for x in "uvw"]
        opx = 100 * c(a, "vo_any").mean() / PX; rpx = 100 * c(a, "region_px").mean() / PX
        bl = c(a, "blobs").mean()
        zmax = [max(np.max(c(a, f"zhi_{x}")), -np.min(c(a, f"zlo_{x}"))) for x in "uvw"]
        N0 = pm(a, lambda s: c(s, "nrmse").mean()); N1 = pm(a, lambda s: c(s, "nrmse_rep").mean())
        E0 = pm(a, lambda s: eall(s).mean()); E1 = pm(a, lambda s: eall(s, True).mean())
        S.setdefault(k, {})[tag] = dict(n=int(len(a)), pct_samples_spiked=sp, pct_samples_u_v_w=comp, outlier_px_pct=opx, region_px_pct=rpx, blobs_per_sample=bl,
                                        max_abs_z_u_v_w=zmax, nrmse=N0, nrmse_masked=N1, eerr=E0, eerr_masked=E1,
                                        e_u_v_w=[pm(a, lambda s, x=x: c(s, "e" + x).mean()) for x in "uvw"],
                                        e_u_v_w_masked=[pm(a, lambda s, x=x: c(s, "e" + x + "_rep").mean()) for x in "uvw"])
        print(f" {k:<24}{tag:<4}{len(a):>5} {sp:>10.1f}% {'/'.join(f'{x:.0f}' for x in comp):>18} {opx:>10.4f}% {rpx:>9.3f}% {bl:>9.2f} "
              f"{'/'.join(f'{x:.1f}' for x in zmax):>26} | {N0:.4f} -> {N1:.4f}".ljust(24) + f"       | {E0:.4f} -> {E1:.4f}")

print("\n" + "=" * 160)
print("3. PER-COMPONENT E_err (plane-mean convention; full set where available, else audit subset); masked in brackets where spikes present")
print("=" * 160)
for k in keys:
    tag = "A" if META[k]["sub_only"] else "F"
    d = S.get(k, {}).get(tag)
    if not d:
        continue
    eu, ev, ew = d["e_u_v_w"]; mu_, mv, mw = d["e_u_v_w_masked"]
    spiked = d["pct_samples_spiked"] > 0
    print(f" {k:<24}[{tag}] E_u {eu:.4f}  E_v {ev:.4f}  E_w {ew:.4f}  | mean {np.mean([eu, ev, ew]):.4f}" +
          (f"   masked: E_u {mu_:.4f}  E_v {mv:.4f}  E_w {mw:.4f}  | mean {np.mean([mu_, mv, mw]):.4f}" if spiked else ""))

print("\n" + "=" * 160)
print("4. DNS k=0 (mean-flow) share of total spectral energy, per plane (full set)")
print("=" * 160)
Dd = ZD["DNS"]
def dc(n): return Dd[:, DCOLS.index(n)]
for pi, yp in enumerate(YP):
    s = Dd[dc("plane") == pi]
    print(f" {yp}: n={len(s)}  k0 share u {np.mean(s[:, DCOLS.index('k0_u')]):.4f} (min {np.min(s[:, DCOLS.index('k0_u')]):.4f})  v {np.mean(s[:, DCOLS.index('k0_v')]):.2e}  w {np.mean(s[:, DCOLS.index('k0_w')]):.2e}")
S["_dns_k0"] = {yp: [float(np.mean(Dd[dc("plane") == pi][:, DCOLS.index(f"k0_{x}")])) for x in "uvw"] for pi, yp in enumerate(YP)}

print("\n" + "=" * 160)
print("5. CROSS-CHECK against the prior audit's records (same checkpoints, audit subset)")
print("=" * 160)
if os.path.exists(AUD):
    A = np.load(AUD, allow_pickle=True); acols = list(A["cols"])
    def ac(a, n): return a[:, acols.index(n)]
    for mine, theirs in (("T1|v4c_s42", "v4c_s42"), ("T1|v4clo_s43", "v4clo_s43"), ("T1|v4f12_s42", "v4f12_s42"), ("T1|v4e_s42", "v4e_s42"), ("T3|v4c_s42_oldrun", "v4c_s42_oldrun")):
        if mine not in ZD:
            continue
        t = mine.split("|")[0][1]
        aud = [A[f"T{t}|{theirs}|{yp}"] for yp in YP]
        audN0 = np.mean([ac(x, "nrmse").mean() for x in aud]); audN1 = np.mean([ac(x, "nrmse_rep").mean() for x in aud])
        audE0 = np.mean([((ac(x, "eerr_u") + ac(x, "eerr_v") + ac(x, "eerr_w")) / 3).mean() for x in aud])
        audE1 = np.mean([((ac(x, "eerr_rep_u") + ac(x, "eerr_rep_v") + ac(x, "eerr_rep_w")) / 3).mean() for x in aud])
        audsp = np.mean([100 * np.mean(ac(x, "vo_any") > 0) for x in aud])
        d = S[mine]["A"]
        print(f" {mine:<20} audit: NRMSE {audN0:.4f}->{audN1:.4f} E_err {audE0:.4f}->{audE1:.4f} spiked {audsp:.1f}%   |  this run: NRMSE {d['nrmse']:.4f}->{d['nrmse_masked']:.4f} "
              f"E_err {d['eerr']:.4f}->{d['eerr_masked']:.4f} spiked {d['pct_samples_spiked']:.1f}%")
print("\n" + "=" * 160)
print("6. PARTIALLY SPIKED CHECKPOINTS: per-plane breakdown (any checkpoint with 0% < spiked samples < 100%)")
print("=" * 160)
for k in keys:
    for tag, subset in (("F", False), ("A", True)):
        a = rec(k, subset)
        if len(a) == 0 or (not subset and META[k]["sub_only"]):
            continue
        sp = 100 * np.mean(c(a, "vo_any") > 0)
        if 0 < sp < 100:
            parts = []
            for pi, yp in enumerate(YP):
                s = a[c(a, "plane") == pi]
                parts.append(f"{yp}: {100*np.mean(c(s,'vo_any')>0):.1f}% (u/v/w {'/'.join(f'{100*np.mean(c(s,f'vo_{x}')>0):.0f}' for x in 'uvw')})")
            print(f" {k:<24}[{tag}] overall {sp:.1f}% | " + " | ".join(parts))
        break

print("\n" + "=" * 160)
print("7. THREE-SPIKE-STATES TEST: spike state (components flagged in >=50% of samples) vs component errors and level")
print("=" * 160)
def state_of(d):
    fl = [x for x, p in zip("uvw", d["pct_samples_u_v_w"]) if p >= 50]
    if not fl:
        return "none" if d["pct_samples_spiked"] < 50 else "mixed"
    return "+".join(fl)
def level_of(e):
    return "pixel-optimal" if e > 0.18 else ("intermediate" if e > 0.07 else "spectrum-matching")
rows7 = []
for k in keys:
    tag = "A" if META[k]["sub_only"] else "F"
    d = S.get(k, {}).get(tag)
    if not d or not k.startswith("T1"):
        continue
    eu, ev, ew = d["e_u_v_w"]; tot = eu + ev + ew
    st = state_of(d); lv = level_of(d["eerr"])
    rows7.append((k, tag, st, eu, ev, ew, d["eerr"], lv, 100 * eu / tot))
    print(f" {k:<24}[{tag}] state {st:<8} E_u {eu:.4f} ({100*eu/tot:4.1f}% of sum)  E_v {ev:.4f}  E_w {ew:.4f}  E_err {d['eerr']:.4f}  -> {lv}")
# component values by state, and the level each state predicts
by = {}
for k, tag, st, eu, ev, ew, e, lv, sh in rows7:
    if tag == "F" and st in ("none", "v", "u+v+w", "v+w"):
        by.setdefault(st, []).append((eu, ev, ew, e))
for st, v in by.items():
    v = np.array(v)
    print(f"   state {st:<6}: n={len(v)}  mean E_u {v[:,0].mean():.4f}  E_v {v[:,1].mean():.4f}  E_w {v[:,2].mean():.4f}  -> E_err {v[:,3].mean():.4f} (range {v[:,3].min():.4f}-{v[:,3].max():.4f})")
if "none" in by and ("u+v+w" in by or "v+w" in by):
    nn = np.array(by["none"]).mean(0); ss = np.array(by.get("u+v+w", by.get("v+w"))).mean(0)
    pred_v = (nn[0] + ss[1] + nn[2]) / 3
    print(f"   prediction for a v-only spike state from the other two states' component values: (E_u,none {nn[0]:.4f} + E_v,spiked {ss[1]:.4f} + E_w,none {nn[2]:.4f})/3 = {pred_v:.4f}")
    if "v" in by:
        print(f"   measured v-only state E_err: {np.array(by['v'])[:,3].mean():.4f}")
S["_states"] = {st: np.array(v).mean(0).tolist() for st, v in by.items()}

print("\n" + "=" * 160)
print("8. MANUSCRIPT TABLE CANDIDATE (Tier 1, final epoch, three-seed means where all seeds share a state; full evaluation set)")
print("=" * 160)
groups = [("v3 (s42)", "const. 0.05*", ["T1|v3_s42"]),
          ("v4e", "0.07", ["T1|v4e_s42", "T1|v4e_s43_ep50", "T1|v4e_s44"]),
          ("v4d (s42, s43)", "0.10", ["T1|v4d_s42", "T1|v4d_s43"]),
          ("v4d (s44)", "0.10", ["T1|v4d_s44"]),
          ("v4f12", "0.12", ["T1|v4f12_s42", "T1|v4f12_s43", "T1|v4f12_s44"]),
          ("v4clo", "0.15", ["T1|v4clo_s42", "T1|v4clo_s43", "T1|v4clo_s44"]),
          ("v4c", "0.30", ["T1|v4c_s42", "T1|v4c_s43", "T1|v4c_s44"])]
for g, lam, ks in groups:
    ds = [S[k]["F"] for k in ks if k in S and "F" in S[k]]
    if len(ds) != len(ks):
        print(f" {g:<16} (missing {len(ks)-len(ds)} of {len(ks)} checkpoints)"); continue
    sts = sorted({state_of(d) for d in ds})
    E = np.array([d["e_u_v_w"] for d in ds]); Em = np.array([d["e_u_v_w_masked"] for d in ds])
    e = np.array([d["eerr"] for d in ds]); em = np.array([d["eerr_masked"] for d in ds])
    zm = np.max([d["max_abs_z_u_v_w"] for d in ds], axis=0)
    print(f" {g:<16} lambda {lam:<12} spikes: {','.join(sts):<8} E_u {E[:,0].mean():.3f}  E_v {E[:,1].mean():.3f}  E_w {E[:,2].mean():.3f}  E_err {e.mean():.4f}"
          f"  | masked E_v {Em[:,1].mean():.3f}  E_w {Em[:,2].mean():.3f}  E_err {em.mean():.4f} | max |z| u/v/w {'/'.join(f'{x:.0f}' for x in zm)}")
print("\n" + "=" * 160)
print("9. BASELINES, Tier 1: three-seed means of per-component E_err (full evaluation set), spike status")
print("=" * 160)
for b in ("kim", "fukami_dscms", "guastoni", "fukami_cnn"):
    ks = [f"T1|{b}_s{s}" for s in (42, 43, 44)]
    ds = [S[k]["F"] for k in ks if k in S and "F" in S[k]]
    if len(ds) != 3:
        print(f" {b:<14} (missing {3-len(ds)} seeds)"); continue
    E = np.array([d["e_u_v_w"] for d in ds]); e = np.array([d["eerr"] for d in ds])
    print(f" {b:<14} spiked samples {max(d['pct_samples_spiked'] for d in ds):.1f}% (max over seeds)  E_u {E[:,0].mean():.4f}  E_v {E[:,1].mean():.4f}  E_w {E[:,2].mean():.4f}"
          f"  -> E_err {e.mean():.4f} +/- {e.std(ddof=1):.4f}  (E_u share {100*E[:,0].mean()/E.mean(0).sum():.1f}%)  max |z| u/v/w {'/'.join(f'{x:.1f}' for x in np.max([d['max_abs_z_u_v_w'] for d in ds], axis=0))}")
json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print("\nwrote", os.path.join(OUT, "summary.json"))
