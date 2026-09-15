"""
analyse_threeseed.py - statistics for run_threeseed.py, per model seed and across seeds.

Reads threeseed_per_sample.npz; writes threeseed_report.txt and threeseed_summary.json.
Every statistic mirrors analyse.py of the published seed-42 run (same seed-blocked SEM over
the 20 data files, same pooled-sd internal control), computed at each model seed, under the
Section III-H flag rule (S) unless marked P (the published per-snapshot rule).
"""
import json
import numpy as np
from scipy import stats as st

Z = np.load("threeseed_per_sample.npz", allow_pickle=True)
SEEDS = [int(s) for s in Z["seeds"]]
LAB = [str(x) for x in Z["labels"]]
COMP = [str(x) for x in Z["components"]]
BAND = [str(x) for x in Z["bands"]]
CA, CP, CS = Z["coh_apparent"], Z["coh_masked_P"], Z["coh_masked_S"]    # [seed, model, comp, band, sample]
FP, FS, RS, ZM = Z["flags_P"], Z["flags_S"], Z["region_S"], Z["zmax"]    # [seed, model, comp, sample] / [seed, model, sample]
DS = Z["sample_seed"]
nS = CA.shape[-1]
assert int(Z["n_done"]) == nS, "run not complete"
FILES = sorted(set(DS.tolist()))
UP = [BAND.index(b) for b in ("33-64", "65-128", "129-256")]
LOW = [BAND.index(b) for b in ("1-16", "17-32")]
LAMBDA = {"Kim": None, "v3": 0.05, "v4e": 0.07, "v4d": 0.10, "v4f12": 0.12, "v4clo": 0.15, "v4c": 0.30}
out, J = [], {"seeds": SEEDS, "n_samples": nS, "n_files": len(FILES)}
P = out.append


def sem_cluster(x):
    means = np.array([x[DS == s].mean() for s in FILES])
    return means.std(ddof=1) / np.sqrt(len(FILES))


def state(k, mi, F=FS):
    frac = float((F[k, mi].sum(axis=0) > 0).mean())
    return ("spiked" if frac > 0.5 else "marginal" if frac > 0 else "clean"), frac


P("=" * 100)
P(f"SPECTRAL COHERENCE WITH THE DNS - Tier 1, y+=15, {nS} samples ({len(FILES)} data files), model seeds {SEEDS}")
P("=" * 100)

# 1 ---------------------------------------------------------------------------------------------
P("\n1. FLAGS AND SPIKE STATES (S = Section III-H rule; P = published per-snapshot rule)")
J["states"] = {}
for k, s in enumerate(SEEDS):
    P(f"-- model seed {s}")
    for mi, l in enumerate(LAB):
        stS, frS = state(k, mi, FS); stP, frP = state(k, mi, FP)
        cells = "  ".join(f"{c}: S {FS[k,mi,ci].mean():5.1f}/{FS[k,mi,ci].max():3d}/{100*np.mean(FS[k,mi,ci]>0):3.0f}%"
                          f"  P {FP[k,mi,ci].mean():5.1f}/{FP[k,mi,ci].max():3d}/{100*np.mean(FP[k,mi,ci]>0):3.0f}%"
                          for ci, c in enumerate(COMP))
        reg = RS[k, mi][RS[k, mi] > 0]
        P(f"   {l:<6} S-state {stS:<8} ({100*frS:5.1f}% of samples)  P-state {stP:<8} ({100*frP:5.1f}%)  "
          f"region {100*reg.mean()/512**2 if reg.size else 0:.3f}% (max {100*reg.max()/512**2 if reg.size else 0:.3f}%)  "
          f"max|z| {ZM[k,mi].max():6.1f}   [mean/max/%samples] {cells}")
        J["states"][f"{l}_s{s}"] = {
            "S": stS, "S_frac": frS, "P": stP, "P_frac": frP, "max_abs_z": float(ZM[k, mi].max()),
            "region_S_mean_pct": float(100 * reg.mean() / 512 ** 2) if reg.size else 0.0,
            "flags_S": {c: {"mean": float(FS[k, mi, ci].mean()), "max": int(FS[k, mi, ci].max()),
                            "frac": float(np.mean(FS[k, mi, ci] > 0))} for ci, c in enumerate(COMP)},
            "flags_P": {c: {"mean": float(FP[k, mi, ci].mean()), "max": int(FP[k, mi, ci].max()),
                            "frac": float(np.mean(FP[k, mi, ci] > 0))} for ci, c in enumerate(COMP)}}

# 2 ---------------------------------------------------------------------------------------------
P("\n2. APPARENT COHERENCE, mean +- sd over samples")
J["apparent"] = {}
for ci, c in enumerate(COMP):
    P(f"-- component {c}")
    P(f"   {'model':<7}{'seed':>5} " + "".join(f"{b:>18s}" for b in BAND))
    for mi, l in enumerate(LAB):
        for k, s in enumerate(SEEDS):
            x = CA[k, mi, ci]
            P(f"   {l:<7}{s:>5} " + "".join(f"{x[bi].mean():10.4f}+-{x[bi].std(ddof=1):6.4f}" for bi in range(len(BAND))))
            for bi, b in enumerate(BAND):
                J["apparent"][f"{l}_s{s}_{c}_{b}"] = [float(x[bi].mean()), float(x[bi].std(ddof=1))]

# 3 ---------------------------------------------------------------------------------------------
P("\n3. INTERNAL CONTROLS - spiked in v only, against the not-spiked models of the same seed, pooled (units of pooled sd)")
J["internal"] = {}
for k, s in enumerate(SEEDS):
    pool = [l for mi, l in enumerate(LAB) if state(k, mi)[0] != "spiked"]
    vonly = [l for mi, l in enumerate(LAB)
             if state(k, mi)[0] == "spiked" and FS[k, mi, 0].sum() == 0 and FS[k, mi, 2].sum() == 0]
    P(f"-- seed {s}: pool {pool};  spiked in v only: {vonly}")
    for l in vonly:
        mi = LAB.index(l)
        for ci, c in enumerate(COMP):
            row = []
            for bi, b in enumerate(BAND):
                f = CA[k, mi, ci, bi]
                cl = np.concatenate([CA[k, LAB.index(m), ci, bi] for m in pool])
                g = (f.mean() - cl.mean()) / cl.std(ddof=1)
                row.append(g)
                J["internal"][f"{l}_s{s}_{c}_{b}"] = {"gap_sd": float(g), "mean": float(f.mean()), "pool_mean": float(cl.mean()),
                                                      "pool": pool}
            P(f"   {l:<6} {c}: flagged px mean {FS[k,mi,ci].mean():5.1f} max {FS[k,mi,ci].max():3d} (P: {FP[k,mi,ci].mean():5.1f}/{FP[k,mi,ci].max():3d}) | "
              + "  ".join(f"{b} {g:+7.2f}" for b, g in zip(BAND, row)))

# 4 ---------------------------------------------------------------------------------------------
P("\n4. APPARENT -> MASKED (S rule: merged region, all three components; P shown for comparison), means over samples")
J["masked"] = {}
for k, s in enumerate(SEEDS):
    P(f"-- seed {s}")
    for mi, l in enumerate(LAB):
        stS = state(k, mi)[0]
        noop_S = bool(np.array_equal(np.nan_to_num(CS[k, mi]), np.nan_to_num(CA[k, mi]))) if RS[k, mi].sum() == 0 else False
        noop_P = bool(np.array_equal(np.nan_to_num(CP[k, mi]), np.nan_to_num(CA[k, mi]))) if FP[k, mi].sum() == 0 else False
        for ci, c in enumerate(COMP):
            P(f"   {l:<6} {stS:<8} {c}: " + "  ".join(
                f"{b} {CA[k,mi,ci,bi].mean():.4f}->{CS[k,mi,ci,bi].mean():.4f} (P {CP[k,mi,ci,bi].mean():.4f})" for bi, b in enumerate(BAND))
              + (f"   [no-op: S {noop_S}, P {noop_P}]" if ci == 0 else ""))
            for bi, b in enumerate(BAND):
                J["masked"][f"{l}_s{s}_{c}_{b}"] = {"apparent": float(CA[k, mi, ci, bi].mean()), "masked_S": float(CS[k, mi, ci, bi].mean()),
                                                    "masked_P": float(CP[k, mi, ci, bi].mean())}
        J["masked"][f"{l}_s{s}_noop"] = {"S": noop_S, "P": noop_P}

# 5 ---------------------------------------------------------------------------------------------
P("\n5. v4e VERSUS Kim, paired on the same samples; t from the SEM blocked on data file (df = %d)" % (len(FILES) - 1))
J["v4e_vs_kim"] = {}
ik, ie = LAB.index("Kim"), LAB.index("v4e")


def paired(A_, B_):
    d = A_ - B_
    sem = sem_cluster(d)
    t = d.mean() / sem
    return d, t, 2 * st.t.sf(abs(t), len(FILES) - 1)


for k, s in enumerate(SEEDS):
    P(f"-- seed {s} (v4e seed {s} against Kim seed {s})")
    for ci, c in enumerate(COMP):
        cells = []
        for bi, b in enumerate(BAND):
            kk, ee = CA[k, ik, ci, bi], CA[k, ie, ci, bi]
            d, t, p = paired(ee, kk)
            gap = 100 * d.mean() / kk.mean()
            ratio = abs(d.mean()) / kk.std(ddof=1)
            cells.append(f"{b} {gap:+6.2f}% t{t:+6.1f} |d|/sd {ratio:4.2f}")
            J["v4e_vs_kim"][f"s{s}_{c}_{b}"] = {"gap_pct": float(gap), "t": float(t), "p": float(p), "abs_d_over_sd_kim": float(ratio),
                                                "frac_below": float(np.mean(d < 0)), "C_kim": float(kk.mean()), "C_v4e": float(ee.mean())}
        P(f"   {c}: " + " | ".join(cells))
P("-- all seeds pooled: per-sample difference averaged over the three matched seed pairs")
for ci, c in enumerate(COMP):
    cells = []
    for bi, b in enumerate(BAND):
        kk, ee = CA[:, ik, ci, bi].mean(0), CA[:, ie, ci, bi].mean(0)
        d, t, p = paired(ee, kk)
        gap = 100 * d.mean() / kk.mean()
        cells.append(f"{b} {gap:+6.2f}% t{t:+6.1f} p{p:.1e}")
        J["v4e_vs_kim"][f"pooled_{c}_{b}"] = {"gap_pct": float(gap), "t": float(t), "p": float(p),
                                              "abs_d_over_sd_kim": float(abs(d.mean()) / kk.std(ddof=1))}
    P(f"   {c}: " + " | ".join(cells))

# 6 ---------------------------------------------------------------------------------------------
P("\n6. v3 VERSUS Kim - cells above Kim (upper three bands x three components = 9; all five bands = 15)")
J["v3_vs_kim"] = {}
iv3 = LAB.index("v3")
for k, s in enumerate(SEEDS):
    up9 = sum(CA[k, iv3, ci, bi].mean() > CA[k, ik, ci, bi].mean() for ci in range(3) for bi in UP)
    all15 = sum(CA[k, iv3, ci, bi].mean() > CA[k, ik, ci, bi].mean() for ci in range(3) for bi in range(len(BAND)))
    J["v3_vs_kim"][f"s{s}"] = {"above_of_9": int(up9), "above_of_15": int(all15)}
    P(f"   seed {s}: above Kim in {up9} of 9 upper-band cells, {all15} of 15 cells")

# 7 ---------------------------------------------------------------------------------------------
P("\n7. BELOW THE DEGRADATION CUTOFF (k = 1-16, 17-32), component v, apparent mean +- sd")
for k, s in enumerate(SEEDS):
    P(f"   seed {s}: " + " | ".join(f"{l} {CA[k,LAB.index(l),1,LOW[0]].mean():.4f}/{CA[k,LAB.index(l),1,LOW[1]].mean():.4f}" for l in LAB))

# 8 ---------------------------------------------------------------------------------------------
P("\n8. v4e VERSUS Kim WITHOUT A SEED PAIRING - all nine cross-seed pairings, and each model's own seed-to-seed range")
J["v4e_vs_kim_cross"] = {}
iv3 = LAB.index("v3")
for ci, c in enumerate(COMP):
    cells = []
    for bi, b in enumerate(BAND):
        kim = [float(CA[j, ik, ci, bi].mean()) for j in range(len(SEEDS))]
        v4e = [float(CA[i, ie, ci, bi].mean()) for i in range(len(SEEDS))]
        v3m = [float(CA[i, iv3, ci, bi].mean()) for i in range(len(SEEDS))]
        gaps = [100 * (e / k_ - 1) for e in v4e for k_ in kim]
        nbelow = int(sum(g < 0 for g in gaps))
        J["v4e_vs_kim_cross"][f"{c}_{b}"] = {"kim_by_seed": kim, "v4e_by_seed": v4e, "v3_by_seed": v3m,
                                             "pairings_v4e_below": nbelow, "gap_pct_min": float(min(gaps)), "gap_pct_max": float(max(gaps)),
                                             "ranges_overlap": bool(max(v4e) >= min(kim) and max(kim) >= min(v4e))}
        cells.append(f"{b} below in {nbelow}/9 ({min(gaps):+.1f}..{max(gaps):+.1f}%)")
    P(f"   {c}: " + " | ".join(cells))
P("   per-seed means (seeds " + ", ".join(map(str, SEEDS)) + "):")
for ci, c in enumerate(COMP):
    for bi, b in enumerate(BAND):
        d = J["v4e_vs_kim_cross"][f"{c}_{b}"]
        P(f"     {c} {b:>8}:  Kim {np.round(d['kim_by_seed'], 4)}   v4e {np.round(d['v4e_by_seed'], 4)}   v3 {np.round(d['v3_by_seed'], 4)}"
          f"   seed ranges overlap (Kim, v4e): {d['ranges_overlap']}")

# 9 ---------------------------------------------------------------------------------------------
P("\n9. RANGES AS THEY WOULD BE QUOTED (all three seeds, S rule, component v unless stated)")
Q = {}
ups = ["33-64", "65-128", "129-256"]
notsp = {k: [l for mi, l in enumerate(LAB) if state(k, mi)[0] != "spiked"] for k in range(len(SEEDS))}
spk = {k: [l for mi, l in enumerate(LAB) if state(k, mi)[0] == "spiked"] for k in range(len(SEEDS))}
for b in ups:
    bi = BAND.index(b)
    ns = [CA[k, LAB.index(l), 1, bi].mean() for k in range(len(SEEDS)) for l in notsp[k]]
    sp = [CA[k, LAB.index(l), 1, bi].mean() for k in range(len(SEEDS)) for l in spk[k]]
    ms = [CS[k, LAB.index(l), 1, bi].mean() for k in range(len(SEEDS)) for l in spk[k]]
    Q[b] = {"not_spiked": [float(min(ns)), float(max(ns))], "spiked": [float(min(sp)), float(max(sp))], "spiked_masked_S": [float(min(ms)), float(max(ms))]}
    P(f"   v {b:>8}: not spiked {min(ns):.3f}-{max(ns):.3f}   spiked {min(sp):.3f}-{max(sp):.3f}   spiked after masking {min(ms):.3f}-{max(ms):.3f}")
for k, s in enumerate(SEEDS):
    for l in [x for x in spk[k] if FS[k, LAB.index(x), 0].sum() == 0 and FS[k, LAB.index(x), 2].sum() == 0]:
        mi = LAB.index(l)
        g = {c: [J["internal"][f"{l}_s{s}_{c}_{b}"]["gap_sd"] for b in ups] for c in COMP}
        other = max(abs(x) for c in ("u", "w") for x in g[c])
        fl = FS[k, mi, 1]
        Q[f"control_{l}_s{s}"] = {"v_gap_sd": g["v"], "max_abs_gap_uw_sd": float(other), "flag_v_mean": float(fl.mean()),
                                  "flag_v_max": int(fl.max()), "flag_v_frac": float(np.mean(fl > 0)), "flag_uw_total": int(FS[k, mi, 0].sum() + FS[k, mi, 2].sum())}
        P(f"   control {l} seed {s}: v {np.round(g['v'], 2)} sd; u and w within {other:.2f} sd; v flags mean {fl.mean():.1f}, max {fl.max()}, "
          f"on {100*np.mean(fl > 0):.0f}% of samples; u+w flags {int(FS[k, mi, 0].sum() + FS[k, mi, 2].sum())}")
noop = {f"{l}_s{s}": J["masked"][f"{l}_s{s}_noop"]["S"] for k, s in enumerate(SEEDS) for mi, l in enumerate(LAB) if state(k, mi)[0] == "clean"}
Q["noop_clean_S"] = noop
P(f"   masking a no-op for every clean model: {all(noop.values())}  ({len(noop)} model-seeds: {sorted(noop)})")
for b in ("1-16", "17-32"):
    bi = BAND.index(b)
    for l in ("v4e", "v4f12", "v4c"):
        vals = [float(CA[k, LAB.index(l), 1, bi].mean()) for k in range(len(SEEDS))]
        Q[f"below_{l}_{b}"] = vals
        P(f"   v {b:>6} {l:<6} by seed {np.round(vals, 4)}")
for c in COMP:
    for b in ups:
        g = [J["v4e_vs_kim"][f"s{s}_{c}_{b}"] for s in SEEDS]
        P(f"   v4e-Kim {c} {b:>8} by seed: gap {[round(x['gap_pct'], 2) for x in g]}%  t {[round(x['t'], 1) for x in g]}  |d|/sd(Kim) {[round(x['abs_d_over_sd_kim'], 2) for x in g]}")
J["quoted"] = Q

txt = "\n".join(out)
open("threeseed_report.txt", "w", encoding="utf-8").write(txt)
json.dump(J, open("threeseed_summary.json", "w"), indent=1)
print(txt)
