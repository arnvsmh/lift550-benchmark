"""
run_threeseed.py - spectral coherence with the DNS at model seeds 42, 43 and 44.

Same pipeline as E:/lift550_paper/analysis_coherence_multisample_20260912
(coh_lib.py copied unmodified; its SHA-256 is recorded in the meta): the same
500 Tier-1 y+=15 validation samples (stride 6), the same data build and float16
cache numerics, the same fp16-autocast inference, the same coherence statistic,
the same v4e seed-42 replacement field, the same best.pt checkpoints.

Extensions over the published run:
  * model seeds 43 and 44 for all seven models (checkpoints: Kim/v3/v4e in
    runs_final, the four sweep models in lift550_data/runs_newseeds);
  * two bands below the degradation cutoff, k = 1-16 and 17-32;
  * masking under BOTH flag rules, so every number can be stated under the rule
    actually used:
      P  the published run's per-snapshot rule (coh_lib.flag_mask): that
         snapshot's DNS min/max +- 3 x that snapshot's DNS std, per component,
         each component's flags dilated 2 px and replaced separately;
      S  the manuscript's Section III-H rule: DNS range over the 1,233-sample
         audit subset +- 3 sigma_train, flags of any component merged, dilated
         2 px, the region replaced in all three components.

Inference only. Writes only to this script's directory.
"""
import os, sys, json, time, math, argparse, hashlib
sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coh_lib as L                                                     # noqa: E402

PUB_DIR    = r"E:/lift550_paper/analysis_coherence_multisample_20260912"
RUNS_FINAL = r"E:/lift550_paper/runs_final"
RUNS_NEW   = r"C:/Users/Arnav Simha/lift550_data/runs_newseeds"
YP, YPV = "yp15", 15.0
LAB = ["Kim", "v3", "v4e", "v4d", "v4f12", "v4clo", "v4c"]
NAME = {"Kim": "kim", "v3": "yasunet_v3", "v4e": "yasunet_v4e", "v4d": "yasunet_v4d",
        "v4f12": "yasunet_v4f12", "v4clo": "yasunet_v4clo", "v4c": "yasunet_v4c"}
LAMBDA = {"Kim": None, "v3": 0.05, "v4e": 0.07, "v4d": 0.10, "v4f12": 0.12, "v4clo": 0.15, "v4c": 0.30}
SWEEP = ("v4d", "v4f12", "v4clo", "v4c")
BANDS = [("1-16", 1, 16), ("17-32", 17, 32), ("33-64", 33, 64), ("65-128", 65, 128), ("129-256", 129, 256)]
BAND_NAMES = [b[0] for b in BANDS]
PUB_BANDS = slice(2, 5)                     # the three published bands
IDX = [np.flatnonzero((L.K_RADIAL >= lo - 0.5) & (L.K_RADIAL < hi + 0.5)) for _, lo, hi in BANDS]
REPL = (42, "v4e")                          # the paper's fill at every model seed


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ckpt_path(lab, seed):
    root = RUNS_NEW if (lab in SWEEP and seed != 42) else RUNS_FINAL
    return os.path.join(root, f"{NAME[lab]}_tier1_seed{seed}", "best.pt")


def build(lab, seed, device):
    """Mirrors coh_lib.build_model, with an explicit path and an identity check."""
    path = ckpt_path(lab, seed)
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    assert cfg["model"] == NAME[lab] and int(cfg["tier"]) == 1 and int(cfg["seed"]) == seed, (path, cfg)
    if cfg["model"] == "kim":
        from baselines_pkg import build_baseline
        m = build_baseline("kim", in_channels=3, out_channels=3).to(device)
        ty = False
    else:
        from yasunet_v3 import YASUNetV2
        m = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128,
                      modes_h=10, modes_w=10, n_refine=6).to(device)
        ty = True
    m.load_state_dict(ck["model_state"])
    m.eval()
    ep = ck.get("epoch")
    del ck
    return m, ty, ep, path


def band_power(F):
    P = (F.real ** 2 + F.imag ** 2).ravel()
    return np.array([P[ix].sum() for ix in IDX])


def coh_from(Fp, Fd, dd):
    """coh_lib.coherence with the DNS transform and band powers precomputed.
    Identical arithmetic: same elements, same C order, same float64 sums."""
    cross = (Fp * np.conj(Fd)).ravel()
    pp = (Fp.real ** 2 + Fp.imag ** 2).ravel()
    out = np.empty(len(IDX))
    for bi, ix in enumerate(IDX):
        den = math.sqrt(pp[ix].sum() * dd[bi])
        out[bi] = abs(cross[ix].sum()) / den if den > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="first N samples only")
    ap.add_argument("--validate", action="store_true",
                    help="compare seed 42 against the published per-sample arrays")
    ap.add_argument("--out_prefix", default="threeseed")
    ap.add_argument("--envelope", default="envelope_iiih_yp15.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    assert REPL[0] in seeds, "the replacement field needs model seed 42"

    LOG = open(os.path.join(HERE, f"log_{a.out_prefix}.txt"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(y) for y in x)
        print(s, flush=True)
        LOG.write(s + "\n"); LOG.flush()

    stats = L.load_stats()
    device = torch.device("cuda")
    vals, idx = L.val_snapshot_list(YP)
    step = max(1, len(vals) // a.n_samples)
    sel = vals[::step][:a.n_samples]
    if a.limit:
        sel = sel[:a.limit]
    comps = ["uxz", "vxz", "wxz"]
    mu = np.array([stats[f"{v}_{YP}"]["mean"] for v in comps], np.float64)
    sd = np.array([stats[f"{v}_{YP}"]["std"] for v in comps], np.float64)
    env = json.load(open(os.path.join(HERE, a.envelope)))
    LO = np.array(env["emin"], np.float64) - 3.0 * sd
    HI = np.array(env["emax"], np.float64) + 3.0 * sd
    log(f"Tier-1 {YP} validation snapshots {len(vals)}; stride {step}; samples {len(sel)}; "
        f"data seeds {sorted(set(s for s, _ in sel))}")
    log(f"III-H thresholds (sigma about the training mean): lo {np.round((LO - mu) / sd, 2)} "
        f"hi {np.round((HI - mu) / sd, 2)}")
    log(f"coh_lib.py sha256 {sha256(os.path.join(HERE, 'coh_lib.py'))}  "
        f"(published {sha256(os.path.join(PUB_DIR, 'coh_lib.py'))})")
    log(f"device {torch.cuda.get_device_name(0)}")

    models, mmeta = {}, {}
    for s in seeds:
        for lab in LAB:
            m, ty, ep, path = build(lab, s, device)
            models[(s, lab)] = (m, ty)
            mmeta[f"{lab}_s{s}"] = {"path": path, "epoch": ep, "lambda": LAMBDA[lab]}
            log(f"  loaded {lab:<6} seed {s}  epoch {ep}  <- {path}")

    nSd, nM, nC, nB, nS = len(seeds), len(LAB), 3, len(BANDS), len(sel)
    shp = (nSd, nM, nC, nB, nS)
    coh_app = np.full(shp, np.nan)
    coh_mP = np.full(shp, np.nan)
    coh_mS = np.full(shp, np.nan)
    flagsP = np.zeros((nSd, nM, nC, nS), np.int64)
    flagsPd = np.zeros_like(flagsP)
    flagsS = np.zeros_like(flagsP)
    regionS = np.zeros((nSd, nM, nS), np.int64)
    zmax = np.zeros((nSd, nM, nC, nS))
    sample_seed = np.array([x for x, _ in sel], np.int32)
    sample_snap = np.array([y for _, y in sel], np.int32)

    def save(n_done, elapsed):
        np.savez_compressed(
            os.path.join(HERE, f"{a.out_prefix}_per_sample.npz"),
            coh_apparent=coh_app, coh_masked_P=coh_mP, coh_masked_S=coh_mS,
            flags_P=flagsP, flags_P_dilated=flagsPd, flags_S=flagsS, region_S=regionS, zmax=zmax,
            seeds=np.array(seeds), labels=np.array(LAB), components=np.array(["u", "v", "w"]),
            bands=np.array(BAND_NAMES), sample_seed=sample_seed, sample_snap=sample_snap,
            n_done=np.array(n_done))
        json.dump({
            "n_samples": nS, "n_done": n_done, "stride": step, "n_val_available": len(vals),
            "yp": YP, "tier": 1, "model_seeds": seeds, "replacement": f"{REPL[1]} seed {REPL[0]}",
            "bands": BAND_NAMES, "elapsed_min": elapsed / 60, "models": mmeta,
            "envelope_audit_subset": env,
            "thresholds_sigma": {"lo": ((LO - mu) / sd).tolist(), "hi": ((HI - mu) / sd).tolist()},
            "coh_lib_sha256": sha256(os.path.join(HERE, "coh_lib.py")),
            "data_seeds": sorted(set(int(s) for s, _ in sel)),
        }, open(os.path.join(HERE, f"{a.out_prefix}_meta.json"), "w"), indent=2, default=str)

    t0 = time.time()
    with torch.no_grad():
        for si, (dseed, snap) in enumerate(sel):
            inp, dns = L.load_sample(idx, dseed, snap, YP, stats, quantize_f16=True)
            Fd = [np.fft.fft2(dns[c].astype(np.float64)) for c in range(3)]
            dd = [band_power(Fd[c]) for c in range(3)]
            preds = {}
            for s in seeds:
                for lab in LAB:
                    m, ty = models[(s, lab)]
                    preds[(s, lab)] = L.predict(m, ty, inp, YPV, stats, YP, device)
            repl = preds[REPL]
            for k, s in enumerate(seeds):
                for mi, lab in enumerate(LAB):
                    p = preds[(s, lab)]
                    ca = [coh_from(np.fft.fft2(p[c].astype(np.float64)), Fd[c], dd[c]) for c in range(3)]
                    for c in range(3):
                        coh_app[k, mi, c, :, si] = ca[c]
                        zmax[k, mi, c, si] = np.abs((p[c] - mu[c]) / sd[c]).max()
                        # rule P - exactly run_coherence.py
                        fm = L.flag_mask(p[c], dns[c])
                        nf = int(fm.sum())
                        flagsP[k, mi, c, si] = nf
                        if nf == 0:
                            coh_mP[k, mi, c, :, si] = ca[c]
                        else:
                            fdl = L.dilate(fm, 2)
                            flagsPd[k, mi, c, si] = int(fdl.sum())
                            pm = p[c].copy(); pm[fdl] = repl[c][fdl]
                            coh_mP[k, mi, c, :, si] = coh_from(np.fft.fft2(pm.astype(np.float64)), Fd[c], dd[c])
                    # rule S - Section III-H
                    fS = [(p[c] > HI[c]) | (p[c] < LO[c]) for c in range(3)]
                    for c in range(3):
                        flagsS[k, mi, c, si] = int(fS[c].sum())
                    anyS = fS[0] | fS[1] | fS[2]
                    if not anyS.any():
                        for c in range(3):
                            coh_mS[k, mi, c, :, si] = ca[c]
                    else:
                        R = L.dilate(anyS, 2)
                        regionS[k, mi, si] = int(R.sum())
                        for c in range(3):
                            pm = p[c].copy(); pm[R] = repl[c][R]
                            coh_mS[k, mi, c, :, si] = coh_from(np.fft.fft2(pm.astype(np.float64)), Fd[c], dd[c])
            if (si + 1) % 25 == 0 or si == 0:
                el = time.time() - t0
                log(f"  {si + 1}/{nS}  {el / 60:.1f} min  {el / (si + 1):.2f} s/sample  "
                    f"ETA {el / (si + 1) * (nS - si - 1) / 60:.1f} min")
            if (si + 1) % 50 == 0:
                save(si + 1, time.time() - t0)
    elapsed = time.time() - t0
    save(nS, elapsed)
    log(f"done: {nS} samples x {nSd} seeds x {nM} models in {elapsed / 60:.1f} min")

    if a.validate:
        pub = np.load(os.path.join(PUB_DIR, "coherence_per_sample.npz"), allow_pickle=True)
        assert [str(x) for x in pub["labels"]] == LAB
        n = nS
        same_samples = bool(np.array_equal(pub["sample_seed"][:n], sample_seed)
                            and np.array_equal(pub["sample_snap"][:n], sample_snap))
        k = seeds.index(42)
        dA = float(np.nanmax(np.abs(coh_app[k][:, :, PUB_BANDS, :] - pub["coh_apparent"][..., :n])))
        dM = float(np.nanmax(np.abs(coh_mP[k][:, :, PUB_BANDS, :] - pub["coh_masked"][..., :n])))
        fl = int((flagsP[k] != pub["flags"][..., :n]).sum())
        fld = int((flagsPd[k] != pub["flags_dilated"][..., :n]).sum())
        ok = same_samples and dA < 1e-4 and dM < 1e-4 and fl == 0 and fld == 0
        rep = {"n_samples_compared": n, "same_samples": same_samples,
               "max_abs_dev_apparent": dA, "max_abs_dev_masked_P": dM,
               "flag_count_mismatches": fl, "dilated_flag_count_mismatches": fld, "pass": ok}
        json.dump(rep, open(os.path.join(HERE, f"{a.out_prefix}_validation.json"), "w"), indent=2)
        log(f"VALIDATION vs published seed-42 per-sample arrays: {rep}")


if __name__ == "__main__":
    main()
