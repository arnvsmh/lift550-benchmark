# phase1_checks.py - inference-only checks for the Phase-1 rewrite, on the 1,233-sample audit subset (all three tiers).
#  (1) spike status of every Tables II-IV checkpoint at Tiers 2 and 3 (Tier 1 was covered by spiketest.py);
#  (2) the v4e Tier-3 lowest-combined-loss checkpoints (s42 epoch 7, s43 epoch 19, s44 epoch 34) and v4c Tier-2 s42 (Figs. 2-3);
#  (3) loss accounting for every spiked checkpoint: share of squared error and of the Charbonnier loss inside the flagged
#      region, and the training-loss terms with the flagged pixels as predicted vs replaced (fill = v4e s42; fill = degraded input);
#  (4) variance outside the flagged region and the share of fluctuation energy inside it;
#  (5) Tier-2 spectra / RMS accumulators for the Figure-3 models (as predicted and masked).
# Same pipeline and outlier rule as spiketest.py. Read-only for all user files; writes only to <scratchpad>/spike_out.
import os, sys, json, time, argparse
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SP, "lift_code"))
import reader as R
from degradation import degrade

P = r"C:\Users\Arnav Simha\Downloads\lift550_paper"
N = r"C:\Users\Arnav Simha\lift550_data"
D = r"D:\lift550_cache"
RAW = r"D:\Research\misc\data\train"
RF, RB, RN = os.path.join(P, "runs_final"), os.path.join(P, "runs_baselines"), os.path.join(N, "runs_newseeds")
JF, JC, JB, JN = os.path.join(P, "results_final"), os.path.join(P, "results_charb"), os.path.join(P, "results_baselines"), os.path.join(N, "results_newseeds")
STATS = json.load(open(os.path.join(P, "code", "norm_stats_train.json")))
VARS = ["uxz", "vxz", "wxz"]; YP = ["yp15", "yp30", "yp50", "yp100"]
YPV = [15.0, 30.0, 50.0, 100.0]
IV = json.load(open(os.path.join(D, "index_val.json")))
OUT = os.path.join(SP, "spike_out")
EPS = 1e-3
FILLS = ("", "_rep", "_in")      # as predicted | flagged region <- v4e s42 prediction | flagged region <- degraded input
COLS = (["vi", "plane", "nrmse", "nrmse_rep", "nrmse_in"]
        + [f"e{c}{f}" for f in FILLS for c in "uvw"]
        + ["vo_u", "vo_v", "vo_w", "vo_any", "region_px", "blobs", "zhi_u", "zhi_v", "zhi_w", "zlo_u", "zlo_v", "zlo_w",
           "sen_tot", "sen_reg", "sep_tot", "sep_reg", "ch_tot", "ch_reg"]
        + [f"{q}{f}" for q in ("spec", "charb", "div", "grad") for f in FILLS]
        + [f"{q}_{c}" for q in ("vout_m", "vout_ref", "vout_dns", "vall_m", "esh") for c in "uvw"])
FIG3 = ["DNS", "kim_s42", "v3_s42", "v4e_s42", "v4c_s42", "v4c_s42_rep", "v4c_s42_in"]
BASE = ("kim", "fukami_cnn", "fukami_dscms", "guastoni")


def eval_set():
    n = IV["total_snapshots"]
    entries = [(yp, i) for yp in YP for i in range(n)]
    sel = entries[::max(1, len(entries) // 5000)][:5000]          # identical to evaluate_cache.py
    full = {yp: [i for (y, i) in sel if y == yp] for yp in YP}
    sub = {yp: set(full[yp][::(1 if yp == "yp100" else 5)]) for yp in YP}   # audit subset: strides 5/1
    return full, sub


def read_dns(yp, vi):
    e = IV["snap_index"][vi]
    return np.stack([R.read_snapshot(os.path.join(RAW, f"{v}_{yp}_{e['seed']}.pl"), e["snap"]).astype(np.float32) for v in VARS])


def make_input(dns, yp, tier):
    deg = degrade(torch.from_numpy(dns.copy()), tier=tier, training=False).numpy()
    deg16 = deg.astype(np.float16).astype(np.float32)
    out = np.empty_like(deg16)
    for c, v in enumerate(VARS):
        s = STATS[f"{v}_{yp}"]
        out[c] = ((deg16[c] - np.float32(s["mean"])) / np.float32(s["std"])).astype(np.float16).astype(np.float32)
    return out


class EvalDS(torch.utils.data.Dataset):
    def __init__(self, items, tiers):
        self.items, self.tiers = items, tiers

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        torch.set_num_threads(1)
        pi, vi = self.items[k]
        yp = YP[pi]
        dns = read_dns(yp, vi)
        xs = np.stack([make_input(dns, yp, t) for t in self.tiers]) if self.tiers else np.zeros((0,), np.float32)
        return torch.from_numpy(dns), torch.from_numpy(xs), pi, vi


def registry():
    M = {1: [], 2: [], 3: []}

    def add(t, name, kind, pt, js):
        M[t].append(dict(name=name, kind=kind, pt=pt, js=js if (js and os.path.exists(js)) else None))
    Y = "yasu"
    # Tier 1: masking reference + every spiked final-epoch checkpoint
    add(1, "v4e_s42", Y, os.path.join(RF, "yasunet_v4e_tier1_seed42", "best_charb.pt"), os.path.join(JC, "yasunet_v4e_tier1_seed42_summary.json"))
    add(1, "v4d_s44", Y, os.path.join(RN, "yasunet_v4d_tier1_seed44", "best.pt"), os.path.join(JN, "yasunet_v4d_tier1_seed44_summary.json"))
    for v in ("v4f12", "v4clo", "v4c"):
        add(1, f"{v}_s42", Y, os.path.join(RF, f"yasunet_{v}_tier1_seed42", "best.pt"), os.path.join(JF, f"yasunet_{v}_tier1_seed42_summary.json"))
        for s in (43, 44):
            add(1, f"{v}_s{s}", Y, os.path.join(RN, f"yasunet_{v}_tier1_seed{s}", "best.pt"), os.path.join(JN, f"yasunet_{v}_tier1_seed{s}_summary.json"))
    # Tiers 2 and 3: every Tables II-IV checkpoint (v4e Charbonnier-selected, v3 and baselines best.pt), plus extras
    for t in (2, 3):
        for s in (42, 43, 44):
            add(t, f"v4e_s{s}", Y, os.path.join(RF, f"yasunet_v4e_tier{t}_seed{s}", "best_charb.pt"), os.path.join(JC, f"yasunet_v4e_tier{t}_seed{s}_summary.json"))
        for s in (42, 43, 44):
            add(t, f"v3_s{s}", Y, os.path.join(RF, f"yasunet_v3_tier{t}_seed{s}", "best.pt"), os.path.join(JF, f"yasunet_v3_tier{t}_seed{s}_summary.json"))
        for b in BASE:
            for s in (42, 43, 44):
                add(t, f"{b}_s{s}", b, os.path.join(RB, f"{b}_tier{t}_seed{s}", "best.pt"), os.path.join(JB, f"{b}_tier{t}_seed{s}_summary.json"))
    add(2, "v4c_s42", Y, os.path.join(RF, "yasunet_v4c_tier2_seed42", "best.pt"), os.path.join(JF, "yasunet_v4c_tier2_seed42_summary.json"))
    for s, ep in ((42, 7), (43, 19), (44, 34)):
        add(3, f"v4e_s{s}_ep{ep}", Y, os.path.join(RF, f"yasunet_v4e_tier3_seed{s}", "best.pt"), os.path.join(JF, f"yasunet_v4e_tier3_seed{s}_summary.json"))
    for s in (42, 43):
        add(3, f"v4c_s{s}", Y, os.path.join(RN, f"yasunet_v4c_tier3_seed{s}", "best.pt"), os.path.join(JN, f"yasunet_v4c_tier3_seed{s}_summary.json"))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)          # >0: smoke test on the first N audit samples of each plane
    a = ap.parse_args()
    from yasunet_v3 import YASUNetV2
    from baselines_pkg import build_baseline
    from scipy import ndimage
    os.makedirs(OUT, exist_ok=True)
    TAG = "p1" if a.limit == 0 else f"p1_smoke{a.limit}"
    LOG = open(os.path.join(OUT, f"log_{TAG}.txt"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(y) for y in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    full, sub = eval_set()
    M = registry()
    tiers = [1, 2, 3]
    items = [(pi, vi) for pi in range(4) for vi in full[YP[pi]] if vi in sub[YP[pi]]]
    if a.limit:
        items = [it for pi in range(4) for it in [x for x in items if x[0] == pi][:a.limit]]
    log(f"{len(items)} audit-subset samples; models per tier {[len(M[t]) for t in tiers]}")

    MU = torch.tensor([[STATS[f"{v}_{yp}"]["mean"] for v in VARS] for yp in YP], dtype=torch.float32, device=dev)
    SD = torch.tensor([[STATS[f"{v}_{yp}"]["std"] for v in VARS] for yp in YP], dtype=torch.float32, device=dev)

    # ---- pass 0: DNS envelope over the audit subset (as spiketest.py / part1.py) ----
    t0 = time.time()
    dl0 = torch.utils.data.DataLoader(EvalDS(items, []), batch_size=a.batch, num_workers=a.workers)
    emin = np.full((4, 3), np.inf); emax = np.full((4, 3), -np.inf)
    for dns, _, pis, _ in dl0:
        d = dns.numpy().reshape(dns.shape[0], 3, -1)
        for i, pi in enumerate(pis.tolist()):
            emin[pi] = np.minimum(emin[pi], d[i].min(1)); emax[pi] = np.maximum(emax[pi], d[i].max(1))
    s_np, m_np = SD.cpu().numpy(), MU.cpu().numpy()
    for pi, yp in enumerate(YP):
        log(f"DNS envelope {yp}: sigma units lo {np.round((emin[pi]-m_np[pi])/s_np[pi],1)} hi {np.round((emax[pi]-m_np[pi])/s_np[pi],1)}")
    log(f"pass 0 done in {time.time()-t0:.0f}s")
    LO = torch.tensor(emin, dtype=torch.float32, device=dev) - 3 * SD
    HI = torch.tensor(emax, dtype=torch.float32, device=dev) + 3 * SD

    models, meta = {}, {}
    for t in tiers:
        for m in M[t]:
            ck = torch.load(m["pt"], map_location="cpu", weights_only=False)
            net = (YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6)
                   if m["kind"] == "yasu" else build_baseline(m["kind"], in_channels=3, out_channels=3))
            net.load_state_dict(ck["model_state"]); net.eval().to(dev)
            je = json.load(open(m["js"])).get("epoch") if m["js"] else None
            models[(t, m["name"])] = net
            meta[f"T{t}|{m['name']}"] = dict(pt=m["pt"], js=m["js"], epoch=ck.get("epoch"), json_epoch=je)
            log(f"loaded T{t} {m['name']:<18} epoch {ck.get('epoch')} (json epoch {je})  <- {m['pt']}")
            del ck

    k1 = torch.fft.fftfreq(512, d=1 / 512).to(dev)
    KZ, KX = torch.meshgrid(k1, k1, indexing="ij")
    binid = torch.floor(torch.sqrt(KX ** 2 + KZ ** 2) + 0.5).long().flatten()
    vidx = torch.nonzero(binid <= 256).squeeze(1); vbin = binid[vidx]

    def spectra(f):
        B = f.shape[0]
        Fh = torch.fft.fft2(f.double())
        Pw = ((Fh.real ** 2 + Fh.imag ** 2) / float(512 * 512) ** 2).reshape(B * 3, -1)[:, vidx]
        Eo = torch.zeros(B * 3, 257, dtype=torch.float64, device=dev); Eo.index_add_(1, vbin, Pw)
        return Eo.reshape(B, 3, 257)

    def eerr(Ep, Et):
        return (Ep - Et).abs().sum(-1) / Et.sum(-1)

    def nrmse(p, t):
        return torch.sqrt(((p - t) ** 2).mean(dim=(1, 2, 3))) / (t.reshape(t.shape[0], -1).std(dim=1, unbiased=False) + 1e-8)

    def dilate(m, r=2):
        for _ in range(r):
            m = F.max_pool2d(F.pad(m, (1, 1, 1, 1), mode="circular"), 3, stride=1)
        return m

    def pgrad(f, dim):            # trainer_yasunet_v4c._periodic_grad
        return (torch.roll(f, shifts=-1, dims=dim) - torch.roll(f, shifts=1, dims=dim)) * 0.5

    def charb_mean(pn, tn):
        return torch.sqrt((pn - tn) ** 2 + EPS ** 2).mean(dim=(1, 2, 3))

    def div_l(pn):
        return ((pgrad(pn[:, 0], -1) + pgrad(pn[:, 2], -2)) ** 2).mean(dim=(1, 2))

    def grad_l(pn, tn):
        lx = torch.sqrt((pgrad(pn, -1) - pgrad(tn, -1)) ** 2 + EPS ** 2).mean(dim=(1, 2, 3))
        lz = torch.sqrt((pgrad(pn, -2) - pgrad(tn, -2)) ** 2 + EPS ** 2).mean(dim=(1, 2, 3))
        return 0.5 * (lx + lz)

    REC = {k: [] for k in meta}
    ACC_S = {k: np.zeros((4, 3, 257)) for k in FIG3}; ACC_V = {k: np.zeros((4, 3)) for k in FIG3}; ACC_N = np.zeros(4)
    dl = torch.utils.data.DataLoader(EvalDS(items, tiers), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    t1 = time.time(); nb = 0; done = 0

    def save():
        np.savez(os.path.join(OUT, f"records_{TAG}.npz"), cols=np.array(COLS), meta=json.dumps(meta),
                 env=json.dumps(dict(emin=emin.tolist(), emax=emax.tolist())),
                 fig3_keys=np.array(FIG3), fig3_S=np.stack([ACC_S[k] for k in FIG3]), fig3_V=np.stack([ACC_V[k] for k in FIG3]), fig3_N=ACC_N,
                 **{k: (np.concatenate(v) if v else np.zeros((0, len(COLS)))) for k, v in REC.items()})

    with torch.no_grad():
        for dns, xs, pis, vis in dl:
            B = dns.shape[0]
            t = dns.to(dev, non_blocking=True)
            pis_t = pis.to(dev); mu = MU[pis_t][:, :, None, None]; sd = SD[pis_t][:, :, None, None]
            lo = LO[pis_t][:, :, None, None]; hi = HI[pis_t][:, :, None, None]
            ypv = torch.tensor([YPV[i] for i in pis.tolist()], device=dev)
            tn = (t - mu) / sd
            Et = spectra(t); Etn = spectra(tn)
            for pi in pis.tolist():
                ACC_N[pi] += 1
            for ti, tier in enumerate(tiers):
                xn = xs[:, ti].to(dev, non_blocking=True)
                xp = xn * sd + mu
                preds = {}
                for m in M[tier]:
                    net = models[(tier, m["name"])]
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = net(xn, ypv) if m["kind"] == "yasu" else net(xn)
                    yn = y.float(); preds[m["name"]] = (yn, yn * sd + mu)
                refn, refp = preds[M[tier][0]["name"]]
                keep = {}
                for m in M[tier]:
                    yn, p = preds[m["name"]]
                    vo = (p < lo) | (p > hi)
                    anyvo = vo.any(dim=1, keepdim=True).float()
                    Rm = dilate(anyvo) > 0
                    Rf = Rm.float()
                    has = Rm.flatten(1).any(1)
                    fp = {"": p, "_rep": torch.where(Rm, refp, p), "_in": torch.where(Rm, xp, p)}
                    fn = {"": yn, "_rep": torch.where(Rm, refn, yn), "_in": torch.where(Rm, xn, yn)}
                    Ep0, Epn0 = spectra(p), spectra(yn)
                    cols = {}
                    for f in FILLS:
                        if f == "" or not bool(has.any()):
                            Ep, Epn = Ep0, Epn0
                        else:
                            Ep, Epn = Ep0.clone(), Epn0.clone()
                            Ep[has] = spectra(fp[f][has]); Epn[has] = spectra(fn[f][has])
                        cols["nrmse" + f] = nrmse(fp[f], t)
                        e3 = eerr(Ep, Et)
                        for c, cc in enumerate("uvw"):
                            cols[f"e{cc}{f}"] = e3[:, c]
                        cols["spec" + f] = eerr(Epn, Etn).mean(1)
                        cols["charb" + f] = charb_mean(fn[f], tn)
                        cols["div" + f] = div_l(fn[f])
                        cols["grad" + f] = grad_l(fn[f], tn)
                    en = yn - tn; epp = p - t
                    ch = torch.sqrt(en ** 2 + EPS ** 2)
                    cols["sen_tot"] = (en ** 2).sum(dim=(1, 2, 3)); cols["sen_reg"] = (en ** 2 * Rf).sum(dim=(1, 2, 3))
                    cols["sep_tot"] = (epp ** 2).sum(dim=(1, 2, 3)); cols["sep_reg"] = (epp ** 2 * Rf).sum(dim=(1, 2, 3))
                    cols["ch_tot"] = ch.sum(dim=(1, 2, 3)); cols["ch_reg"] = (ch * Rf).sum(dim=(1, 2, 3))
                    Mo = 1.0 - Rf; no = Mo.sum(dim=(2, 3))

                    def vout(f):
                        mo = (f * Mo).sum(dim=(2, 3)) / no
                        return (((f - mo[:, :, None, None]) ** 2) * Mo).sum(dim=(2, 3)) / no
                    vm, vr, vd = vout(p), vout(refp), vout(t)
                    fl2 = (p - p.mean(dim=(2, 3), keepdim=True)) ** 2
                    va = fl2.mean(dim=(2, 3)); esh = (fl2 * Rf).sum(dim=(2, 3)) / fl2.sum(dim=(2, 3))
                    z = (p - mu) / sd
                    zhi, zlo = z.amax(dim=(2, 3)), z.amin(dim=(2, 3))
                    for c, cc in enumerate("uvw"):
                        cols[f"vout_m_{cc}"] = vm[:, c]; cols[f"vout_ref_{cc}"] = vr[:, c]; cols[f"vout_dns_{cc}"] = vd[:, c]
                        cols[f"vall_m_{cc}"] = va[:, c]; cols[f"esh_{cc}"] = esh[:, c]
                        cols[f"vo_{cc}"] = vo[:, c].sum(dim=(1, 2)).double()
                        cols[f"zhi_{cc}"] = zhi[:, c]; cols[f"zlo_{cc}"] = zlo[:, c]
                    cols["vo_any"] = anyvo.sum(dim=(1, 2, 3)); cols["region_px"] = Rf.sum(dim=(1, 2, 3))
                    nbl = np.zeros(B, np.float64)
                    am = anyvo[:, 0].bool(); hv = am.flatten(1).any(1).cpu().numpy()
                    if hv.any():
                        amc = am.cpu().numpy()
                        for i in np.nonzero(hv)[0]:
                            nbl[i] = ndimage.label(amc[i], structure=np.ones((3, 3)))[1]
                    cols["blobs"] = torch.from_numpy(nbl).to(dev)
                    cols["vi"] = vis.to(dev).double(); cols["plane"] = pis_t.double()
                    REC[f"T{tier}|{m['name']}"].append(torch.stack([cols[k].double() for k in COLS], 1).cpu().numpy())
                    if tier == 2 and m["name"] == "v4c_s42":
                        keep = {"v4c_s42": fp[""], "v4c_s42_rep": fp["_rep"], "v4c_s42_in": fp["_in"]}
                if tier == 2:
                    fl = {"DNS": t, "kim_s42": preds["kim_s42"][1], "v3_s42": preds["v3_s42"][1], "v4e_s42": refp, **keep}
                    for key in FIG3:
                        Es = spectra(fl[key]).cpu().numpy()
                        vv = ((fl[key] - fl[key].mean(dim=(2, 3), keepdim=True)) ** 2).mean(dim=(2, 3)).double().cpu().numpy()
                        for i, pi in enumerate(pis.tolist()):
                            ACC_S[key][pi] += Es[i]; ACC_V[key][pi] += vv[i]
            nb += 1; done += B
            if nb % 20 == 0:
                el = time.time() - t1
                log(f"  {done}/{len(items)} samples  {el:.0f}s elapsed  ETA {el / done * (len(items) - done) / 60:.1f} min")
            if nb % 40 == 0:
                save()
    save()
    log(f"pass 1 done in {time.time()-t1:.0f}s -> {OUT}")

    # validation: yp100 uses all 291 samples in both the audit subset and the evaluation set, so it must reproduce exactly
    ci = {c: COLS.index(c) for c in COLS}
    for key, mm in meta.items():
        if not mm["js"] or not REC[key]:
            continue
        r = np.concatenate(REC[key]); js = json.load(open(mm["js"]))
        parts = []
        for pi, yp in enumerate(YP):
            s = r[r[:, ci["plane"]] == pi]
            n1 = s[:, ci["nrmse"]].mean(); e1 = s[:, [ci["eu"], ci["ev"], ci["ew"]]].mean(1).mean()
            parts.append(f"{yp}(n={len(s)}) N {n1:.5f}/{js['per_yp'][yp]['nrmse']:.5f} E {e1:.5f}/{js['per_yp'][yp]['spectral_error']:.5f}")
        s = r[r[:, ci["plane"]] == 3]
        d100 = max(abs(s[:, ci["nrmse"]].mean() - js["per_yp"]["yp100"]["nrmse"]),
                   abs(s[:, [ci["eu"], ci["ev"], ci["ew"]]].mean(1).mean() - js["per_yp"]["yp100"]["spectral_error"]))
        log(f"VALIDATE {key:<22} epoch {mm['epoch']}/{mm['json_epoch']} | " + " | ".join(parts) + f" | yp100 |diff| {d100:.2e}")


if __name__ == "__main__":
    main()
