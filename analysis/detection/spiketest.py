# spiketest.py - inference-only artifact test, reproducing the prior audit's method (part1.py) exactly:
#   inputs rebuilt from raw DNS as the cache pipeline does (reader -> degrade(training=False) -> fp16 -> normalize -> fp16),
#   fp16 autocast inference, outlier = beyond the DNS envelope (over the 1,233-sample audit subset) +/- 3 sigma,
#   region = outliers (any component) dilated 2 px (circular 3x3 max-pool twice), masked = region replaced by v4e s42.
# Extensions: runs the FULL 5,000-sample evaluation set (so every checkpoint can be checked against its summary JSON),
#   records per-component E_err, per-sample extreme values in sigma units, and the DNS k=0 spectral share.
# Read-only for all user files; writes only to <scratchpad>/spike_out.
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

COLS = ["vi", "plane", "insub", "nrmse", "nrmse_rep", "eu", "ev", "ew", "eu_rep", "ev_rep", "ew_rep",
        "vo_u", "vo_v", "vo_w", "vo_any", "region_px", "blobs",
        "zhi_u", "zhi_v", "zhi_w", "zlo_u", "zlo_v", "zlo_w"]
DCOLS = ["vi", "plane", "insub", "k0_u", "k0_v", "k0_w", "dzhi_u", "dzhi_v", "dzhi_w", "dzlo_u", "dzlo_v", "dzlo_w"]


def eval_set():
    n = IV["total_snapshots"]
    entries = [(yp, i) for yp in YP for i in range(n)]
    sel = entries[::max(1, len(entries) // 5000)][:5000]          # identical to evaluate_cache.py
    full = {yp: [i for (y, i) in sel if y == yp] for yp in YP}
    sub = {yp: set(full[yp][::(1 if yp == "yp100" else 5)]) for yp in YP}   # prior audit: strides 5/1
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
        self.items, self.tiers = items, tiers                          # items: (plane_idx, vi, insub)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        torch.set_num_threads(1)
        pi, vi, insub = self.items[k]
        yp = YP[pi]
        dns = read_dns(yp, vi)
        xs = np.stack([make_input(dns, yp, t) for t in self.tiers]) if self.tiers else np.zeros((0,), np.float32)
        return torch.from_numpy(dns), torch.from_numpy(xs), pi, vi, insub


def registry():
    M = {1: [], 3: []}

    def add(t, name, kind, pt, js, sub_only=False):
        M[t].append(dict(name=name, kind=kind, pt=pt, js=js, sub_only=sub_only))
    Y = "yasu"
    # ---- Tier 1 ----  (first entry = masking reference, as in the prior audit)
    add(1, "v4e_s42", Y, os.path.join(RF, "yasunet_v4e_tier1_seed42", "best_charb.pt"), os.path.join(JC, "yasunet_v4e_tier1_seed42_summary.json"))
    add(1, "v4e_s43", Y, os.path.join(RF, "yasunet_v4e_tier1_seed43", "best_charb.pt"), os.path.join(JC, "yasunet_v4e_tier1_seed43_summary.json"))
    add(1, "v4e_s44", Y, os.path.join(RF, "yasunet_v4e_tier1_seed44", "best_charb.pt"), os.path.join(JC, "yasunet_v4e_tier1_seed44_summary.json"))
    add(1, "v4e_s43_ep50", Y, os.path.join(RF, "yasunet_v4e_tier1_seed43", "latest.pt"), os.path.join(JF, "yasunet_v4e_tier1_seed43_summary.json"))
    for s in (42, 43, 44):
        add(1, f"v3_s{s}", Y, os.path.join(RF, f"yasunet_v3_tier1_seed{s}", "best.pt"), os.path.join(JF, f"yasunet_v3_tier1_seed{s}_summary.json"))
    for v in ("v4d", "v4f12", "v4clo", "v4c"):
        add(1, f"{v}_s42", Y, os.path.join(RF, f"yasunet_{v}_tier1_seed42", "best.pt"), os.path.join(JF, f"yasunet_{v}_tier1_seed42_summary.json"))
        for s in (43, 44):
            add(1, f"{v}_s{s}", Y, os.path.join(RN, f"yasunet_{v}_tier1_seed{s}", "best.pt"), os.path.join(JN, f"yasunet_{v}_tier1_seed{s}_summary.json"))
    # pre-transition (Charbonnier-selected) checkpoints: audit subset only
    add(1, "v4d_s44_ep35", Y, os.path.join(RN, "yasunet_v4d_tier1_seed44", "best_charb.pt"), None, True)
    add(1, "v4f12_s43_ep15", Y, os.path.join(RN, "yasunet_v4f12_tier1_seed43", "best_charb.pt"), None, True)
    add(1, "v4f12_s44_ep15", Y, os.path.join(RN, "yasunet_v4f12_tier1_seed44", "best_charb.pt"), None, True)
    add(1, "v4clo_s44_ep34", Y, os.path.join(RN, "yasunet_v4clo_tier1_seed44", "best_charb.pt"), None, True)
    add(1, "v4c_s43_ep12", Y, os.path.join(RN, "yasunet_v4c_tier1_seed43", "best_charb.pt"), None, True)
    add(1, "v4c_s44_ep14", Y, os.path.join(RN, "yasunet_v4c_tier1_seed44", "best_charb.pt"), None, True)
    for b in ("kim", "fukami_cnn", "fukami_dscms", "guastoni"):
        for s in (42, 43, 44):
            add(1, f"{b}_s{s}", b, os.path.join(RB, f"{b}_tier1_seed{s}", "best.pt"), os.path.join(JB, f"{b}_tier1_seed{s}_summary.json"))
    # ---- Tier 3 ----
    add(3, "v4e_s42", Y, os.path.join(RF, "yasunet_v4e_tier3_seed42", "best_charb.pt"), os.path.join(JC, "yasunet_v4e_tier3_seed42_summary.json"))
    add(3, "v4c_s42", Y, os.path.join(RN, "yasunet_v4c_tier3_seed42", "best.pt"), os.path.join(JN, "yasunet_v4c_tier3_seed42_summary.json"))
    add(3, "v4c_s43", Y, os.path.join(RN, "yasunet_v4c_tier3_seed43", "best.pt"), os.path.join(JN, "yasunet_v4c_tier3_seed43_summary.json"))
    add(3, "v4c_s42_oldrun", Y, os.path.join(RF, "yasunet_v4c_tier3_seed42", "best.pt"), os.path.join(JF, "yasunet_v4c_tier3_seed42_summary.json"))
    add(3, "v4c_s42_ep14", Y, os.path.join(RN, "yasunet_v4c_tier3_seed42", "best_charb.pt"), None, True)
    add(3, "v4c_s43_ep15", Y, os.path.join(RN, "yasunet_v4c_tier3_seed43", "best_charb.pt"), None, True)
    return M


A_SET = {1: {"v4d_s43", "v4d_s44", "v4f12_s43", "v4f12_s44", "v4clo_s43", "v4clo_s44", "v4c_s43", "v4c_s44",
             "v4d_s44_ep35", "v4f12_s43_ep15", "v4f12_s44_ep15", "v4clo_s44_ep34", "v4c_s43_ep12", "v4c_s44_ep14"},
         3: {"v4c_s42", "v4c_s43", "v4c_s42_ep14", "v4c_s43_ep15"}}
# Job B (trimmed): full set for the seed-42 sweep variants, v3 s42, v4e final-epoch seeds 43/44 and all baselines;
# audit subset only for the remaining v3/v4e seeds; no Tier-3 models (the old Tier-3 run was already tested by the audit).
B_FULL = {1: {"v4d_s42", "v4f12_s42", "v4clo_s42", "v4c_s42", "v3_s42", "v4e_s43_ep50", "v4e_s44"}
             | {f"{b}_s{s}" for b in ("kim", "fukami_cnn", "fukami_dscms", "guastoni") for s in (42, 43, 44)}, 3: set()}
B_SUB = {1: {"v3_s43", "v3_s44", "v4e_s43"}, 3: set()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["A", "B", "all"])
    ap.add_argument("--mode", default="quick", choices=["quick", "full"])
    ap.add_argument("--models", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from yasunet_v3 import YASUNetV2
    from baselines_pkg import build_baseline
    from scipy import ndimage
    os.makedirs(OUT, exist_ok=True)
    TAG = a.mode if a.set == "all" else f"{a.mode}_{a.set}"
    LOG = open(os.path.join(OUT, f"log_{TAG}.txt"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(y) for y in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    full, sub = eval_set()
    M = registry()
    if a.mode == "quick":
        keep = set(a.models.split(",")) if a.models else set()
        M = {t: [m for m in lst if m["name"] in keep or (m is lst[0])] for t, lst in M.items()}
        M = {t: lst for t, lst in M.items() if any(m["name"] in keep for m in lst)}
        items = [(3, vi, True) for vi in full["yp100"]]
    else:
        if a.set == "A":         # the masking reference (first entry per tier) is always kept
            M = {t: [m for i, m in enumerate(lst) if i == 0 or m["name"] in A_SET[t]] for t, lst in M.items()}
        elif a.set == "B":
            M = {t: [dict(m, sub_only=(m["sub_only"] or m["name"] in B_SUB[t])) for i, m in enumerate(lst)
                     if i == 0 or m["name"] in (B_FULL[t] | B_SUB[t])] for t, lst in M.items()}
        M = {t: lst for t, lst in M.items() if len(lst) > 1}
        items = [(pi, vi, vi in sub[YP[pi]]) for pi in range(4) for vi in full[YP[pi]]]
    tiers = sorted(M)
    log(f"mode {a.mode}: {len(items)} samples; tiers {tiers}; models per tier {[len(M[t]) for t in tiers]}")

    MU = torch.tensor([[STATS[f"{v}_{yp}"]["mean"] for v in VARS] for yp in YP], dtype=torch.float32, device=dev)   # [4,3]
    SD = torch.tensor([[STATS[f"{v}_{yp}"]["std"] for v in VARS] for yp in YP], dtype=torch.float32, device=dev)

    # ---- pass 0: DNS envelope over the audit subset (exactly as part1.py) ----
    if a.mode == "full":
        t0 = time.time()
        sub_items = [(pi, vi, True) for pi in range(4) for vi in full[YP[pi]] if vi in sub[YP[pi]]]
        dl0 = torch.utils.data.DataLoader(EvalDS(sub_items, []), batch_size=a.batch, num_workers=a.workers)
        emin = np.full((4, 3), np.inf); emax = np.full((4, 3), -np.inf)
        for dns, _, pis, _, _ in dl0:
            d = dns.numpy().reshape(dns.shape[0], 3, -1)
            for i, pi in enumerate(pis.tolist()):
                emin[pi] = np.minimum(emin[pi], d[i].min(1)); emax[pi] = np.maximum(emax[pi], d[i].max(1))
        s_np, m_np = SD.cpu().numpy(), MU.cpu().numpy()
        for pi, yp in enumerate(YP):
            log(f"DNS envelope {yp}: sigma units lo {np.round((emin[pi]-m_np[pi])/s_np[pi],1)} hi {np.round((emax[pi]-m_np[pi])/s_np[pi],1)}  (n={len(sub[yp])})")
        log(f"pass 0 done in {time.time()-t0:.0f}s")
        LO = torch.tensor(emin, dtype=torch.float32, device=dev) - 3 * SD
        HI = torch.tensor(emax, dtype=torch.float32, device=dev) + 3 * SD
    else:
        emin = emax = None
        LO = torch.full((4, 3), -1e30, device=dev); HI = torch.full((4, 3), 1e30, device=dev)

    # ---- models ----
    models, meta = {}, {}
    for t in tiers:
        for m in M[t]:
            ck = torch.load(m["pt"], map_location="cpu", weights_only=False)
            net = (YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6)
                   if m["kind"] == "yasu" else build_baseline(m["kind"], in_channels=3, out_channels=3))
            net.load_state_dict(ck["model_state"]); net.eval().to(dev)
            je = json.load(open(m["js"]))["epoch"] if m["js"] and os.path.exists(m["js"]) else None
            models[(t, m["name"])] = net
            meta[f"T{t}|{m['name']}"] = dict(pt=m["pt"], js=m["js"], epoch=ck.get("epoch"), json_epoch=je, sub_only=m["sub_only"])
            log(f"loaded T{t} {m['name']:<16} epoch {ck.get('epoch')} (json epoch {je})  <- {m['pt']}")
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

    REC = {k: [] for k in meta}; DREC = []
    ds = EvalDS(items, tiers)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    t1 = time.time(); nb = 0; done = 0

    def save():
        np.savez(os.path.join(OUT, f"records_{TAG}.npz"), cols=np.array(COLS), dcols=np.array(DCOLS),
                 meta=json.dumps(meta), env=json.dumps(dict(emin=None if emin is None else emin.tolist(), emax=None if emax is None else emax.tolist())),
                 DNS=np.concatenate(DREC) if DREC else np.zeros((0, len(DCOLS))),
                 **{k: (np.concatenate(v) if v else np.zeros((0, len(COLS)))) for k, v in REC.items()})

    with torch.no_grad():
        for dns, xs, pis, vis, insub in dl:
            B = dns.shape[0]
            t = dns.to(dev, non_blocking=True)
            pis_t = pis.to(dev); mu = MU[pis_t][:, :, None, None]; sd = SD[pis_t][:, :, None, None]
            lo = LO[pis_t][:, :, None, None]; hi = HI[pis_t][:, :, None, None]
            ypv = torch.tensor([YPV[i] for i in pis.tolist()], device=dev)
            Et = spectra(t)
            dz = (t - mu) / sd
            DREC.append(torch.cat([vis.to(dev).double()[:, None], pis_t.double()[:, None], insub.to(dev).double()[:, None],
                                   (Et[:, :, 0] / Et.sum(-1)), dz.amax(dim=(2, 3)).double(), dz.amin(dim=(2, 3)).double()], 1).cpu().numpy())
            subm = insub.to(dev).bool()
            for ti, tier in enumerate(tiers):
                x = xs[:, ti].to(dev, non_blocking=True)
                preds = {}
                for m in M[tier]:
                    rows = subm if m["sub_only"] else torch.ones(B, dtype=torch.bool, device=dev)
                    if not bool(rows.any()):
                        continue
                    net = models[(tier, m["name"])]
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = net(x[rows], ypv[rows]) if m["kind"] == "yasu" else net(x[rows])
                    preds[m["name"]] = (rows, y.float() * sd[rows] + mu[rows])
                ref_rows, ref_all = preds[M[tier][0]["name"]]
                for m in M[tier]:
                    if m["name"] not in preds:
                        continue
                    rows, p = preds[m["name"]]
                    tt, Ett, ref = t[rows], Et[rows], ref_all[rows]
                    vo = (p < lo[rows]) | (p > hi[rows])
                    anyvo = vo.any(dim=1, keepdim=True).float()
                    Rm = dilate(anyvo) > 0
                    Ep = spectra(p)
                    has = Rm.flatten(1).any(1)
                    prep = torch.where(Rm, ref, p)
                    Er = Ep.clone()
                    if bool(has.any()):
                        Er[has] = spectra(prep[has])
                    nbl = np.zeros(int(rows.sum()), np.float32)
                    am = anyvo[:, 0].bool()
                    hv = am.flatten(1).any(1).cpu().numpy()
                    if hv.any():
                        amc = am.cpu().numpy()
                        for i in np.nonzero(hv)[0]:
                            nbl[i] = ndimage.label(amc[i], structure=np.ones((3, 3)))[1]
                    z = (p - mu[rows]) / sd[rows]
                    rec = torch.cat([vis.to(dev)[rows].double()[:, None], pis_t[rows].double()[:, None], insub.to(dev)[rows].double()[:, None],
                                     nrmse(p, tt).double()[:, None], nrmse(prep, tt).double()[:, None],
                                     eerr(Ep, Ett), eerr(Er, Ett),
                                     vo.sum(dim=(2, 3)).double(), anyvo.sum(dim=(1, 2, 3)).double()[:, None], Rm.float().sum(dim=(1, 2, 3)).double()[:, None],
                                     torch.from_numpy(nbl).to(dev).double()[:, None],
                                     z.amax(dim=(2, 3)).double(), z.amin(dim=(2, 3)).double()], 1).cpu().numpy()
                    REC[f"T{tier}|{m['name']}"].append(rec)
            nb += 1; done += B
            if nb % 25 == 0:
                el = time.time() - t1
                log(f"  {done}/{len(items)} samples  {el:.0f}s elapsed  ETA {el / done * (len(items) - done) / 60:.1f} min")
            if nb % 100 == 0:
                save()
    save()
    log(f"pass 1 done in {time.time()-t1:.0f}s -> {OUT}")

    # ---- validation printout: per-plane means vs summary JSON ----
    for key, mm in meta.items():
        if not mm["js"] or not REC[key]:
            continue
        r = np.concatenate(REC[key]); js = json.load(open(mm["js"]))
        rows = []
        for pi, yp in enumerate(YP):
            s = r[r[:, 1] == pi]
            if len(s) == 0:
                continue
            n1 = s[:, COLS.index("nrmse")].mean(); e1 = s[:, [COLS.index(c) for c in ("eu", "ev", "ew")]].mean(1).mean()
            n0 = js["per_yp"][yp]["nrmse"]; e0 = js["per_yp"][yp]["spectral_error"]
            rows.append((yp, len(s), n1, n0, e1, e0))
        dmax = max(max(abs(x[2] - x[3]), abs(x[4] - x[5])) for x in rows)
        log(f"VALIDATE {key:<22} epoch {mm['epoch']}/{mm['json_epoch']} | " + " | ".join(f"{yp}(n={n}) N {n1:.5f}/{n0:.5f} E {e1:.5f}/{e0:.5f}" for yp, n, n1, n0, e1, e0 in rows) + f" | max|diff| {dmax:.2e}")


if __name__ == "__main__":
    main()
