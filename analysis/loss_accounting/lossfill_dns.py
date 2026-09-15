# lossfill_dns.py - the loss accounting of Section IV-D with a third, strictly stronger fill: the DNS itself.
#
# Derived from phase1_checks.py (the script behind the published loss accounting) and restricted to Tier 1:
#   same 1,233-sample audit subset, same input rebuild (reader -> degrade(training=False) -> fp16 -> normalise -> fp16),
#   same fp16-autocast inference, same Section III-H rule (audit-subset DNS envelope +- 3 sigma_train, any component,
#   dilated 2 px by two periodic 3x3 max-pools), same loss terms (trainer_yasunet_v4c.py), same checkpoints.
# Fills of the spike region:  ""  as predicted | "_rep" v4e seed-42 prediction | "_in" degraded input | "_dns" the DNS target
# The first three reproduce records_p1.npz and serve as the validation; "_dns" is new.
# Also counts flagged pixels on v4clo seed 43's epoch-34 (Charbonnier-selected, pre-onset) checkpoint.
# Inference only. Writes only to this script's directory.
import os, sys, json, time, argparse
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = r"E:\lift550_paper\code"
sys.path.insert(0, CODE)
import reader as R                                                     # noqa: E402
from degradation import degrade                                        # noqa: E402

P = r"C:\Users\Arnav Simha\Downloads\lift550_paper"
N = r"C:\Users\Arnav Simha\lift550_data"
D = r"D:\lift550_cache"
RAW = r"D:\Research\misc\data\train"
RF, RN = os.path.join(P, "runs_final"), os.path.join(N, "runs_newseeds")
STATS = json.load(open(os.path.join(P, "code", "norm_stats_train.json")))
VARS = ["uxz", "vxz", "wxz"]; YP = ["yp15", "yp30", "yp50", "yp100"]
YPV = [15.0, 30.0, 50.0, 100.0]
IV = json.load(open(os.path.join(D, "index_val.json")))
EPS = 1e-3
LAMBDA_DIV, LAMBDA_GRAD = 0.02, 0.05
FILLS = ("", "_rep", "_in", "_dns")
TERMS = ("spec", "charb", "div", "grad")
COLS = (["vi", "plane", "vo_any", "region_px", "zabs"]
        + [f"{q}{f}" for q in TERMS for f in FILLS]
        + ["sen_tot", "sen_reg", "sep_tot", "sep_reg", "ch_tot", "ch_reg"])
TERMINAL = {"v4d": 0.10, "v4f12": 0.12, "v4clo": 0.15, "v4c": 0.30}


def eval_set():
    n = IV["total_snapshots"]
    entries = [(yp, i) for yp in YP for i in range(n)]
    sel = entries[::max(1, len(entries) // 5000)][:5000]          # identical to evaluate_cache.py
    full = {yp: [i for (y, i) in sel if y == yp] for yp in YP}
    sub = {yp: set(full[yp][::(1 if yp == "yp100" else 5)]) for yp in YP}
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
    M = [dict(name="v4e_s42", pt=os.path.join(RF, "yasunet_v4e_tier1_seed42", "best_charb.pt"), lam=None)]   # masking reference
    M.append(dict(name="v4d_s44", pt=os.path.join(RN, "yasunet_v4d_tier1_seed44", "best.pt"), lam=TERMINAL["v4d"]))
    for v in ("v4f12", "v4clo", "v4c"):
        M.append(dict(name=f"{v}_s42", pt=os.path.join(RF, f"yasunet_{v}_tier1_seed42", "best.pt"), lam=TERMINAL[v]))
        for s in (43, 44):
            M.append(dict(name=f"{v}_s{s}", pt=os.path.join(RN, f"yasunet_{v}_tier1_seed{s}", "best.pt"), lam=TERMINAL[v]))
    M.append(dict(name="v4clo_s43_ep34", pt=os.path.join(RN, "yasunet_v4clo_tier1_seed43", "best_charb.pt"), lam=None))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--p1", required=True, help="records_p1.npz from phase1_checks.py, for validation")
    a = ap.parse_args()
    from yasunet_v3 import YASUNetV2
    LOG = open(os.path.join(HERE, "log_lossfill.txt"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(y) for y in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    full, sub = eval_set()
    M = registry()
    items = [(pi, vi) for pi in range(4) for vi in full[YP[pi]] if vi in sub[YP[pi]]]
    log(f"{len(items)} audit-subset samples; {len(M)} checkpoints")

    MU = torch.tensor([[STATS[f"{v}_{yp}"]["mean"] for v in VARS] for yp in YP], dtype=torch.float32, device=dev)
    SD = torch.tensor([[STATS[f"{v}_{yp}"]["std"] for v in VARS] for yp in YP], dtype=torch.float32, device=dev)

    t0 = time.time()
    dl0 = torch.utils.data.DataLoader(EvalDS(items, []), batch_size=a.batch, num_workers=a.workers)
    emin = np.full((4, 3), np.inf); emax = np.full((4, 3), -np.inf)
    for dns, _, pis, _ in dl0:
        d = dns.numpy().reshape(dns.shape[0], 3, -1)
        for i, pi in enumerate(pis.tolist()):
            emin[pi] = np.minimum(emin[pi], d[i].min(1)); emax[pi] = np.maximum(emax[pi], d[i].max(1))
    p1 = np.load(a.p1, allow_pickle=True)
    env1 = json.loads(str(p1["env"]))
    log(f"envelope identical to records_p1.npz: {np.array_equal(emin, np.array(env1['emin'])) and np.array_equal(emax, np.array(env1['emax']))}"
        f"  (pass 0 {time.time() - t0:.0f}s)")
    LO = torch.tensor(emin, dtype=torch.float32, device=dev) - 3 * SD
    HI = torch.tensor(emax, dtype=torch.float32, device=dev) + 3 * SD

    nets, meta = {}, {}
    for m in M:
        ck = torch.load(m["pt"], map_location="cpu", weights_only=False)
        net = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16, hidden_dim=128, modes_h=10, modes_w=10, n_refine=6)
        net.load_state_dict(ck["model_state"]); net.eval().to(dev)
        nets[m["name"]] = net
        meta[m["name"]] = dict(pt=m["pt"], epoch=ck.get("epoch"), lam=m["lam"])
        log(f"loaded {m['name']:<16} epoch {ck.get('epoch')}  <- {m['pt']}")
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

    def dilate(m, r=2):
        for _ in range(r):
            m = F.max_pool2d(F.pad(m, (1, 1, 1, 1), mode="circular"), 3, stride=1)
        return m

    def pgrad(f, dim):
        return (torch.roll(f, shifts=-1, dims=dim) - torch.roll(f, shifts=1, dims=dim)) * 0.5

    def charb_mean(pn, tn):
        return torch.sqrt((pn - tn) ** 2 + EPS ** 2).mean(dim=(1, 2, 3))

    def div_l(pn):
        return ((pgrad(pn[:, 0], -1) + pgrad(pn[:, 2], -2)) ** 2).mean(dim=(1, 2))

    def grad_l(pn, tn):
        lx = torch.sqrt((pgrad(pn, -1) - pgrad(tn, -1)) ** 2 + EPS ** 2).mean(dim=(1, 2, 3))
        lz = torch.sqrt((pgrad(pn, -2) - pgrad(tn, -2)) ** 2 + EPS ** 2).mean(dim=(1, 2, 3))
        return 0.5 * (lx + lz)

    REC = {m["name"]: [] for m in M}
    dl = torch.utils.data.DataLoader(EvalDS(items, [1]), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
    t1 = time.time(); nb = 0; done = 0
    with torch.no_grad():
        for dns, xs, pis, vis in dl:
            B = dns.shape[0]
            t = dns.to(dev, non_blocking=True)
            pis_t = pis.to(dev); mu = MU[pis_t][:, :, None, None]; sd = SD[pis_t][:, :, None, None]
            lo = LO[pis_t][:, :, None, None]; hi = HI[pis_t][:, :, None, None]
            ypv = torch.tensor([YPV[i] for i in pis.tolist()], device=dev)
            tn = (t - mu) / sd
            Etn = spectra(tn)
            xn = xs[:, 0].to(dev, non_blocking=True)
            preds = {}
            for m in M:
                with torch.autocast("cuda", dtype=torch.float16):
                    y = nets[m["name"]](xn, ypv)
                yn = y.float(); preds[m["name"]] = (yn, yn * sd + mu)
            refn, _ = preds[M[0]["name"]]
            for m in M:
                yn, p = preds[m["name"]]
                vo = (p < lo) | (p > hi)
                anyvo = vo.any(dim=1, keepdim=True).float()
                Rm = dilate(anyvo) > 0
                Rf = Rm.float()
                has = Rm.flatten(1).any(1)
                fn = {"": yn, "_rep": torch.where(Rm, refn, yn), "_in": torch.where(Rm, xn, yn), "_dns": torch.where(Rm, tn, yn)}
                Epn0 = spectra(yn)
                cols = {}
                for f in FILLS:
                    if f == "" or not bool(has.any()):
                        Epn = Epn0
                    else:
                        Epn = Epn0.clone(); Epn[has] = spectra(fn[f][has])
                    cols["spec" + f] = eerr(Epn, Etn).mean(1)
                    cols["charb" + f] = charb_mean(fn[f], tn)
                    cols["div" + f] = div_l(fn[f])
                    cols["grad" + f] = grad_l(fn[f], tn)
                en = yn - tn; epp = p - t
                ch = torch.sqrt(en ** 2 + EPS ** 2)
                cols["sen_tot"] = (en ** 2).sum(dim=(1, 2, 3)); cols["sen_reg"] = (en ** 2 * Rf).sum(dim=(1, 2, 3))
                cols["sep_tot"] = (epp ** 2).sum(dim=(1, 2, 3)); cols["sep_reg"] = (epp ** 2 * Rf).sum(dim=(1, 2, 3))
                cols["ch_tot"] = ch.sum(dim=(1, 2, 3)); cols["ch_reg"] = (ch * Rf).sum(dim=(1, 2, 3))
                cols["vo_any"] = anyvo.sum(dim=(1, 2, 3)); cols["region_px"] = Rf.sum(dim=(1, 2, 3))
                cols["zabs"] = ((p - mu) / sd).abs().amax(dim=(1, 2, 3))
                cols["vi"] = vis.to(dev).double(); cols["plane"] = pis_t.double()
                REC[m["name"]].append(torch.stack([cols[k].double() for k in COLS], 1).cpu().numpy())
            nb += 1; done += B
            if nb % 20 == 0:
                el = time.time() - t1
                log(f"  {done}/{len(items)} samples  {el:.0f}s  ETA {el / done * (len(items) - done) / 60:.1f} min")
    R_ = {k: np.concatenate(v) for k, v in REC.items()}
    np.savez(os.path.join(HERE, "lossfill_records.npz"), cols=np.array(COLS), meta=json.dumps(meta),
             env=json.dumps(dict(emin=emin.tolist(), emax=emax.tolist())), **R_)
    log(f"pass 1 done in {time.time() - t1:.0f}s")

    # ---- validation against records_p1.npz: fills "", "_rep", "_in" ----
    c1 = [str(c) for c in p1["cols"]]; i1 = {c: i for i, c in enumerate(c1)}; ci = {c: i for i, c in enumerate(COLS)}
    worst = 0.0; nkeys = 0
    for name, r in R_.items():
        key = f"T1|{name}"
        if key not in p1.files:
            continue
        q = p1[key]
        kq = {(int(x[i1["plane"]]), int(x[i1["vi"]])): x for x in q}
        for x in r:
            y = kq[(int(x[ci["plane"]]), int(x[ci["vi"]]))]
            for term in TERMS:
                for f in ("", "_rep", "_in"):
                    worst = max(worst, abs(x[ci[term + f]] - y[i1[term + f]]))
            worst = max(worst, abs(x[ci["vo_any"]] - y[i1["vo_any"]]), abs(x[ci["region_px"]] - y[i1["region_px"]]))
        nkeys += 1
    log(f"VALIDATION vs records_p1.npz ({nkeys} checkpoints, fills as-predicted / v4e / input, all loss terms and flag counts): "
        f"max abs deviation {worst:.3e}")

    # ---- the result ----
    out = {"validation_max_abs_dev": worst, "checkpoints": {}}
    log("")
    log("total training objective at each run's terminal weight, plane means; fraction of samples on which the spiked output is lower")
    for m in M:
        if m["lam"] is None:
            continue
        r = R_[m["name"]]; lam = m["lam"]
        tot = {f: r[:, ci["charb" + f]] + lam * r[:, ci["spec" + f]] + LAMBDA_DIV * r[:, ci["div" + f]] + LAMBDA_GRAD * r[:, ci["grad" + f]] for f in FILLS}
        rec = {"lambda": lam, "planes": {}}
        line = f"  {m['name']:<10} lam {lam:.2f} |"
        for pi, yp in enumerate(YP):
            s = r[:, ci["plane"]] == pi
            means = {f: float(tot[f][s].mean()) for f in FILLS}
            lower = {f: float(np.mean(tot[""][s] < tot[f][s])) for f in FILLS[1:]}
            specm = {f: float(r[s, ci["spec" + f]].mean()) for f in FILLS}
            rec["planes"][yp] = {"n": int(s.sum()), "total_mean": means, "frac_spiked_lower": lower, "spec_mean": specm}
            line += f" {yp} pred {means['']:.3f} v4e {means['_rep']:.3f} in {means['_in']:.3f} dns {means['_dns']:.3f} (lower on {100 * lower['_dns']:.0f}%) |"
        allf = {f: float(np.mean(tot[""] < tot[f])) for f in FILLS[1:]}
        rec["frac_spiked_lower_all_planes"] = allf
        rec["n_samples"] = int(len(r))
        out["checkpoints"][m["name"]] = rec
        log(line)
        log(f"  {'':<10} all planes: spiked output lower than  v4e-fill {100 * allf['_rep']:.1f}%  input-fill {100 * allf['_in']:.1f}%  DNS-fill {100 * allf['_dns']:.1f}%  of {len(r)} samples")
    ep = R_["v4clo_s43_ep34"]
    out["v4clo_s43_ep34"] = {"samples_flagged": int((ep[:, ci["vo_any"]] > 0).sum()), "n": int(len(ep)),
                             "max_flagged_px": int(ep[:, ci["vo_any"]].max()), "max_abs_z": float(ep[:, ci["zabs"]].max())}
    log(f"v4clo seed 43 epoch-34 checkpoint: {out['v4clo_s43_ep34']}")
    json.dump(out, open(os.path.join(HERE, "lossfill_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
