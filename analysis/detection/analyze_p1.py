# Read-only summaries of spike_out/records_p1.npz (phase1_checks.py) plus the Tier-1 spike-test records. Prints tables only.
import os, sys, json
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(SP, "spike_out")
TAG = sys.argv[1] if len(sys.argv) > 1 else "p1"
Z = np.load(os.path.join(OUT, f"records_{TAG}.npz"), allow_pickle=True)
cols = list(Z["cols"]); meta = json.loads(str(Z["meta"])); ci = {c: i for i, c in enumerate(cols)}
YP = ["yp15", "yp30", "yp50", "yp100"]; PX = 512 * 512
LAM = {"v4e": 0.07, "v4d": 0.10, "v4f12": 0.12, "v4clo": 0.15, "v4c": 0.30}
keys = [k for k in meta if k in Z.files]


def C(a, n): return a[:, ci[n]]
def planes(a): return [a[a[:, ci["plane"]] == pi] for pi in range(4)]
def pm(a, f):            # plane-mean convention: mean over planes of per-plane sample means
    return float(np.mean([f(s).mean() for s in planes(a)]))
def pp(a, f): return [float(f(s).mean()) for s in planes(a)]
def E(a, fill=""): return (C(a, f"eu{fill}") + C(a, f"ev{fill}") + C(a, f"ew{fill}")) / 3
def lam_of(key):
    n = key.split("|")[1].split("_")[0]
    return LAM.get(n)


print("=" * 170); print("A. VALIDATION: yp100 (all 291 samples in both sets) vs summary JSON; other planes are 1-in-5 subsets"); print("=" * 170)
for k in keys:
    mm = meta[k]
    if not mm["js"]: print(f" {k:<22} no JSON"); continue
    js = json.load(open(mm["js"])); a = Z[k]; s = planes(a)[3]
    n1, e1 = C(s, "nrmse").mean(), E(s).mean()
    n0, e0 = js["per_yp"]["yp100"]["nrmse"], js["per_yp"]["yp100"]["spectral_error"]
    print(f" {k:<22} ep {mm['epoch']}/{mm['json_epoch']}  yp100 N {n1:.5f}/{n0:.5f} E {e1:.5f}/{e0:.5f}  |d| {max(abs(n1-n0), abs(e1-e0)):.1e}  "
          f"-> {'REPRODUCES' if max(abs(n1-n0), abs(e1-e0)) < 6e-5 else 'CHECK'}")

print("\n" + "=" * 170); print("B. SPIKE STATUS (audit subset, 1,233 samples; plane-mean convention). masked = region <- v4e s42 | <- degraded input"); print("=" * 170)
print(f" {'checkpoint':<22} {'%smp':>6} {'%smp u/v/w':>16} {'out px%':>8} {'reg px%':>8} {'blobs':>6} {'max|z| u/v/w':>18} | {'NRMSE as-is / v4e-fill / input-fill':<34} | E_err as-is / v4e-fill / input-fill")
for k in keys:
    a = Z[k]
    f_any = 100 * np.mean(C(a, "vo_any") > 0); fc = [100 * np.mean(C(a, f"vo_{c}") > 0) for c in "uvw"]
    op = 100 * C(a, "vo_any").mean() / PX; rp = 100 * C(a, "region_px").mean() / PX
    bl = C(a, "blobs").mean()
    mz = [max(C(a, f"zhi_{c}").max(), -C(a, f"zlo_{c}").min()) for c in "uvw"]
    print(f" {k:<22} {f_any:>5.1f}% {'/'.join(f'{x:.0f}' for x in fc):>16} {op:>7.4f}% {rp:>7.3f}% {bl:>6.2f} {'/'.join(f'{x:.1f}' for x in mz):>18} | "
          f"{pm(a, lambda s: C(s,'nrmse')):.4f} / {pm(a, lambda s: C(s,'nrmse_rep')):.4f} / {pm(a, lambda s: C(s,'nrmse_in')):.4f}      | "
          f"{pm(a, lambda s: E(s)):.4f} / {pm(a, lambda s: E(s,'_rep')):.4f} / {pm(a, lambda s: E(s,'_in')):.4f}"
          f"   [u {pm(a, lambda s: C(s,'eu')):.4f} v {pm(a, lambda s: C(s,'ev')):.4f} w {pm(a, lambda s: C(s,'ew')):.4f}"
          f" -> masked u {pm(a, lambda s: C(s,'eu_rep')):.4f} v {pm(a, lambda s: C(s,'ev_rep')):.4f} w {pm(a, lambda s: C(s,'ew_rep')):.4f}]")
    if f_any > 0:
        print(f" {'':<22} per plane: NRMSE {[round(x,4) for x in pp(a, lambda s: C(s,'nrmse'))]} -> {[round(x,4) for x in pp(a, lambda s: C(s,'nrmse_rep'))]}"
              f" | E_err {[round(x,4) for x in pp(a, lambda s: E(s))]} -> {[round(x,4) for x in pp(a, lambda s: E(s,'_rep'))]} / input-fill {[round(x,4) for x in pp(a, lambda s: E(s,'_in'))]}")

print("\n" + "=" * 170); print("C. LOSS ACCOUNTING for checkpoints with a flagged region (normalized units unless stated; per-plane values, then plane mean)"); print("=" * 170)
for k in keys:
    a = Z[k]
    if np.mean(C(a, "vo_any") > 0) < 0.5: continue
    lam = lam_of(k)
    reg = pp(a, lambda s: C(s, "region_px") / PX * 100)
    shn = pp(a, lambda s: C(s, "sen_reg") / C(s, "sen_tot") * 100); shp = pp(a, lambda s: C(s, "sep_reg") / C(s, "sep_tot") * 100)
    shc = pp(a, lambda s: C(s, "ch_reg") / C(s, "ch_tot") * 100)
    print(f" {k}  (terminal lambda {lam})")
    print(f"   region % of pixels         {[round(x,3) for x in reg]}")
    print(f"   share of squared error  (normalized) {[round(x,1) for x in shn]}   (physical) {[round(x,1) for x in shp]}")
    print(f"   share of Charbonnier sum  {[round(x,2) for x in shc]}")
    for q in ("spec", "charb", "div", "grad"):
        print(f"   {q:<6} as-is {[round(x,4) for x in pp(a, lambda s: C(s,q))]}  v4e-fill {[round(x,4) for x in pp(a, lambda s: C(s,q+'_rep'))]}  input-fill {[round(x,4) for x in pp(a, lambda s: C(s,q+'_in'))]}")
    for L in sorted({lam, 0.30} - {None}):
        tot = {f: pp(a, lambda s, f=f: C(s, "charb" + f) + L * C(s, "spec" + f) + 0.02 * C(s, "div" + f) + 0.05 * C(s, "grad" + f)) for f in ("", "_rep", "_in")}
        print(f"   total training loss at lambda={L:.2f}: as-is {[round(x,4) for x in tot['']]}  v4e-fill {[round(x,4) for x in tot['_rep']]}  input-fill {[round(x,4) for x in tot['_in']]}")

print("\n" + "=" * 170); print("D. SMOOTHNESS OUTSIDE THE REGION (variance outside region / DNS variance on the same pixels) and fluctuation-energy share INSIDE it"); print("=" * 170)
for k in keys:
    a = Z[k]
    if np.mean(C(a, "vo_any") > 0) < 0.5: continue
    rm = {c: pp(a, lambda s, c=c: C(s, f"vout_m_{c}") / C(s, f"vout_dns_{c}") * 100) for c in "uvw"}
    rr = {c: pp(a, lambda s, c=c: C(s, f"vout_ref_{c}") / C(s, f"vout_dns_{c}") * 100) for c in "uvw"}
    sh = {c: pp(a, lambda s, c=c: C(s, f"esh_{c}") * 100) for c in "uvw"}
    print(f" {k:<22} model/DNS % u {[round(x) for x in rm['u']]} v {[round(x) for x in rm['v']]} w {[round(x) for x in rm['w']]}"
          f" | v4e/DNS % u {[round(x) for x in rr['u']]} v {[round(x) for x in rr['v']]} w {[round(x) for x in rr['w']]}"
          f" | energy share in region % u {[round(x,1) for x in sh['u']]} v {[round(x,1) for x in sh['v']]} w {[round(x,1) for x in sh['w']]}")

print("\n" + "=" * 170); print("E. FIGURE-3 TEST (Tier 2, audit subset): spectrum ratio to DNS at selected k, and RMS ratio to DNS"); print("=" * 170)
fk = list(Z["fig3_keys"]); S = Z["fig3_S"]; V = Z["fig3_V"]; Nn = Z["fig3_N"]
ks = [16, 24, 32, 40, 48, 64, 96, 128, 192, 256]
d = fk.index("DNS")
for pi, yp in enumerate(YP):
    for key in fk:
        if key == "DNS": continue
        r = S[fk.index(key), pi] / S[d, pi]
        print(f" {yp:<6} {key:<12} u E(k)/DNS at {ks}: {[round(float(r[0,k]),2) for k in ks]}  | v {[round(float(r[1,k]),2) for k in (32,64,128)]} w {[round(float(r[2,k]),2) for k in (32,64,128)]}"
              f" | RMS/DNS u,v,w {[round(float(x),3) for x in np.sqrt(V[fk.index(key), pi] / V[d, pi])]}")
    print(f" {yp:<6} n={int(Nn[pi])}")

print("\n" + "=" * 170); print("F. TABLES II-IV CHECKPOINTS: spike status (Tier 1 from spiketest records, full set where available; Tiers 2-3 from this run)"); print("=" * 170)
T1 = {}
for fn in ("records_full_A.npz", "records_full_B.npz"):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); c1 = list(z["cols"])
        for kk in json.loads(str(z["meta"])):
            if kk.startswith("T1|") and kk in z.files and len(z[kk]):
                T1[kk] = (z[kk], c1)
want = [f"T1|{m}_s{s}" for m in ("v4e", "v3", "kim", "fukami_cnn", "fukami_dscms", "guastoni") for s in (42, 43, 44)]
for k in want:
    if k in T1:
        a, c1 = T1[k]; n = len(a); fr = 100 * np.mean(a[:, c1.index("vo_any")] > 0)
        mz = max(max(a[:, c1.index(f"zhi_{c}")].max(), -a[:, c1.index(f"zlo_{c}")].min()) for c in "uvw")
        print(f" {k:<22} n={n:>4} samples with a flagged pixel {fr:5.2f}%  max|z| {mz:6.1f}")
    else:
        print(f" {k:<22} not in spiketest records")
for k in keys:
    t, m = k.split("|")
    if t in ("T2", "T3") and (m.split("_s")[0] in ("v4e", "v3", "kim", "fukami_cnn", "fukami_dscms", "guastoni")) and "_ep" not in m:
        a = Z[k]; fr = 100 * np.mean(C(a, "vo_any") > 0)
        mz = max(max(C(a, f"zhi_{c}").max(), -C(a, f"zlo_{c}").min()) for c in "uvw")
        print(f" {k:<22} n={len(a):>4} samples with a flagged pixel {fr:5.2f}%  max|z| {mz:6.1f}")
