"""Read-only: per-run training histories of every YASU-Net v4* run.
Reports best.pt epoch (argmin combined val_loss), best_charb epoch (argmin val_charb),
every single-epoch val_spec drop > 0.02 with the lambda in force, and the full series."""
import json, os
roots = [r"C:\Users\Arnav Simha\Downloads\lift550_paper\runs_final",
         r"C:\Users\Arnav Simha\lift550_data\runs_newseeds"]
for r in roots:
    for d in sorted(os.listdir(r)):
        h = os.path.join(r, d, "history.json")
        if not d.startswith("yasunet_v4") or not os.path.exists(h):
            continue
        H = json.load(open(h))
        ep = [x["epoch"] for x in H]
        vl = [x["val_loss"] for x in H]
        vs = [x.get("val_spec") for x in H]
        vc = [x.get("val_charb") for x in H]
        lam = [x.get("lambda_spectral") for x in H]
        best = ep[vl.index(min(vl))]
        bc = ep[vc.index(min(vc))] if None not in vc else None
        print(f"== {os.path.basename(r)}/{d}: {len(H)} epochs (last {ep[-1]}); best.pt(argmin val_loss)=ep{best}; "
              f"best_charb(argmin val_charb)=ep{bc}")
        if None in vs:
            print("   keys:", sorted(H[0].keys()))
            continue
        drops = []
        for i in range(1, len(H)):
            dd = vs[i - 1] - vs[i]
            if dd > 0.02:
                drops.append(f"ep{ep[i]}: spec {vs[i-1]:.4f}->{vs[i]:.4f} (lam {lam[i]}), "
                             f"charb {vc[i-1]:.5f}->{vc[i]:.5f}")
        print("   drops>0.02:", "; ".join(drops) if drops else "none")
        print("   val_spec :", " ".join(f"{v:.3f}" for v in vs))
        print("   val_charb:", " ".join(f"{v:.4f}" for v in vc))
        print("   lambda   :", sorted(set(lam), key=lam.index))
