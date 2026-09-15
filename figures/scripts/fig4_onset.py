#!/usr/bin/env python3
r"""
Figure 4 - spike onset.  Validation spectral loss against epoch for v4c at Tier 1,
all three seeds.  Was panel (b) of the previous Figure 4; panel (a) is cut.

READS
  E:\lift550_paper\runs_final\yasunet_v4c_tier1_seed42\history.json    seed 42
  E:\lift550_data\histories\yasunet_v4c_tier1_seed43_history.json      seed 43
  E:\lift550_data\histories\yasunet_v4c_tier1_seed44_history.json      seed 44

Each history is a 50-element list with keys epoch, train_loss, val_loss, val_charb,
val_spec, lr, lambda_spectral.  The curve plotted is val_spec.

WHAT THE SOURCES SAY (read off the files, printed by this script)
  lambda schedule, identical in all three runs: 0.05 -> 0.15 at epoch 16,
  0.15 -> 0.30 at epoch 36.
  seed 42 steps twice: 0.3163 -> 0.1626 at epoch 16, 0.1511 -> 0.0508 at epoch 36.
  seed 43 steps once:  0.3219 -> 0.0654 at epoch 16.
  seed 44 steps once:  0.3141 -> 0.0638 at epoch 16.
  Charbonnier loss moves by +1.6 % and +1.7 % at seed 42's two steps, and by +5.3 %
  and +6.0 % at the single steps of seeds 43 and 44 - a large spectral gain at almost
  no pixel cost, which is the signature of spike onset.

  So the number of steps is seed-dependent while onset at a scheduled lambda increase
  is not.  The previous caption said "seeds 43 and 44 (not shown)".
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = {
    42: r"E:\lift550_paper\runs_final\yasunet_v4c_tier1_seed42\history.json",
    43: r"E:\lift550_data\histories\yasunet_v4c_tier1_seed43_history.json",
    44: r"E:\lift550_data\histories\yasunet_v4c_tier1_seed44_history.json",
}
COL = {42: "#d62728", 43: "#1f77b4", 44: "#2ca02c"}
STYLE = {42: "-", 43: "--", 44: "-."}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def main(out):
    hist = {s: json.load(open(p)) for s, p in SRC.items()}

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for e in (16, 36):
        ax.axvline(e, color="0.78", lw=0.6, ls=":", zorder=0)
    ax.text(16, 0.375, r"$\lambda\!:\,0.05\!\to\!0.15$", fontsize=6.2, color="0.35",
            ha="center", va="bottom")
    ax.text(36, 0.375, r"$\lambda\!:\,0.15\!\to\!0.30$", fontsize=6.2, color="0.35",
            ha="center", va="bottom")

    for s in (42, 43, 44):
        ep = [r["epoch"] for r in hist[s]]
        vs = [r["val_spec"] for r in hist[s]]
        ax.plot(ep, vs, ls=STYLE[s], lw=1.0, color=COL[s], label="seed %d" % s)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("validation spectral loss")
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 0.42)
    ax.legend(frameon=False, loc="center right", handlelength=2.0,
              borderaxespad=0.6, labelspacing=0.3)
    ax.tick_params(length=2.5)
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=220)
    plt.close(fig)

    print("verified against the source files:")
    for s in (42, 43, 44):
        h = hist[s]
        lam = [r["lambda_spectral"] for r in h]
        vs = [r["val_spec"] for r in h]
        ch = [r["val_charb"] for r in h]
        chg = [(h[i]["epoch"], lam[i - 1], lam[i]) for i in range(1, len(h))
               if lam[i] != lam[i - 1]]
        drops = [(h[i]["epoch"], vs[i - 1], vs[i], ch[i - 1], ch[i])
                 for i in range(1, len(h)) if vs[i] < 0.6 * vs[i - 1]]
        print("  seed %d  n=%d  lambda changes %s" % (s, len(h), chg))
        for e, a, b, ca, cb in drops:
            print("    step at epoch %2d: val_spec %.4f -> %.4f   "
                  "val_charb %.5f -> %.5f (%+.1f %%)"
                  % (e, a, b, ca, cb, 100 * (cb / ca - 1)))
        print("    final val_spec %.4f" % vs[-1])
    print("\nwrote %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
