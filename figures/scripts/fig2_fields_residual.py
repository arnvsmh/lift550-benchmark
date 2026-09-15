#!/usr/bin/env python3
r"""
Figure 2 - field reconstruction and residual, Tier 2, y+ = 15, seed 42.

Columns : degraded input (denormalised), YASU-Net v4e, v4c, DNS, |v4c - DNS|
Rows    : u, v, w

READS
  E:\lift550_paper\figure_data\arrays_ALL_tier2.npz   fields (verified seed-42 snapshot)
  E:\lift550_paper\code\norm_stats_train.json         training mean/std for denormalisation

WHY THE DENORMALISATION MATTERS
  The `input` entry of the npz is the network's z-scored input: std(input)/std(dns) =
  5.97 / 13.32 / 14.01 for u / v / w.  Every other entry is physical.  The previous
  figure plotted the z-scored array against physical colour limits, driving 34.3 % of
  the u, 78.5 % of the v and 83.1 % of the w input pixels to the colourmap end-stops.

COLOUR LIMITS
  Per row, diverging, centred on the DNS mean, half-width = 99th percentile of
  |x - mean| over the DNS field.  DNS saturation is therefore exactly 1.00 % per row.
  For v and w this reproduces the previous figure's limits (0.0641 vs +-0.064,
  0.1232 vs +-0.118); for u it replaces a symmetric +-0.58 that wasted half the
  colourmap on values u never takes.

RESIDUAL COLUMN
  |v4c - DNS| on a log scale, 1e-3 to 1e1, shared across rows.  v4c's median residual
  exceeds v4e's by only 1.6-4.0 %, but its worst pixel in u is 8.504 - 13.7x the DNS
  field's entire dynamic range.  A linear scale renders that as one more bright speck.
  The spike occupies a few pixels, so each residual panel carries a zoom inset.
"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

FD   = r"E:\lift550_paper\figure_data\arrays_ALL_tier2.npz"
STAT = r"E:\lift550_paper\code\norm_stats_train.json"
COMPS, YP = ["u", "v", "w"], "yp15"
SPIKE_RC = (356, 120)          # row, col - extreme pixel, all three components
ZOOM_HALF = 26                 # half-width of the inset window, px
PCTL = 0.99
RES_VMIN, RES_VMAX = 1e-3, 1e1

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def load():
    d = np.load(FD)
    st = json.load(open(STAT))
    mu = [st[f"{c}xz_{YP}"]["mean"] for c in COMPS]
    sd = [st[f"{c}xz_{YP}"]["std"] for c in COMPS]
    inp = np.stack([d["input"][i] * sd[i] + mu[i] for i in range(3)])
    return d, inp


def limits(dns):
    out = []
    for i in range(3):
        c = float(dns[i].mean())
        h = float(np.quantile(np.abs(dns[i] - c), PCTL))
        out.append((c - h, c + h))
    return out


def main(out_pdf):
    d, inp = load()
    dns, v4e, v4c = d["dns"], d["field_v4e"], d["field_basin"]
    res = np.abs(v4c - dns)
    lims = limits(dns)
    r0, c0 = SPIKE_RC
    sl = (slice(r0 - ZOOM_HALF, r0 + ZOOM_HALF), slice(c0 - ZOOM_HALF, c0 + ZOOM_HALF))

    cols = [("Degraded input", inp), (r"v4e ($\lambda\!=\!0.07$)", v4e),
            (r"v4c ($\lambda\!=\!0.30$)", v4c), ("DNS", dns)]

    fig = plt.figure(figsize=(7.16, 4.15))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, .46, 1, .40],
                          wspace=0.05, hspace=0.10)
    sat, stats = {}, {}
    for r in range(3):
        lo, hi = lims[r]
        for c, (title, arr) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(arr[r], cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title, pad=3)
            if c == 0: ax.set_ylabel(f"${COMPS[r]}$", rotation=0, labelpad=9,
                                     va="center", fontsize=10)
            sat[(COMPS[r], title)] = float(((arr[r] < lo) | (arr[r] > hi)).mean() * 100)
            if title.startswith("v4c"):
                ax.add_patch(Circle((c0, r0), 30, fill=False, ec="#101010", lw=0.7))

        host = fig.add_subplot(gs[r, 4]); host.axis("off")
        cax = host.inset_axes([0.0, 0.14, 0.17, 0.72])
        cb = fig.colorbar(im, cax=cax, ticks=[lo, hi])
        cb.ax.set_yticklabels([f"{lo:+.2f}", f"{hi:+.2f}"])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)

        axr = fig.add_subplot(gs[r, 5])
        imr = axr.imshow(res[r], cmap="viridis",
                         norm=LogNorm(vmin=RES_VMIN, vmax=RES_VMAX), interpolation="none")
        axr.set_xticks([]); axr.set_yticks([])
        if r == 0: axr.set_title(r"$|\mathrm{v4c}-\mathrm{DNS}|$", pad=3)
        axr.add_patch(Rectangle((c0 - ZOOM_HALF, r0 - ZOOM_HALF), 2 * ZOOM_HALF,
                                2 * ZOOM_HALF, fill=False, ec="w", lw=0.6))
        ins = axr.inset_axes([0.56, 0.02, 0.42, 0.42])
        ins.imshow(res[r][sl], cmap="viridis",
                   norm=LogNorm(vmin=RES_VMIN, vmax=RES_VMAX), interpolation="none")
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values(): s.set_color("w"); s.set_linewidth(0.8)
        stats[COMPS[r]] = (float(res[r].max()), float(np.median(res[r])),
                           float(np.median(np.abs(v4e[r] - dns[r]))))

    hostr = fig.add_subplot(gs[:, 6]); hostr.axis("off")
    cbr = fig.colorbar(imr, cax=hostr.inset_axes([0.0, 0.05, 0.20, 0.90]))
    cbr.ax.tick_params(labelsize=6, length=2, pad=1)

    fig.savefig(out_pdf, dpi=600)
    png = str(out_pdf).replace(".pdf", ".png")
    fig.savefig(png, dpi=220)
    plt.close(fig)

    print(f"colour limits (percentile {PCTL} of |x - DNS mean|):")
    for i, c in enumerate(COMPS):
        print(f"  {c}: centre {dns[i].mean():+.4f}  [{lims[i][0]:+.4f}, {lims[i][1]:+.4f}]"
              f"  = {(lims[i][1]-lims[i][0])/2/dns[i].std():.2f} sigma_DNS")
    print("\nend-stop saturation, % of panel pixels outside the colour limits:")
    hdr = [t for t, _ in cols]
    print("  comp " + "".join(f"{h[:16]:>18s}" for h in hdr))
    for c in COMPS:
        print(f"  {c}   " + "".join(f"{sat[(c,h)]:17.3f}%" for h in hdr))
    print("\nresidual: max | median v4c | median v4e | ratio")
    for c in COMPS:
        mx, m4c, m4e = stats[c]
        print(f"  {c}: {mx:8.4f} | {m4c:.4f} | {m4e:.4f} | {m4c/m4e:.4f}")
    print(f"\nwrote {out_pdf}\nwrote {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
