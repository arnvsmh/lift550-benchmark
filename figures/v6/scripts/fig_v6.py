#!/usr/bin/env python3
r"""
Figures for the 6-page version (STCR 2026): reduced rebuilds of Figures 2, 3(a) and 4.

  figA_fields_residual_v.pdf   v only, one row: DNS, v4e, v4c, |v4c - DNS|
  figB_onset.pdf               validation spectral loss vs epoch, seeds 42/43/44
  figAB_combined.pdf           A above B with panel labels (a), (b): one float
  figC_spectrum_u.pdf          E(k) of u at y+ = 15: DNS, Kim, v4e, v4c apparent and masked

  python fig_v6.py [--only {A,B,AB,C} ...] [--outdir DIR]
  builds, checks and writes only the figures named (default: all). The other figures'
  files are not written; their sha256 is compared before and after the run.

READS (same files, snapshot and arrays as the full-size figures)
  E:\lift550_paper\figure_data\arrays_ALL_tier2.npz   via fig2_fields_residual.load(); C: fig3's FD
  E:\lift550_paper\code\norm_stats_train.json
  the three v4c Tier 1 history.json files            via fig4_onset.SRC
  C only:
  E:\lift550_paper\analysis_coherence_threeseed_20260912\envelope_iiih_yp15.json   III-H envelope
  E:\lift550_paper\analysis_final3_provenance_20260912\fig3a_iiih_mask.json        reference numbers

NOTHING IS RE-TYPED. Loading, colour limits (limits()), spike location (SPIKE_RC),
zoom window (ZOOM_HALF), residual norm (RES_VMIN, RES_VMAX), line colours and styles
are imported from ../../scripts, so the full-size and reduced figures cannot drift.
For C, colours, k_c, the npz path and the model order come from fig3_spectra_profiles;
its line widths, zorders and axis limits are literals inside panel_a() and are copied.

GEOMETRY. Every axes is placed in inches and the PDF is saved without a tight bbox,
so each file is exactly IEEEtran \columnwidth (21pc = 3.487 in) wide: include with
width=\columnwidth, scale 1.000. The combined figure is drawn by the same two
functions at a vertical offset, so its panels are A and B verbatim.

PRESENTATION CHANGES ONLY
  A: input column and u, w rows dropped; colourbars under the panels with end labels
     beside the bar. The spike circle (9.5 pt across, 0.9 pt stroke) and the zoom
     inset (0.475 in, 1 pt frame) keep the printed size they have in the full-size
     figure at \textwidth. The zoom box still marks the same 52x52 px window, so it
     shrinks with the panel (8.2 -> 6.0 pt).
  B: fonts 7/6 pt; y ticks every 0.1 (minor 0.05); axis labels, lambda markers and
     legend entries unchanged; legend inside the axes.
  C: fonts 7/6 pt (k_c label 7 -> 6 pt); ticks as B except the x tick labels sit 3.5 pt
     below the axis so 10^0 clears 10^-9 at the log-log corner; legend inside the axes;
     v3 dropped (cut from the 6-page paper). DNS, Kim, v4e and v4c (apparent) are the
     stored spec_yp15_* arrays, drawn verbatim; curves are drawn with path simplification
     off so the PDF keeps all 256 vertices of each (matplotlib's default drops 25-32).

FIGURE C'S MASKED CURVE USES A DIFFERENT RULE FROM THE FULL-SIZE FIGURE.
fig3_spectra_profiles.py masks the u pixels beyond this snapshot's DNS range +- 3 x this
snapshot's DNS std, dilated 7x7 (0.119 % of the plane). C uses the rule printed in
Section III-H of main32.tex (III-C of the 6-page paper): DNS min/max per component over
the 1,233-sample audit subset +- 3 sigma_train, flags of u, v and w merged, dilated 5x5
(two periodic 3x3 max filters), v4c's u replaced by v4e's u there (0.094 %).
spectrum(), dilate() and the flags are copied verbatim from fig3a_iiih_mask.py;
data_checks_C reproduces its JSON.
"""
import sys, json, argparse, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import fig2_fields_residual as F2          # noqa: E402
import fig4_onset as F4                    # noqa: E402
with matplotlib.rc_context():              # fig3 sets rcParams at import (lines.linewidth among them);
    import fig3_spectra_profiles as F3     # noqa: E402  restored on exit, so A and B render as before

COLW = 252 / 72.27        # IEEEtran conference \columnwidth = 21pc, in
TEXTH = 9.25              # IEEEtran conference \textheight, in
V = 1                     # v is row 1 of the (u, v, w) stack

plt.rcParams.update({     # after the imports, which set their own rcParams
    "font.family": "serif", "font.size": 7,
    "axes.titlesize": 7, "axes.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "pdf.fonttype": 42, "savefig.bbox": "standard", "savefig.pad_inches": 0,
})

# Figure A geometry, inches
A_ML = A_MR = 0.02
A_GAP, A_GAPR = 0.035, 0.075                          # field gaps; gap before residual
A_S = (COLW - A_ML - A_MR - 2 * A_GAP - A_GAPR) / 4   # square panel side
A_TITLE, A_CBGAP, A_CBH, A_BOT = 0.15, 0.045, 0.05, 0.06
A_H = A_BOT + A_CBH + A_CBGAP + A_S + A_TITLE
CIRCLE_R, CIRCLE_LW = 41, 0.9                         # px, pt: 9.5 pt across, as printed now
BOX_LW, INSET_LW = 0.75, 1.0                          # pt
INSET = [0.41, 0.015, 0.575, 0.575]                   # panel fraction: 0.475 in, as printed now

# Figure B geometry, inches
B_H = 0.16 * TEXTH
B_L, B_R, B_B, B_T = 0.33, 0.02, 0.255, 0.025
YLABEL = "validation spectral loss"      # 1.14 in at 7 pt on the 1.20 in axis

GAP_AB = 0.09             # combined figure only: band between A and B that holds "(b)"

EXPECT_STEPS = {42: [(16, 0.3163, 0.1626), (36, 0.1511, 0.0508)],
                43: [(16, 0.3219, 0.0654)],
                44: [(16, 0.3141, 0.0638)]}

# Figure C geometry, inches
C_H, C_HMAX = 0.21 * TEXTH, 2.0
C_L, C_R, C_B, C_T = 0.43, 0.02, 0.34, 0.055
C_XLIM, C_YLIM, C_KC_Y = (1, 260), (1e-9, 1e-3), 1.6e-9   # literals in fig3_spectra_profiles.panel_a
ENV_IIIH = r"E:\lift550_paper\analysis_coherence_threeseed_20260912\envelope_iiih_yp15.json"
REF_IIIH = r"E:\lift550_paper\analysis_final3_provenance_20260912\fig3a_iiih_mask.json"
REL_TOL = 1e-12           # C, checks (a) and (d): float agreement with stored arrays / JSON

NAMES = {"A": "figA_fields_residual_v.pdf", "B": "figB_onset.pdf",
         "AB": "figAB_combined.pdf", "C": "figC_spectrum_u.pdf"}


def inch_axes(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def pow10(x):
    return rf"$10^{{{int(round(np.log10(x)))}}}$"


def end_labels(cax, left, right):
    kw = dict(xycoords="axes fraction", textcoords="offset points",
              va="center_baseline", fontsize=6)
    cax.annotate(left, (0, 0.5), (-2, 0), ha="right", **kw)
    cax.annotate(right, (1, 0.5), (2, 0), ha="left", **kw)


def load_fields():
    d, _ = F2.load()                       # _ = denormalised input, not drawn in v6
    dns, v4e, v4c = d["dns"], d["field_v4e"], d["field_basin"]
    return dict(dns=dns, v4e=v4e, v4c=v4c, res=np.abs(v4c - dns), lims=F2.limits(dns))


def draw_fields(fig, F, y0):
    fw, fh = fig.get_size_inches()
    lo, hi = F["lims"][V]
    r0, c0 = F2.SPIKE_RC
    h = F2.ZOOM_HALF
    yp = y0 + A_BOT + A_CBH + A_CBGAP
    xs = [A_ML + i * (A_S + A_GAP) for i in range(3)]
    xs.append(xs[2] + A_S + A_GAPR)
    art = {}
    titles = ("DNS", r"v4e ($\lambda\!=\!0.07$)", r"v4c ($\lambda\!=\!0.30$)")
    for x, key, title in zip(xs, ("dns", "v4e", "v4c"), titles):
        ax = inch_axes(fig, x, yp, A_S, A_S)
        art[key] = ax.imshow(F[key][V], cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, pad=2.5)
        if key == "v4c":
            art["circle"] = ax.add_patch(Circle((c0, r0), CIRCLE_R, fill=False,
                                                ec="#101010", lw=CIRCLE_LW))
    axr = inch_axes(fig, xs[3], yp, A_S, A_S)
    art["res"] = axr.imshow(F["res"][V], cmap="viridis", interpolation="none",
                            norm=LogNorm(vmin=F2.RES_VMIN, vmax=F2.RES_VMAX))
    axr.set_xticks([]); axr.set_yticks([])
    axr.set_title(r"$|\mathrm{v4c}-\mathrm{DNS}|$", pad=2.5)
    art["box"] = axr.add_patch(Rectangle((c0 - h, r0 - h), 2 * h, 2 * h,
                                         fill=False, ec="w", lw=BOX_LW))
    ins = axr.inset_axes(INSET)
    art["inset"] = ins.imshow(F["res"][V][r0 - h:r0 + h, c0 - h:c0 + h], cmap="viridis",
                              interpolation="none",
                              norm=LogNorm(vmin=F2.RES_VMIN, vmax=F2.RES_VMAX))
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color("w"); s.set_linewidth(INSET_LW)

    yc = y0 + A_BOT
    x0, x1 = xs[0] + 0.46, xs[2] + A_S - 0.34
    cb = fig.colorbar(art["dns"], cax=inch_axes(fig, x0, yc, x1 - x0, A_CBH),
                      orientation="horizontal", ticks=[])
    cb.outline.set_linewidth(0.5)
    end_labels(cb.ax, f"{lo:+.3f}".replace("-", "\u2212"), f"{hi:+.3f}")
    fig.text((xs[0] + 0.03) / fw, (yc + A_CBH / 2) / fh, "$v$", fontsize=7,
             ha="left", va="center_baseline")

    xr0, xr1 = xs[3] + 0.23, xs[3] + A_S - 0.18
    cbr = fig.colorbar(art["res"], cax=inch_axes(fig, xr0, yc, xr1 - xr0, A_CBH),
                       orientation="horizontal")
    cbr.set_ticks([1e-2, 1e-1, 1e0])
    cbr.ax.xaxis.set_major_formatter(NullFormatter())
    cbr.minorticks_off()
    cbr.ax.tick_params(length=1.5, width=0.5)
    cbr.outline.set_linewidth(0.5)
    end_labels(cbr.ax, pow10(F2.RES_VMIN), pow10(F2.RES_VMAX))
    return art


def draw_onset(fig, hist, y0):
    ax = inch_axes(fig, B_L, y0 + B_B, COLW - B_L - B_R, B_H - B_B - B_T)
    for e in (16, 36):
        ax.axvline(e, color="0.78", lw=0.6, ls=":", zorder=0)
    lam = [ax.text(16, 0.375, r"$\lambda\!:\,0.05\!\to\!0.15$", fontsize=5.5, color="0.35",
                   ha="center", va="bottom"),
           ax.text(36, 0.375, r"$\lambda\!:\,0.15\!\to\!0.30$", fontsize=5.5, color="0.35",
                   ha="center", va="bottom")]
    lines = {}
    for s in (42, 43, 44):
        ep = [r["epoch"] for r in hist[s]]
        vs = [r["val_spec"] for r in hist[s]]
        lines[s], = ax.plot(ep, vs, ls=F4.STYLE[s], lw=0.9, color=F4.COL[s],
                            label="seed %d" % s)
    ax.set_xlabel("training epoch", labelpad=1.5)
    ax.set_ylabel(YLABEL, labelpad=2)
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 0.42)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(which="major", length=2, width=0.5, pad=1.5)
    ax.tick_params(which="minor", length=1.2, width=0.4)
    leg = ax.legend(frameon=False, loc="center right", handlelength=1.8,
                    borderaxespad=0.4, labelspacing=0.25, handletextpad=0.5)
    return dict(ax=ax, lines=lines, lam=lam, leg=leg)


# ---- Figure C. N .. BINS, spectrum(), dilate() and the III-H flags in load_spectrum() are copied
# verbatim from E:\lift550_paper\analysis_final3_provenance_20260912\fig3a_iiih_mask.py
# (metrics.py binning: |fft2|^2 / (512*512)^2 over bins (K >= k-0.5) & (K < k+0.5)). That script
# loads data and writes its JSON when imported, so it is replicated here, not imported;
# data_checks_C reproduces its JSON.
N = 512
kx = np.fft.fftfreq(N, d=1.0 / N)
KX, KY = np.meshgrid(kx, kx)
K = np.sqrt(KX ** 2 + KY ** 2)
BINS = [(K >= k - 0.5) & (K < k + 0.5) for k in range(257)]
QS = (32, 64, 128)


def spectrum(f):
    P = np.abs(np.fft.fft2(f)) ** 2 / float(N * N) ** 2
    return np.array([P[m].sum() for m in BINS])


def dilate(m, r):
    o = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


def load_spectrum():
    """Stored apparent spectra of u at y+ = 15 (DNS, Kim, v3, v4e, v4c; v3 for reference numbers
    only, not drawn) and the v4c spectrum masked by the Section III-H rule."""
    Z = np.load(F3.FD)
    STATS = json.load(open(F2.STAT))
    ENV = json.load(open(ENV_IIIH))
    dns = Z["dns"].astype(np.float64)                   # from here to Em: fig3a_iiih_mask.py
    v4c = Z["field_basin"].astype(np.float64)
    v4e = Z["field_v4e"].astype(np.float64)
    sig = np.array([STATS[f"{v}_yp15"]["std"] for v in ("uxz", "vxz", "wxz")], np.float64)
    emin, emax = np.array(ENV["emin"], np.float64), np.array(ENV["emax"], np.float64)
    f_iiih = np.zeros((N, N), bool)
    for c in range(3):
        f_iiih |= (v4c[c] < emin[c] - 3 * sig[c]) | (v4c[c] > emax[c] + 3 * sig[c])
    region = dilate(f_iiih, 2)
    mu = v4c[0].copy()
    mu[region] = v4e[0][region]
    Em = spectrum(mu)
    per_comp = [int(((v4c[c] < emin[c] - 3 * sig[c]) | (v4c[c] > emax[c] + 3 * sig[c])).sum())
                for c in range(3)]
    return dict(k=Z["spec_yp15_k"], stored={n: Z["spec_yp15_" + key] for n, key in F3.ORDER},
                E_dns=spectrum(dns[0]), E_mask=Em, flag=f_iiih, region=region, per_comp=per_comp)


def spectrum_curves(S):
    """Legend order -> (k, E(k), style), as fig3_spectra_profiles.panel_a draws them, minus v3."""
    k, st = S["k"], S["stored"]
    sel = k >= 1
    out = {n: (k[sel], st[n][sel], dict(color=F3.C[n], lw=1.35 if n == "DNS" else 0.9,
                                        zorder=3 if n == "DNS" else 2))
           for n in ("DNS", "Kim", "v4e")}
    out["v4c (apparent)"] = (k[sel], st["v4c"][sel], dict(color=F3.C["v4c"], lw=1.0, zorder=4))
    out["v4c (masked)"] = (np.arange(257)[1:257], S["E_mask"][1:257],
                           dict(color=F3.C["v4c"], lw=1.1, ls="--", zorder=5))
    return out


def draw_spectrum(fig, S, y0):
    ax = inch_axes(fig, C_L, y0 + C_B, COLW - C_L - C_R, C_H - C_B - C_T)
    ax.axvline(F3.KCUT, color="0.75", lw=0.6, ls=":", zorder=0)
    kc = ax.text(F3.KCUT * 1.08, C_KC_Y, "$k_c$", color="0.45", fontsize=6, va="bottom")
    lines = {}
    with plt.rc_context({"path.simplify": False}):  # keep every vertex in the PDF (verify_v6.py)
        for n, (x, y, style) in spectrum_curves(S).items():
            lines[n], = ax.loglog(x, y, label=n, **style)
    ax.set_xlabel("$k$", labelpad=1.5)
    ax.set_ylabel("$E(k)$", labelpad=2)
    ax.set_xlim(*C_XLIM)
    ax.set_ylim(*C_YLIM)
    ax.tick_params(which="major", length=2, width=0.5, pad=1.5)
    ax.tick_params(which="minor", length=1.2, width=0.4)
    ax.tick_params(axis="x", which="major", pad=3.5)  # 10^0 then clears 10^-9 at the corner
    leg = ax.legend(frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.3,
                    labelspacing=0.25, handletextpad=0.5)
    return dict(ax=ax, lines=lines, lam=[kc], leg=leg, labels="k_c label")


def check(label, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    return bool(ok)


def layout_report(fig, name, B=None):
    """Ink inside the canvas; no text/text or text/foreign-axes overlap; curves clear
    of the legend and the in-axes labels (B: lambda labels, C: k_c label)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    ticklabels, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                ticklabels.update((t.label1, t.label2))
            for t in axis._update_ticks():          # ticks inside the view limits = drawn ones
                drawn.update((t.label1, t.label2))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and (t not in ticklabels or t in drawn)]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    tb = fig.get_tightbbox(r)
    m = (tb.x0, tb.y0, w - tb.x1, h - tb.y1)
    ok = check(f"{name}: all ink inside the canvas, margins L/B/R/T = "
               + " / ".join(f"{v:+.3f}" for v in m) + " in", min(m) > -1e-3)
    lo_t = min(boxes, key=lambda b: b[1].y0)
    hi_t = max(boxes, key=lambda b: b[1].y1)
    print(f"      lowest text '{lo_t[0].get_text()}' {lo_t[1].y0 / fig.dpi:.3f} in above the bottom;"
          f" highest '{hi_t[0].get_text()}' {h - hi_t[1].y1 / fig.dpi:.3f} in below the top")
    bad = []
    for i, (t1, b1) in enumerate(boxes):
        bad += [f"'{t1.get_text()}' x '{t2.get_text()}'" for t2, b2 in boxes[i + 1:]
                if b1.overlaps(b2)]
        bad += [f"'{t1.get_text()}' x axes" for ax in fig.axes
                if ax is not t1.axes and b1.overlaps(ax.get_window_extent(r))]
    if B is not None:
        obst = [("legend", B["leg"].get_window_extent(r))]
        obst += [(t.get_text(), t.get_window_extent(r)) for t in B["lam"]]
        for s, ln in B["lines"].items():
            q = ln.axes.transData.transform(ln.get_xydata())     # densify in display space, where
            p = np.concatenate([np.linspace(q[i], q[i + 1], 16)  # segments are straight on log axes too
                                for i in range(len(q) - 1)])
            for lab, bb in obst:
                if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) &
                        (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                    bad.append(f"{'seed %d' % s if isinstance(s, int) else s} curve x {lab}")
    ok &= check(f"{name}: no overlaps (text/text, text/other axes"
                + (f", curves/legend/{B.get('labels', 'lambda labels')})" if B else ")")
                + ("" if not bad else " -> " + "; ".join(bad)), not bad)
    return ok


def data_checks(F, hist, arts_A, arts_B):
    ok = True
    dns, v4c = F["dns"], F["v4c"]
    print("\nSPIKE (same test as verify_sources.py):")
    ext = []
    for i, c in enumerate("uvw"):
        ref, sd = dns[i], dns[i].std()
        r_, c_ = np.unravel_index(np.argmax(np.abs((v4c[i] - ref.mean()) / sd)), ref.shape)
        ext.append((int(r_), int(c_)))
        print(f"    {c}: extreme pixel ({r_},{c_})")
    rows, cols = [e[0] for e in ext], [e[1] for e in ext]
    ok &= check(f"extreme pixels span rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"
                " (expected 356-358, 120-121)",
                (min(rows), max(rows), min(cols), max(cols)) == (356, 358, 120, 121))
    ref, sd = dns[V], dns[V].std()
    rr, cc = np.where((v4c[V] < ref.min() - 3 * sd) | (v4c[V] > ref.max() + 3 * sd))
    r0, c0 = F2.SPIKE_RC
    hz = F2.ZOOM_HALF
    print(f"    v flagged: {rr.size} px, rows {rr.min()}-{rr.max()}, cols {cc.min()}-{cc.max()}")
    ok &= check(f"circle centre (row {r0}, col {c0}) unchanged; all flagged v px inside the "
                f"r = {CIRCLE_R} px circle (max distance {np.hypot(rr - r0, cc - c0).max():.1f})",
                (r0, c0) == (356, 120) and np.hypot(rr - r0, cc - c0).max() < CIRCLE_R)
    ok &= check(f"all flagged v px inside the {2 * hz}x{2 * hz} zoom window",
                rr.min() >= r0 - hz and rr.max() < r0 + hz and cc.min() >= c0 - hz and cc.max() < c0 + hz)

    print("\nCOLOUR SCALES:")
    lo, hi = F["lims"][V]
    ok &= check(f"v field limits [{lo:+.4f}, {hi:+.4f}] (expected +-0.0641)",
                (round(lo, 4), round(hi, 4)) == (-0.0641, 0.0641))
    for tag, art in arts_A:
        ok &= check(f"{tag}: DNS/v4e/v4c images drawn at exactly those limits",
                    all(art[k].get_clim() == (lo, hi) for k in ("dns", "v4e", "v4c")))
        ok &= check(f"{tag}: residual and inset LogNorm [{F2.RES_VMIN:g}, {F2.RES_VMAX:g}]",
                    all(isinstance(art[k].norm, LogNorm) and
                        (art[k].norm.vmin, art[k].norm.vmax) == (1e-3, 1e1)
                        for k in ("res", "inset")))
        same = [np.array_equal(np.ma.getdata(art[k].get_array()), F[k][V])
                for k in ("dns", "v4e", "v4c", "res")]
        same.append(np.array_equal(np.ma.getdata(art["inset"].get_array()),
                                   F["res"][V][r0 - hz:r0 + hz, c0 - hz:c0 + hz]))
        ok &= check(f"{tag}: drawn arrays bit-identical to the npz v fields / residual / inset",
                    all(same))

    print("\nONSET CURVES (same test as fig4_onset.py):")
    for s in (42, 43, 44):
        h_ = hist[s]
        lam = [r["lambda_spectral"] for r in h_]
        vs = [r["val_spec"] for r in h_]
        chg = [(h_[i]["epoch"], lam[i - 1], lam[i]) for i in range(1, len(h_)) if lam[i] != lam[i - 1]]
        drops = [(h_[i]["epoch"], round(vs[i - 1], 4), round(vs[i], 4))
                 for i in range(1, len(h_)) if vs[i] < 0.6 * vs[i - 1]]
        ok &= check(f"seed {s}: steps {drops}", drops == EXPECT_STEPS[s])
        ok &= check(f"seed {s}: lambda changes {chg}", chg == [(16, 0.05, 0.15), (36, 0.15, 0.3)])
        print(f"    seed {s}: final val_spec {vs[-1]:.4f}, min over run {min(vs):.4f} "
              f"(epoch {h_[int(np.argmin(vs))]['epoch']})")
        for tag, B in arts_B:
            xy = B["lines"][s].get_xydata()
            ok &= check(f"{tag}: seed {s} line = all {len(h_)} (epoch, val_spec) pairs",
                        np.array_equal(xy, np.array([[r["epoch"], r["val_spec"]] for r in h_])))
    fin = [round(hist[s][-1]["val_spec"], 4) for s in (42, 43, 44)]
    ok &= check(f"converged floor: final val_spec {fin} (0.0408-0.0409)",
                all(0.0408 <= f <= 0.0409 for f in fin))
    return ok


def data_checks_C(S, art):
    """(a)-(f) for Figure C. The stored spectra and fig3a_iiih_mask.json come from earlier runs,
    and last-ulp float agreement is not portable across machines or numpy builds; (a) and (d)
    therefore pass within REL_TOL and print the actual difference and where it is bit-identical."""
    ok = True
    ref = json.load(open(REF_IIIH))
    J = ref["section_III_H"]
    Z = np.load(F3.FD)                                   # re-read: lines are compared with the file
    k, D = Z["spec_yp15_k"], Z["spec_yp15_DNS"]
    sel = k >= 1
    idx = {q: int(np.where(np.isclose(k, q))[0][0]) for q in QS}
    st = {n: Z["spec_yp15_" + key] for n, key in F3.ORDER}

    def ratio(E, q):
        return float(E[idx[q]] / D[idx[q]])

    def bitexact(d):
        return ", ".join(str(q) for q in QS if d[q] == 0) or "none"

    print("\nSPECTRUM (C: u at y+ = 15, Tier 2, seed 42, validation index 0):")
    e = {q: float(abs(S["E_dns"][q] - D[idx[q]]) / abs(D[idx[q]])) for q in QS}
    ok &= check(f"(a) binning gate: spectrum(DNS u) vs stored spec_yp15_DNS at k = 32, 64, 128: max rel err "
                f"{max(e.values()):.1e} (tol {REL_TOL:g}; bit-identical at k = {bitexact(e)}; JSON recorded "
                f"{ref['gate_binning_max_rel_err_at_32_64_128']:g})", max(e.values()) <= REL_TOL)
    print(f"      over all k = 1..256: max rel err {np.max(np.abs(S['E_dns'][1:257] - D[sel]) / D[sel]):.1e}")

    L = art["lines"]
    ok &= check("(b) DNS, Kim, v4e, v4c (apparent) lines: x = spec_yp15_k, y = spec_yp15_{DNS,Kim,v4e,basin}"
                " at k = 1..256, exactly",
                all(np.array_equal(L[n].get_xdata(), k[sel]) and np.array_equal(L[n].get_ydata(), st[m][sel])
                    for n, m in (("DNS", "DNS"), ("Kim", "Kim"), ("v4e", "v4e"), ("v4c (apparent)", "v4c"))))
    legend = [t.get_text() for t in art["leg"].get_texts()]
    ok &= check("(b) v4c (masked) line = the III-H spectrum below at k = 1..256; legend: " + ", ".join(legend)
                + "; no v3 line",
                np.array_equal(L["v4c (masked)"].get_xdata(), np.arange(1, 257))
                and np.array_equal(L["v4c (masked)"].get_ydata(), S["E_mask"][1:257])
                and legend == list(L) == ["DNS", "Kim", "v4e", "v4c (apparent)", "v4c (masked)"]
                and not any(np.array_equal(ln.get_ydata(), st["v3"][sel]) for ln in art["ax"].get_lines()))
    ok &= check("    path simplification off on all 5 curves (each keeps its 256 vertices in the PDF)",
                all(not ln.get_path().should_simplify for ln in L.values()))

    f, reg = S["flag"], S["region"]
    rr, cc = np.where(reg)
    pct = 100.0 * float(reg.sum()) / (N * N)
    print(f"      III-H flags u/v/w {'/'.join(map(str, S['per_comp']))} px, merged {int(f.sum())} px; "
          f"region rows {rr.min()}-{rr.max()}, cols {cc.min()}-{cc.max()}")
    ok &= check("(c) 5x5 box dilation = two passes of a periodic 3x3 max filter",
                np.array_equal(reg, dilate(dilate(f, 1), 1)))
    ok &= check(f"(c) flagged {int(f.sum())} px, region {int(reg.sum())} px = {pct:.3f} % of the plane "
                f"(JSON: {J['flagged_px']} px, {J['region_px']} px, {J['region_pct_of_plane']:.3f} %; "
                f"manuscript 0.094 %)",
                (int(f.sum()), int(reg.sum()), pct) == (J["flagged_px"], J["region_px"], J["region_pct_of_plane"])
                and f"{pct:.3f}" == "0.094")

    m = {q: float(S["E_mask"][q] / D[idx[q]]) for q in QS}
    dm = {q: abs(m[q] / J["ratio_to_dns"][str(q)] - 1) for q in QS}
    ok &= check("(d) masked v4c / DNS at k = 32, 64, 128: " + ", ".join(f"{m[q]:.4f}" for q in QS)
                + f" = JSON to rel {max(dm.values()):.1e} (tol {REL_TOL:g}; bit-identical at k = {bitexact(dm)})",
                max(dm.values()) <= REL_TOL)
    ok &= check(f"(d) rounded: {m[64]:.2f} at k = 64, {m[128]:.2f} at k = 128 (manuscript 0.20, 0.03)",
                (f"{m[64]:.2f}", f"{m[128]:.2f}") == ("0.20", "0.03"))

    a = {q: ratio(st["v4c"], q) for q in (64, 128)}
    ok &= check(f"(e) apparent v4c / DNS (stored): {a[64]:.4f} at k = 64, {a[128]:.4f} at k = 128 -> "
                f"{a[64]:.2f}, {a[128]:.2f} (manuscript 0.99, 0.95)",
                (f"{a[64]:.2f}", f"{a[128]:.2f}") == ("0.99", "0.95"))

    r = {n: {q: ratio(st[n], q) for q in (64, 128)} for n in ("Kim", "v4e", "v3")}
    ok &= check("(f) v4e / DNS from the stored arrays = JSON v4e_ratio_to_dns at k = 32, 64, 128, bit-identical",
                all(ratio(st["v4e"], q) == ref["v4e_ratio_to_dns"][str(q)] for q in QS))
    for n in ("Kim", "v4e", "v3"):
        print(f"      {n:3s} / DNS (stored): {r[n][64]:.3f} at k = 64, {r[n][128]:.3f} at k = 128"
              + ("   reference only, not plotted" if n == "v3" else ""))
    for grp, lab in ((("Kim", "v4e"), "Kim and v4e (new caption range)"),
                     (("Kim", "v3", "v4e"), "Kim, v3 and v4e (manuscript: 0.21-0.28, 0.05-0.07)")):
        lo = {q: min(r[n][q] for n in grp) for q in (64, 128)}
        hi = {q: max(r[n][q] for n in grp) for q in (64, 128)}
        print(f"      {lab}: {lo[64]:.3f}-{hi[64]:.3f} at k = 64, {lo[128]:.3f}-{hi[128]:.3f} at k = 128;"
              f" 2 dp {lo[64]:.2f}-{hi[64]:.2f}, {lo[128]:.2f}-{hi[128]:.2f}")
    print("      masked v4c below both Kim and v4e at k = 64 and 128: "
          + str(all(m[q] < min(r["Kim"][q], r["v4e"][q]) for q in (64, 128))))
    return ok


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main(outdir, only=("all",)):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    want = [n for n in NAMES if "all" in only or n in only]
    keep = [p for n in NAMES if n not in want
            for p in (outdir / NAMES[n], (outdir / NAMES[n]).with_suffix(".png")) if p.exists()]
    before = {p: sha256(p) for p in keep}

    figs, arts_A, arts_B, art_C = {}, [], [], None
    if {"A", "B", "AB"} & set(want):
        F = load_fields()
        hist = {s: json.load(open(p)) for s, p in F4.SRC.items()}
    if "A" in want:
        figs["A"] = plt.figure(figsize=(COLW, A_H))
        arts_A.append(("A", draw_fields(figs["A"], F, 0.0)))
    if "B" in want:
        figs["B"] = plt.figure(figsize=(COLW, B_H))
        arts_B.append(("B", draw_onset(figs["B"], hist, 0.0)))
    H = A_H + GAP_AB + B_H
    if "AB" in want:
        figs["AB"] = fig = plt.figure(figsize=(COLW, H))
        arts_A.append(("A+B", draw_fields(fig, F, B_H + GAP_AB)))
        arts_B.append(("A+B", draw_onset(fig, hist, 0.0)))
        lab = dict(ha="left", va="top", fontsize=7.5, fontweight="bold")
        fig.text(0.01 / COLW, (H - 0.005) / H, "(a)", **lab)
        fig.text(0.01 / COLW, (B_H + GAP_AB - 0.02) / H, "(b)", **lab)
    if "C" in want:
        S = load_spectrum()
        figs["C"] = plt.figure(figsize=(COLW, C_H))
        art_C = draw_spectrum(figs["C"], S, 0.0)

    report = {"A": ("A", None), "B": ("B", dict(arts_B).get("B")),
              "AB": ("A+B", dict(arts_B).get("A+B")), "C": ("C", art_C)}
    print("LAYOUT:")
    ok = True
    for n in want:
        ok &= layout_report(figs[n], *report[n])
    if "C" in want:
        ok &= check(f"C: {COLW:.3f} x {C_H:.3f} in, height <= {C_HMAX:.1f} in", C_H <= C_HMAX)
    if arts_A or arts_B:
        ok &= data_checks(F, hist, arts_A, arts_B)
    if "C" in want:
        ok &= data_checks_C(S, art_C)

    if arts_A:
        print("\nANNOTATION SIZES (A, printed at scale 1.000):")
        print(f"  panel side {A_S:.3f} in; spike circle diameter {2 * CIRCLE_R / 512 * A_S * 72:.1f} pt,"
              f" lw {CIRCLE_LW} pt; zoom box {2 * F2.ZOOM_HALF / 512 * A_S * 72:.1f} pt, lw {BOX_LW} pt;"
              f" inset {INSET[2] * A_S:.3f} in, frame {INSET_LW} pt")

    print("\nOUTPUT (width = \\columnwidth = %.3f in; page fraction = height / %.2f in):" % (COLW, TEXTH))
    for n, fig in figs.items():
        pdf = outdir / NAMES[n]
        fig.savefig(pdf, dpi=600)
        fig.savefig(pdf.with_suffix(".png"), dpi=300)
        hh = fig.get_size_inches()[1]
        print(f"  {n:2s} {hh:.3f} in  {hh / TEXTH:.3f} page  {pdf}")
        plt.close(fig)
    if "AB" in want:
        print(f"  A + B separately {A_H + B_H:.3f} in; combined {H:.3f} in "
              f"(difference {H - A_H - B_H:+.3f} in = the (a)/(b) gap)")
    if before:
        print("\nNOT REBUILT (sha256 before this run -> after):")
        for p, h in before.items():
            h2 = sha256(p)
            ok &= check(f"{p.name:28s} {h} -> {'unchanged' if h2 == h else h2}", h2 == h)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "v6"))
    ap.add_argument("--only", nargs="+", default=["all"], choices=["A", "B", "AB", "C", "all"],
                    help="figures to build, check and write (default: all); the others are not written")
    a = ap.parse_args()
    sys.exit(0 if main(a.outdir, a.only) else 1)
