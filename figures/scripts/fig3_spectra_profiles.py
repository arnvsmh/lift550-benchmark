#!/usr/bin/env python3
r"""
Figure 3 - spectral and statistical fidelity, Tier 2, seed 42.  Writes two PDFs so the
existing subfigure scaffolding in main32.tex keeps its (a)/(b) subcaptions.

  (a) fig3a_spectrum.pdf  radially binned E(k) of u at y+ = 15 only (not four planes:
      the four panels showed the same thing four times).  Curves: DNS, Kim, v3, v4e,
      v4c apparent, v4c masked.
  (b) fig3b_profiles.pdf  RMS fluctuation profiles vs y+, all four planes, apparent only.

READS
  E:\lift550_paper\figure_data\arrays_ALL_tier2.npz     stored spectra + fields
  E:\lift550_paper\figure_data\extra_arrays_tier2.npz   stored RMS profiles

PROVENANCE
  The five apparent curves in (a) are the STORED spec_yp15_* arrays - the same arrays
  the previous figure drew.  Only the masked curve is computed here.  That computation
  was validated by recomputing all five apparent spectra from the stored fields and
  reproducing the stored arrays to a maximum relative error of 5.4e-6 over k = 1..256;
  the stored normalisation is |fft2|^2 / N^4 with N = 512.

  The masked curve replaces the spike region - pixels more than 3 sigma beyond the DNS
  range, dilated by a 7x7 box, 0.119 % of the plane - with the v4e prediction: the same
  fill and the same rule as Table VI.

  Panel (b) plots the stored profile arrays (first 10 evaluation samples per plane).
  The percentages quoted in Section IV-C come from the audit subset, a different sample
  basis; the two agree to about one percentage point.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FD    = r"E:\lift550_paper\figure_data\arrays_ALL_tier2.npz"
EXTRA = r"E:\lift550_paper\figure_data\extra_arrays_tier2.npz"
N, NORM = 512, 512.0 ** 4
KCUT = 32
YP = [15, 30, 50, 100]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})
C = {"DNS": "#000000", "Kim": "#1f77b4", "v3": "#ff7f0e",
     "v4e": "#2ca02c", "v4c": "#d62728"}
ORDER = [("DNS", "DNS"), ("Kim", "Kim"), ("v3", "v3"), ("v4e", "v4e"), ("v4c", "basin")]


def radial_k():
    kx = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    return np.rint(np.sqrt(KX ** 2 + KY ** 2)).astype(int)


def radial(x, K):
    P = np.abs(np.fft.fft2(x)) ** 2 / NORM
    return np.array([P[K == k].sum() for k in range(258)])


def dilate(m, r=3):
    o = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


def panel_a(out):
    d = np.load(FD)
    k = d["spec_yp15_k"]
    stored = {n: d["spec_yp15_" + key] for n, key in ORDER}

    K = radial_k()
    dns_u, v4e_u, v4c_u = d["dns"][0], d["field_v4e"][0], d["field_basin"][0]
    sd = dns_u.std()
    flag = (v4c_u < dns_u.min() - 3 * sd) | (v4c_u > dns_u.max() + 3 * sd)
    mask = dilate(flag, 3)
    v4c_masked = v4c_u.copy()
    v4c_masked[mask] = v4e_u[mask]
    Emask = radial(v4c_masked, K)

    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    sel = k >= 1
    ax.axvline(KCUT, color="0.75", lw=0.6, ls=":", zorder=0)
    ax.text(KCUT * 1.08, 1.6e-9, "$k_c$", color="0.45", fontsize=7, va="bottom")
    for n in ["DNS", "Kim", "v3", "v4e"]:
        ax.loglog(k[sel], stored[n][sel], color=C[n],
                  lw=1.35 if n == "DNS" else 0.9, label=n,
                  zorder=3 if n == "DNS" else 2)
    ax.loglog(k[sel], stored["v4c"][sel], color=C["v4c"], lw=1.0,
              label="v4c (apparent)", zorder=4)
    ax.loglog(np.arange(258)[1:257], Emask[1:257], color=C["v4c"], lw=1.1, ls="--",
              label="v4c (masked)", zorder=5)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$E(k)$")
    ax.set_xlim(1, 260)
    ax.set_ylim(1e-9, 1e-3)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6,
              borderaxespad=0.3, labelspacing=0.25)
    ax.tick_params(which="both", length=2.5)
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=220)
    plt.close(fig)

    print("(a) mask covers %.3f %% of the plane (%d px)" % (mask.mean() * 100, mask.sum()))
    print("    ratio to DNS at k = 32 / 64 / 128:")
    idx = [int(np.where(k == q)[0][0]) for q in (32, 64, 128)]
    for n, _ in ORDER[1:]:
        print("      %-14s " % n +
              " ".join("%7.3f" % (stored[n][i] / stored["DNS"][i]) for i in idx))
    print("      %-14s " % "v4c (masked)" +
          " ".join("%7.3f" % (Emask[q] / stored["DNS"][i]) for q, i in zip((32, 64, 128), idx)))


def panel_b(out):
    prof = np.load(EXTRA, allow_pickle=True)["profiles"].item()
    labs = [r"$u_{\mathrm{rms}}$", r"$v_{\mathrm{rms}}$", r"$w_{\mathrm{rms}}$"]
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 2.05), sharex=True)
    handles = []
    for ax, q, lab in zip(axes, ["urms", "vrms", "wrms"], labs):
        for n, key in ORDER:
            h, = ax.plot(YP, np.asarray(prof[key][q]) * 100, color=C[n], marker="o",
                         ms=1.8, lw=1.3 if n == "DNS" else 0.9, label=n)
            if q == "urms":
                handles.append(h)
        ax.set_title(lab + r"  $(\times 10^{-2})$", pad=2, fontsize=7)
        ax.set_xlabel("$y^+$")
        ax.set_xticks([15, 50, 100])
        ax.tick_params(length=2.5)
    fig.legend(handles, [n for n, _ in ORDER], frameon=False, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.20),
               handlelength=1.4, columnspacing=1.1, fontsize=6.5)
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=220)
    plt.close(fig)

    print("(b) RMS as %% of DNS, planes y+ = 15/30/50/100:")
    for q in ["urms", "vrms", "wrms"]:
        for n, key in ORDER[1:]:
            pct = [100 * v / dv for v, dv in zip(prof[key][q], prof["DNS"][q])]
            print("      %s %-4s " % (q, n) + " ".join("%5.1f%%" % v for v in pct))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outa", required=True)
    ap.add_argument("--outb", required=True)
    a = ap.parse_args()
    panel_a(a.outa)
    print()
    panel_b(a.outb)
    print("\nwrote %s\nwrote %s" % (a.outa, a.outb))
