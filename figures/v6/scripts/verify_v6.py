#!/usr/bin/env python3
"""Check the written v6 PDFs themselves, not the matplotlib objects:
page size, fonts embedded, field rasters pixel-identical to the full-size Figure 2,
onset polylines equal to the history files, Figure C's spectrum polylines equal to the
stored spectra and the Section III-H masked spectrum, and main32.tex and the full-size
assets untouched.   Usage: python verify_v6.py [--render DIR]"""
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors
import pypdf, pdfplumber

sys.dont_write_bytecode = True                           # keep v6/scripts free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_v6 as G                                       # geometry + sources, main() not run

ROOT, OUT = G.ROOT, G.ROOT / "v6"
NAMES = list(G.NAMES.values())                           # A, B, AB, C
OLD2 = ROOT / "fig2_fields_residual.pdf"
OLD_V_ROW = {"I9": "v4e", "I10": "v4c", "I11": "DNS", "I13": "|v4c-DNS|", "I14": "zoom inset"}
UNTOUCHED = {ROOT.parent / "main32.tex": ("288AE972C990EEB647D1D4E163BC6BB6D5281835AFA45F0CBC70FEA4CA5C68DE", 101093),
             ROOT / "fig2_fields_residual.pdf": (None, 2969857),
             ROOT / "fig3a_spectrum.pdf": (None, 31737),
             ROOT / "fig4_onset.pdf": (None, 18604)}


def page_images(path):
    return {im.name.split(".")[0]: np.asarray(im.image)
            for im in pypdf.PdfReader(path).pages[0].images}


def page_fonts(path):
    out = []
    for ref in pypdf.PdfReader(path).pages[0]["/Resources"].get("/Font", {}).values():
        f = ref.get_object()
        sub = str(f.get("/Subtype"))
        d = f["/DescendantFonts"][0].get_object() if sub == "/Type0" else f
        fd = d.get("/FontDescriptor")
        fd = fd.get_object() if fd is not None else {}
        emb = sub == "/Type3" or any(k in fd for k in ("/FontFile", "/FontFile2", "/FontFile3"))
        out.append((str(f.get("/BaseFont")), sub, emb))
    return out


def polylines_vs_history(path, hist):
    x0, w = G.B_L * 72, (G.COLW - G.B_L - G.B_R) * 72
    y0, h = G.B_B * 72, (G.B_H - G.B_B - G.B_T) * 72
    ref = {s: np.array([[r["epoch"], r["val_spec"]] for r in hist[s]]) for s in hist}
    with pdfplumber.open(path) as pdf:
        pg = pdf.pages[0]
        H = float(pg.height)
        cands = [np.array(c["pts"], float) for c in pg.curves if len(c["pts"]) == len(ref[42])]
    best = {}
    for p in cands:
        for conv, yb in (("y from top", H - p[:, 1]), ("y from bottom", p[:, 1])):
            d = np.column_stack([(p[:, 0] - x0) / w * 51, (yb - y0) / h * 0.42])
            for s, R in ref.items():
                e = np.abs(d - R).max(axis=0)
                if s not in best or e.sum() < best[s][1].sum():
                    best[s] = (conv, e)
    return len(cands), best


def stroked_paths(path):
    """Stroked paths of the page content stream, in points, with the stroke state in force
    (RG colour, w, dash array, CTM). matplotlib draws a Line2D as one m l ... l S path."""
    st = dict(rgb=(0.0, 0.0, 0.0), w=1.0, dash=(), ctm=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    stack, cur, out = [], [], []
    for args, op in pypdf.PdfReader(path).pages[0].get_contents().operations:
        if op == b"q":
            stack.append(dict(st))
        elif op == b"Q":
            st = stack.pop()
        elif op == b"cm":
            a, b, c, d, e, f = map(float, args)
            A, B, C, D, E, F = st["ctm"]
            st["ctm"] = (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D,
                         e * A + f * C + E, e * B + f * D + F)
        elif op == b"w":
            st["w"] = float(args[0])
        elif op == b"RG":
            st["rgb"] = tuple(map(float, args))
        elif op == b"d":
            st["dash"] = tuple(map(float, args[0]))
        elif op in (b"m", b"l"):
            cur.append((op.decode(), float(args[0]), float(args[1])))
        elif op in (b"c", b"v", b"y", b"h", b"re"):
            cur.append((op.decode(), np.nan, np.nan))
        elif op in (b"S", b"s"):
            out.append(dict(st, ops=cur))
            cur = []
        elif op in (b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"n"):
            cur = []
    return out


def polylines_vs_spectra(path, S):
    """Each Figure C curve = the one long stroked path in its colour, width and dash, mapped back
    through the axes box and log limits to (log10 k, log10 E) and compared with the arrays:
    stored spec_yp15_* for the four apparent curves, the III-H masked spectrum for the fifth."""
    x0, w = G.C_L * 72, (G.COLW - G.C_L - G.C_R) * 72
    y0, h = G.C_B * 72, (G.C_H - G.C_B - G.C_T) * 72
    (lx0, lx1), (ly0, ly1) = np.log10(G.C_XLIM), np.log10(G.C_YLIM)
    Z = np.load(G.F3.FD)
    k = Z["spec_yp15_k"][1:]
    ref = {"DNS": Z["spec_yp15_DNS"][1:], "Kim": Z["spec_yp15_Kim"][1:], "v4e": Z["spec_yp15_v4e"][1:],
           "v4c (apparent)": Z["spec_yp15_basin"][1:], "v4c (masked)": S["E_mask"][1:257]}
    paths = [p for p in stroked_paths(path) if len(p["ops"]) > 8]   # legend handles, spines: 2
    rows = []
    for name, (_, _, sty) in G.spectrum_curves(S).items():
        hits = [p for p in paths if np.allclose(p["rgb"], mcolors.to_rgb(sty["color"]), atol=1e-6)
                and abs(p["w"] - sty["lw"]) < 1e-6 and bool(p["dash"]) == (sty.get("ls") == "--")]
        nv, e = 0, None
        if len(hits) == 1:
            ops = hits[0]["ops"]
            nv = len(ops)
            xy = np.array([o[1:] for o in ops])
            a, b, c, d, tx, ty = hits[0]["ctm"]
            X, Y = a * xy[:, 0] + c * xy[:, 1] + tx, b * xy[:, 0] + d * xy[:, 1] + ty
            data = np.column_stack([lx0 + (X - x0) / w * (lx1 - lx0), ly0 + (Y - y0) / h * (ly1 - ly0)])
            if "".join(o[0] for o in ops) == "m" + "l" * (len(k) - 1):
                e = np.abs(data - np.column_stack([np.log10(k), np.log10(ref[name])])).max(axis=0)
        rows.append((name, len(hits), nv, e, sty))
    return len(paths), rows


def main(render):
    ok = True
    hist = {s: json.load(open(p)) for s, p in G.F4.SRC.items()}
    print(f"PAGE SIZE AND FONTS (width = COLW within 1e-3 in; C also height <= {G.C_HMAX:.1f} in)")
    for n in NAMES:
        mb = pypdf.PdfReader(OUT / n).pages[0].mediabox
        wi, hi = float(mb.width) / 72, float(mb.height) / 72
        f = page_fonts(OUT / n)
        emb = all(e for _, _, e in f)
        good = emb and abs(wi - G.COLW) < 1e-3 and (n != NAMES[3] or hi <= G.C_HMAX)
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {n:28s} {wi:.3f} x {hi:.3f} in  {hi / G.TEXTH:.3f} page  "
              f"fonts embedded: {emb} {[b for b, _, _ in f]}")

    print("\nFIELD RASTERS vs full-size fig2_fields_residual.pdf (pixel-exact)")
    old = page_images(OLD2)
    newA = page_images(OUT / NAMES[0])
    for n in (NAMES[0], NAMES[2]):
        new = page_images(OUT / n)
        for k, v in new.items():
            if v.shape[:2] not in ((512, 512), (52, 52)):
                continue
            hits = [o for o, ov in old.items() if ov.shape == v.shape and np.array_equal(ov, v)]
            same_as_A = any(np.array_equal(v, a) for a in newA.values() if a.shape == v.shape)
            good = len(hits) == 1 and hits[0] in OLD_V_ROW and same_as_A
            ok &= good
            print(f"  [{'OK' if good else 'FAIL'}] {n} {k} {v.shape[1]}x{v.shape[0]} == old {hits}"
                  f" ({OLD_V_ROW.get(hits[0], '?') if hits else 'no match'})"
                  + ("" if n == NAMES[0] else f", == A: {same_as_A}"))
    n_rast = {n: len(page_images(OUT / n)) for n in NAMES}
    print(f"  embedded images per file (fields + insets + colourbar strips): {n_rast}")

    print("\nONSET POLYLINES IN THE PDF vs history.json (data units)")
    for n in (NAMES[1], NAMES[2]):
        cnt, best = polylines_vs_history(OUT / n, hist)
        for s in (42, 43, 44):
            conv, e = best[s]
            good = cnt == 3 and e[0] < 1e-3 and e[1] < 1e-5
            ok &= good
            print(f"  [{'OK' if good else 'FAIL'}] {n} seed {s}: max |d epoch| {e[0]:.1e}, "
                  f"max |d val_spec| {e[1]:.1e}  ({cnt} 50-point polylines, {conv})")

    print("\nSPECTRUM POLYLINES IN THE PDF vs arrays (data units after the log transform)")
    n_long, rows = polylines_vs_spectra(OUT / NAMES[3], G.load_spectrum())
    for name, nhit, nv, e, sty in rows:
        good = nhit == 1 and nv == 256 and e is not None and e.max() < 1e-6
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {NAMES[3]} {name:14s}: {nhit} path ({sty['lw']} pt"
              f"{', dashed' if sty.get('ls') else ''}), {nv} vertices"
              + (f", max |d log10 k| {e[0]:.1e}, max |d log10 E| {e[1]:.1e}" if e is not None else ""))
    good = n_long == 5
    ok &= good
    print(f"  [{'OK' if good else 'FAIL'}] {NAMES[3]}: {n_long} stroked paths with more than 8 vertices"
          " (the 5 curves; no v3)")

    print("\nUNTOUCHED")
    for p, (sha, size) in UNTOUCHED.items():
        got = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        good = p.stat().st_size == size and (sha is None or got == sha)
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {p}  {p.stat().st_size:,} bytes"
              + (f"  sha256 {'unchanged' if got == sha else 'CHANGED'}" if sha else ""))

    if render:
        Path(render).mkdir(parents=True, exist_ok=True)
        for n in NAMES:
            with pdfplumber.open(OUT / n) as pdf:
                pdf.pages[0].to_image(resolution=450).save(Path(render) / n.replace(".pdf", "_450dpi.png"))
        print(f"\nrendered PDFs at 450 dpi -> {render}")
    print("\nALL PDF CHECKS PASSED" if ok else "\nSOME PDF CHECKS FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=None)
    sys.exit(0 if main(ap.parse_args().render) else 1)
