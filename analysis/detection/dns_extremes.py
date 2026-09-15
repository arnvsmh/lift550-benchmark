# Read-only: DNS extremes (sigma units, per plane and component) over the full 5,000-sample evaluation set and the 1,233-sample audit subset.
import os
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(SP, "spike_out", "records_full_A.npz"), allow_pickle=True)
dc = list(z["dcols"]); D = z["DNS"]
YP = ["yp15", "yp30", "yp50", "yp100"]
gmax = 0.0
for pi, yp in enumerate(YP):
    s = D[D[:, dc.index("plane")] == pi]
    sub = s[s[:, dc.index("insub")] == 1]
    lo = [s[:, dc.index(f"dzlo_{c}")].min() for c in "uvw"]; hi = [s[:, dc.index(f"dzhi_{c}")].max() for c in "uvw"]
    ls = [sub[:, dc.index(f"dzlo_{c}")].min() for c in "uvw"]; hs = [sub[:, dc.index(f"dzhi_{c}")].max() for c in "uvw"]
    gmax = max(gmax, max(abs(x) for x in lo + hi))
    print(f"{yp:<6} full n={len(s):>4}  lo u/v/w {np.round(lo, 1)}  hi {np.round(hi, 1)}  | subset n={len(sub):>3}  lo {np.round(ls, 1)}  hi {np.round(hs, 1)}")
print(f"largest |DNS| over the full evaluation set: {gmax:.2f} sigma")
