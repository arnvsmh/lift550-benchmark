import json, time
from pathlib import Path

ACTIVE = [
    ("yasunet_v4e_tier3_seed44",  ".241  v4e T3 s44"),
    ("yasunet_v4f12_tier1_seed42",".156  lambda=0.12 probe"),
    ("yasunet_v4e_tier2_seed42",  ".155  chain A2 leg2"),
    ("yasunet_v4e_tier2_seed43",  ".162  chain B2 leg2"),
    ("yasunet_v4e_tier2_seed44",  ".230  chain C2 leg2"),
    ("yasunet_v4e_tier1_seed43",  ".142  chain D2 leg2"),
    ("yasunet_v4e_tier1_seed44",  ".220  chain E leg2"),
    ("yasunet_v4e_tier3_seed42",  ".144  chain F leg2"),
    ("yasunet_v4e_tier3_seed43",  ".163  chain G leg2"),
]
print(f"{'run':<28} {'pod/role':<22} {'ep':>3} {'val':>9} {'charb':>8} {'spec':>7} {'lam':>5} {'upd':>6}")
print("-"*92)
for run, label in ACTIVE:
    h = Path(f"runs_final/{run}/history.json")
    if not h.exists():
        print(f"{run:<28} {label:<22} {'--':>3} {'starting/queued':>9}")
        continue
    e = json.load(open(h))[-1]
    age = (time.time() - h.stat().st_mtime)/60
    print(f"{run:<28} {label:<22} {e['epoch']:>3} {e['val_loss']:>9.5f} "
          f"{e.get('val_charb',float('nan')):>8.5f} {e.get('val_spec',float('nan')):>7.4f} "
          f"{e.get('lambda_s', e.get('lam_s','')):>5} {age:>5.0f}m")
