import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset
from yasunet_v3 import YASUNetV2
from trainer_yasunet_v4 import (
    weighted_charbonnier_loss, _highk_weight, spectral_loss_2d,
    divergence_loss, gradient_loss, nearwall_weight,
    LAMBDA_DIV, LAMBDA_GRAD, lambda_spectral_for_epoch,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache_dir",  required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--n_batches", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {args.checkpoint}")
    print(f"  checkpoint epoch: {epoch}, stored val_loss: {ckpt.get('val_loss', float('nan')):.6f}")

    model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16,
                      hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    val_ds = LIFT550Dataset(args.cache_dir, args.stats_path, tier=args.tier, split="val")
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    sample = next(iter(loader))
    _, _, H, W = sample["target"].shape
    hk = _highk_weight(H, W, device)
    lam_now = lambda_spectral_for_epoch(epoch if isinstance(epoch,int) else 50)

    agg = {"charb":0.,"spec":0.,"div":0.,"grad":0.,"nrmse":0.}
    per_yp = {}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches: break
            inp = batch["input"].to(device); target = batch["target"].to(device)
            yp = batch["yp"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
                pred = model(inp, yp)
            pred=pred.float(); target=target.float()
            lc = weighted_charbonnier_loss(pred, target, yp).item()
            ls = spectral_loss_2d(pred, target, hk).item()
            ld = divergence_loss(pred).item()
            lg = gradient_loss(pred, target).item()
            num = torch.sqrt(((pred-target)**2).mean(dim=(1,2,3)))
            den = torch.sqrt((target**2).mean(dim=(1,2,3)))+1e-12
            nr = (num/den)
            agg["charb"]+=lc; agg["spec"]+=ls; agg["div"]+=ld; agg["grad"]+=lg
            agg["nrmse"]+=nr.mean().item(); n+=1
            for b in range(pred.shape[0]):
                k=int(yp[b].item()); per_yp.setdefault(k,[]).append(nr[b].item())

    for k in agg: agg[k]/=n
    print(f"\n  Using lambda_spec={lam_now} (for the weighted contribution below)")
    print("\n  -- Loss components (unweighted) --")
    print(f"    charbonnier (near-wall wtd): {agg['charb']:.5f}")
    print(f"    spectral (2D log high-k)   : {agg['spec']:.5f}")
    print(f"    divergence                 : {agg['div']:.6f}")
    print(f"    gradient                   : {agg['grad']:.5f}")
    print("\n  -- Weighted contribution to total loss --")
    wc = agg['charb']; ws = lam_now*agg['spec']; wd = LAMBDA_DIV*agg['div']; wg = LAMBDA_GRAD*agg['grad']
    tot = wc+ws+wd+wg
    print(f"    charbonnier : {wc:.5f}  ({100*wc/tot:4.1f}% of total)")
    print(f"    spectral    : {ws:.5f}  ({100*ws/tot:4.1f}% of total)")
    print(f"    divergence  : {wd:.6f}  ({100*wd/tot:4.1f}% of total)")
    print(f"    gradient    : {wg:.5f}  ({100*wg/tot:4.1f}% of total)")
    print(f"    TOTAL       : {tot:.5f}")
    print("\n  -- True NRMSE (comparable to Kim 0.1301 / v3 0.1289) --")
    print(f"    overall NRMSE: {agg['nrmse']:.5f}  (over {n} batches)")
    print("    per-y+ NRMSE:")
    for k in sorted(per_yp):
        print(f"      yp{k:<4}: {np.mean(per_yp[k]):.4f}")

if __name__ == "__main__":
    main()
