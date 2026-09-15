import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_a100 import LIFT550Dataset, PROTOCOL_VARS
from normalization import load_stats, denormalize
from yasunet_v3 import YASUNetV2
from trainer_yasunet_v4 import (
    weighted_charbonnier_loss, _highk_weight, spectral_loss_2d,
    divergence_loss, gradient_loss, LAMBDA_DIV, LAMBDA_GRAD,
    lambda_spectral_for_epoch,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache_dir",  required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--n_batches", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats  = load_stats(args.stats_path)
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    epoch  = ckpt.get("epoch", "?")
    print(f"Loaded {args.checkpoint}  (epoch {epoch}, stored val_loss {ckpt.get('val_loss', float('nan')):.6f})")

    model = YASUNetV2(in_channels=3, out_channels=3, base_ch=32, embed_dim=16,
                      hidden_dim=128, modes_h=10, modes_w=10, n_refine=6).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    val_ds = LIFT550Dataset(args.cache_dir, args.stats_path, tier=args.tier, split="val")
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    sample = next(iter(loader)); _,_,H,W = sample["target"].shape
    hk = _highk_weight(H, W, device)
    lam_now = lambda_spectral_for_epoch(epoch if isinstance(epoch,int) else 50)

    comp = {"charb":0.,"spec":0.,"div":0.,"grad":0.}
    nrmse_norm_list = []; nrmse_denorm_list = []; per_yp = {}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches: break
            inp = batch["input"].to(device); target = batch["target"].to(device)
            yp = batch["yp"].to(device); yp_names = batch["yp_name"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
                pred = model(inp, yp)
            pred=pred.float(); target=target.float()
            comp["charb"] += weighted_charbonnier_loss(pred, target, yp).item()
            comp["spec"]  += spectral_loss_2d(pred, target, hk).item()
            comp["div"]   += divergence_loss(pred).item()
            comp["grad"]  += gradient_loss(pred, target).item()
            p_np = pred.cpu().numpy(); t_np = target.cpu().numpy()
            for b in range(p_np.shape[0]):
                ypn = yp_names[b]
                pd = np.empty_like(p_np[b]); td = np.empty_like(t_np[b])
                for c,var in enumerate(PROTOCOL_VARS):
                    key = f"{var}_{ypn}"
                    m,s = stats[key]["mean"], stats[key]["std"]
                    pd[c] = denormalize(p_np[b,c], m, s)
                    td[c] = denormalize(t_np[b,c], m, s)
                rm_d = np.sqrt(np.mean((pd-td)**2)); nr_d = rm_d/(np.std(td)+1e-8)
                rm_n = np.sqrt(np.mean((p_np[b]-t_np[b])**2)); nr_n = rm_n/(np.std(t_np[b])+1e-8)
                nrmse_denorm_list.append(nr_d); nrmse_norm_list.append(nr_n)
                per_yp.setdefault(int(yp[b].item()), []).append(nr_d)
            n += 1

    for k in comp: comp[k]/=n
    wc=comp["charb"]; ws=lam_now*comp["spec"]; wd=LAMBDA_DIV*comp["div"]; wg=LAMBDA_GRAD*comp["grad"]
    tot=wc+ws+wd+wg
    print(f"\n  lambda_spec at this epoch = {lam_now}")
    print("\n  -- Loss components (unweighted / weighted / pct of total) --")
    print(f"    charbonnier : {comp['charb']:.5f} / {wc:.5f} / {100*wc/tot:4.1f}%")
    print(f"    spectral    : {comp['spec']:.5f} / {ws:.5f} / {100*ws/tot:4.1f}%")
    print(f"    divergence  : {comp['div']:.6f} / {wd:.6f} / {100*wd/tot:4.1f}%")
    print(f"    gradient    : {comp['grad']:.5f} / {wg:.5f} / {100*wg/tot:4.1f}%")
    print("\n  -- NRMSE comparison --")
    print(f"    NRMSE (normalized space): {np.mean(nrmse_norm_list):.5f}")
    print(f"    NRMSE (denormalized, LOCKED metric): {np.mean(nrmse_denorm_list):.5f}")
    print(f"      ^ compare to Kim 0.1301 / v3 0.1289")
    print("    per-y+ NRMSE (denormalized):")
    for k in sorted(per_yp):
        print(f"      yp{k:<4}: {np.mean(per_yp[k]):.4f}")

if __name__ == "__main__":
    main()
