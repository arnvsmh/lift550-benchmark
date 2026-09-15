# LIFT-550

Code, run records and evaluation results for the paper

**A Spectral Loss Satisfied by Artifacts in Wall-Bounded Turbulence Super-Resolution**
Arnav Simha and Pranav Kulkarni

In one super-resolution network (YASU-Net) for wall-parallel planes of turbulent channel flow, a heavily weighted,
magnitude-only spectral loss is satisfied by sparse, extreme-amplitude artifacts rather than by better reconstruction.
The paper gives a procedure for detecting them: an amplitude test, masking and rescoring, and loss accounting. The
detection code is in `analysis/detection/`.

## Contents

| Folder | Contents |
|---|---|
| `code/` | Data reader, degradation pipeline, normalization (`norm_stats_train.json` holds the training-set statistics), YASU-Net and baseline models, training scripts, evaluation and metrics |
| `analysis/detection/` | Amplitude test, masking and rescoring on the full evaluation set (`spiketest.py`); Tier 2–3 checks and loss accounting on the audit subset (`phase1_checks.py`); summaries (`analyze_spike.py`, `analyze_p1.py`); DNS extremes and spike-region share. `spike_out/` holds `summary.json` and the run logs |
| `analysis/loss_accounting/` | Loss accounting with the DNS as the fill (`lossfill_dns.py`), with its summary and log |
| `analysis/coherence/` | Spectral coherence with the DNS at model seeds 42, 43 and 44: pipeline, validation gates, statistics and outputs |
| `analysis/provenance/` | Seam energy of the cached arrays, correlation between simulations at matched times, and the masked-spectrum ratios of Figure 2, each with its output |
| `figures/` | `v6/scripts/fig_v6.py` builds Figures 1–3 of the paper using the scripts in `figures/scripts/`; `verify_v6.py` checks the written PDFs |
| `configs/`, `histories/` | Configuration and per-epoch training history for each of 84 runs |
| `results/` | Evaluation summaries (see below) |

Python packages are listed in `requirements.txt`.

## Runs

YASU-Net runs are named `yasunet_<variant>_tier<T>_seed<S>`. The baseline runs (`fukami_cnn`, `fukami_dscms`, `guastoni`
and `kim`, the Kim generator) are trained by `code/trainer_baseline.py`.

Spectral-loss weight λ by epoch, from `configs/` (λ_div = 0.02 and λ_grad = 0.05 for all five variants):

| Variant | Trainer (`code/`) | Epochs 1–15 | 16–35 | 36–50 |
|---|---|---|---|---|
| v4e | `trainer_yasunet_v4e.py` | 0.05 | 0.07 | 0.07 |
| v4d | `trainer_yasunet_v4d.py` | 0.05 | 0.10 | 0.10 |
| v4f12 | `trainer_yasunet_v4f12.py` | 0.05 | 0.12 | 0.12 |
| v4clo | `trainer_yasunet_v4c_lo.py` | 0.05 | 0.10 | 0.15 |
| v4c | `trainer_yasunet_v4c.py` | 0.05 | 0.15 | 0.30 |

`configs/`, `histories/` and `results/` also hold earlier variants (v2, v3, v3fskip, v4, v4b).
`yasunet_v4c_tier3_seed42_run2` is a second run of the v4c Tier-3 seed-42 configuration: the same `config.json`, a
different training history.

## Results

Each `*_summary.json` holds `metrics_mean` and `metrics_std` for RMSE, NRMSE, SSIM and the spectral error
(`spectral_error`), per-plane values (`per_yp`), and the evaluated `epoch` and `val_loss`.

| Folder | Runs | Checkpoint |
|---|---|---|
| `results/final/` | YASU-Net runs | lowest validation loss (`val_loss` in the history) |
| `results/charbonnier_selected/` | v4e (Tiers 1–3, seeds 42–44) and v4d (Tier 1, seed 42) | lowest validation Charbonnier loss (`val_charb`) |
| `results/baselines/` | the four baselines, Tiers 1–3, seeds 42–44 | lowest validation loss (`val_loss`) |
| `results/new_seeds/` | v4c, v4clo, v4d and v4f12 at Tier 1, seeds 43–44; v4c at Tier 3, seeds 42 (second run) and 43 | final epoch, which is also the lowest validation loss |

## Running the scripts

The analysis and figure scripts are the versions that produced the paper's numbers and figures. They still contain
the absolute paths used on our machines (raw DNS files, cached arrays, checkpoints and output folders); change them
before running. `spiketest.py` and `phase1_checks.py` import the training code from a `lift_code/` folder beside them,
which should hold the contents of `code/`. In `code/evaluate_cache.py`, the branch for the original `yasunet` model
imports `yasunet_a100`, which is not included; no run here uses that model.

## Data

The DNS fields are not redistributed here; they are available from the KTH FLOW group upon request.

## Checkpoints and figure data

Trained checkpoints and figure data are not in the repository yet; they will be added.

## Acknowledgment

We thank Prof. Ricardo Vinuesa and Arivazhagan Geetha Balasubramanian for sharing the dataset used in this study.

## License

MIT; see `LICENSE`.
