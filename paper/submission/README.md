# An Open Python Framework for Battery Operational Reliability Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Open-source Python implementation of the Shikdar–Laaksonen (2026) multihorizon hazard framework for battery operational reliability, with reproducibility analysis, methodological corrections, and an exploratory synthetic scaling study.

## Key Results

| Question | Answer |
|----------|--------|
| AUC on NASA 4-cell (per-fold, valid metric) | **0.50** (exactly random — only 2 EOL events across 3 training cells) |
| Exploratory synthetic scaling: N for AUC > 0.95 | 5–8 cells (synthetic only; not validated on real data) |
| Exploratory synthetic scaling: N for AUC > 0.98 | 12–20 cells (curve continues to improve, no plateau within N=2-20) |
| Min. failure events for meaningful AUC (synthetic) | ≥200 events across dataset (at N=12, AUC=0.91; at 100 events, drops to 0.64) |
| Real-data N requirement | Unknown — validated guidelines require ≥15 cells real-data scaling study |
| Transfer test (train synthetic N=20 → test NASA) | Macro AUC **0.88** (95% CI: [0.860, 0.900]) — distributional overlap, NOT validation of real-data training |
| Direct real-data training (NASA 4-cell) | AUC **0.50** — dataset too small for this method |
| Calibration leakage (our re-implementation bug) | +0.016–0.28 inflation; corrected AUC dropped from 0.74 to 0.50 |
| Stacked AUC (NASA, concatenated test folds) | Macro 0.45 (95% CI [0.24, 0.77]); per-horizon 0.26–0.71 — below-random due to between-cell ranking confound |
| Energy unit error | 1000× revenue overstatement (corrected in §6.2) |
| Negative control (shuffled labels) | AUC **0.53** — model does not find spurious patterns |
| Censoring sensitivity | 10% censoring → AUC drops 0.98→0.92; ≥20% → all horizons single-class |
| RSF baseline comparison (N=2-20) | RSF plateaus at AUC ~0.82 for N≥3; XGBoost reaches 0.99 — multi-horizon classification outperforms per-cycle survival modeling |
| Power to detect AUC > 0.95 (synthetic) | ≥0.99 at N ≥ 8 |
| Seed sensitivity (N=8, 10 seeds) | 0.9736 ± 0.0010 (range [0.9714, 0.9750]) |
| Synthetic vs real distribution | KS D=0.33 (p<0.001), per-feature KL 0.06–0.78 nats |

**Exploratory scaling curve (synthetic data only):** AUC rises from 0.84 (N=2) → 0.94 (N=5) → 0.97 (N=8) → 0.99 (N=20), with no plateau observed within N=2-20. A constrained fit (a ≤ 1.0) hits the boundary, confirming insufficient curvature to estimate a plateau. **These results are specific to the synthetic generator and do not transfer quantitatively to real datasets.**

## Quick Start

```bash
cd code
pip install -r requirements.txt
python experiments/run_all.py --quick
```

The `--quick` flag runs XGBoost-only experiments (baseline, dispatch, composite labels, market simulation, ablation) in under 10 seconds. Use `--expt scaling` for the full Monte Carlo scaling study (~2–3 hours). Use `--model {xgboost,lstm,tcn,transformer}` to select a specific model.

## Project Structure

```
battery_paper/
├── code/
│   ├── src/
│   │   ├── data/            # NASALoader, CALCELoader, synthetic generator
│   │   ├── models/          # XGBoostHazard, LSTM/TCN/Transformer, calibration
│   │   ├── dispatch/        # ThresholdPolicy, derating, market simulation
│   │   └── evaluation/      # Cross-validation, metrics, visualization
│   ├── experiments/         # run_all.py + transfer, censoring, min event, seed sensitivity
│   ├── results/             # JSON output from each experiment
│   ├── config.yaml          # Single configuration file (seed=42 in config)
│   └── requirements.txt
├── tests/                   # 8 unit tests (energy, calibration, integration, hazard)
├── CONTRIBUTION.md          # Detailed contribution vs Shikdar–Laaksonen (2026)
├── LICENSE                  # MIT (code)
├── LICENSE_PAPER            # CC BY 4.0 (manuscript)
├── environment.yml          # Conda environment with pinned deps
├── requirements-exact.txt   # Pip alternatives with pinned versions
├── reproduce.sh             # One-command reproduction script
├── docs/
│   ├── primer.md             # Quick primer
│   ├── submission_primer.md  # Primer copy for submission
│   ├── presentation.html     # Slide deck
│   └── submission_presentation.html  # Slide deck copy for submission
└── paper/
    ├── manuscript.md         # Full paper
    ├── Extension_Paper.docx  # Formatted Word document
    ├── figures/              # 9 publication-quality figures
    └── submission/           # Final copies for submission
```

## Methodological Corrections

Three bugs were found and fixed in our re-implementation of the Shikdar–Laaksonen framework (see [CONTRIBUTION.md](CONTRIBUTION.md) for full delineation; these bugs are in our re-implementation, not the original authors' code):

1. **Calibration data leakage** — Fitting the isotonic calibrator on the test set inflated AUC. Corrected AUC is 0.50 (not 0.74). Impact diminishes on larger datasets. Demonstrated via controlled experiment: correct calibration AUC 0.9786 vs leaked AUC 0.9947 (+0.0161 inflation).
2. **Energy unit error** — kWh × $/MWh without dividing by 1000 overstated revenue by 1000×. Corrected: $3.78 per battery across the 4-cell test period (not $3,780).
3. **Inconsistent baselines** — Baseline failure rate used label density; model used dispatch-based metric. Now both use the same conditional dispatch metric.

## Transfer Test (Synthetic → Real)

Training on synthetic data (N=20, 300 cycles each) and evaluating on real NASA 4-cell data yields **macro AUC 0.88** (95% bootstrap CI: [0.860, 0.900]), indicating the synthetic generator captures real-world structure. The CI does not include 0.50, confirming statistical significance. **This does NOT validate real-data training — the direct NASA cross-validation fails (AUC 0.50) due to insufficient failure events.** The transfer test is a distributional similarity check, not a substitute for real-data validation.

## Negative Controls

1. **Label-shuffling test:** Randomly permuted failure labels → macro AUC 0.53 (near random). Confirms no spurious pattern detection.
2. **Random data test:** Training on random Gaussian features with random binary labels → macro AUC ~0.50 across all N. Confirms the XGBoost pipeline does not produce inflated AUC on noise.

## Censoring Sensitivity

| Censoring | Macro AUC | Valid horizons |
|:---:|:---:|:---:|
| 0% | 0.9775 | 4/4 |
| 10% | 0.9239 | 1/4 |
| >20% | NaN | 0/4 |

>20% label censoring renders evaluation impossible (all horizons single-class). Datasets with heavy censoring require survival-specific methods.

## Event Count Analysis

With N=12 cells fixed, varying the number of failure events shows:

| Events | AUC | Meaning |
|:---:|:---:|:---|
| ≥200 | 0.91 | Meaningful discrimination |
| 100 | 0.64 | Near random |
| ≤50 | <0.40 | Signal destroyed |

The scaling curve improvement is primarily driven by event count, not cell count. NASA 4-cell has ~2 EOL events, consistent with AUC ~0.50.

## Citation

```bibtex
@misc{teamdynamic2026battery,
  title = {An Open Python Framework for Battery Operational Reliability Estimation},
  author = {{Team Dynamic}},
  year = {2026},
  doi = {10.5281/zenodo.20532600},
  url = {https://github.com/touhidsiddiqueeraj-bit/An-Open-Python-Framework-for-Battery-Operational-Reliability-Estimation}
}
```

## License

Code: MIT (see [LICENSE](LICENSE))  
Manuscript and figures: CC BY 4.0 (see [LICENSE_PAPER](LICENSE_PAPER))
