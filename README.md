# An Open Python Framework for Battery Operational Reliability Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Open-source Python implementation of the Shikdar–Laaksonen (2026) multihorizon hazard framework for battery operational reliability, with a quantitative synthetic scaling study establishing minimum-data guidelines.

## Key Results

| Question | Answer |
|----------|--------|
| AUC on NASA 4-cell (per-fold, valid metric) | **0.50** (exactly random — only 2 EOL events across 3 training cells) |
| Minimum N for AUC > 0.95 on synthetic data | 5–8 cells |
| Minimum N for AUC > 0.98 on synthetic data | 12–20 cells (curve continues to improve, no plateau) |
| Real-data N requirement | Unknown — likely ≥50; unvalidated (see §7.2) |
| Transfer test (train synthetic N=20 → test NASA) | Macro AUC **0.88** — framework works on real data with sufficient training |
| Calibration leakage inflation | +0.02–0.28 AUC (corrected in §6.1) |
| Energy unit error | 1000× revenue overstatement (corrected in §6.2) |
| Synthetic negative control (shuffled labels) | AUC **0.53** — model does not find spurious patterns |

**Scaling curve:** AUC rises from 0.84 (N=2) → 0.94 (N=5) → 0.97 (N=8) → 0.99 (N=20), with continued improvement and no clear plateau.

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
│   ├── experiments/         # run_all.py entry point + transfer/censoring tests
│   ├── results/             # JSON output from each experiment
│   ├── config.yaml          # Single configuration file (seed=42 in config)
│   └── requirements.txt
├── tests/                   # Unit tests (energy, calibration, integration)
├── CONTRIBUTION.md          # Detailed contribution vs Shikdar–Laaksonen (2026)
├── environment.yml          # Conda environment with pinned deps
├── requirements-exact.txt   # Pip alternatives with pinned versions
├── reproduce.sh             # One-command reproduction script
└── paper/
    ├── manuscript.md         # Full paper
    ├── Extension_Paper.docx  # Formatted Word document
    ├── presentation.html     # Slide deck
    ├── primer.md             # Quick primer
    ├── figures/              # 7 publication-quality figures
    └── submission/           # Final copies for submission
```

## Methodological Corrections

Three bugs were found and fixed in our re-implementation of the Shikdar–Laaksonen framework (see [CONTRIBUTION.md](CONTRIBUTION.md) for full delineation):

1. **Calibration data leakage** — Fitting the isotonic calibrator on the test set inflated AUC. Corrected AUC is 0.50 (not 0.74). Impact diminishes on larger datasets.
2. **Energy unit error** — kWh × $/MWh without dividing by 1000 overstated revenue by 1000×. Corrected: $3.78 (not $3,780).
3. **Inconsistent baselines** — Baseline failure rate used label density; model used dispatch-based metric. Now both use the same conditional dispatch metric.

## Transfer Test (Synthetic → Real)

Training on synthetic data (N=20, 300 cycles each) and evaluating on real NASA 4-cell data yields **macro AUC 0.88**, confirming the framework works on real data when sufficient training examples are available.

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

MIT
