# An Open Python Framework for Battery Operational Reliability Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Open-source Python implementation of the Shikdar–Laaksonen (2026) multihorizon hazard framework for battery operational reliability, with a quantitative synthetic scaling study establishing minimum-data guidelines.

## Key Results

| Question | Answer |
|----------|--------|
| AUC on NASA 4-cell dataset | 0.46 (near-random — only 4 cells) |
| Minimum N for reliable AUC (synthetic) | 8–12 cells (AUC > 0.98) |
| Estimated N for reliable AUC (real data) | 15–25 cells |
| Calibration leakage inflation found | +0.28 AUC |
| Energy unit error found | 1000× revenue overstatement |

**Scaling curve:** AUC rises from 0.84 (N=2) → 0.97 (N=8) → 0.99 (N=50), plateauing above N=12.

## Quick Start

```bash
cd code
pip install -r requirements.txt
python experiments/run_all.py --quick
```

The `--quick` flag runs XGBoost-only experiments (baseline, dispatch, composite labels, market simulation, ablation) in under 10 seconds. Use `--expt scaling` for the full Monte Carlo scaling study (~2–3 hours).

## Project Structure

```
battery_paper/
├── code/
│   ├── src/
│   │   ├── data/            # NASALoader, CALCELoader, synthetic generator
│   │   ├── models/          # XGBoostHazard, LSTM/TCN/Transformer, calibration
│   │   ├── dispatch/        # ThresholdPolicy, derating, market simulation
│   │   └── evaluation/      # Cross-validation, metrics, visualization
│   ├── experiments/         # run_all.py entry point
│   ├── results/             # JSON output from each experiment
│   ├── config.yaml          # Single configuration file
│   └── requirements.txt
└── paper/
    ├── manuscript.md         # Full paper
    ├── Extension_Paper.docx  # Formatted Word document
    ├── presentation.html     # Slide deck
    ├── primer.md             # Quick primer
    ├── figures/              # 7 publication-quality figures
    └── submission/           # Final copies for submission
```

## Methodological Corrections

Three bugs were found and fixed in the original codebase:

1. **Calibration data leakage** — Fitting the isotonic calibrator on the test set inflated AUC from 0.46 to 0.74 (+0.28).
2. **Energy unit error** — kWh × $/MWh without dividing by 1000 overstated revenue by 1000×.
3. **Inconsistent baselines** — Baseline failure rate used label density; model rows used dispatch-based metric.

## Citation

```bibtex
@misc{teamdynamic2026battery,
  title = {An Open Python Framework for Battery Operational Reliability Estimation},
  author = {{Team Dynamic}},
  year = {2026},
  doi = {10.5281/zenodo.15089441},
  url = {https://github.com/touhidsiddiqueeraj-bit/An-Open-Python-Framework-for-Battery-Operational-Reliability-Estimation}
}
```

## License

MIT
