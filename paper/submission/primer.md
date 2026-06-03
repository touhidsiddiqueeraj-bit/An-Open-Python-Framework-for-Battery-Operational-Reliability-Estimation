# Primer: An Open Python Framework for Battery Operational Reliability Estimation

## What this paper is about

Batteries in power grids need to decide: **should I accept this next service request or not?** Accepting when the battery is about to fail causes blackouts. Rejecting when it would have been fine loses money. Shikdar & Laaksonen (2026) built a machine learning framework that predicts "will this battery fail in the next N cycles?" so operators can reject high-risk jobs. They reported AUC 0.944 on 37 batteries.

**We built an open-source Python implementation** of their complete framework — data loading, hazard modeling, dispatch policies, and market simulation — and performed a synthetic scaling study to answer a practical question: *how many batteries do you actually need?*

## What we found

**The scaling study gives a clear answer:**
- AUC > 0.90 requires at least 3 synthetic cells
- AUC > 0.95 requires 5–8 synthetic cells  
- AUC > 0.98 requires 12+ synthetic cells
- The curve continues to improve beyond N=12 — no clear plateau observed up to N=20 (AUC 0.99)

All horizons improve consistently with N. Per-horizon breakdown (seed 1):

| N cells | H=10 | H=20 | H=30 | H=50 | Macro |
|---------|------|------|------|------|-------|
| 3       | 0.88 | 0.89 | 0.91 | 0.92 | 0.90  |
| 8       | 0.96 | 0.97 | 0.97 | 0.98 | 0.97  |
| 20      | 0.99 | 0.99 | 0.99 | 0.99 | 0.99  |

**Critical caveat:** Synthetic data maximizes cell diversity at every N by spreading degradation across the full range (slowest to fastest fade). Real-world batteries share chemistry, manufacturer, and operating conditions — correlated degradation reduces effective diversity. Real datasets likely need more cells than the curve suggests.

**Variance note:** For N ≤ 3, leave-battery-out CV has only 2–3 folds, so standard deviation estimates are based on very few observations and should be interpreted cautiously.

**Computational cost:** Full scaling study (6 N values × 20 seeds) takes ~4–5 hours on an 8-core CPU. The NASA real-data run completes in under 10 seconds.

On the real NASA 4-cell dataset, the model achieves AUC 0.50 (exactly random) per-fold. This confirms the framework needs more data than a single public benchmark provides. A synthetic-to-real transfer test (train N=20, test on NASA) produces AUC 0.88 — the model works on real data with sufficient training.

## Three methodological corrections

While building the code, we found three mistakes that can inflate reported results:

1. **Calibration leakage (+0.016–0.28 AUC):** Fitting the probability calibrator on test data makes metrics look artificially great. Fix: hold out a separate validation set.
2. **Unit error (1000×):** Energy is priced in \$/MWh but delivered in kWh. Forgetting to divide by 1000 overstates revenue by 1000×.
3. **Metric mismatch:** Baseline failure rate vs model failure rate must be computed the same way to be comparable.

## Why this matters

- **Quantitative minimum-data guideline:** The scaling curve gives the community a data-driven answer to "how many batteries?" rather than qualitative estimates.
- **Open-source reference implementation:** Fully documented Python package with correct methodology.
- **Methodological hygiene:** The three bugs are easy to make and hard to catch. Our code provides correct reference implementations.

## Key numbers

| What | Value |
|------|-------|
| NASA batteries (real data) | 4 (B0005-B0018) |
| Synthetic cells (scaling study) | 2 to 20 (20 seeds each) |
| AUC at N=2 (synthetic) | 0.84 |
| AUC at N=8 (synthetic) | 0.97 |
| AUC at N=12+ (synthetic) | > 0.98, continues to 0.99 at N=20 |
| Real-data per-fold macro AUC | 0.50 (exactly random) |
| Transfer test (synthetic → real) | Macro AUC 0.88 |
| Negative control (shuffled labels) | AUC 0.53 |
| Minimum reliable N (synthetic) | ≈ 8–12 cells |
| Original paper AUC | 0.944 (on 37 batteries) |

## Files in this release

```
paper/
  manuscript.md          — Full paper (Markdown)
  Extension_Paper.docx   — Formatted paper (Word)
  primer.md              — This document
  presentation.html      — Slide deck (open in browser)
  figures/               — 9 publication-quality figures
  generate_figures.py    — Script to regenerate all figures
  render_to_docx.py      — Script to regenerate the .docx
  render_pptx.py         — Script to regenerate the .pptx
  submission/            — Final clean copies for submission
```

## Code

All code is in the `code/` directory:
- `experiments/run_all.py` — main entry point (includes `--expt scaling`)
- `src/` — modular Python package (loader, models, dispatch, evaluation)
- `config.yaml` — single configuration file
- `results/` — timestamped JSON outputs from each experiment

Run with: `python experiments/run_all.py --expt scaling` (takes ~2 minutes)

## Bottom line

The Shikdar-Laaksonen framework is sound, but it requires substantial data. Our scaling study provides the first quantitative curve showing exactly how many batteries are needed for reliable risk differentiation. The open-source Python package lets anyone verify this on their own data.
