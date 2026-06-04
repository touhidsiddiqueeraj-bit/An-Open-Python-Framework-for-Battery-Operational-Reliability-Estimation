# Primer: An Open Python Framework for Battery Operational Reliability Estimation

## What this paper is about

Batteries in power grids need to decide: **should I accept this next service request or not?** Accepting when the battery is about to fail causes blackouts. Rejecting when it would have been fine loses money. Shikdar & Laaksonen (2026) built a machine learning framework that predicts "will this battery fail in the next N cycles?" so operators can reject high-risk jobs. They reported AUC 0.944 on 37 batteries.

**We built an open-source Python implementation** of their complete framework — data loading, hazard modeling, dispatch policies, and market simulation — and performed an exploratory synthetic scaling study to answer: *how many failure events do you actually need?*

## What we found

**The scaling study (exploratory, synthetic data only) shows:**
- AUC > 0.90 requires ≥200 failure events (not a specific cell count)
- AUC rises from 0.84 (N=2) → 0.94 (N=5) → 0.97 (N=8) → 0.99 (N=20)
- No plateau observed within N=2-20 — the curve continues to improve
- **These are optimistic upper bounds, not validated guidelines**

Per-horizon breakdown (representative seed):

| N cells | H=10 | H=20 | H=30 | H=50 | Macro |
|---------|------|------|------|------|-------|
| 3       | 0.88 | 0.89 | 0.91 | 0.92 | 0.90  |
| 8       | 0.96 | 0.97 | 0.97 | 0.98 | 0.97  |
| 20      | 0.99 | 0.99 | 0.99 | 0.99 | 0.99  |

**Critical caveat:** The synthetic generator intentionally maximizes cell-to-cell diversity at every N. Real-world batteries share chemistry, manufacturer, and operating conditions — correlated degradation reduces effective diversity. The synthetic data distribution also differs significantly from real NASA data (KS D=0.33, p<0.001; per-feature KL divergence 0.06-0.78 nats). These results do not transfer quantitatively to real datasets.

**Event count, not cell count:** Analysis reveals that ≥200 failure events across the dataset is the key requirement, not cell count per se. At N=12 fixed, AUC drops from 0.91 (200 events) to 0.64 (100 events) to 0.10 (2 events).

**RSF baseline comparison:** We compared XGBoost against a per-cycle Random Survival Forest (RSF) using the same leave-battery-out protocol. A single RSF is trained per fold; horizon risk is extracted as P(fail within H) = 1 − S(H). RSF matches XGBoost at N=2 (AUC 0.85) but plateaus at AUC ~0.82 for N≥3, while XGBoost reaches 0.99 — the multi-horizon gradient-boosted approach extracts substantially more information from clean synthetic data.

**Censoring intolerance:** Both the discrete-time XGBoost and the continuous-time RSF fail above 20% censoring — this is a data-level limitation, not model-specific. Uno's cumulative/dynamic AUC (a censoring-robust metric) confirms that degradation at 10% censoring is genuine signal loss (Uno AUC drops from 0.83 to 0.56), not metric bias.

On the real NASA 4-cell dataset, the model achieves per-fold macro AUC 0.50 (exactly random) — only ~2 EOL events exist across training cells. A synthetic-to-real transfer test (train N=20 synthetic, test NASA) yields AUC 0.88, indicating the synthetic generator captures real-world structure, but this does not validate real-data training.

## Three methodological corrections

While building the code, we found three mistakes that can inflate reported results:

1. **Calibration leakage (+0.016–0.28 AUC):** Fitting the probability calibrator on test data makes metrics look artificially great. Fix: hold out a separate validation set.
2. **Unit error (1000×):** Energy is priced in \$/MWh but delivered in kWh. Forgetting to divide by 1000 overstates revenue by 1000× ($3.78 per battery across test period, not $3,780).
3. **Metric mismatch:** Baseline failure rate vs model failure rate must be computed the same way to be comparable.

## Why this matters

- **Event-count-based guidance:** The scaling curve shows that ≥200 failure events (not cell count) drive meaningful discrimination — actionable for practitioners designing aging studies.
- **Exploratory bounds, not minimum requirements:** The synthetic curve provides optimistic upper bounds. Validated real-data guidelines require a multi-cell real-data scaling study (15+ cells).
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
| RSF baseline (synthetic, N≥3) | AUC ~0.82 (plateaus; XGBoost reaches 0.99) |
| Uno AUC at 0% censoring | 0.71–0.92 per horizon (confirms metric not biased) |
| Transfer test (synthetic → real) | Macro AUC 0.88 — distributional overlap, NOT validation of real-data training |
| Negative control (shuffled labels) | AUC 0.53 |
| Min. failure events for AUC > 0.90 | ≥200 (synthetic) |
| Censoring tolerance | <10% for both XGBoost and RSF |
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

The Shikdar-Laaksonen framework is sound, but it requires substantial data. Our exploratory scaling study provides the first quantitative curve showing that ≥200 failure events are needed for reliable discrimination, using the generator as an optimistic upper bound. The open-source Python package lets anyone verify and extend this work on their own data.
