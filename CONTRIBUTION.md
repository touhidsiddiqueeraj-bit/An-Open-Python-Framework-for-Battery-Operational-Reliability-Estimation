# Contribution Statement

This repository implements and extends the multihorizon discrete-time hazard framework for battery operational reliability introduced by Shikdar and Laaksonen (2026) [1].

## What is from the original paper (§§2–3)

The core framework — hazard modeling via per-horizon XGBoost classifiers, isotonic probability calibration, threshold and derating dispatch policies, and AR(1) market simulation — follows the architecture described in Shikdar and Laaksonen (2026). The framework structure (section numbering 2–3 in the manuscript) replicates their design.

## What is novel (§§4–9)

| Section | Contribution |
|---------|-------------|
| §4 | Synthetic data generator with statistical validation (KS test vs real data), negative control experiment, and diversity-maximizing parameter scaling |
| §5.1 | Real-data case study on NASA 4-cell dataset identifying data insufficiency and demonstrating synthetic-to-real transfer |
| §5.2 | Monte Carlo scaling study (20 seeds, N=2–20) establishing quantitative minimum-data guidelines with bootstrap CIs and DeLong tests |
| §5.2.1 | SOH threshold sensitivity analysis |
| §6 | Identification and correction of three methodological pitfalls: calibration data leakage, energy unit error, inconsistent baselines |
| §7–8 | Discussion of limitations and conclusions distinguishing demonstrated vs assumed results |
| §9 | Open-source release with pinned environment and reproducibility scripts |

## Bugs found and fixed (§6)

All three bugs documented in §6 were identified in our own re-implementation of the Shikdar–Laaksonen framework. They are not errors in the original authors' published code or paper. The original authors' conceptual framework — multihorizon hazard modeling for battery operational reliability — remains valid and is the foundation of this work.

## References

[1] T. A. Shikdar and H. Laaksonen, "Learning When Not to Use a Battery: Multihorizon Failure Intelligence," *International Transactions on Electrical Energy Systems*, vol. 2026, no. 1, p. 6000810, 2026.
