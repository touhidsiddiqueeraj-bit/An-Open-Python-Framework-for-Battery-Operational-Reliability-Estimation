# Supplementary Materials

## Supplementary Figures

| Label | File | Description |
|-------|------|-------------|
| Figure S1 | `figure_s1.png` | Per-feature KL divergence (nats) between the synthetic data generator and the real NASA PCoE dataset across 7 features: SOH, voltage_avg, current_avg, temperature_avg, d_soh, d_capacity, capacity. Values range from 0.06 (SOH) to 0.78 (temperature_avg), confirming distributional differences highlighted in §4.3. |

## Supplementary Data

| File | Description |
|------|-------------|
| `../../code/results/supplementary_analysis.json` | Full JSON output from the supplementary analysis pipeline, including extended censoring sensitivity (0–80%), per-feature KL divergence, plateau assessment, power analysis, and computational cost. |
| `generate_figure_s1.py` | Script used to generate Supplementary Figure S1 from the analysis JSON. |
