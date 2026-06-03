#!/usr/bin/env python3
"""Minimum event count analysis: vary failure events while holding cell count constant.
Uses explicit label masking to control event count, not SOH threshold variation."""
import sys, os, json, yaml
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
results_dir = cfg["execution"]["results_dir"]
os.makedirs(results_dir, exist_ok=True)

from src.data.synthetic import generate_synthetic_nasa
from src.data.composite_failure import CompositeFailureLabeler
from src.models.xgboost_hazard import XGBoostHazard
from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_validation import leave_battery_out_cv

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])
horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]
window = cfg["features"]["window_size"]
model_cfg = cfg["models"]["xgboost"].copy()
model_cfg["window_size"] = window

print("=== MINIMUM EVENT COUNT ANALYSIS ===")
print()

N_CELLS = 12  # enough for signal, fast enough to run
TARGET_EVENTS = [200, 100, 50, 20, 10, 5, 2]

results = {}
for target in TARGET_EVENTS:
    print(f"  Target events: {target}")
    seed_aucs = []
    for seed in range(5):
        df = generate_synthetic_nasa(n_cells=N_CELLS, seed=seed * 100 + 21)
        df = labeler.label(df.copy(), method="single")
        hcols = [f"fail_{h}" for h in horizons]
        y = df[horizon_cols].values.astype(np.float32)
        total_events = int(y.sum())
        # Mask positive labels until only target events remain
        if total_events > target:
            pos_idx = np.where(y.ravel() == 1)[0]
            rng = np.random.default_rng(seed)
            n_remove = total_events - target
            remove_idx = rng.choice(pos_idx, n_remove, replace=False)
            y_flat = y.ravel()
            y_flat[remove_idx] = 0.0
            y = y_flat.reshape(y.shape)
        df_masked = df.copy()
        for i, h in enumerate(horizons):
            df_masked[horizon_cols[i]] = y[:, i]

        cv = leave_battery_out_cv(
            df_masked, XGBoostHazard, model_cfg, feature_cols, hcols,
            None, seed=seed, horizons=horizons)

        if len(cv["predictions"]) == 0:
            continue
        m = compute_metrics(cv["targets"], cv["predictions"], horizons=horizons)
        auc_val = m["macro_avg"]["auc"]
        if not np.isnan(auc_val):
            seed_aucs.append(auc_val)

    if seed_aucs:
        mean_auc = float(np.mean(seed_aucs))
        std_auc = float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else 0.0
        print(f"    Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        results[str(target)] = {
            "target_events": target,
            "n_cells": N_CELLS,
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "n_seeds": len(seed_aucs)}
    else:
        print(f"    No valid AUCs")
        results[str(target)] = {
            "target_events": target,
            "n_cells": N_CELLS,
            "mean_auc": None,
            "std_auc": None,
            "n_seeds": 0}

fp = os.path.join(results_dir, "min_event_analysis.json")
with open(fp, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {fp}")
print("Done.")
