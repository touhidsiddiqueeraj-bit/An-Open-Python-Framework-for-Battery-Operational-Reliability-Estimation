#!/usr/bin/env python3
"""Seed sensitivity: run scaling at N=8 across seeds 0-9, report variance."""
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

print("=== SEED SENSITIVITY (N=8, seeds 0-9) ===")

n_values = [8]  # representative N
results = {"n_values": n_values, "seeds": list(range(10)), "per_seed": {}}

for seed in range(10):
    results["per_seed"][f"seed_{seed}"] = {}
    for N in n_values:
        df = generate_synthetic_nasa(n_cells=N, seed=seed * 100 + N)
        df = labeler.label(df.copy(), method="single")
        hcols = [f"fail_{h}" for h in horizons]
        cv = leave_battery_out_cv(
            df, XGBoostHazard, model_cfg, feature_cols, hcols,
            None, seed=seed, horizons=horizons)
        if len(cv["predictions"]) == 0:
            auc = None
        else:
            m = compute_metrics(cv["targets"], cv["predictions"], horizons=horizons)
            auc = m["macro_avg"]["auc"]
            if auc is not None and isinstance(auc, float) and np.isnan(auc):
                auc = None
        results["per_seed"][f"seed_{seed}"][str(N)] = auc
        print(f"  seed={seed}, N={N}: AUC={auc}")

# Summary stats
output = {"n_values": n_values, "seeds": list(range(10)), "per_seed": results["per_seed"]}
summary = {}
for N in n_values:
    aucs = [results["per_seed"][f"seed_{s}"][str(N)]
            for s in range(10)
            if results["per_seed"][f"seed_{s}"][str(N)] is not None]
    if aucs:
        summary[str(N)] = {
            "mean": round(float(np.mean(aucs)), 4),
            "std": round(float(np.std(aucs, ddof=1)), 4),
            "min": round(float(min(aucs)), 4),
            "max": round(float(max(aucs)), 4)}
        print(f"  N={N}: mean={np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}, "
              f"range=[{min(aucs):.4f}, {max(aucs):.4f}]")
output["summary"] = summary

fp = os.path.join(results_dir, "seed_sensitivity.json")
with open(fp, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {fp}")
print("Done.")
