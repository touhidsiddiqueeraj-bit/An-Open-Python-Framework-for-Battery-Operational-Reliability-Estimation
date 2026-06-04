#!/usr/bin/env python3
"""Run scaling at N=30 and N=50, 20 seeds each. Append to existing results."""
import os, sys, json, time, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.synthetic import generate_synthetic_nasa
from src.data.composite_failure import CompositeFailureLabeler
from src.models.xgboost_hazard import XGBoostHazard
from src.models.calibration import ProbabilityCalibrator
from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_validation import leave_battery_out_cv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")

cfg = {
    "horizons": [10, 20, 30, 50],
    "failure": {"soh_threshold": 0.70, "sudden_drop_threshold": 0.05},
    "calibration": {"method": "isotonic"},
    "features": {
        "input_cols": ["cycle", "capacity", "soh", "voltage_avg", "current_avg",
                        "temperature_avg", "d_capacity", "d_soh"],
        "derived_cols": [],
        "window_size": 20,
    },
    "models": {
        "xgboost": {
            "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
            "early_stopping_rounds": 20, "window_size": 20,
        }
    },
}

def bootstrap_ci(arr, n_bootstrap=10000, seed=42):
    arr = np.array(arr, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means.append(np.nanmean(arr[idx]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def run_n(N, n_seeds=20):
    horizons = cfg["horizons"]
    feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]
    window = cfg["features"]["window_size"]
    model_cfg = cfg["models"]["xgboost"].copy()
    model_cfg["window_size"] = window
    calibrator = ProbabilityCalibrator(method=cfg["calibration"]["method"])

    seed_aucs = []
    for seed in range(n_seeds):
        t0 = time.time()
        df = generate_synthetic_nasa(n_cells=N, n_cycles=300, seed=seed * 100 + N)
        labeler = CompositeFailureLabeler(
            soh_threshold=cfg["failure"]["soh_threshold"],
            sudden_drop=cfg["failure"]["sudden_drop_threshold"])
        df = labeler.label(df.copy(), method="single")
        horizon_cols = [f"fail_{h}" for h in horizons]

        cv = leave_battery_out_cv(
            df, XGBoostHazard, model_cfg, feature_cols, horizon_cols,
            calibrator, seed=seed, horizons=horizons)

        if len(cv["predictions"]) == 0:
            auc = float("nan")
        else:
            m = compute_metrics(cv["targets"], cv["predictions"], horizons=horizons)
            auc = m["macro_avg"]["auc"]
        elapsed = time.time() - t0
        seed_aucs.append(auc)
        print(f"  N={N} seed={seed+1}/{n_seeds} AUC={auc:.4f} ({elapsed:.1f}s)")
        sys.stdout.flush()

    mean_auc = round(float(np.nanmean(seed_aucs)), 4)
    std_auc = round(float(np.nanstd(seed_aucs, ddof=1)), 4) if len(seed_aucs) > 1 else 0.0
    lo, hi = bootstrap_ci(seed_aucs)
    print(f"  N={N}: mean={mean_auc} +/- {std_auc}  CI=[{lo:.4f}, {hi:.4f}]")
    return {
        "N": N, "mean": mean_auc, "std": std_auc,
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "per_seed": seed_aucs
    }

if __name__ == "__main__":
    for N in [50]:
        print(f"\n=== N={N} ===")
        res = run_n(N, n_seeds=20)
        path = os.path.join(RESULTS_DIR, f"scaling_extended_N{N}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved: {path}")
    print("\nDone.")
