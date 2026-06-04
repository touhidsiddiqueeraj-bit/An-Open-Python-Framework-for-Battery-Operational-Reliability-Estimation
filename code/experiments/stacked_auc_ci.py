#!/usr/bin/env python3
"""Compute bootstrap CI for stacked AUC on NASA 4-cell dataset.

Output: horizon-level stacked AUC + 95% bootstrap CI.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.nasa import NASALoader
from src.data.composite_failure import CompositeFailureLabeler
from src.data.synthetic import generate_synthetic_nasa
from src.models.xgboost_hazard import XGBoostHazard
from src.models.calibration import ProbabilityCalibrator
from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_validation import leave_battery_out_cv
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")

def load_nasa_or_synthetic(data_dir):
    loader = NASALoader(data_dir=data_dir)
    df = loader.load_classic()
    if df.empty:
        loader.print_download_instructions()
        df = generate_synthetic_nasa(n_cells=4, seed=42)
    return df

def bootstrap_auc(y_true, y_score, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yi = y_true[idx]
        si = y_score[idx]
        if len(np.unique(yi)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yi, si))
        except:
            continue
    if len(aucs) < 100:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def main():
    cfg = {
        "execution": {"data_dir": os.path.join(BASE, "data")},
        "horizons": [10, 20, 30, 50],
        "failure": {"soh_threshold": 0.70, "sudden_drop_threshold": 0.05},
        "models": {
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 20,
                "window_size": 20,
            }
        },
        "calibration": {"method": "isotonic"},
        "features": {
            "input_cols": [
                "cycle", "capacity", "soh", "voltage_avg", "current_avg",
                "temperature_avg", "d_capacity", "d_soh",
            ],
            "window_size": 20,
        },
        "dispatch": {"tau_range": [0.1, 0.2, 0.3]},
    }

    df = load_nasa_or_synthetic(cfg["execution"]["data_dir"])
    labeler = CompositeFailureLabeler(
        soh_threshold=cfg["failure"]["soh_threshold"],
        sudden_drop=cfg["failure"]["sudden_drop_threshold"])
    df = labeler.label(df.copy(), method="single")

    horizons = cfg["horizons"]
    horizon_cols = [f"fail_{h}" for h in horizons]
    feature_cols = cfg["features"]["input_cols"]

    t0 = time.time()
    cv = leave_battery_out_cv(
        df, XGBoostHazard, cfg["models"]["xgboost"],
        feature_cols, horizon_cols, ProbabilityCalibrator(method="isotonic"),
        horizons=horizons)
    elapsed = time.time() - t0

    targets = cv["targets"]
    predictions = cv["predictions"]

    # Per-fold per-horizon AUC (mean ± std)
    per_fold_aucs = {h: [] for h in horizons}
    for fold in cv["per_fold_raw"]:
        for h in horizons:
            ph = fold["per_horizon"]
            v = ph.get(h) or ph.get(str(h))
            if v is None:
                v = float("nan")
            else:
                v = v.get("auc") if isinstance(v, dict) else v
                if v is None:
                    v = float("nan")
            per_fold_aucs[h].append(v if v is not None else float("nan"))

    print("=== Per-fold AUC (averaged per horizon) ===")
    for h in horizons:
        vals = np.array(per_fold_aucs[h])
        mu = np.nanmean(vals)
        sd = np.nanstd(vals, ddof=1) if np.sum(~np.isnan(vals)) > 1 else 0
        nan_count = int(np.sum(np.isnan(vals)))
        print(f"  H={h}: {mu:.4f} ± {sd:.4f}  ({nan_count}/{len(vals)} NaN folds)")

    print("\n=== Stacked AUC with 95% bootstrap CI ===")
    for h_idx, h in enumerate(horizons):
        yt = targets[:, h_idx]
        yp = predictions[:, h_idx]
        if len(np.unique(yt)) < 2:
            print(f"  H={h}: single class in stacked predictions — skipping")
            continue
        stacked_auc = roc_auc_score(yt, yp)
        mean_auc, ci_lo, ci_hi = bootstrap_auc(yt, yp, n_bootstrap=10000)
        print(f"  H={h}: stacked AUC = {stacked_auc:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Macro average of stacked horizons
    print("\n=== Macro-averaged stacked AUC ===")
    macro_aucs = []
    macro_cis = []
    for h_idx, h in enumerate(horizons):
        yt = targets[:, h_idx]
        yp = predictions[:, h_idx]
        if len(np.unique(yt)) < 2:
            continue
        mean_a, lo, hi = bootstrap_auc(yt, yp, n_bootstrap=10000)
        macro_aucs.append(mean_a)
        macro_cis.append((lo, hi))
    if macro_aucs:
        print(f"  Mean macro AUC = {np.mean(macro_aucs):.4f}")
        print(f"  CI range: [{np.min([c[0] for c in macro_cis]):.4f}, {np.max([c[1] for c in macro_cis]):.4f}]")

    # Also report per-fold (correct) macro-avg with CI
    print("\n=== Per-fold macro AUC (correct metric) ===")
    macro_per_fold = []
    for fold in cv["per_fold_raw"]:
        aucs_fold = []
        for h in horizons:
            ph = fold["per_horizon"]
            v = ph.get(h) or ph.get(str(h))
            if isinstance(v, dict):
                v = v.get("auc")
            if v is not None and not np.isnan(v):
                aucs_fold.append(v)
        if aucs_fold:
            macro_per_fold.append(np.mean(aucs_fold))
    macro_per_fold = np.array(macro_per_fold)
    valid = ~np.isnan(macro_per_fold)
    if valid.sum() > 0:
        mu = np.mean(macro_per_fold[valid])
        sd = np.std(macro_per_fold[valid], ddof=1)
        print(f"  Mean: {mu:.4f} ± {sd:.4f}")
        # Bootstrap CI on per-fold macro AUC
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(10000):
            idx = rng.integers(0, len(macro_per_fold), size=len(macro_per_fold))
            s = macro_per_fold[idx]
            s_valid = s[~np.isnan(s)]
            if len(s_valid) > 0:
                boot.append(np.mean(s_valid))
        if boot:
            print(f"  95% CI: [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")

    print(f"\nRuntime: {elapsed:.2f}s")
    print("Done.")

if __name__ == "__main__":
    main()
