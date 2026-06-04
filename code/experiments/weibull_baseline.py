#!/usr/bin/env python3
"""Weibull AFT baseline: run on synthetic N=8,20, compare AUC vs XGBoost."""
import os, sys, json, time, warnings
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

import pandas as pd
from src.data.synthetic import generate_synthetic_nasa
from src.data.composite_failure import CompositeFailureLabeler
from src.models.xgboost_hazard import XGBoostHazard
from src.evaluation.metrics import compute_metrics
from sklearn.metrics import roc_auc_score
from lifelines import WeibullAFTFitter

labeler = CompositeFailureLabeler(soh_threshold=0.70, sudden_drop=0.05)
horizons = [10, 20, 30, 50]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = ["capacity", "soh", "voltage_avg", "temperature_avg"]
# Reduced feature set: cycle is constant (mean 150.5), current_avg constant (C/2),
# d_capacity/d_soh nearly collinear with capacity/soh.
# Weibull AFT with formula-based regression fails to converge on >4 features with N≤20.
results_dir = "results"

n_vals = [8, 20]
n_seeds = 10
output = {"xgb": {}, "weibull": {}}

for N in n_vals:
    xgb_aucs = []
    weib_aucs = []
    print(f"\nN={N} ...")

    for seed in range(n_seeds):
        syn = generate_synthetic_nasa(n_cells=N, seed=seed)
        syn = labeler.label(syn, method="single")

        # Per-cell data for Weibull AFT
        cell_eol = syn.groupby("cell_id")["eol_cycle"].first()
        cell_feat = syn.groupby("cell_id")[feature_cols].mean()
        all_cids = sorted(syn["cell_id"].unique())

        cell_times = np.array([
            float(cell_eol[cid]) if pd.notna(cell_eol[cid]) else 300.0
            for cid in all_cids])
        cell_events = np.array([
            pd.notna(cell_eol[cid]) for cid in all_cids])
        cell_feat_mat = cell_feat.loc[all_cids].values.astype(np.float64)

        # Cell-level train/test split (80/20)
        n_cells = len(all_cids)
        n_train_cells = int(n_cells * 0.8)
        train_cells = list(all_cids[:n_train_cells])
        test_cells = list(all_cids[n_train_cells:])
        train_cell_idx = [i for i, cid in enumerate(all_cids) if cid in train_cells]
        test_cell_idx = [i for i, cid in enumerate(all_cids) if cid in test_cells]

        cell_row_map = {cid: np.where(syn["cell_id"].values == cid)[0]
                        for cid in all_cids}
        train_rows = np.concatenate([cell_row_map[cid] for cid in train_cells])
        test_rows = np.concatenate([cell_row_map[cid] for cid in test_cells])

        # ── XGBoost (per-cycle) ──
        X_all = syn[feature_cols].values.astype(np.float32)
        y_all = syn[horizon_cols].values.astype(np.float32)
        mdl = XGBoostHazard(config={"n_estimators": 300, "max_depth": 4,
                                     "learning_rate": 0.05, "min_child_weight": 5,
                                     "subsample": 0.8, "colsample_bytree": 0.8,
                                     "early_stopping_rounds": 20})
        mdl.fit(X_all[train_rows], y_all[train_rows], X_all[test_rows], y_all[test_rows])
        preds = mdl.predict_proba(X_all[test_rows])
        m = compute_metrics(y_all[test_rows], preds, horizons=horizons)
        xgb_aucs.append(m["macro_avg"]["auc"] if not np.isnan(m["macro_avg"]["auc"]) else 0.5)

        # ── Weibull AFT ──
        cell_df = pd.DataFrame({
            "T": cell_times[train_cell_idx],
            "E": cell_events[train_cell_idx].astype(bool),
        })
        for idx_f, fname in enumerate(feature_cols):
            cell_df[fname] = cell_feat_mat[train_cell_idx, idx_f]


        weib = WeibullAFTFitter(penalizer=1.0)
        weib._scipy_fit_method = "SLSQP"
        try:
            formula = "+".join(feature_cols)
            cell_df_fit = cell_df.copy()
            cell_df_fit["T"] = cell_df_fit["T"] / 100.0
            weib.fit(cell_df_fit, duration_col="T", event_col="E",
                     formula=formula, show_progress=False)
        except Exception:
            weib_aucs.append(np.nan)
            continue

        try:
            test_df = pd.DataFrame(cell_feat_mat[test_cell_idx], columns=feature_cols)
            surv_scaled = weib.predict_survival_function(test_df)
            surv = surv_scaled.copy()
            surv.index = surv.index * 100.0
        except Exception:
            weib_aucs.append(np.nan)
            continue

        weib_h_aucs = []
        for H in horizons:
            scores, labels = [], []
            for j, cid in enumerate(test_cells):
                surv_fn = surv.iloc[:, j]
                times = surv_fn.index.values
                surv_vals = surv_fn.values
                cycles = cell_row_map[cid]
                for k in cycles:
                    t_start = float(syn.iloc[k]["cycle"])
                    t_end = t_start + H
                    # Nearest-neighbor lookup (closest time ≤ query)
                    idx_start = np.searchsorted(times, t_start, side="right") - 1
                    idx_end = np.searchsorted(times, t_end, side="right") - 1
                    s_start = surv_vals[max(0, idx_start)]
                    s_end = surv_vals[max(0, idx_end)]
                    p = 1.0 - s_end / s_start if s_start > 0 else 1.0
                    p = max(0.0, min(1.0, p))
                    scores.append(p)
                    labels.append(y_all[k, horizons.index(H)])
            scores, labels = np.array(scores), np.array(labels)
            if len(np.unique(labels)) < 2:
                weib_h_aucs.append(np.nan)
            else:
                try:
                    weib_h_aucs.append(roc_auc_score(labels, scores))
                except Exception:
                    weib_h_aucs.append(np.nan)

        weib_aucs.append(np.nanmean(weib_h_aucs) if not np.all(np.isnan(weib_h_aucs)) else np.nan)

    output["xgb"][f"N={N}"] = {
        "mean": round(float(np.nanmean(xgb_aucs)), 4),
        "std": round(float(np.nanstd(xgb_aucs)), 4)}
    output["weibull"][f"N={N}"] = {
        "mean": round(float(np.nanmean(weib_aucs)), 4),
        "std": round(float(np.nanstd(weib_aucs)), 4)}
    print(f"  XGBoost: {output['xgb'][f'N={N}']['mean']:.4f} ± {output['xgb'][f'N={N}']['std']:.4f}")
    print(f"  Weibull: {output['weibull'][f'N={N}']['mean']:.4f} ± {output['weibull'][f'N={N}']['std']:.4f}")

out_path = os.path.join(results_dir, "weibull_baseline.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print("\n=== Summary ===")
print(f"{'N':>4}  {'XGB AUC':>10}  {'Weib AUC':>10}")
for N in n_vals:
    x = output["xgb"][f"N={N}"]
    w = output["weibull"][f"N={N}"]
    print(f"{N:>4}  {x['mean']:.4f}±{x['std']:.4f}  {w['mean']:.4f}±{w['std']:.4f}")

xgb_better = all(
    output["xgb"][f"N={N}"]["mean"] > output["weibull"][f"N={N}"]["mean"]
    for N in n_vals)
output["interpretation"] = (
    "XGBoost discrete-time hazard outperforms Weibull AFT across all N values on synthetic data."
    if xgb_better else
    "Weibull AFT matches or exceeds XGBoost on some N values.")
print(f"Interpretation: {output['interpretation']}")

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
