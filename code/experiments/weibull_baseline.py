#!/usr/bin/env python3
"""CoxPH AFT baseline: run on synthetic scaling N=2-20, compare AUC vs XGBoost.

Uses per-cell survival data: time-to-EOL, event indicator, mean features."""
import sys, os, json, yaml, time, warnings
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
results_dir = cfg["execution"]["results_dir"]
os.makedirs(results_dir, exist_ok=True)

import pandas as pd
from src.data.synthetic import generate_synthetic_nasa
from src.data.composite_failure import CompositeFailureLabeler
from src.models.xgboost_hazard import XGBoostHazard
from src.evaluation.metrics import compute_metrics
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])
horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]

n_vals = [2, 3, 5, 8, 12, 20]
n_seeds = 20
output = {"xgb": {}, "coxph": {}}

for N in n_vals:
    xgb_aucs = []
    cox_aucs = []
    print(f"\nN={N} ...")

    for seed in range(n_seeds):
        syn = generate_synthetic_nasa(n_cells=N, seed=seed)
        syn = labeler.label(syn, method="single")
        X = syn[feature_cols].values.astype(np.float32)
        y = syn[horizon_cols].values.astype(np.float32)

        # Per-cell data for CoxPH AFT
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
        train_cells = all_cids[:n_train_cells]
        test_cells = all_cids[n_train_cells:]
        train_cell_idx = [i for i, cid in enumerate(all_cids) if cid in train_cells]
        test_cell_idx = [i for i, cid in enumerate(all_cids) if cid in test_cells]

        # Map cells to their cycle rows in X
        cell_row_map = {cid: np.where(syn["cell_id"].values == cid)[0]
                        for cid in all_cids}
        train_rows = np.concatenate([cell_row_map[cid] for cid in train_cells])
        test_rows = np.concatenate([cell_row_map[cid] for cid in test_cells])

        # ── XGBoost ──
        mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
        mdl.fit(X[train_rows], y[train_rows], X[test_rows], y[test_rows])
        preds = mdl.predict_proba(X[test_rows])
        m = compute_metrics(y[test_rows], preds, horizons=horizons)
        xgb_aucs.append(m["macro_avg"]["auc"] if not np.isnan(m["macro_avg"]["auc"]) else 0.5)

        # ── CoxPH AFT ──
        cox_target = Surv.from_arrays(
            event=cell_events[train_cell_idx],
            time=cell_times[train_cell_idx])

        if len(np.unique(cell_events[train_cell_idx])) < 2:
            cox_aucs.append(np.nan)
            continue

        weib = CoxPHSurvivalAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weib.fit(cell_feat_mat[train_cell_idx], cox_target)

        pred_surv = weib.predict_survival_function(cell_feat_mat[test_cell_idx])
        utimes = weib.unique_times_

        cox_h_aucs = []
        for H in horizons:
            scores, labels = [], []
            for j, cid in enumerate(test_cells):
                surv_fn = pred_surv[j]
                cycles = cell_row_map[cid]
                for k in cycles:
                    t_start = float(syn.iloc[k]["cycle"])
                    t_end = t_start + H
                    s_start = surv_fn(t_start) if t_start < utimes[-1] else surv_fn(utimes[-1])
                    s_end = surv_fn(t_end) if t_end < utimes[-1] else surv_fn(utimes[-1])
                    p = 1.0 - s_end / s_start if s_start > 0 else 1.0
                    p = max(0.0, min(1.0, p))
                    scores.append(p)
                    labels.append(y[k, horizons.index(H)])
            scores, labels = np.array(scores), np.array(labels)
            if len(np.unique(labels)) < 2:
                cox_h_aucs.append(np.nan)
            else:
                try:
                    cox_h_aucs.append(roc_auc_score(labels, scores))
                except Exception:
                    cox_h_aucs.append(np.nan)

        cox_aucs.append(np.nanmean(cox_h_aucs) if not np.all(np.isnan(cox_h_aucs)) else np.nan)

    output["xgb"][f"N={N}"] = {
        "mean": round(float(np.nanmean(xgb_aucs)), 4),
        "std": round(float(np.nanstd(xgb_aucs)), 4)}
    output["coxph"][f"N={N}"] = {
        "mean": round(float(np.nanmean(cox_aucs)), 4),
        "std": round(float(np.nanstd(cox_aucs)), 4)}
    print(f"  XGBoost: {output['xgb'][f'N={N}']['mean']:.4f} ± {output['xgb'][f'N={N}']['std']:.4f}")
    print(f"  CoxPH: {output['coxph'][f'N={N}']['mean']:.4f} ± {output['coxph'][f'N={N}']['std']:.4f}")

out_path = os.path.join(results_dir, "coxph_baseline.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# Summary
print("\n=== Summary ===")
print(f"{'N':>4}  {'XGB AUC':>10}  {'Weib AUC':>10}")
for N in n_vals:
    x = output["xgb"][f"N={N}"]
    w = output["coxph"][f"N={N}"]
    print(f"{N:>4}  {x['mean']:.4f}±{x['std']:.4f}  {w['mean']:.4f}±{w['std']:.4f}")

# Determine which performs better
xgb_better = all(
    output["xgb"][f"N={N}"]["mean"] > output["coxph"][f"N={N}"]["mean"]
    for N in n_vals)
if xgb_better:
    output["interpretation"] = "XGBoost discrete-time hazard consistently outperforms CoxPH AFT across all N values on synthetic data."
else:
    output["interpretation"] = "CoxPH AFT matches or exceeds XGBoost on some N values."

out_path = os.path.join(results_dir, "coxph_baseline.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Interpretation: {output['interpretation']}")
