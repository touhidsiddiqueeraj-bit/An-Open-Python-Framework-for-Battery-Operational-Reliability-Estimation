#!/usr/bin/env python3
"""Uno's cumulative/dynamic AUC for censoring scenarios.

Uses simple 80/20 train/test split (same as supplementary_analysis.py
censoring experiment) for apples-to-apples comparison."""
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
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])
horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]

n_cells = 20
n_seeds = 5
censoring_levels = [0.0, 0.1, 0.2]
output = {}

for cl in censoring_levels:
    cl_key = f"censor_{int(cl*100)}%"
    std_aucs = {h: [] for h in horizons}
    uno_aucs = {h: [] for h in horizons}

    for seed in range(n_seeds):
        np.random.seed(seed)
        df = generate_synthetic_nasa(n_cells=n_cells, n_cycles=300, seed=seed * 100 + n_cells)
        df = labeler.label(df.copy(), method="single")
        X = df[feature_cols].values.astype(np.float32)
        y = df[horizon_cols].values.astype(np.float32)
        cell_eol = df.groupby("cell_id")["eol_cycle"].first().to_dict()

        # Apply censoring (same approach as supplementary_analysis.py)
        if cl > 0:
            rng = np.random.default_rng(int(cl * 100))
            for i, col in enumerate(horizon_cols):
                n_censor = int(len(y) * cl)
                pos_idx = np.where(y[:, i] == 1)[0]
                if len(pos_idx) == 0:
                    continue
                n_censor = min(n_censor, len(pos_idx))
                if n_censor == 0:
                    continue
                censor_idx = rng.choice(pos_idx, n_censor, replace=False)
                y[censor_idx, i] = 0

        # Simple 80/20 split
        split = int(len(X) * 0.8)
        mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
        mdl.fit(X[:split], y[:split], X[split:], y[split:])
        preds = mdl.predict_proba(X[split:])

        # Standard AUC per horizon
        m = compute_metrics(y[split:], preds, horizons=horizons)
        for h_idx, H in enumerate(horizons):
            auc_val = m["per_horizon"][H]["auc"]
            std_aucs[H].append(auc_val if auc_val is not None else np.nan)

        # Uno's cumulative dynamic AUC
        # Build survival data for train and test splits (aligned with risk scores)
        all_cycles = df["cycle"].values
        all_eol = df["eol_cycle"].values
        all_cid = df["cell_id"].values

        train_times, train_events = [], []
        test_times_list, test_events_list = [], []
        risk_scores_list = None

        for i in range(len(X)):
            cid = all_cid[i]
            cyc = all_cycles[i]
            eol_val = cell_eol[cid]
            if pd.notna(eol_val) and cyc > eol_val:
                continue
            rem = (eol_val - cyc) if pd.notna(eol_val) else (300.0 - cyc + 1)
            evt = 1 if (pd.notna(eol_val) and cyc == eol_val) else 0
            if i < split:
                train_times.append(rem)
                train_events.append(evt)
            else:
                test_times_list.append(rem)
                test_events_list.append(evt)

        train_times = np.array(train_times)
        train_events = np.array(train_events).astype(bool)
        test_times = np.array(test_times_list)
        test_events = np.array(test_events_list).astype(bool)

        surv_train = Surv.from_arrays(event=train_events, time=train_times)
        surv_test = Surv.from_arrays(event=test_events, time=test_times)

        # Filter risk scores to match test survival data (skipping post-EOL)
        test_risk = {h_idx: [] for h_idx in range(len(horizons))}
        for i in range(split, len(X)):
            cid = all_cid[i]
            cyc = all_cycles[i]
            eol_val = cell_eol[cid]
            if pd.notna(eol_val) and cyc > eol_val:
                continue
            for h_idx in range(len(horizons)):
                test_risk[h_idx].append(preds[i - split, h_idx])

        # Compute Uno AUC at each horizon
        for h_idx, H in enumerate(horizons):
            risk_scores = np.array(test_risk[h_idx])
            if len(np.unique(test_events)) < 2:
                uno_aucs[H].append(np.nan)
                continue
            try:
                uno_val, _ = cumulative_dynamic_auc(
                    surv_train, surv_test, risk_scores, [float(H)])
                uno_aucs[H].append(uno_val[0])
            except Exception:
                uno_aucs[H].append(np.nan)

    output[cl_key] = {
        "standard_auc_per_horizon": {
            f"H={H}": {
                "mean": round(float(np.nanmean(std_aucs[H])), 4),
                "std": round(float(np.nanstd(std_aucs[H], ddof=1)), 4)
            } for H in horizons
        },
        "uno_auc_per_horizon": {
            f"H={H}": {
                "mean": round(float(np.nanmean(uno_aucs[H])), 4),
                "std": round(float(np.nanstd(uno_aucs[H], ddof=1)), 4)
            } for H in horizons
        },
    }

out_path = os.path.join(results_dir, "uno_auc_censoring.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# Print compact comparison table
print(f"{'Censoring':>10} {'Metric':>8}", end="")
for H in horizons:
    print(f" {'H='+str(H):>10}", end="")
print()

for cl in censoring_levels:
    cl_key = f"censor_{int(cl*100)}%"
    d = output[cl_key]
    for metric, label in [("standard_auc_per_horizon", "Std AUC"),
                          ("uno_auc_per_horizon", "Uno AUC")]:
        print(f"{cl_key:>10} {label:>8}", end="")
        for H in horizons:
            v = d[metric][f"H={H}"]
            print(f" {v['mean']:.4f}±{v['std']:.4f}", end="")
        print()
