#!/usr/bin/env python3
"""RSF per-cycle scaling baseline: XGBoost vs RSF across N=2-20, 20 seeds."""
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
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])
horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]

n_vals = [2, 3, 5, 8, 12, 20]
n_seeds = 20
output = {"xgb": {}, "rsf": {}}

for N in n_vals:
    xgb_aucs = []
    rsf_aucs = []
    print(f"\nN={N} ...")

    for seed in range(n_seeds):
        np.random.seed(seed)
        df = generate_synthetic_nasa(n_cells=N, n_cycles=300, seed=seed * 100 + N)
        df = labeler.label(df.copy(), method="single")
        X = df[feature_cols].values.astype(np.float32)
        y = df[horizon_cols].values.astype(np.float32)
        cell_ids = df["cell_id"].values
        eol_cycles = df["eol_cycle"].values

        # Per-cell: EOL cycles
        cell_eol = df.groupby("cell_id")["eol_cycle"].first().to_dict()
        cell_ncycles = df.groupby("cell_id").size().to_dict()

        # Leave-battery-out CV
        all_cids = df["cell_id"].unique().tolist()
        np.random.shuffle(all_cids)

        xgb_fold_aucs = []
        rsf_fold_aucs = []
        n_folds = 0
        for test_cell in all_cids:
            train_mask = df["cell_id"] != test_cell
            test_mask = df["cell_id"] == test_cell
            X_train_fold = X[train_mask.values]
            y_train_fold = y[train_mask.values]
            X_test_fold = X[test_mask.values]
            y_test_fold = y[test_mask.values]
            cell_ids_test = cell_ids[test_mask.values]

            if len(X_train_fold) < 10 or len(X_test_fold) < 5:
                continue

            n_folds += 1
            split = int(len(X_train_fold) * 0.8)

            # ── XGBoost ──
            mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
            mdl.fit(X_train_fold[:split], y_train_fold[:split],
                    X_train_fold[split:], y_train_fold[split:])
            preds = mdl.predict_proba(X_test_fold)
            m = compute_metrics(y_test_fold, preds, horizons=horizons)
            xgb_fold_aucs.append(m["macro_avg"]["auc"] if not np.isnan(m["macro_avg"]["auc"]) else 0.5)

            # ── RSF (per-cycle) ──
            # Build per-cycle time-to-EOL for training
            train_cell_ids = df.loc[train_mask, "cell_id"].values
            train_cycles = df.loc[train_mask, "cycle"].values
            train_eol = np.array([cell_eol[cid] for cid in train_cell_ids])
            train_valid = pd.notna(train_eol)
            train_time = np.where(train_valid, train_eol - train_cycles, 300.0 - train_cycles + 1)
            train_event = train_valid & (train_cycles == train_eol)
            train_event = train_event.astype(bool)

            # At/EOL only: include the EOL cycle itself (event) and all pre-EOL cycles
            train_pre_eol = np.where(train_valid, train_cycles <= train_eol, True)

            train_X_rsf = X_train_fold[train_pre_eol].astype(np.float64)
            train_t_rsf = train_time[train_pre_eol]
            train_e_rsf = train_event[train_pre_eol]

            valid = np.isfinite(train_t_rsf) & np.isfinite(train_X_rsf).all(axis=1)
            train_X_rsf = train_X_rsf[valid]
            train_t_rsf = train_t_rsf[valid]
            train_e_rsf = train_e_rsf[valid]

            if len(np.unique(train_e_rsf)) < 2 or train_X_rsf.shape[0] < 10:
                rsf_fold_aucs.append(np.nan)
                continue

            rsf = RandomSurvivalForest(
                n_estimators=200, min_samples_leaf=10, max_depth=6,
                random_state=42, n_jobs=2)
            rsf_target = Surv.from_arrays(event=train_e_rsf, time=train_t_rsf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rsf.fit(train_X_rsf, rsf_target)

            # Per-cycle RSF prediction on test fold
            rsf_h_aucs = []
            for H in horizons:
                scores, labels = [], []
                for i_test, (_, test_row) in enumerate(df[test_mask].iterrows()):
                    cid = test_row["cell_id"]
                    cycle = test_row["cycle"]
                    eol = cell_eol[cid]
                    if pd.notna(eol) and cycle >= eol:
                        continue  # post-EOL, not at risk
                    feats = X_test_fold[i_test].reshape(1, -1).astype(np.float64)
                    surv_fn = rsf.predict_survival_function(feats)[0]
                    utimes = rsf.unique_times_
                    p = 1.0 - surv_fn(min(float(H), utimes[-1]))
                    p = max(0.0, min(1.0, p))
                    scores.append(p)
                    labels.append(y_test_fold[i_test, horizons.index(H)])

                scores, labels = np.array(scores), np.array(labels)
                if len(np.unique(labels)) < 2:
                    rsf_h_aucs.append(np.nan)
                else:
                    try:
                        rsf_h_aucs.append(roc_auc_score(labels, scores))
                    except Exception:
                        rsf_h_aucs.append(np.nan)

            rsf_fold_aucs.append(np.nanmean(rsf_h_aucs) if not np.all(np.isnan(rsf_h_aucs)) else np.nan)

        # Aggregate across folds for this seed
        xgb_aucs.append(np.nanmean(xgb_fold_aucs) if xgb_fold_aucs else np.nan)
        rsf_aucs.append(np.nanmean(rsf_fold_aucs) if rsf_fold_aucs else np.nan)

    output["xgb"][f"N={N}"] = {
        "mean": round(float(np.nanmean(xgb_aucs)), 4),
        "std": round(float(np.nanstd(xgb_aucs, ddof=1)), 4)}
    output["rsf"][f"N={N}"] = {
        "mean": round(float(np.nanmean(rsf_aucs)), 4),
        "std": round(float(np.nanstd(rsf_aucs, ddof=1)), 4)}
    print(f"  XGBoost: {output['xgb'][f'N={N}']['mean']:.4f} ± {output['xgb'][f'N={N}']['std']:.4f}")
    print(f"  RSF:     {output['rsf'][f'N={N}']['mean']:.4f} ± {output['rsf'][f'N={N}']['std']:.4f}")

out_path = os.path.join(results_dir, "rsf_baseline.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print("\n=== Summary ===")
print(f"{'N':>4}  {'XGB AUC':>10}  {'RSF AUC':>10}")
for N in n_vals:
    x = output["xgb"][f"N={N}"]
    r = output["rsf"][f"N={N}"]
    print(f"{N:>4}  {x['mean']:.4f}±{x['std']:.4f}  {r['mean']:.4f}±{r['std']:.4f}")

xgb_wins = all(
    output["xgb"][f"N={N}"]["mean"] > output["rsf"][f"N={N}"]["mean"]
    for N in n_vals if not np.isnan(output["rsf"][f"N={N}"]["mean"]))
output["interpretation"] = (
    "XGBoost consistently outperforms RSF across all N" if xgb_wins
    else "RSF matches or exceeds XGBoost on some N values.")

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
