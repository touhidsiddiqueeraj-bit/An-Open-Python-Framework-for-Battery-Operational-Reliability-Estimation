#!/usr/bin/env python3
"""Censoring baseline: compare RSF vs XGBoost under censoring.

Both models use per-cycle data. XGBoost gets per-horizon binary labels.
RSF gets per-cycle (time_to_eol, event) pairs. Only pre-EOL cycles are
used for RSF (cycles after EOL are excluded — the cell is no longer at risk).
Determines whether >20% censoring failure is specific to discrete-time hazard."""
import sys, os, json, yaml, time, warnings
import numpy as np
import pandas as pd
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

syn = generate_synthetic_nasa(n_cells=20, seed=42)
syn = labeler.label(syn, method="single")
X = syn[feature_cols].values.astype(np.float32)
y = syn[horizon_cols].values.astype(np.float32)
n_total = len(X)
split = int(n_total * 0.8)

# Per-cell: EOL cycle and max observed cycle
cell_eol = syn.groupby("cell_id")["eol_cycle"].first().to_dict()
cell_ncycles = syn.groupby("cell_id").size().to_dict()

# Shift train/test indices to be cell-based (not contiguous rows)
# 80/20 split by cell: first 16 cells train, last 4 test
cell_ids_arr = syn["cell_id"].values
all_cids = sorted(syn["cell_id"].unique())
n_cells = len(all_cids)
train_cell_count = int(n_cells * 0.8)
train_cell_ids = set(all_cids[:train_cell_count])
test_cell_ids = set(all_cids[train_cell_count:])

train_rows = np.where(np.isin(cell_ids_arr, list(train_cell_ids)))[0]
test_rows = np.where(np.isin(cell_ids_arr, list(test_cell_ids)))[0]

print(f"Train cells: {len(train_cell_ids)}, Test cells: {len(test_cell_ids)}")
print(f"Train rows: {len(train_rows)}, Test rows: {len(test_rows)}")

# Pre-compute which cycles are pre-EOL for each cell
is_at_risk = np.zeros(n_total, dtype=bool)
for cid in all_cids:
    eol = cell_eol[cid]
    if pd.notna(eol):
        mask = (cell_ids_arr == cid) & (syn["cycle"].values < eol)
    else:
        mask = cell_ids_arr == cid
    is_at_risk[mask] = True

# For RSF: map each pre-EOL cycle to (time_to_eol, event)
# time_to_eol = eol_cycle - current_cycle
# event = (current_cycle + 1 == eol_cycle)  # True only for the cycle just before EOL
rsf_time = np.full(n_total, np.nan)
rsf_event = np.zeros(n_total, dtype=bool)
for cid in all_cids:
    eol = cell_eol[cid]
    if pd.notna(eol):
        mask = (cell_ids_arr == cid) & (syn["cycle"].values < eol)
        cycles = syn.loc[mask, "cycle"].values
        rsf_time[mask] = eol - cycles
        # Event = True only for the last pre-EOL cycle
        last_pre_eol_cycle = cycles.max()
        event_mask = mask & (syn["cycle"].values == last_pre_eol_cycle)
        rsf_event[event_mask] = True

# Remove NaN rows (post-EOL or failed cells with no risk period)
risk_idx = is_at_risk.copy()

censoring_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
output = {}

for cl in censoring_levels:
    print(f"\n=== Censoring {cl*100:.0f}% ===")
    cl_key = f"censor_{int(cl*100)}%"

    # ── XGBoost (per-cycle, label-masking on all cycles) ──
    y_c = y.copy()
    rng_c = np.random.default_rng(int(cl * 100))
    for i in range(y.shape[1]):
        n_censor = int(len(y_c) * cl)
        pos_idx = np.where(y_c[:, i] == 1)[0]
        if len(pos_idx) == 0:
            continue
        n_censor = min(n_censor, len(pos_idx))
        if n_censor == 0:
            continue
        censor_idx = rng_c.choice(pos_idx, n_censor, replace=False)
        y_c[censor_idx, i] = 0

    mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
    t0 = time.time()
    mdl.fit(X[train_rows], y_c[train_rows], X[test_rows], y_c[test_rows])
    xgb_fit_time = time.time() - t0
    preds_all = mdl.predict_proba(X[test_rows])
    m = compute_metrics(y_c[test_rows], preds_all, horizons=horizons)
    xgb_auc = m["macro_avg"]["auc"]

    # ── RSF (per-cycle, pre-EOL only, right-censoring on events) ──
    # Apply censoring: randomly censor a fraction of event=True cycles
    rsf_y = rsf_event.copy()
    rng_rsf = np.random.default_rng(int(cl * 100) + 42)
    n_censor_rsf = int(rsf_event.sum() * cl)
    event_idx = np.where(rsf_event & is_at_risk)[0]
    if n_censor_rsf > 0 and len(event_idx) > 0:
        n_censor_rsf = min(n_censor_rsf, len(event_idx))
        censor_rsf_idx = rng_rsf.choice(event_idx, n_censor_rsf, replace=False)
        rsf_y[censor_rsf_idx] = False  # censor: event becomes censored

    # Prepare training data for RSF (pre-EOL cycles only)
    train_risk = risk_idx[train_rows]
    rsf_X_train = X[train_rows][train_risk].astype(np.float64)
    rsf_t_train = rsf_time[train_rows][train_risk]
    rsf_e_train = rsf_y[train_rows][train_risk]

    test_risk = risk_idx[test_rows]
    rsf_X_test = X[test_rows][test_risk].astype(np.float64)
    rsf_t_test = rsf_time[test_rows][test_risk]
    rsf_e_test = rsf_y[test_rows][test_risk]

    # Ensure no NaN/inf in training data
    valid_train = np.isfinite(rsf_t_train) & np.isfinite(rsf_X_train).all(axis=1)
    rsf_X_train = rsf_X_train[valid_train]
    rsf_t_train = rsf_t_train[valid_train]
    rsf_e_train = rsf_e_train[valid_train]

    if len(np.unique(rsf_e_train)) < 2:
        print("  RSF: train has <2 event classes, skipping")
        rsf_macro_auc = np.nan
        rsf_aucs = [np.nan] * len(horizons)
        rsf_fit_time = None
    else:
        rsf_target = Surv.from_arrays(event=rsf_e_train, time=rsf_t_train)
        rsf = RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=10, max_depth=8,
            random_state=42, n_jobs=2)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rsf.fit(rsf_X_train, rsf_target)
        rsf_fit_time = time.time() - t0

        # Predict survival for test cycles
        valid_test = np.isfinite(rsf_t_test) & np.isfinite(rsf_X_test).all(axis=1)
        if valid_test.sum() == 0:
            rsf_macro_auc = np.nan
            rsf_aucs = [np.nan] * len(horizons)
        else:
            pred_surv = rsf.predict_survival_function(rsf_X_test[valid_test])
            utimes = rsf.unique_times_

            rsf_aucs = []
            for H in horizons:
                scores = []
                labels = []
                for i, global_idx in enumerate(test_rows[risk_idx[test_rows]][valid_test]):
                    surv_fn = pred_surv[i]
                    # RSF survival function is S(t) = P(remaining_time > t)
                    # P(fail within H) = 1 - S(H)
                    p = 1.0 - surv_fn(min(float(H), utimes[-1]))
                    p = max(0.0, min(1.0, p))
                    scores.append(p)
                    labels.append(y_c[global_idx, horizons.index(H)])

                scores = np.array(scores)
                labels = np.array(labels)
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2:
                    rsf_aucs.append(np.nan)
                else:
                    try:
                        rsf_aucs.append(roc_auc_score(labels, scores))
                    except Exception:
                        rsf_aucs.append(np.nan)

            rsf_macro_auc = np.nanmean(rsf_aucs) if not np.all(np.isnan(rsf_aucs)) else np.nan

    print(f"  XGBoost AUC: {xgb_auc:.4f}  RSF AUC: {rsf_macro_auc:.4f}")
    if rsf_macro_auc is not None and not np.isnan(rsf_macro_auc):
        print(f"  RSF per-horizon: {[f'{a:.4f}' if not np.isnan(a) else 'nan' for a in rsf_aucs]}")
    output[cl_key] = {
        "xgb_fit_time_sec": round(xgb_fit_time, 2),
        "rsf_fit_time_sec": round(rsf_fit_time, 2) if rsf_fit_time is not None else None,
        "xgb_macro_auc": round(xgb_auc, 4) if not np.isnan(xgb_auc) else "nan",
        "rsf_macro_auc": round(rsf_macro_auc, 4) if not np.isnan(rsf_macro_auc) else "nan",
        "rsf_per_horizon_auc": [round(a, 4) if not np.isnan(a) else "nan" for a in rsf_aucs],
        "nan_horizons_rsf": sum(1 for a in rsf_aucs if np.isnan(a)),
        "n_test_cycles_rsf": int(valid_test.sum()),
    }

# Determine interpretation
rsf_ok_above_20 = all(
    output[k]["rsf_macro_auc"] != "nan" and output[k]["rsf_macro_auc"] > 0.5
    for k in ["censor_20%", "censor_30%", "censor_50%", "censor_80%"]
    if k in output)
xgb_fails_above_20 = all(
    output[k]["xgb_macro_auc"] == "nan" or (isinstance(output[k]["xgb_macro_auc"], float) and output[k]["xgb_macro_auc"] < 0.5)
    for k in ["censor_20%", "censor_30%", "censor_50%", "censor_80%"]
    if k in output)

if rsf_ok_above_20 and xgb_fails_above_20:
    output["interpretation"] = ("RSF maintains discriminative performance at >20% censoring "
        "while XGBoost discrete-time hazard fails. Discrete-time hazard models are unsuitable "
        "for datasets with >20% censoring; practitioners should use continuous-time survival methods.")
elif not rsf_ok_above_20:
    output["interpretation"] = ("Both RSF and XGBoost discrete-time hazard fail above 20% censoring. "
        "The dataset itself lacks sufficient failure signal at high censoring rates, "
        "independent of modeling choice.")
else:
    output["interpretation"] = ("Mixed censoring tolerance results. See per-horizon AUCs for details.")

out_path = os.path.join(results_dir, "censoring_baseline.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"\nInterpretation: {output['interpretation']}")
