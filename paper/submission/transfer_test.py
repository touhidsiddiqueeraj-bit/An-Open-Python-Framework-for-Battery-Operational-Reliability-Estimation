#!/usr/bin/env python3
"""Synthetic-to-real transfer + Brier skill score + overfitting test."""
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

from src.data.nasa import NASALoader
from src.data.synthetic import generate_synthetic_nasa
from src.data.composite_failure import CompositeFailureLabeler
from src.models.xgboost_hazard import XGBoostHazard
from src.evaluation.metrics import compute_metrics

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])

horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]

# ── 1. Synthetic-to-real transfer test ────────────────
print("=== 1. SYNTHETIC-TO-REAL TRANSFER TEST ===")
print()

syn = generate_synthetic_nasa(n_cells=20, seed=42)
syn = labeler.label(syn, method="single")
X_syn = syn[feature_cols].values.astype(np.float32)
y_syn = syn[horizon_cols].values.astype(np.float32)

split = int(len(X_syn) * 0.8)
X_tr, y_tr = X_syn[:split], y_syn[:split]
X_eval, y_eval = X_syn[split:], y_syn[split:]

mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
mdl.fit(X_tr, y_tr, X_eval, y_eval)

syn_eval_preds = mdl.predict_proba(X_eval)
syn_eval_metrics = compute_metrics(y_eval, syn_eval_preds, horizons=horizons)
print(f"  Synthetic eval macro AUC: {syn_eval_metrics['macro_avg']['auc']:.4f}")

loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
df_real = loader.load_classic()
df_real = labeler.label(df_real, method="single")
X_real = df_real[feature_cols].values.astype(np.float32)
y_real = df_real[horizon_cols].values.astype(np.float32)

real_preds = mdl.predict_proba(X_real)
real_metrics = compute_metrics(y_real, real_preds, horizons=horizons)
print(f"  NASA transfer macro AUC: {real_metrics['macro_avg']['auc']:.4f}")
for h in horizons:
    m = real_metrics["per_horizon"][h]
    print(f"    H={h}: AUC={m['auc']}, Brier={m['brier']}")

# ── 2. Brier skill score ─────────────────────────────
print()
print("=== 2. BRIER SKILL SCORE (vs constant-hazard baseline) ===")
transfer_output = {}
for h in horizons:
    rate = y_real[:, horizons.index(h)].mean()
    bl_brier = rate * (1.0 - rate)
    mdl_brier = real_metrics["per_horizon"][h]["brier"]
    skill = 1.0 - mdl_brier / bl_brier if bl_brier > 0 else 0.0
    print(f"  H={h}: model_Brier={mdl_brier:.4f}, baseline_Brier={bl_brier:.4f}, skill={skill:.4f}")
    transfer_output[str(h)] = {
        "auc": real_metrics["per_horizon"][h]["auc"],
        "model_brier": mdl_brier,
        "baseline_brier": round(bl_brier, 4),
        "skill_score": round(skill, 4)}

# ── 3. Overfitting test (train/val/test on synthetic) ─
print()
print("=== 3. OVERFITTING TEST (synthetic N=20, 60/20/20 split) ===")
np.random.seed(2026)
idx = np.random.permutation(len(X_syn))
n_tr = int(0.6 * len(idx))
n_val = int(0.2 * len(idx))
i_tr, i_val, i_te = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]

X_tr2, y_tr2 = X_syn[i_tr], y_syn[i_tr]
X_val2, y_val2 = X_syn[i_val], y_syn[i_val]
X_te2, y_te2 = X_syn[i_te], y_syn[i_te]

mdl2 = XGBoostHazard(config=cfg["models"]["xgboost"])
mdl2.fit(X_tr2, y_tr2, X_val2, y_val2)

tr_preds = mdl2.predict_proba(X_tr2)
val_preds = mdl2.predict_proba(X_val2)
te_preds = mdl2.predict_proba(X_te2)

tr_auc = compute_metrics(y_tr2, tr_preds, horizons=horizons)["macro_avg"]["auc"]
val_auc = compute_metrics(y_val2, val_preds, horizons=horizons)["macro_avg"]["auc"]
te_auc = compute_metrics(y_te2, te_preds, horizons=horizons)["macro_avg"]["auc"]

print(f"  Train AUC: {tr_auc:.4f}")
print(f"  Val   AUC: {val_auc:.4f}")
print(f"  Test  AUC: {te_auc:.4f}")
gap = abs(tr_auc - te_auc)
print(f"  Train-test gap: {gap:.4f} {'OK (≤0.02)' if gap <= 0.02 else 'WARNING (>0.02)'}")

# ── Save ─────────────────────────────────────────────
output = {
    "synthetic_eval_macro_auc": syn_eval_metrics["macro_avg"]["auc"],
    "nasa_transfer_macro_auc": real_metrics["macro_avg"]["auc"],
    "nasa_per_horizon": transfer_output,
    "overfitting_test": {
        "train_auc": round(tr_auc, 4),
        "val_auc": round(val_auc, 4),
        "test_auc": round(te_auc, 4),
        "train_test_gap": round(gap, 4)}}
def convert(v):
    if isinstance(v, (np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.int32, np.int64)):
        return int(v)
    return v

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return convert(o)

fp = os.path.join(results_dir, "transfer_test.json")
with open(fp, "w") as f:
    json.dump(clean(output), f, indent=2)
print(f"\nSaved: {fp}")
print("Done.")
