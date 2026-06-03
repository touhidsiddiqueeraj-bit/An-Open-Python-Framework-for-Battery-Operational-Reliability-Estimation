#!/usr/bin/env python3
"""Censoring sensitivity + computational cost + synthetic validation."""
import sys, os, json, yaml, time
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

output = {}

# ── 1. CENSORING SENSITIVITY ─────────────────────────
print("=== 1. CENSORING SENSITIVITY ===")
syn = generate_synthetic_nasa(n_cells=20, seed=42)
syn = labeler.label(syn, method="single")
X = syn[feature_cols].values.astype(np.float32)
y = syn[horizon_cols].values.astype(np.float32)

censoring_levels = [0.0, 0.1, 0.3, 0.5]
censoring_results = {}
for cl in censoring_levels:
    y_c = y.copy()
    rng_c = np.random.default_rng(int(cl * 100))
    # Right-censor: randomly mask the last `cl` fraction of each horizon's labels to 0
    for i in range(y.shape[1]):
        n_censor = int(len(y_c) * cl)
        idx = rng_c.choice(len(y_c), n_censor, replace=False)
        # Only censor positive labels (batteries that have failed are removed from observation)
        pos_idx = np.where(y_c[:, i] == 1)[0]
        censor_idx = rng_c.choice(pos_idx, min(n_censor, len(pos_idx)), replace=False)
        y_c[censor_idx, i] = 0

    split = int(len(X) * 0.8)
    mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
    t0 = time.time()
    mdl.fit(X[:split], y_c[:split], X[split:], y_c[split:])
    fit_time = time.time() - t0

    t0 = time.time()
    preds = mdl.predict_proba(X[split:split + 5])  # small batch for inference time
    infer_time = (time.time() - t0) / max(len(X[split:split + 5]), 1)

    preds_all = mdl.predict_proba(X[split:])
    m = compute_metrics(y_c[split:], preds_all, horizons=horizons)
    auc_val = m["macro_avg"]["auc"]
    auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "nan"
    censoring_results[f"censor_{cl:.0%}"] = {
        "macro_auc": auc_str,
        "fit_time_sec": round(fit_time, 2),
        "infer_time_ms": round(infer_time * 1000, 2)}
    print(f"  Censor={cl:.0%}: macro AUC={auc_str}, "
          f"fit={fit_time:.2f}s, infer={infer_time*1000:.2f}ms/cell")
output["censoring_sensitivity"] = censoring_results

# ── 2. SYNTHETIC VALIDATION (KL divergence) ──────────
print()
print("=== 2. SYNTHETIC VALIDATION (KL divergence) ===")
from scipy.stats import entropy, gaussian_kde

loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
df_real = loader.load_classic()
syn_val = generate_synthetic_nasa(n_cells=20, seed=42)

# Compare SOH distributions
real_soh = df_real["soh"].values
syn_soh = syn_val["soh"].values

# KL divergence via KDE
real_kde = gaussian_kde(real_soh)
syn_kde = gaussian_kde(syn_soh)
grid = np.linspace(0.4, 1.05, 200)
p = real_kde(grid) + 1e-10
q = syn_kde(grid) + 1e-10
kl_div = entropy(p, q)
print(f"  KL divergence (real || synthetic SOH): {kl_div:.4f} nats")

# Wasserstein distance
from scipy.stats import wasserstein_distance
ws = wasserstein_distance(real_soh, syn_soh)
print(f"  Wasserstein distance (real vs synthetic SOH): {ws:.4f}")

# Mean SOH comparison
print(f"  Mean SOH: real={real_soh.mean():.4f}, synthetic={syn_soh.mean():.4f}")

output["synthetic_validation"] = {
    "kl_divergence_nats": round(kl_div, 4),
    "wasserstein_distance": round(ws, 4),
    "real_mean_soh": round(float(real_soh.mean()), 4),
    "synthetic_mean_soh": round(float(syn_soh.mean()), 4),
    "real_n_cells": len(df_real["cell_id"].unique()),
    "synthetic_n_cells": 20}

# ── 3. COMPUTATIONAL COST (N=20 full run) ───────────
print()
print("=== 3. COMPUTATIONAL COST ===")
syn = generate_synthetic_nasa(n_cells=20, seed=42)
syn = labeler.label(syn, method="single")
X_s = syn[feature_cols].values.astype(np.float32)
y_s = syn[horizon_cols].values.astype(np.float32)

def get_mem_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmPeak:"):
                return int(line.split()[1]) // 1024
    return 0

t0 = time.time()
for trial in range(3):
    split_s = int(len(X_s) * 0.8)
    mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
    mdl.fit(X_s[:split_s], y_s[:split_s], X_s[split_s:], y_s[split_s:])
    _ = mdl.predict_proba(X_s[split_s:])
avg_time = (time.time() - t0) / 3
peak_mb = get_mem_mb()

print(f"  Avg train+infer time (N=20): {avg_time:.2f}s")
print(f"  Peak memory: {peak_mb:.0f} MB")
output["computational_cost"] = {
    "train_and_infer_time_sec_N20": round(avg_time, 2),
    "peak_memory_mb": round(peak_mb, 0)}

# ── SAVE ────────────────────────────────────────────
fp = os.path.join(results_dir, "supplementary_analysis.json")
with open(fp, "w") as f:
    json.dump(output, f, indent=2, default=lambda x: "nan" if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x)
print(f"\nSaved: {fp}")
print("Done.")
