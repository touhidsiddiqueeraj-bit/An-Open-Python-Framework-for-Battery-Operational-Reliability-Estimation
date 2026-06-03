#!/usr/bin/env python3
"""Generate synthetic overlay plot + NASA per-cell breakdown."""
import sys, os, yaml, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.nasa import NASALoader
from src.data.synthetic import generate_synthetic_nasa
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(fig_dir, exist_ok=True)

# ── 1. Synthetic vs real SOH overlay ──
loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
df_real = loader.load_classic()
syn = generate_synthetic_nasa(n_cells=20, seed=42)

real_soh = df_real["soh"].values
syn_soh = syn["soh"].values

real_kde = gaussian_kde(real_soh)
syn_kde = gaussian_kde(syn_soh)
grid = np.linspace(0.4, 1.05, 200)
p = real_kde(grid)
q = syn_kde(grid)

from scipy.stats import entropy, wasserstein_distance
kl_div = entropy(p + 1e-10, q + 1e-10)
ws = wasserstein_distance(real_soh, syn_soh)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(real_soh, bins=30, density=True, alpha=0.5, label="Real NASA (n=4 cells)", color="steelblue")
ax.hist(syn_soh, bins=30, density=True, alpha=0.5, label="Synthetic (n=20 cells)", color="coral")
ax.plot(grid, real_kde(grid), "b-", lw=1.5)
ax.plot(grid, syn_kde(grid), "r-", lw=1.5)
ax.set_xlabel("State of Health (SOH)")
ax.set_ylabel("Density")
ax.set_title(f"SOH Distribution: Real vs Synthetic\nKL divergence = {kl_div:.3f} nats, Wasserstein = {ws:.3f}")
ax.legend()
ax.set_xlim(0.4, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "soh_distribution_comparison.png"), dpi=200)
print(f"Saved: {fig_dir}/soh_distribution_comparison.png")

# ── 2. Per-cell breakdown for NASA 4-cell ──
# Simulate leave-battery-out predictions for NASA data
from src.data.composite_failure import CompositeFailureLabeler
from src.models.xgboost_hazard import XGBoostHazard

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])

df = labeler.label(df_real, method="single")
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]
horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]

cell_ids = df["cell_id"].unique()
print(f"Cells: {cell_ids}")

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()

for idx, cell in enumerate(cell_ids):
    train = df[df["cell_id"] != cell]
    test = df[df["cell_id"] == cell]
    X_tr = train[feature_cols].values.astype(np.float32)
    y_tr = train[horizon_cols].values.astype(np.float32)
    X_te = test[feature_cols].values.astype(np.float32)
    y_te = test[horizon_cols].values.astype(np.float32)

    mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
    mdl.fit(X_tr, y_tr, X_tr[:len(X_tr)//5], y_tr[:len(X_tr)//5])
    preds = mdl.predict_proba(X_te)

    ax = axes[idx]
    # Plot predicted probabilities vs actual SOH for H=20
    h_idx = horizons.index(20)
    y_true = y_te[:, h_idx]
    y_pred = preds[:, h_idx]

    n_eol = int(y_true.sum())
    n_total = len(y_true)
    # Compute AUC if both classes present
    if n_eol > 0 and n_eol < n_total:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = float("nan")

    ax.scatter(test["soh"], y_pred, c=["red" if yy else "blue" for yy in y_true],
               alpha=0.6, s=10, edgecolors="none")
    ax.set_title(f"Cell {cell}: AUC(H=20)={auc:.2f}, EOL={n_eol}/{n_total}")
    ax.set_xlabel("SOH")
    ax.set_ylabel("P(fail | H=20)")
    ax.set_ylim(-0.05, 1.05)

fig.suptitle("NASA 4-Cell: Per-Cell Predictions at H=20 (Leave-Battery-Out CV)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "nasa_per_cell_breakdown.png"), dpi=200)
print(f"Saved: {fig_dir}/nasa_per_cell_breakdown.png")
print("Done.")
