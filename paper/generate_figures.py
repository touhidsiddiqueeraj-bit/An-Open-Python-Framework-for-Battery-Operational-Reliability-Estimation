#!/usr/bin/env python3
"""Generate all 6 manuscript figures from experiment results."""

import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from src.data.nasa import NASALoader
import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE, "..", "code")
RESULTS = os.path.join(CODE_DIR, "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.chdir(CODE_DIR)

# Load config
with open(os.path.join(CODE_DIR, "config.yaml")) as f:
    cfg = yaml.safe_load(f)

# Find latest result files
def latest(pat):
    files = sorted(glob.glob(os.path.join(RESULTS, pat)))
    return files[-1] if files else None

# ── FIGURE 1: Degradation curves ────────────────────────
def fig1_degradation():
    loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
    df = loader.load_classic()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"B0005": "#1b3a5c", "B0006": "#c0392b", "B0007": "#27ae60", "B0018": "#f39c12"}
    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        sub = df[df["cell_id"] == cid]
        ax.plot(sub["cycle"], sub["soh"], label=cid, color=colors.get(cid, "#333"),
                linewidth=1.5)
    ax.axhline(0.70, ls="--", color="gray", alpha=0.7, label="EOL threshold (SOH=0.70)")
    ax.set_xlabel("Cycle number", fontsize=11)
    ax.set_ylabel("State of Health (SOH)", fontsize=11)
    ax.set_title("Figure 1: NASA Battery Degradation Curves", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.05)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "degradation_curves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  degradation_curves.png")

# ── FIGURE 2: AUC by horizon ────────────────────────────
def fig2_auc():
    fp = latest("baseline_*.json")
    if not fp:
        return
    with open(fp) as f:
        d = json.load(f)
    horizons = [10, 20, 30, 50]
    raw_aucs = [d["raw"]["per_horizon"][str(h)]["auc"] for h in horizons]
    cal_aucs = [d["calibrated"]["per_horizon"][str(h)]["auc"] for h in horizons]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(horizons))
    w = 0.35
    ax.bar(x - w/2, raw_aucs, w, label="Raw XGBoost", color="#1b3a5c", edgecolor="k")
    ax.bar(x + w/2, cal_aucs, w, label="Calibrated", color="#c0392b", edgecolor="k")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.7, label="Random (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in horizons])
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Figure 2: Model Discrimination by Horizon", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "auc_by_horizon.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  auc_by_horizon.png")

# ── FIGURE 3: Dataset composition (cycles per cell + EOL) ──
def fig3_composition():
    loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
    df = loader.load_classic()
    cells = ["B0005", "B0006", "B0007", "B0018"]
    totals = []
    eols = []
    for c in cells:
        sub = df[df["cell_id"] == c]
        totals.append(len(sub))
        eols.append((sub["soh"] < 0.70).sum())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(cells))
    w = 0.35
    bars1 = ax.bar(x - w/2, totals, w, label="Total cycles", color="#1b3a5c", edgecolor="k")
    bars2 = ax.bar(x + w/2, eols, w, label="EOL cycles (SOH<0.70)", color="#c0392b", edgecolor="k")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_ylabel("Cycle count", fontsize=11)
    ax.set_title("Figure 3: Dataset Composition", fontsize=12)
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dataset_composition.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  dataset_composition.png")

# ── FIGURE 4: Feature correlation ───────────────────────
def fig4_correlation():
    loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
    df = loader.load_classic()
    feat_cols = ["soh", "voltage_avg", "current_avg", "temperature_avg", "cycle"]
    # Compute derived
    df["d_soh"] = df.groupby("cell_id")["soh"].diff()
    df["d_capacity"] = df.groupby("cell_id")["capacity"].diff()
    feat_cols += ["d_soh", "d_capacity"]
    corr = df[feat_cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, ax=ax, fmt=".2f", cbar_kws={"shrink": 0.8})
    ax.set_title("Figure 4: Feature Correlation Matrix", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "feature_correlation.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  feature_correlation.png")

# ── FIGURE 5: Ablation ──────────────────────────────────
def fig5_ablation():
    fp = latest("ablation_*.json")
    if not fp:
        return
    with open(fp) as f:
        d = json.load(f)
    labels = list(d.keys())
    rates = [d[k]["failure_rate"] for k in labels]
    colors = ["#7f8c8d", "#1b3a5c", "#2980b9", "#c0392b"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), rates, color=colors[:len(labels)], edgecolor="k")
    ax.axhline(0.0063, ls="--", color="gray", alpha=0.7, label="Always-dispatch baseline")
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Failure rate", fontsize=11)
    ax.set_title("Figure 5: Ablation Study — Failure Rates", fontsize=12)
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ablation.png")

# ── FIGURE 6: Real vs synthetic degradation ─────────────
def fig6_real_vs_synthetic():
    from src.data.synthetic import generate_synthetic_nasa
    loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
    df_real = loader.load_classic()
    df_syn = generate_synthetic_nasa(n_cells=4, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = ["#1b3a5c", "#c0392b", "#27ae60", "#f39c12"]
    for i, cid in enumerate(["B0005", "B0006", "B0007", "B0018"]):
        sub = df_real[df_real["cell_id"] == cid]
        axes[0].plot(sub["cycle"], sub["soh"], label=cid, color=colors[i], linewidth=1.2)
    axes[0].axhline(0.70, ls="--", color="gray", alpha=0.5)
    axes[0].set_xlabel("Cycle number", fontsize=10)
    axes[0].set_ylabel("SOH", fontsize=10)
    axes[0].set_title("Real NASA Data", fontsize=11)
    axes[0].legend(fontsize=8)
    for i, cid in enumerate(df_syn["cell_id"].unique()):
        sub = df_syn[df_syn["cell_id"] == cid]
        axes[1].plot(sub["cycle"], sub["soh"], label=f"Syn-{i+1}", color=colors[i], linewidth=1.2)
    axes[1].axhline(0.70, ls="--", color="gray", alpha=0.5)
    axes[1].set_xlabel("Cycle number", fontsize=10)
    axes[1].set_title("Synthetic Data", fontsize=11)
    axes[1].legend(fontsize=8)
    fig.suptitle("Figure 6: Real vs Synthetic Degradation Trajectories", fontsize=12, y=1.02)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "real_vs_synthetic.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  real_vs_synthetic.png")

# ── FIGURE 7: Scaling curve (AUC vs N) ────────────────────
def fig7_scaling_curve():
    fp = latest("scaling_monte_carlo_*.json")
    if not fp:
        print("  Scaling results not found, skipping fig7.")
        return
    with open(fp) as f:
        d = json.load(f)
    N = np.array(d["N"])
    auc_mean = np.array(d["auc_mean"])
    auc_std = np.array(d["auc_std"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N, auc_mean, "-o", color="#1b3a5c", linewidth=2, markersize=6, zorder=3)
    ax.fill_between(N, auc_mean - auc_std, auc_mean + auc_std,
                    alpha=0.2, color="#1b3a5c", label="±1 std (3 seeds)")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="Random (AUC=0.5)")
    ax.axvspan(0, 5, alpha=0.06, color="#e74c3c", label="Insufficient (N≤5)")
    ax.axvspan(5, 12, alpha=0.06, color="#f39c12", label="Marginal (5<N<12)")
    ax.axvspan(12, 55, alpha=0.06, color="#27ae60", label="Reliable (N≥12)")
    ax.set_xlabel("Number of batteries (N)", fontsize=12)
    ax.set_ylabel("Macro-averaged AUC", fontsize=12)
    ax.set_title("Figure 1: Model Discrimination vs Dataset Size", fontsize=13)
    ax.set_xlim(1, 55)
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks(N)
    ax.legend(fontsize=9, loc="lower right")
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scaling_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  scaling_curve.png")

if __name__ == "__main__":
    print("Generating figures...")
    fig1_degradation()
    fig2_auc()
    fig3_composition()
    fig4_correlation()
    fig5_ablation()
    fig6_real_vs_synthetic()
    fig7_scaling_curve()
    print("Done.")
