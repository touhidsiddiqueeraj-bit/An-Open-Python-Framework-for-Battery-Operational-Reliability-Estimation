import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_calibration(y_true, y_pred, n_bins=15, ax=None):
    """Reliability diagram: predicted probability vs observed frequency."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    observed = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        observed[i] = y_true[mask].mean() if mask.sum() > 0 else 0
    ax.plot(bin_centers, observed, "o-", label="Model", color="#1b3a5c")
    ax.plot([0, 1], [0, 1], "--", label="Perfect", color="gray")
    ax.fill_between(bin_centers, observed, bin_centers,
                     alpha=0.15, color="#1b3a5c")
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency", fontsize=11)
    ax.set_title("Calibration Curve", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    sns.despine()
    return ax


def plot_risk_tradeoff(results, ax=None):
    """Energy vs failure rate trade-off across policies."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"Always dispatch": "X", "SOH threshold": "s",
               "RUL threshold": "D", "Risk-threshold": "o",
               "Derating": "v", "RL-Adaptive": "^"}
    for policy, data in results.items():
        ax.scatter(data["failure_rate"], data["energy"],
                    marker=markers.get(policy, "o"), s=120,
                    label=policy, zorder=5, edgecolors="k", linewidth=0.5)
        if "tau_sweep" in data:
            sweep = data["tau_sweep"]
            ax.plot(sweep["failure_rates"], sweep["energies"],
                     "-", alpha=0.4, color="gray")
    ax.set_xlabel("Failure rate", fontsize=11)
    ax.set_ylabel("Delivered energy (kWh)", fontsize=11)
    ax.set_title("Operational Energy–Risk Trade-off", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    sns.despine()
    return ax


def plot_survival(cycles, survival_prob, horizon, ax=None):
    """Plot survival probability over degradation trajectory."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, survival_prob, "-", color="#1b3a5c", linewidth=1.5)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Cycle number", fontsize=11)
    ax.set_ylabel(f"Survival probability (H={horizon})", fontsize=11)
    ax.set_title(f"Survival Function (H={horizon})", fontsize=12)
    ax.set_ylim(0, 1.05)
    sns.despine()
    return ax


def plot_multi_horizon(cycles, probs, horizons, ax=None):
    """Plot failure probabilities for multiple horizons."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(horizons)))
    for i, h in enumerate(horizons):
        ax.plot(cycles, probs[:, i], label=f"H={h}",
                color=colors[i], linewidth=1.5)
    ax.set_xlabel("Cycle number", fontsize=11)
    ax.set_ylabel("Failure probability", fontsize=11)
    ax.set_title("Multihorizon Failure Prediction", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    sns.despine()
    return ax


def plot_auc_comparison(scores, ax=None):
    """Bar chart comparing AUC across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    models = list(scores.keys())
    aucs = [scores[m]["auc"] for m in models]
    bars = ax.bar(models, aucs, color="#1b3a5c", edgecolor="k", linewidth=0.5)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Model Discrimination Comparison", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    sns.despine()
    return ax


def save_figure(fig, path, dpi=200, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")
