#!/usr/bin/env python3
"""Local experiment runner for the extension paper.

Usage:
  python run_all.py --quick          XGBoost + CALCE + dispatch + market
  python run_all.py --full           Also include DL models (CPU: slow)
  python run_all.py --expt baseline  Single experiment by name
  python run_all.py --list           Show available experiments
"""

import os, sys, json, yaml, argparse, time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.nasa import NASALoader
from src.data.calce import CALCELoader
from src.data.synthetic import generate_synthetic_nasa, generate_synthetic_calce
from src.data.composite_failure import CompositeFailureLabeler
from src.data.augmentation import OperationalAugmenter
from src.models.xgboost_hazard import XGBoostHazard
from src.models.calibration import ProbabilityCalibrator
from src.dispatch.threshold import ThresholdPolicy
from src.dispatch.derating import ContinuousDeratingPolicy
from src.dispatch.market_sim import MarketSimulator
from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_validation import leave_battery_out_cv
from src.evaluation.visualization import (
    plot_calibration, plot_risk_tradeoff, plot_auc_comparison, save_figure)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "paper", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg, color=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def save_results(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}_{TAG}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log(f"Saved: {path}")
    return path


def load_config():
    path = os.path.join(BASE, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── DATA LOADING ──────────────────────────────────────────

def load_nasa(cfg):
    loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
    df = loader.load_classic()
    if df.empty:
        log("Real NASA data not found. Using synthetic data.")
        loader.print_download_instructions()
        df = generate_synthetic_nasa(n_cells=4, seed=42)
        log(f"Synthetic NASA: {len(df)} rows, {df['cell_id'].nunique()} cells")
    else:
        log(f"NASA loaded: {len(df)} rows, {df['cell_id'].nunique()} cells")
    return df


def load_calce(cfg, chemistries=None):
    if chemistries is None:
        chemistries = ["LCO"]
    loader = CALCELoader(
        data_dir=cfg["execution"]["data_dir"], chemistries=chemistries)
    log(f"Loading CALCE ({', '.join(chemistries)})...")
    df = loader.load_all()
    if df.empty:
        log("Real CALCE data not found. Using synthetic data.")
        for chem in chemistries:
            synth = generate_synthetic_calce(chemistry=chem, n_cells=2, seed=99)
            df = pd.concat([df, synth], ignore_index=True)
    log(f"  CALCE loaded: {len(df)} rows, {df['cell_id'].nunique()} cells")
    return df


def label_data(df, method, cfg):
    labeler = CompositeFailureLabeler(
        soh_threshold=cfg["failure"]["soh_threshold"],
        sudden_drop=cfg["failure"]["sudden_drop_threshold"])
    return labeler.label(df.copy(), method=method)


def get_feature_specs(df, cfg):
    horizons = cfg["horizons"]
    horizon_cols = [f"fail_{h}" for h in horizons]
    feature_cols = cfg["features"]["input_cols"]
    window = cfg["features"]["window_size"]
    return horizons, horizon_cols, feature_cols, window


def run_xgboost_cv(df, feature_cols, horizon_cols, window, cfg, horizons=None):
    model_cfg = cfg["models"]["xgboost"]
    model_cfg["window_size"] = window
    calibrator = ProbabilityCalibrator(method=cfg["calibration"]["method"])
    log("Running leave-battery-out CV with XGBoost...")
    t0 = time.time()
    cv = leave_battery_out_cv(
        df, XGBoostHazard, model_cfg, feature_cols, horizon_cols, calibrator,
        horizons=horizons)
    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s")
    return cv


# ── EXPERIMENTS ────────────────────────────────────────────

def exp_baseline(cfg):
    log("=" * 50)
    log("EXPERIMENT: Baseline Replication")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None
    df = label_data(df, "single", cfg)
    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    cv = run_xgboost_cv(df, feature_cols, horizon_cols, window, cfg,
                        horizons=horizons)

    metrics_raw = compute_metrics(cv["targets"], cv["predictions"], horizons=horizons)
    metrics_cal = compute_metrics(cv["targets"], cv["calibrated"], horizons=horizons)

    # Per-fold stats
    per_fold_aucs = {h: [] for h in horizons}
    for fold in cv["per_fold_raw"]:
        for h in horizons:
            v = fold["per_horizon"][h]["auc"]
            per_fold_aucs[h].append(v if v is not None else float("nan"))

    log("--- Results ---")
    for h in horizons:
        m = metrics_raw["per_horizon"][h]
        vals = np.array(per_fold_aucs[h])
        mu = np.nanmean(vals)
        sd = np.nanstd(vals, ddof=1) if np.sum(~np.isnan(vals)) > 1 else 0
        log(f"  H={h}: raw_AUC={m['auc']} (per-fold {mu:.3f}±{sd:.3f}), "
            f"cal_AUC={metrics_cal['per_horizon'][h]['auc']}")
    mu_macro = np.nanmean([
        metrics_raw["per_horizon"][h]["auc"]
        if metrics_raw["per_horizon"][h]["auc"] is not None else float("nan")
        for h in horizons
    ])
    log(f"  Macro raw AUC: {mu_macro:.4f}")
    log(f"  Macro cal AUC: {metrics_cal['macro_avg']['auc']:.4f}")

    output = {"raw": metrics_raw, "calibrated": metrics_cal,
              "per_fold_raw": cv["per_fold_raw"],
              "per_fold_cal": cv["per_fold_cal"]}
    save_results("baseline", output)

    # Dispatch trade-off sweep
    P_cal = cv["calibrated"]
    y_true = cv["targets"]
    for tau in cfg["dispatch"]["tau_range"]:
        policy = ThresholdPolicy(tau=tau)
        E, _dec = policy.decide(P_cal[:, 0], 0.5)
        n_d = (_dec).sum()
        flr = (y_true[:, 0] * _dec).sum() / n_d if n_d > 0 else 0
        log(f"  τ={tau:.2f}: energy={E.sum():.1f}, fail_rate={flr:.4f}")

    return output


def exp_models(cfg):
    log("=" * 50)
    log("EXPERIMENT: Multi-Model Benchmark")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None
    df = label_data(df, "single", cfg)
    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    results = {}
    model_cfg = cfg["models"]["xgboost"]
    model_cfg["window_size"] = window

    single_model = cfg["execution"].get("model", None)

    # XGBoost (always included unless --model specified something else)
    if single_model is None or single_model == "xgboost":
        cv_xgb = run_xgboost_cv(df, feature_cols, horizon_cols, window, cfg)
        m = compute_metrics(cv_xgb["targets"], cv_xgb["predictions"], horizons=horizons)
        results["XGBoost"] = m["macro_avg"]
        log(f"  XGBoost: AUC={m['macro_avg']['auc']:.4f}")

    # DL models (only in --full mode, or when --model specifies one)
    run_dl = (cfg["execution"]["mode"] == "full") or (single_model is not None)
    if run_dl:
        from src.models.lstm_hazard import LSTMHazard
        from src.models.tcn_hazard import TCNHazard
        from src.models.transformer_hazard import TransformerHazard

        model_map = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer"}
        for name, cls in [("LSTM", LSTMHazard), ("TCN", TCNHazard),
                          ("Transformer", TransformerHazard)]:
            if single_model is not None and name.lower() != single_model:
                continue
            log(f"  Training {name} (this will take a while on CPU)...")
            mc = cfg["models"].get(name.lower(), cfg["models"]["xgboost"])
            mc["window_size"] = window
            try:
                cv = leave_battery_out_cv(
                    df, cls, mc, feature_cols, horizon_cols, None)
                m = compute_metrics(cv["targets"], cv["predictions"],
                                    horizons=horizons)
                results[name] = m["macro_avg"]
                log(f"  {name}: AUC={m['macro_avg']['auc']:.4f}")
            except Exception as e:
                log(f"  {name} FAILED: {e}")
                results[name] = {"auc": 0, "brier": 0}
    else:
        log("  Skipping DL models (use --full to include).")

    save_results("model_comparison", results)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_auc_comparison(results, ax)
    save_figure(fig, os.path.join(FIGURES_DIR, "model_comparison.png"))

    return results


def exp_chemistry(cfg):
    log("=" * 50)
    log("EXPERIMENT: Cross-Chemistry Validation")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None
    df = label_data(df, "single", cfg)
    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    results = {}
    model_cfg = cfg["models"]["xgboost"]
    model_cfg["window_size"] = window
    calibrator = ProbabilityCalibrator(method=cfg["calibration"]["method"])

    for chem in ["LCO"]:  # start with one; LFP/K2 need more disk space
        log(f"Testing on {chem}...")
        df_calce = load_calce(cfg, [chem])
        if df_calce.empty:
            log(f"  No {chem} data, skipping.")
            continue
        df_calce = label_data(df_calce, "single", cfg)
        combined = pd.concat([df, df_calce], ignore_index=True)
        cv = leave_battery_out_cv(
            combined, XGBoostHazard, model_cfg, feature_cols, horizon_cols,
            calibrator)
        m = compute_metrics(cv["targets"], cv["calibrated"], horizons=horizons)
        results[chem] = m["macro_avg"]
        log(f"  {chem}: AUC={m['macro_avg']['auc']:.4f}")

    save_results("cross_chemistry", results)
    return results


def exp_composite(cfg):
    log("=" * 50)
    log("EXPERIMENT: Composite Failure Labels")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None

    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    results = {}
    for method_name, method in [("single", "single"), ("multi", "multi")]:
        log(f"  Labels: {method_name}")
        df_labeled = label_data(df, method, cfg)
        cv = run_xgboost_cv(df_labeled, feature_cols,
                            [f"fail_{h}" for h in horizons], window, cfg)
        m = compute_metrics(cv["targets"], cv["calibrated"], horizons=horizons)
        results[method_name] = m["macro_avg"]
        log(f"    AUC={m['macro_avg']['auc']:.4f}")

    save_results("composite_failure", results)
    return results


def exp_dispatch(cfg):
    log("=" * 50)
    log("EXPERIMENT: Dispatch Policy Comparison")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None
    df = label_data(df, "single", cfg)
    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    cv = run_xgboost_cv(df, feature_cols, horizon_cols, window, cfg)
    P_cal = cv["calibrated"]
    y_true = cv["targets"]

    results = {}
    policies = {
        "Always dispatch": ThresholdPolicy(tau=1.0),  # tau=1 accepts all
        "Threshold(τ=0.2)": ThresholdPolicy(tau=0.20),
        "Threshold(τ=0.1)": ThresholdPolicy(tau=0.10),
        "Derating(α=2)": ContinuousDeratingPolicy(alpha=2.0),
        "Derating(α=5)": ContinuousDeratingPolicy(alpha=5.0),
    }

    for pname, policy in policies.items():
        E, dec = policy.decide(P_cal[:, 0], 0.5)
        n_d = dec.sum()
        flr = (y_true[:, 0] * dec).sum() / n_d if n_d > 0 else 0
        results[pname] = {"energy": round(E.sum(), 2),
                          "failure_rate": round(flr, 4)}
        log(f"  {pname:25s}: energy={E.sum():.1f}, fail_rate={flr:.4f}")

    save_results("dispatch_comparison", results)
    return results


def exp_market(cfg):
    log("=" * 50)
    log("EXPERIMENT: Market Simulation")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None
    df = label_data(df, "single", cfg)
    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    cv = run_xgboost_cv(df, feature_cols, horizon_cols, window, cfg)
    P_cal = cv["calibrated"]

    # Build per-cell eol_cycle from the labeled DataFrame
    eol_by_cell = df.groupby("cell_id")["eol_cycle"].first().dropna()
    # Use the median eol_cycle across cells for the stacked simulation
    eol_cycle = int(eol_by_cell.median()) if len(eol_by_cell) > 0 else 200

    ms = MarketSimulator(
        price_mean=cfg["market"]["price_mean"],
        price_std=cfg["market"]["price_std"],
        price_ar_coeff=cfg["market"]["price_ar_coeff"],
        service_energy_kwh=cfg["market"]["service_energy_kwh"],
        penalty_cost=cfg["market"]["penalty_cost"],
        seed=42)

    results = {}
    for pname, policy_fn in [
        ("Always dispatch", ThresholdPolicy(tau=1.0)),
        ("Threshold(τ=0.2)", ThresholdPolicy(tau=0.20)),
        ("Derating(α=2)", ContinuousDeratingPolicy(alpha=2.0)),
    ]:
        log(f"  Simulating {pname}...")
        t0 = time.time()
        mc = ms.monte_carlo(
            P_cal[:150, 0], eol_cycle, horizon=20,
            dispatch_policy=policy_fn,
            n_scenarios=cfg["market"]["n_scenarios"])
        r = {
            "mean_revenue": float(np.round(mc["revenue"].mean(), 2)),
            "std_revenue": float(np.round(mc["revenue"].std(), 2)),
            "mean_energy": float(np.round(mc["energy_delivered"].mean(), 2)),
            "mean_failure_rate": float(np.round(mc["failure_rate"].mean(), 4)),
        }
        results[pname] = r
        log(f"    revenue={r['mean_revenue']} ±{r['std_revenue']}, "
            f"fail={r['mean_failure_rate']}")

    save_results("market_simulation", results)
    return results


def exp_ablation(cfg):
    log("=" * 50)
    log("EXPERIMENT: Ablation Study")
    log("=" * 50)

    df = load_nasa(cfg)
    if df.empty:
        return None

    horizons, horizon_cols, feature_cols, window = get_feature_specs(df, cfg)

    results = {}

    mc = cfg["models"]["xgboost"]
    mc["window_size"] = window
    calib = ProbabilityCalibrator(method=cfg["calibration"]["method"])

    # 1. Always dispatch (baseline) — use same dispatch metric as model rows
    df_s = label_data(df, "single", cfg)
    cv_base = leave_battery_out_cv(
        df_s, XGBoostHazard, mc, feature_cols, horizon_cols, None)
    policy = ThresholdPolicy(tau=1.0)
    P_base = np.ones(len(cv_base["targets"])) * 0.5
    E, dec = policy.decide(P_base, 0.5)
    flr = (cv_base["targets"][:, 0] * dec).sum() / dec.sum() if dec.sum() > 0 else 0
    results["Always dispatch"] = {
        "failure_rate": round(flr, 4),
        "energy": round(E.sum(), 2),
    }

    # 2. Raw hazard (no calibration)
    cv = leave_battery_out_cv(
        df_s, XGBoostHazard, mc, feature_cols, horizon_cols, None)
    P = cv["predictions"][:, 0]
    policy = ThresholdPolicy(tau=0.20)
    E, dec = policy.decide(P, 0.5)
    n_d = dec.sum()
    flr = (cv["targets"][:, 0] * dec).sum() / n_d if n_d > 0 else 0
    results["Raw Hazard"] = {"failure_rate": round(flr, 4),
                             "energy": round(E.sum(), 2)}

    # 3. Hazard + Calibration
    cv = leave_battery_out_cv(
        df_s, XGBoostHazard, mc, feature_cols, horizon_cols, calib)
    P = cv["calibrated"][:, 0]
    E, dec = policy.decide(P, 0.5)
    n_d = dec.sum()
    flr = (cv["targets"][:, 0] * dec).sum() / n_d if n_d > 0 else 0
    results["+ Calibration"] = {"failure_rate": round(flr, 4),
                                "energy": round(E.sum(), 2)}

    # 4. + Composite failure
    df_m = label_data(df, "multi", cfg)
    cv = leave_battery_out_cv(
        df_m, XGBoostHazard, mc, feature_cols,
        [f"fail_{h}" for h in horizons], calib)
    P = cv["calibrated"][:, 0]
    E, dec = policy.decide(P, 0.5)
    n_d = dec.sum()
    flr = (cv["targets"][:, 0] * dec).sum() / n_d if n_d > 0 else 0
    results["+ Composite Labels"] = {"failure_rate": round(flr, 4),
                                     "energy": round(E.sum(), 2)}

    log("-" * 40)
    for k, v in results.items():
        log(f"  {k:25s}: fail_rate={v['failure_rate']:.4f}, "
            f"energy={v['energy']:.1f}")

    save_results("ablation", results)
    return results


def _bootstrap_ci(values, n_resamples=10000, ci=0.95):
    """Compute percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return float("nan"), float("nan")
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        boot = np.random.choice(values, size=len(values), replace=True)
        means[i] = boot.mean()
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return round(lower, 4), round(upper, 4)


def _delong_test(auc1, auc2, n1, n2):
    """Approximate DeLong test for paired AUC comparison.

    Uses the normal approximation: z = (auc1 - auc2) / sqrt(var1 + var2).
    Returns z-statistic and p-value (two-sided).
    """
    se1 = np.sqrt((auc1 * (1 - auc1) + (n1 - 1) * (auc1 / (2 - auc1) - auc1**2)) / n1)
    se2 = np.sqrt((auc2 * (1 - auc2) + (n2 - 1) * (auc2 / (2 - auc2) - auc2**2)) / n2)
    se_pooled = np.sqrt(se1**2 + se2**2)
    if se_pooled < 1e-10:
        return 0.0, 1.0
    z = (auc1 - auc2) / se_pooled
    p = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 4), round(p, 6)


def _run_scaling_at_threshold(n_values, n_seeds, soh_threshold, cfg):
    """Run scaling study at a given SOH threshold. Returns dict of results."""
    horizons = cfg["horizons"]
    feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]
    window = cfg["features"]["window_size"]
    model_cfg = cfg["models"]["xgboost"].copy()
    model_cfg["window_size"] = window
    calibrator = ProbabilityCalibrator(method=cfg["calibration"]["method"])

    results = {"soh_threshold": soh_threshold, "N": n_values,
               "auc_mean": [], "auc_ci_low": [], "auc_ci_high": [],
               "per_seed": {}}

    for idx, N in enumerate(n_values):
        seed_aucs = []
        for seed in range(n_seeds):
            df = generate_synthetic_nasa(n_cells=N, n_cycles=300, seed=seed * 100 + N)
            labeler = CompositeFailureLabeler(
                soh_threshold=soh_threshold,
                sudden_drop=cfg["failure"]["sudden_drop_threshold"])
            df = labeler.label(df.copy(), method="single")
            horizon_cols = [f"fail_{h}" for h in horizons]

            cv = leave_battery_out_cv(
                df, XGBoostHazard, model_cfg, feature_cols, horizon_cols,
                calibrator, seed=seed, horizons=horizons)

            if len(cv["predictions"]) == 0:
                auc = float("nan")
            else:
                m = compute_metrics(cv["targets"], cv["predictions"], horizons=horizons)
                auc = m["macro_avg"]["auc"]
            seed_aucs.append(auc)
            if f"seed_{seed}" not in results["per_seed"]:
                results["per_seed"][f"seed_{seed}"] = {}
            results["per_seed"][f"seed_{seed}"][str(N)] = auc

        mean_auc = round(np.nanmean(seed_aucs), 4) if seed_aucs else float("nan")
        lo, hi = _bootstrap_ci(seed_aucs)
        results["auc_mean"].append(mean_auc)
        results["auc_ci_low"].append(lo)
        results["auc_ci_high"].append(hi)

    return results


def exp_scaling(cfg):
    log("=" * 50)
    log("EXPERIMENT: Synthetic Scaling Study")
    log("=" * 50)

    n_values = [2, 3, 5, 8, 12, 20]
    n_seeds = 20
    horizons = cfg["horizons"]
    feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]
    window = cfg["features"]["window_size"]
    model_cfg = cfg["models"]["xgboost"].copy()
    model_cfg["window_size"] = window
    calibrator = ProbabilityCalibrator(method=cfg["calibration"]["method"])

    results = {"N": n_values, "n_seeds": n_seeds,
               "auc_mean": [], "auc_std": [],
               "auc_ci_low": [], "auc_ci_high": [],
               "delong": {}, "per_seed": {}}

    all_seed_aucs = []

    for idx, N in enumerate(n_values):
        log(f"  N={N} ({idx+1}/{len(n_values)})")
        seed_aucs = []
        for seed in range(n_seeds):
            df = generate_synthetic_nasa(n_cells=N, n_cycles=300, seed=seed * 100 + N)
            labeler = CompositeFailureLabeler(
                soh_threshold=cfg["failure"]["soh_threshold"],
                sudden_drop=cfg["failure"]["sudden_drop_threshold"])
            df = labeler.label(df.copy(), method="single")
            horizon_cols = [f"fail_{h}" for h in horizons]

            t0 = time.time()
            cv = leave_battery_out_cv(
                df, XGBoostHazard, model_cfg, feature_cols, horizon_cols,
                calibrator, seed=seed, horizons=horizons)
            elapsed = time.time() - t0

            if len(cv["predictions"]) == 0:
                auc = float("nan")
            else:
                m = compute_metrics(cv["targets"], cv["predictions"], horizons=horizons)
                auc = m["macro_avg"]["auc"]
            seed_aucs.append(auc)
            if f"seed_{seed}" not in results["per_seed"]:
                results["per_seed"][f"seed_{seed}"] = {}
            results["per_seed"][f"seed_{seed}"][str(N)] = auc
            sys.stdout.write(f"\r    N={N} seed={seed+1}/{n_seeds} AUC={auc:.4f} ({elapsed:.1f}s)  ")
            sys.stdout.flush()
        sys.stdout.write("\n")

        mean_auc = round(np.nanmean(seed_aucs), 4)
        std_auc = round(np.nanstd(seed_aucs, ddof=1), 4) if len(seed_aucs) > 1 else 0.0
        lo, hi = _bootstrap_ci(seed_aucs)
        results["auc_mean"].append(mean_auc)
        results["auc_std"].append(std_auc)
        results["auc_ci_low"].append(lo)
        results["auc_ci_high"].append(hi)
        all_seed_aucs.append(seed_aucs)
        log(f"  N={N}: mean AUC={mean_auc:.4f} +/- {std_auc:.4f}  "
            f"95% CI=[{lo:.4f}, {hi:.4f}]")

    # DeLong tests between adjacent N values
    log("  --- DeLong significance tests (adjacent N) ---")
    for i in range(len(n_values) - 1):
        n1_val = n_values[i]
        n2_val = n_values[i + 1]
        auc1 = results["auc_mean"][i]
        auc2 = results["auc_mean"][i + 1]
        n_obs = max(len([x for x in all_seed_aucs[i] if not np.isnan(x)]),
                    len([x for x in all_seed_aucs[i + 1] if not np.isnan(x)]))
        z, p = _delong_test(auc1, auc2, n_obs, n_obs)
        results["delong"][f"{n1_val}_vs_{n2_val}"] = {"z": z, "p": p}
        sig = "significant" if p < 0.05 else "not significant"
        log(f"    N={n1_val} vs N={n2_val}: z={z:.4f}, p={p:.6f} ({sig})")

    save_results("scaling_monte_carlo", results)
    return results


def exp_soh_sensitivity(cfg):
    log("=" * 50)
    log("EXPERIMENT: SOH Threshold Sensitivity")
    log("=" * 50)

    n_values = [2, 5, 12, 20]
    n_seeds = 10
    thresholds = [0.70, 0.75, 0.80]

    results = {"thresholds": {}, "n_values": n_values}
    for thresh in thresholds:
        log(f"  SOH threshold = {thresh}")
        r = _run_scaling_at_threshold(n_values, n_seeds, thresh, cfg)
        results["thresholds"][str(thresh)] = {
            "auc_mean": r["auc_mean"],
            "auc_ci_low": r["auc_ci_low"],
            "auc_ci_high": r["auc_ci_high"],
        }
        for idx, N in enumerate(n_values):
            log(f"    N={N}: AUC={r['auc_mean'][idx]}  CI=[{r['auc_ci_low'][idx]}, {r['auc_ci_high'][idx]}]")

    save_results("soh_sensitivity", results)
    return results


def exp_calibration_leakage(cfg):
    log("=" * 50)
    log("EXPERIMENT: Calibration Leakage Demonstration")
    log("=" * 50)

    N = 20
    n_seeds = 10
    horizons = cfg["horizons"]
    feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]
    window = cfg["features"]["window_size"]
    model_cfg = cfg["models"]["xgboost"].copy()
    model_cfg["window_size"] = window

    correct_aucs = []
    leaked_aucs = []

    for seed in range(n_seeds):
        df = generate_synthetic_nasa(n_cells=N, n_cycles=300, seed=seed * 100 + N)
        labeler = CompositeFailureLabeler(
            soh_threshold=cfg["failure"]["soh_threshold"],
            sudden_drop=cfg["failure"]["sudden_drop_threshold"])
        df = labeler.label(df.copy(), method="single")
        horizon_cols = [f"fail_{h}" for h in horizons]

        cell_ids = np.asarray(df["cell_id"].unique())
        rng_seed = np.random.default_rng(seed)
        rng_seed.shuffle(cell_ids)

        for test_cell in cell_ids[:3]:
            train_mask = df["cell_id"] != test_cell
            test_mask = df["cell_id"] == test_cell

            X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
            y_train = df.loc[train_mask, horizon_cols].values.astype(np.float32)
            X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
            y_test = df.loc[test_mask, horizon_cols].values.astype(np.float32)

            split = int(len(X_train) * 0.8)
            X_tr, y_tr = X_train[:split], y_train[:split]
            X_val, y_val = X_train[split:], y_train[split:]

            # Train model
            mdl = XGBoostHazard(config=model_cfg)
            mdl.fit(X_tr, y_tr, X_val, y_val)
            y_pred = mdl.predict_proba(X_test)
            y_pred_val = mdl.predict_proba(X_val)

            # Correct calibration: fit on val, apply to test
            cal_correct = ProbabilityCalibrator(method="isotonic")
            cal_correct.fit(y_pred_val, y_val, horizons=horizons)
            y_cal_correct = cal_correct.transform(y_pred, horizons=horizons)

            # Leaked calibration: fit on test, apply to test
            cal_leaked = ProbabilityCalibrator(method="isotonic")
            cal_leaked.fit(y_pred, y_test, horizons=horizons)
            y_cal_leaked = cal_leaked.transform(y_pred, horizons=horizons)

            m_correct = compute_metrics(y_test, y_cal_correct, horizons=horizons)
            m_leaked = compute_metrics(y_test, y_cal_leaked, horizons=horizons)

            correct_aucs.append(m_correct["macro_avg"]["auc"])
            leaked_aucs.append(m_leaked["macro_avg"]["auc"])

    mean_correct = round(np.mean(correct_aucs), 4)
    mean_leaked = round(np.mean(leaked_aucs), 4)
    inflation = round(mean_leaked - mean_correct, 4)

    results = {
        "N": N,
        "n_seeds": n_seeds,
        "correct_auc_mean": mean_correct,
        "leaked_auc_mean": mean_leaked,
        "inflation": inflation,
        "per_fold_correct": correct_aucs,
        "per_fold_leaked": leaked_aucs,
    }

    log(f"  Correct calibration (fit on val): mean AUC = {mean_correct:.4f}")
    log(f"  Leaked calibration (fit on test): mean AUC = {mean_leaked:.4f}")
    log(f"  Inflation: +{inflation:.4f} AUC")

    save_results("calibration_leakage_demo", results)
    return results


def exp_ks_test(cfg):
    log("=" * 50)
    log("EXPERIMENT: KS Test — Synthetic vs Real Degradation")
    log("=" * 50)

    loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
    df_real = loader.load_classic()
    if df_real.empty:
        log("  No real NASA data found, skipping.")
        return None

    df_syn = generate_synthetic_nasa(n_cells=4, seed=42)

    # Compute degradation rates (d_soh)
    real_rates = df_real.groupby("cell_id")["soh"].diff().dropna().values
    syn_rates = df_syn.groupby("cell_id")["soh"].diff().dropna().values

    stat, p_value = ks_2samp(real_rates, syn_rates)

    results = {
        "real_n_obs": len(real_rates),
        "syn_n_obs": len(syn_rates),
        "real_mean_dsoh": float(np.mean(real_rates)),
        "syn_mean_dsoh": float(np.mean(syn_rates)),
        "ks_statistic": float(stat),
        "ks_p_value": float(p_value),
    }

    log(f"  Real d_soh: mean={results['real_mean_dsoh']:.6f}, n={results['real_n_obs']}")
    log(f"  Syn d_soh: mean={results['syn_mean_dsoh']:.6f}, n={results['syn_n_obs']}")
    log(f"  KS statistic: {stat:.4f}, p-value: {p_value:.6f}")
    if p_value < 0.05:
        log("  → Distributions differ significantly (reject H0)")
    else:
        log("  → No significant difference detected")

    save_results("ks_test", results)
    return results


# ── MAIN ──────────────────────────────────────────────────

EXPERIMENTS = {
    "baseline": (exp_baseline, "Replicate original paper"),
    "models": (exp_models, "Multi-model benchmark"),
    "chemistry": (exp_chemistry, "Cross-chemistry validation"),
    "composite": (exp_composite, "Composite failure labels"),
    "dispatch": (exp_dispatch, "Dispatch policy comparison"),
    "market": (exp_market, "Market simulation"),
    "ablation": (exp_ablation, "Ablation study"),
    "scaling": (exp_scaling, "Synthetic scaling study (20 seeds, bootstrap CIs)"),
    "soh_sensitivity": (exp_soh_sensitivity, "SOH threshold sensitivity analysis"),
    "calibration_leakage": (exp_calibration_leakage, "Calibration leakage demo on synthetic data"),
    "ks_test": (exp_ks_test, "KS test: synthetic vs real degradation rates"),
}

EXPERIMENT_ORDER_QUICK = ["baseline", "dispatch", "composite", "market", "ablation", "ks_test"]
EXPERIMENT_ORDER_FULL = ["baseline", "models", "chemistry",
                         "composite", "dispatch", "market", "ablation"]


def main():
    parser = argparse.ArgumentParser(description="Extension paper experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: XGBoost only, skip DL")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: include DL models (CPU: slow)")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single model: xgboost, lstm, tcn, transformer")
    parser.add_argument("--expt", type=str, default=None,
                        help=f"Single experiment: {list(EXPERIMENTS.keys())}")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, (_, desc) in EXPERIMENTS.items():
            print(f"  {name:15s}  {desc}")
        return

    cfg = load_config()

    # Set execution mode
    if args.quick:
        cfg["execution"]["mode"] = "quick"
    elif args.full:
        cfg["execution"]["mode"] = "full"

    if args.model:
        cfg["execution"]["model"] = args.model.lower()
        cfg["execution"]["mode"] = "single_model"
        log(f"Single model: {args.model}")
    else:
        cfg["execution"].pop("model", None)

    log(f"Mode: {cfg['execution']['mode']}")
    log(f"Data dir: {os.path.abspath(cfg['execution']['data_dir'])}")

    # Run selected experiment or all
    if args.expt:
        if args.expt not in EXPERIMENTS:
            print(f"Unknown experiment: {args.expt}")
            print(f"Choose from: {list(EXPERIMENTS.keys())}")
            return
        fn, desc = EXPERIMENTS[args.expt]
        log(f"Starting: {desc}")
        t0 = time.time()
        result = fn(cfg)
        elapsed = time.time() - t0
        if result is None:
            log("Experiment aborted (no data).")
        else:
            log(f"Completed in {elapsed:.1f}s")
    else:
        order = EXPERIMENT_ORDER_QUICK if cfg["execution"]["mode"] == "quick" else EXPERIMENT_ORDER_FULL
        for name in order:
            fn, desc = EXPERIMENTS[name]
            log(f"\n{'='*50}")
            log(f"Starting: {desc}")
            log(f"{'='*50}")
            t0 = time.time()
            result = fn(cfg)
            elapsed = time.time() - t0
            if result is None:
                log("Skipped (no data).")
            else:
                log(f"Completed in {elapsed:.1f}s")
            log("")

    log("All done.")


if __name__ == "__main__":
    main()
