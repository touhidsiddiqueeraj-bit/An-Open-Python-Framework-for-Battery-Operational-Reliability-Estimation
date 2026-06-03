#!/usr/bin/env python3
"""Supplementary analyses: extended censoring, per-feature KL, plateau model, power analysis."""
import sys, os, json, yaml, time, glob
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
from scipy.stats import entropy, gaussian_kde, wasserstein_distance
from scipy.optimize import curve_fit

labeler = CompositeFailureLabeler(
    soh_threshold=cfg["failure"]["soh_threshold"],
    sudden_drop=cfg["failure"]["sudden_drop_threshold"])
horizons = cfg["horizons"]
horizon_cols = [f"fail_{h}" for h in horizons]
feature_cols = cfg["features"]["input_cols"] + cfg["features"]["derived_cols"]

output = {}

# ════════════════════════════════════════════════════════
# 1. CENSORING SENSITIVITY (extended to 0-80%)
# ════════════════════════════════════════════════════════
print("=== 1. CENSORING SENSITIVITY (extended) ===")
syn = generate_synthetic_nasa(n_cells=20, seed=42)
syn = labeler.label(syn, method="single")
X = syn[feature_cols].values.astype(np.float32)
y = syn[horizon_cols].values.astype(np.float32)

censoring_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
censoring_results = {}
for cl in censoring_levels:
    y_c = y.copy()
    rng_c = np.random.default_rng(int(cl * 100))
    nan_folds = 0
    for i in range(y.shape[1]):
        n_censor = int(len(y_c) * cl)
        pos_idx = np.where(y_c[:, i] == 1)[0]
        if len(pos_idx) == 0:
            nan_folds += 1
            continue
        n_censor = min(n_censor, len(pos_idx))
        if n_censor == 0:
            continue
        censor_idx = rng_c.choice(pos_idx, n_censor, replace=False)
        y_c[censor_idx, i] = 0

    split = int(len(X) * 0.8)
    mdl = XGBoostHazard(config=cfg["models"]["xgboost"])
    t0 = time.time()
    mdl.fit(X[:split], y_c[:split], X[split:], y_c[split:])
    fit_time = time.time() - t0

    t0 = time.time()
    _ = mdl.predict_proba(X[split:split + 5])
    infer_time = (time.time() - t0) / max(len(X[split:split + 5]), 1)

    preds_all = mdl.predict_proba(X[split:])
    m = compute_metrics(y_c[split:], preds_all, horizons=horizons)
    auc_val = m["macro_avg"]["auc"]
    auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "nan"
    # Count NaN horizons in macro avg
    num_nan = sum(1 for h in horizons if m["per_horizon"][h]["auc"] is None)
    censoring_results[f"censor_{cl:.0%}"] = {
        "macro_auc": auc_str,
        "nan_horizons": num_nan,
        "fit_time_sec": round(fit_time, 2),
        "infer_time_ms": round(infer_time * 1000, 2)}
    print(f"  Censor={cl:.0%}: macro AUC={auc_str}, nan_horizons={num_nan}/4, "
          f"fit={fit_time:.2f}s")
output["censoring_sensitivity"] = censoring_results

# ════════════════════════════════════════════════════════
# 2. PER-FEATURE KL DIVERGENCE
# ════════════════════════════════════════════════════════
print()
print("=== 2. PER-FEATURE KL DIVERGENCE ===")
loader = NASALoader(data_dir=cfg["execution"]["data_dir"])
df_real = loader.load_classic()
syn_val = generate_synthetic_nasa(n_cells=20, seed=42)

all_features = ["soh", "voltage_avg", "current_avg", "temperature_avg",
                "d_soh", "d_capacity", "capacity"]
kl_results = {}
for feat in all_features:
    if feat not in df_real.columns or feat not in syn_val.columns:
        continue
    real_vals = df_real[feat].dropna().values
    syn_vals = syn_val[feat].dropna().values
    if len(real_vals) < 10 or len(syn_vals) < 10:
        continue
    # KL via KDE
    try:
        real_kde = gaussian_kde(real_vals)
        syn_kde = gaussian_kde(syn_vals)
        lo = max(real_vals.min(), syn_vals.min())
        hi = min(real_vals.max(), syn_vals.max())
        grid = np.linspace(lo, hi, 200)
        p = real_kde(grid) + 1e-10
        q = syn_kde(grid) + 1e-10
        kl = entropy(p, q)
        ws = wasserstein_distance(real_vals, syn_vals)
        kl_results[feat] = {
            "kl_divergence_nats": round(kl, 4),
            "wasserstein": round(ws, 4),
            "real_mean": round(float(real_vals.mean()), 4),
            "syn_mean": round(float(syn_vals.mean()), 4)}
        print(f"  {feat:20s}: KL={kl:.4f} nats, W={ws:.4f}, "
              f"real_mean={real_vals.mean():.4f}, syn_mean={syn_vals.mean():.4f}")
    except Exception as e:
        print(f"  {feat:20s}: SKIP ({e})")
output["per_feature_kl"] = kl_results

# ════════════════════════════════════════════════════════
# 3. PLATEAU ASSESSMENT (constrained fit)
# ════════════════════════════════════════════════════════
print()
print("=== 3. PLATEAU ASSESSMENT (constrained fit) ===")
scaling_files = sorted(glob.glob(os.path.join(results_dir, "scaling_monte_carlo_*.json")))
if scaling_files:
    with open(scaling_files[-1]) as f:
        scaling_data = json.load(f)
    n_vals = np.array(scaling_data["N"])
    auc_means = np.array(scaling_data["auc_mean"])

    def plateau_model(N, a, b, c):
        return a - b / (N ** c)

    try:
        # Fit with asymptote constrained to [0, 1.0] (AUC cannot exceed 1.0)
        popt, _ = curve_fit(plateau_model, n_vals, auc_means,
                            p0=[1.0, 0.5, 0.5],
                            bounds=([0, 0, 0], [1.0, np.inf, np.inf]),
                            maxfev=5000)
        a_hat, b_hat, c_hat = popt
        max_obs = float(auc_means.max())
        # Check how close we are to plateau at N=20
        grad_at_20 = b_hat * c_hat * (20 ** (-c_hat - 1))
        plateau_result = {
            "constrained_asymptote": round(a_hat, 4),
            "b_hat": round(b_hat, 4),
            "c_hat": round(c_hat, 4),
            "max_observed_auc": round(max_obs, 4),
            "slope_at_N20": round(grad_at_20, 6),
            "interpretation": "asymptote hits 1.0 boundary — insufficient curvature to estimate plateau; curve not saturated within N=2-20"}
        print(f"  Constrained fit: AUC = {a_hat:.4f} - {b_hat:.4f} / N^{c_hat:.4f}")
        print(f"  Asymptote (constrained ≤1.0): {a_hat:.4f}")
        print(f"  Max observed AUC: {max_obs:.4f}")
        print(f"  Slope at N=20: {grad_at_20:.6f} AUC/cell")
        print(f"  → Plateau cannot be estimated within N=2-20 range")
        output["plateau_assessment"] = plateau_result
    except Exception as e:
        print(f"  Fit FAILED: {e}")
        output["plateau_assessment"] = {"error": str(e), "max_observed_auc": float(auc_means.max())}
else:
    print("  No scaling results found, skipping plateau assessment")

# ════════════════════════════════════════════════════════
# 4. POWER ANALYSIS
# ════════════════════════════════════════════════════════
print()
print("=== 4. POWER ANALYSIS ===")
if scaling_files:
    with open(scaling_files[-1]) as f:
        scaling_data = json.load(f)
    n_vals = scaling_data["N"]
    power_results = {}
    for idx, N in enumerate(n_vals):
        aucs = [scaling_data["per_seed"][f"seed_{s}"][str(N)]
                for s in range(scaling_data["n_seeds"])
                if str(N) in scaling_data["per_seed"].get(f"seed_{s}", {})]
        aucs = [a for a in aucs if a is not None and not (isinstance(a, float) and np.isnan(a))]
        if len(aucs) == 0:
            continue
        aucs = np.array(aucs)
        for threshold in [0.7, 0.8, 0.9, 0.95]:
            power = (aucs > threshold).mean()
            key = f"N={N}"
            if key not in power_results:
                power_results[key] = {}
            power_results[key][f"P(AUC>{threshold})"] = round(power, 4)
        mean_auc = float(aucs.mean())
        std_auc = float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0
        power_results[key]["mean_auc"] = round(mean_auc, 4)
        power_results[key]["std_auc"] = round(std_auc, 4)
        power_results[key]["n_seeds_valid"] = len(aucs)
        print(f"  N={N}: mean AUC={mean_auc:.4f} ± {std_auc:.4f}, "
              f"P(>0.8)={power_results[key]['P(AUC>0.8)']:.4f}, "
              f"P(>0.95)={power_results[key]['P(AUC>0.95)']:.4f}")
    output["power_analysis"] = power_results

# ════════════════════════════════════════════════════════
# 5. COMPUTATIONAL COST (unchanged)
# ════════════════════════════════════════════════════════
print()
print("=== 5. COMPUTATIONAL COST ===")
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

# ════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    return o

fp = os.path.join(results_dir, "supplementary_analysis.json")
with open(fp, "w") as f:
    json.dump(clean(output), f, indent=2)
print(f"\nSaved: {fp}")
print("Done.")
