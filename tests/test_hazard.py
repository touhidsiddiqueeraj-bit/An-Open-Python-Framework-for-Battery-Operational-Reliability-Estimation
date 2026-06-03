"""Unit tests for core hazard model logic."""
import sys, os, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from src.models.xgboost_hazard import XGBoostHazard
from src.data.synthetic import generate_synthetic_nasa
from src.data.composite_failure import CompositeFailureLabeler
from src.evaluation.metrics import compute_metrics


def test_fit_predict_roundtrip():
    """XGBoostHazard.fit() followed by predict_proba() returns valid-shape output."""
    df = generate_synthetic_nasa(n_cells=4, seed=42)
    labeler = CompositeFailureLabeler(soh_threshold=0.70, sudden_drop=0.05)
    df = labeler.label(df, method="single")
    feature_cols = ["soh", "voltage_avg", "current_avg", "temperature_avg",
                    "cycle", "d_soh", "d_capacity"]
    horizon_cols = ["fail_10", "fail_20", "fail_30", "fail_50"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[horizon_cols].values.astype(np.float32)
    split = int(len(X) * 0.8)
    mdl = XGBoostHazard(config={"n_estimators": 10, "max_depth": 2, "verbosity": 0})
    mdl.fit(X[:split], y[:split], X[split:], y[split:])
    preds = mdl.predict_proba(X[split:])
    assert preds.shape == (len(X[split:]), 4), f"Expected ({len(X[split:])}, 4), got {preds.shape}"
    assert preds.min() >= 0.0 and preds.max() <= 1.0, "Probabilities out of [0, 1] range"


def test_fit_without_validation():
    """XGBoostHazard.fit() without validation set still produces predictions."""
    df = generate_synthetic_nasa(n_cells=3, seed=42)
    labeler = CompositeFailureLabeler(soh_threshold=0.70, sudden_drop=0.05)
    df = labeler.label(df, method="single")
    feature_cols = ["soh", "voltage_avg", "current_avg", "temperature_avg",
                    "cycle", "d_soh", "d_capacity"]
    horizon_cols = ["fail_10", "fail_20", "fail_30", "fail_50"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[horizon_cols].values.astype(np.float32)
    mdl = XGBoostHazard(config={"n_estimators": 10, "max_depth": 2, "verbosity": 0})
    mdl.fit(X, y)
    preds = mdl.predict_proba(X)
    assert preds.shape == (len(X), 4)
    assert np.all(np.isfinite(preds))


def test_reproducible_seed():
    """Same seed produces same predictions."""
    df = generate_synthetic_nasa(n_cells=3, seed=42)
    labeler = CompositeFailureLabeler(soh_threshold=0.70, sudden_drop=0.05)
    df = labeler.label(df, method="single")
    feature_cols = ["soh", "voltage_avg", "current_avg", "temperature_avg",
                    "cycle", "d_soh", "d_capacity"]
    horizon_cols = ["fail_10", "fail_20", "fail_30", "fail_50"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[horizon_cols].values.astype(np.float32)
    split = int(len(X) * 0.8)
    mdl1 = XGBoostHazard(config={"n_estimators": 10, "max_depth": 2, "verbosity": 0,
                                  "random_state": 42})
    mdl1.fit(X[:split], y[:split], X[split:], y[split:])
    preds1 = mdl1.predict_proba(X[split:])
    mdl2 = XGBoostHazard(config={"n_estimators": 10, "max_depth": 2, "verbosity": 0,
                                  "random_state": 42})
    mdl2.fit(X[:split], y[:split], X[split:], y[split:])
    preds2 = mdl2.predict_proba(X[split:])
    assert np.allclose(preds1, preds2, atol=1e-6), "Reproducible seed failed"


def test_random_data_gives_valid_auc():
    """Training on random features still produces valid AUC ~0.5."""
    rng = np.random.default_rng(2026)
    n = 200
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = rng.binomial(1, 0.3, size=(n, 4)).astype(np.float32)
    split = n // 2
    mdl = XGBoostHazard(config={"n_estimators": 10, "max_depth": 2, "verbosity": 0})
    mdl.fit(X[:split], y[:split], X[split:], y[split:])
    preds = mdl.predict_proba(X[split:])
    m = compute_metrics(y[split:], preds, horizons=[10, 20, 30, 50])
    auc = m["macro_avg"]["auc"]
    assert auc is not None and not np.isnan(auc), "AUC should be computable"
    # On random data with ~0.3 event rate, AUC should be near 0.5
    assert 0.3 <= auc <= 0.7, f"AUC={auc} on random data should be near 0.5"


def test_multi_horizon_independence():
    """Each horizon trains independently; one failing horizon should not affect others."""
    df = generate_synthetic_nasa(n_cells=4, seed=42)
    labeler = CompositeFailureLabeler(soh_threshold=0.70, sudden_drop=0.05)
    df = labeler.label(df, method="single")
    feature_cols = ["soh", "voltage_avg", "current_avg", "temperature_avg",
                    "cycle", "d_soh", "d_capacity"]
    horizon_cols = ["fail_10", "fail_20", "fail_30", "fail_50"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[horizon_cols].values.astype(np.float32)
    # Corrupt labels for horizon 0 (all zeros — no signal)
    y[:, 0] = 0.0
    split = int(len(X) * 0.8)
    mdl = XGBoostHazard(config={"n_estimators": 10, "max_depth": 2, "verbosity": 0})
    mdl.fit(X[:split], y[:split], X[split:], y[split:])
    preds = mdl.predict_proba(X[split:])
    m = compute_metrics(y[split:], preds, horizons=[10, 20, 30, 50])
    # H=10 should be NaN (single class), others should be valid
    assert m["per_horizon"][10]["auc"] is None, "H=10 should be NaN (only one class)"
    for h in [20, 30, 50]:
        if m["per_horizon"][h]["auc"] is not None:
            assert m["per_horizon"][h]["auc"] > 0.5, f"H={h} should be >0.5"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in locals().items() if k.startswith("test_")]
    n_pass = 0
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    exit(0 if n_fail == 0 else 1)
