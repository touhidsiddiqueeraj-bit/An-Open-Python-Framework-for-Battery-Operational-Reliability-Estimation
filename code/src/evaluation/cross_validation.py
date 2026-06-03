import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


def leave_battery_out_cv(df, model_cls, model_config,
                         feature_cols, horizon_cols,
                         calibrator=None, seed=42, horizons=None):
    """Leave-one-battery-out cross-validation supporting both
    tree-based (flat features) and sequence (sliding window) models.

    Detection:
      - If the model is XGBoostHazard (tree-based), use flat 2D features.
      - Otherwise (LSTM/TCN/Transformer), build 3D sliding windows.
    """
    cell_ids = np.asarray(df["cell_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(cell_ids)

    is_tree = model_cls.__name__ == "XGBoostHazard"

    all_preds, all_targets = [], []
    all_calibrated = []
    fitted_models = []
    per_fold_raw_metrics = []
    per_fold_cal_metrics = []

    for test_cell in cell_ids:
        train_mask = df["cell_id"] != test_cell
        test_mask = df["cell_id"] == test_cell

        X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, horizon_cols].values.astype(np.float32)
        X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
        y_test = df.loc[test_mask, horizon_cols].values.astype(np.float32)

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        if is_tree:
            split = int(len(X_train) * 0.8)
            X_tr, y_tr = X_train[:split], y_train[:split]
            X_val_inner, y_val_inner = X_train[split:], y_train[split:]

            mdl = model_cls(config=model_config)
            mdl.fit(X_tr, y_tr, X_val_inner, y_val_inner)

            y_val_pred = mdl.predict_proba(X_val_inner)
            y_pred = mdl.predict_proba(X_test)

        else:
            window = model_config.get("window_size", 20)
            X_train_seq, y_train_seq = _build_sequences(
                X_train, y_train, df.loc[train_mask, "cell_id"].values, window)
            X_test_seq, y_test_seq = _build_sequences(
                X_test, y_test, df.loc[test_mask, "cell_id"].values, window)

            if len(X_train_seq) < 10 or len(X_test_seq) < 5:
                continue

            split = int(len(X_train_seq) * 0.8)
            X_tr, y_tr = X_train_seq[:split], y_train_seq[:split]
            X_val_inner, y_val_inner = X_train_seq[split:], y_train_seq[split:]

            mdl = model_cls(
                input_shape=(X_tr.shape[1], X_tr.shape[2]),
                n_horizons=len(horizon_cols),
                config=model_config)
            mdl.fit(X_tr, y_tr, X_val_inner, y_val_inner)
            y_val_pred = mdl.predict_proba(X_val_inner)
            y_pred = mdl.predict_proba(X_test_seq)
            y_test = y_test_seq

        fitted_models.append(mdl)

        # Calibrate on VALIDATION set, apply to TEST set
        if calibrator is not None:
            cal = calibrator.__class__(method=calibrator.method)
            cal.fit(y_val_pred, y_val_inner, horizons=horizon_cols)
            y_cal = cal.transform(y_pred, horizons=horizon_cols)
        else:
            y_cal = y_pred

        all_preds.append(y_pred)
        all_targets.append(y_test)
        all_calibrated.append(y_cal)

        # Per-fold metrics
        per_fold_raw_metrics.append(
            compute_metrics(y_test, y_pred, horizons=horizons))
        per_fold_cal_metrics.append(
            compute_metrics(y_test, y_cal, horizons=horizons))

    if not all_preds:
        return {"predictions": np.array([]), "targets": np.array([]),
                "calibrated": np.array([]), "models": []}

    return {
        "predictions": np.vstack(all_preds),
        "targets": np.vstack(all_targets),
        "calibrated": np.vstack(all_calibrated),
        "models": fitted_models,
        "per_fold_raw": per_fold_raw_metrics,
        "per_fold_cal": per_fold_cal_metrics,
    }


def _build_sequences(features, targets, cell_ids, window):
    cells = np.unique(cell_ids)
    X_seq, y_seq = [], []
    for c in cells:
        mask = cell_ids == c
        f = features[mask]
        t = targets[mask]
        for i in range(len(f) - window + 1):
            X_seq.append(f[i:i + window])
            y_seq.append(t[i + window - 1])
    return (np.array(X_seq, dtype=np.float32),
            np.array(y_seq, dtype=np.float32))
