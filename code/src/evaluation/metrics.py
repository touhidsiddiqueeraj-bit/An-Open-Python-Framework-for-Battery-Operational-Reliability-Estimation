import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def compute_metrics(y_true, y_pred, horizons=None):
    """Compute discrimination and calibration metrics per horizon.

    Parameters
    ----------
    y_true : ndarray (n_samples, n_horizons) or (n_samples,)
    y_pred : ndarray (n_samples, n_horizons) or (n_samples,)
    horizons : list of horizon labels (optional)

    Returns
    -------
    dict with keys 'auc', 'brier', 'ece' (per horizon and macro avg)
    """
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    n_horizons = y_true.shape[1]
    h_labels = horizons or list(range(n_horizons))
    results = {"per_horizon": {}}

    aucs, briers, eces = [], [], []
    for i, h in enumerate(h_labels):
        yt, yp = y_true[:, i], y_pred[:, i]
        n_pos = yt.sum()
        if n_pos > 0 and n_pos < len(yt):
            auc = roc_auc_score(yt, yp)
        else:
            auc = float("nan")
        brier = brier_score_loss(yt, yp)
        ece = _expected_calibration_error(yt, yp, n_bins=10)

        results["per_horizon"][h] = {
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "brier": round(brier, 4),
            "ece": round(ece, 4),
        }
        aucs.append(auc)
        briers.append(brier)
        eces.append(ece)

    results["macro_avg"] = {
        "auc": round(np.nanmean(aucs), 4),
        "brier": round(np.mean(briers), 4),
        "ece": round(np.mean(eces), 4),
    }
    return results


def compute_operational_metrics(energy_offered, failures_actual,
                                energy_requested=None):
    """Compute operational performance metrics."""
    total_offered = np.sum(energy_offered)
    total_failures = np.sum(failures_actual) if failures_actual is not None else 0
    n_dispatch = np.sum(energy_offered > 0) if isinstance(energy_offered, np.ndarray) else 1
    failure_rate = total_failures / n_dispatch if n_dispatch > 0 else 0.0
    return {
        "delivered_energy": round(total_offered, 2),
        "failure_rate": round(failure_rate, 4),
        "n_dispatch": int(n_dispatch),
    }


def _expected_calibration_error(y_true, y_pred, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[1:-1])  # 0..n_bins-1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)
