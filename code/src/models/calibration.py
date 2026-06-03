import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Calibrate raw model probabilities into consistent risk estimates.

    Uses isotonic regression (monotonic non-parametric mapping) as in
    the original paper. For horizons where isotonic overfits, falls
    back to Platt scaling (sigmoid calibration).
    """

    def __init__(self, method="isotonic"):
        self.method = method
        self.calibrators = {}

    def fit(self, y_pred, y_true, horizons=None):
        """Learn calibration mapping for each horizon.

        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_horizons)
            Raw predicted probabilities.
        y_true : ndarray of shape (n_samples, n_horizons)
            Ground truth binary labels.
        horizons : list, optional
            Horizon labels for storage.
        """
        n_horizons = y_pred.shape[1]
        for i in range(n_horizons):
            h_label = horizons[i] if horizons else i
            if self.method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
                iso.fit(y_pred[:, i], y_true[:, i])
                self.calibrators[h_label] = iso
            else:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1.0, class_weight="balanced")
                lr.fit(y_pred[:, i:i+1], y_true[:, i])
                self.calibrators[h_label] = lr

    def transform(self, y_pred, horizons=None):
        """Apply calibration.

        Returns calibrated probabilities of shape (n_samples, n_horizons).
        """
        n_horizons = y_pred.shape[1]
        calibrated = np.zeros_like(y_pred)
        for i in range(n_horizons):
            h_label = horizons[i] if horizons else i
            cal = self.calibrators[h_label]
            if self.method == "isotonic":
                calibrated[:, i] = cal.predict(y_pred[:, i])
            else:
                calibrated[:, i] = cal.predict_proba(y_pred[:, i:i+1])[:, 1]
        return np.clip(calibrated, 0.0, 1.0)

    def fit_transform(self, y_pred, y_true, horizons=None):
        self.fit(y_pred, y_true, horizons)
        return self.transform(y_pred, horizons)
