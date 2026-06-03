import numpy as np
import xgboost as xgb


class XGBoostHazard:
    """Multihorizon hazard model using gradient boosted trees.

    Trains one XGBClassifier per prediction horizon with early stopping
    on a held-out validation set.
    """

    def __init__(self, config=None):
        c = config or {}
        self.params = {
            "n_estimators": c.get("n_estimators", 300),
            "max_depth": c.get("max_depth", 4),
            "learning_rate": c.get("learning_rate", 0.05),
            "subsample": c.get("subsample", 0.8),
            "colsample_bytree": c.get("colsample_bytree", 0.8),
            "min_child_weight": c.get("min_child_weight", 5),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": 42,
        }
        self.early_stopping = c.get("early_stopping_rounds", 20)
        self.models_ = []
        self.n_horizons_ = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train one model per horizon with early stopping."""
        self.n_horizons_ = y_train.shape[1]
        self.models_ = []

        if X_val is not None and y_val is not None:
            for i in range(self.n_horizons_):
                est = xgb.XGBClassifier(**self.params,
                                        early_stopping_rounds=self.early_stopping)
                est.fit(X_train, y_train[:, i],
                        eval_set=[(X_val, y_val[:, i])],
                        verbose=False)
                self.models_.append(est)
        else:
            for i in range(self.n_horizons_):
                est = xgb.XGBClassifier(**self.params)
                est.fit(X_train, y_train[:, i])
                self.models_.append(est)
        return self

    def predict_proba(self, X):
        """Return failure probabilities for each horizon.

        Returns shape (n_samples, n_horizons).
        """
        if not self.models_:
            raise RuntimeError("Model not fitted.")
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        return probs

    @property
    def name(self):
        return "XGBoost"
