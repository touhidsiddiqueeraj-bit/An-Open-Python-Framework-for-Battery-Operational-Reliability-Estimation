import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier


class XGBoostHazard:
    """Multihorizon hazard model using gradient boosted trees.

    Trains one multi-output classifier that jointly predicts failure
    probability for all horizons H ∈ {10, 20, 30, 50}.
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
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model.

        y_train : array-like of shape (n_samples, n_horizons)
        """
        self.model = MultiOutputClassifier(
            xgb.XGBClassifier(**self.params), n_jobs=1)
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val[:, i]) for i in range(y_val.shape[1])]
            # MultiOutputClassifier doesn't support eval_set natively;
            # we train per-horizon with early stopping instead.
            self.model.estimators_ = []
            for i in range(y_train.shape[1]):
                est = xgb.XGBClassifier(**self.params,
                                        early_stopping_rounds=self.early_stopping)
                est.fit(X_train, y_train[:, i],
                        eval_set=[(X_val, y_val[:, i])],
                        verbose=False)
                self.model.estimators_.append(est)
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return failure probabilities for each horizon.

        Returns shape (n_samples, n_horizons).
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        # MultiOutputClassifier returns a list of (n, 2) arrays
        probs = self.model.predict_proba(X)
        return np.column_stack([p[:, 1] for p in probs])

    @property
    def name(self):
        return "XGBoost"
