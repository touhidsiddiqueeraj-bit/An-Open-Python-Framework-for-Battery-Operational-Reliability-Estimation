import numpy as np


class ThresholdPolicy:
    """Binary accept/deny dispatch based on risk threshold.

    Accept service request if calibrated failure probability ≤ τ.
    This is the policy used in the original paper.
    """

    def __init__(self, tau=0.20):
        self.tau = tau

    def decide(self, P_cal, E_requested):
        """Return offered energy for each sample.

        Parameters
        ----------
        P_cal : float or ndarray of shape (n_samples,) or (n_samples, n_horizons)
        E_requested : float or ndarray

        Returns
        -------
        E_offered : float or ndarray — either E_requested or 0
        decisions : bool or ndarray — boolean
        """
        scalar_input = np.isscalar(P_cal)
        P = np.asarray(P_cal, dtype=float)
        if P.ndim == 2:
            P = P[:, 0]
        decisions = P <= self.tau
        E_offered = np.where(decisions, E_requested, 0.0)
        if scalar_input:
            return float(E_offered), bool(decisions)
        return E_offered, decisions

    def sweep(self, P_cal, E_requested, taus):
        """Evaluate multiple thresholds at once.

        Returns dict {tau: (E_offered_total, failure_rate, n_accepted)}
        """
        results = {}
        for tau in taus:
            self.tau = tau
            offered, dec = self.decide(P_cal, E_requested)
            results[tau] = {
                "energy": offered.sum(),
                "n_accepted": dec.sum(),
                "accept_rate": dec.mean(),
            }
        return results

    @property
    def name(self):
        return f"Threshold(τ={self.tau})"
