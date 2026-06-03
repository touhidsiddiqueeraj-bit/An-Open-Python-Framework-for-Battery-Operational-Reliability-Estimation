import numpy as np


class ContinuousDeratingPolicy:
    """Offer partial energy based on risk level.

    Instead of binary accept/deny, the offered energy decreases
    continuously as risk increases:

        E_offered = E_requested × max(0, 1 - α × P_cal)

    where α controls the aggressiveness of derating.
    """

    def __init__(self, alpha=2.0, min_offer=0.0):
        self.alpha = alpha
        self.min_offer = min_offer

    def decide(self, P_cal, E_requested):
        scalar_input = np.isscalar(P_cal)
        P = np.asarray(P_cal, dtype=float)
        if P.ndim == 2:
            P = P[:, 0]
        ratio = np.clip(1.0 - self.alpha * P, self.min_offer, 1.0)
        E_offered = ratio * E_requested
        if scalar_input:
            return float(E_offered), float(ratio)
        return E_offered, ratio

    def sweep(self, P_cal, E_requested, alphas):
        results = {}
        for alpha in alphas:
            self.alpha = alpha
            offered, _ = self.decide(P_cal, E_requested)
            results[alpha] = {"energy": offered.sum()}
        return results

    @property
    def name(self):
        return f"Derating(α={self.alpha})"
