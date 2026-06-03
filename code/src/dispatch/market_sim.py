import numpy as np
import pandas as pd


class MarketSimulator:
    """Lightweight flexibility market simulation.

    Models a day-ahead energy market with:
      - Stochastic prices (AR(1) process)
      - Sequential service requests
      - Battery reliability check before dispatch
      - Revenue tracking with failure penalties
    """

    def __init__(self, price_mean=50.0, price_std=15.0,
                 price_ar_coeff=0.7, service_energy_kwh=0.5,
                 penalty_cost=500.0, seed=42):
        self.price_mean = price_mean
        self.price_std = price_std
        self.price_ar = price_ar_coeff
        self.service_energy = service_energy_kwh
        self.penalty_cost = penalty_cost
        self.rng = np.random.default_rng(seed)

    def run(self, P_cal, eol_cycle, horizon=20, dispatch_policy=None,
            tau=0.20):
        """Run a single market simulation trajectory.

        Parameters
        ----------
        P_cal : ndarray (n_cycles,) — calibrated failure probabilities
        eol_cycle : int — cycle at which battery actually fails
        dispatch_policy : callable with signature (P, E) -> E_offered
        tau : float — risk threshold if no policy given

        Returns
        -------
        dict with keys: revenue, failures, energy_delivered, n_accepted
        """
        n = len(P_cal)
        prices = self._ar1_prices(n)
        revenue = 0.0
        failures = 0
        energy_delivered = 0.0
        n_accepted = 0

        for t in range(n):
            price = prices[t]
            E_req = self.service_energy

            if dispatch_policy is not None:
                E_offer, _ = dispatch_policy.decide(P_cal[t], E_req)
            else:
                E_offer = E_req if P_cal[t] <= tau else 0.0

            if E_offer > 0:
                n_accepted += 1
                # Failure occurs if we dispatch and the battery fails
                # within the horizon
                actual_failure = (t < eol_cycle <= t + horizon
                                  if pd.notna(eol_cycle) else False)
                if actual_failure:
                    failures += 1
                    revenue += E_offer * (price / 1000.0) - self.penalty_cost
                else:
                    revenue += E_offer * (price / 1000.0)
                energy_delivered += E_offer

        return {
            "revenue": float(revenue),
            "failures": int(failures),
            "energy_delivered": float(energy_delivered),
            "n_accepted": int(n_accepted),
            "failure_rate": float(failures / n_accepted) if n_accepted > 0 else 0.0,
        }

    def monte_carlo(self, P_cal, eol_cycle, horizon=20,
                    dispatch_policy=None, tau=0.20, n_scenarios=1000):
        """Run Monte Carlo simulation across price scenarios.

        Returns DataFrame of results.
        """
        records = []
        for s in range(n_scenarios):
            self.rng = np.random.default_rng(42 + s)
            result = self.run(P_cal, eol_cycle, horizon,
                              dispatch_policy, tau)
            result["scenario"] = s
            records.append(result)
        return pd.DataFrame(records)

    def _ar1_prices(self, n):
        eps = self.rng.normal(0, self.price_std, n)
        prices = np.zeros(n)
        prices[0] = self.price_mean + eps[0]
        for t in range(1, n):
            prices[t] = (self.price_mean
                         + self.price_ar * (prices[t - 1] - self.price_mean)
                         + eps[t])
        return np.maximum(prices, 0.0)
