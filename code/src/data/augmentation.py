import numpy as np
import pandas as pd


class OperationalAugmenter:
    """Augment laboratory cycling data with field-like operational variability.

    Introduces:
      - Partial cycles (random SoC windows)
      - Random rest periods between cycles
      - Temperature perturbations
      - Measurement noise

    Uses an empirical capacity-fade model to assign post-augmentation
    health labels so that the resulting trajectory is physically plausible.
    """

    def __init__(self, seed=42, partial_cycle_prob=0.3,
                 rest_prob=0.2, temp_noise_std=1.5, capacity_noise_std=0.005):
        self.rng = np.random.default_rng(seed)
        self.partial_cycle_prob = partial_cycle_prob
        self.rest_prob = rest_prob
        self.temp_noise_std = temp_noise_std
        self.capacity_noise_std = capacity_noise_std

    def augment(self, df, n_virtual_cells=3):
        """Generate augmented versions of each degradation trajectory."""
        records = []
        for cell_id, group in df.groupby("cell_id"):
            group = group.sort_values("cycle").reset_index(drop=True)
            for vc in range(n_virtual_cells):
                aug = self._augment_single(group, f"{cell_id}_aug{vc}")
                records.append(aug)
        if records:
            result = pd.concat(records, ignore_index=True)
            result["augmented"] = True
            original = df.copy()
            original["augmented"] = False
            return pd.concat([original, result], ignore_index=True)
        return df

    def _augment_single(self, group, new_id):
        aug = group.copy()
        aug["cell_id"] = new_id
        n = len(aug)

        # Partial cycling: insert capacity recovery for some cycles
        for i in range(1, n):
            if self.rng.random() < self.partial_cycle_prob:
                factor = self.rng.uniform(0.6, 0.95)
                aug.loc[i, "capacity"] *= factor
                aug.loc[i, "soh"] = (aug.loc[i, "capacity"]
                                     / group["capacity"].iloc[:5].mean())

        # Rest periods: occasionally keep capacity constant
        for i in range(1, n):
            if self.rng.random() < self.rest_prob:
                if i < n - 1:
                    aug.loc[i, "capacity"] = aug.loc[i - 1, "capacity"]
                    aug.loc[i, "soh"] = aug.loc[i - 1, "soh"]

        # Temperature and measurement noise
        if "temperature_avg" in aug.columns:
            aug["temperature_avg"] += self.rng.normal(
                0, self.temp_noise_std, size=n)
        aug["capacity"] += self.rng.normal(
            0, self.capacity_noise_std, size=n)
        aug["soh"] = aug["capacity"] / group["capacity"].iloc[:5].mean()

        # Recompute differentials
        aug["d_soh"] = aug["soh"].diff().fillna(0)
        aug["d_capacity"] = aug["capacity"].diff().fillna(0)

        return aug
