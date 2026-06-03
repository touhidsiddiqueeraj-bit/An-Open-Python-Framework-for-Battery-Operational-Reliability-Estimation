"""Generate synthetic battery degradation data mimicking NASA PCoE dataset.

Creates realistic capacity fade trajectories with:
  - Exponential + linear degradation
  - Cycle-to-cycle noise
  - Cell-to-cell variability
  - Voltage, current, temperature measurements
  - Multiple cells with different EOL points
"""

import numpy as np
import pandas as pd


def generate_synthetic_nasa(n_cells=4, n_cycles=300, seed=42):
    """Generate synthetic battery data mimicking NASA PCoE degradation.

    Supports arbitrary n_cells by generating unique cell IDs and scaling
    degradation parameters across the population so that smaller datasets
    still produce realistic variance while larger datasets add diversity.

    Returns a DataFrame with the same columns as NASALoader output.
    """
    rng = np.random.default_rng(seed)
    records = []

    for idx in range(n_cells):
        cell_id = f"BAT-SYN-{idx+1:03d}"
        base_capacity = 2.0 + rng.uniform(-0.1, 0.1)

        # Scale fade rate across cells so small-N still has variance
        # and large-N adds more extreme examples
        fade_base = 0.001 + 0.003 * (idx / max(n_cells - 1, 1))
        noise_scale = 0.008

        # EOL (SOH<0.70) occurs at different cycles spread across range
        min_eol = 80
        max_eol = 220
        eol_at = int(min_eol + (max_eol - min_eol) * (idx / max(n_cells - 1, 1)))

        for t in range(1, n_cycles + 1):
            if t < eol_at - 25:
                capacity = base_capacity * (1.0 - fade_base * t)
            else:
                progress = (t - (eol_at - 25)) / 25
                cap_before = base_capacity * (1.0 - fade_base * (eol_at - 25))
                capacity = cap_before * (1.0 - 0.20 * progress ** 2)
            capacity += rng.normal(0, noise_scale)
            capacity = max(capacity, 0.0)

            voltage = 3.7 + rng.normal(0, 0.05)
            current = -2.0 + rng.normal(0, 0.1)
            temperature = 25.0 + rng.normal(0, 2.0) + 0.02 * t
            measurement_time = t * 1.5 + rng.normal(0, 0.1)

            records.append({
                "cell_id": cell_id,
                "cycle": t,
                "voltage_avg": voltage,
                "current_avg": current,
                "temperature_avg": temperature,
                "capacity": capacity,
                "time": measurement_time,
                "dataset": "synthetic_nasa",
            })

    df = pd.DataFrame(records)

    # Engineer features (same as NASALoader._engineer_features)
    df = df.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    bol = df.groupby("cell_id")["capacity"].transform(
        lambda x: x.iloc[:5].mean() if len(x) >= 5 else x.mean())
    df["soh"] = df["capacity"] / bol
    df["d_soh"] = df.groupby("cell_id")["soh"].diff().fillna(0)
    df["d_capacity"] = df.groupby("cell_id")["capacity"].diff().fillna(0)
    for col in ["voltage_avg", "current_avg", "temperature_avg"]:
        df[f"{col}_ma3"] = df.groupby("cell_id")[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

    return df


def generate_synthetic_calce(chemistry="LCO", n_cells=3, n_cycles=250, seed=99):
    """Generate synthetic CALCE-like data for a different chemistry."""
    rng = np.random.default_rng(seed)
    records = []

    prefixes = {"LCO": "CS2", "LFP": "A123", "K2": "K2"}
    chem_fade = {"LCO": 0.003, "LFP": 0.0015, "K2": 0.0025}
    prefix = prefixes.get(chemistry, "GEN")
    fade = chem_fade.get(chemistry, 0.002)

    for i in range(n_cells):
        cell_id = f"{prefix}_{i+1}_{chemistry}"
        base_cap = 1.5 + rng.uniform(-0.1, 0.1)

        for t in range(1, n_cycles + 1):
            capacity = base_cap * (1.0 - fade * t / 8)
            if t > 180 + i * 15:
                capacity -= 0.1 * (t - 180 - i * 15) / 40
            capacity += rng.normal(0, 0.012)
            capacity = max(capacity, 0.0)

            records.append({
                "cell_id": cell_id,
                "cycle": t,
                "voltage_avg": 3.3 + rng.normal(0, 0.05),
                "current_avg": -1.5 + rng.normal(0, 0.1),
                "temperature_avg": 28.0 + rng.normal(0, 2.0),
                "capacity": capacity,
                "time": t * 2.0,
                "chemistry": chemistry,
                "dataset": "synthetic_calce",
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    bol = df.groupby("cell_id")["capacity"].transform(
        lambda x: x.iloc[:5].mean() if len(x) >= 5 else x.mean())
    df["soh"] = df["capacity"] / bol
    df["d_soh"] = df.groupby("cell_id")["soh"].diff().fillna(0)
    df["d_capacity"] = df.groupby("cell_id")["capacity"].diff().fillna(0)
    return df
