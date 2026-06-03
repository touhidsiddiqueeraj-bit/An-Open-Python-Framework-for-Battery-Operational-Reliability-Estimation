import numpy as np
import pandas as pd


class CompositeFailureLabeler:
    """Generate multi-criteria failure labels.

    Instead of a single SOH < 70 % threshold, this labeler marks
    operational failure based on multiple indicators:

      1. SOH drops below threshold
      2. Internal resistance rise exceeds threshold (if available)
      3. Sudden single-cycle capacity drop exceeds threshold
      4. Incomplete discharge (future: requires voltage info)
    """

    def __init__(self, soh_threshold=0.70, resistance_pct=100,
                 sudden_drop=0.05):
        self.soh_threshold = soh_threshold
        self.resistance_pct = resistance_pct
        self.sudden_drop = sudden_drop

    def label(self, df, method="multi"):
        """Add `eol_cycle`, `failure_event` and horizon labels.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: cell_id, cycle, soh, d_capacity
        method : str
            'single' — only SOH < threshold (original paper)
            'multi'  — composite of all criteria (extension)
        """
        df = df.copy()
        eol_cycles = {}

        for cid, group in df.groupby("cell_id"):
            group = group.sort_values("cycle")
            eol = None
            for _, row in group.iterrows():
                failed = False
                if row["soh"] < self.soh_threshold:
                    failed = True
                if method == "multi":
                    if row.get("d_capacity", 0) < -self.sudden_drop:
                        failed = True
                    bol_resist = group.iloc[:5].get("resistance", pd.Series([np.nan]))
                    if "resistance" in group.columns and not bol_resist.isna().all():
                        if row["resistance"] > bol_resist.mean() * (1 + self.resistance_pct / 100):
                            failed = True
                if failed:
                    eol = row["cycle"]
                    break
            eol_cycles[cid] = eol

        df["eol_cycle"] = df["cell_id"].map(eol_cycles)

        for H in [10, 20, 30, 50]:
            df[f"fail_{H}"] = df.apply(
                lambda r: 1 if (pd.notna(r["eol_cycle"])
                                and r["cycle"] < r["eol_cycle"]
                                and r["cycle"] + H >= r["eol_cycle"])
                else 0, axis=1)

        return df
