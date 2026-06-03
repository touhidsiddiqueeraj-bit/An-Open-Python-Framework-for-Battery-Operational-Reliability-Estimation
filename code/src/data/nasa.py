import os, glob, re, zipfile, io
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

EXPECTED_FILES = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]
NASA_DOWNLOAD_URL = (
    "https://ti.arc.nasa.gov/tech/dash/groups/pcoe/"
    "prognostic-data-repository/battery/"
)


class NASALoader:
    """Load the NASA PCoE battery aging dataset from local .mat files.

    The four classic cells (B0005, B0006, B0007, B0018) are loaded
    from locally-downloaded .mat files.  Each file contains a struct
    keyed by the battery ID (e.g. 'B0005') with a 'cycle' field.
    """
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def check_available(self):
        found = []
        for fname in EXPECTED_FILES:
            if os.path.exists(os.path.join(self.raw_dir, fname)):
                found.append(fname)
        return found

    def print_download_instructions(self):
        print("=" * 60)
        print("NASA CLASSIC DATASET — Download")
        print("=" * 60)
        for f in EXPECTED_FILES:
            path = os.path.join(self.raw_dir, f)
            status = "✓" if os.path.exists(path) else "✗ missing"
            print(f"  {status}  {f}")
        if not self.check_available():
            print("\nPlace .mat files in:", self.raw_dir)
        print("=" * 60)

    def load_classic(self):
        found = self.check_available()
        if not found:
            self.print_download_instructions()
            return pd.DataFrame()

        records = []
        for fname in tqdm(found, desc="Loading NASA .mat files"):
            path = os.path.join(self.raw_dir, fname)
            cell_id = os.path.splitext(fname)[0]
            try:
                raw = loadmat(path)
                # Key in file matches the battery ID (e.g. 'B0005')
                file_key = [k for k in raw if not k.startswith("__")][0]
                cycles = raw[file_key][0, 0]["cycle"].ravel()
                data = self._parse_cycles(cycles)
                if not data:
                    continue
                df = pd.DataFrame(data)
                df["cell_id"] = cell_id
                df["dataset"] = "nasa_classic"
                records.append(df)
            except Exception as e:
                print(f"  Error loading {fname}: {e}")

        if not records:
            return pd.DataFrame()
        combined = pd.concat(records, ignore_index=True)
        return self._engineer_features(combined)

    def _parse_cycles(self, cycles):
        out = {"cycle": [], "voltage_avg": [], "current_avg": [],
               "temperature_avg": [], "capacity": [], "discharge_time": []}
        for i, c in enumerate(cycles):
            typ = str(c["type"].ravel()[0])
            if "discharge" not in typ.lower():
                continue
            out["cycle"].append(i)
            meas = c["data"].ravel()[0]
            out["voltage_avg"].append(float(np.mean(meas["Voltage_measured"].ravel())))
            out["current_avg"].append(float(np.mean(meas["Current_measured"].ravel())))
            out["temperature_avg"].append(float(np.mean(meas["Temperature_measured"].ravel())))
            out["capacity"].append(float(meas["Capacity"].ravel()[0]))
            time_arr = meas["Time"].ravel()
            out["discharge_time"].append(float(time_arr[-1] if len(time_arr) > 0 else 0))
        return out

    def _engineer_features(self, df):
        if df.empty:
            return df
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
