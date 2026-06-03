import os, re, zipfile, io, requests
import numpy as np
import pandas as pd
from tqdm import tqdm

CALCE_BASE = "https://web.calce.umd.edu/batteries/data/"

# Identified CALCE datasets by chemistry group
CALCE_FILES = {
    "LCO": [f"CS2_{i}.zip" for i in [3, 5, 6, 7, 8, 9, 21, 24, 33, 34, 35, 36, 37, 38]],
    "LiCoO2": [f"CX2_{i}.zip" for i in [3, 4, 8, 16, 31, 32, 33, 34, 35, 36, 37, 38]],
    "LFP": ([f"A123_{i}.zip" for i in [3, 5]] +
            [f"A123_094.zip"]),
    "K2": ["K2_016.zip", "K2_039.zip"],
}


class CALCELoader:
    """Download and process CALCE battery aging datasets."""

    def __init__(self, data_dir, chemistries=None, redownload=False):
        self.data_dir = os.path.abspath(data_dir)
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.redownload = redownload
        if chemistries is None:
            chemistries = list(CALCE_FILES.keys())
        self.chemistries = chemistries

    def load_all(self):
        records = []
        for chem in self.chemistries:
            files = CALCE_FILES.get(chem, [])
            for fname in files:
                url = CALCE_BASE + fname
                zpath = os.path.join(self.raw_dir, fname)
                if not os.path.exists(zpath) or self.redownload:
                    self._download(url, zpath)
                records.append(self._parse_zip(zpath, chem))
        combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        return self._engineer_features(combined)

    def list_available(self):
        for chem, files in CALCE_FILES.items():
            print(f"{chem}: {len(files)} files  ({', '.join(files[:5])} ...)")

    # ── private ──────────────────────────────────────────

    def _download(self, url, path):
        print(f"Downloading {os.path.basename(path)} ...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f:
            pbar = tqdm(total=total, unit="B", unit_scale=True,
                        desc=os.path.basename(path))
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

    def _parse_zip(self, zpath, chemistry):
        """Parse a single CALCE zip into a DataFrame."""
        rows = []
        cell_id = os.path.splitext(os.path.basename(zpath))[0]
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                txt_files = [f for f in z.namelist()
                             if f.endswith(".txt") or f.endswith(".csv")]
                for tf in txt_files:
                    try:
                        content = z.read(tf).decode("utf-8", errors="replace")
                        df = self._parse_text(content, tf)
                        if df is not None and len(df) > 0:
                            df["cell_id"] = f"{cell_id}_{chemistry}"
                            df["chemistry"] = chemistry
                            rows.append(df)
                    except Exception:
                        continue
        except (zipfile.BadZipFile, Exception):
            pass
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _parse_text(self, content, fname):
        """Try to parse CALCE text files which vary in format."""
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) < 2:
            return None

        # Detect delimiters and headers
        header = lines[0].lower()
        if "cycle" in header or "time" in header:
            delim = "\t" if "\t" in lines[1] else ","
            data_lines = lines[1:]
        elif "cycle" in header.lower():
            delim = "\t" if "\t" in lines[1] else ","
            data_lines = lines[1:]
        else:
            return None

        parsed = []
        for line in data_lines:
            parts = line.split(delim)
            try:
                nums = [float(p) for p in parts]
                parsed.append(nums)
            except ValueError:
                continue
        if not parsed:
            return None

        arr = np.array(parsed)
        cols = [f"col{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        df.columns = self._infer_column_names(header, arr.shape[1])
        if "time" in df.columns:
            df["cycle"] = df.groupby("cell_id").cumcount() + 1 if "cell_id" in df.columns else range(1, len(df) + 1)
        if "capacity" not in df.columns and "col0" in df.columns:
            df["capacity"] = df["col0"]
            df = df.drop(columns=["col0"])
        return df

    def _infer_column_names(self, header, ncols):
        known = {
            "time": ["time", "t", "timestamp", "seconds"],
            "voltage_avg": ["voltage", "volt", "v", "potential"],
            "current_avg": ["current", "i", "amp"],
            "temperature_avg": ["temperature", "temp", "t", "celsius"],
            "capacity": ["capacity", "cap", "ah", "c", "discharge capacity"],
        }
        cols = []
        h = header.lower()
        for i in range(ncols):
            matched = False
            for key, aliases in known.items():
                if any(a in h.split() for a in aliases):
                    if key not in cols:
                        cols.append(key)
                        matched = True
                        break
            if not matched:
                cols.append(f"col{i}")
        return cols

    def _engineer_features(self, df):
        if df.empty:
            return df
        df = df.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
        bol = df.groupby("cell_id")["capacity"].transform(
            lambda x: x.iloc[:5].mean() if len(x) > 5 else x.mean())
        df["soh"] = df["capacity"] / bol
        df["d_soh"] = df.groupby("cell_id")["soh"].diff().fillna(0)
        df["d_capacity"] = df.groupby("cell_id")["capacity"].diff().fillna(0)
        for col in ["voltage_avg", "current_avg", "temperature_avg"]:
            if col in df.columns:
                df[f"{col}_ma3"] = df.groupby("cell_id")[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean())
        return df
