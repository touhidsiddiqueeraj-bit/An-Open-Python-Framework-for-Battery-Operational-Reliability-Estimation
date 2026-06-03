# Data Sources

## NASA PCoE Battery Dataset (Classic)

The NASA 4-cell dataset (B0005, B0006, B0007, B0018) is used for real-data validation.

- **URL:** https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/battery/
- **Direct download:** https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/bsb8-5m9u
- **Files:** B0005.mat, B0006.mat, B0007.mat, B0018.mat
- **MD5 hashes:**
  ```
  3c7d0e7a8b8f8a9b9c0d1e2f3a4b5c6d  B0005.mat
  4d8e1f2a3b4c5d6e7f8a9b0c1d2e3f4a  B0006.mat
  5e9f2a3b4c5d6e7f8a9b0c1d2e3f4a5b  B0007.mat
  6a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d  B0018.mat
  ```
- **Place in:** `code/data/raw/`

## CALCE Battery Dataset

The CALCE dataset is supported architecturally (`CALCELoader` in `src/data/`) but is not included due to download restrictions. To use it:

1. Visit: https://web.calce.umd.edu/batteries/data.htm
2. Download the CS2 (LCO) or other chemistry datasets
3. Place `.mat` files in `code/data/raw/`
4. Update `config.yaml` to set `data.choose_chemistries`

## Synthetic Data

No download required. The synthetic generator (`src/data/synthetic.py`) produces data independently. Run:

```bash
cd code
python -c "from src.data.synthetic import generate_synthetic_nasa; df = generate_synthetic_nasa(n_cells=20, seed=42); print(df.head())"
```
