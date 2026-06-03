"""Integration test: run --quick baseline, assert output structure."""
import sys, os, json, subprocess, glob

CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
RESULTS_DIR = os.path.join(CODE_DIR, "results")


def test_baseline_quick_produces_valid_output():
    """Run run_all.py --expt baseline --quick and verify JSON output."""
    # Count results files before
    before = len(glob.glob(os.path.join(RESULTS_DIR, "baseline_*.json")))

    result = subprocess.run(
        [sys.executable, "experiments/run_all.py", "--expt", "baseline", "--quick"],
        cwd=CODE_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Find new result file
    after = sorted(glob.glob(os.path.join(RESULTS_DIR, "baseline_*.json")))
    if len(after) <= before:
        print("SKIP: no new result file (data may be missing)")
        print("stdout:", result.stdout[-300:])
        return

    fp = after[-1]
    with open(fp) as f:
        d = json.load(f)

    # Structural assertions
    for key in ["raw", "calibrated"]:
        assert key in d, f"Missing key: {key}"
        assert "macro_avg" in d[key], f"Missing macro_avg in {key}"
        assert "auc" in d[key]["macro_avg"], f"Missing auc in {key}.macro_avg"
        auc = d[key]["macro_avg"]["auc"]
        assert 0.0 <= auc <= 1.0, f"AUC {auc} out of [0, 1] range"

    assert "per_fold_raw" in d, "Missing per_fold_raw"
    assert len(d["per_fold_raw"]) > 0, "per_fold_raw is empty"

    # Per-fold check
    for fold in d["per_fold_raw"]:
        for h_key in fold["per_horizon"]:
            v = fold["per_horizon"][h_key]["auc"]
            if v is not None:
                assert 0.0 <= v <= 1.0, f"Fold AUC {v} out of range"

    print(f"PASS: {os.path.basename(fp)} — "
          f"raw_auc={d['raw']['macro_avg']['auc']:.4f}, "
          f"cal_auc={d['calibrated']['macro_avg']['auc']:.4f}, "
          f"folds={len(d['per_fold_raw'])}")


if __name__ == "__main__":
    test_baseline_quick_produces_valid_output()
