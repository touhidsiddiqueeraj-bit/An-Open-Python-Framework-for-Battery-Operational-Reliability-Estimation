#!/bin/bash
# ─────────────────────────────────────────────────────────
# Local training launcher for the extension paper
# ─────────────────────────────────────────────────────────
# Usage:
#   ./local_train.sh           Quick mode (XGBoost only)
#   ./local_train.sh --full    Full mode (includes DL models)
#   ./local_train.sh --expt baseline   Single experiment
#   ./local_train.sh --list    List experiments
# ─────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " Battery Hazard Extension — Local Training"
echo "========================================"
echo "Date: $(date)"
echo "Mode: ${1:-quick}"
echo ""

# Check Python
PYTHON="python3"
command -v $PYTHON >/dev/null 2>&1 || { echo "Error: python3 not found"; exit 1; }

# Create data directories
mkdir -p data/{raw,processed}

# Run experiments
exec $PYTHON experiments/run_all.py "$@"
