#!/usr/bin/env bash
# Reproduce the battery reliability study (CPU-only, ~6 seconds for --quick)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

echo "=== Setting up environment ==="

# Create virtual environment
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install pinned dependencies
pip install -r "${REPO_DIR}/requirements-exact.txt"

# Run the quick experiment
echo "=== Running quick experiment ==="
cd "${REPO_DIR}/code"
python experiments/run_all.py --quick

echo "=== Regenerating figures ==="
cd "${REPO_DIR}/paper"
python generate_figures.py
python render_to_docx.py

echo "=== Done ==="
echo "Outputs:"
echo "  - code/results/   (experiment results)"
echo "  - paper/figures/  (manuscript figures)"
echo "  - paper/Extension_Paper.docx"
