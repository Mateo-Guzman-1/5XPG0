#!/bin/bash
# setup_venv.sh — create a local virtual environment and install the training
# dependencies. Run it once, then `source .venv/bin/activate`.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "done. activate with:  source .venv/bin/activate"
