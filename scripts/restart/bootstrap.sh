#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash scripts/restart/bootstrap.sh --config configs/restart/restart_v1.yaml'
  exit 0
fi
[[ "${1:-}" == "--config" && -f "${2:-}" ]] || { echo 'Specify --config'; exit 2; }
# Use the verified Python 3.12 base without touching its packages.
PYTHON_BASE=${RESTART_PYTHON_BASE:-/opt/miniconda/bin/python}
if [[ ! -x .venv/bin/python ]]; then "$PYTHON_BASE" -m venv .venv; fi
.venv/bin/python -m pip install -c configs/restart/environment.lock.txt -e '.[dev,hf]' 'vllm==0.11.0' scipy jsonschema
.venv/bin/python -m pip check
.venv/bin/python -c 'import sys; print(sys.executable)'
