#!/usr/bin/env bash
set -euo pipefail
pseudoroute oracle-sweep --config configs/experiment/oracle_tiny.yaml --output-dir "${1:-artifacts/paper/oracle}"
