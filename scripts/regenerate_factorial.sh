#!/usr/bin/env bash
set -euo pipefail
pseudoroute dapq-factorial --config configs/experiment/dapq_factorial_tiny.yaml --output-dir "${1:-artifacts/paper/factorial}"
