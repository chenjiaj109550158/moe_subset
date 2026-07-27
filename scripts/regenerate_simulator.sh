#!/usr/bin/env bash
set -euo pipefail
pseudoroute simulate-offload --config configs/experiment/simulate_offload_tiny.yaml --output-dir "${1:-artifacts/paper/simulator}"
