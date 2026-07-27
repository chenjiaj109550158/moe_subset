#!/usr/bin/env bash
set -euo pipefail
pseudoroute benchmark-offload --config configs/experiment/benchmark_offload_tiny.yaml --output-dir "${1:-artifacts/paper/runtime}"
