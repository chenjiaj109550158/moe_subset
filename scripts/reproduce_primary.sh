#!/usr/bin/env bash
set -euo pipefail
pseudoroute reproduce --suite primary --config-root configs/paper --output-dir "${1:-artifacts/reproduction/primary}"
