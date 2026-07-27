#!/usr/bin/env bash
set -euo pipefail
pseudoroute inspect-model --config configs/model/tiny_moe.yaml
pytest

