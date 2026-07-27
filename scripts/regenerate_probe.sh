#!/usr/bin/env bash
set -euo pipefail
root="${1:-artifacts/paper}"
pseudoroute train-predictor --config configs/experiment/train_predictor_tiny.yaml --output-dir "$root/train_predictor"
python - "$root" <<'PY'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1]).resolve()
config = yaml.safe_load(Path("configs/experiment/evaluate_probe_tiny.yaml").read_text())
config["predictor_dir"] = str(root / "train_predictor")
Path("/tmp/pseudoroute_probe_reproduction.yaml").write_text(yaml.safe_dump(config))
PY
pseudoroute evaluate-probe --config /tmp/pseudoroute_probe_reproduction.yaml --output-dir "$root/probe"
