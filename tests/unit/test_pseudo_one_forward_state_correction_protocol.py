from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_one_forward_state_correction_v1.yaml")
SAMPLES = Path("configs/analysis/pseudo_one_forward_state_correction_v1_samples.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_forward_state_correction_protocol_is_frozen_and_calibration_free() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    assert config["analysis_id"] == samples["analysis_id"]
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["anchor_correction"]["coefficients"] == list(range(1, 9))
    assert len(samples["partitions"]["development"]) == 4
    assert samples["selection"]["new_rows_added"] is False
    forbidden = set(config["information_boundary"]["forbidden"])
    assert {"future_true_tokens_after_sampled_next", "learned_or_fitted_parameters"} <= forbidden
    assert config["analysis"]["actual_accuracy_authorized"] is False


def test_one_forward_state_correction_fingerprints_are_deterministic() -> None:
    assert _sha256(CONFIG) == "24f2f6e74f5a01ae589acd7a22dc084f7f9c0117181d964e343b41ed6ac93925"
    assert _sha256(SAMPLES) == "350cc9ddcc003daa6973d247434bc600ecfc108ad7b9773f5e0cb997936ec80c"
