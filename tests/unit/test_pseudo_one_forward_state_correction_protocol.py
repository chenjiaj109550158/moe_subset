from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.pseudo_one_forward_state_correction import SPECS

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
    assert _sha256(CONFIG) == "7f3c519996c0a190930ef9de4edc138200e3a659287bc171d966e09219803c44"
    assert _sha256(SAMPLES) == "350cc9ddcc003daa6973d247434bc600ecfc108ad7b9773f5e0cb997936ec80c"


def test_one_forward_state_correction_runner_matches_frozen_variants() -> None:
    assert [spec.key for spec in SPECS] == [
        "recent_sequence_causal_uncorrected",
        "recent_sequence_causal_hidden_velocity_linear_norm",
        "recent_sequence_causal_residual_velocity_linear_norm",
    ]
    assert [spec.hidden_state_correction for spec in SPECS] == [
        "none",
        "recent_linear_norm",
        "none",
    ]
    assert [spec.residual_correction for spec in SPECS] == [
        "none",
        "none",
        "recent_linear_norm",
    ]
    assert all(spec.policy().content == "recent_sequence_causal" for spec in SPECS)
