from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_executed_previous_subset_smoke_v1.yaml")
MANIFEST = Path("configs/analysis/pseudo_executed_previous_subset_smoke_v1_samples.json")


def test_executed_previous_subset_smoke_protocol_is_frozen() -> None:
    config_payload = CONFIG.read_bytes()
    manifest_payload = MANIFEST.read_bytes()
    assert hashlib.sha256(config_payload).hexdigest() == (
        "117ce2f4f05d91760d36a7e20136120dbead5853d97af89b37d95faeefcd7f4c"
    )
    assert hashlib.sha256(manifest_payload).hexdigest() == (
        "2f968509cb7fe6e3a97b803ffd002f64bfab847b12e818508959901d065b688a"
    )
    config = yaml.safe_load(config_payload)
    manifest = json.loads(manifest_payload)
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["operating_point"] == {
        "horizon": 8,
        "budget_per_layer": 32,
        "resident_fraction": 0.25,
        "route_token_cap": 16,
        "boundary_rule": "non_overlapping_start_zero_h8",
    }
    assert config["policies"] == [
        "pseudo_executed_previous_subset",
        "provided_previous_residual_control",
        "previous_route_commitment_reference",
    ]
    assert config["scope"]["task_accuracy"] == "forbidden"
    assert config["scope"]["held_out_or_development_expansion"] == "forbidden"
    assert config["executed_residual"]["full_pre_mask_scores_saved_before_execution_mask"]
    assert manifest["selection"]["new_rows_added"] is False
    assert manifest["rows"] == [
        {"row_index": 786, "sample_id": "test-786"},
        {"row_index": 394, "sample_id": "test-394"},
    ]
