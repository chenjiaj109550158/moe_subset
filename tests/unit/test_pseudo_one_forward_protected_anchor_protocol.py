from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.pseudo_one_forward_protected_anchor import DAMPED, SPECS, UNDAMPED

CONFIG = Path("configs/analysis/pseudo_one_forward_protected_anchor_v2.yaml")
SAMPLES = Path("configs/analysis/pseudo_one_forward_protected_anchor_v2_samples.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protected_anchor_protocol_is_frozen_before_results() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    assert config["analysis_id"] == samples["analysis_id"]
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["fixed_mechanism"]["anchor_one_correction_coefficient"] == 0.0
    assert config["analytic_constraints"]["horizon_damped_coefficients"] == [
        index / 8 for index in range(8)
    ]
    assert config["analytic_constraints"]["correction_delta_norm_cap_relative_to_fresh"] == 0.25
    assert len(config["variants"]) == 5
    assert len(samples["partitions"]["development"]) == 4
    assert samples["selection"]["new_rows_added"] is False
    forbidden = set(config["information_boundary"]["forbidden"])
    assert {
        "future_true_tokens_after_sampled_next",
        "task_accuracy",
        "learned_or_fitted_parameters",
    } <= forbidden
    assert config["analysis"]["held_out_route_authorized"] is False
    assert config["analysis"]["actual_accuracy_authorized"] is False


def test_protected_anchor_protocol_hashes_are_deterministic() -> None:
    assert _sha256(CONFIG) == "5bac51be6be2ab90aed563eb9208ce23afa772c9b109e9fa1bf66ee319bb81ee"
    assert _sha256(SAMPLES) == "d37ff792acb6cb0d2fe0158040f7e3db8d3555783492c46ad4349362ada302ea"


def test_protected_anchor_runner_matches_frozen_variants() -> None:
    assert len(SPECS) == 5
    assert DAMPED == tuple(index / 8 for index in range(8))
    assert UNDAMPED == tuple(float(index) for index in range(8))
    assert all(spec.coefficients[0] == 0 for spec in SPECS)
    assert [spec.max_relative_delta_norm for spec in SPECS] == [
        None,
        None,
        None,
        0.25,
        0.25,
    ]
    assert SPECS[-1].selection == "anchor_one_top8_plus_later_corrected_utility_fill"
