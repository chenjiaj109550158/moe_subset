import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.subset_trace import sha256_file

CONFIG = Path("configs/analysis/pseudo_mass_preserving_residual_held_out_v1.yaml")
SAMPLES = Path("configs/analysis/pseudo_mass_preserving_residual_held_out_v1_samples.json")
EXPECTED_CONFIG_SHA256 = "2effc6808f2637f64f60836a90175f664305f5874460a458922da26bfd96416f"
EXPECTED_SAMPLES_SHA256 = "c83dbce41acb7bce2b0cf399c2632965d784ae289f0ca81bb6a6f105aa1762f3"


def test_mass_preserving_residual_held_out_protocol_is_frozen_and_disjoint() -> None:
    assert sha256_file(CONFIG) == EXPECTED_CONFIG_SHA256
    assert sha256_file(SAMPLES) == EXPECTED_SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    assert config["status"] == (
        "protocol_frozen_before_reference_repair_or_held_out_model_execution"
    )
    assert config["selected_candidate"]["key"] == ("natural_top8_intersection_zero_missing")
    assert config["previous_route_commitment"]["boundary_zero"] == (
        "frozen_static_frequency_top32_from_pinned_count_tensor"
    )
    assert config["selected_candidate"]["extra_forward_count"] == (
        "one_batched_h8_traversal_per_boundary"
    )
    assert (
        config["accuracy_authorization"]["task_accuracy_authorized_before_held_out_gate"] is False
    )
    assert samples["selection"]["accuracy_correctness_or_ground_truth_used"] is False
    development = samples["partitions"]["development_reference"]
    held_out = samples["partitions"]["held_out_route"]
    assert len(development) == 4
    assert len(held_out) == 8
    assert {row["sample_id"] for row in development}.isdisjoint(
        {row["sample_id"] for row in held_out}
    )
    assert [row["sample_id"] for row in held_out] == [
        "test-88",
        "test-898",
        "test-141",
        "test-819",
        "test-1250",
        "test-1110",
        "test-1049",
        "test-1222",
    ]
