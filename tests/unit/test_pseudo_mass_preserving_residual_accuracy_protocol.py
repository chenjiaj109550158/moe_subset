from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/pseudo_mass_preserving_residual_accuracy_pilot_v1.yaml"
SAMPLES = ROOT / "configs/benchmark/pseudo_mass_preserving_residual_accuracy_pilot_v1_samples.json"
CONFIG_SHA256 = "7370d4c3208fcd32ab7b0f903409a7132c14cd784b7360cc7cf7863720f0d46b"
SAMPLES_SHA256 = "ac1aeb3da4778d09628c2e3188fd524392ce253b9fc04b8c795c0312f82dadfe"
IDS = [
    "test-44",
    "test-632",
    "test-444",
    "test-519",
    "test-1311",
    "test-1264",
    "test-825",
    "test-252",
    "test-668",
    "test-892",
    "test-1136",
    "test-850",
    "test-760",
    "test-842",
    "test-87",
    "test-658",
]
POLICIES = [
    "hard_oracle_commitment",
    "previous_route_commitment",
    "natural_top8_intersection_zero_missing",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_accuracy_protocol_fingerprints_scope_and_gate_are_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["pilot_id"] == "pseudo_mass_preserving_residual_accuracy_pilot_v1"
    assert config["status"] == "protocol_frozen_before_any_new_accuracy_generation"
    assert config["source_gate"]["strong_candidate_gate_pass"] is True
    assert config["model"]["revision"] == "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["policies"]["order"] == POLICIES
    assert config["policies"]["previous_route_commitment"]["boundary_zero"] == (
        "frozen_static_frequency_top32_from_pinned_count_tensor"
    )
    candidate = config["policies"]["natural_top8_intersection_zero_missing"]
    assert candidate["captured_natural_weights"] == "preserve"
    assert candidate["missing_natural_mass"] == "zero"
    assert candidate["extra_pseudo_traversals_per_boundary"] == 1
    assert config["execution"]["actual"]["total_sample_policy_rows"] == 48
    assert config["execution"]["actual"]["checksum_pinned_reused_rows"] == 8
    assert config["execution"]["actual"]["expected_new_rows"] == 40
    assert config["accuracy_gate"]["minimum_policy_successes"] == 15
    assert config["accuracy_gate"]["conclusion_ceiling"] == "NARROW"

    rows = samples["accuracy"]
    assert [row["sample_id"] for row in rows] == IDS
    assert len({row["sample_id"] for row in rows}) == 16
    assert all(row["source_v17_correct"] for row in rows)
    assert sum(row["source_v17_generated_tokens"] for row in rows) == 4112
    assert samples["selection"]["accuracy_correctness_or_ground_truth_used_to_select_ids"] is False
    assert samples["work_assignment"]["policy_order"] == POLICIES
    assert samples["work_assignment"]["new_rows"] == 40


def test_accuracy_partitions_are_fixed_disjoint_and_reuse_only_wave_one_oracle() -> None:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    wave_one = [row for row in samples["accuracy"] if row["partition"] == "closed_loop_wave_1"]
    wave_two = [row for row in samples["accuracy"] if row["partition"] == "closed_loop_wave_2"]
    assert len(wave_one) == len(wave_two) == 8
    assert {row["sample_id"] for row in wave_one}.isdisjoint({row["sample_id"] for row in wave_two})
    reused = samples["work_assignment"]["reused"]
    assert reused == {
        "partition": "closed_loop_wave_1",
        "policy": "hard_oracle_commitment",
        "rows": 8,
    }
