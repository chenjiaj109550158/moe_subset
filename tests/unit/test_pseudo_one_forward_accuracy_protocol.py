from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/pseudo_one_forward_accuracy_pilot_v1.yaml"
SAMPLES = ROOT / "configs/benchmark/pseudo_one_forward_accuracy_pilot_v1_samples.json"
CONFIG_SHA256 = "d9515855b897189fde9f36fba151af5467ebc93e46bbf09bb79ad5b39e5f10af"
SAMPLES_SHA256 = "fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046"
IDS = [
    "test-44",
    "test-632",
    "test-444",
    "test-519",
    "test-1311",
    "test-1264",
    "test-825",
    "test-252",
]
POLICIES = [
    "recent_sequence_causal",
    "sampled_unigram_full_continuation",
    "future_exact_content_oracle",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_accuracy_protocol_fingerprints_and_scope_are_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["pilot_id"] == "pseudo_one_forward_accuracy_pilot_v1"
    assert config["status"] == "protocol_frozen_before_accuracy_generation"
    assert config["model"]["revision"] == ("0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
    assert config["operating_point"] == {
        "horizon": 8,
        "budget_per_layer": 32,
        "routed_experts_per_layer": 128,
        "resident_fraction": 0.25,
        "native_top_k": 8,
        "subset_selection": "first_four_anchor_core_plus_history_fill",
        "boundary_zero_history": "last_eight_full_expert_prefill_natural_routes",
        "later_history": "current_policy_previous_realized_window_pre_mask_natural_routes",
        "boundary_zero_shadow_expert_access": "full_native_top8",
        "later_shadow_expert_access": ("current_policy_previous_realized_layer_local_b32_subset"),
        "shadow_expert_contribution": "fresh_native_moe_residual",
        "actual_execution": ("hard_mask_outside_subset_before_native_topk_and_normalization"),
    }
    assert config["policies"]["order"] == POLICIES
    assert config["accuracy_gate"]["minimum_policy_successes"] == 7
    assert config["accuracy_gate"]["conclusion_ceiling"] == "PILOT_NARROW"
    assert config["execution"]["actual"]["expected_sample_policy_rows"] == 24
    assert [row["sample_id"] for row in samples["accuracy"]] == IDS
    assert samples["work_assignment"]["policy_order"] == POLICIES
    assert all(row["source_v17_correct"] for row in samples["accuracy"])
    assert sum(row["source_v17_generated_tokens"] for row in samples["accuracy"]) == 2101
    assert samples["selection"]["accuracy_correctness_or_ground_truth_used_to_select_ids"] is False
    assert (
        config["policies"]["future_exact_content_oracle"][
            "excluded_from_single_extra_forward_claim"
        ]
        is True
    )
