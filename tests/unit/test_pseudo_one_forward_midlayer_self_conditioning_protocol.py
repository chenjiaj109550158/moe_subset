from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.pseudo_one_forward_midlayer_self_conditioning import SPECS

CONFIG = Path("configs/analysis/pseudo_one_forward_midlayer_self_conditioning_v1.yaml")
SAMPLES = Path("configs/analysis/pseudo_one_forward_midlayer_self_conditioning_v1_samples.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_midlayer_protocol_is_frozen_calibration_free_and_one_forward() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["analysis_id"] == samples["analysis_id"]
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["midlayer_self_conditioning"]["refresh_count_per_boundary"] == 1
    assert config["midlayer_self_conditioning"]["refresh_after_zero_based_layer"] == 23
    assert config["midlayer_self_conditioning"]["shifted_targets"] == {
        "source_anchor_indices_one_based": [1, 2, 3, 4, 5, 6, 7],
        "target_anchor_indices_one_based": [2, 3, 4, 5, 6, 7, 8],
    }
    assert [row["key"] for row in config["variants"]] == [
        "midpoint_greedy_shift_uncapped",
        "midpoint_expected_top8_shift_uncapped",
        "midpoint_greedy_shift_cap25",
    ]
    assert [row["sample_id"] for row in samples["partitions"]["development"]] == [
        "test-0",
        "test-439",
        "test-879",
        "test-1318",
    ]
    assert samples["selection"]["new_rows_added"] is False
    assert config["audits"]["require_one_causal_forward_per_boundary"] is True
    assert config["analysis"]["held_out_route_authorized"] is False
    assert config["analysis"]["actual_accuracy_authorized"] is False
    forbidden = set(config["information_boundary"]["forbidden"])
    assert {
        "future_true_tokens_after_sampled_next",
        "benchmark_answer",
        "task_accuracy",
        "learned_or_fitted_parameters",
        "offline_expert_priors",
        "default_vector_values",
        "retrieved_history_states",
    } <= forbidden


def test_midlayer_protocol_hashes_are_deterministic() -> None:
    assert _sha256(CONFIG) == ("75f4c028871632139a3f68f290726cb4834931a3a89cd2cc16883e050ebbb67b")
    assert _sha256(SAMPLES) == ("8314356459cbf1bf36bfafffecc0a0a2bb7fd750c22b85ef50cb921ff4b8f535")


def test_midlayer_runner_matches_frozen_variants() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    variants = config["variants"]

    assert [spec.key for spec in SPECS] == [row["key"] for row in variants]
    assert [spec.prediction for spec in SPECS] == [row["content_prediction"] for row in variants]
    assert [spec.max_relative_embedding_delta_norm for spec in SPECS] == [
        row["maximum_embedding_delta_norm_relative_to_mid_hidden"] for row in variants
    ]
    assert [spec.mode for spec in SPECS] == [
        "greedy_shift",
        "expected_top8_shift",
        "greedy_shift",
    ]
    assert all(
        spec.policy().selection == "first_four_anchor_core_plus_history_fill" for spec in SPECS
    )
