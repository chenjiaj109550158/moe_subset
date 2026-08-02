from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.pseudo_one_forward_token_aligned_retrieval import SPECS

CONFIG = Path("configs/analysis/pseudo_one_forward_token_aligned_retrieval_v1.yaml")
SAMPLES = Path("configs/analysis/pseudo_one_forward_token_aligned_retrieval_v1_samples.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_token_aligned_retrieval_protocol_is_frozen_and_calibration_free() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["analysis_id"] == samples["analysis_id"]
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["operating_point"]["content"] == "recent_sequence"
    assert len(config["variants"]) == 5
    assert len(samples["partitions"]["development"]) == 4
    assert samples["selection"]["new_rows_added"] is False
    assert [row["sample_id"] for row in samples["partitions"]["development"]] == [
        "test-0",
        "test-439",
        "test-879",
        "test-1318",
    ]

    forbidden = set(config["information_boundary"]["forbidden"])
    assert {
        "future_true_tokens_after_sampled_next",
        "vanilla_future_trajectory",
        "benchmark_answer",
        "task_accuracy",
        "learned_or_fitted_parameters",
        "offline_route_transition_tables",
        "offline_expert_priors",
        "default_vector_values",
    } <= forbidden
    assert config["analysis"]["held_out_route_authorized"] is False
    assert config["analysis"]["actual_accuracy_authorized"] is False
    assert config["analysis"]["learned_predictor_training"] is False


def test_token_aligned_retrieval_protocol_hashes_are_deterministic() -> None:
    assert _sha256(CONFIG) == ("19a368fe6a5e819cc0e5b3a6c3528ccf9baf94a66ba4733a278e9c0d1e5cadff")
    assert _sha256(SAMPLES) == ("dc18107f178100f41df1ffccdf23f07ddd3f611044f4b7f7ac5da1edbad36c69")


def test_token_aligned_retrieval_variants_match_frozen_design() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    variants = config["variants"]

    assert [variant["key"] for variant in variants] == [
        "exact_recent_residual_additive_norm",
        "exact_recent_residual_replace_norm",
        "exact_then_embedding_nearest_residual_additive_norm",
        "exact_recent_router_input_additive_norm",
        "exact_then_embedding_nearest_router_input_additive_norm",
    ]
    assert [variant["target"] for variant in variants] == [
        "fresh_moe_residual",
        "fresh_moe_residual",
        "fresh_moe_residual",
        "router_input",
        "router_input",
    ]
    assert config["audits"]["require_one_causal_forward_per_boundary"] is True
    assert config["audits"]["require_retrieved_indices_precede_boundary"] is True
    assert [spec.key for spec in SPECS] == [variant["key"] for variant in variants]
    assert [spec.retrieval_mode for spec in SPECS] == [variant["retrieval"] for variant in variants]
    assert [spec.target for spec in SPECS] == [variant["target"] for variant in variants]
    assert [spec.mixing for spec in SPECS] == [variant["mixing"] for variant in variants]
    assert all(
        spec.policy().selection == "first_four_anchor_core_plus_history_fill" for spec in SPECS
    )
