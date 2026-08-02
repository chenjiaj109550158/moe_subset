from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.pseudo_one_forward_context_continuation import (
    CONFIG_SHA256 as RUNNER_CONFIG_SHA256,
)
from pseudoroute.benchmark.pseudo_one_forward_context_continuation import (
    SAMPLES_SHA256 as RUNNER_SAMPLES_SHA256,
)
from pseudoroute.benchmark.pseudo_one_forward_context_continuation import SPECS

ANALYSIS_ID = "pseudo_one_forward_context_continuation_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
CONFIG_SHA256 = "519502aa463af860f4f639ff1233bf14d77bfca6ee6cb5aeff28f80817316897"
SAMPLES_SHA256 = "20a38b86b2b04ded8ded9faaf1c9eb0a90cb73668e9c23980188367a3369b4d4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_context_continuation_protocol_fingerprints_and_scope() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    assert RUNNER_CONFIG_SHA256 == CONFIG_SHA256
    assert RUNNER_SAMPLES_SHA256 == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["analysis_id"] == samples["analysis_id"] == ANALYSIS_ID
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["dataset"]["new_rows_or_traces"] is False
    assert config["operating_point"] == {
        "horizon": 8,
        "budget_per_layer": 32,
        "resident_fraction": 0.25,
        "route_token_cap": 64,
        "attention": "causal",
        "subset_selection": "first_four_anchor_core_plus_history_fill",
        "boundary_zero_expert_access": "full_native_top8",
        "later_expert_access": "current_policy_previous_realized_layer_local_b32_subset",
        "expert_contribution": "fresh_native_moe_residual",
    }
    assert [variant["key"] for variant in config["variants"]] == [
        "longest_suffix_full_continuation",
        "sampled_unigram_full_continuation",
        "longest_suffix_partial_recent_fill",
    ]
    assert [spec.key for spec in SPECS] == [
        "longest_suffix_full_continuation",
        "sampled_unigram_full_continuation",
        "longest_suffix_partial_recent_fill",
    ]
    assert config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"] == 0.02
    assert config["analysis"]["held_out_route_authorized"] is False
    assert config["analysis"]["actual_accuracy_authorized"] is False
    assert config["analysis"]["learned_predictor_training"] is False
    assert config["information_boundary"]["forbidden"] == [
        "future_true_tokens_after_sampled_next",
        "vanilla_future_trajectory",
        "benchmark_answer",
        "correctness",
        "task_accuracy",
        "learned_or_fitted_parameters",
        "offline_ngram_or_continuation_tables",
        "offline_expert_priors",
        "offline_route_transition_tables",
        "default_vector_values",
        "retrieved_history_hidden_states",
    ]
    assert samples["selection"]["accuracy_correctness_or_ground_truth_used"] is False
    assert samples["selection"]["new_rows_added"] is False
    assert samples["partitions"]["development"] == [
        {"row_index": 0, "sample_id": "test-0"},
        {"row_index": 439, "sample_id": "test-439"},
        {"row_index": 879, "sample_id": "test-879"},
        {"row_index": 1318, "sample_id": "test-1318"},
    ]
