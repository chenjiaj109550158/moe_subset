from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/analysis/pseudo_state_component_swap_v1.yaml"
SAMPLES = ROOT / "configs/analysis/pseudo_state_component_swap_v1_samples.json"
CONFIG_SHA256 = "6fc9c4cd92de0108c198de03de63badd38f0a5c9885aad33e15a893ea5efe53f"
SAMPLES_SHA256 = "2a75b1cb96f46189d469d1d5a431174351fa726f44a71909cd1708c5ed27bb17"
IDS = ["test-0", "test-439", "test-879", "test-1318"]
VARIANTS = [
    "exact_future_unpatched",
    "exact_future_attention_output_oracle",
    "exact_future_moe_residual_oracle",
    "exact_future_attention_and_moe_oracle",
    "exact_future_hidden_after_layer_7_oracle",
    "exact_future_hidden_after_layer_15_oracle",
    "exact_future_hidden_after_layer_23_oracle",
    "exact_future_hidden_after_layer_31_oracle",
    "exact_future_hidden_after_layer_39_oracle",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_component_swap_protocol_fingerprints_and_scope_are_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["analysis_id"] == "pseudo_state_component_swap_v1"
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["variants"]["order"] == VARIANTS
    assert [row["sample_id"] for row in samples["partitions"]["development"]] == IDS
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["operating_point"]["route_token_cap"] == 64
    assert config["dataset"]["new_rows"] is False
    assert config["execution"]["expected_sample_variant_rows"] == 36
    assert config["analysis"]["task_accuracy_authorized"] is False
    assert config["analysis"]["held_out_route_authorized"] is False
    assert config["analysis"]["learned_predictor_training"] is False
    assert config["information_boundary"]["deployability"]["all_component_swap_variants"] is False
    assert config["next_candidate_mapping"] == {
        "attention_dominant": "batched_recent_unigram_current_multiview",
        "moe_residual_dominant": "mass_preserving_previous_subset_residual",
        "coupled_attention_and_residual": "mass_preserving_batched_multiview",
        "component_path_unresolved": "stop_without_candidate_execution",
        "candidate_requires_separate_frozen_protocol_before_model_output": True,
    }
    checkpoints = [config["variants"][key]["hidden_after_layer_patch"] for key in VARIANTS[4:]]
    assert checkpoints == [7, 15, 23, 31, 39]
