from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/qwen_penultimate_joint_offload_speed_v1.yaml"
SAMPLES = ROOT / "configs/benchmark/qwen_penultimate_joint_offload_speed_v1_samples.json"
CONFIG_SHA256 = "6ba8a77f9a749d7baab2b9d784825c7b835fb32538d843cb7d32ea8c1eb27f6d"
SAMPLES_SHA256 = "202486d1591fc46c21ac298c0ff953ccb3245441a6961c7279826db4ec1d62d4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_joint_offload_protocol_fingerprints_are_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256


def test_joint_offload_scope_and_work_order_are_fixed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["authorization"][
        "user_approved_actual_joint_offload_implementation_and_speed_measurement"
    ]
    assert config["model"]["revision"] == "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    assert config["model"]["precision"] == "bfloat16"
    assert config["offload"]["gpu_slots_per_layer"] == 32
    assert config["offload"]["resident_fraction"] == 0.25
    assert config["dataset"]["exact_sample_ids"] == ["test-44", "test-632"]
    assert [row["sample_id"] for row in samples["samples"]] == ["test-44", "test-632"]
    assert config["execution"]["measured"]["row_order"] == samples["work_order"]
    assert not samples["selection"][
        "accuracy_correctness_answer_or_new_runtime_used_to_select_replacements"
    ]


def test_joint_policy_commits_only_bridge_and_uses_one_layer_call() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = config["policies"]["penultimate_unigram_joint_h8_b32_async_prefetch"]
    later = policy["later_windows"]

    assert policy["horizon"] == 8
    assert policy["subset_budget_per_layer"] == 32
    assert later["joint_sequence_tokens"] == 9
    assert later["native_attention_calls_per_layer"] == 1
    assert later["native_router_calls_per_layer"] == 1
    assert later["native_expert_calls_per_layer"] == 1
    assert later["production_cache_commit"] == "bridge_kv_only_exactly_one_position"
    assert later["pseudo_kv_commit"] == "forbidden"
    assert later["next_subset_prefetch"].startswith("layer_local_async")
    assert later["just_sampled_last_token_used_for_planning"] is False


def test_joint_speed_and_accuracy_gates_are_predeclared() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = config["decision_gate"]

    assert gate["traditional_reference_exact_identity_required"] == 2
    assert gate["candidate_correct_required"] == 2
    assert gate["candidate_zero_production_expert_misses_required"]
    assert gate["minimum_speedup_ratio_for_positive_engineering_signal"] == 1.0
    assert gate["strong_speedup_ratio"] == 1.05
    assert config["execution"]["mechanism_smoke"]["sequential_reference_excluded_from_speed"]
    assert config["measurements"]["primary_speed_metric"].startswith(
        "aggregate_post_prefill_decode_forwards_per_second"
    )
