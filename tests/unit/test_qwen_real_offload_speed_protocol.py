from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/qwen_real_offload_speed_pilot_v1.yaml"
SAMPLES = ROOT / "configs/benchmark/qwen_real_offload_speed_pilot_v1_samples.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_offload_protocol_fingerprints_are_frozen() -> None:
    assert _sha256(CONFIG) == "874ac3aef1802ffbb2274e2aeeb2bdb655a2350e348b09673d32354225a014d0"
    assert _sha256(SAMPLES) == "2a10f7d065a028e0ca52e6fa44757f40d8f9cc23f9a59004f87482b8b1bbf3df"


def test_real_offload_protocol_has_equal_resident_budget_and_true_copies() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    samples = json.loads(SAMPLES.read_text())

    assert config["offload"]["gpu_slots_per_layer"] == 32
    assert config["offload"]["resident_fraction"] == 0.25
    assert config["offload"]["host_store"] == "pinned_cpu_bfloat16"
    assert config["offload"]["transfer_direction"] == "cpu_to_cuda"
    assert config["offload"]["speculative_transfer_compute_overlap"] is False
    assert config["decode"]["max_new_tokens"] == 512
    assert [row["sample_id"] for row in samples["samples"]] == ["test-44", "test-632"]
    assert not samples["selection"][
        "accuracy_correctness_answer_runtime_or_policy_result_used_to_select_ids"
    ]

    lossless = config["policies"]["lossless_dynamic_top8_lru_b32"]
    candidate = config["policies"]["natural_top8_intersection_zero_missing_h8_b32"]
    assert lossless["resident_cache_budget_per_layer"] == 32
    assert lossless["executed_route"] == "exact_native_top8"
    assert candidate["subset_budget_per_layer"] == 32
    assert candidate["horizon"] == 8
    assert candidate["missing_natural_mass"] == "zero"
    assert candidate["substitute_experts"] == "none"


def test_real_offload_work_order_is_fixed_ab_ba() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["execution"]["measured"]["row_order"] == [
        ["test-44", "lossless_dynamic_top8_lru_b32"],
        ["test-44", "natural_top8_intersection_zero_missing_h8_b32"],
        ["test-632", "natural_top8_intersection_zero_missing_h8_b32"],
        ["test-632", "lossless_dynamic_top8_lru_b32"],
    ]
