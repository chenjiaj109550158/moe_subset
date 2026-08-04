from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.subset_trace import sha256_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1.yaml"
SAMPLES = ROOT / "configs/benchmark/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1_samples.json"
ORIGINAL = ROOT / "configs/benchmark/pseudo_embedding_qwen_gsm8k_v1_samples.json"
CONFIG_SHA256 = "57242f91f204c4e765d893dd754e4e523f8fdce62c303827d2e375cec4cc654c"
SAMPLES_SHA256 = "0f4fd75760390fb8ea468af888c8dcd0b22483cdb2af0f96f51a9833c6614d10"
IDS = (
    "test-44",
    "test-632",
    "test-444",
    "test-519",
    "test-1311",
    "test-1264",
    "test-825",
    "test-252",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_penultimate_joint_protocol_fingerprints_and_scope_are_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))

    assert config["pilot_id"] == "pseudo_penultimate_joint_qwen_gsm8k_wave1_v1"
    assert config["status"] == "protocol_frozen_before_any_penultimate_result"
    assert config["authorization"] == {
        "user_approved_wave_1_accuracy": True,
        "approved_at_utc": "2026-08-04",
        "actual_fused_or_overlapped_runtime_approved": False,
        "checkpoint": "stop_after_wave_1_report_for_user_confirmation",
        "offload_during_accuracy": False,
        "dataset_expansion": "forbidden",
        "token_cap_expansion": "forbidden",
    }
    assert config["model"]["revision"] == ("0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["operating_point"]["resident_fraction"] == 0.25
    assert config["operating_point"]["pseudo_residual_execution"] == (
        "natural_top8_intersection_zero_missing"
    )
    assert config["policy"]["later_windows"]["sampled_last_token_used_for_planning"] is False
    assert config["policy"]["later_windows"]["bridge_execution"] == (
        "current_window_hard_b32_production_semantics"
    )
    assert config["logical_accuracy_simulation"]["offload_engine"] == "disabled"
    assert config["logical_accuracy_simulation"]["intended_fused_runtime_not_implemented"]
    assert config["accuracy_gate"]["minimum_candidate_successes"] == 7
    assert config["accuracy_gate"]["strong_preservation_successes"] == 8
    assert config["accuracy_gate"]["user_confirmation_required_for_fused_runtime"]

    rows = samples["accuracy"]
    assert tuple(row["sample_id"] for row in rows) == IDS
    assert all(row["source_v17_correct"] for row in rows)
    assert sum(row["source_v17_generated_tokens"] for row in rows) == 2101
    original_rows = original["partitions"]["closed_loop_wave_1"]["rows"]
    assert [(row["row_index"], row["sample_id"], row["sha256_rank"]) for row in rows] == [
        (row["row_index"], row["sample_id"], row["sha256_rank"]) for row in original_rows
    ]
    assert samples["selection"]["accuracy_correctness_or_ground_truth_used_to_select_ids"] is False
    assert samples["selection"]["replacement_after_execution"] is False


def test_online_baseline_rows_are_checksum_pinned_external_references() -> None:
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    for reference in samples["accuracy"]:
        baseline = reference["online_baseline"]
        path = ROOT / baseline["path"]
        assert _sha256(path) == baseline["file_sha256"]
        row = json.loads(path.read_text(encoding="utf-8"))
        expected = row.pop("row_payload_sha256")
        assert expected == baseline["row_payload_sha256"]
        assert expected == sha256_json(row)
        assert row["sample_id"] == reference["sample_id"]
        assert row["row_index"] == reference["row_index"]
        assert row["policy"] == "natural_top8_intersection_zero_missing"
        assert row["evaluation_mode"] == "actual_hard_closed_loop_generation"
        assert row["correct"] is True
        assert row["identity_materialized"] is False
        assert row.get("actual_offload_metrics") is None
