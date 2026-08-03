from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/pseudo_one_forward_hard_oracle_v1.yaml"
SAMPLES = ROOT / "configs/benchmark/pseudo_one_forward_accuracy_pilot_v1_samples.json"
CONFIG_SHA256 = "60e03148a24ad16859ca21631a3644d4db6930b0bdaddc482060646b77e902b6"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hard_oracle_protocol_fingerprints_and_scope_are_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))

    assert config["pilot_id"] == "pseudo_one_forward_hard_oracle_v1"
    assert config["status"] == "posthoc_protocol_frozen_before_hard_oracle_generation"
    assert config["dataset"]["sample_ids"] == IDS
    assert [row["sample_id"] for row in samples["accuracy"]] == IDS
    assert config["source"]["resolved_sample_manifest_sha256"] == SAMPLES_SHA256
    assert config["operating_point"] == {
        "horizon": 8,
        "budget_per_layer": 32,
        "routed_experts_per_layer": 128,
        "resident_fraction": 0.25,
        "native_top_k": 8,
        "subset_selection": "future_selected_routing_mass",
        "oracle_lookahead": ("natural_full_expert_greedy_from_current_hard_policy_boundary_state"),
        "oracle_lookahead_cache": "copy_on_write_or_exact_rewind",
        "actual_execution": "hard_mask_outside_subset_before_native_topk_and_normalization",
    }
    assert config["policy"]["name"] == "hard_oracle_commitment"
    assert config["policy"]["label_or_ground_truth_used"] is False
    assert config["policy"]["frozen_v17_future_trajectory_used"] is False
    assert config["policy"]["identity_materialized"] is False
    assert config["execution"]["expected_sample_policy_rows"] == 8
    assert config["execution"]["network_downloads"] is False
    assert config["accuracy_reporting"]["frozen_reference_successes"] == 8
    assert config["accuracy_reporting"]["accuracy_selects_policy_or_samples"] is False
    assert config["accuracy_reporting"]["does_not_modify_parent_pilot_decision"] is True
    assert config["accuracy_reporting"]["conclusion_ceiling"] == "PILOT_NARROW"
