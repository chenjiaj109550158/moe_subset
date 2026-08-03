from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/analysis/pseudo_mass_preserving_residual_v1.yaml"
SAMPLES = ROOT / "configs/analysis/pseudo_mass_preserving_residual_v1_samples.json"
CONFIG_SHA256 = "6482b21ef9fb42c323844674ce4b8fb4aadff920ed4a15765882d405b39713f3"
SAMPLES_SHA256 = "a0942180ef7cd8c4bf9b003069a96b93554653e5ca3b7c75a5babdc5e567d76d"
IDS = ["test-0", "test-439", "test-879", "test-1318"]
VARIANTS = [
    "natural_top8_intersection_zero_missing",
    "rerouted_top8_scaled_by_captured_natural_mass",
    "captured_natural_plus_substitute_missing_mass",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mass_preserving_residual_protocol_is_frozen() -> None:
    assert _sha256(CONFIG) == CONFIG_SHA256
    assert _sha256(SAMPLES) == SAMPLES_SHA256
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    assert config["analysis_id"] == "pseudo_mass_preserving_residual_v1"
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["variants"]["order"] == VARIANTS
    assert [row["sample_id"] for row in samples["partitions"]["development"]] == IDS
    assert config["operating_point"]["horizon"] == 8
    assert config["operating_point"]["budget_per_layer"] == 32
    assert config["operating_point"]["resident_fraction"] == 0.25
    assert config["operating_point"]["pseudo_content"] == ("sampled_unigram_full_continuation")
    assert config["execution"]["expected_new_sample_variant_rows"] == 12
    assert config["selection_rule"]["ranking_uses_accuracy_or_correctness"] is False
    assert config["analysis"]["held_out_route_authorized"] is False
    assert config["analysis"]["task_accuracy_authorized"] is False
    assert config["analysis"]["learned_predictor_training"] is False
    assert config["progress_gate"]["held_out_route_requires_separate_frozen_protocol"] is True
