import json
from pathlib import Path

import yaml

from pseudoroute.benchmark.subset_trace import sha256_json

CONFIG = Path("configs/analysis/pseudo_executed_embedding_composition_v1.yaml")
SAMPLES = Path("configs/analysis/pseudo_executed_embedding_composition_v1_samples.json")


def test_composition_protocol_has_deterministic_identity_and_disjoint_rows() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    assert config["analysis_id"] == samples["analysis_id"]
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert sha256_json(config) == sha256_json(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    development = samples["partitions"]["development"]
    held_out = samples["partitions"]["held_out_route"]
    assert len(development) == 4
    assert len(held_out) == 8
    assert {row["sample_id"] for row in development}.isdisjoint(
        row["sample_id"] for row in held_out
    )
    assert samples["selection"]["accuracy_correctness_or_ground_truth_used"] is False


def test_composition_candidates_are_calibration_free_and_diagnostics_separate() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    variants = config["development_variants"]
    assert all("exact_future" not in key for key in variants["deployable"])
    assert set(variants["diagnostics"]) == {
        "exact_future_independent",
        "exact_future_causal",
    }
    assert "learned_or_fitted_parameters" in config["information_boundary"]["forbidden"]
    assert config["ranking"]["accuracy_or_correctness_used"] is False
