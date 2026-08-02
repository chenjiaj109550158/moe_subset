import hashlib
import json
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_embedding_qwen_gsm8k_residual_window_v1.yaml")
MANIFEST = Path("configs/analysis/pseudo_embedding_qwen_gsm8k_residual_window_v1_samples.json")


def test_residual_window_protocol_is_frozen_and_calibration_free() -> None:
    config_payload = CONFIG.read_bytes()
    manifest_payload = MANIFEST.read_bytes()
    assert hashlib.sha256(config_payload).hexdigest() == (
        "cb14de9d2c4ef319b67848569e8eb985cc8ca4d5a398a06acef2e9a737a9e413"
    )
    assert hashlib.sha256(manifest_payload).hexdigest() == (
        "cd6957e34f72f69290ebd9cb147be54ccee3a3dcfe6ef5358f6d34dc850d1255"
    )
    config = yaml.safe_load(config_payload)
    manifest = json.loads(manifest_payload)
    assert config["status"] == "protocol_frozen_before_model_execution"
    assert config["operating_point"] == {
        "horizon": 8,
        "budget_per_layer": 32,
        "resident_fraction": 0.25,
        "boundary_rule": "non_overlapping_start_zero_h8",
    }
    assert config["residual_smoke"]["variants"] == [
        "zero",
        "previous_window_position_aligned",
        "previous_window_last_repeated",
        "previous_window_mean_repeated",
    ]
    forbidden = config["information_boundary"]["forbidden_inputs"]
    assert "learned_or_fitted_parameters" in forbidden
    assert "default_vector_values" in forbidden
    assert config["development"]["mode"] == (
        "teacher_forced_saved_v17_tokens_on_policy_own_hard_subset_state"
    )
    rows = manifest["partitions"]
    assert [len(rows[key]["rows"]) for key in rows] == [2, 4, 8, 8, 8]


def test_new_held_out_sha_ranking_is_exact_and_disjoint() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    excluded_source = json.loads(
        Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1_samples.json").read_text(
            encoding="utf-8"
        )
    )
    excluded = {
        row["row_index"]
        for partition in excluded_source["partitions"].values()
        for row in partition["rows"]
    }
    ranked = []
    for row_index in range(selection["source_population"]):
        if row_index in excluded:
            continue
        sample_id = f"test-{row_index}"
        payload = selection["new_held_out_rank_payload"].format(
            row_index=row_index, sample_id=sample_id
        )
        ranked.append((hashlib.sha256(payload.encode()).hexdigest(), row_index, sample_id))
    ranked.sort()
    expected = [
        {"row_index": row_index, "sample_id": sample_id, "sha256_rank": digest}
        for digest, row_index, sample_id in ranked[:8]
    ]
    actual = manifest["partitions"]["held_out_route_new"]["rows"]
    assert actual == expected
    assert not ({row["row_index"] for row in actual} & excluded)
