from __future__ import annotations

import inspect
import json

from pseudoroute.benchmark import pseudo_mass_preserving_residual_accuracy as accuracy
from pseudoroute.benchmark import pseudo_one_forward_accuracy_pilot as legacy_pseudo
from pseudoroute.benchmark import subset_closed_loop


def _audit(*, first: bool) -> dict[str, object]:
    return {
        "subset_residual_execution": accuracy.CANDIDATE,
        "full_pre_mask_scores_all_experts": True,
        "execution_weights_finite_nonnegative": True,
        "default_fingerprint": None,
        "execution_weight_sum_min": 0.0,
        "execution_weight_sum_max": 1.001,
        "execution_subsets_supplied": not first,
        "execution_scope": ("full_native_topk" if first else "previous_realized_window_subset"),
        "executed_ids_within_supplied_subset": None if first else True,
    }


def test_work_assignment_skips_only_checksum_pinned_wave_one_oracle() -> None:
    _, samples, _, _ = accuracy._protocol()
    smoke = accuracy._work(samples, stage="smoke", partition="all")
    wave_one = accuracy._work(samples, stage="actual", partition="wave1")
    wave_two = accuracy._work(samples, stage="actual", partition="wave2")
    all_rows = accuracy._work(samples, stage="actual", partition="all")

    assert len(smoke) == 2
    assert len(wave_one) == 16
    assert len(wave_two) == 24
    assert len(all_rows) == 40
    assert all(policy != "hard_oracle_commitment" for _, policy in wave_one)
    assert sum(policy == "hard_oracle_commitment" for _, policy in wave_two) == 8
    assert len({(row["sample_id"], policy) for row, policy in all_rows}) == 40


def test_wave_one_oracle_reuse_validates_all_external_row_checksums() -> None:
    config, samples, _, _ = accuracy._protocol()
    rows = accuracy._legacy_oracle_rows(config, samples)
    assert len(rows) == 8
    assert [row["sample_id"] for row in rows] == list(accuracy.IDS[:8])
    assert all(row["row_source"].startswith("checksum_pinned_") for row in rows)
    assert all(row["hard_mask_executed"] for row in rows)
    assert not any(row["identity_materialized"] for row in rows)


def test_candidate_weight_audit_requires_full_first_and_resident_later() -> None:
    row: dict[str, object] = {
        "planning_rows": [
            {"probe_audit": _audit(first=True)},
            {"probe_audit": _audit(first=False)},
        ]
    }
    assert accuracy._candidate_weight_audit(row)

    bad_mass = json.loads(json.dumps(row))
    bad_mass["planning_rows"][1]["probe_audit"]["execution_weight_sum_max"] = 1.1
    assert not accuracy._candidate_weight_audit(bad_mass)

    bad_subset = json.loads(json.dumps(row))
    bad_subset["planning_rows"][1]["probe_audit"]["executed_ids_within_supplied_subset"] = False
    assert not accuracy._candidate_weight_audit(bad_subset)


def test_new_row_checksum_round_trip_and_scope(tmp_path) -> None:
    path = tmp_path / "row.json"
    payload = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": accuracy.PILOT_ID,
        "config_sha256": accuracy.CONFIG_SHA256,
        "sample_manifest_sha256": accuracy.SAMPLES_SHA256,
        "stage": "actual",
        "row_index": 44,
        "sample_id": "test-44",
        "policy": accuracy.CANDIDATE,
        "max_new_tokens": 512,
    }
    accuracy._write_checksummed(path, payload)
    loaded = accuracy._load_checksummed(
        path,
        stage="actual",
        row_index=44,
        sample_id="test-44",
        policy=accuracy.CANDIDATE,
        max_new_tokens=512,
    )
    assert loaded is not None
    assert loaded["row_payload_sha256"]


def test_optional_evaluator_injection_points_preserve_legacy_defaults() -> None:
    pseudo_parameter = inspect.signature(legacy_pseudo.run_policy_sample).parameters[
        "subset_residual_execution"
    ]
    static_parameter = inspect.signature(subset_closed_loop.run_policy_sample).parameters[
        "static_subsets"
    ]
    assert pseudo_parameter.default == "renormalized_reroute"
    assert static_parameter.default is None
