from __future__ import annotations

from pathlib import Path

from pseudoroute.benchmark import qwen_real_offload_speed as speed
from pseudoroute.benchmark.subset_trace import sha256_json, write_json_atomic


def test_candidate_identity_diagnostic_recovers_measured_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(speed, "OUTPUT", tmp_path)
    reference = {"row_index": 44, "sample_id": "test-44"}
    config = {
        "source": {
            "model_id": "Qwen/test",
            "model_revision": "revision",
            "precision": "bfloat16",
        }
    }
    engine_audit = {
        "slots_per_layer": 32,
        "all_cpu_sources_pinned": True,
        "no_full_expert_parameter_on_cuda": True,
    }
    write_json_atomic(
        tmp_path / "setup" / "execution.json",
        {
            "pid": 123,
            "physical_gpu": 0,
            "engine_audit": engine_audit,
        },
    )
    metrics = {
        "h2d_bytes": 1024,
        "cache_hits": 10,
        "cache_misses": 2,
        "transfer_batches": 2,
        "transfer_ms": 1.0,
        "exposed_stall_ms": 1.0,
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
        "phase_metrics": {"production": {"cache_misses": 0}},
    }
    diagnostic: dict[str, object] = {
        "schema_version": 1,
        "state": "failed_identity_diagnostic",
        "pilot_id": speed.PILOT_ID,
        "config_sha256": speed.CONFIG_SHA256,
        "sample_manifest_sha256": speed.SAMPLES_SHA256,
        "stage": "actual",
        "row_index": 44,
        "sample_id": "test-44",
        "policy": speed.CANDIDATE,
        "generated_token_ids": [1, 9, 3],
        "reference_token_ids": [1, 2, 3],
        "first_token_divergence_from_required_reference": 1,
        "generated_tokens": 3,
        "correct": True,
        "subset_trajectory_sha256": "subset",
        "first_route_divergence": 0,
        "actual_offload_metrics": metrics,
        "prefill_wall_seconds_measured": 2.0,
        "decode_wall_seconds_measured": 4.0,
        "planning_wall_seconds_measured": 1.0,
        "prefetch_wall_seconds_measured": 0.5,
        "production_wall_seconds_measured": 2.5,
        "execution_git_head": "head",
        "pid": 123,
        "ppid": 12,
    }
    diagnostic["failure_payload_sha256"] = sha256_json(diagnostic)
    write_json_atomic(
        tmp_path / "actual" / "00044" / f"{speed.CANDIDATE}.123.DIAGNOSTIC.json",
        diagnostic,
    )

    row = speed._recover_candidate_diagnostic(
        config,
        reference,
        max_new_tokens=512,
    )

    assert row is not None
    assert row["state"] == "complete"
    assert row["required_reference_exact_token_identity"] is False
    assert row["candidate_reference_identity_gate_pass"] is False
    assert row["exact_token_agreement_with_required_reference"] == 2 / 3
    assert row["post_prefill_decode_forwards"] == 2
    assert row["decode_forwards_per_second_measured"] == 0.5
    assert row["actual_h2d_copy_executed"] is True
    assert row["offload_engine_audit"] == engine_audit
    assert row["recovered_from_identity_failure_diagnostic"].endswith(".123.DIAGNOSTIC.json")
