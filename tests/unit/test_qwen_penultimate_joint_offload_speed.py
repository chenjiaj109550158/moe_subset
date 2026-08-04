from __future__ import annotations

import json
from pathlib import Path

import pytest

from pseudoroute.benchmark.qwen_penultimate_joint_offload_speed import (
    CONFIG_SHA256,
    JOINT,
    PILOT_ID,
    SAMPLES_SHA256,
    TRADITIONAL,
    _cap,
    _focused_decision,
    _load_checksummed,
    _policy_summary,
    _protocol,
    _work,
    _write_checksummed,
)


def _summary_row(
    *,
    forwards: int,
    decode_wall: float,
    generated: int,
    inference_wall: float,
    h2d_bytes: int,
) -> dict[str, object]:
    return {
        "post_prefill_decode_forwards": forwards,
        "decode_wall_seconds_measured": decode_wall,
        "generated_tokens": generated,
        "end_to_end_inference_wall_seconds_measured": inference_wall,
        "correct": True,
        "required_reference_exact_token_identity": True,
        "production_expert_misses": 0,
        "joint_calls": 2,
        "joint_input_tokens": 18,
        "actual_offload_metrics": {
            "h2d_bytes": h2d_bytes,
            "transfer_ms": h2d_bytes / 10,
            "exposed_stall_ms": h2d_bytes / 100,
            "async_prefetch_batches": 3,
            "async_prefetch_bytes": h2d_bytes // 2,
            "deferred_ready_waits": 4,
            "deferred_ready_wait_ms": h2d_bytes / 200,
            "peak_allocated_bytes": h2d_bytes * 10,
            "peak_reserved_bytes": h2d_bytes * 20,
        },
    }


def test_policy_summary_uses_aggregate_forward_time_and_measured_transfer() -> None:
    rows = [
        _summary_row(
            forwards=10,
            decode_wall=2.0,
            generated=11,
            inference_wall=3.0,
            h2d_bytes=100,
        ),
        _summary_row(
            forwards=20,
            decode_wall=4.0,
            generated=24,
            inference_wall=4.0,
            h2d_bytes=200,
        ),
    ]

    summary = _policy_summary(rows)  # type: ignore[arg-type]

    assert summary["aggregate_post_prefill_decode_forwards_per_second_measured"] == 5.0
    assert summary["aggregate_end_to_end_tokens_per_second_measured"] == 5.0
    assert summary["actual_h2d_bytes"] == 300
    assert summary["cuda_event_transfer_seconds"] == 0.03
    assert summary["cuda_event_exposed_stall_seconds"] == 0.003
    assert summary["async_prefetch_batches"] == 6
    assert summary["deferred_ready_waits"] == 8
    assert summary["production_expert_misses"] == 0


@pytest.mark.parametrize(
    ("speedup", "invariants", "expected"),
    [
        (0.99, True, "NARROW_NO_SPEEDUP"),
        (1.00, True, "NARROW_POSITIVE_ENGINEERING_SIGNAL"),
        (1.05, True, "NARROW_STRONG_ENGINEERING_SIGNAL"),
        (1.20, False, "STOP_PIVOT_INVARIANT_OR_ACCURACY_FAILURE"),
    ],
)
def test_focused_decision_respects_frozen_speed_and_invariant_gates(
    speedup: float,
    invariants: bool,
    expected: str,
) -> None:
    assert (
        _focused_decision(
            speedup=speedup,
            traditional_identity=True,
            joint_correct=True,
            joint_zero_miss=True,
            invariants=invariants,
            positive_threshold=1.0,
            strong_threshold=1.05,
        )
        == expected
    )


def test_focused_decision_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        _focused_decision(
            speedup=1.0,
            traditional_identity=True,
            joint_correct=True,
            joint_zero_miss=True,
            invariants=True,
            positive_threshold=1.1,
            strong_threshold=1.0,
        )


def test_protocol_keeps_smoke_cap_and_ab_ba_actual_order() -> None:
    config, samples = _protocol()

    assert _cap(config, "smoke") == 17
    assert _cap(config, "actual") == 512
    assert [(row["sample_id"], policy) for row, policy in _work(config, samples, "actual")] == [
        ("test-44", TRADITIONAL),
        ("test-44", JOINT),
        ("test-632", JOINT),
        ("test-632", TRADITIONAL),
    ]


def test_atomic_row_checksum_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    row: dict[str, object] = {
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "stage": "smoke",
        "row_index": 44,
        "sample_id": "test-44",
        "policy": JOINT,
        "max_new_tokens": 17,
    }
    _write_checksummed(path, row)

    loaded = _load_checksummed(
        path,
        stage="smoke",
        row_index=44,
        sample_id="test-44",
        policy=JOINT,
        max_new_tokens=17,
    )
    assert loaded is not None
    assert loaded["sample_id"] == "test-44"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sample_id"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _load_checksummed(
            path,
            stage="smoke",
            row_index=44,
            sample_id="test-44",
            policy=JOINT,
            max_new_tokens=17,
        )
