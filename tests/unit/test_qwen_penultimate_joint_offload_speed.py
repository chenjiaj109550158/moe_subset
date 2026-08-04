from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pseudoroute.benchmark import qwen_penultimate_joint_offload_speed as speed
from pseudoroute.benchmark.prefetch import NativeRoute, SubsetRouteRecord
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
            "pinned_cpu_expert_bytes": 4000,
            "gpu_expert_slot_capacity_bytes": 1000,
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
    assert summary["native_model_calls_post_prefill_including_bootstrap"] == 4
    assert summary["native_input_positions_post_prefill_including_pseudo"] == 36
    assert summary["extra_pseudo_input_positions"] == 6
    assert summary["gpu_expert_slot_capacity_bytes"] == 1000


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


def test_runtime_protocol_selector_reads_frozen_v2_and_restores_v1() -> None:
    try:
        speed._select_pilot("v2")
        config, samples = speed._protocol()
        assert config["pilot_id"] == "qwen_penultimate_joint_offload_speed_v2"
        assert [row["sample_id"] for row in samples["samples"]] == ["test-44", "test-632"]
    finally:
        speed._select_pilot("v1")


def test_batching_route_parity_records_layer_slot_and_subset_overlap() -> None:
    sequential: list[SubsetRouteRecord] = []
    joint: list[SubsetRouteRecord] = []
    logits = torch.zeros(1, speed.EXPERTS)
    weights = torch.full((1, speed.TOP_K), 1 / speed.TOP_K)
    base_ids = torch.arange(speed.TOP_K).reshape(1, -1)
    allowed = tuple(range(speed.BUDGET))
    for layer in range(speed.LAYERS):
        left_route = NativeRoute(logits, weights, base_ids)
        right_ids = base_ids.clone()
        if layer == 0:
            right_ids[0, -1] = speed.TOP_K
        right_route = NativeRoute(logits, weights, right_ids)
        sequential.append(
            SubsetRouteRecord(
                layer=layer,
                natural=left_route,
                executed=left_route,
                allowed=allowed,
            )
        )
        joint.append(
            SubsetRouteRecord(
                layer=layer,
                natural=right_route,
                executed=right_route,
                allowed=allowed,
            )
        )
    sequential_next = {layer: allowed for layer in range(speed.LAYERS)}
    joint_next = dict(sequential_next)
    joint_next[0] = tuple((*range(speed.BUDGET - 1), speed.BUDGET))

    result = speed._batching_route_parity(
        tuple(sequential),
        tuple(joint),
        sequential_next,
        joint_next,
    )

    assert result["batching_difference_metrics_recorded"] is True
    assert result["sequential_bridge_natural_exact_layers"] == speed.LAYERS - 1
    assert result["sequential_bridge_executed_exact_layers"] == speed.LAYERS - 1
    assert result["sequential_bridge_natural_slot_agreement"] == (
        speed.LAYERS * speed.TOP_K - 1
    ) / (speed.LAYERS * speed.TOP_K)
    assert result["sequential_next_b32_exact_layers"] == speed.LAYERS - 1
    assert result["sequential_next_b32_min_overlap_fraction"] == 31 / 32
