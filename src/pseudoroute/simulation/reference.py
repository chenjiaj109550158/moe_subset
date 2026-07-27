"""Minimal M9 reference timeline for comparing fixed and adaptive plans."""

from __future__ import annotations

from dataclasses import dataclass

from pseudoroute.execution.adaptive import ResidencyPlan
from pseudoroute.types import ExpertKey, ModelSpec


@dataclass(frozen=True)
class ReferenceEvent:
    timestamp_us: float
    event: str
    boundary: int
    bytes: int
    reason: str


@dataclass(frozen=True)
class ReferenceSimulationSummary:
    simulation_scope: str
    mode: str
    total_time_us: float
    transfer_time_us: float
    compute_time_us: float
    transfer_bytes: int
    replans: int


def simulate_reference_timeline(
    spec: ModelSpec,
    plans: list[ResidencyPlan],
    *,
    mode: str,
    generated_tokens: int,
    bandwidth_bytes_per_us: float,
    fixed_latency_us: float,
    compute_us_per_token: float,
) -> tuple[ReferenceSimulationSummary, list[ReferenceEvent]]:
    if bandwidth_bytes_per_us <= 0 or fixed_latency_us < 0 or compute_us_per_token < 0:
        raise ValueError("invalid reference simulation hardware")
    timestamp = 0.0
    transfer_time = 0.0
    transfer_bytes = 0
    events = []
    resident: frozenset[ExpertKey] = frozenset()
    static = frozenset(plans[0].static_experts) if plans else frozenset()
    for plan in plans:
        if not static <= plan.allowed:
            raise RuntimeError("static expert was evicted")
        if resident and any(key in static for key in plan.eviction_delta):
            raise RuntimeError("timeline attempts to evict a static expert")
        byte_count = sum(spec.expert_bytes[key] for key in plan.load_delta)
        duration = (fixed_latency_us if byte_count else 0.0) + byte_count / bandwidth_bytes_per_us
        events.append(ReferenceEvent(timestamp, "replan", plan.boundary, 0, plan.reason))
        if byte_count:
            events.append(
                ReferenceEvent(timestamp, "load_start", plan.boundary, byte_count, plan.reason)
            )
            timestamp += duration
            events.append(
                ReferenceEvent(timestamp, "load_end", plan.boundary, byte_count, plan.reason)
            )
        transfer_time += duration
        transfer_bytes += byte_count
        resident = plan.allowed
    compute_time = generated_tokens * compute_us_per_token
    timestamp += compute_time
    return (
        ReferenceSimulationSummary(
            "m9_reference_serial_transfer_no_overlap",
            mode,
            timestamp,
            transfer_time,
            compute_time,
            transfer_bytes,
            len(plans),
        ),
        events,
    )
