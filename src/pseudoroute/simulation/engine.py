"""M8 event-driven expert-cache and transfer simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pseudoroute.types import ExpertKey


class CacheState(StrEnum):
    CPU = "cpu"
    LOADING = "loading"
    RESIDENT = "resident"


@dataclass(frozen=True)
class SimHardware:
    expert_capacity_bytes: int
    h2d_bandwidth_bytes_per_s: float
    fixed_latency_us: float
    max_concurrent_transfers: int
    dense_compute_us_per_token: float
    expert_compute_us: float
    overlap_transfers: bool

    @property
    def bandwidth_bytes_per_us(self) -> float:
        return self.h2d_bandwidth_bytes_per_s / 1_000_000


@dataclass(frozen=True)
class RouteToken:
    token_index: int
    experts_by_layer: tuple[tuple[ExpertKey, ...], ...]


@dataclass(frozen=True)
class CacheEvent:
    timestamp_us: float
    event: Literal[
        "probe_start",
        "probe_end",
        "load_start",
        "load_end",
        "evict",
        "stall_start",
        "stall_end",
        "expert_compute_start",
        "expert_compute_end",
    ]
    expert: ExpertKey | None
    layer_idx: int | None
    bytes: int
    stream: str
    reason: str
    token_index: int | None


@dataclass
class CacheEntry:
    state: CacheState
    completion_us: float
    last_used: int
    frequency: int
    prefetched: bool
    used_since_prefetch: bool


@dataclass(frozen=True)
class SimulationSummary:
    simulated: bool
    baseline: str
    tokens: int
    total_time_us: float
    mean_tpot_us: float
    p50_tpot_us: float
    p95_tpot_us: float
    p99_tpot_us: float
    exposed_stall_us: float
    transfer_bytes: int
    bytes_per_token: float
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    wasted_prefetch_bytes: int
    route_coverage: float
    peak_resident_bytes: int
    no_overlap_bound_us: float
    perfect_overlap_bound_us: float


class ExpertCache:
    def __init__(self, sizes: dict[ExpertKey, int], capacity_bytes: int, policy: str) -> None:
        self.sizes = sizes
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.entries: dict[ExpertKey, CacheEntry] = {}
        self.global_frequency: dict[ExpertKey, int] = {}
        self.wasted_prefetch_bytes = 0
        self.peak_bytes = 0

    @property
    def occupied_bytes(self) -> int:
        return sum(self.sizes[key] for key in self.entries)

    def promote_completed(self, timestamp: float) -> None:
        for entry in self.entries.values():
            if entry.state is CacheState.LOADING and entry.completion_us <= timestamp:
                entry.state = CacheState.RESIDENT

    def evict_for(
        self,
        required_bytes: int,
        *,
        timestamp: float,
        protected: frozenset[ExpertKey],
        events: list[CacheEvent],
        reason: str,
        token_index: int | None,
    ) -> None:
        self.promote_completed(timestamp)
        while self.occupied_bytes + required_bytes > self.capacity_bytes:
            candidates = [
                (key, entry)
                for key, entry in self.entries.items()
                if key not in protected and entry.state is CacheState.RESIDENT
            ]
            if not candidates:
                raise RuntimeError("expert cache capacity cannot fit required working set")
            if self.policy == "lfu":
                victim, entry = min(
                    candidates, key=lambda item: (item[1].frequency, item[1].last_used, item[0])
                )
            else:
                victim, entry = min(
                    candidates, key=lambda item: (item[1].last_used, item[1].frequency, item[0])
                )
            if entry.prefetched and not entry.used_since_prefetch:
                self.wasted_prefetch_bytes += self.sizes[victim]
            del self.entries[victim]
            events.append(
                CacheEvent(
                    timestamp,
                    "evict",
                    victim,
                    victim.layer_idx,
                    self.sizes[victim],
                    "cache",
                    reason,
                    token_index,
                )
            )

    def reserve(self, key: ExpertKey, completion: float, *, prefetched: bool) -> None:
        self.entries[key] = CacheEntry(
            CacheState.LOADING,
            completion,
            -1,
            self.global_frequency.get(key, 0),
            prefetched,
            False,
        )
        self.peak_bytes = max(self.peak_bytes, self.occupied_bytes)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))]


def _event_order(event: CacheEvent) -> float:
    # Python's stable sort preserves causal insertion order for equal timestamps.
    return event.timestamp_us


def _planned_experts(
    routes: tuple[RouteToken, ...], baseline: str, boundary: int, horizon: int, budget: int
) -> frozenset[ExpertKey]:
    if baseline == "lossless_predictor":
        source = routes[max(0, boundary - 1)]
        return frozenset(key for layer in source.experts_by_layer for key in layer)
    if baseline == "one_step_commitment":
        source = routes[boundary]
        return frozenset(key for layer in source.experts_by_layer for key in layer)
    if baseline == "multi_step_commitment":
        selected = []
        windows = routes[boundary : boundary + horizon]
        for layer_idx in range(len(routes[0].experts_by_layer)):
            counts: dict[ExpertKey, int] = {}
            for token in windows:
                for key in token.experts_by_layer[layer_idx]:
                    counts[key] = counts.get(key, 0) + 1
            selected.extend(sorted(counts, key=lambda key: (-counts[key], key))[:budget])
        return frozenset(selected)
    return frozenset()


def simulate(
    routes: tuple[RouteToken, ...],
    sizes: dict[ExpertKey, int],
    hardware: SimHardware,
    *,
    baseline: Literal[
        "on_demand",
        "lru",
        "lfu",
        "lossless_predictor",
        "one_step_commitment",
        "multi_step_commitment",
        "custom_plan",
    ],
    horizon: int,
    per_layer_budget: int,
    probe_us: float,
    prefetch_schedule: dict[int, frozenset[ExpertKey]] | None = None,
) -> tuple[SimulationSummary, list[CacheEvent]]:
    if not routes or horizon < 1 or per_layer_budget < 1:
        raise ValueError("simulation requires routes and positive horizon/budget")
    if hardware.expert_capacity_bytes < max(sizes.values()):
        raise ValueError("cache cannot hold one expert")
    policy = "lfu" if baseline == "lfu" else "lru"
    cache = ExpertCache(sizes, hardware.expert_capacity_bytes, policy)
    events: list[CacheEvent] = []
    timestamp = 0.0
    transfer_bytes = 0
    transfer_service_us = 0.0
    exposed_stall = 0.0
    hits = 0
    misses = 0
    natural_count = 0
    executed_count = 0
    token_latencies = []

    def issue_batch(
        keys: list[ExpertKey],
        issue_time: float,
        *,
        prefetched: bool,
        reason: str,
        token: int,
        protected_keys: frozenset[ExpertKey] = frozenset(),
    ) -> float:
        nonlocal transfer_bytes, transfer_service_us
        missing = [key for key in sorted(set(keys)) if key not in cache.entries]
        latest = issue_time
        cache.evict_for(
            sum(sizes[key] for key in missing),
            timestamp=issue_time,
            protected=frozenset(missing) | protected_keys,
            events=events,
            reason=reason,
            token_index=token,
        )
        for wave_start in range(0, len(missing), hardware.max_concurrent_transfers):
            wave = missing[wave_start : wave_start + hardware.max_concurrent_transfers]
            byte_count = sum(sizes[key] for key in wave)
            duration = hardware.fixed_latency_us + byte_count / hardware.bandwidth_bytes_per_us
            completion = issue_time + duration
            for key in wave:
                events.append(
                    CacheEvent(
                        issue_time,
                        "load_start",
                        key,
                        key.layer_idx,
                        sizes[key],
                        "h2d",
                        reason,
                        token,
                    )
                )
                events.append(
                    CacheEvent(
                        completion, "load_end", key, key.layer_idx, sizes[key], "h2d", reason, token
                    )
                )
                cache.reserve(key, completion, prefetched=prefetched)
            transfer_bytes += byte_count
            transfer_service_us += duration
            latest = max(latest, completion)
            issue_time = completion
        return latest

    for token in routes:
        token_start = timestamp
        commitment = baseline in {"one_step_commitment", "multi_step_commitment"}
        planned: frozenset[ExpertKey] = frozenset()
        if (
            baseline == "custom_plan"
            and prefetch_schedule is not None
            and token.token_index in prefetch_schedule
        ):
            events.append(
                CacheEvent(
                    timestamp,
                    "probe_start",
                    None,
                    None,
                    0,
                    "compute",
                    baseline,
                    token.token_index,
                )
            )
            timestamp += probe_us
            events.append(
                CacheEvent(
                    timestamp,
                    "probe_end",
                    None,
                    None,
                    0,
                    "compute",
                    baseline,
                    token.token_index,
                )
            )
            planned = prefetch_schedule[token.token_index]
            prefetch_end = issue_batch(
                list(planned),
                timestamp,
                prefetched=True,
                reason="custom_plan_prefetch",
                token=token.token_index,
                protected_keys=planned,
            )
            if not hardware.overlap_transfers:
                timestamp = prefetch_end
        elif (
            baseline in {"lossless_predictor", "one_step_commitment", "multi_step_commitment"}
            and token.token_index % horizon == 0
        ):
            events.append(
                CacheEvent(
                    timestamp, "probe_start", None, None, 0, "compute", baseline, token.token_index
                )
            )
            timestamp += probe_us
            events.append(
                CacheEvent(
                    timestamp, "probe_end", None, None, 0, "compute", baseline, token.token_index
                )
            )
            planned = _planned_experts(
                routes, baseline, token.token_index, horizon, per_layer_budget
            )
            prefetch_end = issue_batch(
                list(planned),
                timestamp,
                prefetched=True,
                reason="window_prefetch",
                token=token.token_index,
            )
            if not hardware.overlap_transfers:
                timestamp = prefetch_end
        elif commitment:
            boundary = token.token_index - token.token_index % horizon
            planned = _planned_experts(routes, baseline, boundary, horizon, per_layer_budget)
        timestamp += hardware.dense_compute_us_per_token
        for natural_layer in token.experts_by_layer:
            natural_count += len(natural_layer)
            required = tuple(key for key in natural_layer if not commitment or key in planned)
            executed_count += len(required)
            protected = frozenset(required)
            for key in required:
                cache.promote_completed(timestamp)
                entry = cache.entries.get(key)
                if entry is not None:
                    hits += 1
                    if entry.completion_us > timestamp:
                        events.append(
                            CacheEvent(
                                timestamp,
                                "stall_start",
                                key,
                                key.layer_idx,
                                0,
                                "compute",
                                "wait_for_prefetch",
                                token.token_index,
                            )
                        )
                        exposed_stall += entry.completion_us - timestamp
                        timestamp = entry.completion_us
                        cache.promote_completed(timestamp)
                        events.append(
                            CacheEvent(
                                timestamp,
                                "stall_end",
                                key,
                                key.layer_idx,
                                0,
                                "compute",
                                "wait_for_prefetch",
                                token.token_index,
                            )
                        )
                else:
                    misses += 1
                    cache.evict_for(
                        sizes[key],
                        timestamp=timestamp,
                        protected=protected,
                        events=events,
                        reason="demand_capacity",
                        token_index=token.token_index,
                    )
                    completion = issue_batch(
                        [key],
                        timestamp,
                        prefetched=False,
                        reason="demand_load",
                        token=token.token_index,
                    )
                    events.append(
                        CacheEvent(
                            timestamp,
                            "stall_start",
                            key,
                            key.layer_idx,
                            0,
                            "compute",
                            "demand_load",
                            token.token_index,
                        )
                    )
                    exposed_stall += completion - timestamp
                    timestamp = completion
                    cache.promote_completed(timestamp)
                    events.append(
                        CacheEvent(
                            timestamp,
                            "stall_end",
                            key,
                            key.layer_idx,
                            0,
                            "compute",
                            "demand_load",
                            token.token_index,
                        )
                    )
                    entry = cache.entries[key]
                entry.last_used = token.token_index
                entry.frequency += 1
                cache.global_frequency[key] = entry.frequency
                entry.used_since_prefetch = True
                events.append(
                    CacheEvent(
                        timestamp,
                        "expert_compute_start",
                        key,
                        key.layer_idx,
                        0,
                        "compute",
                        baseline,
                        token.token_index,
                    )
                )
                timestamp += hardware.expert_compute_us
                events.append(
                    CacheEvent(
                        timestamp,
                        "expert_compute_end",
                        key,
                        key.layer_idx,
                        0,
                        "compute",
                        baseline,
                        token.token_index,
                    )
                )
            if baseline == "on_demand":
                for key in list(cache.entries):
                    if cache.entries[key].state is CacheState.RESIDENT:
                        del cache.entries[key]
                        events.append(
                            CacheEvent(
                                timestamp,
                                "evict",
                                key,
                                key.layer_idx,
                                sizes[key],
                                "cache",
                                "on_demand_no_cache",
                                token.token_index,
                            )
                        )
        token_latencies.append(timestamp - token_start)
    for key, entry in cache.entries.items():
        if entry.prefetched and not entry.used_since_prefetch:
            cache.wasted_prefetch_bytes += sizes[key]
    probe_count = (
        math.ceil(len(routes) / horizon)
        if baseline in {"lossless_predictor", "one_step_commitment", "multi_step_commitment"}
        else 0
    )
    if baseline == "custom_plan" and prefetch_schedule is not None:
        probe_count = len(prefetch_schedule)
    compute_floor = (
        len(routes) * hardware.dense_compute_us_per_token
        + executed_count * hardware.expert_compute_us
    )
    no_overlap = compute_floor + transfer_service_us + probe_us * probe_count
    perfect_overlap = compute_floor + probe_us * probe_count
    if timestamp > no_overlap and timestamp - no_overlap <= 1e-9:
        no_overlap = timestamp
    summary = SimulationSummary(
        True,
        baseline,
        len(routes),
        timestamp,
        timestamp / len(routes),
        _percentile(token_latencies, 0.5),
        _percentile(token_latencies, 0.95),
        _percentile(token_latencies, 0.99),
        exposed_stall,
        transfer_bytes,
        transfer_bytes / len(routes),
        hits,
        misses,
        hits / max(1, hits + misses),
        cache.wasted_prefetch_bytes,
        executed_count / natural_count,
        cache.peak_bytes,
        no_overlap,
        perfect_overlap,
    )
    validate_timeline(summary, events, sizes, hardware)
    return summary, sorted(events, key=_event_order)


def validate_timeline(
    summary: SimulationSummary,
    events: list[CacheEvent],
    sizes: dict[ExpertKey, int],
    hardware: SimHardware,
) -> None:
    if summary.peak_resident_bytes > hardware.expert_capacity_bytes:
        raise RuntimeError("simulated cache exceeds capacity")
    completion: dict[ExpertKey, float] = {}
    starts: dict[ExpertKey, list[float]] = {}
    for event in sorted(events, key=_event_order):
        if event.event == "load_start" and event.expert is not None:
            starts.setdefault(event.expert, []).append(event.timestamp_us)
        elif event.event == "load_end" and event.expert is not None:
            completion[event.expert] = event.timestamp_us
        elif event.event == "evict" and event.expert is not None:
            completion.pop(event.expert, None)
        elif event.event == "expert_compute_start" and event.expert is not None:
            if event.expert not in completion or completion[event.expert] > event.timestamp_us:
                raise RuntimeError("expert used before load completion")
    if summary.perfect_overlap_bound_us > summary.no_overlap_bound_us:
        raise RuntimeError("invalid overlap bounds")
    if summary.total_time_us + 1e-9 < summary.perfect_overlap_bound_us:
        raise RuntimeError("timeline is faster than the perfect-overlap bound")
    if summary.total_time_us - 1e-9 > summary.no_overlap_bound_us:
        raise RuntimeError("timeline exceeds the no-overlap bound")
