import pytest

from pseudoroute.simulation.engine import RouteToken, SimHardware, simulate
from pseudoroute.types import ExpertKey


def _hardware(*, overlap: bool = True, capacity: int = 200, concurrent: int = 2) -> SimHardware:
    return SimHardware(capacity, 10_000_000, 5.0, concurrent, 10.0, 2.0, overlap)


def test_synthetic_on_demand_timeline_and_bounds_are_exact() -> None:
    key = ExpertKey(0, 0)
    summary, events = simulate(
        (RouteToken(0, ((key,),)),),
        {key: 100},
        _hardware(capacity=100, concurrent=1),
        baseline="on_demand",
        horizon=1,
        per_layer_budget=1,
        probe_us=0,
    )
    assert summary.total_time_us == 27.0
    assert summary.transfer_bytes == 100
    assert summary.no_overlap_bound_us == 27.0
    assert summary.perfect_overlap_bound_us == 12.0
    assert [event.event for event in events if event.event.startswith("expert_compute")] == [
        "expert_compute_start",
        "expert_compute_end",
    ]


def test_concurrent_prefetch_shares_aggregate_bandwidth() -> None:
    keys = (ExpertKey(0, 0), ExpertKey(0, 1))
    summary, events = simulate(
        (RouteToken(0, (keys,)),),
        {key: 100 for key in keys},
        _hardware(),
        baseline="one_step_commitment",
        horizon=1,
        per_layer_budget=2,
        probe_us=0,
    )
    starts = [event for event in events if event.event == "load_start"]
    ends = [event for event in events if event.event == "load_end"]
    assert {event.timestamp_us for event in starts} == {0.0}
    assert {event.timestamp_us for event in ends} == {25.0}
    assert summary.exposed_stall_us == 15.0
    assert summary.peak_resident_bytes == 200


def test_cache_capacity_and_load_completion_invariants() -> None:
    keys = tuple(ExpertKey(0, index) for index in range(3))
    routes = tuple(RouteToken(index, ((key,),)) for index, key in enumerate(keys))
    summary, events = simulate(
        routes,
        {key: 100 for key in keys},
        _hardware(capacity=100, concurrent=1),
        baseline="lru",
        horizon=1,
        per_layer_budget=1,
        probe_us=0,
    )
    assert summary.peak_resident_bytes <= 100
    completion = {}
    for event in events:
        if event.event == "load_end":
            completion[event.expert] = event.timestamp_us
        if event.event == "expert_compute_start":
            assert completion[event.expert] <= event.timestamp_us


def test_lossless_predictor_falls_back_for_every_missing_expert() -> None:
    first, second = ExpertKey(0, 0), ExpertKey(0, 1)
    routes = (RouteToken(0, ((first,),)), RouteToken(1, ((second,),)))
    summary, events = simulate(
        routes,
        {first: 100, second: 100},
        _hardware(capacity=100, concurrent=1),
        baseline="lossless_predictor",
        horizon=1,
        per_layer_budget=1,
        probe_us=0,
    )
    assert summary.route_coverage == 1.0
    assert any(event.reason == "demand_load" and event.expert == second for event in events)


def test_impossible_working_set_fails_capacity_instead_of_overcommitting() -> None:
    keys = (ExpertKey(0, 0), ExpertKey(0, 1))
    with pytest.raises(RuntimeError, match="capacity"):
        simulate(
            (RouteToken(0, (keys,)),),
            {key: 100 for key in keys},
            _hardware(capacity=100, concurrent=1),
            baseline="lru",
            horizon=1,
            per_layer_budget=2,
            probe_us=0,
        )
