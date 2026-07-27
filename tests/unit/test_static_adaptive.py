import torch

from pseudoroute.analysis.static_residency import calibrate_static_signals, rankings
from pseudoroute.config import load_config
from pseudoroute.execution.adaptive import (
    ResidencyPlan,
    partition_budget,
    plan_static_dynamic,
    run_static_adaptive_closed_loop,
)
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.simulation.reference import simulate_reference_timeline
from pseudoroute.types import ExpertKey


def _adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=23, device="cpu"))


def _documents() -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        ("a", torch.tensor([[1, 2, 3, 4]])),
        ("b", torch.tensor([[5, 6, 7, 8]])),
    )


def test_static_rankings_have_every_signal_and_deterministic_experts() -> None:
    adapter = _adapter()
    signals = calibrate_static_signals(adapter, _documents(), max_interventions=4, seed=3)
    result = rankings(signals)
    assert set(result) == {
        "frequency",
        "covariance_variance",
        "quality_sensitivity",
        "downstream_influence",
        "miss_risk",
        "combined",
    }
    assert all(
        sorted(order) == [0, 1, 2, 3] for method in result.values() for order in method.values()
    )
    assert torch.isfinite(signals.combined).all()


def test_static_dynamic_partition_respects_remaining_capacity() -> None:
    ranking = {0: (0, 1, 2, 3), 1: (3, 2, 1, 0)}
    static, dynamic_budget = partition_budget(ranking, total_budget=3, static_fraction=0.5)
    assert static == frozenset({ExpertKey(0, 0), ExpertKey(1, 3)})
    assert dynamic_budget == 2
    adapter = _adapter()
    plan = plan_static_dynamic(
        adapter,
        torch.tensor([[1, 2]]),
        horizon=2,
        static=static,
        dynamic_budget=dynamic_budget,
        resident=static,
        reason="initial",
    )
    for layer in range(2):
        assert len([key for key in plan.allowed if key.layer_idx == layer]) == 3
    assert static <= plan.allowed
    assert not set(plan.eviction_delta) & set(static)


def test_adaptive_termination_replans_at_next_token_and_cannot_loop() -> None:
    adapter = _adapter()
    ranking = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    summary, plans, terminations = run_static_adaptive_closed_loop(
        adapter,
        torch.tensor([[1, 2]]),
        max_new_tokens=4,
        horizon=3,
        total_budget=2,
        static_fraction=0.5,
        static_method="frequency",
        ranking=ranking,
        adaptive=True,
        threshold=-1.0,
    )
    assert summary.replans == 4
    assert summary.terminations == 3
    assert [plan.boundary for plan in plans] == [2, 3, 4, 5]
    assert all(event.reason == "aggregate_out_of_subset_mass" for event in terminations)
    assert all(event.realized_window_length == 1 for event in terminations)
    assert summary.mean_realized_horizon == 1.0


def test_fixed_horizon_has_no_termination_events() -> None:
    adapter = _adapter()
    ranking = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}
    summary, plans, terminations = run_static_adaptive_closed_loop(
        adapter,
        torch.tensor([[1, 2]]),
        max_new_tokens=5,
        horizon=2,
        total_budget=3,
        static_fraction=1 / 3,
        static_method="combined",
        ranking=ranking,
        adaptive=False,
        threshold=0.0,
    )
    assert not terminations
    assert summary.replans == 3
    assert [plan.boundary for plan in plans] == [2, 4, 6]


def test_reference_simulator_bytes_and_static_invariant() -> None:
    adapter = _adapter()
    static = (ExpertKey(0, 0), ExpertKey(1, 0))
    plan = ResidencyPlan(
        2,
        "initial",
        2,
        static,
        (ExpertKey(0, 1), ExpertKey(1, 1)),
        (*static, ExpertKey(0, 1), ExpertKey(1, 1)),
        (),
    )
    summary, events = simulate_reference_timeline(
        adapter.spec,
        [plan],
        mode="fixed",
        generated_tokens=2,
        bandwidth_bytes_per_us=1024,
        fixed_latency_us=10,
        compute_us_per_token=5,
    )
    expected_bytes = sum(adapter.spec.expert_bytes[key] for key in plan.load_delta)
    assert summary.transfer_bytes == expected_bytes
    assert summary.transfer_time_us == 10 + expected_bytes / 1024
    assert summary.total_time_us == summary.transfer_time_us + 10
    assert [event.event for event in events] == ["replan", "load_start", "load_end"]
