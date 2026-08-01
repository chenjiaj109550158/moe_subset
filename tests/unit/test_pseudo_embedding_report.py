from __future__ import annotations

import pytest

from pseudoroute.benchmark.pseudo_embedding_report import (
    Aggregate,
    _paired_bootstrap,
)


def _row(hit: int, mass: float) -> dict[str, object]:
    return {
        "route_hits": hit,
        "route_slots": 8,
        "selected_mass_hit": mass,
        "selected_mass_total": 1.0,
        "full_mass_hit": mass,
        "full_mass_total": 1.0,
        "fallback_loads": 8 - hit,
        "prefetch_loads": 2,
        "subset_churn": 4,
        "lossless_transfer_bytes": 10,
        "natural_reference_bytes": 20,
    }


def test_route_aggregate_uses_count_and_mass_denominators() -> None:
    state = Aggregate()
    state.update(_row(6, 0.8))
    state.update(_row(4, 0.6))
    row = state.as_dict()
    assert row["mean_route_hit"] == 10 / 16
    assert row["mean_selected_mass"] == 0.7
    assert row["fallback_frequency"] == 6 / 16
    assert row["estimated_transfer_reduction"] == 0.5


def test_paired_bootstrap_is_sample_paired_and_deterministic() -> None:
    values = {
        "pseudo": {"a": (0.8, 0.9), "b": (0.6, 0.7)},
        "previous": {"a": (0.7, 0.8), "b": (0.5, 0.6)},
    }
    first = _paired_bootstrap(
        values,
        "pseudo",
        "previous",
        samples=100,
        seed=17,
    )
    second = _paired_bootstrap(
        values,
        "pseudo",
        "previous",
        samples=100,
        seed=17,
    )
    assert first == second
    assert first["mean_route_hit_difference"] == pytest.approx(0.1)
    assert first["mean_selected_mass_difference"] == pytest.approx(0.1)
