from __future__ import annotations

from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_report import (
    Aggregate,
    _development_gate,
    _paired_bootstrap,
)

CONFIG = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml")


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


def test_development_gate_records_negative_oracle_gap_recovery() -> None:
    suite = load_pseudo_embedding_config(CONFIG)
    aggregates: dict[str, dict[str, object]] = {
        "hard_oracle_commitment": {
            "mean_route_hit": 0.95,
            "mean_selected_mass": 0.97,
            "estimated_transfer_reduction": 0.7,
        },
        "previous_route_commitment": {
            "mean_route_hit": 0.61,
            "mean_selected_mass": 0.63,
            "estimated_transfer_reduction": 0.36,
        },
        "static_frequency": {
            "mean_route_hit": 0.34,
            "mean_selected_mass": 0.35,
            "estimated_transfer_reduction": 0.31,
        },
    }
    for variant in suite.variants:
        aggregates[variant.key] = {
            "mean_route_hit": 0.56,
            "mean_selected_mass": 0.58,
            "estimated_transfer_reduction": 0.35,
        }

    costs = {
        variant.key: {
            "boundaries": 1,
            "mean_probe_latency_seconds_measured": 1.0,
        }
        for variant in suite.variants
    }
    gate = _development_gate(
        suite,
        aggregates,
        costs,
        {"all_required_invariants_pass": True},
    )

    assert gate["decision"] == "STOP/PIVOT"
    candidate = gate["candidates"][0]
    assert candidate["route_hit_improvement_over_previous"] == pytest.approx(-0.05)
    assert candidate["selected_mass_improvement_over_previous"] == pytest.approx(-0.05)
    assert candidate["development_route_hit_gap_recovery"] < 0
    assert candidate["development_selected_mass_gap_recovery"] < 0
    assert candidate["eligible"] is False
