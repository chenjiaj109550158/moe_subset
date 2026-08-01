from __future__ import annotations

import pytest
import torch

from pseudoroute.benchmark.pseudo_embedding_calibration_free_analysis import (
    Aggregate,
    _top_b,
    calibration_free_subsets,
)


def test_top_b_uses_ascending_expert_id_tie_break() -> None:
    assert _top_b(torch.ones(8), 3) == (0, 1, 2)


def test_calibration_free_candidates_use_zero_pseudo_fallback_at_boundary_zero() -> None:
    probabilities = torch.zeros(2, 8, 128)
    probabilities[:, :, 127] = 1.0
    ids = torch.arange(8).reshape(1, 1, 8).expand(8, 2, 8)
    weights = torch.full((8, 2, 8), 0.125)
    candidates = calibration_free_subsets(probabilities, ids, weights, 0)
    observed = {layers[0] for layers in candidates.values()}
    assert len(observed) == 1
    assert 127 in observed.pop()


def test_one_known_plus_history_uses_horizon_derived_weight() -> None:
    probabilities = torch.zeros(1, 8, 128)
    probabilities[0, 0, 127] = 1.0
    ids = torch.arange(8).reshape(1, 1, 8).expand(8, 1, 8)
    weights = torch.full((8, 1, 8), 0.125)
    candidates = calibration_free_subsets(probabilities, ids, weights, 8)
    subset = candidates["one_known_plus_seven_history"][0]
    assert set(range(8)).issubset(subset)
    assert 127 in subset


def test_aggregate_separates_route_count_and_selected_mass() -> None:
    aggregate = Aggregate()
    aggregate.update(
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0.5, 0.25, 0.15, 0.1]]),
        (0, 1, 3),
        (),
    )
    row = aggregate.as_dict()
    assert row["mean_route_hit"] == 0.75
    assert row["mean_selected_mass"] == pytest.approx(0.85)
