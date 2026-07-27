import itertools

import torch

from pseudoroute.oracle.planner import (
    adjusted_scores,
    adjusted_top_b,
    aggregate_window_utility,
    exact_knapsack,
    make_subset_plan,
)
from pseudoroute.types import ExpertKey


def test_interval_utility_and_uncertainty_are_exact() -> None:
    probabilities = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.5, 0.5]],
        ]
    )
    mean, uncertainty, score = aggregate_window_utility(
        probabilities, (1, 3), gamma=1.0, uncertainty_beta=2.0
    )
    assert torch.equal(mean, torch.tensor([1.5, 1.5], dtype=torch.float64))
    assert torch.equal(uncertainty, torch.tensor([0.5, 0.5], dtype=torch.float64))
    assert torch.equal(score, torch.tensor([2.5, 2.5], dtype=torch.float64))


def test_adjusted_top_b_honors_static_and_transfer_penalties() -> None:
    utility = {0: torch.tensor([5.0, 4.0, 3.0])}
    resident = frozenset({ExpertKey(0, 1)})
    adjusted = adjusted_scores(
        utility,
        resident=resident,
        selected_previous=frozenset({ExpertKey(0, 1)}),
        load_costs={ExpertKey(0, 0): 10.0, ExpertKey(0, 2): 1.0},
        eviction_costs={},
        load_lambda=1.0,
        eviction_lambda=0.0,
    )
    selected = adjusted_top_b(adjusted, {0: 2}, mandatory=frozenset({ExpertKey(0, 2)}))
    assert selected == {0: (1, 2)}


def test_exact_knapsack_matches_brute_force_with_variable_sizes() -> None:
    keys = tuple(ExpertKey(0, index) for index in range(5))
    values = {key: value for key, value in zip(keys, (8.0, 7.0, 6.0, 5.0, 4.0), strict=True)}
    sizes = {key: value for key, value in zip(keys, (6, 5, 4, 3, 2), strict=True)}
    selected = exact_knapsack(values, sizes, capacity_bytes=10, mandatory=frozenset({keys[4]}))
    brute = max(
        (
            (sum(values[key] for key in subset), tuple(sorted(subset)))
            for count in range(len(keys) + 1)
            for subset in itertools.combinations(keys, count)
            if keys[4] in subset and sum(sizes[key] for key in subset) <= 10
        ),
        key=lambda item: (item[0], tuple(reversed(item[1]))),
    )[1]
    assert selected == brute


def test_subset_plan_has_exact_load_and_eviction_deltas() -> None:
    sizes = {ExpertKey(0, index): (index + 1) * 10 for index in range(3)}
    plan = make_subset_plan(
        {0: (0, 2)},
        {0: torch.tensor([1.0, 2.0, 3.0])},
        resident=frozenset({ExpertKey(0, 0), ExpertKey(0, 1)}),
        static=frozenset({ExpertKey(0, 0)}),
        expert_bytes=sizes,
    )
    assert plan.load_delta == (ExpertKey(0, 2),)
    assert plan.eviction_delta == (ExpertKey(0, 1),)
    assert plan.selected_bytes == 40
    assert plan.load_bytes == 30
    assert plan.eviction_bytes == 20
