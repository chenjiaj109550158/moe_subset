import itertools

import torch

from pseudoroute.oracle.selectors import (
    OracleSelector,
    cost_aware_utilities,
    oracle_utilities,
    select_top_b,
)
from pseudoroute.oracle.windows import OracleWindow
from pseudoroute.types import ExpertKey


def window() -> OracleWindow:
    probabilities = torch.tensor(
        [
            [[0.55, 0.25, 0.15, 0.05], [0.10, 0.20, 0.30, 0.40]],
            [[0.10, 0.60, 0.20, 0.10], [0.45, 0.15, 0.30, 0.10]],
            [[0.20, 0.10, 0.65, 0.05], [0.05, 0.60, 0.10, 0.25]],
        ],
        dtype=torch.float64,
    )
    topk_scores, topk_ids = probabilities.topk(2, dim=-1)
    topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    return OracleWindow(
        sample_id="synthetic",
        start=0,
        horizon=3,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        router_logits=probabilities.log(),
    )


def brute_force_objective(scores: torch.Tensor, budget: int) -> float:
    return max(
        sum(float(scores[index]) for index in subset)
        for subset in itertools.combinations(range(scores.numel()), budget)
    )


def test_every_additive_oracle_matches_brute_force() -> None:
    for selector in OracleSelector:
        utilities = oracle_utilities(
            window(),
            selector,
            num_experts=4,
            gamma=0.9,
            load_costs={
                ExpertKey(layer, expert): float(expert + 1)
                for layer in range(2)
                for expert in range(4)
            },
            load_cost_lambda=0.05,
        )
        budgets = {0: 2, 1: 2}
        selected = select_top_b(utilities, budgets)
        for layer, subset in selected.items():
            achieved = sum(float(utilities[layer][expert]) for expert in subset)
            assert achieved == brute_force_objective(utilities[layer], budgets[layer])


def test_cost_awareness_prefers_resident_or_cheaper_expert() -> None:
    base = {0: torch.tensor([1.0, 0.99, 0.98])}
    adjusted = cost_aware_utilities(
        base,
        resident=frozenset({ExpertKey(0, 2)}),
        load_costs={ExpertKey(0, 0): 10.0, ExpertKey(0, 1): 1.0},
        load_cost_lambda=0.1,
    )
    assert select_top_b(adjusted, {0: 1}) == {0: (2,)}


def test_budget_is_enforced_and_invalid_budget_fails() -> None:
    selected = select_top_b({0: torch.tensor([3.0, 2.0, 1.0])}, {0: 2})
    assert len(selected[0]) == 2
    try:
        select_top_b({0: torch.ones(2)}, {0: 3})
    except ValueError as error:
        assert "invalid budget" in str(error)
    else:
        raise AssertionError("invalid budget was accepted")
