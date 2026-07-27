"""M8 uncertainty- and transfer-aware utility aggregation and subset planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.types import ExpertKey


@dataclass(frozen=True)
class AggregatedUtility:
    mean: dict[int, Tensor]
    uncertainty: dict[int, Tensor]
    score: dict[int, Tensor]


@dataclass(frozen=True)
class CostAwareSubsetPlan:
    subset_by_layer: dict[int, tuple[int, ...]]
    static_members: tuple[ExpertKey, ...]
    dynamic_members: tuple[ExpertKey, ...]
    load_delta: tuple[ExpertKey, ...]
    eviction_delta: tuple[ExpertKey, ...]
    selected_bytes: int
    load_bytes: int
    eviction_bytes: int
    predicted_utility: float


def aggregate_window_utility(
    branch_probabilities: Tensor,
    anchors: tuple[int, ...],
    *,
    gamma: float,
    uncertainty_beta: float,
    uncertainty_mode: str = "mean_plus_std",
) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate ``[branches, anchors, experts]`` using interval quadrature."""
    if branch_probabilities.ndim != 3 or branch_probabilities.shape[1] != len(anchors):
        raise ValueError("branch probabilities must have shape [branches, anchors, experts]")
    if not anchors or tuple(sorted(set(anchors))) != anchors or anchors[0] < 1:
        raise ValueError("anchors must be unique, sorted, and positive")
    if not 0 < gamma <= 1 or uncertainty_beta < 0:
        raise ValueError("invalid aggregation parameters")
    intervals = torch.tensor(
        [anchors[0], *(right - left for left, right in zip(anchors, anchors[1:], strict=False))],
        dtype=torch.float64,
        device=branch_probabilities.device,
    )
    discounts = torch.tensor(
        [gamma ** (anchor - 1) for anchor in anchors],
        dtype=torch.float64,
        device=branch_probabilities.device,
    )
    branch_utility = (branch_probabilities.double() * (intervals * discounts)[None, :, None]).sum(
        dim=1
    )
    mean = branch_utility.mean(dim=0)
    uncertainty = branch_utility.std(dim=0, unbiased=False)
    if uncertainty_mode == "mean_plus_std":
        score = mean + uncertainty_beta * uncertainty
    elif uncertainty_mode == "mean_minus_std":
        score = mean - uncertainty_beta * uncertainty
    else:
        raise ValueError("uncertainty mode must be mean_plus_std or mean_minus_std")
    return mean, uncertainty, score


def adjusted_scores(
    utility: dict[int, Tensor],
    *,
    resident: frozenset[ExpertKey],
    selected_previous: frozenset[ExpertKey],
    load_costs: dict[ExpertKey, float],
    eviction_costs: dict[ExpertKey, float],
    load_lambda: float,
    eviction_lambda: float,
) -> dict[int, Tensor]:
    if load_lambda < 0 or eviction_lambda < 0:
        raise ValueError("cost penalties must be non-negative")
    result = {}
    for layer, values in utility.items():
        scores = values.double().clone()
        for expert in range(values.numel()):
            key = ExpertKey(layer, expert)
            if key not in resident:
                scores[expert] -= load_lambda * load_costs.get(key, 0.0)
            if key in resident and key not in selected_previous:
                scores[expert] -= eviction_lambda * eviction_costs.get(key, 0.0)
        result[layer] = scores
    return result


def adjusted_top_b(
    scores: dict[int, Tensor],
    budgets: dict[int, int],
    *,
    mandatory: frozenset[ExpertKey] = frozenset(),
) -> dict[int, tuple[int, ...]]:
    selected = {}
    for layer, values in scores.items():
        required = sorted(key.expert_idx for key in mandatory if key.layer_idx == layer)
        if len(required) > budgets[layer]:
            raise ValueError("mandatory experts exceed layer budget")
        remaining = sorted(
            (expert for expert in range(values.numel()) if expert not in set(required)),
            key=lambda expert: (-float(values[expert]), expert),
        )
        selected[layer] = tuple(sorted((*required, *remaining[: budgets[layer] - len(required)])))
    return selected


def exact_knapsack(
    values: dict[ExpertKey, float],
    sizes: dict[ExpertKey, int],
    *,
    capacity_bytes: int,
    mandatory: frozenset[ExpertKey] = frozenset(),
) -> tuple[ExpertKey, ...]:
    if capacity_bytes < 0 or set(values) != set(sizes):
        raise ValueError("invalid capacity or mismatched knapsack inputs")
    mandatory_bytes = sum(sizes[key] for key in mandatory)
    if mandatory_bytes > capacity_bytes:
        raise ValueError("mandatory experts exceed byte capacity")
    candidates = sorted(set(values) - mandatory)
    states: dict[int, tuple[float, tuple[ExpertKey, ...]]] = {mandatory_bytes: (0.0, ())}
    for key in candidates:
        additions: dict[int, tuple[float, tuple[ExpertKey, ...]]] = {}
        for used, (score, chosen) in states.items():
            next_used = used + sizes[key]
            if next_used <= capacity_bytes:
                proposal = (score + values[key], (*chosen, key))
                current = states.get(next_used) or additions.get(next_used)
                if (
                    current is None
                    or proposal[0] > current[0]
                    or (proposal[0] == current[0] and proposal[1] < current[1])
                ):
                    additions[next_used] = proposal
        for used, proposal in additions.items():
            current = states.get(used)
            if (
                current is None
                or proposal[0] > current[0]
                or (proposal[0] == current[0] and proposal[1] < current[1])
            ):
                states[used] = proposal
    _, (_, chosen) = max(
        states.items(), key=lambda item: (item[1][0], -item[0], tuple(reversed(item[1][1])))
    )
    return tuple(sorted((*mandatory, *chosen)))


def greedy_knapsack(
    values: dict[ExpertKey, float],
    sizes: dict[ExpertKey, int],
    *,
    capacity_bytes: int,
    mandatory: frozenset[ExpertKey] = frozenset(),
) -> tuple[ExpertKey, ...]:
    chosen = list(sorted(mandatory))
    used = sum(sizes[key] for key in chosen)
    for key in sorted(
        set(values) - mandatory,
        key=lambda item: (-values[item] / sizes[item], item),
    ):
        if used + sizes[key] <= capacity_bytes:
            chosen.append(key)
            used += sizes[key]
    return tuple(sorted(chosen))


def make_subset_plan(
    subset_by_layer: dict[int, tuple[int, ...]],
    utility: dict[int, Tensor],
    *,
    resident: frozenset[ExpertKey],
    static: frozenset[ExpertKey],
    expert_bytes: dict[ExpertKey, int],
) -> CostAwareSubsetPlan:
    selected = frozenset(
        ExpertKey(layer, expert) for layer, experts in subset_by_layer.items() for expert in experts
    )
    if not static <= selected:
        raise ValueError("subset omits mandatory static experts")
    loads = tuple(sorted(selected - resident))
    evictions = tuple(sorted((resident - selected) - static))
    return CostAwareSubsetPlan(
        subset_by_layer,
        tuple(sorted(static)),
        tuple(sorted(selected - static)),
        loads,
        evictions,
        sum(expert_bytes[key] for key in selected),
        sum(expert_bytes[key] for key in loads),
        sum(expert_bytes[key] for key in evictions),
        sum(float(utility[key.layer_idx][key.expert_idx]) for key in selected),
    )
