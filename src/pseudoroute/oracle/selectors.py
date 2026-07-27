"""Additive oracle utilities and deterministic budgeted selection."""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import Tensor

from pseudoroute.oracle.windows import OracleWindow
from pseudoroute.types import ExpertKey


class OracleSelector(StrEnum):
    BINARY_COUNT = "binary_count"
    SELECTED_ROUTING_MASS = "selected_routing_mass"
    FULL_ROUTER_MASS = "full_router_mass"
    COST_AWARE_SELECTED_MASS = "cost_aware_selected_mass"


def _discounts(horizon: int, gamma: float, *, device: torch.device) -> Tensor:
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    return torch.tensor([gamma**offset for offset in range(horizon)], device=device)


def binary_count_utility(window: OracleWindow, num_experts: int) -> dict[int, Tensor]:
    utilities: dict[int, Tensor] = {}
    for layer in range(window.topk_ids.shape[1]):
        scores = torch.zeros(num_experts, dtype=torch.float64)
        ids = window.topk_ids[:, layer].reshape(-1).cpu()
        scores.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
        utilities[layer] = scores
    return utilities


def selected_routing_mass_utility(
    window: OracleWindow, num_experts: int, *, gamma: float
) -> dict[int, Tensor]:
    discounts = _discounts(window.horizon, gamma, device=window.topk_weights.device)
    utilities: dict[int, Tensor] = {}
    for layer in range(window.topk_ids.shape[1]):
        scores = torch.zeros(num_experts, dtype=torch.float64)
        ids = window.topk_ids[:, layer].reshape(-1).cpu()
        weighted = (window.topk_weights[:, layer] * discounts[:, None]).reshape(-1).double().cpu()
        scores.scatter_add_(0, ids, weighted)
        utilities[layer] = scores
    return utilities


def full_router_mass_utility(window: OracleWindow, *, gamma: float) -> dict[int, Tensor]:
    if window.router_logits is None:
        raise ValueError("full-router-mass oracle requires router logits")
    discounts = _discounts(window.horizon, gamma, device=window.router_logits.device)
    probabilities = window.router_logits.double().softmax(dim=-1)
    return {
        layer: (probabilities[:, layer] * discounts[:, None]).sum(dim=0).cpu()
        for layer in range(probabilities.shape[1])
    }


def select_top_b(
    utilities: dict[int, Tensor], budgets: dict[int, int]
) -> dict[int, tuple[int, ...]]:
    selected: dict[int, tuple[int, ...]] = {}
    if set(utilities) != set(budgets):
        raise ValueError("utility and budget layers must match")
    for layer, scores in utilities.items():
        budget = budgets[layer]
        if budget < 1 or budget > scores.numel():
            raise ValueError(f"invalid budget {budget} for layer {layer}")
        ranking = sorted(range(scores.numel()), key=lambda expert: (-float(scores[expert]), expert))
        selected[layer] = tuple(sorted(ranking[:budget]))
    return selected


def cost_aware_utilities(
    utilities: dict[int, Tensor],
    *,
    resident: frozenset[ExpertKey],
    load_costs: dict[ExpertKey, float],
    load_cost_lambda: float,
) -> dict[int, Tensor]:
    if load_cost_lambda < 0:
        raise ValueError("load_cost_lambda must be non-negative")
    adjusted: dict[int, Tensor] = {}
    for layer, scores in utilities.items():
        values = scores.clone()
        for expert in range(scores.numel()):
            key = ExpertKey(layer, expert)
            if key not in resident:
                values[expert] -= load_cost_lambda * load_costs.get(key, 0.0)
        adjusted[layer] = values
    return adjusted


def oracle_utilities(
    window: OracleWindow,
    selector: OracleSelector,
    *,
    num_experts: int,
    gamma: float,
    resident: frozenset[ExpertKey] = frozenset(),
    load_costs: dict[ExpertKey, float] | None = None,
    load_cost_lambda: float = 0.0,
) -> dict[int, Tensor]:
    if selector is OracleSelector.BINARY_COUNT:
        return binary_count_utility(window, num_experts)
    if selector is OracleSelector.FULL_ROUTER_MASS:
        return full_router_mass_utility(window, gamma=gamma)
    mass = selected_routing_mass_utility(window, num_experts, gamma=gamma)
    if selector is OracleSelector.SELECTED_ROUTING_MASS:
        return mass
    if selector is OracleSelector.COST_AWARE_SELECTED_MASS:
        return cost_aware_utilities(
            mass,
            resident=resident,
            load_costs=load_costs or {},
            load_cost_lambda=load_cost_lambda,
        )
    raise ValueError(f"unsupported oracle selector: {selector}")
