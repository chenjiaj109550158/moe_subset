"""M9 static/dynamic oracle planning and adaptive closed-loop reference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.execution.closed_loop import _natural_rollout
from pseudoroute.execution.routing_policy import MaskedSubstitutionPolicy
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.oracle.selectors import selected_routing_mass_utility
from pseudoroute.types import ExpertKey


@dataclass(frozen=True)
class ResidencyPlan:
    boundary: int
    reason: str
    planned_horizon: int
    static_experts: tuple[ExpertKey, ...]
    dynamic_experts: tuple[ExpertKey, ...]
    load_delta: tuple[ExpertKey, ...]
    eviction_delta: tuple[ExpertKey, ...]

    @property
    def allowed(self) -> frozenset[ExpertKey]:
        return frozenset((*self.static_experts, *self.dynamic_experts))


@dataclass(frozen=True)
class TerminationEvent:
    generated_step: int
    token_position: int
    realized_window_length: int
    reason: str
    aggregate_out_of_subset_mass: float
    threshold: float


@dataclass(frozen=True)
class AdaptiveSummary:
    mode: str
    static_method: str
    static_fraction: float
    horizon: int
    generated_tokens: str
    exact_token_rate: float
    replans: int
    terminations: int
    mean_realized_horizon: float
    mean_out_of_subset_mass: float
    transfer_bytes: int


def partition_budget(
    rankings_by_layer: dict[int, tuple[int, ...]],
    *,
    total_budget: int,
    static_fraction: float,
) -> tuple[frozenset[ExpertKey], int]:
    if total_budget < 1 or not 0 <= static_fraction <= 1:
        raise ValueError("invalid residency budget or static fraction")
    static_count = min(total_budget, math.floor(total_budget * static_fraction + 1e-12))
    static = frozenset(
        ExpertKey(layer, expert)
        for layer, ranking in rankings_by_layer.items()
        for expert in ranking[:static_count]
    )
    return static, total_budget - static_count


def plan_static_dynamic(
    adapter: TinyMoEAdapter,
    prefix: Tensor,
    *,
    horizon: int,
    static: frozenset[ExpertKey],
    dynamic_budget: int,
    resident: frozenset[ExpertKey],
    reason: str,
) -> ResidencyPlan:
    window, _ = _natural_rollout(adapter, prefix, horizon)
    utilities = selected_routing_mass_utility(
        window, next(iter(adapter.spec.num_experts_by_layer.values())), gamma=1.0
    )
    dynamic: list[ExpertKey] = []
    for layer, scores in utilities.items():
        static_ids = {key.expert_idx for key in static if key.layer_idx == layer}
        candidates = sorted(
            (expert for expert in range(scores.numel()) if expert not in static_ids),
            key=lambda expert: (-float(scores[expert]), expert),
        )
        dynamic.extend(ExpertKey(layer, expert) for expert in candidates[:dynamic_budget])
    selected = frozenset((*static, *dynamic))
    evictions = tuple(sorted((resident - selected) - static))
    return ResidencyPlan(
        int(prefix.shape[1]),
        reason,
        horizon,
        tuple(sorted(static)),
        tuple(sorted(dynamic)),
        tuple(sorted(selected - resident)),
        evictions,
    )


@torch.inference_mode()
def run_static_adaptive_closed_loop(
    adapter: TinyMoEAdapter,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    horizon: int,
    total_budget: int,
    static_fraction: float,
    static_method: str,
    ranking: dict[int, tuple[int, ...]],
    adaptive: bool,
    threshold: float,
) -> tuple[AdaptiveSummary, list[ResidencyPlan], list[TerminationEvent]]:
    static, dynamic_budget = partition_budget(
        ranking, total_budget=total_budget, static_fraction=static_fraction
    )
    if total_budget < adapter.model.config.top_k:
        raise ValueError("substitution requires total budget >= model top-k")
    _, reference = _natural_rollout(adapter, prompt, max_new_tokens)
    trajectory = prompt.clone()
    generated: list[int] = []
    plans = []
    terminations = []
    out_masses = []
    realized_lengths = []
    # The first plan charges static preload bytes; later plans must retain them.
    resident: frozenset[ExpertKey] = frozenset()
    policy = MaskedSubstitutionPolicy()
    schedule: dict[int, frozenset[ExpertKey]] = {}
    next_reason = "initial"
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        planned = min(horizon, remaining)
        plan = plan_static_dynamic(
            adapter,
            trajectory,
            horizon=planned,
            static=static,
            dynamic_budget=dynamic_budget,
            resident=resident,
            reason=next_reason,
        )
        plans.append(plan)
        resident = plan.allowed
        realized = 0
        next_reason = "fixed_horizon"
        for _ in range(planned):
            position = trajectory.shape[1] - 1
            schedule[position] = plan.allowed
            output = adapter.forward_with_policy(
                trajectory, policy, allowed_experts_by_position=schedule
            )
            records = [
                record for record in output.executed_routes if record.token_position == position
            ]
            aggregate = sum(record.out_of_subset_mass for record in records) / len(records)
            out_masses.append(aggregate)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            trajectory = torch.cat((trajectory, token), dim=1)
            realized += 1
            if adaptive and aggregate > threshold and len(generated) < max_new_tokens:
                terminations.append(
                    TerminationEvent(
                        len(generated) - 1,
                        position,
                        realized,
                        "aggregate_out_of_subset_mass",
                        aggregate,
                        threshold,
                    )
                )
                next_reason = "early_termination_out_of_subset_mass"
                break
        realized_lengths.append(realized)
        if len(plans) > max_new_tokens:
            raise RuntimeError("adaptive planning exceeded one re-plan per generated token")
    reference_ids = reference.reshape(-1).tolist()
    matches = [value == target for value, target in zip(generated, reference_ids, strict=True)]
    transfer_bytes = sum(
        adapter.spec.expert_bytes[key] for plan in plans for key in plan.load_delta
    )
    return (
        AdaptiveSummary(
            "adaptive" if adaptive else "fixed",
            static_method,
            static_fraction,
            horizon,
            json.dumps(generated),
            sum(matches) / len(matches),
            len(plans),
            len(terminations),
            sum(realized_lengths) / len(realized_lengths),
            sum(out_masses) / len(out_masses),
            transfer_bytes,
        ),
        plans,
        terminations,
    )
