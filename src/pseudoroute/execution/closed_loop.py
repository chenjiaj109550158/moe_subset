"""Fixed-window oracle planning and authoritative closed-loop tiny generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from pseudoroute.execution.routing_policy import (
    ExecutedRoute,
    LosslessFallbackPolicy,
    MaskedSubstitutionPolicy,
    MaskedTruncationPolicy,
    NaturalRoutingPolicy,
    RoutingPolicy,
)
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.base import TraceLevel, TraceRequest
from pseudoroute.oracle.selectors import select_top_b, selected_routing_mass_utility
from pseudoroute.oracle.windows import OracleWindow
from pseudoroute.types import ExpertKey


@dataclass(frozen=True)
class SubsetPlan:
    decision_boundary: int
    planned_horizon: int
    subset_by_layer: dict[int, tuple[int, ...]]
    load_delta: tuple[ExpertKey, ...]
    eviction_delta: tuple[ExpertKey, ...]
    transfer_bytes: int
    information_regime: str = "oracle"

    @property
    def allowed_experts(self) -> frozenset[ExpertKey]:
        return frozenset(
            ExpertKey(layer, expert)
            for layer, experts in self.subset_by_layer.items()
            for expert in experts
        )


@dataclass(frozen=True)
class ClosedLoopRouteRow:
    policy: str
    horizon: int
    budget_ratio: float
    generated_step: int
    token_position: int
    layer_idx: int
    natural_topk_ids: str
    natural_topk_weights: str
    executed_topk_ids: str
    executed_topk_weights: str
    out_of_subset_mass: float
    fallback_used: bool
    missing_natural_experts: str


@dataclass(frozen=True)
class ClosedLoopWindowRow:
    policy: str
    horizon: int
    budget_ratio: float
    boundary: int
    planned_horizon: int
    subset_by_layer: str
    load_delta: str
    eviction_delta: str
    planned_transfer_bytes: int
    fallback_transfer_bytes: int
    planning_prefix: str


@dataclass(frozen=True)
class ClosedLoopSummary:
    policy: str
    horizon: int
    budget_ratio: float
    generated_tokens: str
    base_tokens: str
    first_divergence: int | None
    exact_token_rate: float
    exact_sequence_match: bool
    mean_next_token_kl: float
    reference_perplexity: float
    mean_out_of_subset_mass: float
    planned_transfer_bytes: int
    fallback_transfer_bytes: int
    total_transfer_bytes: int
    transfer_bytes_per_token: float
    boundary_count: int


def _natural_rollout(
    adapter: TinyMoEAdapter, prefix: Tensor, horizon: int
) -> tuple[OracleWindow, Tensor]:
    trajectory = prefix.clone()
    step_traces = []
    generated = []
    for _ in range(horizon):
        result = adapter.run_base_forward(
            trajectory, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS)
        )
        final_position = trajectory.shape[1] - 1
        traces = sorted(
            (trace for trace in result.traces if trace.token_position == final_position),
            key=lambda trace: trace.layer_idx,
        )
        step_traces.append(traces)
        token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        trajectory = torch.cat((trajectory, token), dim=1)
    topk_ids = torch.stack(
        [torch.stack([trace.topk_ids.reshape(-1) for trace in traces]) for traces in step_traces]
    )
    topk_weights = torch.stack(
        [
            torch.stack([trace.topk_weights.reshape(-1) for trace in traces])
            for traces in step_traces
        ]
    )
    router_logits = torch.stack(
        [torch.stack([trace.raw_logits.reshape(-1) for trace in traces]) for traces in step_traces]
    )
    return (
        OracleWindow(
            sample_id="closed-loop-planning",
            start=int(prefix.shape[1]),
            horizon=horizon,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=router_logits,
        ),
        torch.cat(generated, dim=1),
    )


@torch.inference_mode()
def plan_oracle_window(
    adapter: TinyMoEAdapter,
    prefix: Tensor,
    *,
    horizon: int,
    budget_ratio: float,
    gamma: float,
    resident: frozenset[ExpertKey],
) -> SubsetPlan:
    window, _ = _natural_rollout(adapter, prefix, horizon)
    budgets = {
        layer: min(
            adapter.spec.num_experts_by_layer[layer],
            max(1, math.ceil(budget_ratio * adapter.spec.top_k_by_layer[layer])),
        )
        for layer in adapter.spec.moe_layer_indices
    }
    num_experts = next(iter(adapter.spec.num_experts_by_layer.values()))
    utilities = selected_routing_mass_utility(window, num_experts, gamma=gamma)
    subsets = select_top_b(utilities, budgets)
    selected = frozenset(
        ExpertKey(layer, expert) for layer, experts in subsets.items() for expert in experts
    )
    loads = tuple(sorted(selected - resident))
    evictions = tuple(sorted(resident - selected))
    return SubsetPlan(
        decision_boundary=int(prefix.shape[1]),
        planned_horizon=horizon,
        subset_by_layer=subsets,
        load_delta=loads,
        eviction_delta=evictions,
        transfer_bytes=sum(adapter.spec.expert_bytes[key] for key in loads),
    )


def _policy(name: str) -> RoutingPolicy:
    if name == "natural":
        return NaturalRoutingPolicy()
    if name == "lossless_fallback":
        return LosslessFallbackPolicy()
    if name == "masked_substitution":
        return MaskedSubstitutionPolicy()
    if name == "masked_truncation_preserve":
        return MaskedTruncationPolicy(renormalize_weights=False)
    if name == "masked_truncation_renormalize":
        return MaskedTruncationPolicy(renormalize_weights=True)
    raise ValueError(f"unknown routing policy: {name}")


def _route_rows(
    policy_name: str,
    horizon: int,
    budget_ratio: float,
    step: int,
    position: int,
    records: tuple[ExecutedRoute, ...],
) -> list[ClosedLoopRouteRow]:
    import json

    return [
        ClosedLoopRouteRow(
            policy_name,
            horizon,
            budget_ratio,
            step,
            position,
            record.layer_idx,
            json.dumps(record.natural_topk_ids.reshape(-1).tolist()),
            json.dumps(record.natural_topk_weights.reshape(-1).tolist()),
            json.dumps(record.executed_topk_ids.reshape(-1).tolist()),
            json.dumps(record.executed_topk_weights.reshape(-1).tolist()),
            record.out_of_subset_mass,
            record.fallback_used,
            json.dumps([[key.layer_idx, key.expert_idx] for key in record.missing_natural_experts]),
        )
        for record in records
        if record.token_position == position
    ]


@torch.inference_mode()
def evaluate_closed_loop(
    adapter: TinyMoEAdapter,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    horizon: int,
    budget_ratio: float,
    gamma: float,
    policy_name: str,
) -> tuple[ClosedLoopSummary, list[ClosedLoopRouteRow], list[ClosedLoopWindowRow]]:
    import json

    if max_new_tokens < 1:
        raise ValueError("closed-loop evaluation requires generated tokens")
    _, base_tokens = _natural_rollout(adapter, prompt, max_new_tokens)
    trajectory = prompt.clone()
    generated: list[int] = []
    route_rows: list[ClosedLoopRouteRow] = []
    window_rows: list[ClosedLoopWindowRow] = []
    resident: frozenset[ExpertKey] = frozenset()
    planned_bytes = 0
    fallback_bytes = 0
    kls: list[float] = []
    reference_nlls: list[float] = []
    out_masses: list[float] = []
    policy = _policy(policy_name)
    policy_schedule: dict[int, frozenset[ExpertKey]] = {}
    while len(generated) < max_new_tokens:
        realized_horizon = min(horizon, max_new_tokens - len(generated))
        planning_prefix = trajectory.reshape(-1).tolist()
        plan = plan_oracle_window(
            adapter,
            trajectory,
            horizon=realized_horizon,
            budget_ratio=budget_ratio,
            gamma=gamma,
            resident=resident,
        )
        planned_bytes += plan.transfer_bytes
        window_fallback = 0
        for _ in range(realized_horizon):
            natural = adapter.run_base_forward(trajectory)
            position = trajectory.shape[1] - 1
            policy_schedule[position] = plan.allowed_experts
            executed = adapter.forward_with_policy(
                trajectory, policy, allowed_experts_by_position=policy_schedule
            )
            final_records = tuple(
                record for record in executed.executed_routes if record.token_position == position
            )
            for record in final_records:
                out_masses.append(record.out_of_subset_mass)
                if record.fallback_used:
                    transferred = sum(
                        adapter.spec.expert_bytes[key] for key in record.missing_natural_experts
                    )
                    window_fallback += transferred
                    fallback_bytes += transferred
            natural_log_probs = natural.logits[:, -1].log_softmax(dim=-1)
            executed_log_probs = executed.logits[:, -1].log_softmax(dim=-1)
            natural_probs = natural_log_probs.exp()
            kls.append(
                float((natural_probs * (natural_log_probs - executed_log_probs)).sum(dim=-1).mean())
            )
            reference = base_tokens[:, len(generated)]
            reference_nlls.append(float(-executed_log_probs.gather(-1, reference[:, None]).mean()))
            token = executed.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            route_rows.extend(
                _route_rows(
                    policy_name,
                    horizon,
                    budget_ratio,
                    len(generated) - 1,
                    position,
                    final_records,
                )
            )
            trajectory = torch.cat((trajectory, token), dim=1)
        window_rows.append(
            ClosedLoopWindowRow(
                policy_name,
                horizon,
                budget_ratio,
                plan.decision_boundary,
                realized_horizon,
                json.dumps(plan.subset_by_layer, sort_keys=True),
                json.dumps([[key.layer_idx, key.expert_idx] for key in plan.load_delta]),
                json.dumps([[key.layer_idx, key.expert_idx] for key in plan.eviction_delta]),
                plan.transfer_bytes,
                window_fallback,
                json.dumps(planning_prefix),
            )
        )
        resident = plan.allowed_experts
    base = base_tokens.reshape(-1).tolist()
    matches = [actual == expected for actual, expected in zip(generated, base, strict=True)]
    divergence = next((index for index, match in enumerate(matches) if not match), None)
    total_bytes = planned_bytes + fallback_bytes
    summary = ClosedLoopSummary(
        policy_name,
        horizon,
        budget_ratio,
        json.dumps(generated),
        json.dumps(base),
        divergence,
        sum(matches) / len(matches),
        all(matches),
        sum(kls) / len(kls),
        math.exp(sum(reference_nlls) / len(reference_nlls)),
        sum(out_masses) / len(out_masses),
        planned_bytes,
        fallback_bytes,
        total_bytes,
        total_bytes / max_new_tokens,
        len(window_rows),
    )
    return summary, route_rows, window_rows


def summary_dict(summary: ClosedLoopSummary) -> dict[str, object]:
    return asdict(summary)
