"""Sampled non-mutating expert interventions for the deterministic tiny model."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.execution import MaskedSubstitutionPolicy, MaskedTruncationPolicy
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import LayerActivationTrace
from pseudoroute.types import ExpertKey


@dataclass(frozen=True)
class InterventionTarget:
    sample_id: str
    token_position: int
    layer_idx: int
    expert_idx: int


@dataclass(frozen=True)
class InterventionRow:
    information_regime: str
    sample_id: str
    token_position: int
    layer_idx: int
    expert_idx: int
    intervention: str
    immediate_output_l2: float
    immediate_output_relative_l2: float
    next_token_kl: float
    downstream_positions: int
    downstream_topk_flip_count: int
    downstream_topk_flip_rate: float
    downstream_router_kl: float


def _activation_map(
    activations: tuple[LayerActivationTrace, ...],
) -> dict[tuple[int, int], LayerActivationTrace]:
    return {(item.token_position, item.layer_idx): item for item in activations}


def sample_active_targets(
    adapter: TinyMoEAdapter,
    samples: tuple[tuple[str, Tensor], ...],
    *,
    max_targets: int,
    seed: int,
) -> tuple[InterventionTarget, ...]:
    candidates = []
    with torch.inference_mode():
        for sample_id, token_ids in samples:
            output = adapter.model(token_ids, capture_trace=True)
            for trace in output.traces:
                for expert in trace.topk_ids.reshape(-1).tolist():
                    candidates.append(
                        InterventionTarget(
                            sample_id, trace.token_position, trace.layer_idx, int(expert)
                        )
                    )
    unique = list(dict.fromkeys(candidates))
    random.Random(seed).shuffle(unique)
    return tuple(unique[:max_targets])


def _policy(name: str) -> MaskedSubstitutionPolicy | MaskedTruncationPolicy:
    if name == "zero_selected_contribution":
        return MaskedTruncationPolicy(renormalize_weights=False)
    if name == "remove_and_renormalize":
        return MaskedTruncationPolicy(renormalize_weights=True)
    if name == "substitute_next_available":
        return MaskedSubstitutionPolicy()
    raise ValueError(f"unknown intervention: {name}")


def run_intervention(
    adapter: TinyMoEAdapter,
    sample_id: str,
    token_ids: Tensor,
    target: InterventionTarget,
    *,
    intervention: str,
) -> InterventionRow:
    if target.sample_id != sample_id:
        raise ValueError("intervention target belongs to another sample")
    all_experts = frozenset(adapter.spec.expert_bytes)
    excluded = ExpertKey(target.layer_idx, target.expert_idx)
    allowed = all_experts - {excluded}
    schedule = {target.token_position: allowed}
    with torch.inference_mode():
        baseline = adapter.model(token_ids, capture_trace=True, capture_activations=True)
        changed = adapter.model(
            token_ids,
            capture_trace=True,
            capture_activations=True,
            policy=_policy(intervention),
            allowed_experts_by_position=schedule,
        )
    baseline_activations = _activation_map(baseline.activations)
    changed_activations = _activation_map(changed.activations)
    base_state = baseline_activations[(target.token_position, target.layer_idx)].post_moe_state
    changed_state = changed_activations[(target.token_position, target.layer_idx)].post_moe_state
    if base_state is None or changed_state is None:
        raise ValueError("intervention requires post-MoE activation capture")
    difference = torch.linalg.vector_norm(changed_state.double() - base_state.double())
    baseline_norm = torch.linalg.vector_norm(base_state.double()).clamp_min(1e-30)
    base_log_probs = baseline.logits[:, target.token_position].double().log_softmax(dim=-1)
    changed_log_probs = changed.logits[:, target.token_position].double().log_softmax(dim=-1)
    base_probabilities = base_log_probs.exp()
    next_token_kl = float(
        (base_probabilities * (base_log_probs - changed_log_probs)).sum(dim=-1).mean()
    )
    downstream_keys = sorted(
        key
        for key in baseline_activations
        if key[0] > target.token_position
        or (key[0] == target.token_position and key[1] > target.layer_idx)
    )
    flips = 0
    router_kls = []
    for key in downstream_keys:
        base_activation = baseline_activations[key]
        changed_activation = changed_activations[key]
        base_ids = base_activation.topk_ids.sort(dim=-1).values
        changed_ids = changed_activation.topk_ids.sort(dim=-1).values
        flips += int(not bool(torch.equal(base_ids, changed_ids)))
        base_lp = base_activation.router_logits.double().log_softmax(dim=-1)
        changed_lp = changed_activation.router_logits.double().log_softmax(dim=-1)
        router_kls.append(float((base_lp.exp() * (base_lp - changed_lp)).sum(dim=-1).mean()))
    count = len(downstream_keys)
    return InterventionRow(
        "offline_teacher_forced",
        sample_id,
        target.token_position,
        target.layer_idx,
        target.expert_idx,
        intervention,
        float(difference),
        float(difference / baseline_norm),
        next_token_kl,
        count,
        flips,
        flips / count if count else 0.0,
        sum(router_kls) / count if count else 0.0,
    )


def run_sampled_interventions(
    adapter: TinyMoEAdapter,
    samples: tuple[tuple[str, Tensor], ...],
    *,
    max_targets: int,
    seed: int,
    interventions: tuple[str, ...],
) -> list[InterventionRow]:
    targets = sample_active_targets(adapter, samples, max_targets=max_targets, seed=seed)
    by_id = dict(samples)
    return [
        run_intervention(
            adapter,
            target.sample_id,
            by_id[target.sample_id],
            target,
            intervention=intervention,
        )
        for target in targets
        for intervention in interventions
    ]
