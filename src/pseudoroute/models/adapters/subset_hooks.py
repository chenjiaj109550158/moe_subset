"""Temporary native-router hooks for validated trained-model subset execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from pseudoroute.execution.routing_policy import (
    ExecutedRoute,
    LosslessFallbackPolicy,
    MaskedSubstitutionPolicy,
    NaturalRoutingPolicy,
    RoutingPolicy,
)
from pseudoroute.types import ExpertKey, MissPolicy


@dataclass(frozen=True)
class NativeRouteSemantics:
    score_kind: str
    normalize_selected: bool
    scaling_factor: float = 1.0


def native_scores(logits: Tensor, semantics: NativeRouteSemantics) -> Tensor:
    if semantics.score_kind == "softmax_float32":
        return logits.float().softmax(dim=-1)
    if semantics.score_kind == "selected_logits_softmax":
        return logits
    raise ValueError(f"unsupported score kind: {semantics.score_kind}")


def select_allowed(
    logits: Tensor,
    allowed: tuple[int, ...],
    *,
    top_k: int,
    semantics: NativeRouteSemantics,
) -> tuple[Tensor, Tensor, Tensor]:
    if len(allowed) < top_k:
        raise ValueError("hard commitment requires at least native top-k allowed experts")
    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    if semantics.score_kind == "selected_logits_softmax":
        allowed_values = logits.index_select(-1, allowed_tensor)
        selected_values, positions = allowed_values.topk(top_k, dim=-1)
        ids = allowed_tensor[positions]
        weights = selected_values.softmax(dim=-1, dtype=selected_values.dtype)
        return logits, ids, weights
    scores = native_scores(logits, semantics)
    selected, positions = scores.index_select(-1, allowed_tensor).topk(top_k, dim=-1)
    ids = allowed_tensor[positions]
    if semantics.normalize_selected:
        selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    else:
        selected = selected * semantics.scaling_factor
    return scores, ids, selected.to(logits.dtype)


def install_tuple_router_hooks(
    routers: tuple[tuple[int, nn.Module], ...],
    *,
    policy: RoutingPolicy,
    allowed_experts: frozenset[ExpertKey] | None,
    top_k_by_layer: dict[int, int],
    semantics_by_layer: dict[int, NativeRouteSemantics],
    records: list[ExecutedRoute],
    output_order: str = "logits_weights_ids",
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install reversible hooks on routers returning native routing tuples."""

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, router in routers:
        allowed = tuple(
            sorted(
                key.expert_idx
                for key in (allowed_experts or frozenset())
                if key.layer_idx == layer_idx
            )
        )

        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            index: int = layer_idx,
            layer_allowed: tuple[int, ...] = allowed,
        ) -> object:
            if not isinstance(output, tuple) or len(output) < 3:
                raise RuntimeError("native router did not return a three-item tuple")
            if output_order == "logits_weights_ids":
                logits, natural_weights, natural_ids = output[:3]
            else:
                natural_ids, natural_weights, auxiliary = output[:3]
                if not _inputs or not isinstance(_inputs[0], Tensor):
                    raise RuntimeError("DeepSeek gate hook requires router input")
                gate = _module
                state = _inputs[0].reshape(-1, _inputs[0].shape[-1])
                logits = torch.nn.functional.linear(
                    state.float(), cast(Tensor, gate.weight).float()
                )
            if not all(
                isinstance(value, Tensor) for value in (logits, natural_weights, natural_ids)
            ):
                raise RuntimeError("native router tuple contains non-tensor route values")
            semantics = semantics_by_layer[index]
            scores = native_scores(logits, semantics)
            if semantics.score_kind == "selected_logits_softmax":
                full_probabilities = logits.float().softmax(dim=-1)
            else:
                full_probabilities = scores
            allowed_set = set(layer_allowed)
            missing = (
                tuple(
                    ExpertKey(index, int(expert))
                    for expert in natural_ids.reshape(-1).tolist()
                    if int(expert) not in allowed_set
                )
                if allowed_experts is not None
                else ()
            )
            out_mass = 0.0
            if allowed_experts is not None:
                mask = torch.ones_like(full_probabilities, dtype=torch.bool)
                if layer_allowed:
                    mask[..., list(layer_allowed)] = False
                out_mass = float(full_probabilities.masked_select(mask).sum()) / max(
                    1, logits.reshape(-1, logits.shape[-1]).shape[0]
                )
            executed_ids = natural_ids
            executed_weights = natural_weights
            fallback = False
            miss_policy: MissPolicy | None = None
            if isinstance(policy, MaskedSubstitutionPolicy):
                _, executed_ids, executed_weights = select_allowed(
                    logits,
                    layer_allowed,
                    top_k=top_k_by_layer[index],
                    semantics=semantics,
                )
                miss_policy = MissPolicy.SUBSTITUTE
            elif isinstance(policy, LosslessFallbackPolicy):
                fallback = bool(missing)
                miss_policy = MissPolicy.LOSSLESS_FALLBACK
            elif not isinstance(policy, NaturalRoutingPolicy):
                raise NotImplementedError(
                    "trained tuple-router hook supports natural, lossless fallback, "
                    "and substitution"
                )
            records.append(
                ExecutedRoute(
                    token_position=0,
                    layer_idx=index,
                    natural_logits=logits.detach().cpu(),
                    natural_topk_ids=natural_ids.detach().cpu(),
                    natural_topk_weights=natural_weights.detach().cpu(),
                    executed_topk_ids=executed_ids.detach().cpu(),
                    executed_topk_weights=executed_weights.detach().cpu(),
                    out_of_subset_mass=out_mass,
                    missing_natural_experts=missing,
                    fallback_used=fallback,
                    miss_policy=miss_policy,
                )
            )
            if output_order == "logits_weights_ids":
                return (logits, executed_weights, executed_ids, *output[3:])
            return (executed_ids, executed_weights, auxiliary, *output[3:])

        handles.append(router.register_forward_hook(hook))
    return handles
