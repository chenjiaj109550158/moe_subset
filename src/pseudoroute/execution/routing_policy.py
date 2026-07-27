"""Explicit M3 routing and miss-policy semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from pseudoroute.types import ExpertKey, MissPolicy


@dataclass(frozen=True)
class TokenRoutingContext:
    token_position: int
    allowed_experts: frozenset[ExpertKey] | None = None

    def allowed_for_layer(self, layer_idx: int) -> tuple[int, ...] | None:
        if self.allowed_experts is None:
            return None
        return tuple(
            sorted(key.expert_idx for key in self.allowed_experts if key.layer_idx == layer_idx)
        )


@dataclass(frozen=True)
class ExecutedRoute:
    token_position: int
    layer_idx: int
    natural_logits: Tensor
    natural_topk_ids: Tensor
    natural_topk_weights: Tensor
    executed_topk_ids: Tensor
    executed_topk_weights: Tensor
    out_of_subset_mass: float
    missing_natural_experts: tuple[ExpertKey, ...]
    fallback_used: bool
    miss_policy: MissPolicy | None


class RoutingPolicy(Protocol):
    def choose(
        self,
        layer_idx: int,
        natural_logits: Tensor,
        natural_topk: Tensor,
        token_context: TokenRoutingContext,
    ) -> ExecutedRoute: ...


def _natural_weights(logits: Tensor, topk_ids: Tensor) -> Tensor:
    probabilities = logits.softmax(dim=-1)
    selected = probabilities.gather(-1, topk_ids)
    return selected / selected.sum(dim=-1, keepdim=True)


def _missing(
    layer_idx: int, topk_ids: Tensor, allowed: tuple[int, ...] | None
) -> tuple[ExpertKey, ...]:
    if allowed is None:
        return ()
    allowed_set = set(allowed)
    return tuple(
        ExpertKey(layer_idx, int(expert))
        for expert in topk_ids.reshape(-1).tolist()
        if int(expert) not in allowed_set
    )


def _out_mass(logits: Tensor, allowed: tuple[int, ...] | None) -> float:
    if allowed is None:
        return 0.0
    probabilities = logits.softmax(dim=-1)
    mask = torch.ones_like(probabilities, dtype=torch.bool)
    if allowed:
        mask[..., list(allowed)] = False
    return float(probabilities.detach().masked_select(mask).sum()) / logits.shape[0]


@dataclass(frozen=True)
class NaturalRoutingPolicy:
    def authorize_natural_route(self) -> bool:
        return True

    def choose(
        self,
        layer_idx: int,
        natural_logits: Tensor,
        natural_topk: Tensor,
        token_context: TokenRoutingContext,
    ) -> ExecutedRoute:
        weights = _natural_weights(natural_logits, natural_topk)
        return ExecutedRoute(
            token_context.token_position,
            layer_idx,
            natural_logits.detach().clone(),
            natural_topk.detach().clone(),
            weights.detach().clone(),
            natural_topk.detach().clone(),
            weights.detach().clone(),
            0.0,
            (),
            False,
            None,
        )


@dataclass(frozen=True)
class MaskedSubstitutionPolicy:
    def choose(
        self,
        layer_idx: int,
        natural_logits: Tensor,
        natural_topk: Tensor,
        token_context: TokenRoutingContext,
    ) -> ExecutedRoute:
        allowed = token_context.allowed_for_layer(layer_idx)
        if allowed is None:
            return NaturalRoutingPolicy().choose(
                layer_idx, natural_logits, natural_topk, token_context
            )
        if len(allowed) < natural_topk.shape[-1]:
            raise ValueError("substitution requires at least top-k allowed experts")
        probabilities = natural_logits.softmax(dim=-1)
        allowed_tensor = torch.tensor(allowed, device=natural_logits.device)
        allowed_scores = probabilities[..., allowed_tensor]
        positions = allowed_scores.topk(natural_topk.shape[-1], dim=-1).indices
        executed_ids = allowed_tensor[positions]
        executed_weights = probabilities.gather(-1, executed_ids)
        executed_weights = executed_weights / executed_weights.sum(dim=-1, keepdim=True)
        natural_weights = _natural_weights(natural_logits, natural_topk)
        return ExecutedRoute(
            token_context.token_position,
            layer_idx,
            natural_logits.detach().clone(),
            natural_topk.detach().clone(),
            natural_weights.detach().clone(),
            executed_ids.detach().clone(),
            executed_weights.detach().clone(),
            _out_mass(natural_logits, allowed),
            _missing(layer_idx, natural_topk, allowed),
            False,
            MissPolicy.SUBSTITUTE,
        )


@dataclass(frozen=True)
class MaskedTruncationPolicy:
    renormalize_weights: bool

    def choose(
        self,
        layer_idx: int,
        natural_logits: Tensor,
        natural_topk: Tensor,
        token_context: TokenRoutingContext,
    ) -> ExecutedRoute:
        allowed = token_context.allowed_for_layer(layer_idx)
        if allowed is None:
            return NaturalRoutingPolicy().choose(
                layer_idx, natural_logits, natural_topk, token_context
            )
        if natural_topk.shape[0] != 1:
            raise ValueError("M3 truncation supports batch size 1")
        natural_weights = _natural_weights(natural_logits, natural_topk)
        keep = [
            index
            for index, expert in enumerate(natural_topk[0].tolist())
            if int(expert) in set(allowed)
        ]
        executed_ids = natural_topk[:, keep]
        executed_weights = natural_weights[:, keep]
        if self.renormalize_weights and executed_weights.numel():
            executed_weights = executed_weights / executed_weights.sum(dim=-1, keepdim=True)
        return ExecutedRoute(
            token_context.token_position,
            layer_idx,
            natural_logits.detach().clone(),
            natural_topk.detach().clone(),
            natural_weights.detach().clone(),
            executed_ids.detach().clone(),
            executed_weights.detach().clone(),
            _out_mass(natural_logits, allowed),
            _missing(layer_idx, natural_topk, allowed),
            False,
            MissPolicy.TRUNCATE,
        )


@dataclass(frozen=True)
class LosslessFallbackPolicy:
    def choose(
        self,
        layer_idx: int,
        natural_logits: Tensor,
        natural_topk: Tensor,
        token_context: TokenRoutingContext,
    ) -> ExecutedRoute:
        allowed = token_context.allowed_for_layer(layer_idx)
        weights = _natural_weights(natural_logits, natural_topk)
        missing = _missing(layer_idx, natural_topk, allowed)
        return ExecutedRoute(
            token_context.token_position,
            layer_idx,
            natural_logits.detach().clone(),
            natural_topk.detach().clone(),
            weights.detach().clone(),
            natural_topk.detach().clone(),
            weights.detach().clone(),
            _out_mass(natural_logits, allowed),
            missing,
            bool(missing),
            MissPolicy.LOSSLESS_FALLBACK,
        )
