"""Causal policies. Evaluator labels and future continuations never enter this module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.benchmark.context_continuation import build_context_continuation


@dataclass(frozen=True)
class WindowConfig:
    residency_window_tokens: int = 8
    pseudo_compute_tokens: int = 4
    pseudo_content_horizon: int = 8
    selector_core_anchors: int = 4

    def __post_init__(self) -> None:
        if self.residency_window_tokens < 1:
            raise ValueError("window must contain production forwards")
        if self.pseudo_compute_tokens not in (0, 4, 8):
            raise ValueError("supported G values are 0, 4 and 8")
        if self.pseudo_content_horizon != 8 or self.selector_core_anchors != 4:
            raise ValueError("restart-v1 retains content horizon 8 and first-four core")


def anchor_ids(
    prompt: tuple[int, ...], generated: tuple[int, ...], config: WindowConfig
) -> tuple[int, ...]:
    if config.pseudo_compute_tokens == 0:
        return ()
    if not generated or len(prompt) + len(generated) - 1 < 7:
        raise ValueError("legacy pseudo content requires seven preceding known tokens")
    # Construct exactly the legacy H=8 sequence BEFORE truncating compute positions.
    plan = build_context_continuation(
        len(generated) - 1,
        prompt,
        generated,
        "sampled_unigram_full_continuation",
        horizon=config.pseudo_content_horizon,
    )
    return plan.anchor_token_ids[: config.pseudo_compute_tokens]


def top_ids(scores: Tensor, budget: int) -> tuple[int, ...]:
    if scores.ndim != 1 or not 0 < budget <= scores.numel():
        raise ValueError("invalid scores or budget")
    # Stable descending sort preserves ascending logical ID for exact ties.
    return tuple(sorted(scores.argsort(descending=True, stable=True)[:budget].cpu().tolist()))


def normalized(scores: Tensor) -> Tensor:
    total = scores.sum()
    return torch.where(
        total > 0, scores / total.clamp_min(1e-30), torch.full_like(scores, 1 / scores.numel())
    )


def cost_aware_subset(
    history: Tensor,
    current: tuple[int, ...],
    expert_bytes: Tensor,
    *,
    budget: int,
    d: int = 8,
    beta: float = 0.05,
    pseudo_probabilities: Tensor | None = None,
) -> tuple[int, ...]:
    utility = normalized(history.float())
    if pseudo_probabilities is not None:
        utility = 0.5 * utility + 0.5 * normalized(pseudo_probabilities[:4].float().mean(0))
    resident = torch.zeros_like(utility, dtype=torch.bool)
    resident[list(current)] = True
    cost = beta * expert_bytes / expert_bytes.mean() * (~resident)
    return top_ids(d * utility - cost, budget)


def legacy_subset(
    probabilities: Tensor, ids: Tensor, history: Tensor, budget: int
) -> tuple[int, ...]:
    if probabilities.shape[0] not in (4, 8) or ids.shape[0] != probabilities.shape[0]:
        raise ValueError("legacy selector requires four or eight computed anchors")
    # Preserve double-precision ranking and hard core priority; tail anchors are ignored.
    p, i, h = probabilities.double().cpu(), ids.long().cpu(), history.double().cpu()
    union = set(i[:4].flatten().tolist())
    scores = p[:4].sum(0)
    core = sorted(union, key=lambda e: (-float(scores[e]), e))[:budget]
    fill = sorted((e for e in range(h.numel()) if e not in union), key=lambda e: (-float(h[e]), e))
    selected = tuple(sorted(core + fill[: budget - len(core)]))
    if len(selected) != budget:
        raise ValueError("selector failed to fill slots")
    return selected
