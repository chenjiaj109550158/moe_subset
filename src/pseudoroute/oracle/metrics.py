"""Open-loop local-routing-consistency metrics."""

from __future__ import annotations

import torch

from pseudoroute.oracle.windows import OracleWindow


def segment_hit_rate(window: OracleWindow, subsets: dict[int, tuple[int, ...]]) -> float:
    hits = 0
    total = int(window.topk_ids.numel())
    for layer, subset in subsets.items():
        available = torch.tensor(subset, device=window.topk_ids.device)
        hits += int(torch.isin(window.topk_ids[:, layer], available).sum())
    return hits / total if total else 0.0


def selected_mass_coverage(window: OracleWindow, subsets: dict[int, tuple[int, ...]]) -> float:
    captured = 0.0
    total = float(window.topk_weights.sum())
    for layer, subset in subsets.items():
        available = torch.tensor(subset, device=window.topk_ids.device)
        mask = torch.isin(window.topk_ids[:, layer], available)
        captured += float(window.topk_weights[:, layer].masked_select(mask).sum())
    return captured / total if total else 0.0


def full_mass_coverage(window: OracleWindow, subsets: dict[int, tuple[int, ...]]) -> float:
    if window.router_logits is None:
        raise ValueError("full mass coverage requires router logits")
    probabilities = window.router_logits.double().softmax(dim=-1)
    captured = 0.0
    for layer, subset in subsets.items():
        captured += float(probabilities[:, layer, list(subset)].sum())
    total = float(probabilities.sum())
    return captured / total if total else 0.0


def expert_union_sizes(window: OracleWindow) -> dict[int, int]:
    return {
        layer: len(set(window.topk_ids[:, layer].reshape(-1).tolist()))
        for layer in range(window.topk_ids.shape[1])
    }


def union_coverage(window: OracleWindow, subsets: dict[int, tuple[int, ...]]) -> float:
    covered = 0
    total = 0
    for layer, subset in subsets.items():
        natural = set(window.topk_ids[:, layer].reshape(-1).tolist())
        covered += len(natural.intersection(subset))
        total += len(natural)
    return covered / total if total else 0.0
