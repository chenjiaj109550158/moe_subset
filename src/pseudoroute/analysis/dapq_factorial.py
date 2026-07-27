"""Offline DapQ-style capture, metrics, and paired bootstrap aggregation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.analysis.pseudo_sequences import OfflinePseudoSequence
from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
from pseudoroute.models.adapters.tiny import TinyMoEAdapter


@dataclass(frozen=True)
class FactorialCapture:
    sequence: OfflinePseudoSequence
    pre_rope_query: Tensor | None
    post_rope_query: Tensor | None
    post_attention_state: Tensor | None
    router_input: Tensor | None
    router_logits: Tensor
    topk_ids: Tensor
    topk_weights: Tensor


@dataclass(frozen=True)
class FactorialMetricRow:
    information_regime: str
    sample_id: str
    condition: str
    context_swapped: bool
    position_offset: int
    content_seed: int
    layer_idx: int
    horizon: int
    metric: str
    value: float


@dataclass(frozen=True)
class BootstrapRow:
    information_regime: str
    condition: str
    context_swapped: bool
    position_offset: int
    layer_idx: int
    horizon: int
    metric: str
    mean: float
    ci_lower: float
    ci_upper: float
    num_samples: int
    bootstrap_samples: int


def capture_factorial(
    adapter: TinyMoEAdapter | HFMixtralAdapter, sequence: OfflinePseudoSequence
) -> FactorialCapture:
    records = adapter.capture_offline_factorial(sequence)
    layers = adapter.spec.moe_layer_indices
    by_key = {(record.token_position, record.layer_idx): record for record in records}
    ordered = [
        by_key[(position, layer)]
        for position in range(sequence.boundary, sequence.boundary + sequence.horizon)
        for layer in layers
    ]
    shape = (sequence.horizon, len(layers), -1)

    def stack(name: str) -> Tensor | None:
        values = [getattr(record, name) for record in ordered]
        if any(value is None for value in values):
            return None
        return torch.stack([value.reshape(-1) for value in values if value is not None]).reshape(
            shape
        )

    def required_stack(name: str) -> Tensor:
        value = stack(name)
        if value is None:
            raise ValueError(f"adapter omitted required factorial capture: {name}")
        return value

    return FactorialCapture(
        sequence=sequence,
        pre_rope_query=stack("pre_rope_query"),
        post_rope_query=stack("post_rope_query"),
        post_attention_state=stack("post_attention_state"),
        router_input=stack("router_input"),
        router_logits=required_stack("router_logits"),
        topk_ids=required_stack("topk_ids").long(),
        topk_weights=required_stack("topk_weights"),
    )


def _cosine(left: Tensor, right: Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left, right, dim=-1).mean())


def _probability_metrics(candidate: Tensor, target: Tensor) -> tuple[float, float]:
    p = target.double().softmax(dim=-1)
    q = candidate.double().softmax(dim=-1)
    midpoint = (p + q) / 2
    kl = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).mean()
    js = (
        0.5
        * (
            (p * (p.clamp_min(1e-12).log() - midpoint.log())).sum(-1)
            + (q * (q.clamp_min(1e-12).log() - midpoint.log())).sum(-1)
        ).mean()
    )
    return float(kl), float(js)


def _topk_metrics(candidate: Tensor, target: Tensor) -> tuple[float, float]:
    recalls = []
    jaccards = []
    for candidate_ids, target_ids in zip(candidate.tolist(), target.tolist(), strict=True):
        candidate_set, target_set = set(candidate_ids), set(target_ids)
        overlap = len(candidate_set & target_set)
        recalls.append(overlap / len(target_set))
        jaccards.append(overlap / len(candidate_set | target_set))
    return sum(recalls) / len(recalls), sum(jaccards) / len(jaccards)


def _rank_correlation(candidate: Tensor, target: Tensor) -> float:
    candidate_ranks = candidate.argsort(dim=-1).argsort(dim=-1).double()
    target_ranks = target.argsort(dim=-1).argsort(dim=-1).double()
    return _cosine(
        candidate_ranks - candidate_ranks.mean(dim=-1, keepdim=True),
        target_ranks - target_ranks.mean(dim=-1, keepdim=True),
    )


def _window_metrics(candidate: Tensor, target: Tensor, budget: int) -> tuple[float, float]:
    candidate_mass = candidate.double().softmax(dim=-1).sum(dim=0)
    target_probabilities = target.double().softmax(dim=-1)
    target_mass = target_probabilities.sum(dim=0)
    selected = set(candidate_mass.topk(budget).indices.tolist())
    target_selected = set(target_mass.topk(budget).indices.tolist())
    recall = len(selected & target_selected) / budget
    coverage = float(target_probabilities[:, sorted(selected)].sum() / target_probabilities.sum())
    return recall, coverage


def compare_capture(
    candidate: FactorialCapture,
    target: FactorialCapture,
    *,
    horizons: tuple[int, ...],
    top_k: int,
) -> list[FactorialMetricRow]:
    if candidate.sequence.sample_id != target.sequence.sample_id:
        raise ValueError("factorial metrics require paired examples from the same document")
    rows: list[FactorialMetricRow] = []
    for horizon in horizons:
        if horizon > candidate.sequence.horizon:
            continue
        for layer in range(candidate.router_logits.shape[1]):
            candidate_logits = candidate.router_logits[:horizon, layer]
            target_logits = target.router_logits[:horizon, layer]
            probability_kl, js = _probability_metrics(candidate_logits, target_logits)
            recall, jaccard = _topk_metrics(
                candidate.topk_ids[:horizon, layer], target.topk_ids[:horizon, layer]
            )
            utility_recall, mass_coverage = _window_metrics(candidate_logits, target_logits, top_k)
            sorted_target = target_logits.sort(dim=-1, descending=True).values
            margin = (sorted_target[:, top_k - 1] - sorted_target[:, top_k]).abs().clamp_min(1e-12)
            margin_error = float(
                ((candidate_logits - target_logits).abs().amax(dim=-1) / margin).mean()
            )
            metrics = {
                "router_logit_cosine": _cosine(candidate_logits, target_logits),
                "logit_mse": float(torch.nn.functional.mse_loss(candidate_logits, target_logits)),
                "probability_kl": probability_kl,
                "jensen_shannon": js,
                "topk_recall": recall,
                "topk_jaccard": jaccard,
                "expert_rank_correlation": _rank_correlation(candidate_logits, target_logits),
                "topb_window_utility_recall": utility_recall,
                "routing_mass_coverage": mass_coverage,
                "margin_normalized_error": margin_error,
            }
            optional_states = (
                ("pre_rope_query_cosine", candidate.pre_rope_query, target.pre_rope_query),
                ("post_rope_query_cosine", candidate.post_rope_query, target.post_rope_query),
                (
                    "full_state_cosine",
                    candidate.post_attention_state,
                    target.post_attention_state,
                ),
                (
                    "router_visible_state_cosine",
                    candidate.router_input,
                    target.router_input,
                ),
            )
            for name, candidate_state, target_state in optional_states:
                if candidate_state is not None and target_state is not None:
                    metrics[name] = _cosine(
                        candidate_state[:horizon, layer], target_state[:horizon, layer]
                    )
            rows.extend(
                FactorialMetricRow(
                    "offline_teacher_forced",
                    candidate.sequence.sample_id,
                    candidate.sequence.condition.value,
                    candidate.sequence.context_swapped,
                    candidate.sequence.position_offset,
                    candidate.sequence.content_seed,
                    layer,
                    horizon,
                    name,
                    value,
                )
                for name, value in metrics.items()
            )
    return rows


_LOWER_IS_BETTER = {
    "logit_mse",
    "probability_kl",
    "jensen_shannon",
    "margin_normalized_error",
}


def add_position_dominance(rows: list[FactorialMetricRow]) -> list[FactorialMetricRow]:
    """Append the paired DC/SP versus SC/DP position-dominance contrast."""
    values = {
        (
            row.sample_id,
            row.context_swapped,
            row.position_offset,
            row.layer_idx,
            row.horizon,
            row.metric,
            row.condition,
        ): row
        for row in rows
    }
    contrasts = []
    base_keys = {key[:-1] for key in values}
    for key in sorted(base_keys):
        dcsp = values.get((*key, "DC_SP"))
        scdp = values.get((*key, "SC_DP"))
        if dcsp is None or scdp is None:
            continue
        value = (
            scdp.value - dcsp.value if dcsp.metric in _LOWER_IS_BETTER else dcsp.value - scdp.value
        )
        contrasts.append(
            FactorialMetricRow(
                "offline_teacher_forced",
                dcsp.sample_id,
                "POSITION_DOMINANCE",
                dcsp.context_swapped,
                dcsp.position_offset,
                dcsp.content_seed,
                dcsp.layer_idx,
                dcsp.horizon,
                dcsp.metric,
                value,
            )
        )
    return [*rows, *contrasts]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap(
    rows: list[FactorialMetricRow],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> list[BootstrapRow]:
    if bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    grouped: dict[tuple[str, bool, int, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[
            (
                row.condition,
                row.context_swapped,
                row.position_offset,
                row.layer_idx,
                row.horizon,
                row.metric,
            )
        ][row.sample_id] = row.value
    randomizer = random.Random(seed)
    alpha = (1 - confidence) / 2
    output = []
    for key, by_sample in sorted(grouped.items()):
        sample_ids = sorted(by_sample)
        replicates = []
        for _ in range(bootstrap_samples):
            drawn = randomizer.choices(sample_ids, k=len(sample_ids))
            replicates.append(sum(by_sample[sample] for sample in drawn) / len(drawn))
        output.append(
            BootstrapRow(
                "offline_teacher_forced",
                key[0],
                key[1],
                key[2],
                key[3],
                key[4],
                key[5],
                sum(by_sample.values()) / len(by_sample),
                _quantile(replicates, alpha),
                _quantile(replicates, 1 - alpha),
                len(sample_ids),
                bootstrap_samples,
            )
        )
    return output
