"""M9 calibration signals and deterministic static expert rankings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.analysis.expert_criticality import run_sampled_interventions
from pseudoroute.analysis.router_geometry import (
    covariance_aware_logit_variance,
    empirical_covariance_sketch,
)
from pseudoroute.models.adapters.tiny import TinyMoEAdapter


@dataclass(frozen=True)
class StaticRankingSignals:
    frequency: Tensor
    covariance_variance: Tensor
    quality_sensitivity: Tensor
    downstream_influence: Tensor
    miss_risk: Tensor
    combined: Tensor


def _normalize(values: Tensor) -> Tensor:
    minimum = values.min(dim=-1, keepdim=True).values
    span = values.max(dim=-1, keepdim=True).values - minimum
    return (values - minimum) / span.clamp_min(1e-30)


def calibrate_static_signals(
    adapter: TinyMoEAdapter,
    documents: tuple[tuple[str, Tensor], ...],
    *,
    max_interventions: int,
    seed: int,
) -> StaticRankingSignals:
    layers = len(adapter.spec.moe_layer_indices)
    experts = next(iter(adapter.spec.num_experts_by_layer.values()))
    frequency = torch.zeros(layers, experts, dtype=torch.float64)
    misses = torch.zeros_like(frequency)
    router_inputs: list[list[Tensor]] = [[] for _ in range(layers)]
    with torch.inference_mode():
        for _, tokens in documents:
            output = adapter.model(tokens, capture_trace=True, capture_activations=True)
            selected_by_position: dict[tuple[int, int], set[int]] = {}
            for activation in output.activations:
                ids = {int(value) for value in activation.topk_ids.reshape(-1).tolist()}
                selected_by_position[(activation.token_position, activation.layer_idx)] = ids
                for expert in ids:
                    frequency[activation.layer_idx, expert] += 1
                if activation.token_position > 0:
                    previous = selected_by_position[
                        (activation.token_position - 1, activation.layer_idx)
                    ]
                    for expert in ids - previous:
                        misses[activation.layer_idx, expert] += 1
                if activation.router_input is not None:
                    router_inputs[activation.layer_idx].append(activation.router_input.cpu())
    covariance_variance = torch.zeros_like(frequency)
    for layer_idx in range(layers):
        states = torch.cat(router_inputs[layer_idx]).double()
        sketch = empirical_covariance_sketch(states, rank=states.shape[-1])
        weight = adapter._layer(layer_idx).router.weight.detach().cpu().double()  # noqa: SLF001
        covariance_variance[layer_idx] = covariance_aware_logit_variance(weight, sketch.covariance)
    intervention_rows = run_sampled_interventions(
        adapter,
        documents,
        max_targets=max_interventions,
        seed=seed,
        interventions=("zero_selected_contribution",),
    )
    quality = torch.zeros_like(frequency)
    influence = torch.zeros_like(frequency)
    counts = torch.zeros_like(frequency)
    for row in intervention_rows:
        quality[row.layer_idx, row.expert_idx] += row.next_token_kl
        influence[row.layer_idx, row.expert_idx] += row.downstream_router_kl
        counts[row.layer_idx, row.expert_idx] += 1
    quality /= counts.clamp_min(1)
    influence /= counts.clamp_min(1)
    miss_risk = misses / frequency.clamp_min(1)
    components = (
        _normalize(frequency),
        _normalize(covariance_variance),
        _normalize(quality),
        _normalize(influence),
        _normalize(miss_risk),
    )
    bytes_by_layer = torch.tensor(
        [
            [
                adapter.spec.expert_bytes[key]
                for key in sorted(adapter.spec.expert_bytes)
                if key.layer_idx == layer
            ]
            for layer in range(layers)
        ],
        dtype=torch.float64,
    )
    combined = sum(components, start=torch.zeros_like(frequency)) / bytes_by_layer
    return StaticRankingSignals(
        frequency, covariance_variance, quality, influence, miss_risk, combined
    )


def rankings(signals: StaticRankingSignals) -> dict[str, dict[int, tuple[int, ...]]]:
    methods = {
        "frequency": signals.frequency,
        "covariance_variance": signals.covariance_variance,
        "quality_sensitivity": signals.quality_sensitivity,
        "downstream_influence": signals.downstream_influence,
        "miss_risk": signals.miss_risk,
        "combined": signals.combined,
    }
    return {
        name: {
            layer: tuple(
                sorted(
                    range(values.shape[1]),
                    key=lambda expert: (-float(values[layer, expert]), expert),
                )
            )
            for layer in range(values.shape[0])
        }
        for name, values in methods.items()
    }
