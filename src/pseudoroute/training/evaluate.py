"""Common-target M6 probe evaluation, transition fitting, and latency accounting."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.probes.base import FutureRoutingProbe
from pseudoroute.probes.heuristics import MarkovTransitionProbe
from pseudoroute.training.dataset import PredictorDataset
from pseudoroute.types import DeployableDecodeState


@dataclass(frozen=True)
class ProbeEvaluationRow:
    information_regime: str
    probe_name: str
    split: str
    rows: int
    utility_mse: float
    utility_cosine: float
    topb_recall: float
    routing_mass_coverage: float
    subset_regret: float
    latency_median_us: float
    latency_p95_us: float
    temporary_output_bytes: int
    equal_budget_experts: int
    equal_cost_ceiling_us: float
    equal_cost_slack_us: float


def fit_markov_transitions(
    adapter: TinyMoEAdapter,
    documents: tuple[tuple[str, Tensor], ...],
    *,
    train_sample_ids: frozenset[str],
    smoothing: float,
) -> Tensor:
    layers = len(adapter.spec.moe_layer_indices)
    experts = next(iter(adapter.spec.num_experts_by_layer.values()))
    counts = torch.full((layers, experts, experts), smoothing, dtype=torch.float64)
    with torch.inference_mode():
        for sample_id, token_ids in documents:
            if sample_id not in train_sample_ids:
                continue
            traces = adapter.model(token_ids, capture_trace=True).traces
            by_key = {(trace.token_position, trace.layer_idx): trace for trace in traces}
            for position in range(int(token_ids.numel()) - 1):
                for layer in adapter.spec.moe_layer_indices:
                    current = by_key[(position, layer)]
                    following = by_key[(position + 1, layer)]
                    for source, source_weight in zip(
                        current.topk_ids.reshape(-1).tolist(),
                        current.topk_weights.reshape(-1).tolist(),
                        strict=True,
                    ):
                        for destination, destination_weight in zip(
                            following.topk_ids.reshape(-1).tolist(),
                            following.topk_weights.reshape(-1).tolist(),
                            strict=True,
                        ):
                            counts[layer, int(source), int(destination)] += float(
                                source_weight
                            ) * float(destination_weight)
    return counts / counts.sum(dim=-1, keepdim=True)


def save_markov_probe(root: Path, probe: MarkovTransitionProbe) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {"transitions": probe.transitions.cpu().contiguous()}, str(root / "model.safetensors")
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "safe_format": "safetensors",
                "probe_name": "markov_transition",
                "information_regime": "online_pre_sample",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_markov_probe(root: Path) -> MarkovTransitionProbe:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("safe_format") != "safetensors"
        or manifest.get("probe_name") != "markov_transition"
    ):
        raise ValueError("unsafe or invalid Markov model format")
    return MarkovTransitionProbe(load_file(str(root / "model.safetensors"))["transitions"])


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def evaluate_probe(
    probe: FutureRoutingProbe,
    dataset: PredictorDataset,
    states: tuple[DeployableDecodeState, ...],
    *,
    split: str,
    top_b: int,
    latency_repetitions: int,
    anchor_candidates: tuple[int, ...] | None = None,
) -> tuple[dict[str, float | int | str], list[float]]:
    indices = dataset.indices(split).tolist()
    regimes = {states[index].information_regime.value for index in indices}
    if len(regimes) != 1:
        raise ValueError("probe evaluation split mixes information regimes")
    information_regime = regimes.pop()
    predictions = []
    targets = []
    latencies = []
    output_bytes = 0
    for index in indices:
        state = states[index]
        horizon = dataset.rows[index].horizon
        requested_horizons = (horizon,)
        if anchor_candidates is not None:
            requested_horizons = tuple(
                sorted({*(anchor for anchor in anchor_candidates if anchor <= horizon), horizon})
            )
        for _ in range(latency_repetitions):
            started = time.perf_counter_ns()
            output = probe.predict(state, requested_horizons)
            latencies.append((time.perf_counter_ns() - started) / 1000)
        flat = torch.cat(
            [
                output.aggregate_utility[layer].detach().cpu().double()
                for layer in range(dataset.num_layers)
            ]
        )
        if flat.shape != dataset.targets[index].shape or not bool(torch.isfinite(flat).all()):
            raise ValueError(f"{probe.name} produced invalid utility shape or values")
        if bool((flat < 0).any()):
            raise ValueError(f"{probe.name} produced negative utility")
        predictions.append(flat)
        targets.append(dataset.targets[index])
        output_bytes = max(
            output_bytes,
            flat.numel() * flat.element_size(),
            int(output.metadata.get("temporary_bytes", 0)),
        )
    predicted = torch.stack(predictions)
    target = torch.stack(targets)
    mse = float(torch.nn.functional.mse_loss(predicted, target))
    cosine = float(torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean())
    recalls, coverages, regrets = [], [], []
    for prediction, truth in zip(predicted, target, strict=True):
        for layer in range(dataset.num_layers):
            start, end = layer * dataset.num_experts, (layer + 1) * dataset.num_experts
            predicted_set = set(prediction[start:end].topk(top_b).indices.tolist())
            oracle_set = set(truth[start:end].topk(top_b).indices.tolist())
            recalls.append(len(predicted_set & oracle_set) / top_b)
            captured = float(truth[start:end][list(sorted(predicted_set))].sum())
            oracle_mass = float(truth[start:end][list(sorted(oracle_set))].sum())
            total = float(truth[start:end].sum())
            coverages.append(captured / total if total else 0.0)
            regrets.append(oracle_mass - captured)
    return (
        {
            "information_regime": information_regime,
            "probe_name": probe.name,
            "split": split,
            "rows": len(indices),
            "utility_mse": mse,
            "utility_cosine": cosine,
            "topb_recall": sum(recalls) / len(recalls),
            "routing_mass_coverage": sum(coverages) / len(coverages),
            "subset_regret": sum(regrets) / len(regrets),
            "latency_median_us": statistics.median(latencies),
            "latency_p95_us": _percentile(latencies, 0.95),
            "temporary_output_bytes": output_bytes,
            "equal_budget_experts": top_b,
        },
        latencies,
    )


def finalize_equal_cost(results: list[dict[str, float | int | str]]) -> list[ProbeEvaluationRow]:
    ceiling = max(float(result["latency_p95_us"]) for result in results)
    return [
        ProbeEvaluationRow(
            str(result["information_regime"]),
            str(result["probe_name"]),
            str(result["split"]),
            int(result["rows"]),
            float(result["utility_mse"]),
            float(result["utility_cosine"]),
            float(result["topb_recall"]),
            float(result["routing_mass_coverage"]),
            float(result["subset_regret"]),
            float(result["latency_median_us"]),
            float(result["latency_p95_us"]),
            int(result["temporary_output_bytes"]),
            int(result["equal_budget_experts"]),
            ceiling,
            ceiling - float(result["latency_p95_us"]),
        )
        for result in results
    ]
