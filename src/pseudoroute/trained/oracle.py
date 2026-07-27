"""Common trained-trace oracle sweep, cache baselines, tails, and bootstrap CIs."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor

from pseudoroute.models.introspection import load_model_manifest
from pseudoroute.oracle.metrics import (
    expert_union_sizes,
    full_mass_coverage,
    segment_hit_rate,
    selected_mass_coverage,
)
from pseudoroute.oracle.selectors import (
    OracleSelector,
    oracle_utilities,
    select_top_b,
)
from pseudoroute.oracle.windows import OracleSample, OracleWindow
from pseudoroute.tracing.validation import validate_trace
from pseudoroute.trained.config import TrainedModelConfig, TrainedSuiteConfig
from pseudoroute.types import ExpertKey, ModelSpec


@dataclass(frozen=True)
class TrainedOracleRow:
    model: str
    domain: str
    sample_id: str
    sample_role: str
    layer_idx: int
    start: int
    horizon: int
    budget: int
    budget_multiple: float
    method: str
    context_position_bucket: str
    router_margin: float
    router_margin_bucket: str
    route_hit_rate: float
    selected_mass_coverage: float
    full_mass_coverage: float | None
    expert_union_size: int
    subset_churn: float
    estimated_h2d_bytes: int
    estimated_h2d_bytes_per_token: float
    natural_route: str
    selected_subset: str


@dataclass(frozen=True)
class TrainedOracleAggregate:
    model: str
    domain: str
    layer_idx: int
    horizon: int
    budget: int
    budget_multiple: float
    method: str
    context_position_bucket: str
    router_margin_bucket: str
    metric: str
    count: int
    sample_count: int
    mean: float
    median: float
    p05: float
    p95: float
    worst: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


def _write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def _load_domain_samples(trace_root: Path) -> tuple[OracleSample, ...]:
    manifest = validate_trace(trace_root)
    samples = []
    for shard in manifest.shards:
        tensors = load_file(str(trace_root / shard.path))
        samples.append(
            OracleSample(
                sample_id=shard.sample_id,
                token_ids=tensors["token_ids"],
                topk_ids=tensors["router_topk_ids"],
                topk_weights=tensors["router_topk_weights"],
                router_logits=tensors.get("router_logits"),
                layer_indices=tuple(int(value) for value in tensors["layer_ids"].tolist()),
            )
        )
    return tuple(samples)


def _sample_roles(samples: tuple[OracleSample, ...]) -> dict[str, str]:
    midpoint = max(1, len(samples) // 2)
    return {
        sample.sample_id: "calibration" if index < midpoint else "evaluation"
        for index, sample in enumerate(samples)
    }


def _static_frequency(
    samples_by_domain: dict[str, tuple[OracleSample, ...]],
    roles_by_domain: dict[str, dict[str, str]],
    spec: ModelSpec,
) -> dict[int, Tensor]:
    counts = {
        layer: torch.zeros(spec.num_experts_by_layer[layer], dtype=torch.float64)
        for layer in spec.moe_layer_indices
    }
    for domain, samples in samples_by_domain.items():
        for sample in samples:
            if roles_by_domain[domain][sample.sample_id] != "calibration":
                continue
            axis_by_layer = {layer: axis for axis, layer in enumerate(sample.layer_indices)}
            for layer in spec.moe_layer_indices:
                ids = sample.topk_ids[:, axis_by_layer[layer]].reshape(-1)
                counts[layer].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
    return counts


def _budgets(spec: ModelSpec, suite: TrainedSuiteConfig, layer: int) -> tuple[int, ...]:
    top_k = spec.top_k_by_layer[layer]
    values = {
        min(spec.num_experts_by_layer[layer], max(top_k, int(math.ceil(value * top_k))))
        for value in suite.oracle.budget_multiples
    }
    values.update(
        min(spec.num_experts_by_layer[layer], max(top_k, value))
        for value in suite.oracle.absolute_budgets
    )
    return tuple(sorted(values))


def _context_bucket(start: int) -> str:
    if start < 16:
        return "early_0_15"
    if start < 32:
        return "middle_16_31"
    return "late_32_plus"


def _margin(window: OracleWindow, axis: int, top_k: int) -> float:
    if window.router_logits is None or top_k >= window.router_logits.shape[-1]:
        return float("nan")
    values = window.router_logits[:, axis].float().topk(top_k + 1, dim=-1).values
    return float((values[:, top_k - 1] - values[:, top_k]).mean())


def _fixed_metrics(
    window: OracleWindow, layer: int, axis: int, subset: tuple[int, ...]
) -> tuple[float, float, float | None]:
    single = OracleWindow(
        window.sample_id,
        window.start,
        window.horizon,
        window.topk_ids[:, axis : axis + 1],
        window.topk_weights[:, axis : axis + 1],
        window.router_logits[:, axis : axis + 1] if window.router_logits is not None else None,
        (layer,),
    )
    full = full_mass_coverage(single, {layer: subset}) if window.router_logits is not None else None
    return (
        segment_hit_rate(single, {layer: subset}),
        selected_mass_coverage(single, {layer: subset}),
        full,
    )


def _cache_metrics(
    window: OracleWindow,
    *,
    axis: int,
    layer: int,
    budget: int,
    expert_bytes: dict[ExpertKey, int],
    policy: str,
) -> tuple[float, float, float | None, int, tuple[int, ...]]:
    cache: list[int] = []
    frequencies: Counter[int] = Counter()
    captured_full = 0.0
    total_full = 0.0
    transfer = 0
    for token in range(window.horizon):
        ids = [int(value) for value in window.topk_ids[token, axis].tolist()]
        for expert in ids:
            frequencies[expert] += 1
            if expert in cache:
                cache.remove(expert)
                cache.append(expert)
                continue
            transfer += expert_bytes[ExpertKey(layer, expert)]
            if len(cache) >= budget:
                if policy == "lru":
                    cache.pop(0)
                else:
                    victim = min(cache, key=lambda value: (frequencies[value], value))
                    cache.remove(victim)
            cache.append(expert)
        if window.router_logits is not None:
            probabilities = window.router_logits[token, axis].double().softmax(dim=-1)
            captured_full += float(probabilities[cache].sum())
            total_full += float(probabilities.sum())
    return (
        1.0,
        1.0,
        captured_full / total_full if total_full else None,
        transfer,
        tuple(sorted(cache)),
    )


def _on_demand_bytes(
    window: OracleWindow, axis: int, layer: int, expert_bytes: dict[ExpertKey, int]
) -> int:
    return sum(
        expert_bytes[ExpertKey(layer, int(expert))]
        for expert in window.topk_ids[:, axis].reshape(-1).tolist()
    )


def _margin_thresholds(rows: list[TrainedOracleRow]) -> tuple[float, float]:
    values = np.array([row.router_margin for row in rows if math.isfinite(row.router_margin)])
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, 1 / 3)), float(np.quantile(values, 2 / 3))


def _assign_margin_bucket(value: float, low: float, high: float) -> str:
    if not math.isfinite(value):
        return "unavailable"
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def run_model_oracle(
    suite: TrainedSuiteConfig,
    model: TrainedModelConfig,
    model_root: Path,
) -> tuple[list[TrainedOracleRow], list[TrainedOracleAggregate]]:
    samples_by_domain: dict[str, tuple[OracleSample, ...]] = {
        dataset.key: _load_domain_samples(model_root / "traces" / dataset.key)
        for dataset in suite.datasets
    }
    roles_by_domain: dict[str, dict[str, str]] = {
        domain: _sample_roles(samples) for domain, samples in samples_by_domain.items()
    }
    spec = load_model_manifest(
        model_root / "traces" / suite.datasets[0].key / "model_manifest.json"
    )
    static = _static_frequency(samples_by_domain, roles_by_domain, spec)
    rows: list[TrainedOracleRow] = []
    previous_subsets: dict[tuple[str, str, int, int, int, str], tuple[int, ...]] = {}
    for domain, samples in samples_by_domain.items():
        for sample in samples:
            role = roles_by_domain[domain][sample.sample_id]
            if role != "evaluation":
                continue
            axis_by_layer = {layer: axis for axis, layer in enumerate(sample.layer_indices)}
            for horizon in suite.oracle.horizons:
                for start in range(0, int(sample.token_ids.numel()) - horizon + 1, 4):
                    window = OracleWindow(
                        sample.sample_id,
                        start,
                        horizon,
                        sample.topk_ids[start : start + horizon],
                        sample.topk_weights[start : start + horizon],
                        sample.router_logits[start : start + horizon]
                        if sample.router_logits is not None
                        else None,
                        sample.layer_indices,
                    )
                    unions = expert_union_sizes(window)
                    utility_by_selector = {
                        selector: oracle_utilities(
                            window,
                            selector,
                            num_experts=next(iter(spec.num_experts_by_layer.values())),
                            gamma=1.0,
                        )
                        for selector in (
                            OracleSelector.BINARY_COUNT,
                            OracleSelector.SELECTED_ROUTING_MASS,
                            OracleSelector.FULL_ROUTER_MASS,
                        )
                        if selector is not OracleSelector.FULL_ROUTER_MASS
                        or model.full_router_mass_valid
                    }
                    for layer in spec.moe_layer_indices:
                        axis = axis_by_layer[layer]
                        margin = _margin(window, axis, spec.top_k_by_layer[layer])
                        for budget in _budgets(spec, suite, layer):
                            budget_multiple = budget / spec.top_k_by_layer[layer]
                            subsets: dict[str, tuple[int, ...]] = {}
                            for selector in (
                                OracleSelector.BINARY_COUNT,
                                OracleSelector.SELECTED_ROUTING_MASS,
                                OracleSelector.FULL_ROUTER_MASS,
                            ):
                                if (
                                    selector is OracleSelector.FULL_ROUTER_MASS
                                    and not model.full_router_mass_valid
                                ):
                                    continue
                                utilities = utility_by_selector[selector]
                                subsets[selector.value] = select_top_b(
                                    {layer: utilities[layer]}, {layer: budget}
                                )[layer]
                            static_ranking = sorted(
                                range(static[layer].numel()),
                                key=lambda expert: (-float(static[layer][expert]), expert),
                            )
                            subsets["static_frequency"] = tuple(sorted(static_ranking[:budget]))
                            previous_scores = torch.zeros_like(static[layer])
                            if start > 0:
                                ids = sample.topk_ids[start - 1, axis]
                                weights = sample.topk_weights[start - 1, axis]
                                previous_scores.scatter_add_(0, ids, weights.double())
                            previous_ranking = sorted(
                                range(previous_scores.numel()),
                                key=lambda expert: (
                                    -float(previous_scores[expert]),
                                    -float(static[layer][expert]),
                                    expert,
                                ),
                            )
                            subsets["previous_route"] = tuple(sorted(previous_ranking[:budget]))
                            methods = list(subsets) + ["on_demand", "lru", "lfu"]
                            for method in methods:
                                subset = subsets.get(method, ())
                                if method == "on_demand":
                                    hit, selected = 1.0, 1.0
                                    full = None
                                    transfer = _on_demand_bytes(
                                        window, axis, layer, spec.expert_bytes
                                    )
                                    natural_subset = tuple(
                                        sorted(set(window.topk_ids[:, axis].reshape(-1).tolist()))
                                    )
                                    subset = natural_subset
                                elif method in {"lru", "lfu"}:
                                    hit, selected, full, transfer, subset = _cache_metrics(
                                        window,
                                        axis=axis,
                                        layer=layer,
                                        budget=budget,
                                        expert_bytes=spec.expert_bytes,
                                        policy=method,
                                    )
                                else:
                                    hit, selected, full = _fixed_metrics(
                                        window, layer, axis, subset
                                    )
                                    transfer = sum(
                                        spec.expert_bytes[ExpertKey(layer, expert)]
                                        for expert in subset
                                    )
                                scenario_key = (
                                    sample.sample_id,
                                    domain,
                                    layer,
                                    horizon,
                                    budget,
                                    method,
                                )
                                previous = previous_subsets.get(scenario_key, ())
                                churn = (
                                    1.0
                                    - len(set(previous).intersection(subset))
                                    / max(1, len(set(previous).union(subset)))
                                    if previous
                                    else 1.0
                                )
                                previous_subsets[scenario_key] = subset
                                natural_route = window.topk_ids[:, axis].tolist()
                                rows.append(
                                    TrainedOracleRow(
                                        model.key,
                                        domain,
                                        sample.sample_id,
                                        role,
                                        layer,
                                        start,
                                        horizon,
                                        budget,
                                        budget_multiple,
                                        method,
                                        _context_bucket(start),
                                        margin,
                                        "pending",
                                        hit,
                                        selected,
                                        full,
                                        unions[layer],
                                        churn,
                                        transfer,
                                        transfer / horizon,
                                        json.dumps(natural_route),
                                        json.dumps(subset),
                                    )
                                )
    low, high = _margin_thresholds(rows)
    rows = [
        TrainedOracleRow(
            **{
                **asdict(row),
                "router_margin_bucket": _assign_margin_bucket(row.router_margin, low, high),
            }
        )
        for row in rows
    ]
    aggregates = aggregate_oracle_rows(suite, rows)
    stratified = aggregate_oracle_rows(suite, rows, stratify=True)
    _write_csv(model_root / "oracle_windows.csv", rows)
    _write_csv(model_root / "oracle_aggregates.csv", aggregates)
    _write_csv(model_root / "oracle_stratified_aggregates.csv", stratified)
    worst = sorted(rows, key=lambda row: (row.route_hit_rate, row.selected_mass_coverage))[:200]
    (model_root / "oracle_worst_cases.json").write_text(
        json.dumps([asdict(row) for row in worst], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows, aggregates


def aggregate_oracle_rows(
    suite: TrainedSuiteConfig,
    rows: list[TrainedOracleRow],
    *,
    stratify: bool = False,
) -> list[TrainedOracleAggregate]:
    metrics = (
        "route_hit_rate",
        "selected_mass_coverage",
        "full_mass_coverage",
        "expert_union_size",
        "subset_churn",
        "estimated_h2d_bytes_per_token",
    )
    groups: dict[tuple[object, ...], list[TrainedOracleRow]] = defaultdict(list)
    for row in rows:
        base_key: tuple[object, ...] = (
            row.model,
            row.domain,
            row.layer_idx,
            row.horizon,
            row.budget,
            row.budget_multiple,
            row.method,
        )
        group_key = (
            (*base_key, row.context_position_bucket, row.router_margin_bucket)
            if stratify
            else base_key
        )
        groups[group_key].append(row)
    rng = np.random.default_rng(suite.trace.seed)
    alpha = (1 - suite.oracle.confidence) / 2
    aggregates = []
    bootstrap_indices: dict[int, np.ndarray] = {}
    for aggregate_key, members in groups.items():
        model_key = str(aggregate_key[0])
        domain = str(aggregate_key[1])
        layer_idx = int(str(aggregate_key[2]))
        horizon = int(str(aggregate_key[3]))
        budget = int(str(aggregate_key[4]))
        budget_multiple = float(str(aggregate_key[5]))
        method = str(aggregate_key[6])
        context_bucket = str(aggregate_key[7]) if stratify else "all"
        margin_bucket = str(aggregate_key[8]) if stratify else "all"
        for metric in metrics:
            pairs = [
                (row.sample_id, float(value))
                for row in members
                if (value := getattr(row, metric)) is not None
            ]
            if not pairs:
                continue
            values = np.array([value for _, value in pairs], dtype=np.float64)
            by_sample: dict[str, list[float]] = defaultdict(list)
            for sample_id, value in pairs:
                by_sample[sample_id].append(value)
            sample_means = np.array([mean(sample_values) for sample_values in by_sample.values()])
            if sample_means.size not in bootstrap_indices:
                bootstrap_indices[sample_means.size] = rng.integers(
                    0,
                    sample_means.size,
                    size=(suite.oracle.bootstrap_samples, sample_means.size),
                )
            draws = sample_means[bootstrap_indices[sample_means.size]].mean(axis=1)
            aggregates.append(
                TrainedOracleAggregate(
                    model_key,
                    domain,
                    layer_idx,
                    horizon,
                    budget,
                    budget_multiple,
                    method,
                    context_bucket,
                    margin_bucket,
                    metric,
                    len(values),
                    len(sample_means),
                    float(values.mean()),
                    float(np.median(values)),
                    float(np.quantile(values, 0.05)),
                    float(np.quantile(values, 0.95)),
                    float(values.min()),
                    float(np.quantile(draws, alpha)),
                    float(np.quantile(draws, 1 - alpha)),
                )
            )
    return aggregates


def aggregate_saved_oracle_windows(
    suite: TrainedSuiteConfig, model_root: Path
) -> list[TrainedOracleAggregate]:
    """Regenerate context/margin-stratified summaries from the saved window table."""
    rows: list[TrainedOracleRow] = []
    with (model_root / "oracle_windows.csv").open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            full = raw["full_mass_coverage"]
            rows.append(
                TrainedOracleRow(
                    raw["model"],
                    raw["domain"],
                    raw["sample_id"],
                    raw["sample_role"],
                    int(raw["layer_idx"]),
                    int(raw["start"]),
                    int(raw["horizon"]),
                    int(raw["budget"]),
                    float(raw["budget_multiple"]),
                    raw["method"],
                    raw["context_position_bucket"],
                    float(raw["router_margin"]),
                    raw["router_margin_bucket"],
                    float(raw["route_hit_rate"]),
                    float(raw["selected_mass_coverage"]),
                    float(full) if full else None,
                    int(raw["expert_union_size"]),
                    float(raw["subset_churn"]),
                    int(raw["estimated_h2d_bytes"]),
                    float(raw["estimated_h2d_bytes_per_token"]),
                    raw["natural_route"],
                    raw["selected_subset"],
                )
            )
    aggregates = aggregate_oracle_rows(suite, rows, stratify=True)
    _write_csv(model_root / "oracle_stratified_aggregates.csv", aggregates)
    return aggregates
