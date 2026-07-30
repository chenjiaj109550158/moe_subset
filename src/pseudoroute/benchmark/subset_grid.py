"""Open-loop subset-oracle grid, transfer simulation, tails, and point selection."""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor

from pseudoroute.benchmark.prefetch import load_default_vectors
from pseudoroute.benchmark.subset_config import SubsetModelConfig, SubsetOracleSuiteConfig
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    validate_trace_manifest,
    write_json_atomic,
)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer-compatible scalar, got {type(value).__name__}")
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected float-compatible scalar, got {type(value).__name__}")
    return float(value)


@dataclass
class AggregateState:
    route_hits: int = 0
    route_slots: int = 0
    selected_mass_hit: float = 0.0
    selected_mass_total: float = 0.0
    full_mass_hit: float = 0.0
    full_mass_total: float = 0.0
    lossless_bytes: int = 0
    reference_bytes: int = 0
    tokens: int = 0
    prefetch_loads: int = 0
    fallback_loads: int = 0
    churn: int = 0
    hit_values: list[float] = field(default_factory=list)
    selected_mass_values: list[float] = field(default_factory=list)
    full_mass_values: list[float] = field(default_factory=list)

    def update(self, row: dict[str, object]) -> None:
        self.route_hits += _as_int(row["route_hits"])
        self.route_slots += _as_int(row["route_slots"])
        self.selected_mass_hit += _as_float(row["selected_mass_hit"])
        self.selected_mass_total += _as_float(row["selected_mass_total"])
        full_hit = row["full_mass_hit"]
        full_total = row["full_mass_total"]
        if full_hit != "" and full_total != "":
            self.full_mass_hit += _as_float(full_hit)
            self.full_mass_total += _as_float(full_total)
            self.full_mass_values.append(_as_float(row["full_mass_coverage"]))
        self.lossless_bytes += _as_int(row["lossless_transfer_bytes"])
        self.reference_bytes += _as_int(row["natural_reference_bytes"])
        self.tokens += _as_int(row["tokens"])
        self.prefetch_loads += _as_int(row["prefetch_loads"])
        self.fallback_loads += _as_int(row["fallback_loads"])
        self.churn += _as_int(row["subset_churn"])
        self.hit_values.append(_as_float(row["route_hit_rate"]))
        self.selected_mass_values.append(_as_float(row["selected_mass_coverage"]))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def _aggregate_row(
    model: str,
    key: tuple[object, ...],
    state: AggregateState,
    *,
    stratified: bool,
) -> dict[str, object]:
    task, horizon, budget, method, *strata = key
    route_hit = state.route_hits / state.route_slots if state.route_slots else 0.0
    selected_mass = (
        state.selected_mass_hit / state.selected_mass_total if state.selected_mass_total else 0.0
    )
    full_mass: float | str = (
        state.full_mass_hit / state.full_mass_total if state.full_mass_total else ""
    )
    transfer_reduction = (
        1 - state.lossless_bytes / state.reference_bytes if state.reference_bytes else 0.0
    )
    row: dict[str, object] = {
        "model": model,
        "task": task,
        "horizon": horizon,
        "budget": budget,
        "method": method,
        "stratified": stratified,
        "rows": len(state.hit_values),
        "tokens": state.tokens,
        "mean_route_hit": route_hit,
        "median_route_hit": _percentile(state.hit_values, 0.5),
        "p05_route_hit": _percentile(state.hit_values, 0.05),
        "p95_route_hit": _percentile(state.hit_values, 0.95),
        "worst_route_hit": min(state.hit_values),
        "mean_selected_mass": selected_mass,
        "median_selected_mass": _percentile(state.selected_mass_values, 0.5),
        "p05_selected_mass": _percentile(state.selected_mass_values, 0.05),
        "p95_selected_mass": _percentile(state.selected_mass_values, 0.95),
        "worst_selected_mass": min(state.selected_mass_values),
        "mean_full_router_mass": full_mass,
        "median_full_router_mass": (
            _percentile(state.full_mass_values, 0.5) if state.full_mass_values else ""
        ),
        "p05_full_router_mass": (
            _percentile(state.full_mass_values, 0.05) if state.full_mass_values else ""
        ),
        "p95_full_router_mass": (
            _percentile(state.full_mass_values, 0.95) if state.full_mass_values else ""
        ),
        "worst_full_router_mass": min(state.full_mass_values) if state.full_mass_values else "",
        "lossless_fallback_frequency": 1 - route_hit,
        "prefetch_loads": state.prefetch_loads,
        "fallback_loads": state.fallback_loads,
        "subset_churn": state.churn,
        "lossless_transfer_bytes": state.lossless_bytes,
        "natural_reference_bytes": state.reference_bytes,
        "estimated_h2d_bytes_per_token": state.lossless_bytes / max(1, state.tokens),
        "estimated_transfer_reduction": transfer_reduction,
    }
    if strata:
        row.update(
            {
                "layer": strata[0],
                "context_bucket": strata[1],
                "router_margin_bucket": strata[2],
            }
        )
    else:
        row.update({"layer": "", "context_bucket": "", "router_margin_bucket": ""})
    return row


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _top_b(scores: Tensor, budget: int) -> tuple[int, ...]:
    ranking = sorted(range(scores.numel()), key=lambda expert: (-float(scores[expert]), expert))
    return tuple(sorted(ranking[:budget]))


def _selected_mass_scores(ids: Tensor, weights: Tensor, experts: int) -> Tensor:
    scores = torch.zeros(experts, dtype=torch.float64)
    scores.scatter_add_(0, ids.reshape(-1).cpu(), weights.reshape(-1).double().cpu())
    return scores


def _binary_scores(ids: Tensor, experts: int) -> Tensor:
    scores = torch.zeros(experts, dtype=torch.float64)
    flat = ids.reshape(-1).cpu()
    scores.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
    return scores


def _full_mass_scores(logits: Tensor) -> Tensor:
    return logits.double().softmax(dim=-1).sum(dim=0).cpu()


def _margin_thresholds(
    manifest: dict[str, Any], model_root: Path, model: SubsetModelConfig
) -> dict[tuple[str, int], tuple[float, float]]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in manifest["shards"]:
        tensors = load_file(str(model_root / record["path"]))
        logits = tensors["router_logits"].float()
        top = logits.topk(model.native_top_k + 1, dim=-1).values
        margins = top[..., model.native_top_k - 1] - top[..., model.native_top_k]
        for layer in range(model.routed_layers):
            values[(str(record["task"]), layer)].extend(
                float(value) for value in margins[:, layer].tolist()
            )
    return {
        key: (
            _percentile(layer_values, 1 / 3),
            _percentile(layer_values, 2 / 3),
        )
        for key, layer_values in values.items()
    }


def _context_bucket(start: int) -> str:
    if start < 32:
        return "0-31"
    if start < 64:
        return "32-63"
    return "64-127"


def _margin_bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "mid"
    return "high"


def _method_scores(
    method: str,
    ids: Tensor,
    weights: Tensor,
    logits: Tensor,
    previous_ids: Tensor | None,
    previous_weights: Tensor | None,
    static_scores: Tensor,
    experts: int,
) -> Tensor:
    if method == "future_selected_routing_mass":
        return _selected_mass_scores(ids, weights, experts)
    if method == "future_binary_count":
        return _binary_scores(ids, experts)
    if method == "future_full_router_mass":
        return _full_mass_scores(logits)
    if method == "static_frequency" or previous_ids is None or previous_weights is None:
        return static_scores.double()
    if method == "previous_route":
        return _selected_mass_scores(previous_ids, previous_weights, experts)
    raise ValueError(f"unknown subset selection method: {method}")


def _row_metrics(
    *,
    model: SubsetModelConfig,
    task: str,
    sample_id: str,
    start: int,
    horizon: int,
    layer: int,
    budget: int,
    method: str,
    ids: Tensor,
    weights: Tensor,
    logits: Tensor,
    subset: tuple[int, ...],
    previous_subset: tuple[int, ...],
    expert_bytes: int,
    margin_threshold: tuple[float, float],
    bandwidth_gib_per_second: float,
    fixed_latency_microseconds_per_load: float,
) -> dict[str, object]:
    subset_tensor = torch.tensor(subset, dtype=torch.long)
    mask = torch.isin(ids.cpu(), subset_tensor)
    hits = int(mask.sum())
    slots = int(ids.numel())
    selected_hit = float(weights.cpu().masked_select(mask).double().sum())
    selected_total = float(weights.double().sum())
    full_hit: float | str = ""
    full_total: float | str = ""
    full_coverage: float | str = ""
    if model.full_router_mass == "native_full_softmax":
        probabilities = logits.double().softmax(dim=-1)
        full_hit = float(probabilities[:, list(subset)].sum())
        full_total = float(probabilities.sum())
        full_coverage = float(full_hit) / float(full_total)
    prefetch_loads = len(set(subset) - set(previous_subset))
    fallback_loads = slots - hits
    natural_loads = slots
    lossless_bytes = (prefetch_loads + fallback_loads) * expert_bytes
    reference_bytes = natural_loads * expert_bytes
    top = logits.float().topk(model.native_top_k + 1, dim=-1).values
    margin = float((top[:, model.native_top_k - 1] - top[:, model.native_top_k]).mean())
    bandwidth = bandwidth_gib_per_second * 1024**3
    loads = prefetch_loads + fallback_loads
    simulated_stall_ms = (
        lossless_bytes / bandwidth + loads * fixed_latency_microseconds_per_load / 1e6
    ) * 1000
    natural_union = sorted(set(int(value) for value in ids.reshape(-1).tolist()))
    return {
        "model": model.key,
        "task": task,
        "sample_id": sample_id,
        "boundary": start,
        "horizon": horizon,
        "tokens": int(ids.shape[0]),
        "layer": layer,
        "budget": budget,
        "all_expert_reference": budget == model.routed_experts_per_layer,
        "method": method,
        "context_bucket": _context_bucket(start),
        "router_margin": margin,
        "router_margin_bucket": _margin_bucket(margin, margin_threshold),
        "route_hits": hits,
        "route_slots": slots,
        "route_hit_rate": hits / slots,
        "selected_mass_hit": selected_hit,
        "selected_mass_total": selected_total,
        "selected_mass_coverage": selected_hit / selected_total if selected_total else 0.0,
        "full_mass_hit": full_hit,
        "full_mass_total": full_total,
        "full_mass_coverage": full_coverage,
        "expert_union_size": len(natural_union),
        "subset_churn": len(set(subset).symmetric_difference(previous_subset)),
        "prefetch_loads": prefetch_loads,
        "fallback_loads": fallback_loads,
        "fallback_frequency": 1 - hits / slots,
        "expert_bytes": expert_bytes,
        "lossless_transfer_bytes": lossless_bytes,
        "natural_reference_bytes": reference_bytes,
        "estimated_h2d_bytes_per_token": lossless_bytes / int(ids.shape[0]),
        "estimated_transfer_reduction": 1 - lossless_bytes / reference_bytes,
        "simulated_exposed_stall_ms": simulated_stall_ms,
        "timing_kind": "simulated_not_measured_runtime",
        "natural_expert_union": json.dumps(natural_union, separators=(",", ":")),
        "subset": json.dumps(subset, separators=(",", ":")),
        "natural_ids": json.dumps(ids.tolist(), separators=(",", ":")),
        "natural_weights": json.dumps(weights.float().tolist(), separators=(",", ":")),
    }


def _update_worst(
    heaps: dict[tuple[str, str], list[tuple[float, int, dict[str, object]]]],
    row: dict[str, object],
    counter: int,
    limit: int,
) -> None:
    for metric in ("route_hit_rate", "selected_mass_coverage"):
        key = (str(row["task"]), metric)
        heap = heaps[key]
        item = (-_as_float(row[metric]), counter, dict(row))
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def run_open_loop_grid(
    suite: SubsetOracleSuiteConfig,
    model: SubsetModelConfig,
    output_root: Path,
) -> dict[str, object]:
    manifest = validate_trace_manifest(suite, model, output_root)
    trace_root = output_root / "natural_traces" / model.key
    grid_root = output_root / "open_loop" / model.key
    envelope_path = grid_root / "envelope.json"
    if envelope_path.is_file():
        existing_envelope = cast(
            dict[str, Any], json.loads(envelope_path.read_text(encoding="utf-8"))
        )
        if existing_envelope.get("config_fingerprint") != suite.fingerprint():
            raise ValueError(f"open-loop envelope config mismatch for {model.key}")
        for artifact in existing_envelope["artifacts"]:
            path = grid_root / artifact["path"]
            if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"open-loop checksum mismatch: {path}")
        return cast(dict[str, object], existing_envelope)
    grid_root.mkdir(parents=True, exist_ok=True)
    source_defaults = (
        Path(suite.source_accuracy.artifact_root) / "models" / model.key / "default_vectors"
    )
    static_count = load_default_vectors(source_defaults).count.double()
    if tuple(static_count.shape) != (model.routed_layers, model.routed_experts_per_layer):
        raise ValueError(f"static frequency shape mismatch for {model.key}")
    expert_bytes_by_layer = {
        int(layer): int(value) for layer, value in manifest["expert_bytes_by_layer"].items()
    }
    margins = _margin_thresholds(manifest, trace_root, model)
    methods: list[str] = list(suite.selection.methods)
    if model.key == "qwen3_30b_a3b":
        methods.extend(suite.selection.qwen_additional_methods)
    raw_path = grid_root / "oracle_grid_raw.csv.gz"
    raw_temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp")
    global_states: dict[tuple[object, ...], AggregateState] = defaultdict(AggregateState)
    stratified_states: dict[tuple[object, ...], AggregateState] = defaultdict(AggregateState)
    worst: dict[tuple[str, str], list[tuple[float, int, dict[str, object]]]] = defaultdict(list)
    raw_rows = 0
    fieldnames: list[str] | None = None
    previous_subsets: dict[tuple[str, str, int, str, int, int], tuple[int, ...]] = {}
    with gzip.open(raw_temporary, "wt", newline="", encoding="utf-8") as stream:
        writer: csv.DictWriter[str] | None = None
        for shard_record in manifest["shards"]:
            task = str(shard_record["task"])
            sample_id = str(shard_record["sample_id"])
            tensors = load_file(str(trace_root / shard_record["path"]))
            topk_ids = tensors["router_topk_ids"]
            topk_weights = tensors["router_topk_weights"]
            router_logits = tensors["router_logits"]
            total_tokens = int(topk_ids.shape[0])
            for horizon in suite.trace.horizons:
                for start in range(0, total_tokens, horizon):
                    end = min(start + horizon, total_tokens)
                    previous_start = start - horizon
                    for layer in range(model.routed_layers):
                        ids = topk_ids[start:end, layer]
                        weights = topk_weights[start:end, layer]
                        logits = router_logits[start:end, layer]
                        previous_ids = (
                            topk_ids[previous_start:start, layer] if previous_start >= 0 else None
                        )
                        previous_weights = (
                            topk_weights[previous_start:start, layer]
                            if previous_start >= 0
                            else None
                        )
                        for method in methods:
                            scores = _method_scores(
                                method,
                                ids,
                                weights,
                                logits,
                                previous_ids,
                                previous_weights,
                                static_count[layer],
                                model.routed_experts_per_layer,
                            )
                            for budget in model.budgets:
                                subset = _top_b(scores, budget)
                                state_key = (sample_id, task, horizon, method, budget, layer)
                                previous_subset = previous_subsets.get(state_key, ())
                                row = _row_metrics(
                                    model=model,
                                    task=task,
                                    sample_id=sample_id,
                                    start=start,
                                    horizon=horizon,
                                    layer=layer,
                                    budget=budget,
                                    method=method,
                                    ids=ids,
                                    weights=weights,
                                    logits=logits,
                                    subset=subset,
                                    previous_subset=previous_subset,
                                    expert_bytes=expert_bytes_by_layer[layer],
                                    margin_threshold=margins[(task, layer)],
                                    bandwidth_gib_per_second=(
                                        suite.transfer_model.bandwidth_gib_per_second
                                    ),
                                    fixed_latency_microseconds_per_load=(
                                        suite.transfer_model.fixed_latency_microseconds_per_load
                                    ),
                                )
                                previous_subsets[state_key] = subset
                                if writer is None:
                                    fieldnames = list(row)
                                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                                    writer.writeheader()
                                writer.writerow(row)
                                global_key = (task, horizon, budget, method)
                                stratum_key = (
                                    task,
                                    horizon,
                                    budget,
                                    method,
                                    layer,
                                    row["context_bucket"],
                                    row["router_margin_bucket"],
                                )
                                global_states[global_key].update(row)
                                stratified_states[stratum_key].update(row)
                                _update_worst(
                                    worst,
                                    row,
                                    raw_rows,
                                    suite.open_loop.worst_case_count_per_model_task_metric,
                                )
                                raw_rows += 1
        stream.flush()
    if fieldnames is None or raw_rows == 0:
        raise RuntimeError(f"open-loop grid produced no rows for {model.key}")
    os.replace(raw_temporary, raw_path)
    aggregate_rows = [
        _aggregate_row(model.key, key, state, stratified=False)
        for key, state in sorted(global_states.items())
    ]
    stratified_rows = [
        _aggregate_row(model.key, key, state, stratified=True)
        for key, state in sorted(stratified_states.items())
    ]
    aggregate_path = grid_root / "oracle_grid_aggregates.csv"
    stratified_path = grid_root / "oracle_grid_stratified.csv"
    _write_csv_atomic(aggregate_path, aggregate_rows)
    _write_csv_atomic(stratified_path, stratified_rows)
    worst_rows = [
        {"criterion": metric, **item[2]}
        for (task, metric), heap in sorted(worst.items())
        for item in sorted(heap, key=lambda value: -value[0])
    ]
    worst_path = grid_root / "worst_cases.json"
    write_json_atomic(worst_path, worst_rows)
    artifacts = []
    for path in (raw_path, aggregate_path, stratified_path, worst_path):
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    envelope: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "model": model.key,
        "raw_rows": raw_rows,
        "aggregate_rows": len(aggregate_rows),
        "stratified_rows": len(stratified_rows),
        "trace_manifest_sha256": sha256_file(trace_root / "manifest.json"),
        "artifacts": artifacts,
    }
    write_json_atomic(envelope_path, envelope)
    (grid_root / "DONE").write_text("complete\n", encoding="utf-8")
    return envelope


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _gate_task_row(
    suite: SubsetOracleSuiteConfig,
    model: SubsetModelConfig,
    oracle: dict[str, str],
    baselines: Iterable[dict[str, str]],
) -> dict[str, object]:
    gate = suite.operating_point_selection
    baseline_mass = max(float(row["mean_selected_mass"]) for row in baselines)
    horizon = int(oracle["horizon"])
    budget = int(oracle["budget"])
    resident_fraction = budget / model.routed_experts_per_layer
    mean_route_hit = float(oracle["mean_route_hit"])
    mean_selected_mass = float(oracle["mean_selected_mass"])
    p05_route_hit = float(oracle["p05_route_hit"])
    worst_route_hit = float(oracle["worst_route_hit"])
    transfer_reduction = float(oracle["estimated_transfer_reduction"])
    h2d_bytes = float(oracle["estimated_h2d_bytes_per_token"])
    fallback_frequency = float(oracle["lossless_fallback_frequency"])
    baseline_improvement = mean_selected_mass - baseline_mass
    below_all_expert = budget < model.routed_experts_per_layer
    evidence = {
        "model": model.key,
        "task": oracle["task"],
        "horizon": horizon,
        "budget": budget,
        "resident_fraction": resident_fraction,
        "mean_route_hit": mean_route_hit,
        "mean_selected_mass": mean_selected_mass,
        "p05_route_hit": p05_route_hit,
        "worst_route_hit": worst_route_hit,
        "estimated_transfer_reduction": transfer_reduction,
        "estimated_h2d_bytes_per_token": h2d_bytes,
        "lossless_fallback_frequency": fallback_frequency,
        "best_baseline_selected_mass": baseline_mass,
        "selected_mass_improvement_over_best_baseline": baseline_improvement,
        "below_all_expert": below_all_expert,
    }
    checks = {
        "residency_pass": resident_fraction <= gate.maximum_resident_fraction,
        "mean_hit_pass": mean_route_hit >= gate.minimum_mean_route_hit,
        "selected_mass_pass": mean_selected_mass >= gate.minimum_mean_selected_routing_mass,
        "p05_hit_pass": p05_route_hit >= gate.minimum_p05_route_hit,
        "worst_hit_pass": worst_route_hit >= gate.minimum_worst_route_hit,
        "transfer_pass": transfer_reduction >= gate.minimum_transfer_reduction,
        "baseline_pass": (
            baseline_improvement >= gate.minimum_selected_mass_improvement_over_best_baseline
        ),
        "fallback_pass": fallback_frequency <= gate.maximum_lossless_fallback_frequency,
        "below_all_expert_pass": below_all_expert,
    }
    return {**evidence, **checks, "task_open_loop_pass": all(checks.values())}


def select_operating_points(suite: SubsetOracleSuiteConfig, output_root: Path) -> dict[str, object]:
    task_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for model in suite.models:
        rows = _read_csv(output_root / "open_loop" / model.key / "oracle_grid_aggregates.csv")
        lookup = {
            (row["task"], int(row["horizon"]), int(row["budget"]), row["method"]): row
            for row in rows
        }
        candidates: list[dict[str, object]] = []
        for horizon in suite.trace.horizons:
            for budget in model.budgets:
                evidence_rows = []
                for task in suite.trace.sample_rows:
                    oracle = lookup[(task, horizon, budget, "future_selected_routing_mass")]
                    baselines = (
                        lookup[(task, horizon, budget, "previous_route")],
                        lookup[(task, horizon, budget, "static_frequency")],
                    )
                    evidence = _gate_task_row(suite, model, oracle, baselines)
                    task_rows.append(evidence)
                    evidence_rows.append(evidence)
                candidates.append(
                    {
                        "model": model.key,
                        "horizon": horizon,
                        "budget": budget,
                        "all_tasks_pass": all(row["task_open_loop_pass"] for row in evidence_rows),
                        "task_evidence": evidence_rows,
                        "resident_fraction": budget / model.routed_experts_per_layer,
                        "worst_task_h2d": max(
                            _as_float(row["estimated_h2d_bytes_per_token"]) for row in evidence_rows
                        ),
                        "worst_task_hit": min(
                            _as_float(row["worst_route_hit"]) for row in evidence_rows
                        ),
                    }
                )
        eligible = [candidate for candidate in candidates if candidate["all_tasks_pass"]]
        eligible.sort(
            key=lambda candidate: (
                _as_float(candidate["resident_fraction"]),
                _as_float(candidate["worst_task_h2d"]),
                -_as_float(candidate["worst_task_hit"]),
                -_as_int(candidate["horizon"]),
                _as_int(candidate["budget"]),
            )
        )
        if eligible:
            selected.append(
                {
                    "model": model.key,
                    "horizon": _as_int(eligible[0]["horizon"]),
                    "budget": _as_int(eligible[0]["budget"]),
                    "method": suite.operating_point_selection.hard_oracle_method,
                    "selection_uses_accuracy": False,
                    "selection_rule": list(suite.operating_point_selection.deterministic_order),
                }
            )
    selected_keys = {
        (str(row["model"]), _as_int(row["horizon"]), _as_int(row["budget"])) for row in selected
    }
    for row in task_rows:
        row["selected_for_closed_loop"] = (
            str(row["model"]),
            _as_int(row["horizon"]),
            _as_int(row["budget"]),
        ) in selected_keys
    _write_csv_atomic(output_root / "operating_points.csv", task_rows)
    result: dict[str, object] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "selection_uses_smoke_accuracy": False,
        "selection_requires_all_six_tasks": True,
        "selected": selected,
        "models_without_qualifying_point": [
            model.key
            for model in suite.models
            if model.key not in {row["model"] for row in selected}
        ],
    }
    write_json_atomic(output_root / "selected_operating_points.json", result)
    return result
