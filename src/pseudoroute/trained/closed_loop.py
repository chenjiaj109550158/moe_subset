"""Deterministic trained-model closed-loop evaluation under common routing policies."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from pseudoroute.execution.routing_policy import (
    LosslessFallbackPolicy,
    MaskedSubstitutionPolicy,
    RoutingPolicy,
)
from pseudoroute.models.base import MoEModelAdapter, TraceLevel, TraceRequest
from pseudoroute.oracle.selectors import select_top_b, selected_routing_mass_utility
from pseudoroute.oracle.windows import OracleWindow
from pseudoroute.tracing.validation import validate_trace
from pseudoroute.trained.config import TrainedModelConfig, TrainedSuiteConfig
from pseudoroute.trained.loader import input_device
from pseudoroute.types import ExpertKey, RouterTrace


@dataclass(frozen=True)
class NaturalTrajectory:
    generated_tokens: tuple[int, ...]
    step_traces: tuple[tuple[RouterTrace, ...], ...]
    token_nlls: tuple[float, ...]


@dataclass(frozen=True)
class TrainedClosedLoopRow:
    model: str
    domain: str
    sample_id: str
    policy: str
    information_regime: str
    timing_kind: str
    horizon: int
    budget: int
    generated_tokens: str
    generated_text: str
    natural_tokens: str
    exact_token_agreement: float
    first_divergence: int | None
    nll: float
    perplexity: float
    relative_perplexity_increase: float
    gsm8k_answer: str | None
    gsm8k_ground_truth: str | None
    gsm8k_exact_match: bool | None
    mean_out_of_subset_mass: float
    fallback_frequency: float
    planned_transfer_bytes: int
    fallback_transfer_bytes: int
    estimated_h2d_bytes_per_token: float
    simulated_exposed_stall_ms: float


def _trace_step(
    traces: tuple[RouterTrace, ...], adapter: MoEModelAdapter, position: int
) -> tuple[RouterTrace, ...]:
    selected = tuple(
        sorted(
            (trace for trace in traces if trace.token_position == position),
            key=lambda trace: trace.layer_idx,
        )
    )
    if tuple(trace.layer_idx for trace in selected) != adapter.spec.moe_layer_indices:
        raise RuntimeError("closed-loop trace did not capture every MoE layer")
    return selected


def natural_trajectory(
    adapter: MoEModelAdapter, prompt: Tensor, max_new_tokens: int
) -> NaturalTrajectory:
    cache: object | None = None
    current = prompt
    generated: list[int] = []
    traces: list[tuple[RouterTrace, ...]] = []
    nlls: list[float] = []
    for _step in range(max_new_tokens):
        result = adapter.run_base_forward(
            current,
            kv_cache=cache,
            use_cache=True,
            trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS),
        )
        cache = result.kv_cache
        local_position = current.shape[1] - 1
        traces.append(_trace_step(result.traces, adapter, local_position))
        log_probs = result.logits[:, -1].float().log_softmax(dim=-1)
        token = log_probs.argmax(dim=-1)
        nlls.append(float(-log_probs[0, int(token.item())]))
        generated.append(int(token.item()))
        current = token[:, None].to(input_device(adapter))
    return NaturalTrajectory(tuple(generated), tuple(traces), tuple(nlls))


def _window_from_steps(
    trajectory: NaturalTrajectory,
    adapter: MoEModelAdapter,
    start: int,
    horizon: int,
) -> OracleWindow:
    selected = trajectory.step_traces[start : start + horizon]
    return OracleWindow(
        "closed-loop-natural",
        start,
        horizon,
        torch.stack(
            [torch.stack([trace.topk_ids.reshape(-1) for trace in step]) for step in selected]
        ),
        torch.stack(
            [torch.stack([trace.topk_weights.reshape(-1) for trace in step]) for step in selected]
        ),
        torch.stack(
            [torch.stack([trace.raw_logits.reshape(-1) for trace in step]) for step in selected]
        ),
        adapter.spec.moe_layer_indices,
    )


def _oracle_subset(
    trajectory: NaturalTrajectory,
    adapter: MoEModelAdapter,
    start: int,
    horizon: int,
    budget: int,
) -> frozenset[ExpertKey]:
    window = _window_from_steps(trajectory, adapter, start, horizon)
    count = next(iter(adapter.spec.num_experts_by_layer.values()))
    utilities = selected_routing_mass_utility(window, count, gamma=1.0)
    budgets = {
        layer: min(adapter.spec.num_experts_by_layer[layer], budget)
        for layer in adapter.spec.moe_layer_indices
    }
    subsets = select_top_b(utilities, budgets)
    return frozenset(
        ExpertKey(layer, expert) for layer, experts in subsets.items() for expert in experts
    )


def _static_rankings(model_root: Path, adapter: MoEModelAdapter) -> dict[int, tuple[int, ...]]:
    counts = {
        layer: torch.zeros(adapter.spec.num_experts_by_layer[layer], dtype=torch.float64)
        for layer in adapter.spec.moe_layer_indices
    }
    for domain in ("wikitext", "gsm8k"):
        root = model_root / "traces" / domain
        manifest = validate_trace(root)
        calibration = manifest.shards[: max(1, len(manifest.shards) // 2)]
        for shard in calibration:
            tensors = load_file(str(root / shard.path))
            layer_ids = tuple(int(value) for value in tensors["layer_ids"].tolist())
            for axis, layer in enumerate(layer_ids):
                ids = tensors["router_topk_ids"][:, axis].reshape(-1)
                counts[layer].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
    return {
        layer: tuple(
            sorted(range(values.numel()), key=lambda expert: (-float(values[expert]), expert))
        )
        for layer, values in counts.items()
    }


def _ranked_subset(rankings: dict[int, tuple[int, ...]], budget: int) -> frozenset[ExpertKey]:
    return frozenset(
        ExpertKey(layer, expert)
        for layer, ranking in rankings.items()
        for expert in ranking[:budget]
    )


def _previous_subset(
    records: tuple[Any, ...],
    rankings: dict[int, tuple[int, ...]],
    adapter: MoEModelAdapter,
    budget: int,
) -> frozenset[ExpertKey]:
    by_layer = {record.layer_idx: record for record in records}
    selected: list[ExpertKey] = []
    for layer in adapter.spec.moe_layer_indices:
        previous = (
            [int(value) for value in by_layer[layer].natural_topk_ids.reshape(-1).tolist()]
            if layer in by_layer
            else []
        )
        ranking = previous + [expert for expert in rankings[layer] if expert not in previous]
        selected.extend(ExpertKey(layer, expert) for expert in ranking[:budget])
    return frozenset(selected)


def _extract_answer(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def _ground_truth_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    marker = answer.split("####")[-1]
    return _extract_answer(marker)


def _evaluate_policy(
    suite: TrainedSuiteConfig,
    model: TrainedModelConfig,
    adapter: MoEModelAdapter,
    tokenizer: Any,
    model_root: Path,
    sample: dict[str, Any],
    natural: NaturalTrajectory,
    *,
    policy_name: str,
    budget: int,
    rankings: dict[int, tuple[int, ...]],
) -> TrainedClosedLoopRow:
    prompt = tokenizer(
        sample["rendered_text"],
        return_tensors="pt",
        truncation=True,
        max_length=suite.trace.max_length,
    )["input_ids"].to(input_device(adapter))
    if policy_name == "natural":
        generated = list(natural.generated_tokens)
        nlls = list(natural.token_nlls)
        planned_bytes = 0
        fallback_bytes = 0
        out_masses: list[float] = []
        fallback_count = 0
        route_count = 0
    else:
        policy: RoutingPolicy = (
            LosslessFallbackPolicy()
            if policy_name == "lossless_oracle_residency"
            else MaskedSubstitutionPolicy()
        )
        prefill = adapter.run_base_forward(prompt, use_cache=True)
        cache = prefill.kv_cache
        first_log_probs = prefill.logits[:, -1].float().log_softmax(dim=-1)
        first_token = first_log_probs.argmax(dim=-1)
        generated = [int(first_token.item())]
        reference_first = natural.generated_tokens[0]
        nlls = [float(-first_log_probs[0, reference_first])]
        current = first_token[:, None].to(input_device(adapter))
        resident: frozenset[ExpertKey] = frozenset()
        planned_bytes = 0
        fallback_bytes = 0
        out_masses = []
        fallback_count = 0
        route_count = 0
        previous_records: tuple[Any, ...] = ()
        active_subset: frozenset[ExpertKey] = frozenset()
        for step in range(1, suite.closed_loop.max_new_tokens):
            if (step - 1) % suite.closed_loop.horizon == 0:
                realized = min(
                    suite.closed_loop.horizon,
                    suite.closed_loop.max_new_tokens - step,
                )
                if policy_name in {"lossless_oracle_residency", "hard_oracle_commitment"}:
                    active_subset = _oracle_subset(natural, adapter, step, max(1, realized), budget)
                elif policy_name == "previous_route_commitment":
                    active_subset = _previous_subset(previous_records, rankings, adapter, budget)
                elif policy_name == "static_frequency_commitment":
                    active_subset = _ranked_subset(rankings, budget)
                else:
                    raise ValueError(f"unknown trained policy: {policy_name}")
                loads = active_subset - resident
                planned_bytes += sum(adapter.spec.expert_bytes[key] for key in loads)
                resident = active_subset
            result = adapter.forward_with_policy(
                current,
                policy,
                allowed_experts=active_subset,
                kv_cache=cache,
                use_cache=True,
            )
            cache = result.kv_cache
            previous_records = result.executed_routes
            for record in result.executed_routes:
                route_count += 1
                out_masses.append(record.out_of_subset_mass)
                if record.fallback_used:
                    fallback_count += 1
                    fallback_bytes += sum(
                        adapter.spec.expert_bytes[key] for key in record.missing_natural_experts
                    )
            log_probs = result.logits[:, -1].float().log_softmax(dim=-1)
            reference = natural.generated_tokens[step]
            nlls.append(float(-log_probs[0, reference]))
            token = log_probs.argmax(dim=-1)
            generated.append(int(token.item()))
            current = token[:, None].to(input_device(adapter))
    matches = [
        actual == expected
        for actual, expected in zip(generated, natural.generated_tokens, strict=True)
    ]
    first_divergence = next(
        (i for i, matches_value in enumerate(matches) if not matches_value), None
    )
    nll = sum(nlls) / len(nlls)
    perplexity = math.exp(nll)
    natural_perplexity = math.exp(sum(natural.token_nlls) / len(natural.token_nlls))
    total_transfer = planned_bytes + fallback_bytes
    load_bytes = next(iter(adapter.spec.expert_bytes.values()))
    estimated_loads = total_transfer / max(1, load_bytes)
    stall_seconds = (
        total_transfer / (suite.closed_loop.simulated_bandwidth_gib_s * 1024**3)
        + estimated_loads * suite.closed_loop.simulated_fixed_latency_us / 1e6
    )
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    predicted_answer = _extract_answer(generated_text) if sample["domain"] == "gsm8k" else None
    ground_truth = _ground_truth_answer(sample.get("answer"))
    return TrainedClosedLoopRow(
        model.key,
        sample["domain"],
        sample["sample_id"],
        policy_name,
        "oracle" if "oracle" in policy_name else "online_history_or_static",
        "simulated_transfer_not_measured_runtime",
        suite.closed_loop.horizon,
        budget,
        json.dumps(generated),
        generated_text,
        json.dumps(natural.generated_tokens),
        sum(matches) / len(matches),
        first_divergence,
        nll,
        perplexity,
        perplexity / natural_perplexity - 1,
        predicted_answer,
        ground_truth,
        predicted_answer == ground_truth if ground_truth is not None else None,
        sum(out_masses) / len(out_masses) if out_masses else 0.0,
        fallback_count / route_count if route_count else 0.0,
        planned_bytes,
        fallback_bytes,
        total_transfer / len(generated),
        stall_seconds * 1000,
    )


def run_closed_loop(
    suite: TrainedSuiteConfig,
    model: TrainedModelConfig,
    adapter: MoEModelAdapter,
    tokenizer: Any,
    model_root: Path,
) -> list[TrainedClosedLoopRow]:
    source = json.loads((model_root / "source_samples.json").read_text(encoding="utf-8"))
    by_id = {record["sample_id"]: record for record in source}
    rankings = _static_rankings(model_root, adapter)
    top_k = next(iter(adapter.spec.top_k_by_layer.values()))
    budget = min(
        next(iter(adapter.spec.num_experts_by_layer.values())),
        int(math.ceil(suite.closed_loop.budget_multiple * top_k)),
    )
    rows = []
    for sample_id in suite.closed_loop.sample_ids:
        sample = by_id[sample_id]
        prompt = tokenizer(
            sample["rendered_text"],
            return_tensors="pt",
            truncation=True,
            max_length=suite.trace.max_length,
        )["input_ids"].to(input_device(adapter))
        natural = natural_trajectory(adapter, prompt, suite.closed_loop.max_new_tokens)
        for policy in (
            "natural",
            "lossless_oracle_residency",
            "hard_oracle_commitment",
            "previous_route_commitment",
            "static_frequency_commitment",
        ):
            rows.append(
                _evaluate_policy(
                    suite,
                    model,
                    adapter,
                    tokenizer,
                    model_root,
                    sample,
                    natural,
                    policy_name=policy,
                    budget=budget,
                    rankings=rankings,
                )
            )
    path = model_root / "closed_loop.csv"
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    worst = sorted(rows, key=lambda row: (row.exact_token_agreement, -row.perplexity))[:20]
    (model_root / "closed_loop_worst_cases.json").write_text(
        json.dumps([asdict(row) for row in worst], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows
