"""True cached closed-loop generation for fixed-window oracle and hard subsets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from pseudoroute.benchmark.config import AccuracyModelConfig, AccuracySuiteConfig
from pseudoroute.benchmark.prefetch import (
    NativeRouteCaptureContext,
    PrefetchModelOps,
    SubsetExecutionContext,
    SubsetRouteRecord,
    build_prefetch_ops,
    load_default_vectors,
    physical_expert_bytes,
)
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
    _model_example,
    _read_jsonl,
)
from pseudoroute.benchmark.scoring import score_response
from pseudoroute.benchmark.subset_config import SubsetModelConfig, SubsetOracleSuiteConfig
from pseudoroute.benchmark.subset_trace import sha256_json, write_json_atomic
from pseudoroute.benchmark.tasks import BenchmarkExample, load_examples
from pseudoroute.utils.determinism import seed_everything

ActualPolicy = Literal[
    "lossless_oracle_residency", "hard_oracle_commitment", "previous_route_commitment"
]


@dataclass
class RouteAccounting:
    route_hits: int = 0
    route_slots: int = 0
    selected_mass_hit: float = 0.0
    selected_mass_total: float = 0.0
    route_calls: int = 0
    route_divergent_calls: int = 0
    first_route_divergence: int | None = None
    prefetch_loads: int = 0
    fallback_loads: int = 0
    subset_churn: int = 0
    planned_transfer_bytes: int = 0
    fallback_transfer_bytes: int = 0
    natural_reference_bytes: int = 0


def _source_model(accuracy: AccuracySuiteConfig, model: SubsetModelConfig) -> AccuracyModelConfig:
    matches = [candidate for candidate in accuracy.models if candidate.key == model.key]
    if len(matches) != 1:
        raise ValueError(f"missing source model {model.key}")
    return matches[0]


def _source_rows(root: Path, model: str, task: str) -> list[dict[str, Any]]:
    path = root / "models" / model / "results" / "vanilla" / task / "samples.jsonl"
    rows = [row for row in _read_jsonl(path) if row.get("state") == "complete"]
    return sorted(rows, key=lambda row: int(row["row_index"]))


def _top_b(scores: Tensor, budget: int) -> tuple[int, ...]:
    ranking = sorted(range(scores.numel()), key=lambda expert: (-float(scores[expert]), expert))
    return tuple(sorted(ranking[:budget]))


def _static_subsets(
    suite: SubsetOracleSuiteConfig, model: SubsetModelConfig, budget: int
) -> dict[int, tuple[int, ...]]:
    root = Path(suite.source_accuracy.artifact_root) / "models" / model.key / "default_vectors"
    counts = load_default_vectors(root).count.double()
    if tuple(counts.shape) != (model.routed_layers, model.routed_experts_per_layer):
        raise ValueError(f"static frequency shape mismatch for {model.key}")
    return {layer: _top_b(counts[layer], budget) for layer in range(model.routed_layers)}


def subsets_from_route_steps(
    steps: list[tuple[SubsetRouteRecord, ...]],
    ops: PrefetchModelOps,
    budget: int,
) -> dict[int, tuple[int, ...]]:
    scores = {
        layer: torch.zeros(ops.num_experts, dtype=torch.float64) for layer in range(ops.num_layers)
    }
    for records in steps:
        if tuple(record.layer for record in records) != tuple(range(ops.num_layers)):
            raise RuntimeError("route step did not retain every layer in order")
        for record in records:
            ids = record.natural.ids.reshape(-1).long()
            weights = record.natural.weights.reshape(-1).double()
            scores[record.layer].scatter_add_(0, ids, weights)
    return {layer: _top_b(values, budget) for layer, values in scores.items()}


def _sample_token(logits: Tensor, model: nn.Module, *, do_sample: bool) -> Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    generation = cast(Any, model).generation_config
    scores = logits.float()
    temperature = getattr(generation, "temperature", None)
    if temperature is not None and float(temperature) != 1.0:
        if float(temperature) <= 0:
            raise ValueError("sampling temperature must be positive")
        scores = scores / float(temperature)
    top_k = int(getattr(generation, "top_k", 0) or 0)
    if 0 < top_k < scores.shape[-1]:
        threshold = scores.topk(top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    top_p = float(getattr(generation, "top_p", 1.0) or 1.0)
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_original = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        scores = scores.masked_fill(remove_original, -torch.inf)
    probabilities = scores.softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1)


def _eos_ids(model: nn.Module) -> set[int]:
    value = getattr(cast(Any, model).generation_config, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def _finished(
    model: nn.Module,
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    example: BenchmarkExample,
    prompt_ids: Tensor,
    generated: list[int],
    scores: Tensor,
) -> bool:
    if generated[-1] in _eos_ids(model):
        return True
    if not model_config.honor_task_stop_strings or not example.stop_strings:
        return False
    from transformers import StopStringCriteria

    criteria = StopStringCriteria(tokenizer, list(example.stop_strings))
    continuation = torch.tensor(generated, dtype=torch.long, device=prompt_ids.device)[None]
    sequence = torch.cat((prompt_ids, continuation), dim=-1)
    result = criteria(sequence, scores)
    return bool(cast(Tensor, result).item())


def _cache_length(cache: object) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if getter is None or not callable(getter):
        raise RuntimeError(f"cache {type(cache).__name__} does not expose get_seq_length")
    return int(getter())


def _rewind_cache(cache: object, length: int) -> None:
    crop = getattr(cache, "crop", None)
    if crop is None or not callable(crop):
        raise RuntimeError(f"cache {type(cache).__name__} does not support exact crop rewind")
    crop(length)
    if _cache_length(cache) != length:
        raise RuntimeError("cache rewind length mismatch")


def _cache_layers(cache: object) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"cache {type(cache).__name__} does not expose populated layers")
    return layers


def _cache_has_sliding_layers(cache: object) -> bool:
    return any(getattr(layer, "sliding_window", None) is not None for layer in _cache_layers(cache))


def _tensor_mutation_signature(value: object) -> tuple[object, ...]:
    if not isinstance(value, Tensor):
        return (None,)
    try:
        version: int | None = int(value._version)
    except RuntimeError:
        version = None
    return (id(value), int(value.data_ptr()), tuple(value.shape), version)


def _cache_mutation_signature(cache: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            id(layer),
            _tensor_mutation_signature(getattr(layer, "keys", None)),
            _tensor_mutation_signature(getattr(layer, "values", None)),
            getattr(layer, "cumulative_length", None),
        )
        for layer in _cache_layers(cache)
    )


def _fork_cache_copy_on_write(cache: object) -> object:
    layers = _cache_layers(cache)
    fork = copy.copy(cache)
    fork_layers = [copy.copy(layer) for layer in layers]
    if any(original is cloned for original, cloned in zip(layers, fork_layers, strict=True)):
        raise RuntimeError("cache fork reused a mutable layer object")
    cast(Any, fork).layers = fork_layers
    if len(_cache_layers(fork)) != len(layers):
        raise RuntimeError("cache fork layer count changed")
    if _cache_length(fork) != _cache_length(cache):
        raise RuntimeError("cache fork sequence length changed")
    return fork


def _rng_state(device: torch.device) -> tuple[Tensor, Tensor]:
    return torch.random.get_rng_state(), torch.cuda.get_rng_state(device)


def _restore_rng(device: torch.device, state: tuple[Tensor, Tensor]) -> None:
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], device)


def _forward_capture(
    model: nn.Module,
    ops: PrefetchModelOps,
    current: Tensor,
    cache: object,
    *,
    policy: Literal["natural", "lossless", "hard"],
    allowed: dict[int, tuple[int, ...]] | None = None,
) -> tuple[Any, tuple[SubsetRouteRecord, ...]]:
    context: NativeRouteCaptureContext | SubsetExecutionContext
    if policy == "hard":
        context = SubsetExecutionContext(ops, policy, allowed)
    else:
        context = NativeRouteCaptureContext(ops, allowed)
    with context, torch.inference_mode():
        output = cast(
            Any,
            model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            ),
        )
        if output.past_key_values is not cache:
            raise RuntimeError("closed-loop requires an in-place rewindable cache object")
        records = context.drain()
    if tuple(record.layer for record in records) != tuple(range(ops.num_layers)):
        raise RuntimeError("closed-loop forward did not capture every routed layer")
    return output, records


def natural_lookahead_and_rewind(
    model: nn.Module,
    tokenizer: Any,
    ops: PrefetchModelOps,
    model_config: AccuracyModelConfig,
    example: BenchmarkExample,
    prompt_ids: Tensor,
    generated: list[int],
    current: Tensor,
    cache: object,
    *,
    horizon: int,
    do_sample: bool,
) -> list[tuple[SubsetRouteRecord, ...]]:
    """Roll out naturally, then restore or preserve the boundary cache and RNG."""
    device = current.device
    boundary_length = _cache_length(cache)
    state = _rng_state(device)
    use_fork = _cache_has_sliding_layers(cache)
    boundary_signature = _cache_mutation_signature(cache) if use_fork else ()
    rollout_cache = _fork_cache_copy_on_write(cache) if use_fork else cache
    rollout_current = current
    rollout_generated = list(generated)
    steps: list[tuple[SubsetRouteRecord, ...]] = []
    try:
        for _ in range(horizon):
            output, records = _forward_capture(
                model, ops, rollout_current, rollout_cache, policy="natural"
            )
            steps.append(records)
            scores = cast(Tensor, output.logits[:, -1])
            token = _sample_token(scores, model, do_sample=do_sample)
            rollout_generated.append(int(token.item()))
            rollout_current = token[:, None].to(device)
            if _finished(
                model,
                tokenizer,
                model_config,
                example,
                prompt_ids,
                rollout_generated,
                scores,
            ):
                break
    finally:
        try:
            if use_fork:
                if _cache_length(cache) != boundary_length:
                    raise RuntimeError("forked lookahead changed original cache length")
                if _cache_mutation_signature(cache) != boundary_signature:
                    raise RuntimeError("forked lookahead mutated original cache state")
            else:
                _rewind_cache(cache, boundary_length)
        finally:
            _restore_rng(device, state)
    if not steps:
        raise RuntimeError("oracle natural lookahead produced no route step")
    return steps


def _account_records(
    accounting: RouteAccounting,
    records: tuple[SubsetRouteRecord, ...],
    allowed: dict[int, tuple[int, ...]],
    expert_bytes: dict[int, int],
    *,
    policy: ActualPolicy,
    decode_step: int,
    natural_route_digest: Any,
    executed_route_digest: Any,
) -> None:
    for record in records:
        allowed_tensor = torch.tensor(allowed[record.layer], dtype=torch.long)
        for digest, route in (
            (natural_route_digest, record.natural),
            (executed_route_digest, record.executed),
        ):
            digest.update(record.layer.to_bytes(4, "little"))
            digest.update(str(tuple(route.ids.shape)).encode())
            digest.update(route.ids.contiguous().numpy().tobytes())
        natural_ids = record.natural.ids.reshape(-1)
        mask = torch.isin(natural_ids, allowed_tensor)
        hits = int(mask.sum())
        slots = int(natural_ids.numel())
        accounting.route_hits += hits
        accounting.route_slots += slots
        accounting.selected_mass_hit += float(
            record.natural.weights.reshape(-1).masked_select(mask).double().sum()
        )
        accounting.selected_mass_total += float(record.natural.weights.double().sum())
        accounting.route_calls += 1
        divergent = not torch.equal(record.natural.ids, record.executed.ids)
        if divergent:
            accounting.route_divergent_calls += 1
            if accounting.first_route_divergence is None:
                accounting.first_route_divergence = decode_step
        missing = slots - hits
        accounting.natural_reference_bytes += slots * expert_bytes[record.layer]
        if policy == "lossless_oracle_residency":
            accounting.fallback_loads += missing
            accounting.fallback_transfer_bytes += missing * expert_bytes[record.layer]


def _token_agreement(generated: list[int], natural: list[int]) -> tuple[float, int | None]:
    total = max(len(generated), len(natural))
    matches = sum(
        index < len(generated) and index < len(natural) and generated[index] == natural[index]
        for index in range(total)
    )
    first = next(
        (
            index
            for index in range(total)
            if index >= len(generated)
            or index >= len(natural)
            or generated[index] != natural[index]
        ),
        None,
    )
    return matches / total if total else 1.0, first


def _perplexity(nll: float) -> float:
    return math.exp(nll) if nll < 700 else float("inf")


def run_policy_sample(
    suite: SubsetOracleSuiteConfig,
    accuracy: AccuracySuiteConfig,
    model_subset: SubsetModelConfig,
    model_config: AccuracyModelConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: PrefetchModelOps,
    example: BenchmarkExample,
    source: dict[str, Any],
    *,
    policy: ActualPolicy,
    horizon: int,
    budget: int,
    max_new_tokens: int,
) -> dict[str, object]:
    rendered = str(source["rendered_prompt"])
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, rendered)
    source_tokens = [int(value) for value in source["generated_token_ids"]][:max_new_tokens]
    do_sample = accuracy.do_sample_for(model_config, example.task)
    seed_everything(accuracy.decode.seed)
    torch.cuda.reset_peak_memory_stats(torch.device(model_config.device))
    started = time.time()
    with torch.inference_mode():
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("closed-loop model did not return a KV cache")
    cache_replay_semantics = (
        "copy_on_write_layer_fork_discard_original_boundary_verified"
        if _cache_has_sliding_layers(cache)
        else "in_place_lookahead_exact_crop_rewind_verified"
    )
    prefill_scores = cast(Tensor, prefill.logits[:, -1])
    first = _sample_token(prefill_scores, model, do_sample=do_sample)
    generated = [int(first.item())]
    nlls = []
    if source_tokens:
        nlls.append(float(-prefill_scores.float().log_softmax(dim=-1)[0, source_tokens[0]].item()))
    prompt_ids = inputs["input_ids"]
    current = first[:, None].to(prompt_ids.device)
    accounting = RouteAccounting()
    expert_bytes = {layer: physical_expert_bytes(ops, layer) for layer in range(ops.num_layers)}
    static = _static_subsets(suite, model_subset, budget)
    natural_route_digest = hashlib.sha256()
    executed_route_digest = hashlib.sha256()
    subset_digest = hashlib.sha256()
    active = static
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(ops.num_layers)}
    history: list[tuple[SubsetRouteRecord, ...]] = []
    decode_step = 0
    finished = _finished(
        model, tokenizer, model_config, example, prompt_ids, generated, prefill_scores
    )
    while len(generated) < max_new_tokens and not finished:
        if decode_step % horizon == 0:
            realized = min(horizon, max_new_tokens - len(generated))
            if policy in {"lossless_oracle_residency", "hard_oracle_commitment"}:
                future = natural_lookahead_and_rewind(
                    model,
                    tokenizer,
                    ops,
                    model_config,
                    example,
                    prompt_ids,
                    generated,
                    current,
                    cache,
                    horizon=max(1, realized),
                    do_sample=do_sample,
                )
                active = subsets_from_route_steps(future, ops, budget)
            elif history:
                active = subsets_from_route_steps(history[-horizon:], ops, budget)
            else:
                active = static
            for layer in range(ops.num_layers):
                old = set(resident[layer])
                new = set(active[layer])
                loads = len(new - old)
                accounting.prefetch_loads += loads
                accounting.subset_churn += len(new.symmetric_difference(old))
                accounting.planned_transfer_bytes += loads * expert_bytes[layer]
                resident[layer] = active[layer]
                subset_digest.update(layer.to_bytes(4, "little"))
                subset_digest.update(
                    torch.tensor(active[layer], dtype=torch.int64).numpy().tobytes()
                )
        execution_policy: Literal["lossless", "hard"] = (
            "lossless" if policy == "lossless_oracle_residency" else "hard"
        )
        output, records = _forward_capture(
            model,
            ops,
            current,
            cache,
            policy=execution_policy,
            allowed=active,
        )
        history.append(records)
        _account_records(
            accounting,
            records,
            active,
            expert_bytes,
            policy=policy,
            natural_route_digest=natural_route_digest,
            executed_route_digest=executed_route_digest,
            decode_step=decode_step,
        )
        scores = cast(Tensor, output.logits[:, -1])
        source_index = len(generated)
        if source_index < len(source_tokens):
            nlls.append(
                float(-scores.float().log_softmax(dim=-1)[0, source_tokens[source_index]].item())
            )
        token = _sample_token(scores, model, do_sample=do_sample)
        generated.append(int(token.item()))
        current = token[:, None].to(prompt_ids.device)
        decode_step += 1
        finished = _finished(model, tokenizer, model_config, example, prompt_ids, generated, scores)
    elapsed = time.time() - started
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    score = score_response(example, generated_text)
    agreement, first_divergence = _token_agreement(generated, source_tokens)
    nll = sum(nlls) / len(nlls) if nlls else float("nan")
    total_bytes = accounting.planned_transfer_bytes + accounting.fallback_transfer_bytes
    loads = accounting.prefetch_loads + accounting.fallback_loads
    simulated_stall_ms = (
        total_bytes / (suite.transfer_model.bandwidth_gib_per_second * 1024**3)
        + loads * suite.transfer_model.fixed_latency_microseconds_per_load / 1e6
    ) * 1000
    return {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "model": model_subset.key,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "precision_tier": model_config.precision_tier,
        "task": example.task,
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "policy": policy,
        "information_regime": (
            "routing_information_oracle"
            if policy in {"lossless_oracle_residency", "hard_oracle_commitment"}
            else "online_previous_route"
        ),
        "evaluation_mode": "actual_closed_loop_generation",
        "cache_replay_semantics": cache_replay_semantics,
        "horizon": horizon,
        "budget": budget,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "generated_tokens": len(generated),
        "v17_natural_token_ids": source_tokens,
        "exact_token_agreement": agreement,
        "first_token_divergence": first_divergence,
        "first_route_divergence": accounting.first_route_divergence,
        "route_divergent_calls": accounting.route_divergent_calls,
        "route_calls": accounting.route_calls,
        "executed_route_changed": accounting.route_divergent_calls > 0,
        "natural_route_ids_sha256": natural_route_digest.hexdigest(),
        "executed_route_ids_sha256": executed_route_digest.hexdigest(),
        "subset_trajectory_sha256": subset_digest.hexdigest(),
        "route_hit_rate": (
            accounting.route_hits / accounting.route_slots if accounting.route_slots else 1.0
        ),
        "selected_routing_mass_coverage": (
            accounting.selected_mass_hit / accounting.selected_mass_total
            if accounting.selected_mass_total
            else 1.0
        ),
        "fallback_frequency": (
            1 - accounting.route_hits / accounting.route_slots
            if policy == "lossless_oracle_residency" and accounting.route_slots
            else 0.0
        ),
        "prefetch_loads": accounting.prefetch_loads,
        "fallback_loads": accounting.fallback_loads,
        "subset_churn": accounting.subset_churn,
        "planned_transfer_bytes": accounting.planned_transfer_bytes,
        "fallback_transfer_bytes": accounting.fallback_transfer_bytes,
        "total_transfer_bytes": total_bytes,
        "natural_reference_bytes": accounting.natural_reference_bytes,
        "estimated_transfer_reduction": (
            1 - total_bytes / accounting.natural_reference_bytes
            if accounting.natural_reference_bytes
            else 0.0
        ),
        "simulated_exposed_stall_ms": simulated_stall_ms,
        "transfer_timing_kind": suite.transfer_model.timing_kind,
        "vanilla_token_nll_on_policy_context": nll,
        "perplexity": _perplexity(nll),
        "nll_tokens": len(nlls),
        "correct": score.correct,
        "parsed_answer": score.parsed_answer,
        "score_detail": score.detail,
        "elapsed_seconds_measured": elapsed,
        "tokens_per_second_measured": len(generated) / elapsed if elapsed else None,
        "runtime_kind": "measured_closed_loop_including_oracle_lookahead",
        "peak_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(torch.device(model_config.device))
        ),
        "physical_gpu": model_subset.physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "source_v17_row_sha256": sha256_json(source),
        "logical_shard": int(source["row_index"]) % suite.closed_loop.sample_shards,
    }


def _sample_path(
    root: Path,
    task: str,
    row_index: int,
    policy: str,
    *,
    stage: Literal["mechanism_smoke", "selected_smoke", "full"],
) -> Path:
    return root / "closed_loop" / stage / task / f"{row_index:05d}" / f"{policy}.json"


def _load_valid_sample(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if row.get("state") != "complete" or row.get("config_fingerprint") != fingerprint:
        raise ValueError(f"incompatible closed-loop sample artifact: {path}")
    return row


def _scoped_source_rows(
    suite: SubsetOracleSuiteConfig,
    model: SubsetModelConfig,
    task: str,
    *,
    shard_index: int,
    smoke: bool,
) -> list[dict[str, Any]]:
    rows = _source_rows(Path(suite.source_accuracy.artifact_root), model.key, task)
    if smoke:
        smoke_id = suite.closed_loop.smoke_rows[cast(Any, task)]
        rows = [row for row in rows if row["sample_id"] == smoke_id]
        if len(rows) != 1:
            raise ValueError(f"missing frozen smoke row {model.key}/{task}")
        return rows
    rows = [
        row
        for row in rows
        if int(row["row_index"]) % suite.closed_loop.sample_shards == shard_index
    ]
    if not rows:
        raise ValueError(f"empty frozen full shard {model.key}/{task}/{shard_index}")
    return rows


def run_model_closed_loop(
    suite: SubsetOracleSuiteConfig,
    accuracy: AccuracySuiteConfig,
    model_subset: SubsetModelConfig,
    output_root: Path,
    *,
    horizon: int,
    budget: int,
    policies: tuple[ActualPolicy, ...],
    shard_index: int,
    stage: Literal["mechanism_smoke", "selected_smoke", "full"],
    physical_gpu: int | None = None,
) -> list[dict[str, object]]:
    if shard_index < 0 or shard_index >= suite.closed_loop.sample_shards:
        raise ValueError("closed-loop shard index is outside the frozen shard count")
    actual_physical_gpu = model_subset.physical_gpu if physical_gpu is None else physical_gpu
    if actual_physical_gpu not in (0, 1):
        raise ValueError(f"unsupported physical GPU: {actual_physical_gpu}")
    model_subset = model_subset.model_copy(update={"physical_gpu": actual_physical_gpu})
    model_config = _source_model(accuracy, model_subset).model_copy(
        update={"device": f"cuda:{actual_physical_gpu}"}
    )
    smoke = stage != "full"
    resumed: list[dict[str, object]] = []
    all_complete = True
    for dataset in accuracy.datasets:
        for source in _scoped_source_rows(
            suite,
            model_subset,
            dataset.key,
            shard_index=shard_index,
            smoke=smoke,
        ):
            for policy in policies:
                path = _sample_path(
                    output_root / "models" / model_subset.key,
                    dataset.key,
                    int(source["row_index"]),
                    policy,
                    stage=stage,
                )
                existing = _load_valid_sample(path, suite.fingerprint())
                if existing is None:
                    all_complete = False
                else:
                    resumed.append(cast(dict[str, object], existing))
    if all_complete:
        return resumed
    seed_everything(accuracy.decode.seed)
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = build_prefetch_ops(model, model_config.architecture)
    completed_rows: list[dict[str, object]] = []
    for dataset in accuracy.datasets:
        examples = load_examples(dataset, cache_dir=accuracy.dataset_cache_dir)
        source_rows = _scoped_source_rows(
            suite, model_subset, dataset.key, shard_index=shard_index, smoke=smoke
        )
        for source in source_rows:
            row_index = int(source["row_index"])
            example = _model_example(examples[row_index], model_config)
            if example.sample_id != source["sample_id"]:
                raise ValueError(f"dataset/v17 sample mismatch at {dataset.key}/{row_index}")
            override = model_config.max_new_tokens_overrides.get(dataset.key)
            task_cap = override if override is not None else example.max_new_tokens
            max_tokens = (
                min(task_cap, suite.closed_loop.smoke_max_new_tokens) if smoke else task_cap
            )
            example = replace(example, max_new_tokens=max_tokens)
            for policy in policies:
                path = _sample_path(
                    output_root / "models" / model_subset.key,
                    dataset.key,
                    row_index,
                    policy,
                    stage=stage,
                )
                existing = _load_valid_sample(path, suite.fingerprint())
                if existing is not None:
                    completed_rows.append(cast(dict[str, object], existing))
                    continue
                try:
                    row = run_policy_sample(
                        suite,
                        accuracy,
                        model_subset,
                        model_config,
                        model,
                        tokenizer,
                        ops,
                        example,
                        source,
                        policy=policy,
                        horizon=horizon,
                        budget=budget,
                        max_new_tokens=max_tokens,
                    )
                    write_json_atomic(path, row)
                    completed_rows.append(row)
                    print(
                        json.dumps(
                            {
                                "stage": f"closed_loop_{stage}",
                                "model": model_subset.key,
                                "task": dataset.key,
                                "row_index": row_index,
                                "policy": policy,
                                "tokens": row["generated_tokens"],
                                "correct": row["correct"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except BaseException as error:
                    failure = {
                        "state": "failed",
                        "suite_id": suite.suite_id,
                        "config_fingerprint": suite.fingerprint(),
                        "model": model_subset.key,
                        "task": dataset.key,
                        "row_index": row_index,
                        "sample_id": source["sample_id"],
                        "policy": policy,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                        "pid": os.getpid(),
                        "ppid": os.getppid(),
                        "physical_gpu": model_subset.physical_gpu,
                    }
                    write_json_atomic(path.with_name(f"{policy}.FAILED.json"), failure)
                    raise
    return completed_rows
