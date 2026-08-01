"""Sample-atomic teacher-forced route evaluation for the focused Qwen pilot."""

from __future__ import annotations

import json
import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from pseudoroute.benchmark.config import (
    AccuracyModelConfig,
    AccuracySuiteConfig,
    load_accuracy_suite_config,
)
from pseudoroute.benchmark.prefetch import (
    Qwen3MoePrefetchOps,
    SubsetRouteRecord,
    load_default_vectors,
    physical_expert_bytes,
)
from pseudoroute.benchmark.pseudo_embedding_config import (
    PseudoEmbeddingSuiteConfig,
    SampleManifest,
    load_sample_manifest,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoProbeResult,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
    _read_jsonl,
)
from pseudoroute.benchmark.subset_closed_loop import _forward_capture
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.utils.determinism import seed_everything


def _source_model(accuracy: AccuracySuiteConfig, physical_gpu: int) -> AccuracyModelConfig:
    matches = [model for model in accuracy.models if model.key == "qwen3_30b_a3b"]
    if len(matches) != 1:
        raise ValueError("the v17 source is missing the frozen Qwen model")
    if physical_gpu not in (0, 1):
        raise ValueError("physical GPU must be zero or one")
    return matches[0].model_copy(update={"device": f"cuda:{physical_gpu}"})


def _source_rows(suite: PseudoEmbeddingSuiteConfig) -> dict[int, dict[str, Any]]:
    path = (
        Path(suite.source_accuracy.artifact_root)
        / "models/qwen3_30b_a3b/results/vanilla/gsm8k/samples.jsonl"
    )
    rows = {
        int(row["row_index"]): row for row in _read_jsonl(path) if row.get("state") == "complete"
    }
    if set(rows) != set(range(suite.dataset.frozen_rows)):
        raise ValueError("the frozen Qwen/GSM8K v17 source row set changed")
    return rows


def _variants(suite: PseudoEmbeddingSuiteConfig) -> tuple[QwenPseudoVariant, ...]:
    mandatory = tuple(
        QwenPseudoVariant(
            variant.key,
            variant.content,
            variant.attention,
            variant.expert_contribution,
        )
        for variant in suite.variants
    )
    expected = QwenPseudoVariant(
        "expected_top8_independent_default_topk",
        "expected_top_m",
        "independent",
        "default_vector_selected_topk_mixture",
    )
    return (*mandatory, expected)


def _top_b(scores: Tensor, budget: int) -> tuple[int, ...]:
    ranking = sorted(
        range(scores.numel()),
        key=lambda expert: (-float(scores[expert]), expert),
    )
    return tuple(sorted(ranking[:budget]))


def _scores_from_records(
    records: list[tuple[SubsetRouteRecord, ...]],
    experts: int,
) -> dict[int, Tensor]:
    scores = {layer: torch.zeros(experts, dtype=torch.float64) for layer in range(len(records[0]))}
    for step in records:
        for record in step:
            scores[record.layer].scatter_add_(
                0,
                record.natural.ids.reshape(-1).long(),
                record.natural.weights.reshape(-1).double(),
            )
    return scores


def _subsets_from_records(
    records: list[tuple[SubsetRouteRecord, ...]],
    experts: int,
    budget: int,
) -> dict[int, tuple[int, ...]]:
    return {
        layer: _top_b(scores, budget)
        for layer, scores in _scores_from_records(records, experts).items()
    }


def _route_metrics(
    *,
    sample_id: str,
    boundary: int,
    layer: int,
    method: str,
    ids: Tensor,
    weights: Tensor,
    logits: Tensor,
    subset: tuple[int, ...],
    previous_subset: tuple[int, ...],
    expert_bytes: int,
) -> dict[str, object]:
    subset_tensor = torch.tensor(subset, dtype=torch.long)
    mask = torch.isin(ids.cpu(), subset_tensor)
    hits = int(mask.sum())
    slots = int(ids.numel())
    selected_hit = float(weights.cpu().masked_select(mask).double().sum())
    selected_total = float(weights.double().sum())
    probabilities = logits.double().softmax(dim=-1)
    full_hit = float(probabilities[:, list(subset)].sum())
    full_total = float(probabilities.sum())
    prefetch_loads = len(set(subset) - set(previous_subset))
    fallback_loads = slots - hits
    lossless_bytes = (prefetch_loads + fallback_loads) * expert_bytes
    reference_bytes = slots * expert_bytes
    top = logits.float().topk(9, dim=-1).values
    margin = float((top[:, 7] - top[:, 8]).mean())
    return {
        "sample_id": sample_id,
        "boundary": boundary,
        "realized_tokens": int(ids.shape[0]),
        "layer": layer,
        "method": method,
        "budget": len(subset),
        "resident_fraction": len(subset) / 128,
        "context_bucket": ("0-31" if boundary < 32 else "32-63" if boundary < 64 else "64-127"),
        "router_margin": margin,
        "route_hits": hits,
        "route_slots": slots,
        "route_hit_rate": hits / slots,
        "selected_mass_hit": selected_hit,
        "selected_mass_total": selected_total,
        "selected_mass_coverage": selected_hit / selected_total,
        "full_mass_hit": full_hit,
        "full_mass_total": full_total,
        "full_mass_coverage": full_hit / full_total,
        "fallback_loads": fallback_loads,
        "fallback_frequency": 1 - hits / slots,
        "prefetch_loads": prefetch_loads,
        "subset_churn": len(set(subset).symmetric_difference(previous_subset)),
        "lossless_transfer_bytes": lossless_bytes,
        "natural_reference_bytes": reference_bytes,
        "estimated_transfer_reduction": (
            1 - lossless_bytes / reference_bytes if reference_bytes else 0.0
        ),
        "subset": list(subset),
    }


def _stack_probe_tensor(
    results: list[QwenPseudoProbeResult],
    attribute: str,
) -> Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    cast(dict[int, Tensor], getattr(result, attribute))[layer]
                    for layer in range(len(result.raw_router_logits))
                ]
            )
            for result in results
        ]
    )


def _save_tensors_atomic(path: Path, tensors: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def _probe_cost(result: QwenPseudoProbeResult, boundary: int) -> dict[str, object]:
    return {
        "boundary": boundary,
        "variant": result.variant,
        **result.cost.__dict__,
    }


def _audit_summary(result: QwenPseudoProbeResult, boundary: int) -> dict[str, object]:
    return {
        "boundary": boundary,
        "variant": result.variant,
        **result.audit,
    }


def _development_trace(
    suite: PseudoEmbeddingSuiteConfig,
    row_index: int,
) -> tuple[Path, dict[str, Any]]:
    root = Path("artifacts/benchmark_subset_oracle_v1_r2_authoritative")
    manifest = cast(
        dict[str, Any],
        json.loads(
            (root / "natural_traces/qwen3_30b_a3b/manifest.json").read_text(encoding="utf-8")
        ),
    )
    matches = [
        record
        for record in manifest["shards"]
        if record["task"] == suite.dataset.key and int(record["row_index"]) == row_index
    ]
    if len(matches) != 1:
        raise ValueError(f"missing authoritative development trace row {row_index}")
    path = root / "natural_traces/qwen3_30b_a3b" / str(matches[0]["path"])
    if path.stat().st_size != int(matches[0]["bytes"]) or sha256_file(path) != matches[0]["sha256"]:
        raise ValueError(f"authoritative development trace checksum changed: {path}")
    return path, cast(dict[str, Any], matches[0])


def _validate_development_parity(
    suite: PseudoEmbeddingSuiteConfig,
    row_index: int,
    natural: dict[str, Tensor],
) -> dict[str, object]:
    path, record = _development_trace(suite, row_index)
    authoritative = load_file(str(path))
    if not torch.equal(natural["token_ids"], authoritative["token_ids"]):
        raise RuntimeError("development replay token IDs changed")
    ids_equal = torch.equal(
        natural["router_topk_ids"],
        authoritative["router_topk_ids"],
    )
    max_logits = float(
        (natural["router_logits"] - authoritative["router_logits"].float()).abs().max()
    )
    max_weights = float(
        (natural["router_topk_weights"] - authoritative["router_topk_weights"].float()).abs().max()
    )
    return {
        "authoritative_trace_path": str(path),
        "authoritative_trace_sha256": record["sha256"],
        "authoritative_route_tensors_used_for_scoring": True,
        "current_replay_route_ids_equal": ids_equal,
        "current_replay_max_router_logit_delta": max_logits,
        "current_replay_max_router_weight_delta": max_weights,
        "current_replay_within_authoritative_tolerance": (
            ids_equal and max_logits <= 1e-3 and max_weights <= 1e-3
        ),
        "cross_process_bfloat16_drift_is_not_scoring_input": True,
    }


def run_route_sample(
    suite: PseudoEmbeddingSuiteConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    source: dict[str, Any],
    probes: tuple[QwenPseudoEmbeddingProbe, ...],
    static_subsets: dict[int, tuple[int, ...]],
    *,
    partition: str,
    max_route_tokens: int,
    physical_gpu: int,
    anchor_token_provider: (
        Callable[
            [QwenPseudoVariant, int, tuple[int, ...], tuple[int, ...]],
            tuple[int, ...] | None,
        ]
        | None
    ) = None,
    include_reference_methods: bool = True,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    started = time.time()
    rendered = str(source["rendered_prompt"])
    source_tokens = [int(value) for value in source["generated_token_ids"]]
    route_tokens = min(max_route_tokens, len(source_tokens) - 1)
    if route_tokens < 1:
        raise ValueError("route replay requires at least two saved generated tokens")
    authoritative: dict[str, Tensor] | None = None
    if partition == "development":
        authoritative_path, _ = _development_trace(
            suite,
            int(source["row_index"]),
        )
        loaded = load_file(str(authoritative_path))
        authoritative = {
            "token_ids": loaded["token_ids"][:route_tokens],
            "router_logits": loaded["router_logits"][:route_tokens],
            "router_topk_ids": loaded["router_topk_ids"][:route_tokens],
            "router_topk_weights": loaded["router_topk_weights"][:route_tokens],
        }
        if not torch.equal(
            authoritative["token_ids"],
            torch.tensor(source_tokens[:route_tokens], dtype=torch.int64),
        ):
            raise RuntimeError("authoritative development tokens differ from frozen v17")
    model_config = _source_model(
        load_accuracy_suite_config(suite.source_accuracy.config),
        physical_gpu,
    )
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, rendered)
    prompt_token_ids = tuple(int(value) for value in inputs["input_ids"][0].tolist())
    with torch.inference_mode():
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("Qwen route replay did not return a cache")
    prefill_argmax = int(prefill.logits[:, -1].argmax())
    argmax_matches = int(prefill_argmax == source_tokens[0])
    argmax_compared = 1
    first_argmax_divergence = None if argmax_matches else 0
    current_token_id = int(inputs["input_ids"][0, -1])
    planning_logits = cast(Tensor, prefill.logits[:, -1]).detach()
    natural_steps: list[tuple[SubsetRouteRecord, ...]] = []
    boundary_probe_results: dict[str, list[QwenPseudoProbeResult]] = {
        probe.variant.key: [] for probe in probes
    }
    boundaries: list[int] = []
    metrics: list[dict[str, object]] = []
    costs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    resident: dict[str, dict[int, tuple[int, ...]]] = {}
    previous_scores: dict[int, Tensor] | None = None
    expert_bytes = physical_expert_bytes(ops, 0)
    for boundary in range(0, route_tokens, suite.operating_point.horizon):
        with torch.inference_mode():
            end = min(boundary + suite.operating_point.horizon, route_tokens)
            boundaries.append(boundary)
            results = []
            for probe in probes:
                anchor_token_ids = (
                    anchor_token_provider(
                        probe.variant,
                        boundary,
                        prompt_token_ids,
                        tuple(source_tokens),
                    )
                    if anchor_token_provider is not None
                    else None
                )
                results.append(
                    probe.predict(
                        cache,
                        sampled_next_token_id=source_tokens[boundary],
                        current_token_id=current_token_id,
                        next_token_logits=(
                            planning_logits if probe.variant.content == "expected_top_m" else None
                        ),
                        expected_top_m=suite.optional_expected_embedding.top_m,
                        anchor_token_ids=anchor_token_ids,
                    )
                )
            for result in results:
                boundary_probe_results[result.variant].append(result)
                costs.append(_probe_cost(result, boundary))
                audits.append(_audit_summary(result, boundary))
            window_steps: list[tuple[SubsetRouteRecord, ...]] = []
            for position in range(boundary, end):
                token = torch.tensor(
                    [[source_tokens[position]]],
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                )
                output, records = _forward_capture(
                    model,
                    ops,
                    token,
                    cache,
                    policy="natural",
                )
                if tuple(record.layer for record in records) != tuple(range(ops.num_layers)):
                    raise RuntimeError("Qwen route replay missed a routed layer")
                window_steps.append(records)
                natural_steps.append(records)
                current_token_id = source_tokens[position]
                planning_logits = cast(Tensor, output.logits[:, -1]).detach()
                if position + 1 < len(source_tokens):
                    predicted = int(output.logits[:, -1].argmax())
                    argmax_compared += 1
                    if predicted == source_tokens[position + 1]:
                        argmax_matches += 1
                    elif first_argmax_divergence is None:
                        first_argmax_divergence = position + 1
            if authoritative is None:
                natural_scores = _scores_from_records(window_steps, ops.num_experts)
            else:
                natural_scores = {
                    layer: torch.zeros(ops.num_experts, dtype=torch.float64)
                    for layer in range(ops.num_layers)
                }
                for layer in range(ops.num_layers):
                    natural_scores[layer].scatter_add_(
                        0,
                        authoritative["router_topk_ids"][boundary:end, layer].reshape(-1).long(),
                        authoritative["router_topk_weights"][boundary:end, layer]
                        .reshape(-1)
                        .double(),
                    )
            oracle = {
                layer: _top_b(scores, suite.operating_point.budget_per_layer)
                for layer, scores in natural_scores.items()
            }
            previous = (
                {
                    layer: _top_b(
                        scores,
                        suite.operating_point.budget_per_layer,
                    )
                    for layer, scores in previous_scores.items()
                }
                if previous_scores is not None
                else static_subsets
            )
            subsets: dict[str, dict[int, tuple[int, ...]]] = {}
            if include_reference_methods:
                subsets.update(
                    {
                        "hard_oracle_commitment": oracle,
                        "previous_route_commitment": previous,
                        "static_frequency": static_subsets,
                    }
                )
            subsets.update({result.variant: result.subsets for result in results})
            for method, by_layer in subsets.items():
                resident.setdefault(
                    method,
                    {layer: () for layer in range(ops.num_layers)},
                )
                for layer in range(ops.num_layers):
                    if authoritative is None:
                        layer_ids = torch.stack(
                            [step[layer].natural.ids.reshape(-1) for step in window_steps]
                        )
                        layer_weights = torch.stack(
                            [step[layer].natural.weights.reshape(-1) for step in window_steps]
                        )
                        layer_logits = torch.stack(
                            [step[layer].natural.logits.reshape(-1) for step in window_steps]
                        )
                    else:
                        layer_ids = authoritative["router_topk_ids"][boundary:end, layer]
                        layer_weights = authoritative["router_topk_weights"][boundary:end, layer]
                        layer_logits = authoritative["router_logits"][boundary:end, layer]
                    metrics.append(
                        _route_metrics(
                            sample_id=str(source["sample_id"]),
                            boundary=boundary,
                            layer=layer,
                            method=method,
                            ids=layer_ids,
                            weights=layer_weights,
                            logits=layer_logits,
                            subset=by_layer[layer],
                            previous_subset=resident[method][layer],
                            expert_bytes=expert_bytes,
                        )
                    )
                    resident[method][layer] = by_layer[layer]
            previous_scores = natural_scores
    replay_natural = {
        "token_ids": torch.tensor(source_tokens[:route_tokens], dtype=torch.int64),
        "router_logits": torch.stack(
            [
                torch.stack([record.natural.logits[0].float() for record in step])
                for step in natural_steps
            ]
        ),
        "router_topk_ids": torch.stack(
            [torch.stack([record.natural.ids[0] for record in step]) for step in natural_steps]
        ),
        "router_topk_weights": torch.stack(
            [
                torch.stack([record.natural.weights[0].float() for record in step])
                for step in natural_steps
            ]
        ),
    }
    natural = authoritative if authoritative is not None else replay_natural
    parity: dict[str, object] = {
        "teacher_forced_saved_v17_trajectory": True,
        "teacher_forcing_requires_current_argmax_match": False,
        "v17_argmax_matches": argmax_matches,
        "v17_argmax_compared": argmax_compared,
        "v17_argmax_agreement": argmax_matches / argmax_compared,
        "first_v17_argmax_divergence": first_argmax_divergence,
    }
    if partition == "development":
        parity.update(
            _validate_development_parity(
                suite,
                int(source["row_index"]),
                replay_natural,
            )
        )
    tensors = dict(natural)
    if authoritative is not None:
        tensors.update(
            {
                f"current_replay__{key}": value
                for key, value in replay_natural.items()
                if key != "token_ids"
            }
        )
    tensors["boundaries"] = torch.tensor(boundaries, dtype=torch.int64)
    for variant, results in boundary_probe_results.items():
        tensors[f"{variant}__raw_router_logits"] = _stack_probe_tensor(results, "raw_router_logits")
        tensors[f"{variant}__pre_topk_probabilities"] = _stack_probe_tensor(
            results, "pre_topk_probabilities"
        )
        tensors[f"{variant}__pseudo_topk_ids"] = _stack_probe_tensor(results, "pseudo_topk_ids")
        tensors[f"{variant}__pseudo_topk_weights"] = _stack_probe_tensor(
            results, "pseudo_topk_weights"
        )
        tensors[f"{variant}__subsets"] = torch.tensor(
            [[result.subsets[layer] for layer in range(ops.num_layers)] for result in results],
            dtype=torch.int64,
        )
    row: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "partition": partition,
        "model": suite.model.key,
        "model_revision": suite.model.revision,
        "precision": suite.model.precision,
        "task": suite.dataset.key,
        "dataset_revision": suite.dataset.revision,
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "information_regime": "online_post_sample_probe_with_offline_teacher_forced_scoring",
        "evaluation_mode": "route_replay_open_loop",
        "ground_truth_route_source": (
            "checksum_verified_authoritative_frozen_trace"
            if authoritative is not None
            else "current_load_teacher_forced_saved_v17_trajectory"
        ),
        "horizon": suite.operating_point.horizon,
        "budget": suite.operating_point.budget_per_layer,
        "route_tokens": route_tokens,
        "boundaries": boundaries,
        "variants": list(boundary_probe_results),
        "metrics": metrics,
        "probe_costs": costs,
        "cache_rng_audits": audits,
        "parity": parity,
        "source_v17_row_sha256": sha256_json(source),
        "source_token_ids_sha256": sha256_json(source_tokens),
        "elapsed_seconds_measured": time.time() - started,
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }
    return row, tensors


def _sample_paths(
    output: Path,
    partition: str,
    row_index: int,
) -> tuple[Path, Path]:
    root = output / "route" / partition / "samples" / f"{row_index:05d}"
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _load_valid_sample(
    suite: PseudoEmbeddingSuiteConfig,
    json_path: Path,
    tensor_path: Path,
) -> dict[str, Any] | None:
    if not json_path.is_file() and not tensor_path.is_file():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial route sample artifact: {json_path}")
    row = cast(
        dict[str, Any],
        json.loads(json_path.read_text(encoding="utf-8")),
    )
    if (
        row.get("state") != "complete"
        or row.get("config_fingerprint") != suite.fingerprint()
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible or corrupt route sample artifact: {json_path}")
    return row


def run_route_partition(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
    partition: str,
    *,
    physical_gpu: int,
    variant_keys: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    if partition not in {"mechanism_smoke", "development", "held_out_route"}:
        raise ValueError(f"unsupported route partition: {partition}")
    references = manifest.partitions[partition].rows
    source_rows = _source_rows(suite)
    existing: list[dict[str, object]] = []
    missing = []
    for reference in references:
        json_path, tensor_path = _sample_paths(
            output,
            partition,
            reference.row_index,
        )
        row = _load_valid_sample(suite, json_path, tensor_path)
        if row is None:
            missing.append(reference)
        else:
            existing.append(cast(dict[str, object], row))
    if not missing:
        return existing
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    model_config = _source_model(accuracy, physical_gpu)
    seed_everything(suite.decode.seed)
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (
        ops.num_layers != suite.model.routed_layers
        or ops.num_experts != suite.model.routed_experts_per_layer
        or ops.top_k != suite.model.native_top_k
    ):
        raise ValueError("runtime Qwen model facts changed")
    defaults = load_default_vectors(Path(suite.default_vectors.root))
    if defaults.fingerprint != suite.default_vectors.artifact_fingerprint:
        raise ValueError("default-vector artifact fingerprint changed")
    selected_variants = tuple(
        variant
        for variant in _variants(suite)
        if variant_keys is None or variant.key in set(variant_keys)
    )
    if not selected_variants or (
        variant_keys is not None
        and {variant.key for variant in selected_variants} != set(variant_keys)
    ):
        raise ValueError(f"unknown or empty pseudo variant selection: {variant_keys}")
    probes = tuple(
        QwenPseudoEmbeddingProbe(
            model,
            ops,
            defaults,
            variant,
            anchors=suite.probe.anchors,
            budget=suite.operating_point.budget_per_layer,
        )
        for variant in selected_variants
    )
    static_subsets = {
        layer: _top_b(
            defaults.count[layer].double(),
            suite.operating_point.budget_per_layer,
        )
        for layer in range(ops.num_layers)
    }
    completed = list(existing)
    max_tokens = (
        suite.operating_point.horizon
        if partition == "mechanism_smoke"
        else suite.route_evaluation.max_decode_tokens_per_sample
    )
    for reference in missing:
        source = source_rows[reference.row_index]
        if source["sample_id"] != reference.sample_id:
            raise ValueError(f"sample manifest/source mismatch: {reference}")
        json_path, tensor_path = _sample_paths(
            output,
            partition,
            reference.row_index,
        )
        try:
            row, tensors = run_route_sample(
                suite,
                model,
                tokenizer,
                ops,
                source,
                probes,
                static_subsets,
                partition=partition,
                max_route_tokens=max_tokens,
                physical_gpu=physical_gpu,
            )
        except Exception as error:
            failure = json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")
            write_json_atomic(
                failure,
                {
                    "schema_version": 1,
                    "state": "failed",
                    "suite_id": suite.suite_id,
                    "config_fingerprint": suite.fingerprint(),
                    "partition": partition,
                    "row_index": reference.row_index,
                    "sample_id": reference.sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                },
            )
            raise
        _save_tensors_atomic(tensor_path, tensors)
        row.update(
            {
                "tensor_path": str(tensor_path.relative_to(output)),
                "tensor_bytes": tensor_path.stat().st_size,
                "tensor_sha256": sha256_file(tensor_path),
            }
        )
        write_json_atomic(json_path, row)
        completed.append(row)
        print(
            json.dumps(
                {
                    "stage": f"route_{partition}",
                    "sample_id": reference.sample_id,
                    "tokens": row["route_tokens"],
                    "variants": row["variants"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return completed


def load_focused_suite_and_manifest(
    config_path: Path,
) -> tuple[PseudoEmbeddingSuiteConfig, SampleManifest]:
    from pseudoroute.benchmark.pseudo_embedding_config import (
        load_pseudo_embedding_config,
    )

    suite = load_pseudo_embedding_config(config_path)
    manifest = load_sample_manifest(suite.sample_manifest.path)
    return suite, manifest
