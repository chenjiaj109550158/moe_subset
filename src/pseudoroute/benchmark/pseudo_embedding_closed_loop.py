"""Sample-atomic true hard closed-loop generation for the focused Qwen pilot."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Literal, cast

import torch
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
)
from pseudoroute.benchmark.pseudo_embedding_route import (
    _source_model,
    _source_rows,
    _subsets_from_records,
    _top_b,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
    _model_example,
)
from pseudoroute.benchmark.scoring import score_response
from pseudoroute.benchmark.subset_closed_loop import (
    RouteAccounting,
    _cache_has_sliding_layers,
    _finished,
    _forward_capture,
    _perplexity,
    _sample_token,
    _token_agreement,
    natural_lookahead_and_rewind,
    subsets_from_route_steps,
)
from pseudoroute.benchmark.subset_trace import sha256_json, write_json_atomic
from pseudoroute.benchmark.tasks import BenchmarkExample, load_examples
from pseudoroute.utils.determinism import seed_everything

FocusedPolicy = Literal[
    "hard_oracle_commitment",
    "previous_route_commitment",
    "pseudo_embedding_commitment",
]


def _digest_route(digest: Any, record: SubsetRouteRecord, *, executed: bool) -> None:
    route = record.executed if executed else record.natural
    digest.update(record.layer.to_bytes(4, "little"))
    digest.update(str(tuple(route.ids.shape)).encode())
    digest.update(route.ids.contiguous().numpy().tobytes())


def _account_hard_records(
    accounting: RouteAccounting,
    records: tuple[SubsetRouteRecord, ...],
    allowed: dict[int, tuple[int, ...]],
    expert_bytes: dict[int, int],
    *,
    decode_step: int,
    natural_digest: Any,
    executed_digest: Any,
) -> None:
    for record in records:
        _digest_route(natural_digest, record, executed=False)
        _digest_route(executed_digest, record, executed=True)
        allowed_tensor = torch.tensor(allowed[record.layer], dtype=torch.long)
        natural_ids = record.natural.ids.reshape(-1)
        inside = torch.isin(natural_ids, allowed_tensor)
        hits = int(inside.sum())
        slots = int(natural_ids.numel())
        accounting.route_hits += hits
        accounting.route_slots += slots
        accounting.selected_mass_hit += float(
            record.natural.weights.reshape(-1).masked_select(inside).double().sum()
        )
        accounting.selected_mass_total += float(record.natural.weights.double().sum())
        accounting.route_calls += 1
        if not torch.equal(record.natural.ids, record.executed.ids):
            accounting.route_divergent_calls += 1
            if accounting.first_route_divergence is None:
                accounting.first_route_divergence = decode_step
        accounting.natural_reference_bytes += slots * expert_bytes[record.layer]


def _selected_variant(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
) -> QwenPseudoVariant:
    selection_path = output / "development_selection.json"
    gate_path = output / "held_out_route_gate.json"
    selection = cast(
        dict[str, Any],
        json.loads(selection_path.read_text(encoding="utf-8")),
    )
    gate = cast(dict[str, Any], json.loads(gate_path.read_text(encoding="utf-8")))
    if (
        selection.get("config_fingerprint") != suite.fingerprint()
        or not selection.get("progress_gate_pass")
        or gate.get("config_fingerprint") != suite.fingerprint()
        or not gate.get("held_out_route_gate_pass")
        or not gate.get("actual_closed_loop_authorized")
    ):
        raise RuntimeError("actual closed loop is forbidden until both frozen route gates pass")
    key = selection.get("selected_variant")
    matches = [variant for variant in suite.variants if variant.key == key]
    if len(matches) != 1 or gate.get("selected_variant") != key:
        raise RuntimeError("development selection and held-out gate disagree")
    value = matches[0]
    return QwenPseudoVariant(
        value.key,
        value.content,
        value.attention,
        value.expert_contribution,
    )


def _static_subsets(
    suite: PseudoEmbeddingSuiteConfig,
    ops: Qwen3MoePrefetchOps,
) -> dict[int, tuple[int, ...]]:
    defaults = load_default_vectors(Path(suite.default_vectors.root))
    if defaults.fingerprint != suite.default_vectors.artifact_fingerprint:
        raise ValueError("default-vector artifact fingerprint changed")
    return {
        layer: _top_b(defaults.count[layer].double(), suite.operating_point.budget_per_layer)
        for layer in range(ops.num_layers)
    }


def run_focused_policy_sample(
    suite: PseudoEmbeddingSuiteConfig,
    accuracy: AccuracySuiteConfig,
    model_config: AccuracyModelConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    example: BenchmarkExample,
    source: dict[str, Any],
    static: dict[int, tuple[int, ...]],
    *,
    policy: FocusedPolicy,
    max_new_tokens: int,
    probe: QwenPseudoEmbeddingProbe | None,
    physical_gpu: int,
) -> dict[str, object]:
    if (policy == "pseudo_embedding_commitment") != (probe is not None):
        raise ValueError("only pseudo_embedding_commitment may receive a pseudo probe")
    rendered = str(source["rendered_prompt"])
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, rendered)
    source_tokens = [int(value) for value in source["generated_token_ids"]][:max_new_tokens]
    do_sample = accuracy.do_sample_for(model_config, example.task)
    if do_sample:
        raise ValueError("the frozen Qwen/GSM8K focused pilot must be greedy")
    seed_everything(suite.decode.seed)
    device = torch.device(model_config.device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    with torch.inference_mode():
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("focused closed loop did not return a KV cache")
    prefill_scores = cast(Tensor, prefill.logits[:, -1])
    first = _sample_token(prefill_scores, model, do_sample=False)
    generated = [int(first.item())]
    nlls: list[float] = []
    if source_tokens:
        nlls.append(float(-prefill_scores.float().log_softmax(dim=-1)[0, source_tokens[0]]))
    prompt_ids = inputs["input_ids"]
    current = first[:, None].to(prompt_ids.device)
    last_cached_token_id = int(prompt_ids[0, -1])
    accounting = RouteAccounting()
    expert_bytes = {layer: physical_expert_bytes(ops, layer) for layer in range(ops.num_layers)}
    natural_digest = hashlib.sha256()
    executed_digest = hashlib.sha256()
    subset_digest = hashlib.sha256()
    active = static
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(ops.num_layers)}
    previous_window: list[tuple[SubsetRouteRecord, ...]] | None = None
    current_window: list[tuple[SubsetRouteRecord, ...]] = []
    boundary_count = 0
    probe_latency = 0.0
    probe_attention_queries = 0
    probe_attention_calls = 0
    probe_router_calls = 0
    probe_syncs = 0
    probe_peak_temporary = 0
    probe_audits: list[dict[str, object]] = []
    decode_step = 0
    finished = _finished(
        model,
        tokenizer,
        model_config,
        example,
        prompt_ids,
        generated,
        prefill_scores,
    )
    while len(generated) < max_new_tokens and not finished:
        if decode_step % suite.operating_point.horizon == 0:
            boundary_count += 1
            current_window = []
            remaining = min(
                suite.operating_point.horizon,
                max_new_tokens - len(generated),
            )
            if policy == "hard_oracle_commitment":
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
                    horizon=max(1, remaining),
                    do_sample=False,
                )
                active = subsets_from_route_steps(
                    future,
                    ops,
                    suite.operating_point.budget_per_layer,
                )
            elif policy == "previous_route_commitment":
                active = (
                    _subsets_from_records(
                        previous_window,
                        ops.num_experts,
                        suite.operating_point.budget_per_layer,
                    )
                    if previous_window is not None
                    else static
                )
            else:
                if probe is None:
                    raise AssertionError("pseudo probe disappeared")
                predicted = probe.predict(
                    cache,
                    sampled_next_token_id=int(current.item()),
                    current_token_id=last_cached_token_id,
                )
                active = predicted.subsets
                probe_latency += predicted.cost.latency_seconds_measured
                probe_attention_queries += predicted.cost.attention_queries
                probe_attention_calls += predicted.cost.attention_calls
                probe_router_calls += predicted.cost.router_calls
                probe_syncs += predicted.cost.cpu_gpu_synchronizations
                probe_peak_temporary = max(
                    probe_peak_temporary,
                    predicted.cost.temporary_cuda_bytes_measured,
                )
                probe_audits.append(predicted.audit)
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
        processed_token_id = int(current.item())
        output, records = _forward_capture(
            model,
            ops,
            current,
            cache,
            policy="hard",
            allowed=active,
        )
        current_window.append(records)
        _account_hard_records(
            accounting,
            records,
            active,
            expert_bytes,
            decode_step=decode_step,
            natural_digest=natural_digest,
            executed_digest=executed_digest,
        )
        scores = cast(Tensor, output.logits[:, -1])
        source_index = len(generated)
        if source_index < len(source_tokens):
            nlls.append(float(-scores.float().log_softmax(dim=-1)[0, source_tokens[source_index]]))
        token = _sample_token(scores, model, do_sample=False)
        generated.append(int(token.item()))
        current = token[:, None].to(prompt_ids.device)
        last_cached_token_id = processed_token_id
        decode_step += 1
        if decode_step % suite.operating_point.horizon == 0:
            previous_window = current_window
        finished = _finished(
            model,
            tokenizer,
            model_config,
            example,
            prompt_ids,
            generated,
            scores,
        )
    elapsed = time.time() - started
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    score = score_response(example, generated_text)
    agreement, first_token_divergence = _token_agreement(generated, source_tokens)
    nll = sum(nlls) / len(nlls) if nlls else float("nan")
    audit_pass = all(
        bool(row["production_cache_signature_unchanged"])
        and bool(row["production_rng_unchanged"])
        and bool(row["shadow_cache_discarded"])
        and not bool(row["forbidden_inputs_present"])
        for row in probe_audits
    )
    return {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "model": suite.model.key,
        "model_id": suite.model.model_id,
        "model_revision": suite.model.revision,
        "precision": suite.model.precision,
        "task": suite.dataset.key,
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "policy": policy,
        "information_regime": (
            "routing_information_oracle_from_current_policy_context"
            if policy == "hard_oracle_commitment"
            else "online_previous_pre_mask_route_from_current_policy_context"
            if policy == "previous_route_commitment"
            else "online_post_sample_current_policy_cache_only"
        ),
        "evaluation_mode": "actual_hard_closed_loop_generation",
        "identity_materialized": False,
        "outside_subset_router_logits_masked": True,
        "native_topk_and_normalization_after_mask": True,
        "cache_replay_semantics": (
            "copy_on_write_layer_fork_discard_original_boundary_verified"
            if _cache_has_sliding_layers(cache)
            else "oracle_in_place_exact_crop_rewind_and_pseudo_copy_on_write"
        ),
        "horizon": suite.operating_point.horizon,
        "budget": suite.operating_point.budget_per_layer,
        "resident_fraction": suite.operating_point.resident_fraction,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "generated_tokens": len(generated),
        "v17_natural_token_ids": source_tokens,
        "exact_token_agreement": agreement,
        "first_token_divergence": first_token_divergence,
        "first_route_divergence": accounting.first_route_divergence,
        "route_divergent_calls": accounting.route_divergent_calls,
        "route_calls": accounting.route_calls,
        "executed_route_changed": accounting.route_divergent_calls > 0,
        "natural_route_ids_sha256": natural_digest.hexdigest(),
        "executed_route_ids_sha256": executed_digest.hexdigest(),
        "subset_trajectory_sha256": subset_digest.hexdigest(),
        "route_hit_rate": (
            accounting.route_hits / accounting.route_slots if accounting.route_slots else 1.0
        ),
        "selected_routing_mass_coverage": (
            accounting.selected_mass_hit / accounting.selected_mass_total
            if accounting.selected_mass_total
            else 1.0
        ),
        "natural_outside_subset_frequency": (
            1 - accounting.route_hits / accounting.route_slots if accounting.route_slots else 0.0
        ),
        "runtime_fallback_loads": 0,
        "prefetch_loads": accounting.prefetch_loads,
        "subset_churn": accounting.subset_churn,
        "simulated_planned_transfer_bytes": accounting.planned_transfer_bytes,
        "simulated_natural_reference_bytes": accounting.natural_reference_bytes,
        "simulated_estimated_transfer_reduction": (
            1 - accounting.planned_transfer_bytes / accounting.natural_reference_bytes
            if accounting.natural_reference_bytes
            else 0.0
        ),
        "simulated_stall_ms": None,
        "simulated_stall_omission": "no_stall_latency_model_frozen_in_focused_v1_config",
        "v17_token_nll_on_policy_context": nll,
        "perplexity": _perplexity(nll),
        "nll_tokens": len(nlls),
        "correct": score.correct,
        "parsed_answer": score.parsed_answer,
        "score_detail": score.detail,
        "boundary_count": boundary_count,
        "probe_variant": probe.variant.key if probe is not None else None,
        "probe_latency_seconds_measured": probe_latency,
        "probe_attention_queries": probe_attention_queries,
        "probe_attention_calls": probe_attention_calls,
        "probe_router_calls": probe_router_calls,
        "probe_cpu_gpu_synchronizations": probe_syncs,
        "probe_peak_temporary_cuda_bytes_measured": probe_peak_temporary,
        "probe_cache_rng_information_audit_pass": audit_pass,
        "elapsed_seconds_measured": elapsed,
        "tokens_per_second_measured": len(generated) / elapsed if elapsed else None,
        "runtime_kind": "measured_actual_closed_loop_including_policy_planning",
        "peak_cuda_allocated_bytes_measured": int(torch.cuda.max_memory_allocated(device)),
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "source_v17_row_sha256": sha256_json(source),
        "source_v17_correct": bool(source["correct"]),
        "source_v17_generated_tokens": len(source_tokens),
        "label_or_correctness_used_by_policy": False,
        "future_v17_tokens_used_by_policy": False,
    }


def _sample_path(output: Path, wave: int, row_index: int, policy: str) -> Path:
    return output / "closed_loop" / f"wave_{wave}" / f"{row_index:05d}" / f"{policy}.json"


def _failed_path(output: Path, wave: int, row_index: int, policy: str) -> Path:
    path = _sample_path(output, wave, row_index, policy)
    return path.with_name(f"{policy}.FAILED.json")


def _write_checksummed_row(path: Path, row: dict[str, object]) -> None:
    payload = dict(row)
    payload["row_payload_sha256"] = sha256_json(row)
    write_json_atomic(path, payload)


def _load_checksummed_row(
    path: Path,
    suite: PseudoEmbeddingSuiteConfig,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    expected = row.pop("row_payload_sha256", None)
    if (
        row.get("state") != "complete"
        or row.get("config_fingerprint") != suite.fingerprint()
        or expected != sha256_json(row)
    ):
        raise ValueError(f"incompatible or corrupt closed-loop sample artifact: {path}")
    row["row_payload_sha256"] = expected
    return row


def _wave_partition(wave: int) -> str:
    if wave not in (1, 2):
        raise ValueError("closed-loop wave must be one or two")
    return f"closed_loop_wave_{wave}"


def run_closed_loop_wave(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
    *,
    wave: int,
    physical_gpu: int,
) -> list[dict[str, object]]:
    selected_variant = _selected_variant(suite, output)
    partition = _wave_partition(wave)
    references = manifest.partitions[partition].rows
    policies = suite.policies
    complete: list[dict[str, object]] = []
    missing = []
    for reference in references:
        for policy in policies:
            path = _sample_path(output, wave, reference.row_index, policy)
            existing = _load_checksummed_row(path, suite)
            if existing is None:
                missing.append((reference, policy))
            else:
                complete.append(cast(dict[str, object], existing))
    if not missing:
        return complete
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    model_config = _source_model(accuracy, physical_gpu)
    dataset_matches = [item for item in accuracy.datasets if item.key == suite.dataset.key]
    if len(dataset_matches) != 1:
        raise ValueError("the v17 source is missing frozen GSM8K")
    examples = load_examples(dataset_matches[0], cache_dir=accuracy.dataset_cache_dir)
    sources = _source_rows(suite)
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
    static = _static_subsets(suite, ops)
    pseudo_probe = QwenPseudoEmbeddingProbe(
        model,
        ops,
        defaults,
        selected_variant,
        anchors=suite.probe.anchors,
        budget=suite.operating_point.budget_per_layer,
    )
    for reference, policy in missing:
        source = sources[reference.row_index]
        if source["sample_id"] != reference.sample_id:
            raise ValueError(f"sample manifest/source mismatch: {reference}")
        example = _model_example(examples[reference.row_index], model_config)
        if example.sample_id != reference.sample_id:
            raise ValueError(f"sample manifest/dataset mismatch: {reference}")
        max_tokens = min(example.max_new_tokens, suite.dataset.original_v17_max_new_tokens)
        path = _sample_path(output, wave, reference.row_index, policy)
        try:
            row = run_focused_policy_sample(
                suite,
                accuracy,
                model_config,
                model,
                tokenizer,
                ops,
                example,
                source,
                static,
                policy=policy,
                max_new_tokens=max_tokens,
                probe=pseudo_probe if policy == "pseudo_embedding_commitment" else None,
                physical_gpu=physical_gpu,
            )
            row["wave"] = wave
            _write_checksummed_row(path, row)
            complete.append(row)
            print(
                json.dumps(
                    {
                        "stage": f"closed_loop_wave_{wave}",
                        "sample_id": reference.sample_id,
                        "policy": policy,
                        "tokens": row["generated_tokens"],
                        "correct": row["correct"],
                        "elapsed_seconds": row["elapsed_seconds_measured"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:
            write_json_atomic(
                _failed_path(output, wave, reference.row_index, policy),
                {
                    "schema_version": 1,
                    "state": "failed",
                    "suite_id": suite.suite_id,
                    "config_fingerprint": suite.fingerprint(),
                    "wave": wave,
                    "row_index": reference.row_index,
                    "sample_id": reference.sample_id,
                    "policy": policy,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                },
            )
            raise
    return complete


def wave_eta_checkpoint(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
) -> dict[str, object]:
    rows = []
    for reference in manifest.partitions["closed_loop_wave_1"].rows:
        for policy in suite.policies:
            row = _load_checksummed_row(
                _sample_path(output, 1, reference.row_index, policy),
                suite,
            )
            if row is None:
                raise ValueError("wave-one ETA requires every frozen sample/policy row")
            rows.append(row)
    wave_seconds = sum(float(row["elapsed_seconds_measured"]) for row in rows)
    projected_total_hours = 2 * wave_seconds / 3600
    requires_question = (
        projected_total_hours
        > suite.closed_loop.ask_before_second_wave_if_projected_total_hours_exceeds
    )
    result = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "wave_one_sample_policy_rows": len(rows),
        "wave_one_elapsed_seconds_measured_sum": wave_seconds,
        "projected_16_row_three_policy_total_hours": projected_total_hours,
        "threshold_hours": (
            suite.closed_loop.ask_before_second_wave_if_projected_total_hours_exceeds
        ),
        "second_wave_requires_explicit_user_confirmation": requires_question,
        "second_wave_may_proceed_without_new_confirmation": not requires_question,
        "projection_method": "two_times_sum_of_wave_one_sample_policy_measured_runtimes",
    }
    write_json_atomic(output / "closed_loop" / "wave_1_eta_checkpoint.json", result)
    return result


def paired_accuracy_bootstrap(
    correctness: dict[str, dict[str, bool]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    import numpy as np

    ids = sorted(next(iter(correctness.values())))
    if any(sorted(values) != ids for values in correctness.values()):
        raise ValueError("paired accuracy bootstrap sample mismatch")
    baseline = correctness["vanilla_v17"]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(samples, len(ids)))
    result = []
    for policy, values in sorted(correctness.items()):
        if policy == "vanilla_v17":
            continue
        difference = torch.tensor(
            [int(values[sample_id]) - int(baseline[sample_id]) for sample_id in ids],
            dtype=torch.float64,
        ).numpy()
        draws = difference[indices].mean(axis=1)
        result.append(
            {
                "policy": policy,
                "baseline": "vanilla_v17",
                "samples": len(ids),
                "mean_accuracy_difference": float(difference.mean()),
                "paired_bootstrap_ci95": [
                    float(np.quantile(draws, 0.025)),
                    float(np.quantile(draws, 0.975)),
                ],
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            }
        )
    return result


def aggregate_closed_loop(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
) -> dict[str, object]:
    source = _source_rows(suite)
    rows: list[dict[str, Any]] = []
    references = [
        reference
        for wave in (1, 2)
        for reference in manifest.partitions[_wave_partition(wave)].rows
    ]
    for wave in (1, 2):
        for reference in manifest.partitions[_wave_partition(wave)].rows:
            for policy in suite.policies:
                row = _load_checksummed_row(
                    _sample_path(output, wave, reference.row_index, policy),
                    suite,
                )
                if row is None:
                    raise ValueError("closed-loop aggregate is incomplete")
                if row["sample_id"] != reference.sample_id or int(row["wave"]) != wave:
                    raise ValueError("closed-loop sample provenance mismatch")
                rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = {
        policy: [row for row in rows if row["policy"] == policy] for policy in suite.policies
    }
    summary = {}
    for aggregate_policy, policy_rows in grouped.items():
        successes = sum(bool(row["correct"]) for row in policy_rows)
        total_tokens = sum(int(row["generated_tokens"]) for row in policy_rows)
        total_seconds = sum(float(row["elapsed_seconds_measured"]) for row in policy_rows)
        summary[aggregate_policy] = {
            "samples": len(policy_rows),
            "successes": successes,
            "accuracy_measured": successes / len(policy_rows),
            "accuracy_gate_pass": successes >= suite.accuracy_gate.minimum_policy_successes,
            "mean_exact_token_agreement": sum(
                float(row["exact_token_agreement"]) for row in policy_rows
            )
            / len(policy_rows),
            "mean_route_hit_rate": sum(float(row["route_hit_rate"]) for row in policy_rows)
            / len(policy_rows),
            "mean_selected_routing_mass": sum(
                float(row["selected_routing_mass_coverage"]) for row in policy_rows
            )
            / len(policy_rows),
            "mean_nll": sum(float(row["v17_token_nll_on_policy_context"]) for row in policy_rows)
            / len(policy_rows),
            "total_generated_tokens": total_tokens,
            "total_runtime_seconds_measured": total_seconds,
            "aggregate_tokens_per_second_measured": total_tokens / total_seconds,
            "total_probe_latency_seconds_measured": sum(
                float(row["probe_latency_seconds_measured"]) for row in policy_rows
            ),
            "simulated_total_planned_transfer_bytes": sum(
                int(row["simulated_planned_transfer_bytes"]) for row in policy_rows
            ),
            "simulated_total_natural_reference_bytes": sum(
                int(row["simulated_natural_reference_bytes"]) for row in policy_rows
            ),
        }
    correctness: dict[str, dict[str, bool]] = {
        "vanilla_v17": {
            reference.sample_id: bool(source[reference.row_index]["correct"])
            for reference in references
        }
    }
    for aggregate_policy, policy_rows in grouped.items():
        correctness[aggregate_policy] = {
            str(row["sample_id"]): bool(row["correct"]) for row in policy_rows
        }
    paired = paired_accuracy_bootstrap(
        correctness,
        samples=suite.accuracy_gate.bootstrap_samples,
        seed=suite.accuracy_gate.bootstrap_seed,
    )
    result = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "samples": len(references),
        "sample_policy_rows": len(rows),
        "vanilla_v17": {
            "samples": len(references),
            "successes": sum(correctness["vanilla_v17"].values()),
            "accuracy_measured_preexisting": (
                sum(correctness["vanilla_v17"].values()) / len(references)
            ),
        },
        "policies": summary,
        "paired_accuracy_bootstrap": paired,
        "allowed_drop_questions_frozen": suite.accuracy_gate.allowed_drop_questions,
        "minimum_policy_successes_frozen": suite.accuracy_gate.minimum_policy_successes,
        "measured_vs_simulated": {
            "measured": [
                "task_accuracy",
                "tokens",
                "token_agreement",
                "nll_perplexity",
                "route_divergence",
                "probe_latency_memory_calls",
                "closed_loop_runtime",
            ],
            "simulated": ["expert_transfer_bytes"],
            "not_estimated": ["transfer_stall_due_to_no_frozen_focused_v1_latency_model"],
            "identity_materialized_rows": 0,
            "actual_hard_closed_loop_rows": len(rows),
        },
    }
    write_json_atomic(output / "closed_loop" / "summary.json", result)
    return result
