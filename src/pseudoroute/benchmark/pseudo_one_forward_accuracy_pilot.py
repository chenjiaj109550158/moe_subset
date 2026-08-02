"""Run the frozen one-forward Qwen/GSM8K eight-row accuracy pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from torch import Tensor, nn

from pseudoroute.benchmark.config import (
    AccuracyModelConfig,
    AccuracySuiteConfig,
    load_accuracy_suite_config,
)
from pseudoroute.benchmark.context_continuation import build_context_continuation
from pseudoroute.benchmark.prefetch import (
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
    SubsetRouteRecord,
    physical_expert_bytes,
)
from pseudoroute.benchmark.pseudo_embedding_closed_loop import (
    _account_hard_records,
    paired_accuracy_bootstrap,
)
from pseudoroute.benchmark.pseudo_embedding_content_smoke import recent_anchor_token_ids
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    candidate_subsets,
    route_scores,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoProbeResult,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
    _model_example,
    _read_jsonl,
    _software_hardware,
)
from pseudoroute.benchmark.scoring import score_response
from pseudoroute.benchmark.subset_closed_loop import (
    RouteAccounting,
    _finished,
    _forward_capture,
    _perplexity,
    _sample_token,
    _token_agreement,
    natural_token_lookahead_copy_on_write,
)
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.benchmark.tasks import BenchmarkExample, load_examples
from pseudoroute.utils.determinism import seed_everything

PilotPolicy = Literal[
    "recent_sequence_causal",
    "sampled_unigram_full_continuation",
    "future_exact_content_oracle",
]
Stage = Literal["smoke", "actual"]

PILOT_ID = "pseudo_one_forward_accuracy_pilot_v1"
CONFIG = Path("configs/benchmark/pseudo_one_forward_accuracy_pilot_v1.yaml")
SAMPLES = Path("configs/benchmark/pseudo_one_forward_accuracy_pilot_v1_samples.json")
CONFIG_SHA256 = "d9515855b897189fde9f36fba151af5467ebc93e46bbf09bb79ad5b39e5f10af"
SAMPLES_SHA256 = "fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046"
OUTPUT = Path("artifacts/pseudo_one_forward_accuracy_pilot_v1")
HORIZON = 8
BUDGET = 32
LAYERS = 48
EXPERTS = 128
TOP_K = 8
POLICIES: tuple[PilotPolicy, ...] = (
    "recent_sequence_causal",
    "sampled_unigram_full_continuation",
    "future_exact_content_oracle",
)
SELECTOR = "first_four_anchor_core_plus_history_fill"


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("one-forward accuracy config fingerprint changed")
    if sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("one-forward accuracy sample fingerprint changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config.get("pilot_id") != PILOT_ID or samples.get("pilot_id") != PILOT_ID:
        raise ValueError("one-forward accuracy pilot identity changed")
    if tuple(config["policies"]["order"]) != POLICIES:
        raise ValueError("one-forward accuracy policy order changed")
    if tuple(samples["work_assignment"]["policy_order"]) != POLICIES:
        raise ValueError("one-forward accuracy work policy order changed")
    expected_ids = (
        "test-44",
        "test-632",
        "test-444",
        "test-519",
        "test-1311",
        "test-1264",
        "test-825",
        "test-252",
    )
    if tuple(row["sample_id"] for row in samples["accuracy"]) != expected_ids:
        raise ValueError("one-forward accuracy sample order changed")
    if (
        config["operating_point"]["horizon"],
        config["operating_point"]["budget_per_layer"],
        config["model"]["routed_layers"],
        config["model"]["routed_experts_per_layer"],
        config["model"]["native_top_k"],
    ) != (HORIZON, BUDGET, LAYERS, EXPERTS, TOP_K):
        raise ValueError("one-forward accuracy model operating point changed")
    for key in (
        "accuracy_config",
        "vanilla_rows",
        "original_pseudo_sample_manifest",
    ):
        path = Path(config["source"][key])
        expected = config["source"][f"{key}_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"one-forward accuracy source changed: {path}")
    route_manifest = Path("artifacts/pseudo_one_forward_context_continuation_v1") / (
        "artifact_manifest.json"
    )
    if sha256_file(route_manifest) != config["source"]["route_analysis_artifact_manifest_sha256"]:
        raise ValueError("one-forward accuracy route-analysis source changed")
    original = _json(Path(config["source"]["original_pseudo_sample_manifest"]))
    original_rows = original["partitions"]["closed_loop_wave_1"]["rows"]
    if [(row["row_index"], row["sample_id"], row["sha256_rank"]) for row in original_rows] != [
        (row["row_index"], row["sample_id"], row["sha256_rank"]) for row in samples["accuracy"]
    ]:
        raise ValueError("one-forward accuracy rows differ from original frozen wave one")
    return config, samples


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_rows(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["row_index"]): row
        for row in _read_jsonl(Path(config["source"]["vanilla_rows"]))
        if row.get("state") == "complete"
    }
    if set(rows) != set(range(1319)):
        raise ValueError("frozen Qwen/GSM8K source row set changed")
    return rows


def _check_reference(reference: dict[str, Any], source: dict[str, Any]) -> None:
    if (
        int(source["row_index"]) != int(reference["row_index"])
        or source["sample_id"] != reference["sample_id"]
        or source["rendered_prompt_sha256"] != reference["rendered_prompt_sha256"]
        or source["target_sha256"] != reference["target_sha256"]
        or int(source["generated_tokens"]) != int(reference["source_v17_generated_tokens"])
        or bool(source["correct"]) is not bool(reference["source_v17_correct"])
    ):
        raise ValueError(f"frozen source provenance changed: {reference['sample_id']}")


def _sample_path(stage: Stage, row_index: int, policy: PilotPolicy) -> Path:
    return OUTPUT / stage / f"{row_index:05d}" / f"{policy}.json"


def _failure_path(stage: Stage, row_index: int, policy: PilotPolicy) -> Path:
    path = _sample_path(stage, row_index, policy)
    return path.with_name(f"{policy}.{os.getpid()}.FAILED.json")


def _write_checksummed(path: Path, row: dict[str, object]) -> None:
    payload = dict(row)
    payload["row_payload_sha256"] = sha256_json(row)
    write_json_atomic(path, payload)


def _load_checksummed(
    path: Path,
    *,
    stage: Stage,
    row_index: int,
    sample_id: str,
    policy: PilotPolicy,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _json(path)
    expected = payload.pop("row_payload_sha256", None)
    if (
        payload.get("state") != "complete"
        or payload.get("pilot_id") != PILOT_ID
        or payload.get("config_sha256") != CONFIG_SHA256
        or payload.get("sample_manifest_sha256") != SAMPLES_SHA256
        or payload.get("stage") != stage
        or int(payload.get("row_index", -1)) != row_index
        or payload.get("sample_id") != sample_id
        or payload.get("policy") != policy
        or int(payload.get("max_new_tokens", -1)) != max_new_tokens
        or expected != sha256_json(payload)
    ):
        raise ValueError(f"incompatible or corrupt accuracy artifact: {path}")
    payload["row_payload_sha256"] = expected
    return payload


def _probe(model: nn.Module, ops: Qwen3MoePrefetchOps, policy: PilotPolicy) -> Any:
    return QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            policy,
            "provided_sequence",
            "causal",
            "native_expert_execution",
        ),
        anchors=tuple(range(1, HORIZON + 1)),
        budget=BUDGET,
    )


def _anchor_plan(
    policy: PilotPolicy,
    *,
    model: nn.Module,
    prompt_token_ids: tuple[int, ...],
    generated_token_ids: tuple[int, ...],
    current: Tensor,
    cache: object,
) -> tuple[tuple[int, ...], dict[str, object]]:
    boundary = len(generated_token_ids) - 1
    if boundary < 0:
        raise ValueError("accuracy planning requires a sampled token")
    started = time.perf_counter()
    if policy == "recent_sequence_causal":
        anchors = recent_anchor_token_ids(
            boundary,
            prompt_token_ids,
            generated_token_ids,
            horizon=HORIZON,
        )
        return anchors, {
            "content_source": "current_policy_recent_known_tokens",
            "future_v17_tokens_used": False,
            "full_expert_natural_lookahead_used": False,
            "content_planning_latency_seconds_measured": time.perf_counter() - started,
            "continuation_match": False,
            "continuation_fallback": False,
            "oracle_natural_forward_calls": 0,
            "oracle_terminal_padding_tokens": 0,
            "oracle_lookahead_latency_seconds_measured": 0.0,
            "oracle_cache_unchanged": None,
            "oracle_rng_unchanged": None,
            "oracle_shadow_cache_discarded": None,
        }
    if policy == "sampled_unigram_full_continuation":
        plan = build_context_continuation(
            boundary,
            prompt_token_ids,
            generated_token_ids,
            "sampled_unigram_full_continuation",
            horizon=HORIZON,
        )
        audit = asdict(plan)
        audit.update(
            {
                "content_source": "current_policy_known_context_unigram_continuation",
                "future_v17_tokens_used": False,
                "full_expert_natural_lookahead_used": False,
                "content_planning_latency_seconds_measured": (
                    plan.planning_latency_seconds_measured
                ),
                "continuation_match": not plan.fallback_used,
                "continuation_fallback": plan.fallback_used,
                "oracle_natural_forward_calls": 0,
                "oracle_terminal_padding_tokens": 0,
                "oracle_lookahead_latency_seconds_measured": 0.0,
                "oracle_cache_unchanged": None,
                "oracle_rng_unchanged": None,
                "oracle_shadow_cache_discarded": None,
            }
        )
        return plan.anchor_token_ids, audit
    lookahead = natural_token_lookahead_copy_on_write(
        model,
        current,
        cache,
        horizon=HORIZON,
    )
    return lookahead.anchor_token_ids, {
        "content_source": "current_policy_full_expert_greedy_natural_successors",
        "future_v17_tokens_used": False,
        "full_expert_natural_lookahead_used": True,
        "content_planning_latency_seconds_measured": (lookahead.latency_seconds_measured),
        "continuation_match": False,
        "continuation_fallback": False,
        "oracle_natural_forward_calls": lookahead.natural_forward_calls,
        "oracle_successor_tokens_generated": lookahead.successor_tokens_generated,
        "oracle_terminal_padding_tokens": lookahead.terminal_padding_tokens,
        "oracle_lookahead_latency_seconds_measured": (lookahead.latency_seconds_measured),
        "oracle_cache_unchanged": lookahead.production_cache_signature_unchanged,
        "oracle_rng_unchanged": lookahead.production_rng_unchanged,
        "oracle_shadow_cache_discarded": lookahead.shadow_cache_discarded,
    }


def _planning_audit_pass(
    policy: PilotPolicy,
    result: QwenPseudoProbeResult,
    content: dict[str, object],
) -> bool:
    audit = result.audit
    common = (
        bool(audit["production_cache_signature_unchanged"])
        and bool(audit["production_rng_unchanged"])
        and bool(audit["shadow_cache_discarded"])
        and bool(audit["one_causal_forward_per_boundary"])
        and bool(audit["shadow_expert_execution"])
        and not bool(audit["forbidden_inputs_present"])
        and result.cost.attention_calls == LAYERS
        and result.cost.attention_queries == LAYERS * HORIZON
        and result.cost.router_calls == LAYERS
        and result.cost.expert_calls == LAYERS
        and result.cost.lm_head_calls == 0
        and content["future_v17_tokens_used"] is False
    )
    if policy != "future_exact_content_oracle":
        return common and content["full_expert_natural_lookahead_used"] is False
    return (
        common
        and content["full_expert_natural_lookahead_used"] is True
        and content["oracle_cache_unchanged"] is True
        and content["oracle_rng_unchanged"] is True
        and content["oracle_shadow_cache_discarded"] is True
        and 0 <= cast(int, content["oracle_natural_forward_calls"]) <= HORIZON - 1
    )


def _exact_matches(generated: list[int], reference: list[int]) -> tuple[int, int]:
    total = max(len(generated), len(reference))
    return (
        sum(
            index < len(generated)
            and index < len(reference)
            and generated[index] == reference[index]
            for index in range(total)
        ),
        total,
    )


def run_policy_sample(
    config: dict[str, Any],
    accuracy: AccuracySuiteConfig,
    model_config: AccuracyModelConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    example: BenchmarkExample,
    source: dict[str, Any],
    *,
    policy: PilotPolicy,
    stage: Stage,
    max_new_tokens: int,
    physical_gpu: int,
    execution_git_head: str,
) -> dict[str, object]:
    if accuracy.do_sample_for(model_config, example.task):
        raise ValueError("frozen Qwen/GSM8K accuracy pilot must be greedy")
    rendered = str(source["rendered_prompt"])
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, rendered)
    prompt_token_ids = tuple(int(value) for value in inputs["input_ids"][0].tolist())
    source_tokens = [int(value) for value in source["generated_token_ids"]][:max_new_tokens]
    seed_everything(int(config["decode"]["seed"]))
    device = torch.device(model_config.device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    with NativeRouteCaptureContext(ops) as prompt_capture, torch.inference_mode():
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
        prompt_records = prompt_capture.drain()
    if tuple(record.layer for record in prompt_records) != tuple(range(LAYERS)):
        raise RuntimeError("accuracy prefill did not capture all routed layers")
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("accuracy closed loop did not return a KV cache")
    prefill_scores = cast(Tensor, prefill.logits[:, -1])
    first = _sample_token(prefill_scores, model, do_sample=False)
    generated = [int(first.item())]
    nlls: list[float] = []
    if source_tokens:
        nlls.append(float(-prefill_scores.float().log_softmax(dim=-1)[0, source_tokens[0]]))
    prompt_ids = inputs["input_ids"]
    current = first[:, None].to(device)
    last_cached_token_id = int(prompt_ids[0, -1])
    history = route_scores(prompt_records, tail_tokens=HORIZON, experts=EXPERTS)
    active: dict[int, tuple[int, ...]] | None = None
    current_window: list[tuple[SubsetRouteRecord, ...]] = []
    accounting = RouteAccounting()
    expert_bytes = {layer: physical_expert_bytes(ops, layer) for layer in range(LAYERS)}
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(LAYERS)}
    natural_digest = hashlib.sha256()
    executed_digest = hashlib.sha256()
    subset_digest = hashlib.sha256()
    probe = _probe(model, ops, policy)
    planning_rows: list[dict[str, object]] = []
    probe_latency = 0.0
    content_latency = 0.0
    oracle_latency = 0.0
    oracle_calls = 0
    probe_attention_queries = 0
    probe_attention_calls = 0
    probe_router_calls = 0
    probe_expert_calls = 0
    probe_syncs = 0
    probe_peak_temporary = 0
    continuation_matches = 0
    continuation_fallbacks = 0
    boundary_count = 0
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
        if decode_step % HORIZON == 0:
            previous_execution_subset = active
            current_window = []
            anchors, content = _anchor_plan(
                policy,
                model=model,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=tuple(generated),
                current=current,
                cache=cache,
            )
            result = probe.predict(
                cache,
                sampled_next_token_id=int(current.item()),
                current_token_id=last_cached_token_id,
                anchor_token_ids=anchors,
                execution_subsets=previous_execution_subset,
                execution_subset_source=(
                    "full_native_top8_prefill_access"
                    if previous_execution_subset is None
                    else "current_policy_previous_realized_window_subset"
                ),
            )
            active = candidate_subsets(result, history, SELECTOR, budget=BUDGET)
            passed = _planning_audit_pass(policy, result, content)
            if not passed:
                raise RuntimeError(f"planning audit failed at boundary {boundary_count}")
            planning_rows.append(
                {
                    "boundary": boundary_count,
                    "generated_token_index": len(generated) - 1,
                    "anchor_token_ids": list(anchors),
                    "previous_execution_subset_supplied": (previous_execution_subset is not None),
                    "planning_audit_pass": passed,
                    "content": content,
                    "probe_cost": asdict(result.cost),
                    "probe_audit": result.audit,
                }
            )
            probe_latency += result.cost.latency_seconds_measured
            content_latency += cast(float, content["content_planning_latency_seconds_measured"])
            oracle_latency += cast(float, content["oracle_lookahead_latency_seconds_measured"])
            oracle_calls += cast(int, content["oracle_natural_forward_calls"])
            probe_attention_queries += result.cost.attention_queries
            probe_attention_calls += result.cost.attention_calls
            probe_router_calls += result.cost.router_calls
            probe_expert_calls += result.cost.expert_calls
            probe_syncs += result.cost.cpu_gpu_synchronizations
            probe_peak_temporary = max(
                probe_peak_temporary,
                result.cost.temporary_cuda_bytes_measured,
            )
            continuation_matches += int(bool(content["continuation_match"]))
            continuation_fallbacks += int(bool(content["continuation_fallback"]))
            for layer in range(LAYERS):
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
            boundary_count += 1
        if active is None:
            raise AssertionError("accuracy subset planning did not run")
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
        current = token[:, None].to(device)
        last_cached_token_id = processed_token_id
        decode_step += 1
        if decode_step % HORIZON == 0:
            history = route_scores(current_window, experts=EXPERTS)
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
    exact_matches, comparison_tokens = _exact_matches(generated, source_tokens)
    nll = sum(nlls) / len(nlls) if nlls else float("nan")
    total_bytes = accounting.planned_transfer_bytes
    all_planning_audits_pass = all(bool(row["planning_audit_pass"]) for row in planning_rows)
    deployable = policy != "future_exact_content_oracle"
    return {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "stage": stage,
        "model": config["model"]["key"],
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "precision": config["model"]["precision"],
        "task": config["dataset"]["key"],
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "policy": policy,
        "policy_role": (
            "deployable_calibration_free" if deployable else "nondeployable_content_oracle"
        ),
        "information_regime": (
            "online_post_sample_current_policy_known_context_only"
            if deployable
            else "current_policy_full_expert_future_content_oracle"
        ),
        "evaluation_mode": "actual_hard_closed_loop_generation",
        "hard_mask_executed": True,
        "identity_materialized": False,
        "outside_subset_router_logits_masked": True,
        "native_topk_and_normalization_after_mask": True,
        "horizon": HORIZON,
        "budget": BUDGET,
        "resident_fraction": BUDGET / EXPERTS,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "generated_tokens": len(generated),
        "v17_natural_token_ids": source_tokens,
        "exact_token_matches": exact_matches,
        "exact_token_comparison_tokens": comparison_tokens,
        "exact_token_agreement": agreement,
        "first_token_divergence": first_token_divergence,
        "first_route_divergence": accounting.first_route_divergence,
        "route_divergent_calls": accounting.route_divergent_calls,
        "route_calls": accounting.route_calls,
        "executed_route_changed": accounting.route_divergent_calls > 0,
        "natural_route_ids_sha256": natural_digest.hexdigest(),
        "executed_route_ids_sha256": executed_digest.hexdigest(),
        "subset_trajectory_sha256": subset_digest.hexdigest(),
        "route_hits": accounting.route_hits,
        "route_slots": accounting.route_slots,
        "route_hit_rate": (
            accounting.route_hits / accounting.route_slots if accounting.route_slots else 1.0
        ),
        "selected_mass_hit": accounting.selected_mass_hit,
        "selected_mass_total": accounting.selected_mass_total,
        "selected_routing_mass_coverage": (
            accounting.selected_mass_hit / accounting.selected_mass_total
            if accounting.selected_mass_total
            else 1.0
        ),
        "prefetch_loads": accounting.prefetch_loads,
        "subset_churn": accounting.subset_churn,
        "simulated_planned_transfer_bytes": total_bytes,
        "simulated_natural_reference_bytes": accounting.natural_reference_bytes,
        "simulated_estimated_transfer_reduction": (
            1 - total_bytes / accounting.natural_reference_bytes
            if accounting.natural_reference_bytes
            else 0.0
        ),
        "v17_token_nll_on_policy_context": nll,
        "perplexity": _perplexity(nll),
        "nll_tokens": len(nlls),
        "correct": score.correct,
        "parsed_answer": score.parsed_answer,
        "score_detail": score.detail,
        "boundary_count": boundary_count,
        "planning_rows": planning_rows,
        "all_planning_audits_pass": all_planning_audits_pass,
        "probe_latency_seconds_measured": probe_latency,
        "content_planning_latency_seconds_measured": content_latency,
        "oracle_lookahead_latency_seconds_measured": oracle_latency,
        "oracle_natural_forward_calls": oracle_calls,
        "probe_attention_queries": probe_attention_queries,
        "probe_attention_calls": probe_attention_calls,
        "probe_router_calls": probe_router_calls,
        "probe_expert_calls": probe_expert_calls,
        "probe_cpu_gpu_synchronizations": probe_syncs,
        "probe_peak_temporary_cuda_bytes_measured": probe_peak_temporary,
        "continuation_matched_boundaries": continuation_matches,
        "continuation_fallback_boundaries": continuation_fallbacks,
        "one_extra_pseudo_forward_per_boundary": True,
        "single_extra_forward_deployable_constraint_satisfied": deployable,
        "future_v17_tokens_used_by_policy": False,
        "label_or_correctness_used_during_policy_execution": False,
        "elapsed_seconds_measured": elapsed,
        "tokens_per_second_measured": len(generated) / elapsed if elapsed else None,
        "runtime_kind": "measured_actual_closed_loop_including_policy_planning",
        "peak_cuda_allocated_bytes_measured": int(torch.cuda.max_memory_allocated(device)),
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "execution_git_head": execution_git_head,
        "source_v17_row_sha256": sha256_json(source),
        "source_v17_correct": bool(source["correct"]),
        "source_v17_generated_tokens": int(source["generated_tokens"]),
        "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
        "source_target_sha256": source["target_sha256"],
    }


def _work(
    samples: dict[str, Any],
    *,
    stage: Stage,
    shard_index: int,
    shard_count: int,
) -> list[tuple[dict[str, Any], PilotPolicy]]:
    if shard_count != 2 or not 0 <= shard_index < shard_count:
        raise ValueError("accuracy pilot requires the frozen two shards")
    if stage == "smoke":
        reference = next(
            row
            for row in samples["accuracy"]
            if row["sample_id"] == samples["mechanism_smoke"]["sample_id"]
        )
        return [(reference, policy) for policy in POLICIES] if shard_index == 0 else []
    units = [(reference, policy) for reference in samples["accuracy"] for policy in POLICIES]
    return [unit for index, unit in enumerate(units) if index % shard_count == shard_index]


def run(
    *,
    stage: Stage,
    physical_gpu: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    config, samples = _protocol()
    units = _work(
        samples,
        stage=stage,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    cap = (
        int(samples["mechanism_smoke"]["max_new_tokens"])
        if stage == "smoke"
        else int(config["dataset"]["original_v17_max_new_tokens"])
    )
    missing = [
        (reference, policy)
        for reference, policy in units
        if _load_checksummed(
            _sample_path(stage, int(reference["row_index"]), policy),
            stage=stage,
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=cap,
        )
        is None
    ]
    if not missing:
        return {"state": "already_complete", "stage": stage, "rows": len(units)}
    accuracy = load_accuracy_suite_config(config["source"]["accuracy_config"])
    model_config = _source_model(accuracy, physical_gpu)
    datasets = [dataset for dataset in accuracy.datasets if dataset.key == "gsm8k"]
    if len(datasets) != 1:
        raise ValueError("frozen v17 source is missing GSM8K")
    examples = load_examples(datasets[0], cache_dir=accuracy.dataset_cache_dir)
    sources = _source_rows(config)
    seed_everything(int(config["decode"]["seed"]))
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed model facts changed")
    revision = _git_head()
    completed = 0
    for reference, policy in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        _check_reference(reference, source)
        example = _model_example(examples[row_index], model_config)
        if example.sample_id != reference["sample_id"]:
            raise ValueError(f"dataset/source mismatch: {reference['sample_id']}")
        max_tokens = min(example.max_new_tokens, cap)
        path = _sample_path(stage, row_index, policy)
        try:
            row = run_policy_sample(
                config,
                accuracy,
                model_config,
                model,
                tokenizer,
                ops,
                example,
                source,
                policy=policy,
                stage=stage,
                max_new_tokens=max_tokens,
                physical_gpu=physical_gpu,
                execution_git_head=revision,
            )
            _write_checksummed(path, row)
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "sample_id": reference["sample_id"],
                        "policy": policy,
                        "tokens": row["generated_tokens"],
                        "correct": row["correct"],
                        "elapsed_seconds": row["elapsed_seconds_measured"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException as error:
            write_json_atomic(
                _failure_path(stage, row_index, policy),
                {
                    "schema_version": 1,
                    "state": "failed",
                    "pilot_id": PILOT_ID,
                    "config_sha256": CONFIG_SHA256,
                    "sample_manifest_sha256": SAMPLES_SHA256,
                    "stage": stage,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": policy,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "physical_gpu": physical_gpu,
                    "execution_git_head": revision,
                },
            )
            raise
    return {
        "state": "complete",
        "stage": stage,
        "shard_index": shard_index,
        "completed_now": completed,
    }


def audit_smoke() -> dict[str, object]:
    config, samples = _protocol()
    reference = samples["accuracy"][0]
    rows = []
    for policy in POLICIES:
        row = _load_checksummed(
            _sample_path("smoke", int(reference["row_index"]), policy),
            stage="smoke",
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=int(samples["mechanism_smoke"]["max_new_tokens"]),
        )
        if row is None:
            raise ValueError("accuracy mechanism smoke is incomplete")
        rows.append(row)
    all_pass = all(
        bool(row["hard_mask_executed"])
        and not bool(row["identity_materialized"])
        and bool(row["all_planning_audits_pass"])
        and bool(row["executed_route_changed"])
        and not bool(row["future_v17_tokens_used_by_policy"])
        and not bool(row["label_or_correctness_used_during_policy_execution"])
        and int(row["probe_attention_calls"]) == int(row["boundary_count"]) * LAYERS
        and int(row["probe_attention_queries"]) == int(row["boundary_count"]) * LAYERS * HORIZON
        for row in rows
    )
    if not all_pass:
        raise RuntimeError("accuracy mechanism smoke audit failed")
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "sample_policy_rows": len(rows),
        "all_pass": True,
        "actual_hard_mask_execution": True,
        "identity_materialized_rows": 0,
        "all_planning_audits_pass": True,
        "future_v17_tokens_used": False,
        "accuracy_or_correctness_selected_policy": False,
        "actual_execution_authorized": True,
        "config_scope": config["status"],
    }
    write_json_atomic(OUTPUT / "smoke" / "audit.json", result)
    return result


def _actual_rows(
    config: dict[str, Any],
    samples: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    cap = int(config["dataset"]["original_v17_max_new_tokens"])
    for reference in samples["accuracy"]:
        for policy in POLICIES:
            row = _load_checksummed(
                _sample_path("actual", int(reference["row_index"]), policy),
                stage="actual",
                row_index=int(reference["row_index"]),
                sample_id=str(reference["sample_id"]),
                policy=policy,
                max_new_tokens=cap,
            )
            if row is None:
                raise ValueError("accuracy aggregate is incomplete")
            rows.append(row)
    return rows


def _policy_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    successes = sum(bool(row["correct"]) for row in rows)
    generated = sum(int(row["generated_tokens"]) for row in rows)
    runtime = sum(float(row["elapsed_seconds_measured"]) for row in rows)
    route_hits = sum(int(row["route_hits"]) for row in rows)
    route_slots = sum(int(row["route_slots"]) for row in rows)
    mass_hit = sum(float(row["selected_mass_hit"]) for row in rows)
    mass_total = sum(float(row["selected_mass_total"]) for row in rows)
    exact_matches = sum(int(row["exact_token_matches"]) for row in rows)
    exact_total = sum(int(row["exact_token_comparison_tokens"]) for row in rows)
    nll_tokens = sum(int(row["nll_tokens"]) for row in rows)
    weighted_nll = (
        sum(float(row["v17_token_nll_on_policy_context"]) * int(row["nll_tokens"]) for row in rows)
        / nll_tokens
    )
    planned = sum(int(row["simulated_planned_transfer_bytes"]) for row in rows)
    natural = sum(int(row["simulated_natural_reference_bytes"]) for row in rows)
    boundaries = sum(int(row["boundary_count"]) for row in rows)
    return {
        "samples": len(rows),
        "successes": successes,
        "accuracy_measured": successes / len(rows),
        "accuracy_gate_pass": successes >= 7,
        "strong_preservation_signal": successes == 8,
        "total_generated_tokens": generated,
        "total_runtime_seconds_measured": runtime,
        "aggregate_tokens_per_second_measured": generated / runtime,
        "exact_token_matches": exact_matches,
        "exact_token_comparison_tokens": exact_total,
        "exact_token_agreement_weighted": exact_matches / exact_total,
        "samples_with_any_token_divergence": sum(
            row["first_token_divergence"] is not None for row in rows
        ),
        "mean_first_token_divergence_when_present": (
            sum(
                int(row["first_token_divergence"])
                for row in rows
                if row["first_token_divergence"] is not None
            )
            / max(1, sum(row["first_token_divergence"] is not None for row in rows))
        ),
        "route_hit_rate": route_hits / route_slots,
        "selected_routing_mass_coverage": mass_hit / mass_total,
        "weighted_v17_token_nll_on_policy_context": weighted_nll,
        "weighted_perplexity": math.exp(weighted_nll) if weighted_nll < 700 else float("inf"),
        "boundaries": boundaries,
        "probe_latency_seconds_measured": sum(
            float(row["probe_latency_seconds_measured"]) for row in rows
        ),
        "content_planning_latency_seconds_measured": sum(
            float(row["content_planning_latency_seconds_measured"]) for row in rows
        ),
        "oracle_lookahead_latency_seconds_measured": sum(
            float(row["oracle_lookahead_latency_seconds_measured"]) for row in rows
        ),
        "oracle_natural_forward_calls": sum(
            int(row["oracle_natural_forward_calls"]) for row in rows
        ),
        "continuation_matched_boundaries": sum(
            int(row["continuation_matched_boundaries"]) for row in rows
        ),
        "continuation_fallback_boundaries": sum(
            int(row["continuation_fallback_boundaries"]) for row in rows
        ),
        "simulated_planned_transfer_bytes": planned,
        "simulated_natural_reference_bytes": natural,
        "simulated_estimated_transfer_reduction": 1 - planned / natural,
        "all_planning_audits_pass": all(bool(row["all_planning_audits_pass"]) for row in rows),
        "actual_hard_closed_loop_rows": len(rows),
        "identity_materialized_rows": sum(bool(row["identity_materialized"]) for row in rows),
    }


def aggregate() -> dict[str, object]:
    config, samples = _protocol()
    smoke = audit_smoke()
    rows = _actual_rows(config, samples)
    grouped = {policy: [row for row in rows if row["policy"] == policy] for policy in POLICIES}
    summaries = {policy: _policy_summary(grouped[policy]) for policy in POLICIES}
    sources = _source_rows(config)
    correctness: dict[str, dict[str, bool]] = {
        "vanilla_v17": {
            str(reference["sample_id"]): bool(sources[int(reference["row_index"])]["correct"])
            for reference in samples["accuracy"]
        }
    }
    for policy in POLICIES:
        correctness[policy] = {
            str(row["sample_id"]): bool(row["correct"]) for row in grouped[policy]
        }
    paired = paired_accuracy_bootstrap(
        correctness,
        samples=int(config["accuracy_gate"]["paired_bootstrap_samples"]),
        seed=int(config["accuracy_gate"]["paired_bootstrap_seed"]),
    )
    recent = correctness["recent_sequence_causal"]
    unigram = correctness["sampled_unigram_full_continuation"]
    ids = [str(reference["sample_id"]) for reference in samples["accuracy"]]
    direct = {
        "baseline": "recent_sequence_causal",
        "candidate": "sampled_unigram_full_continuation",
        "paired_gains": sum(unigram[sample] and not recent[sample] for sample in ids),
        "paired_losses": sum(recent[sample] and not unigram[sample] for sample in ids),
        "paired_equal": sum(recent[sample] == unigram[sample] for sample in ids),
        "accuracy_difference": (
            cast(
                float,
                summaries["sampled_unigram_full_continuation"]["accuracy_measured"],
            )
            - cast(float, summaries["recent_sequence_causal"]["accuracy_measured"])
        ),
    }
    unigram_successes = cast(int, summaries["sampled_unigram_full_continuation"]["successes"])
    decision = (
        "PILOT_NARROW"
        if unigram_successes == 8
        else "PILOT_NARROW_WITH_ONE_ALLOWED_LOSS"
        if unigram_successes == 7
        else "STOP_PIVOT"
    )
    per_sample = [
        {
            "sample_id": sample,
            "row_index": next(
                int(reference["row_index"])
                for reference in samples["accuracy"]
                if reference["sample_id"] == sample
            ),
            "vanilla_v17_correct": correctness["vanilla_v17"][sample],
            "policies": {
                policy: {
                    "correct": correctness[policy][sample],
                    "generated_tokens": next(
                        int(row["generated_tokens"])
                        for row in grouped[policy]
                        if row["sample_id"] == sample
                    ),
                    "exact_token_agreement": next(
                        float(row["exact_token_agreement"])
                        for row in grouped[policy]
                        if row["sample_id"] == sample
                    ),
                    "first_token_divergence": next(
                        row["first_token_divergence"]
                        for row in grouped[policy]
                        if row["sample_id"] == sample
                    ),
                    "elapsed_seconds_measured": next(
                        float(row["elapsed_seconds_measured"])
                        for row in grouped[policy]
                        if row["sample_id"] == sample
                    ),
                }
                for policy in POLICIES
            },
        }
        for sample in ids
    ]
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "samples": len(ids),
        "sample_policy_rows": len(rows),
        "vanilla_v17": {
            "samples": len(ids),
            "successes": sum(correctness["vanilla_v17"].values()),
            "accuracy_measured_preexisting": (sum(correctness["vanilla_v17"].values()) / len(ids)),
            "regenerated": False,
        },
        "policies": summaries,
        "sampled_unigram_vs_recent_sequence": direct,
        "paired_accuracy_bootstrap": paired,
        "per_sample": per_sample,
        "accuracy_gate": {
            "allowed_drop_questions_frozen": 1,
            "minimum_policy_successes_frozen": 7,
            "strong_preservation_successes_frozen": 8,
        },
        "smoke_audit": smoke,
        "decision": decision,
        "conclusion_ceiling": "PILOT_NARROW",
        "future_exact_deployable": False,
    }
    write_json_atomic(OUTPUT / "actual" / "summary.json", result)
    write_json_atomic(OUTPUT / "actual" / "per_sample.json", {"rows": per_sample})
    write_json_atomic(
        OUTPUT / "actual" / "paired_accuracy_bootstrap.json",
        {"rows": paired},
    )
    write_json_atomic(
        OUTPUT / "actual" / "decision.json",
        {
            "pilot_id": PILOT_ID,
            "decision": decision,
            "selected_deployable_policy": None,
            "accuracy_selected_or_tuned_variant": False,
            "sampled_unigram_accuracy_gate_pass": unigram_successes >= 7,
            "future_exact_deployable": False,
            "full_dataset_go": False,
        },
    )
    return result


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# One-forward Qwen/GSM8K eight-row accuracy pilot v1",
        "",
        "All values below use actual hard closed-loop generation on the same eight frozen rows.",
        "Frozen vanilla is reused rather than regenerated.",
        "",
        (
            "| Policy | Correct | Accuracy | Token agreement | Route hit | Selected mass | "
            "Runtime s | Probe s | Oracle s |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        values = summary["policies"][policy]
        lines.append(
            f"| {policy} | {values['successes']}/8 | "
            f"{float(values['accuracy_measured']):.4f} | "
            f"{float(values['exact_token_agreement_weighted']):.6f} | "
            f"{float(values['route_hit_rate']):.6f} | "
            f"{float(values['selected_routing_mass_coverage']):.6f} | "
            f"{float(values['total_runtime_seconds_measured']):.2f} | "
            f"{float(values['probe_latency_seconds_measured']):.2f} | "
            f"{float(values['oracle_lookahead_latency_seconds_measured']):.2f} |"
        )
    direct = summary["sampled_unigram_vs_recent_sequence"]
    lines.extend(
        [
            "",
            "## Paired deployable comparison",
            "",
            f"Sampled-unigram versus recent-sequence gains/losses/equal: "
            f"{direct['paired_gains']}/{direct['paired_losses']}/{direct['paired_equal']}.",
            "",
            "## Scope boundary",
            "",
            f"Focused decision: **{summary['decision']}**. The maximum conclusion is "
            "PILOT_NARROW because N=8. Future-exact uses up to seven full-expert "
            "autoregressive lookahead calls plus one pseudo traversal per boundary and is "
            "not deployable. Accuracy, token identity, NLL, route coverage, probe cost, "
            "and total generation runtime are measured. Transfer reduction is simulated. "
            "No production offloading speedup or full-dataset accuracy claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _manifest() -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(value for value in OUTPUT.rglob("*") if value.is_file()):
        relative = str(path.relative_to(OUTPUT))
        if relative in excluded:
            continue
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(OUTPUT / "artifact_manifest.json", manifest)
    return manifest


def finalize() -> dict[str, object]:
    config, samples = _protocol()
    summary = cast(dict[str, Any], aggregate())
    rows = _actual_rows(config, samples)
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "git_head_at_report": _git_head(),
            "row_execution_git_heads": sorted({str(row["execution_git_head"]) for row in rows}),
            "row_pids": sorted({int(row["pid"]) for row in rows}),
            "row_ppids": sorted({int(row["ppid"]) for row in rows}),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "actual_closed_loop_gsm8k_accuracy",
                "generated_tokens_and_exact_token_agreement",
                "v17_token_nll_on_policy_context_and_perplexity",
                "route_hit_and_selected_routing_mass",
                "probe_and_total_generation_runtime",
                "peak_cuda_memory",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": [
                "production_offloading_runtime",
                "production_runtime_speedup",
                "full_dataset_accuracy",
            ],
        },
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "smoke_sample_policy_rows_validated": 3,
            "actual_sample_policy_rows_validated": len(rows),
            "atomic_json_rows": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "pilot_id": PILOT_ID,
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "sample_ids": [row["sample_id"] for row in samples["accuracy"]],
            "source_v17_vanilla_rows_sha256": config["source"]["vanilla_rows_sha256"],
            "vanilla_regenerated": False,
            "learned_or_fitted_parameters": False,
            "offline_continuation_tables": False,
            "future_v17_tokens_used": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "pilot_id": PILOT_ID,
            "decision": summary["decision"],
            "scope": "Qwen_GSM8K_H8_B32_eight_row_actual_closed_loop_pilot",
            "conclusion_ceiling": "PILOT_NARROW",
            "future_exact_deployable": False,
            "runtime_speedup_claim": False,
            "full_dataset_go": False,
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report(summary))
    manifest = _manifest()
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "decision": summary["decision"],
        "actual_sample_policy_rows": len(rows),
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": True,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    config, samples = _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "report_v1":
        raise ValueError("accuracy pilot pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual_files = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual_files != recorded:
        raise ValueError("accuracy pilot artifact file set changed")
    for artifact in manifest["artifacts"]:
        path = OUTPUT / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"accuracy pilot checksum changed: {path}")
    rows = _actual_rows(config, samples)
    if len(rows) != 24:
        raise ValueError("accuracy pilot actual row count changed")
    if any(
        not bool(row["hard_mask_executed"])
        or bool(row["identity_materialized"])
        or not bool(row["all_planning_audits_pass"])
        or bool(row["future_v17_tokens_used_by_policy"])
        or bool(row["label_or_correctness_used_during_policy_execution"])
        for row in rows
    ):
        raise ValueError("accuracy pilot row audit changed")
    deployable = [row for row in rows if row["policy"] != "future_exact_content_oracle"]
    if any(
        not bool(row["single_extra_forward_deployable_constraint_satisfied"])
        or int(row["oracle_natural_forward_calls"]) != 0
        for row in deployable
    ):
        raise ValueError("deployable single-extra-forward claim changed")
    oracle = [row for row in rows if row["policy"] == "future_exact_content_oracle"]
    if any(
        bool(row["single_extra_forward_deployable_constraint_satisfied"])
        or int(row["oracle_natural_forward_calls"]) <= 0
        for row in oracle
    ):
        raise ValueError("future-exact oracle cost boundary changed")
    resume = _json(OUTPUT / "resume_audit.json")
    if (
        resume["smoke_sample_policy_rows_validated"] != 3
        or resume["actual_sample_policy_rows_validated"] != 24
        or not resume["checksum_resume_pass"]
    ):
        raise ValueError("accuracy pilot resume audit changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "actual_sample_policy_rows": len(rows),
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run-smoke", "smoke-audit", "run", "aggregate", "finalize", "validate"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.command == "run-smoke":
        result = run(
            stage="smoke",
            physical_gpu=args.gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.command == "smoke-audit":
        result = audit_smoke()
    elif args.command == "run":
        result = run(
            stage="actual",
            physical_gpu=args.gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
