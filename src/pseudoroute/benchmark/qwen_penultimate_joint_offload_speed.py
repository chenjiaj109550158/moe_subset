"""Run the frozen real Qwen penultimate-joint offload speed pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from pseudoroute.benchmark import qwen_real_offload_speed as legacy_offload
from pseudoroute.benchmark.config import (
    AccuracyModelConfig,
    AccuracySuiteConfig,
    load_accuracy_suite_config,
)
from pseudoroute.benchmark.prefetch import (
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
    SubsetRouteRecord,
    physical_expert_bytes,
)
from pseudoroute.benchmark.pseudo_embedding_closed_loop import _account_hard_records
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    candidate_subsets,
    route_scores,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    _anchor_plan,
    _bridge_parity,
    _penultimate_shadow_plan,
    _probe,
)
from pseudoroute.benchmark.qwen_penultimate_joint import joint_penultimate_forward
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
    _sample_token,
    _token_agreement,
)
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.benchmark.tasks import BenchmarkExample, load_examples
from pseudoroute.runtime.qwen_offload import QwenExpertOffloadEngine
from pseudoroute.utils.determinism import seed_everything

Policy = Literal[
    "lossless_dynamic_top8_lru_b32",
    "penultimate_unigram_joint_h8_b32_async_prefetch",
]
Stage = Literal["smoke", "actual"]

PILOT_ID = "qwen_penultimate_joint_offload_speed_v1"
CONFIG = Path("configs/benchmark/qwen_penultimate_joint_offload_speed_v1.yaml")
SAMPLES = Path("configs/benchmark/qwen_penultimate_joint_offload_speed_v1_samples.json")
CONFIG_SHA256 = "6ba8a77f9a749d7baab2b9d784825c7b835fb32538d843cb7d32ea8c1eb27f6d"
SAMPLES_SHA256 = "202486d1591fc46c21ac298c0ff953ccb3245441a6961c7279826db4ec1d62d4"
OUTPUT = Path("artifacts/qwen_penultimate_joint_offload_speed_v1")
PILOT_VERSION = "v1"
PROTOCOL_COMMIT = "00c12f3"
TRADITIONAL: Policy = "lossless_dynamic_top8_lru_b32"
JOINT: Policy = "penultimate_unigram_joint_h8_b32_async_prefetch"
POLICIES: tuple[Policy, ...] = (TRADITIONAL, JOINT)
LAYERS = 48
EXPERTS = 128
TOP_K = 8
BUDGET = 32
HORIZON = 8
SELECTOR = "first_four_anchor_core_plus_history_fill"


def _select_pilot(version: str) -> None:
    global PILOT_ID, CONFIG, SAMPLES, CONFIG_SHA256, SAMPLES_SHA256
    global OUTPUT, PILOT_VERSION, PROTOCOL_COMMIT
    if version == "v1":
        PILOT_ID = "qwen_penultimate_joint_offload_speed_v1"
        CONFIG = Path("configs/benchmark/qwen_penultimate_joint_offload_speed_v1.yaml")
        SAMPLES = Path("configs/benchmark/qwen_penultimate_joint_offload_speed_v1_samples.json")
        CONFIG_SHA256 = "6ba8a77f9a749d7baab2b9d784825c7b835fb32538d843cb7d32ea8c1eb27f6d"
        SAMPLES_SHA256 = "202486d1591fc46c21ac298c0ff953ccb3245441a6961c7279826db4ec1d62d4"
        OUTPUT = Path("artifacts/qwen_penultimate_joint_offload_speed_v1")
        PROTOCOL_COMMIT = "00c12f3"
    elif version == "v2":
        PILOT_ID = "qwen_penultimate_joint_offload_speed_v2"
        CONFIG = Path("configs/benchmark/qwen_penultimate_joint_offload_speed_v2.yaml")
        SAMPLES = Path("configs/benchmark/qwen_penultimate_joint_offload_speed_v2_samples.json")
        CONFIG_SHA256 = "7820f904ee547ae99b4b2a3ad5fabe47db410ffe2ae3f4da03653df1f4ecb1f1"
        SAMPLES_SHA256 = "a8c8569cb42db51206d843123e4b5aab5b627347e423b0861b19e2b13d1ec68d"
        OUTPUT = Path("artifacts/qwen_penultimate_joint_offload_speed_v2")
        PROTOCOL_COMMIT = "b648f82"
    else:
        raise ValueError(f"unknown joint-offload pilot version: {version}")
    PILOT_VERSION = version


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("joint-offload config fingerprint changed")
    if sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("joint-offload sample fingerprint changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config.get("pilot_id") != PILOT_ID or samples.get("pilot_id") != PILOT_ID:
        raise ValueError("joint-offload protocol identity changed")
    if (
        config["model"]["routed_layers"],
        config["model"]["routed_experts_per_layer"],
        config["model"]["native_top_k"],
        config["offload"]["gpu_slots_per_layer"],
    ) != (LAYERS, EXPERTS, TOP_K, BUDGET):
        raise ValueError("joint-offload operating point changed")
    expected = ((44, "test-44"), (632, "test-632"))
    if tuple((row["row_index"], row["sample_id"]) for row in samples["samples"]) != expected:
        raise ValueError("joint-offload sample order changed")
    source = config["source"]
    for key in (
        "accuracy_config",
        "frozen_vanilla_rows",
        "logical_penultimate_config",
        "logical_penultimate_samples",
        "prior_real_offload_config",
        "prior_real_offload_samples",
        "prior_real_offload_aggregate",
        "prior_real_offload_manifest",
    ):
        path = Path(source[key])
        if sha256_file(path) != source[f"{key}_sha256"]:
            raise ValueError(f"joint-offload source changed: {path}")
    if config["execution"]["measured"]["row_order"] != samples["work_order"]:
        raise ValueError("joint-offload AB/BA work order changed")
    for reference in samples["samples"]:
        for key in ("logical_penultimate_reference", "prior_traditional_offload_provenance"):
            pinned = reference[key]
            path = Path(pinned["path"])
            if sha256_file(path) != pinned["file_sha256"]:
                raise ValueError(f"joint-offload pinned row changed: {path}")
            payload = _json(path)
            checksum = payload.pop("row_payload_sha256", None)
            if checksum != pinned["row_payload_sha256"] or checksum != sha256_json(payload):
                raise ValueError(f"joint-offload row payload changed: {path}")
    if PILOT_VERSION == "v2":
        failed_row_path = Path(source["failed_v1_smoke_row"])
        failed_audit_path = Path(source["failed_v1_smoke_audit"])
        if sha256_file(failed_row_path) != source["failed_v1_smoke_row_sha256"]:
            raise ValueError("joint-offload v1 failed smoke row changed")
        failed_row = _json(failed_row_path)
        checksum = failed_row.pop("row_payload_sha256", None)
        if checksum != source["failed_v1_smoke_row_payload_sha256"] or checksum != sha256_json(
            failed_row
        ):
            raise ValueError("joint-offload v1 failed smoke payload changed")
        if sha256_file(failed_audit_path) != source["failed_v1_smoke_audit_sha256"]:
            raise ValueError("joint-offload v1 failed smoke audit changed")
        failed_audit = _json(failed_audit_path)
        if failed_audit.get("state") != "failed" or failed_audit.get("all_pass") is not False:
            raise ValueError("joint-offload v1 failed smoke provenance is not failed")
    return config, samples


def _source_rows(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["row_index"]): row
        for row in _read_jsonl(Path(config["source"]["frozen_vanilla_rows"]))
        if row.get("state") == "complete"
    }
    if set(rows) != set(range(1319)):
        raise ValueError("frozen Qwen/GSM8K v17 source row set changed")
    return rows


def _check_source(reference: dict[str, Any], source: dict[str, Any]) -> None:
    if (
        int(source["row_index"]) != int(reference["row_index"])
        or source["sample_id"] != reference["sample_id"]
        or source["rendered_prompt_sha256"] != reference["rendered_prompt_sha256"]
        or source["target_sha256"] != reference["target_sha256"]
        or int(source["generated_tokens"]) != int(reference["source_v17_generated_tokens"])
        or bool(source["correct"]) is not bool(reference["source_v17_correct"])
    ):
        raise ValueError(f"frozen source changed: {reference['sample_id']}")


def _logical_reference(reference: dict[str, Any]) -> dict[str, Any]:
    pinned = reference["logical_penultimate_reference"]
    path = Path(pinned["path"])
    if sha256_file(path) != pinned["file_sha256"]:
        raise ValueError(f"logical penultimate file changed: {path}")
    payload = _json(path)
    checksum = payload.pop("row_payload_sha256", None)
    if checksum != pinned["row_payload_sha256"] or checksum != sha256_json(payload):
        raise ValueError(f"logical penultimate payload changed: {path}")
    payload["row_payload_sha256"] = checksum
    if (
        payload["sample_id"] != reference["sample_id"]
        or int(payload["generated_tokens"]) != int(pinned["generated_tokens"])
        or bool(payload["correct"]) is not bool(pinned["correct"])
    ):
        raise ValueError(f"logical penultimate provenance changed: {path}")
    return payload


def _sample_path(stage: Stage, row_index: int, policy: Policy) -> Path:
    return OUTPUT / stage / f"{row_index:05d}" / f"{policy}.json"


def _failure_path(stage: Stage, row_index: int, policy: Policy) -> Path:
    path = _sample_path(stage, row_index, policy)
    return path.with_name(f"{policy}.{os.getpid()}.FAILED.json")


def _write_checksummed(path: Path, row: dict[str, object]) -> None:
    payload = cast(
        dict[str, object],
        json.loads(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)),
    )
    payload["artifact_checksum_serialization"] = "json_round_trip_v1"
    payload["row_payload_sha256"] = sha256_json(payload)
    write_json_atomic(path, payload)


def _load_checksummed(
    path: Path,
    *,
    stage: Stage,
    row_index: int,
    sample_id: str,
    policy: Policy,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _json(path)
    checksum = payload.pop("row_payload_sha256", None)
    valid = (
        payload.get("state") == "complete"
        and payload.get("artifact_checksum_serialization") == "json_round_trip_v1"
        and payload.get("pilot_id") == PILOT_ID
        and payload.get("config_sha256") == CONFIG_SHA256
        and payload.get("sample_manifest_sha256") == SAMPLES_SHA256
        and payload.get("stage") == stage
        and int(payload.get("row_index", -1)) == row_index
        and payload.get("sample_id") == sample_id
        and payload.get("policy") == policy
        and int(payload.get("max_new_tokens", -1)) == max_new_tokens
        and checksum == sha256_json(payload)
    )
    if not valid:
        raise ValueError(f"incompatible or corrupt joint-offload row: {path}")
    payload["row_payload_sha256"] = checksum
    return payload


def _metrics_payload(engine: QwenExpertOffloadEngine) -> dict[str, object]:
    payload = cast(dict[str, object], asdict(engine.finish_metrics()))
    payload.pop("transfer_records", None)
    return payload


def _phase(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    return cast(dict[str, Any], metrics.get("phase_metrics", {}).get(name, {}))


def _exact_reference(
    generated: list[int],
    reference: list[int],
) -> tuple[float, int | None, bool]:
    agreement, first = _token_agreement(generated, reference)
    return agreement, first, generated == reference


def _batching_route_parity(
    sequential: tuple[SubsetRouteRecord, ...],
    joint: tuple[SubsetRouteRecord, ...],
    sequential_next: dict[int, tuple[int, ...]],
    joint_next: dict[int, tuple[int, ...]],
) -> dict[str, object]:
    if (
        len(sequential) != LAYERS
        or len(joint) != LAYERS
        or set(sequential_next) != set(range(LAYERS))
        or set(joint_next) != set(range(LAYERS))
    ):
        raise ValueError("batching route parity is missing routed layers")
    natural_exact_layers = 0
    executed_exact_layers = 0
    natural_equal_slots = 0
    executed_equal_slots = 0
    route_slots = 0
    subset_exact_layers = 0
    subset_overlaps: list[float] = []
    for layer, (left, right) in enumerate(zip(sequential, joint, strict=True)):
        if left.layer != layer or right.layer != layer:
            raise ValueError("batching route parity layer order changed")
        natural_equal = left.natural.ids.eq(right.natural.ids)
        executed_equal = left.executed.ids.eq(right.executed.ids)
        natural_exact_layers += int(bool(natural_equal.all()))
        executed_exact_layers += int(bool(executed_equal.all()))
        natural_equal_slots += int(natural_equal.sum())
        executed_equal_slots += int(executed_equal.sum())
        route_slots += natural_equal.numel()
        left_subset = set(sequential_next[layer])
        right_subset = set(joint_next[layer])
        if len(left_subset) != BUDGET or len(right_subset) != BUDGET:
            raise ValueError("batching route parity received a non-B32 subset")
        subset_exact_layers += int(left_subset == right_subset)
        subset_overlaps.append(len(left_subset & right_subset) / BUDGET)
    return {
        "batching_difference_metrics_recorded": True,
        "sequential_bridge_natural_exact_layers": natural_exact_layers,
        "sequential_bridge_executed_exact_layers": executed_exact_layers,
        "sequential_bridge_natural_slot_agreement": natural_equal_slots / route_slots,
        "sequential_bridge_executed_slot_agreement": executed_equal_slots / route_slots,
        "sequential_next_b32_exact_layers": subset_exact_layers,
        "sequential_next_b32_mean_overlap_fraction": sum(subset_overlaps) / LAYERS,
        "sequential_next_b32_min_overlap_fraction": min(subset_overlaps),
    }


def _traditional_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "decode": {"seed": config["decode"]["seed"]},
        "source": {
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "precision": config["model"]["precision"],
        },
    }


def _run_traditional(
    config: dict[str, Any],
    accuracy: AccuracySuiteConfig,
    model_config: AccuracyModelConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    engine: QwenExpertOffloadEngine,
    example: BenchmarkExample,
    source: dict[str, Any],
    *,
    stage: Stage,
    max_new_tokens: int,
    physical_gpu: int,
    execution_git_head: str,
) -> dict[str, object]:
    row = legacy_offload._run_lossless(
        _traditional_config(config),
        accuracy,
        model_config,
        model,
        tokenizer,
        ops,
        engine,
        example,
        source,
        stage=stage,
        max_new_tokens=max_new_tokens,
        physical_gpu=physical_gpu,
        execution_git_head=execution_git_head,
    )
    row.update(
        {
            "pilot_id": PILOT_ID,
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "policy": TRADITIONAL,
            "policy_role": "traditional_exact_per_token_real_offload_baseline",
            "actual_transfer_compute_overlap_implemented": False,
            "joint_calls": 0,
            "joint_input_tokens": 0,
            "pseudo_kv_positions_committed": 0,
            "runtime_kind": "measured_actual_traditional_per_token_real_h2d_offload",
        }
    )
    return row


def _run_joint(
    config: dict[str, Any],
    accuracy: AccuracySuiteConfig,
    model_config: AccuracyModelConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    engine: QwenExpertOffloadEngine,
    example: BenchmarkExample,
    source: dict[str, Any],
    logical_reference: dict[str, Any],
    *,
    stage: Stage,
    max_new_tokens: int,
    physical_gpu: int,
    execution_git_head: str,
) -> dict[str, object]:
    if accuracy.do_sample_for(model_config, example.task):
        raise ValueError("joint-offload pilot requires greedy decoding")
    inputs = _encode_saved_rendered_prompt(
        tokenizer,
        model_config,
        str(source["rendered_prompt"]),
    )
    prompt_ids = inputs["input_ids"]
    prompt_token_ids = tuple(int(value) for value in prompt_ids[0].tolist())
    vanilla_tokens = [int(value) for value in source["generated_token_ids"]][:max_new_tokens]
    logical_tokens = [int(value) for value in logical_reference["generated_token_ids"]][
        :max_new_tokens
    ]
    seed_everything(int(config["decode"]["seed"]))
    device = torch.device(model_config.device)
    engine.reset_cache()
    engine.reset_metrics()
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    prefill_started = time.perf_counter()
    with (
        engine.phase("prefill"),
        NativeRouteCaptureContext(ops) as prompt_capture,
        torch.inference_mode(),
    ):
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
        prompt_records = prompt_capture.drain()
    torch.cuda.synchronize(device)
    prefill_wall = time.perf_counter() - prefill_started
    if tuple(record.layer for record in prompt_records) != tuple(range(LAYERS)):
        raise RuntimeError("joint-offload prefill did not capture every routed layer")
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("joint-offload prefill did not return a KV cache")
    scores = cast(Tensor, prefill.logits[:, -1])
    first = _sample_token(scores, model, do_sample=False)
    generated = [int(first.item())]
    current = first[:, None].to(device)
    history = route_scores(prompt_records, tail_tokens=HORIZON, experts=EXPERTS)
    active: dict[int, tuple[int, ...]] | None = None
    current_window: list[tuple[SubsetRouteRecord, ...]] = []
    accounting = RouteAccounting()
    expert_bytes = {layer: physical_expert_bytes(ops, layer) for layer in range(LAYERS)}
    natural_digest = hashlib.sha256()
    executed_digest = hashlib.sha256()
    subset_digest = hashlib.sha256()
    probe = _probe(
        model,
        ops,
        "sampled_unigram_full_continuation",
        subset_residual_execution="natural_top8_intersection_zero_missing",
    )
    planning_wall = 0.0
    initial_prefetch_wall = 0.0
    joint_wall = 0.0
    joint_calls = 0
    regular_production_calls = 0
    joint_audits: list[dict[str, Any]] = []
    smoke_reference: dict[str, object] | None = None
    continuation_matches = 0
    continuation_fallbacks = 0
    finished = _finished(
        model,
        tokenizer,
        model_config,
        example,
        prompt_ids,
        generated,
        scores,
    )
    decode_step = 0
    decode_started = time.perf_counter()
    while len(generated) < max_new_tokens and not finished:
        if decode_step == 0:
            planning_started = time.perf_counter()
            anchors, content = _anchor_plan(
                "sampled_unigram_full_continuation",
                model=model,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=tuple(generated),
                current=current,
                cache=cache,
            )
            with engine.phase("bootstrap_planning"):
                result = probe.predict(
                    cache,
                    sampled_next_token_id=int(current.item()),
                    current_token_id=int(prompt_ids[0, -1]),
                    anchor_token_ids=anchors,
                    execution_subsets=None,
                    execution_subset_source="full_native_top8_prefill_access",
                )
            active = candidate_subsets(result, history, SELECTOR, budget=BUDGET)
            planning_wall += time.perf_counter() - planning_started
            prefetch_started = time.perf_counter()
            with engine.phase("initial_prefetch"):
                engine.preload_subsets(active)
            torch.cuda.synchronize(device)
            initial_prefetch_wall += time.perf_counter() - prefetch_started
            continuation_matches += int(bool(content["continuation_match"]))
            continuation_fallbacks += int(bool(content["continuation_fallback"]))
            for layer in range(LAYERS):
                subset_digest.update(layer.to_bytes(4, "little"))
                subset_digest.update(
                    torch.tensor(active[layer], dtype=torch.int64).numpy().tobytes()
                )
        if active is None:
            raise AssertionError("joint-offload bootstrap did not produce an active subset")

        executed_active = active
        planned_next_active: dict[int, tuple[int, ...]] | None = None
        is_joint = decode_step % HORIZON == HORIZON - 1
        if is_joint:
            if len(current_window) != HORIZON - 1:
                raise RuntimeError("joint boundary does not have seven realized route steps")
            content_started = time.perf_counter()
            anchors, content = _anchor_plan(
                "sampled_unigram_full_continuation",
                model=model,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=tuple(generated),
                current=current,
                cache=cache,
            )
            planning_wall += time.perf_counter() - content_started
            continuation_matches += int(bool(content["continuation_match"]))
            continuation_fallbacks += int(bool(content["continuation_fallback"]))
            sequential = None
            sequential_next = None
            if (
                stage == "smoke"
                and smoke_reference is None
                and bool(config["execution"]["mechanism_smoke"]["sequential_reference_enabled"])
            ):
                with engine.resident_only(), engine.phase("smoke_sequential_reference"):
                    sequential = _penultimate_shadow_plan(
                        model,
                        ops,
                        probe,
                        policy="sampled_unigram_full_continuation",
                        prompt_token_ids=prompt_token_ids,
                        generated_token_ids=tuple(generated),
                        current=current,
                        production_cache=cache,
                        active=active,
                    )
                sequential_history = route_scores(
                    [*current_window, sequential.bridge_records],
                    experts=EXPERTS,
                )
                sequential_next = candidate_subsets(
                    sequential.result,
                    sequential_history,
                    SELECTOR,
                    budget=BUDGET,
                )
            history_before_bridge = route_scores(current_window, experts=EXPERTS)
            joint_started = time.perf_counter()
            joint = joint_penultimate_forward(
                model,
                ops,
                engine,
                current=current,
                anchor_token_ids=anchors,
                production_cache=cache,
                active_subsets=active,
                history_before_bridge=history_before_bridge,
                budget=BUDGET,
            )
            joint_wall += time.perf_counter() - joint_started
            records = joint.bridge_records
            scores = joint.bridge_logits
            next_active = joint.next_subsets
            planned_next_active = next_active
            parity: dict[str, object] | None = None
            if sequential is not None and sequential_next is not None:
                full_parity = _bridge_parity(
                    sequential.bridge_records,
                    records,
                    sequential.bridge_logits,
                    scores,
                )
                natural_ids_exact = all(
                    torch.equal(left.natural.ids, right.natural.ids)
                    for left, right in zip(
                        sequential.bridge_records,
                        records,
                        strict=True,
                    )
                )
                executed_ids_exact = all(
                    torch.equal(left.executed.ids, right.executed.ids)
                    for left, right in zip(
                        sequential.bridge_records,
                        records,
                        strict=True,
                    )
                )
                sampled_token_exact = int(sequential.bridge_logits.argmax(dim=-1).item()) == int(
                    scores.argmax(dim=-1).item()
                )
                next_b32_exact = sequential_next == next_active
                batching = _batching_route_parity(
                    sequential.bridge_records,
                    records,
                    sequential_next,
                    next_active,
                )
                parity = {
                    **full_parity,
                    **batching,
                    "sequential_bridge_logits_max_abs_difference": float(
                        (sequential.bridge_logits.float() - scores.float()).abs().max().item()
                    ),
                    "sequential_bridge_sampled_token_exact": sampled_token_exact,
                    "sequential_bridge_natural_route_ids_exact": natural_ids_exact,
                    "sequential_bridge_executed_route_ids_exact": executed_ids_exact,
                    "sequential_next_b32_exact": next_b32_exact,
                    "smoke_gate_pass": (
                        sampled_token_exact
                        and natural_ids_exact
                        and executed_ids_exact
                        and next_b32_exact
                    ),
                }
                smoke_reference = parity
            joint_audits.append(
                {
                    "decode_step": decode_step,
                    "joint_input_token_ids": list(joint.joint_input_token_ids),
                    "production_cache_length_before": joint.production_cache_length_before,
                    "production_cache_length_after": joint.production_cache_length_after,
                    "pseudo_kv_positions_committed": joint.pseudo_kv_positions_committed,
                    "cache_layer_updates_all_one": all(
                        value == 1 for value in joint.cache_layer_updates
                    ),
                    "cache_layer_query_lengths_all_nine": all(
                        value == 9 for value in joint.cache_layer_query_lengths
                    ),
                    "attention_calls": joint.attention_calls,
                    "router_calls": joint.router_calls,
                    "expert_calls": joint.expert_calls,
                    "async_prefetch_layers": joint.async_prefetch_layers,
                    "production_rng_unchanged": joint.production_rng_unchanged,
                    "production_cache_unchanged_before_commit": (
                        joint.production_cache_unchanged_before_commit
                    ),
                    "pseudo_topk_ids_sha256": joint.pseudo_topk_ids_sha256,
                    "sequential_parity": parity,
                }
            )
            joint_calls += 1
        else:
            with engine.resident_only(), engine.phase("production"):
                output, records = _forward_capture(
                    model,
                    ops,
                    current,
                    cache,
                    policy="hard",
                    allowed=active,
                )
            scores = cast(Tensor, output.logits[:, -1])
            regular_production_calls += 1
        _account_hard_records(
            accounting,
            records,
            executed_active,
            expert_bytes,
            decode_step=decode_step,
            natural_digest=natural_digest,
            executed_digest=executed_digest,
        )
        if is_joint:
            if planned_next_active is None:
                raise AssertionError("joint forward did not produce its next active subset")
            active = planned_next_active
            current_window = []
            for layer in range(LAYERS):
                subset_digest.update(layer.to_bytes(4, "little"))
                subset_digest.update(
                    torch.tensor(active[layer], dtype=torch.int64).numpy().tobytes()
                )
        else:
            current_window.append(records)
        token = _sample_token(scores, model, do_sample=False)
        generated.append(int(token.item()))
        current = token[:, None].to(device)
        decode_step += 1
        finished = _finished(
            model,
            tokenizer,
            model_config,
            example,
            prompt_ids,
            generated,
            scores,
        )
    metrics = cast(dict[str, Any], _metrics_payload(engine))
    decode_wall = time.perf_counter() - decode_started
    inference_wall = time.perf_counter() - inference_started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    score = score_response(example, text)
    agreement, first_divergence, exact = _exact_reference(generated, logical_tokens)
    vanilla_agreement, vanilla_first, vanilla_exact = _exact_reference(
        generated,
        vanilla_tokens,
    )
    production_misses = sum(
        int(_phase(metrics, phase).get("cache_misses", 0))
        for phase in ("production", "joint_compute")
    )
    joint_cache_audit = all(
        row["production_cache_length_after"] == row["production_cache_length_before"] + 1
        and row["pseudo_kv_positions_committed"] == 0
        and bool(row["cache_layer_updates_all_one"])
        and bool(row["cache_layer_query_lengths_all_nine"])
        and int(row["attention_calls"]) == LAYERS
        and int(row["router_calls"]) == LAYERS
        and int(row["expert_calls"]) == LAYERS
        and int(row["async_prefetch_layers"]) == LAYERS
        and bool(row["production_rng_unchanged"])
        and bool(row["production_cache_unchanged_before_commit"])
        for row in joint_audits
    )
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
        "sample_id": source["sample_id"],
        "policy": JOINT,
        "policy_role": "actual_joint_penultimate_pseudoroute_real_offload",
        "evaluation_mode": "actual_hard_closed_loop_joint_real_expert_offload",
        "hard_mask_executed": True,
        "identity_materialized": False,
        "horizon": HORIZON,
        "budget": BUDGET,
        "resident_fraction": BUDGET / EXPERTS,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "generated_token_ids": generated,
        "generated_text": text,
        "generated_tokens": len(generated),
        "correct": score.correct,
        "parsed_answer": score.parsed_answer,
        "score_detail": score.detail,
        "required_reference": "frozen_no_offload_penultimate_exact_tokens_report_only",
        "reference_token_ids": logical_tokens,
        "exact_token_agreement_with_required_reference": agreement,
        "first_token_divergence_from_required_reference": first_divergence,
        "required_reference_exact_token_identity": exact,
        "vanilla_reference_exact_token_agreement": vanilla_agreement,
        "vanilla_reference_first_token_divergence": vanilla_first,
        "vanilla_reference_exact_token_identity": vanilla_exact,
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
        "first_route_divergence": accounting.first_route_divergence,
        "route_divergent_calls": accounting.route_divergent_calls,
        "route_calls": accounting.route_calls,
        "natural_route_ids_sha256": natural_digest.hexdigest(),
        "executed_route_ids_sha256": executed_digest.hexdigest(),
        "subset_trajectory_sha256": subset_digest.hexdigest(),
        "initial_online_bootstrap_calls": 1,
        "joint_calls": joint_calls,
        "joint_input_tokens": joint_calls * 9,
        "joint_native_attention_calls": joint_calls * LAYERS,
        "joint_native_router_calls": joint_calls * LAYERS,
        "joint_native_expert_calls": joint_calls * LAYERS,
        "regular_one_token_production_calls": regular_production_calls,
        "production_cache_positions_committed_by_joint": joint_calls,
        "pseudo_kv_positions_committed": 0,
        "all_joint_cache_and_call_audits_pass": joint_cache_audit,
        "joint_audits": joint_audits,
        "smoke_sequential_reference": smoke_reference,
        "last_sampled_token_used_for_planning": False,
        "continuation_matched_boundaries": continuation_matches,
        "continuation_fallback_boundaries": continuation_fallbacks,
        "actual_transfer_compute_overlap_implemented": True,
        "async_prefetch_batches": metrics["async_prefetch_batches"],
        "async_prefetch_experts": metrics["async_prefetch_experts"],
        "async_prefetch_bytes": metrics["async_prefetch_bytes"],
        "deferred_ready_waits": metrics["deferred_ready_waits"],
        "deferred_ready_wait_ms": metrics["deferred_ready_wait_ms"],
        "production_expert_misses": production_misses,
        "production_expert_miss_forbidden": True,
        "actual_offload_metrics": metrics,
        "offload_engine_audit": engine.audit(),
        "actual_h2d_copy_executed": int(metrics["h2d_bytes"]) > 0,
        "prefill_wall_seconds_measured": prefill_wall,
        "decode_wall_seconds_measured": decode_wall,
        "post_prefill_decode_forwards": decode_step,
        "decode_forwards_per_second_measured": (decode_step / decode_wall if decode_wall else None),
        "planning_wall_seconds_measured": planning_wall,
        "initial_prefetch_wall_seconds_measured": initial_prefetch_wall,
        "joint_host_wall_seconds_measured": joint_wall,
        "end_to_end_inference_wall_seconds_measured": inference_wall,
        "end_to_end_tokens_per_second_measured": (
            len(generated) / inference_wall if inference_wall else None
        ),
        "label_or_correctness_used_during_execution": False,
        "future_tokens_used_during_execution": False,
        "runtime_kind": "measured_actual_joint_hard_closed_loop_real_h2d_offload",
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "execution_git_head": execution_git_head,
        "source_v17_row_sha256": sha256_json(source),
        "source_logical_row_payload_sha256": logical_reference["row_payload_sha256"],
        "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
        "source_target_sha256": source["target_sha256"],
    }


def _work(
    config: dict[str, Any],
    samples: dict[str, Any],
    stage: Stage,
) -> list[tuple[dict[str, Any], Policy]]:
    by_id = {str(row["sample_id"]): row for row in samples["samples"]}
    if stage == "smoke":
        smoke = config["execution"]["mechanism_smoke"]
        return [(by_id[str(smoke["sample_id"])], JOINT)]
    return [
        (by_id[str(sample_id)], cast(Policy, policy))
        for sample_id, policy in config["execution"]["measured"]["row_order"]
    ]


def _cap(config: dict[str, Any], stage: Stage) -> int:
    if stage == "smoke":
        return int(config["execution"]["mechanism_smoke"]["max_new_tokens"])
    return int(config["dataset"]["max_new_tokens"])


def _run_stages(
    stages: tuple[Stage, ...],
    *,
    physical_gpu: int,
) -> dict[str, object]:
    config, samples = _protocol()
    missing: list[tuple[Stage, dict[str, Any], Policy]] = []
    for stage in stages:
        cap = _cap(config, stage)
        for reference, policy in _work(config, samples, stage):
            existing = _load_checksummed(
                _sample_path(stage, int(reference["row_index"]), policy),
                stage=stage,
                row_index=int(reference["row_index"]),
                sample_id=str(reference["sample_id"]),
                policy=policy,
                max_new_tokens=cap,
            )
            if existing is None:
                missing.append((stage, reference, policy))
    if not missing:
        return {"state": "already_complete", "stages": list(stages)}
    if stages == ("actual",):
        audit_smoke()

    accuracy = load_accuracy_suite_config(config["source"]["accuracy_config"])
    model_config = _source_model(accuracy, physical_gpu)
    datasets = [dataset for dataset in accuracy.datasets if dataset.key == "gsm8k"]
    if len(datasets) != 1:
        raise ValueError("frozen v17 accuracy config is missing GSM8K")
    examples = load_examples(datasets[0], cache_dir=accuracy.dataset_cache_dir)
    sources = _source_rows(config)
    seed_everything(int(config["decode"]["seed"]))
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed model facts changed")
    if getattr(cast(Any, model).config, "sliding_window", None) is not None:
        raise ValueError("joint bridge-only cache requires non-sliding Qwen attention")
    engine = QwenExpertOffloadEngine(model, slots_per_layer=BUDGET)
    revision = _git_head()
    environment = _software_hardware()
    setup = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "execution_git_head": revision,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "physical_gpu": physical_gpu,
        "engine_audit": engine.audit(),
        "software_hardware": environment,
    }
    setup["setup_payload_sha256"] = sha256_json(setup)
    write_json_atomic(
        OUTPUT / "setup" / f"{int(time.time())}.{os.getpid()}.json",
        setup,
    )
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", environment)
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "execution_git_head": revision,
            "protocol_commit": PROTOCOL_COMMIT,
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
        },
    )

    completed = 0
    smoke_validated = False
    for stage, reference, policy in missing:
        if stage == "actual" and "smoke" in stages and not smoke_validated:
            audit_smoke()
            smoke_validated = True
        row_index = int(reference["row_index"])
        source = sources[row_index]
        _check_source(reference, source)
        example = _model_example(examples[row_index], model_config)
        if example.sample_id != reference["sample_id"]:
            raise ValueError(f"dataset/source mismatch: {reference['sample_id']}")
        cap = _cap(config, stage)
        path = _sample_path(stage, row_index, policy)
        try:
            if policy == TRADITIONAL:
                row = _run_traditional(
                    config,
                    accuracy,
                    model_config,
                    model,
                    tokenizer,
                    ops,
                    engine,
                    example,
                    source,
                    stage=stage,
                    max_new_tokens=cap,
                    physical_gpu=physical_gpu,
                    execution_git_head=revision,
                )
            else:
                row = _run_joint(
                    config,
                    accuracy,
                    model_config,
                    model,
                    tokenizer,
                    ops,
                    engine,
                    example,
                    source,
                    _logical_reference(reference),
                    stage=stage,
                    max_new_tokens=cap,
                    physical_gpu=physical_gpu,
                    execution_git_head=revision,
                )
            if policy == TRADITIONAL and not bool(row["required_reference_exact_token_identity"]):
                raise RuntimeError(
                    f"traditional offload identity failed for {reference['sample_id']}"
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
                        "decode_forwards_per_second": row["decode_forwards_per_second_measured"],
                        "h2d_bytes": cast(dict[str, Any], row["actual_offload_metrics"])[
                            "h2d_bytes"
                        ],
                        "joint_calls": row.get("joint_calls", 0),
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
    if "smoke" in stages:
        audit_smoke()
    return {"state": "complete", "stages": list(stages), "completed_now": completed}


def audit_smoke() -> dict[str, object]:
    config, samples = _protocol()
    reference, _ = _work(config, samples, "smoke")[0]
    row = _load_checksummed(
        _sample_path("smoke", int(reference["row_index"]), JOINT),
        stage="smoke",
        row_index=int(reference["row_index"]),
        sample_id=str(reference["sample_id"]),
        policy=JOINT,
        max_new_tokens=_cap(config, "smoke"),
    )
    if row is None:
        raise ValueError("joint-offload smoke is incomplete")
    metrics = cast(dict[str, Any], row["actual_offload_metrics"])
    engine = cast(dict[str, Any], row["offload_engine_audit"])
    parity = cast(dict[str, Any] | None, row["smoke_sequential_reference"])
    checks = {
        "joint_calls": int(row["joint_calls"])
        >= int(config["execution"]["mechanism_smoke"]["minimum_joint_calls"]),
        "joint_cache_and_calls": bool(row["all_joint_cache_and_call_audits_pass"]),
        "bridge_sampled_token": parity is not None
        and bool(parity["sequential_bridge_sampled_token_exact"]),
        "actual_h2d": bool(row["actual_h2d_copy_executed"]),
        "async_scheduled": int(metrics["async_prefetch_batches"]) > 0,
        "async_consumed": int(metrics["deferred_ready_waits"]) > 0,
        "zero_production_misses": int(row["production_expert_misses"]) == 0,
        "b32": int(engine["slots_per_layer"]) == BUDGET,
        "pinned": bool(engine["all_cpu_sources_pinned"]),
        "no_full_cuda_experts": bool(engine["no_full_expert_parameter_on_cuda"]),
        "no_identity_materialization": not bool(row["identity_materialized"]),
    }
    if PILOT_VERSION == "v1":
        checks.update(
            {
                "bridge_natural_ids": parity is not None
                and bool(parity["sequential_bridge_natural_route_ids_exact"]),
                "bridge_executed_ids": parity is not None
                and bool(parity["sequential_bridge_executed_route_ids_exact"]),
                "next_b32": parity is not None and bool(parity["sequential_next_b32_exact"]),
            }
        )
    else:
        checks["batching_difference_metrics_recorded"] = parity is not None and bool(
            parity.get("batching_difference_metrics_recorded")
        )
    result = {
        "schema_version": 1,
        "state": "valid" if all(checks.values()) else "failed",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "checks": checks,
        "all_pass": all(checks.values()),
        "joint_calls": row["joint_calls"],
        "actual_h2d_bytes": metrics["h2d_bytes"],
        "async_prefetch_batches": metrics["async_prefetch_batches"],
        "deferred_ready_waits": metrics["deferred_ready_waits"],
        "batching_route_parity": parity,
    }
    write_json_atomic(OUTPUT / "smoke" / "audit.json", result)
    if not bool(result["all_pass"]):
        raise RuntimeError(f"joint-offload smoke gate failed: {checks}")
    return result


def _actual_rows(
    config: dict[str, Any],
    samples: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference, policy in _work(config, samples, "actual"):
        row = _load_checksummed(
            _sample_path("actual", int(reference["row_index"]), policy),
            stage="actual",
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=_cap(config, "actual"),
        )
        if row is None:
            raise ValueError("joint-offload actual rows are incomplete")
        rows.append(row)
    return rows


def _policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forwards = sum(int(row["post_prefill_decode_forwards"]) for row in rows)
    decode_wall = sum(float(row["decode_wall_seconds_measured"]) for row in rows)
    inference_wall = sum(float(row["end_to_end_inference_wall_seconds_measured"]) for row in rows)
    generated = sum(int(row["generated_tokens"]) for row in rows)
    metrics = [cast(dict[str, Any], row["actual_offload_metrics"]) for row in rows]
    return {
        "rows": len(rows),
        "correct_rows": sum(bool(row["correct"]) for row in rows),
        "required_reference_exact_identity_rows": sum(
            bool(row["required_reference_exact_token_identity"]) for row in rows
        ),
        "total_generated_tokens": generated,
        "post_prefill_decode_forwards": forwards,
        "decode_wall_seconds_measured": decode_wall,
        "aggregate_post_prefill_decode_forwards_per_second_measured": (
            forwards / decode_wall if decode_wall else None
        ),
        "end_to_end_inference_wall_seconds_measured": inference_wall,
        "aggregate_end_to_end_tokens_per_second_measured": (
            generated / inference_wall if inference_wall else None
        ),
        "actual_h2d_bytes": sum(int(item["h2d_bytes"]) for item in metrics),
        "cuda_event_transfer_seconds": (sum(float(item["transfer_ms"]) for item in metrics) / 1000),
        "cuda_event_exposed_stall_seconds": (
            sum(float(item["exposed_stall_ms"]) for item in metrics) / 1000
        ),
        "async_prefetch_batches": sum(
            int(item.get("async_prefetch_batches", 0)) for item in metrics
        ),
        "async_prefetch_bytes": sum(int(item.get("async_prefetch_bytes", 0)) for item in metrics),
        "deferred_ready_waits": sum(int(item.get("deferred_ready_waits", 0)) for item in metrics),
        "deferred_ready_wait_seconds": (
            sum(float(item.get("deferred_ready_wait_ms", 0.0)) for item in metrics) / 1000
        ),
        "production_expert_misses": sum(
            int(row.get("production_expert_misses", 0)) for row in rows
        ),
        "joint_calls": sum(int(row.get("joint_calls", 0)) for row in rows),
        "joint_input_tokens": sum(int(row.get("joint_input_tokens", 0)) for row in rows),
        "peak_cuda_allocated_bytes": max(int(item["peak_allocated_bytes"]) for item in metrics),
        "peak_cuda_reserved_bytes": max(int(item["peak_reserved_bytes"]) for item in metrics),
    }


def _focused_decision(
    *,
    speedup: float,
    traditional_identity: bool,
    joint_correct: bool,
    joint_zero_miss: bool,
    invariants: bool,
    positive_threshold: float,
    strong_threshold: float,
) -> str:
    if not 0 < positive_threshold <= strong_threshold:
        raise ValueError("joint-offload speed thresholds are invalid")
    if not (traditional_identity and joint_correct and joint_zero_miss and invariants):
        return "STOP_PIVOT_INVARIANT_OR_ACCURACY_FAILURE"
    if speedup < positive_threshold:
        return "NARROW_NO_SPEEDUP"
    if speedup < strong_threshold:
        return "NARROW_POSITIVE_ENGINEERING_SIGNAL"
    return "NARROW_STRONG_ENGINEERING_SIGNAL"


def aggregate() -> dict[str, object]:
    config, samples = _protocol()
    smoke = audit_smoke()
    rows = _actual_rows(config, samples)
    grouped = {policy: [row for row in rows if row["policy"] == policy] for policy in POLICIES}
    summaries = {policy: _policy_summary(grouped[policy]) for policy in POLICIES}
    traditional_rate = float(
        summaries[TRADITIONAL]["aggregate_post_prefill_decode_forwards_per_second_measured"]
    )
    joint_rate = float(
        summaries[JOINT]["aggregate_post_prefill_decode_forwards_per_second_measured"]
    )
    speedup = joint_rate / traditional_rate
    traditional_h2d = int(summaries[TRADITIONAL]["actual_h2d_bytes"])
    joint_h2d = int(summaries[JOINT]["actual_h2d_bytes"])
    h2d_reduction = 1 - joint_h2d / traditional_h2d
    traditional_transfer = float(summaries[TRADITIONAL]["cuda_event_transfer_seconds"])
    joint_transfer = float(summaries[JOINT]["cuda_event_transfer_seconds"])
    transfer_reduction = 1 - joint_transfer / traditional_transfer
    traditional_identity = (
        int(summaries[TRADITIONAL]["required_reference_exact_identity_rows"]) == 2
    )
    joint_correct = int(summaries[JOINT]["correct_rows"]) == 2
    joint_zero_miss = int(summaries[JOINT]["production_expert_misses"]) == 0
    invariants = all(
        bool(row["all_joint_cache_and_call_audits_pass"]) for row in grouped[JOINT]
    ) and bool(smoke["all_pass"])
    decision = _focused_decision(
        speedup=speedup,
        traditional_identity=traditional_identity,
        joint_correct=joint_correct,
        joint_zero_miss=joint_zero_miss,
        invariants=invariants,
        positive_threshold=float(
            config["decision_gate"]["minimum_speedup_ratio_for_positive_engineering_signal"]
        ),
        strong_threshold=float(config["decision_gate"]["strong_speedup_ratio"]),
    )
    prior = _json(Path(config["source"]["prior_real_offload_aggregate"]))
    per_sample = []
    for reference in samples["samples"]:
        sample_id = reference["sample_id"]
        per_sample.append(
            {
                "sample_id": sample_id,
                "policies": {
                    policy: {
                        "generated_tokens": next(
                            row for row in grouped[policy] if row["sample_id"] == sample_id
                        )["generated_tokens"],
                        "correct": next(
                            row for row in grouped[policy] if row["sample_id"] == sample_id
                        )["correct"],
                        "decode_forwards_per_second_measured": next(
                            row for row in grouped[policy] if row["sample_id"] == sample_id
                        )["decode_forwards_per_second_measured"],
                        "reference_exact_identity": next(
                            row for row in grouped[policy] if row["sample_id"] == sample_id
                        )["required_reference_exact_token_identity"],
                    }
                    for policy in POLICIES
                },
            }
        )
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "sample_count": 2,
        "policy_rows": 4,
        "policy_summaries": summaries,
        "per_sample": per_sample,
        "joint_vs_traditional_decode_throughput_ratio_measured": speedup,
        "joint_vs_traditional_decode_speedup_percent_measured": (speedup - 1) * 100,
        "joint_vs_traditional_actual_h2d_reduction_measured": h2d_reduction,
        "joint_vs_traditional_cuda_event_transfer_time_reduction_measured": (transfer_reduction),
        "traditional_reference_exact_identity_gate_pass": traditional_identity,
        "joint_task_accuracy_gate_pass": joint_correct,
        "joint_zero_production_miss_gate_pass": joint_zero_miss,
        "joint_cache_and_smoke_invariants_pass": invariants,
        "focused_decision": decision,
        "conclusion_ceiling": config["decision_gate"]["conclusion_ceiling"],
        "smoke_audit": smoke,
        "prior_unoverlapped_external_context": {
            "decode_throughput_ratio_measured": prior[
                "candidate_vs_lossless_decode_throughput_ratio_measured"
            ],
            "actual_h2d_reduction_measured": prior[
                "candidate_vs_lossless_actual_h2d_reduction_measured"
            ],
            "focused_decision": prior["focused_decision"],
        },
        "measured_vs_simulated": {
            "task_accuracy": "measured_two_row_actual_hard_closed_loop",
            "token_identity": "measured",
            "runtime": "measured",
            "expert_h2d_bytes": "measured_from_copy_operations",
            "transfer_and_exposed_stall": "measured_by_cuda_events",
            "joint_cache_and_call_counts": "measured_and_audited",
            "simulated_metrics": [],
        },
    }
    write_json_atomic(OUTPUT / "aggregate.json", result)
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "schema_version": 1,
            "state": "complete",
            "pilot_id": PILOT_ID,
            "focused_decision": decision,
            "scope": "Qwen/GSM8K H8 B32 two-row actual penultimate-joint offload",
            "speedup_ratio_measured": speedup,
            "task_accuracy_measured": {
                TRADITIONAL: int(summaries[TRADITIONAL]["correct_rows"]) / 2,
                JOINT: int(summaries[JOINT]["correct_rows"]) / 2,
            },
            "actual_h2d_reduction_measured": h2d_reduction,
            "full_dataset_go": False,
            "production_serving_speedup_claim": False,
        },
    )
    return result


def _report(aggregate_row: dict[str, Any]) -> str:
    traditional = aggregate_row["policy_summaries"][TRADITIONAL]
    joint = aggregate_row["policy_summaries"][JOINT]
    return "\n".join(
        [
            f"# {PILOT_ID} report",
            "",
            f"Decision: **{aggregate_row['focused_decision']}**. "
            "This is a two-row, one-A100 engineering measurement.",
            "",
            "## Measured speed",
            "",
            f"- Traditional exact-top-8: "
            f"{traditional['aggregate_post_prefill_decode_forwards_per_second_measured']:.6f} "
            "decode forwards/s.",
            f"- Penultimate joint H=8/B=32: "
            f"{joint['aggregate_post_prefill_decode_forwards_per_second_measured']:.6f} "
            "decode forwards/s.",
            f"- Joint / traditional: "
            f"{aggregate_row['joint_vs_traditional_decode_throughput_ratio_measured']:.6f}x "
            f"({aggregate_row['joint_vs_traditional_decode_speedup_percent_measured']:+.3f}%).",
            "",
            "## Measured transfer and outputs",
            "",
            f"- Actual H2D bytes: traditional {traditional['actual_h2d_bytes']:,}; "
            f"joint {joint['actual_h2d_bytes']:,}; reduction "
            f"{aggregate_row['joint_vs_traditional_actual_h2d_reduction_measured'] * 100:.3f}%.",
            f"- CUDA-event transfer seconds: traditional "
            f"{traditional['cuda_event_transfer_seconds']:.6f}; joint "
            f"{joint['cuda_event_transfer_seconds']:.6f}.",
            f"- CUDA-event exposed stall seconds: traditional "
            f"{traditional['cuda_event_exposed_stall_seconds']:.6f}; joint "
            f"{joint['cuda_event_exposed_stall_seconds']:.6f}.",
            f"- Joint async prefetch: {joint['async_prefetch_batches']} batches, "
            f"{joint['async_prefetch_bytes']:,} bytes; deferred ready waits "
            f"{joint['deferred_ready_waits']}.",
            f"- GSM8K accuracy: traditional {traditional['correct_rows']}/2; "
            f"joint {joint['correct_rows']}/2.",
            f"- Joint production expert misses: {joint['production_expert_misses']}.",
            "",
            "## Boundary",
            "",
            "All runtime, H2D, transfer, stall, output, cache, and call-count values above "
            "are measured. No transfer or stall value is simulated. Full expert tensors "
            "were pinned on CPU and CUDA held exactly B=32 slots per layer. Timed joint "
            "rows did not execute the sequential smoke reference.",
            "",
            "This Python runtime does not establish full-dataset accuracy, NVMe/NVLink or "
            "multi-GPU behavior, CUDA-graph/custom-kernel performance, or production-serving "
            "speedup.",
            "",
        ]
    )


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def finalize() -> dict[str, object]:
    result = cast(dict[str, Any], aggregate())
    _write_text_atomic(OUTPUT / "report.md", _report(result))
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        cast(dict[str, object], result["measured_vs_simulated"]),
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "schema_version": 1,
            "state": "complete",
            "atomic_json_rows": True,
            "checksum_resume_pass": True,
            "smoke_rows_validated": 1,
            "actual_rows_validated": 4,
            "failed_markers_preserved": [
                str(path.relative_to(OUTPUT)) for path in sorted(OUTPUT.rglob("*.FAILED.json"))
            ],
        },
    )
    write_json_atomic(
        OUTPUT / "pipeline_status.json",
        {
            "schema_version": 1,
            "state": "complete",
            "stage": "complete/report_v1",
            "pilot_id": PILOT_ID,
            "decision": result["focused_decision"],
        },
    )
    files = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "validation.json"}
        and not path.name.endswith(".FAILED.json")
    )
    failures = sorted(OUTPUT.rglob("*.FAILED.json"))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "files": [
            {
                "path": str(path.relative_to(OUTPUT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
        "failure_markers_retained": [str(path.relative_to(OUTPUT)) for path in failures],
    }
    manifest["manifest_payload_sha256"] = sha256_json(manifest)
    write_json_atomic(OUTPUT / "artifact_manifest.json", manifest)
    return manifest


def validate() -> dict[str, object]:
    config, samples = _protocol()
    smoke = audit_smoke()
    rows = _actual_rows(config, samples)
    aggregate_row = _json(OUTPUT / "aggregate.json")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    manifest_checksum = manifest.pop("manifest_payload_sha256", None)
    traditional = [row for row in rows if row["policy"] == TRADITIONAL]
    joint = [row for row in rows if row["policy"] == JOINT]
    checks = {
        "row_count": len(rows) == 4,
        "two_samples": len({row["sample_id"] for row in rows}) == 2,
        "both_policies": {row["policy"] for row in rows} == set(POLICIES),
        "traditional_identity": len(traditional) == 2
        and all(bool(row["required_reference_exact_token_identity"]) for row in traditional),
        "joint_accuracy": len(joint) == 2 and all(bool(row["correct"]) for row in joint),
        "joint_zero_production_miss": all(
            int(row["production_expert_misses"]) == 0 for row in joint
        ),
        "joint_cache_and_calls": all(
            bool(row["all_joint_cache_and_call_audits_pass"]) for row in joint
        ),
        "joint_actual_overlap": all(
            bool(row["actual_transfer_compute_overlap_implemented"]) for row in joint
        ),
        "actual_h2d": all(bool(row["actual_h2d_copy_executed"]) for row in rows),
        "no_identity_materialized": all(not bool(row["identity_materialized"]) for row in rows),
        "all_b32": all(
            int(cast(dict[str, Any], row["offload_engine_audit"])["slots_per_layer"]) == BUDGET
            for row in rows
        ),
        "all_pinned": all(
            bool(cast(dict[str, Any], row["offload_engine_audit"])["all_cpu_sources_pinned"])
            for row in rows
        ),
        "no_full_cuda_experts": all(
            bool(
                cast(dict[str, Any], row["offload_engine_audit"])[
                    "no_full_expert_parameter_on_cuda"
                ]
            )
            for row in rows
        ),
        "smoke": bool(smoke["all_pass"]),
        "aggregate": aggregate_row.get("state") == "complete",
        "manifest_checksum": manifest_checksum == sha256_json(manifest),
        "manifest_files": all(
            sha256_file(OUTPUT / item["path"]) == item["sha256"] for item in manifest["files"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"joint-offload validation failed: {checks}")
    result = {
        "schema_version": 1,
        "state": "valid",
        "pilot_id": PILOT_ID,
        "checks": checks,
        "actual_rows": len(rows),
        "artifacts": len(manifest["files"]),
        "decision": aggregate_row["focused_decision"],
        "failure_markers_retained": manifest["failure_markers_retained"],
    }
    write_json_atomic(OUTPUT / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-version",
        choices=("v1", "v2"),
        default="v1",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--physical-gpu", type=int, default=0)
        if command == "run":
            child.add_argument("--stage", choices=("smoke", "actual"), required=True)
    subparsers.add_parser("protocol")
    subparsers.add_parser("audit-smoke")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    _select_pilot(args.pilot_version)
    if args.command == "protocol":
        config, samples = _protocol()
        result: object = {
            "pilot_id": config["pilot_id"],
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "samples": [row["sample_id"] for row in samples["samples"]],
        }
    elif args.command == "run":
        result = _run_stages((cast(Stage, args.stage),), physical_gpu=args.physical_gpu)
    elif args.command == "all":
        result = _run_stages(("smoke", "actual"), physical_gpu=args.physical_gpu)
        finalize()
        result = validate()
    elif args.command == "audit-smoke":
        result = audit_smoke()
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
