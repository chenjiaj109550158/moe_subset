"""Run and report the frozen two-row real Qwen expert-offload speed pilot."""

from __future__ import annotations

import argparse
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

from pseudoroute.benchmark.config import (
    AccuracyModelConfig,
    AccuracySuiteConfig,
    load_accuracy_suite_config,
)
from pseudoroute.benchmark.prefetch import NativeRouteCaptureContext, Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_route import _source_model
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import run_policy_sample
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
    _model_example,
    _read_jsonl,
    _software_hardware,
)
from pseudoroute.benchmark.scoring import score_response
from pseudoroute.benchmark.subset_closed_loop import (
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
    "natural_top8_intersection_zero_missing_h8_b32",
]
Stage = Literal["smoke", "actual"]

PILOT_ID = "qwen_real_offload_speed_pilot_v1"
CONFIG = Path("configs/benchmark/qwen_real_offload_speed_pilot_v1.yaml")
SAMPLES = Path("configs/benchmark/qwen_real_offload_speed_pilot_v1_samples.json")
CONFIG_SHA256 = "9276363796e711baf487415a85fde49fc525ae42e7be701972867fb8713ed1ec"
SAMPLES_SHA256 = "aafd570fe56d878074cc6f5666dd26df8da528086776400226c15727b77ee819"
OUTPUT = Path("artifacts/qwen_real_offload_speed_pilot_v1")
LOSSLESS: Policy = "lossless_dynamic_top8_lru_b32"
CANDIDATE: Policy = "natural_top8_intersection_zero_missing_h8_b32"
POLICIES: tuple[Policy, ...] = (LOSSLESS, CANDIDATE)
LAYERS = 48
EXPERTS = 128
TOP_K = 8
BUDGET = 32
HORIZON = 8


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
        raise ValueError("real-offload config fingerprint changed")
    if sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("real-offload sample fingerprint changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config.get("pilot_id") != PILOT_ID or samples.get("pilot_id") != PILOT_ID:
        raise ValueError("real-offload pilot identity changed")
    if (
        config["model"]["routed_layers"],
        config["model"]["routed_experts_per_layer"],
        config["model"]["native_top_k"],
        config["offload"]["gpu_slots_per_layer"],
    ) != (LAYERS, EXPERTS, TOP_K, BUDGET):
        raise ValueError("real-offload operating point changed")
    expected = ((44, "test-44"), (632, "test-632"))
    if tuple((row["row_index"], row["sample_id"]) for row in samples["samples"]) != expected:
        raise ValueError("real-offload sample order changed")
    source = config["source"]
    for key in (
        "accuracy_config",
        "source_sample_manifest",
        "frozen_vanilla_rows",
        "frozen_mask_only_candidate_config",
        "frozen_mask_only_candidate_manifest",
    ):
        path = Path(source[key])
        if sha256_file(path) != source[f"{key}_sha256"]:
            raise ValueError(f"real-offload source changed: {path}")
    if config["execution"]["measured"]["row_order"] != [
        ["test-44", LOSSLESS],
        ["test-44", CANDIDATE],
        ["test-632", CANDIDATE],
        ["test-632", LOSSLESS],
    ]:
        raise ValueError("real-offload AB/BA work order changed")
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
        raise ValueError(f"frozen vanilla source changed: {reference['sample_id']}")


def _mask_reference(
    config: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    path = (
        Path(config["source"]["frozen_mask_only_candidate_root"])
        / f"{int(reference['row_index']):05d}"
        / "natural_top8_intersection_zero_missing.json"
    )
    payload = _json(path)
    checksum = payload.pop("row_payload_sha256", None)
    expected = reference["mask_only_candidate_row_payload_sha256"]
    if checksum != expected or checksum != sha256_json(payload):
        raise ValueError(f"mask-only candidate checksum changed: {path}")
    payload["row_payload_sha256"] = checksum
    if (
        payload.get("sample_id") != reference["sample_id"]
        or int(payload.get("generated_tokens", -1))
        != int(reference["mask_only_candidate_generated_tokens"])
        or bool(payload.get("correct")) is not bool(reference["mask_only_candidate_correct"])
    ):
        raise ValueError(f"mask-only candidate provenance changed: {reference['sample_id']}")
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
        raise ValueError(f"incompatible or corrupt real-offload row: {path}")
    payload["row_payload_sha256"] = checksum
    return payload


def _recover_candidate_diagnostic(
    config: dict[str, Any],
    reference: dict[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    row_index = int(reference["row_index"])
    path = _sample_path("actual", row_index, CANDIDATE)
    existing = _load_checksummed(
        path,
        stage="actual",
        row_index=row_index,
        sample_id=str(reference["sample_id"]),
        policy=CANDIDATE,
        max_new_tokens=max_new_tokens,
    )
    if existing is not None:
        return existing
    diagnostic_paths = sorted(
        path.parent.glob(f"{CANDIDATE}.*.DIAGNOSTIC.json"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    if not diagnostic_paths:
        return None
    diagnostic = _json(diagnostic_paths[0])
    checksum = diagnostic.pop("failure_payload_sha256", None)
    if (
        checksum != sha256_json(diagnostic)
        or diagnostic.get("state") != "failed_identity_diagnostic"
        or diagnostic.get("pilot_id") != PILOT_ID
        or diagnostic.get("config_sha256") != CONFIG_SHA256
        or diagnostic.get("sample_manifest_sha256") != SAMPLES_SHA256
        or int(diagnostic.get("row_index", -1)) != row_index
        or diagnostic.get("sample_id") != reference["sample_id"]
        or diagnostic.get("policy") != CANDIDATE
    ):
        raise ValueError(f"incompatible candidate identity diagnostic: {diagnostic_paths[0]}")
    diagnostic["failure_payload_sha256"] = checksum
    setup_rows = [_json(item) for item in sorted((OUTPUT / "setup").glob("*.json"))]
    setup = next(
        (item for item in setup_rows if int(item.get("pid", -1)) == int(diagnostic["pid"])),
        None,
    )
    if setup is None:
        raise ValueError("candidate diagnostic has no matching setup audit")
    metrics = cast(dict[str, Any], diagnostic["actual_offload_metrics"])
    production = cast(dict[str, Any], metrics["phase_metrics"]["production"])
    if int(production["cache_misses"]) != 0:
        raise ValueError("candidate diagnostic contains a production expert miss")
    generated = [int(value) for value in diagnostic["generated_token_ids"]]
    reference_tokens = [int(value) for value in diagnostic["reference_token_ids"]]
    agreement, first = _token_agreement(generated, reference_tokens)
    prefill_wall = float(diagnostic["prefill_wall_seconds_measured"])
    decode_wall = float(diagnostic["decode_wall_seconds_measured"])
    decode_forwards = max(0, len(generated) - 1)
    row: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "stage": "actual",
        "model": "qwen3_30b_a3b",
        "model_id": config["source"]["model_id"],
        "model_revision": config["source"]["model_revision"],
        "precision": config["source"]["precision"],
        "task": "gsm8k",
        "row_index": row_index,
        "sample_id": reference["sample_id"],
        "policy": CANDIDATE,
        "policy_role": "deployable_calibration_free_pseudoroute_real_offload",
        "evaluation_mode": "actual_hard_closed_loop_real_expert_offload",
        "hard_mask_executed": True,
        "identity_materialized": False,
        "horizon": HORIZON,
        "budget": BUDGET,
        "resident_fraction": BUDGET / EXPERTS,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "generated_token_ids": generated,
        "generated_text": None,
        "generated_tokens": len(generated),
        "required_reference": "frozen_mask_only_candidate_exact_tokens",
        "reference_token_ids": reference_tokens,
        "exact_token_agreement_with_required_reference": agreement,
        "first_token_divergence_from_required_reference": first,
        "required_reference_exact_token_identity": False,
        "candidate_reference_identity_gate_pass": False,
        "correct": bool(diagnostic["correct"]),
        "parsed_answer": None,
        "score_detail": "measured_before_identity_gate; recovered_from_atomic_diagnostic",
        "subset_trajectory_sha256": diagnostic.get("subset_trajectory_sha256"),
        "first_route_divergence": diagnostic.get("first_route_divergence"),
        "prefill_wall_seconds_measured": prefill_wall,
        "decode_wall_seconds_measured": decode_wall,
        "post_prefill_decode_forwards": decode_forwards,
        "decode_forwards_per_second_measured": (
            decode_forwards / decode_wall if decode_wall else None
        ),
        "planning_wall_seconds_measured": diagnostic["planning_wall_seconds_measured"],
        "prefetch_wall_seconds_measured": diagnostic["prefetch_wall_seconds_measured"],
        "production_wall_seconds_measured": diagnostic["production_wall_seconds_measured"],
        "end_to_end_inference_wall_seconds_measured": prefill_wall + decode_wall,
        "end_to_end_tokens_per_second_measured": (
            len(generated) / (prefill_wall + decode_wall) if prefill_wall + decode_wall else None
        ),
        "actual_offload_metrics": metrics,
        "offload_engine_audit": setup["engine_audit"],
        "actual_h2d_copy_executed": int(metrics["h2d_bytes"]) > 0,
        "production_expert_miss_forbidden": True,
        "label_or_correctness_used_during_execution": False,
        "future_tokens_used_during_execution": False,
        "runtime_kind": "measured_actual_hard_closed_loop_real_h2d_offload",
        "physical_gpu": int(setup["physical_gpu"]),
        "pid": int(diagnostic["pid"]),
        "ppid": int(diagnostic["ppid"]),
        "execution_git_head": diagnostic["execution_git_head"],
        "recovered_from_identity_failure_diagnostic": str(diagnostic_paths[0].relative_to(OUTPUT)),
        "measurement_completeness": {
            "generated_tokens": "measured",
            "task_correctness": "measured_before_identity_gate",
            "runtime": "measured",
            "h2d_and_cuda_events": "measured",
            "generated_text_and_parsed_answer": "not_retained_in_v1_diagnostic",
        },
    }
    _write_checksummed(path, row)
    return _load_checksummed(
        path,
        stage="actual",
        row_index=row_index,
        sample_id=str(reference["sample_id"]),
        policy=CANDIDATE,
        max_new_tokens=max_new_tokens,
    )


def _metrics_payload(engine: QwenExpertOffloadEngine) -> dict[str, object]:
    payload = cast(dict[str, object], asdict(engine.finish_metrics()))
    payload.pop("transfer_records", None)
    return payload


def _exact_reference(
    generated: list[int],
    reference: list[int],
) -> tuple[float, int | None, bool]:
    agreement, first = _token_agreement(generated, reference)
    return agreement, first, generated == reference


def _run_lossless(
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
    if accuracy.do_sample_for(model_config, example.task):
        raise ValueError("real-offload pilot requires greedy decoding")
    inputs = _encode_saved_rendered_prompt(
        tokenizer,
        model_config,
        str(source["rendered_prompt"]),
    )
    prompt_ids = inputs["input_ids"]
    reference_tokens = [int(value) for value in source["generated_token_ids"]][:max_new_tokens]
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
    if tuple(row.layer for row in prompt_records) != tuple(range(LAYERS)):
        raise RuntimeError("lossless prefill did not capture every routed layer")
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("lossless prefill did not return a KV cache")
    scores = cast(Tensor, prefill.logits[:, -1])
    first = _sample_token(scores, model, do_sample=False)
    generated = [int(first.item())]
    current = first[:, None].to(device)
    finished = _finished(
        model,
        tokenizer,
        model_config,
        example,
        prompt_ids,
        generated,
        scores,
    )
    decode_forwards = 0
    decode_started = time.perf_counter()
    while len(generated) < max_new_tokens and not finished:
        with engine.phase("production"):
            output, records = _forward_capture(
                model,
                ops,
                current,
                cache,
                policy="natural",
            )
        if any(tuple(record.natural.ids.shape)[-1:] != (TOP_K,) for record in records):
            raise RuntimeError("lossless decode route was not native top-8")
        scores = cast(Tensor, output.logits[:, -1])
        token = _sample_token(scores, model, do_sample=False)
        generated.append(int(token.item()))
        current = token[:, None].to(device)
        decode_forwards += 1
        finished = _finished(
            model,
            tokenizer,
            model_config,
            example,
            prompt_ids,
            generated,
            scores,
        )
    torch.cuda.synchronize(device)
    inference_ended = time.perf_counter()
    decode_wall = inference_ended - decode_started
    inference_wall = inference_ended - inference_started
    metrics = _metrics_payload(engine)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    score = score_response(example, text)
    agreement, first_divergence, exact = _exact_reference(generated, reference_tokens)
    return {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "stage": stage,
        "model": "qwen3_30b_a3b",
        "model_id": config["source"]["model_id"],
        "model_revision": config["source"]["model_revision"],
        "precision": config["source"]["precision"],
        "task": "gsm8k",
        "row_index": int(source["row_index"]),
        "sample_id": source["sample_id"],
        "policy": LOSSLESS,
        "policy_role": "traditional_lossless_exact_native_top8_offload",
        "evaluation_mode": "actual_closed_loop_real_expert_offload",
        "hard_mask_executed": False,
        "identity_materialized": False,
        "router_scope_experts": EXPERTS,
        "native_top_k": TOP_K,
        "horizon": 1,
        "budget": BUDGET,
        "resident_fraction": BUDGET / EXPERTS,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "generated_token_ids": generated,
        "generated_text": text,
        "generated_tokens": len(generated),
        "required_reference": "frozen_v17_vanilla_exact_tokens",
        "reference_token_ids": reference_tokens,
        "exact_token_agreement_with_required_reference": agreement,
        "first_token_divergence_from_required_reference": first_divergence,
        "required_reference_exact_token_identity": exact,
        "correct": score.correct,
        "parsed_answer": score.parsed_answer,
        "score_detail": score.detail,
        "prefill_wall_seconds_measured": prefill_wall,
        "decode_wall_seconds_measured": decode_wall,
        "post_prefill_decode_forwards": decode_forwards,
        "decode_forwards_per_second_measured": (
            decode_forwards / decode_wall if decode_wall else None
        ),
        "planning_wall_seconds_measured": 0.0,
        "prefetch_wall_seconds_measured": 0.0,
        "production_wall_seconds_measured": decode_wall,
        "end_to_end_inference_wall_seconds_measured": inference_wall,
        "end_to_end_tokens_per_second_measured": (
            len(generated) / inference_wall if inference_wall else None
        ),
        "actual_offload_metrics": metrics,
        "offload_engine_audit": engine.audit(),
        "actual_h2d_copy_executed": int(cast(Any, metrics["h2d_bytes"])) > 0,
        "production_expert_miss_forbidden": False,
        "label_or_correctness_used_during_execution": False,
        "future_tokens_used_during_execution": False,
        "runtime_kind": "measured_actual_closed_loop_real_h2d_offload",
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "execution_git_head": execution_git_head,
        "source_v17_row_sha256": sha256_json(source),
        "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
        "source_target_sha256": source["target_sha256"],
    }


def _run_candidate(
    config: dict[str, Any],
    accuracy: AccuracySuiteConfig,
    model_config: AccuracyModelConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    engine: QwenExpertOffloadEngine,
    example: BenchmarkExample,
    source: dict[str, Any],
    mask_reference: dict[str, Any],
    *,
    stage: Stage,
    max_new_tokens: int,
    physical_gpu: int,
    execution_git_head: str,
) -> dict[str, object]:
    runtime_config = {
        "decode": {"seed": config["decode"]["seed"]},
        "model": {
            "key": "qwen3_30b_a3b",
            "id": config["source"]["model_id"],
            "revision": config["source"]["model_revision"],
            "precision": config["source"]["precision"],
        },
        "dataset": {"key": "gsm8k"},
    }
    row = run_policy_sample(
        runtime_config,
        accuracy,
        model_config,
        model,
        tokenizer,
        ops,
        example,
        source,
        policy="sampled_unigram_full_continuation",
        stage=stage,
        max_new_tokens=max_new_tokens,
        physical_gpu=physical_gpu,
        offload_engine=engine,
        execution_git_head=execution_git_head,
        subset_residual_execution="natural_top8_intersection_zero_missing",
    )
    generated = [int(value) for value in cast(list[int], row["generated_token_ids"])]
    reference_tokens = [int(value) for value in mask_reference["generated_token_ids"]][
        :max_new_tokens
    ]
    agreement, first_divergence, exact = _exact_reference(generated, reference_tokens)
    row.update(
        {
            "schema_version": 1,
            "pilot_id": PILOT_ID,
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "policy": CANDIDATE,
            "policy_role": "deployable_calibration_free_pseudoroute_real_offload",
            "evaluation_mode": "actual_hard_closed_loop_real_expert_offload",
            "required_reference": "frozen_mask_only_candidate_exact_tokens",
            "reference_token_ids": reference_tokens,
            "exact_token_agreement_with_required_reference": agreement,
            "first_token_divergence_from_required_reference": first_divergence,
            "required_reference_exact_token_identity": exact,
            "offload_engine_audit": engine.audit(),
            "actual_h2d_copy_executed": int(
                cast(dict[str, Any], row["actual_offload_metrics"])["h2d_bytes"]
            )
            > 0,
            "production_expert_miss_forbidden": True,
            "label_or_correctness_used_during_execution": False,
            "future_tokens_used_during_execution": False,
            "runtime_kind": "measured_actual_hard_closed_loop_real_h2d_offload",
            "source_mask_only_row_payload_sha256": mask_reference["row_payload_sha256"],
        }
    )
    return row


def _work(
    config: dict[str, Any], samples: dict[str, Any], stage: Stage
) -> list[tuple[dict[str, Any], Policy]]:
    by_id = {str(row["sample_id"]): row for row in samples["samples"]}
    if stage == "smoke":
        smoke = config["execution"]["mechanism_smoke"]
        reference = by_id[str(smoke["sample_id"])]
        return [(reference, cast(Policy, policy)) for policy in smoke["policies"]]
    return [
        (by_id[str(sample_id)], cast(Policy, policy))
        for sample_id, policy in config["execution"]["measured"]["row_order"]
    ]


def _cap(config: dict[str, Any], stage: Stage) -> int:
    if stage == "smoke":
        return int(config["execution"]["mechanism_smoke"]["max_new_tokens"])
    return int(config["decode"]["max_new_tokens"])


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
            if existing is None and stage == "actual" and policy == CANDIDATE:
                existing = _recover_candidate_diagnostic(
                    config,
                    reference,
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
    engine = QwenExpertOffloadEngine(model, slots_per_layer=BUDGET)
    revision = _git_head()
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
        "software_hardware": _software_hardware(),
    }
    setup["setup_payload_sha256"] = sha256_json(setup)
    write_json_atomic(
        OUTPUT / "setup" / f"{int(time.time())}.{os.getpid()}.json",
        setup,
    )
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())

    completed = 0
    completed_stages: set[Stage] = set()
    for stage, reference, policy in missing:
        if stage == "actual" and "smoke" in stages and "smoke" not in completed_stages:
            audit_smoke()
            completed_stages.add("smoke")
        row_index = int(reference["row_index"])
        source = sources[row_index]
        _check_source(reference, source)
        example = _model_example(examples[row_index], model_config)
        if example.sample_id != reference["sample_id"]:
            raise ValueError(f"dataset/source mismatch: {reference['sample_id']}")
        cap = _cap(config, stage)
        path = _sample_path(stage, row_index, policy)
        try:
            if policy == LOSSLESS:
                row = _run_lossless(
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
                row = _run_candidate(
                    config,
                    accuracy,
                    model_config,
                    model,
                    tokenizer,
                    ops,
                    engine,
                    example,
                    source,
                    _mask_reference(config, reference),
                    stage=stage,
                    max_new_tokens=cap,
                    physical_gpu=physical_gpu,
                    execution_git_head=revision,
                )
            row[f"{policy}_reference_identity_gate_pass"] = bool(
                row["required_reference_exact_token_identity"]
            )
            if not bool(row["required_reference_exact_token_identity"]):
                failure_diagnostic: dict[str, object] = {
                    "schema_version": 1,
                    "state": "failed_identity_diagnostic",
                    "pilot_id": PILOT_ID,
                    "config_sha256": CONFIG_SHA256,
                    "sample_manifest_sha256": SAMPLES_SHA256,
                    "stage": stage,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": policy,
                    "generated_token_ids": row["generated_token_ids"],
                    "reference_token_ids": row["reference_token_ids"],
                    "first_token_divergence_from_required_reference": row[
                        "first_token_divergence_from_required_reference"
                    ],
                    "generated_tokens": row["generated_tokens"],
                    "correct": row["correct"],
                    "subset_trajectory_sha256": row.get("subset_trajectory_sha256"),
                    "first_route_divergence": row.get("first_route_divergence"),
                    "actual_offload_metrics": row["actual_offload_metrics"],
                    "prefill_wall_seconds_measured": row["prefill_wall_seconds_measured"],
                    "decode_wall_seconds_measured": row["decode_wall_seconds_measured"],
                    "planning_wall_seconds_measured": row["planning_wall_seconds_measured"],
                    "prefetch_wall_seconds_measured": row["prefetch_wall_seconds_measured"],
                    "production_wall_seconds_measured": row["production_wall_seconds_measured"],
                    "execution_git_head": revision,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                }
                failure_diagnostic["failure_payload_sha256"] = sha256_json(failure_diagnostic)
                diagnostic_path = _sample_path(stage, row_index, policy).with_name(
                    f"{policy}.{os.getpid()}.DIAGNOSTIC.json"
                )
                write_json_atomic(diagnostic_path, failure_diagnostic)
                if policy == LOSSLESS:
                    raise RuntimeError(
                        f"lossless offload identity failed for {reference['sample_id']}"
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
    rows = []
    for reference, policy in _work(config, samples, "smoke"):
        row = _load_checksummed(
            _sample_path("smoke", int(reference["row_index"]), policy),
            stage="smoke",
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=_cap(config, "smoke"),
        )
        if row is None:
            raise ValueError("real-offload mechanism smoke is incomplete")
        rows.append(row)
    passed = all(
        bool(row["required_reference_exact_token_identity"])
        and bool(row["actual_h2d_copy_executed"])
        and not bool(row["identity_materialized"])
        and int(cast(dict[str, Any], row["offload_engine_audit"])["slots_per_layer"]) == BUDGET
        for row in rows
    )
    if not passed:
        raise RuntimeError("real-offload mechanism smoke audit failed")
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "sample_policy_rows": len(rows),
        "all_pass": True,
        "actual_h2d_copy_rows": len(rows),
        "identity_materialized_rows": 0,
        "required_reference_exact_identity_rows": len(rows),
    }
    write_json_atomic(OUTPUT / "smoke" / "audit.json", result)
    return result


def _actual_rows(
    config: dict[str, Any],
    samples: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    cap = _cap(config, "actual")
    for reference, policy in _work(config, samples, "actual"):
        row = _load_checksummed(
            _sample_path("actual", int(reference["row_index"]), policy),
            stage="actual",
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=cap,
        )
        if row is None:
            raise ValueError("real-offload aggregate is incomplete")
        rows.append(row)
    return rows


def _policy_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    decode_forwards = sum(int(row["post_prefill_decode_forwards"]) for row in rows)
    decode_wall = sum(float(row["decode_wall_seconds_measured"]) for row in rows)
    tokens = sum(int(row["generated_tokens"]) for row in rows)
    inference_wall = sum(float(row["end_to_end_inference_wall_seconds_measured"]) for row in rows)
    offload = [cast(dict[str, Any], row["actual_offload_metrics"]) for row in rows]
    return {
        "rows": len(rows),
        "correct_rows": sum(bool(row["correct"]) for row in rows),
        "required_reference_exact_identity_rows": sum(
            bool(row["required_reference_exact_token_identity"]) for row in rows
        ),
        "recovered_identity_diagnostic_rows": sum(
            row.get("recovered_from_identity_failure_diagnostic") is not None for row in rows
        ),
        "total_generated_tokens": tokens,
        "post_prefill_decode_forwards": decode_forwards,
        "decode_wall_seconds_measured": decode_wall,
        "aggregate_post_prefill_decode_forwards_per_second_measured": (
            decode_forwards / decode_wall if decode_wall else None
        ),
        "end_to_end_inference_wall_seconds_measured": inference_wall,
        "aggregate_end_to_end_tokens_per_second_measured": (
            tokens / inference_wall if inference_wall else None
        ),
        "prefill_wall_seconds_measured": sum(
            float(row["prefill_wall_seconds_measured"]) for row in rows
        ),
        "planning_wall_seconds_measured": sum(
            float(row["planning_wall_seconds_measured"]) for row in rows
        ),
        "prefetch_wall_seconds_measured": sum(
            float(row["prefetch_wall_seconds_measured"]) for row in rows
        ),
        "production_wall_seconds_measured": sum(
            float(row["production_wall_seconds_measured"]) for row in rows
        ),
        "actual_h2d_bytes": sum(int(item["h2d_bytes"]) for item in offload),
        "cache_hits": sum(int(item["cache_hits"]) for item in offload),
        "cache_misses": sum(int(item["cache_misses"]) for item in offload),
        "transfer_batches": sum(int(item["transfer_batches"]) for item in offload),
        "cuda_event_transfer_seconds": sum(float(item["transfer_ms"]) for item in offload) / 1000,
        "cuda_event_exposed_stall_seconds": sum(float(item["exposed_stall_ms"]) for item in offload)
        / 1000,
        "peak_cuda_allocated_bytes": max(int(item["peak_allocated_bytes"]) for item in offload),
        "peak_cuda_reserved_bytes": max(int(item["peak_reserved_bytes"]) for item in offload),
        "identity_materialized_rows": sum(bool(row["identity_materialized"]) for row in rows),
        "actual_closed_loop_real_offload_rows": len(rows),
    }


def aggregate() -> dict[str, object]:
    config, samples = _protocol()
    smoke = audit_smoke()
    rows = _actual_rows(config, samples)
    grouped = {policy: [row for row in rows if row["policy"] == policy] for policy in POLICIES}
    summaries = {policy: _policy_summary(grouped[policy]) for policy in POLICIES}
    lossless_rate = cast(
        float,
        summaries[LOSSLESS]["aggregate_post_prefill_decode_forwards_per_second_measured"],
    )
    candidate_rate = cast(
        float,
        summaries[CANDIDATE]["aggregate_post_prefill_decode_forwards_per_second_measured"],
    )
    speedup = candidate_rate / lossless_rate
    lossless_h2d = int(cast(Any, summaries[LOSSLESS]["actual_h2d_bytes"]))
    candidate_h2d = int(cast(Any, summaries[CANDIDATE]["actual_h2d_bytes"]))
    h2d_reduction = 1 - candidate_h2d / lossless_h2d
    lossless_transfer = float(cast(Any, summaries[LOSSLESS]["cuda_event_transfer_seconds"]))
    candidate_transfer = float(cast(Any, summaries[CANDIDATE]["cuda_event_transfer_seconds"]))
    transfer_time_reduction = 1 - candidate_transfer / lossless_transfer
    lossless_accuracy = int(cast(Any, summaries[LOSSLESS]["correct_rows"])) / int(
        cast(Any, summaries[LOSSLESS]["rows"])
    )
    candidate_accuracy = int(cast(Any, summaries[CANDIDATE]["correct_rows"])) / int(
        cast(Any, summaries[CANDIDATE]["rows"])
    )
    per_sample = [
        {
            "sample_id": sample_id,
            "policies": {
                policy: {
                    "generated_tokens": int(row["generated_tokens"]),
                    "correct": bool(row["correct"]),
                    "reference_exact_identity": bool(
                        row["required_reference_exact_token_identity"]
                    ),
                    "first_token_divergence": row["first_token_divergence_from_required_reference"],
                    "decode_forwards_per_second_measured": float(
                        row["decode_forwards_per_second_measured"]
                    ),
                }
                for policy in POLICIES
                for row in grouped[policy]
                if row["sample_id"] == sample_id
            },
        }
        for sample_id in ("test-44", "test-632")
    ]
    lossless_identity = all(
        bool(row["required_reference_exact_token_identity"]) for row in grouped[LOSSLESS]
    )
    candidate_identity = all(
        bool(row["required_reference_exact_token_identity"]) for row in grouped[CANDIDATE]
    )
    all_identity = lossless_identity and candidate_identity
    decision = (
        "STOP_ENGINE_CORRECTNESS_FAILURE"
        if not lossless_identity
        else "STOP_PIVOT_CANDIDATE_IDENTITY_GATE_FAILURE"
        if not candidate_identity
        else "NARROW_ENGINEERING_SPEEDUP"
        if speedup > 1
        else "NARROW_ENGINEERING_NO_SPEEDUP"
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "measured_hardware_scope": "single_A100_SXM4_80GB_observed_PCIe",
        "sample_count": 2,
        "policy_rows": len(rows),
        "policy_summaries": summaries,
        "candidate_vs_lossless_decode_throughput_ratio_measured": speedup,
        "candidate_vs_lossless_actual_h2d_reduction_measured": h2d_reduction,
        "candidate_vs_lossless_cuda_event_transfer_time_reduction_measured": (
            transfer_time_reduction
        ),
        "lossless_actual_h2d_bytes": lossless_h2d,
        "candidate_actual_h2d_bytes": candidate_h2d,
        "lossless_task_accuracy_measured": lossless_accuracy,
        "candidate_task_accuracy_measured": candidate_accuracy,
        "per_sample": per_sample,
        "lossless_reference_exact_identity_gate_pass": lossless_identity,
        "candidate_reference_exact_identity_gate_pass": candidate_identity,
        "candidate_vs_lossless_decode_speedup_percent_measured": (speedup - 1) * 100,
        "all_required_reference_exact_identity": all_identity,
        "all_actual_h2d_copy": all(bool(row["actual_h2d_copy_executed"]) for row in rows),
        "identity_materialized_rows": sum(bool(row["identity_materialized"]) for row in rows),
        "measured_vs_simulated": {
            "task_accuracy": "measured_two_row_closed_loop",
            "token_identity": "measured",
            "runtime": "measured",
            "expert_h2d_bytes": "measured_from_copy_operations",
            "transfer_and_stall": "measured_by_cuda_events",
            "candidate_route_transfer_estimates_in_rows": "simulated_diagnostic_only",
        },
        "focused_decision": decision,
        "conclusion_ceiling": config["reporting"]["conclusion_ceiling"],
        "smoke_audit": smoke,
    }
    write_json_atomic(OUTPUT / "aggregate.json", result)
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "schema_version": 1,
            "actual_h2d_reduction_measured": h2d_reduction,
            "cuda_event_transfer_time_reduction_measured": transfer_time_reduction,
            "task_accuracy_measured": {LOSSLESS: lossless_accuracy, CANDIDATE: candidate_accuracy},
            "state": "complete",
            "pilot_id": PILOT_ID,
            "focused_decision": decision,
            "scope": "Qwen/GSM8K H=8 B=32 two-row real-offload engineering pilot",
            "speedup_ratio_measured": speedup,
            "all_required_reference_exact_identity": all_identity,
            "lossless_reference_exact_identity_gate_pass": lossless_identity,
            "candidate_reference_exact_identity_gate_pass": candidate_identity,
            "claim_limit": "not a full-dataset or production serving speedup claim",
        },
    )
    candidate_divergences = ", ".join(
        f"{row['sample_id']}: token {row['first_token_divergence_from_required_reference']}"
        for row in grouped[CANDIDATE]
    )
    candidate_production_misses = sum(
        int(
            cast(dict[str, Any], row["actual_offload_metrics"])["phase_metrics"]["production"][
                "cache_misses"
            ]
        )
        for row in grouped[CANDIDATE]
    )
    report = (
        "# Qwen real expert-offload speed pilot\n\n"
        f"Decision: **{decision}**. This is a two-row engineering result only.\n\n"
        "## Measured speed and transfer\n\n"
        f"- Traditional exact-top-8: {lossless_rate:.6f} post-prefill forwards/s.\n"
        f"- H=8/B=32 pseudo subset: {candidate_rate:.6f} post-prefill forwards/s.\n"
        f"- Candidate / traditional: {speedup:.6f}x "
        f"({(speedup - 1) * 100:.3f}%). No speedup was measured.\n"
        f"- Actual H2D: {lossless_h2d:,} vs {candidate_h2d:,} bytes; "
        f"candidate reduction {h2d_reduction * 100:.3f}%.\n"
        f"- CUDA-event transfer time reduction: {transfer_time_reduction * 100:.3f}%.\n"
        f"- Candidate production expert misses: {candidate_production_misses}.\n\n"
        "## Closed-loop outputs\n\n"
        f"- GSM8K accuracy: traditional {int(lossless_accuracy * 2)}/2; "
        f"candidate {int(candidate_accuracy * 2)}/2. This is measured task accuracy.\n"
        f"- Traditional frozen-vanilla exact identity: {lossless_identity} (2/2).\n"
        f"- Candidate frozen mask-only exact identity: {candidate_identity} (0/2); "
        f"first divergences were {candidate_divergences}.\n"
        f"- Actual closed-loop rows: {len(rows)}; identity-materialized rows: 0.\n\n"
        "## Physical and information boundary\n\n"
        "- Full routed-expert weights were held in pinned CPU BF16 storage; CUDA held "
        "32 expert slots per layer (25%). Every reported transfer was an actual CPU-to-CUDA "
        "copy. The candidate used its own closed-loop context and had zero production misses.\n"
        "- Runtime, outputs, H2D bytes, and CUDA-event transfer/stall are measured. "
        "Legacy route/transfer estimates stored in candidate rows are simulated diagnostics only.\n"
        "- This unoverlapped Python reference runner includes instrumentation overhead. "
        "It does not measure NVMe, NVLink, multi-GPU, transfer/compute overlap, a fused serving "
        "kernel, or full-dataset accuracy, and it cannot support a production speedup claim.\n"
    )
    (OUTPUT / "report.md").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "report_payload.json", {"markdown": report})
    return result


def finalize() -> dict[str, object]:
    aggregate()
    payload = _json(OUTPUT / "report_payload.json")
    report_path = OUTPUT / "report.md"
    report_path.write_text(str(payload["markdown"]), encoding="utf-8")
    files = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and path.name != "validation.json"
        and not path.name.endswith(".FAILED.json")
    )
    failures = sorted(path for path in OUTPUT.rglob("*.FAILED.json"))
    manifest = {
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
    checks = {
        "row_count": len(rows) == 4,
        "two_samples": len({row["sample_id"] for row in rows}) == 2,
        "both_policies": {row["policy"] for row in rows} == set(POLICIES),
        "lossless_reference_identity": all(
            bool(row["required_reference_exact_token_identity"])
            for row in rows
            if row["policy"] == LOSSLESS
        ),
        "candidate_reference_identity_audited": all(
            isinstance(row["required_reference_exact_token_identity"], bool)
            for row in rows
            if row["policy"] == CANDIDATE
        ),
        "candidate_production_zero_miss": all(
            int(
                cast(dict[str, Any], row["actual_offload_metrics"])["phase_metrics"]["production"][
                    "cache_misses"
                ]
            )
            == 0
            for row in rows
            if row["policy"] == CANDIDATE
        ),
        "all_actual_h2d": all(bool(row["actual_h2d_copy_executed"]) for row in rows),
        "no_identity_materialized": all(not bool(row["identity_materialized"]) for row in rows),
        "all_b32": all(
            int(cast(dict[str, Any], row["offload_engine_audit"])["slots_per_layer"]) == BUDGET
            for row in rows
        ),
        "all_pinned": all(
            bool(cast(dict[str, Any], row["offload_engine_audit"])["all_cpu_sources_pinned"])
            for row in rows
        ),
        "no_full_cuda_expert_parameters": all(
            bool(
                cast(dict[str, Any], row["offload_engine_audit"])[
                    "no_full_expert_parameter_on_cuda"
                ]
            )
            for row in rows
        ),
        "aggregate_complete": aggregate_row.get("state") == "complete",
        "smoke_complete": bool(smoke["all_pass"]),
        "manifest_checksum": manifest_checksum == sha256_json(manifest),
        "manifest_files": all(
            sha256_file(OUTPUT / item["path"]) == item["sha256"] for item in manifest["files"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"real-offload validation failed: {checks}")
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "checks": checks,
        "all_pass": True,
        "actual_rows": len(rows),
        "failure_markers_retained": manifest["failure_markers_retained"],
    }
    write_json_atomic(OUTPUT / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--physical-gpu", type=int, default=0)
        if command == "run":
            child.add_argument("--stage", choices=("smoke", "actual"), required=True)
    subparsers.add_parser("audit-smoke")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args()
    if args.command == "run":
        result = _run_stages((cast(Stage, args.stage),), physical_gpu=args.physical_gpu)
    elif args.command == "all":
        result = _run_stages(("smoke", "actual"), physical_gpu=args.physical_gpu)
        audit_smoke()
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
