"""Calibration-free native-Qwen prompt-route mechanism smoke."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from safetensors.torch import load_file
from torch import Tensor, nn

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import NativeRouteCaptureContext, Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_calibration_free_analysis import (
    Aggregate,
    _top_b_with_core,
)
from pseudoroute.benchmark.pseudo_embedding_content_smoke import (
    RECENT_INDEPENDENT,
    SAMPLED,
)
from pseudoroute.benchmark.pseudo_embedding_route import (
    _save_tensors_atomic,
    _source_model,
    _source_rows,
    _top_b,
    load_focused_suite_and_manifest,
)
from pseudoroute.benchmark.qwen_pseudo import _restore_rng, _rng_equal, _rng_snapshot
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
    _software_hardware,
)
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_length,
    _cache_mutation_signature,
)
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

DEFAULT_CONFIG = Path(
    "configs/analysis/pseudo_embedding_calibration_free_prompt_route_smoke_v1.yaml"
)
CONFIG_SHA256 = "6c348925820b914b1b8a021f68e121373b085e2973c19dc50030a1477bc3cf03"
ANALYSIS_ID = "pseudo_embedding_calibration_free_prompt_route_smoke_v1"
CONTROL = SAMPLED
PROMPT_PRIOR = "prompt_or_recent_route_prior"
KNOWN_HISTORY = "sampled_known_plus_prompt_or_recent_history"
EQUAL_SAMPLED = "equal_sampled_pseudo_prompt_or_recent_history"
CORE_HISTORY = "sampled_top8_core_plus_prompt_or_recent_history_fill"
EQUAL_RECENT = "equal_recent_pseudo_prompt_or_recent_history"
CANDIDATES = (PROMPT_PRIOR, KNOWN_HISTORY, EQUAL_SAMPLED, CORE_HISTORY, EQUAL_RECENT)
METHODS = (CONTROL, *CANDIDATES)


def _load_protocol(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("prompt-route protocol bytes changed")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("prompt-route protocol root must be a mapping")
    row = cast(dict[str, Any], raw)
    if (
        row.get("analysis_id") != ANALYSIS_ID
        or row.get("status") != "protocol_frozen_before_model_execution"
        or row.get("route_token_cap") != 16
        or tuple(row.get("candidates", ())) != CANDIDATES
        or row.get("artifact_root")
        != "artifacts/pseudo_embedding_calibration_free_prompt_route_smoke_v1"
    ):
        raise ValueError("prompt-route frozen scope changed")
    references = tuple((item["row_index"], item["sample_id"]) for item in row["rows"])
    if references != ((786, "test-786"), (394, "test-394")):
        raise ValueError("prompt-route sample scope changed")
    if any(
        row["calibration_free"][key] != "forbidden"
        for key in (
            "learned_parameters",
            "fitted_coefficients",
            "offline_expert_priors",
            "offline_route_transition_tables",
            "default_vector_values",
            "future_true_tokens_in_deployable_methods",
        )
    ):
        raise ValueError("prompt-route calibration-free boundary changed")
    return row


def _history_distribution(ids: Tensor, weights: Tensor, experts: int) -> Tensor:
    if ids.shape != weights.shape or ids.ndim != 2:
        raise ValueError("route history IDs and weights must be [tokens, top-k]")
    scores = torch.zeros(experts, dtype=torch.float64)
    scores.scatter_add_(0, ids.reshape(-1).long(), weights.reshape(-1).double())
    if not scores.sum():
        raise ValueError("route history has zero selected mass")
    return scores / scores.sum()


def candidate_subsets(
    sampled_probabilities: Tensor,
    recent_probabilities: Tensor,
    history_ids: Tensor,
    history_weights: Tensor,
    *,
    budget: int = 32,
    top_k: int = 8,
    experts: int = 128,
) -> dict[str, tuple[int, ...]]:
    """Construct every frozen prompt-route candidate for one routed layer."""
    expected = (8, experts)
    if (
        tuple(sampled_probabilities.shape) != expected
        or tuple(recent_probabilities.shape) != expected
    ):
        raise ValueError("prompt-route pseudo probability tensor has the wrong shape")
    history = _history_distribution(history_ids, history_weights, experts)
    sampled = sampled_probabilities.double()
    recent = recent_probabilities.double()
    return {
        PROMPT_PRIOR: _top_b(history, budget),
        KNOWN_HISTORY: _top_b(sampled[0] + 7 * history, budget),
        EQUAL_SAMPLED: _top_b(
            sampled[0] + 0.5 * sampled[1:].sum(dim=0) + 3.5 * history,
            budget,
        ),
        CORE_HISTORY: _top_b_with_core(history, sampled[0], budget, top_k),
        EQUAL_RECENT: _top_b(
            recent[0] + 0.5 * recent[1:].sum(dim=0) + 3.5 * history,
            budget,
        ),
    }


def native_weight_normalization_audit(
    weights: Tensor,
    native_dtype: torch.dtype,
) -> tuple[float, float]:
    """Return observed top-k sum error and the native floating-point epsilon bound."""
    if not native_dtype.is_floating_point:
        raise ValueError("native router weight dtype must be floating point")
    error = float((weights.float().sum(dim=-1) - 1).abs().max())
    tolerance = float(torch.finfo(native_dtype).eps)
    return error, tolerance


def _sample_paths(output: Path, row_index: int) -> tuple[Path, Path]:
    root = output / "prompt_route/samples" / f"{row_index:05d}"
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid_sample(
    output: Path,
    row_index: int,
    sample_id: str,
) -> dict[str, Any] | None:
    json_path, tensor_path = _sample_paths(output, row_index)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial prompt-route sample artifact: {json_path}")
    row = cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_id") != sample_id
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"corrupt or incompatible prompt-route sample: {json_path}")
    return row


def _capture_prompt_route(
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    model_config: Any,
    source: dict[str, Any],
    physical_gpu: int,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    rendered = str(source["rendered_prompt"])
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, rendered)
    prompt_length = int(inputs["input_ids"].shape[1])
    if prompt_length < 8:
        raise ValueError("prompt-route capture requires at least eight prompt tokens")
    device = next(model.parameters()).device
    rng_before = _rng_snapshot(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with NativeRouteCaptureContext(ops) as capture, torch.inference_mode():
            output = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
            records = capture.drain()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        rng_unchanged = _rng_equal(rng_before, _rng_snapshot(device))
    finally:
        _restore_rng(device, rng_before)
    if not rng_unchanged:
        raise RuntimeError("prompt-route capture changed generation RNG")
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError("prompt-route prefill did not return a production cache")
    signature_before = _cache_mutation_signature(cache)
    if tuple(record.layer for record in records) != tuple(range(48)):
        raise RuntimeError("prompt-route capture missed a routed layer")
    logits = torch.stack([record.natural.logits[-8:].float() for record in records], dim=1)
    ids = torch.stack([record.natural.ids[-8:] for record in records], dim=1)
    weights = torch.stack([record.natural.weights[-8:].float() for record in records], dim=1)
    if logits.shape != (8, 48, 128) or ids.shape != (8, 48, 8):
        raise RuntimeError("prompt-route capture tensor shape changed")
    weight_sum_error, weight_sum_tolerance = native_weight_normalization_audit(
        weights,
        next(model.parameters()).dtype,
    )
    if weight_sum_error > weight_sum_tolerance:
        raise RuntimeError("prompt-route selected weights are not normalized")
    signature_after = _cache_mutation_signature(cache)
    cache_unchanged = signature_before == signature_after
    if not cache_unchanged:
        raise RuntimeError("prompt-route extraction mutated the production cache")
    tensors = {
        "prompt_token_ids": inputs["input_ids"][0, -8:].detach().cpu().long(),
        "prompt_router_logits": logits,
        "prompt_router_topk_ids": ids,
        "prompt_router_topk_weights": weights,
    }
    row: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "model_revision": ("0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"),
        "precision": "bfloat16",
        "prompt_tokens": prompt_length,
        "captured_prompt_tokens": 8,
        "capture_source": "same_request_current_policy_native_prefill",
        "route_normalization": "native_topk_normalized_weights",
        "max_selected_weight_sum_error": weight_sum_error,
        "native_bfloat16_weight_sum_tolerance": weight_sum_tolerance,
        "prompt_sha256": sha256_json(rendered),
        "input_ids_sha256": sha256_json(inputs["input_ids"].detach().cpu().tolist()),
        "capture_prefill_latency_seconds_measured": elapsed,
        "capture_peak_temporary_cuda_bytes_measured": max(
            0,
            torch.cuda.max_memory_allocated(device) - allocated_before,
        ),
        "production_cache_sequence_length": _cache_length(cache),
        "production_cache_sequence_length_matches_prompt": _cache_length(cache) == prompt_length,
        "production_cache_signature_unchanged_after_extraction": cache_unchanged,
        "generation_rng_unchanged": rng_unchanged,
        "native_attention_rope_router": True,
        "layer_scoped_expert_ids": True,
        "future_generated_tokens_used": False,
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }
    return row, tensors


def _source_audit(protocol: dict[str, Any]) -> dict[str, object]:
    root = Path(protocol["source_content_smoke"]["artifact_root"])
    manifest = root / "artifact_manifest.json"
    if sha256_file(manifest) != protocol["source_content_smoke"]["artifact_manifest_sha256"]:
        raise ValueError("content-smoke artifact manifest changed")
    for reference in protocol["rows"]:
        path = root / "route/samples" / f"{int(reference['row_index']):05d}.safetensors"
        if sha256_file(path) != reference["content_tensor_sha256"]:
            raise ValueError(f"content-smoke source tensor changed: {path}")
    return {
        "state": "valid",
        "content_artifact_manifest_sha256": sha256_file(manifest),
        "content_tensors_validated": 2,
    }


def _raw_row(
    sample_id: str,
    boundary: int,
    layer: int,
    method: str,
    ids: Tensor,
    weights: Tensor,
    subset: tuple[int, ...],
    previous_subset: tuple[int, ...],
) -> tuple[dict[str, object], Aggregate]:
    aggregate = Aggregate()
    aggregate.update(ids, weights, subset, previous_subset)
    return (
        {
            "sample_id": sample_id,
            "boundary": boundary,
            "layer": layer,
            "method": method,
            "subset": list(subset),
            **aggregate.as_dict(),
        },
        aggregate,
    )


def _aggregate(
    protocol: dict[str, Any],
    output: Path,
) -> tuple[
    dict[str, dict[str, int | float]],
    dict[str, object],
    list[dict[str, object]],
]:
    content_root = Path(protocol["source_content_smoke"]["artifact_root"])
    totals = {key: Aggregate() for key in METHODS}
    by_boundary = {(key, boundary): Aggregate() for key in METHODS for boundary in (0, 8)}
    by_sample = {
        (key, str(reference["sample_id"])): Aggregate()
        for reference in protocol["rows"]
        for key in METHODS
    }
    by_layer_block = {(key, block): Aggregate() for key in METHODS for block in range(6)}
    by_anchor = {(key, anchor): Aggregate() for key in METHODS for anchor in range(1, 9)}
    raw_rows: list[dict[str, object]] = []
    for reference in protocol["rows"]:
        row_index = int(reference["row_index"])
        sample_id = str(reference["sample_id"])
        content = load_file(str(content_root / "route/samples" / f"{row_index:05d}.safetensors"))
        prompt = load_file(str(_sample_paths(output, row_index)[1]))
        generated_ids = content["router_topk_ids"]
        generated_weights = content["router_topk_weights"]
        sampled = content[f"{SAMPLED}__pre_topk_probabilities"]
        recent = content[f"{RECENT_INDEPENDENT}__pre_topk_probabilities"]
        control_subsets = content[f"{CONTROL}__subsets"]
        resident: dict[str, dict[int, tuple[int, ...]]] = {
            key: {layer: () for layer in range(48)} for key in METHODS
        }
        for boundary_index, boundary in enumerate((0, 8)):
            history_ids = (
                prompt["prompt_router_topk_ids"]
                if boundary == 0
                else generated_ids[boundary - 8 : boundary]
            )
            history_weights = (
                prompt["prompt_router_topk_weights"]
                if boundary == 0
                else generated_weights[boundary - 8 : boundary]
            )
            end = boundary + 8
            for layer in range(48):
                candidate = candidate_subsets(
                    sampled[boundary_index, layer],
                    recent[boundary_index, layer],
                    history_ids[:, layer],
                    history_weights[:, layer],
                )
                subsets = {
                    CONTROL: tuple(
                        int(value) for value in control_subsets[boundary_index, layer].tolist()
                    ),
                    **candidate,
                }
                for method in METHODS:
                    row, aggregate = _raw_row(
                        sample_id,
                        boundary,
                        layer,
                        method,
                        generated_ids[boundary:end, layer],
                        generated_weights[boundary:end, layer],
                        subsets[method],
                        resident[method][layer],
                    )
                    raw_rows.append(row)
                    totals[method].merge(aggregate)
                    by_boundary[(method, boundary)].merge(aggregate)
                    by_sample[(method, sample_id)].merge(aggregate)
                    by_layer_block[(method, layer // 8)].merge(aggregate)
                    for anchor in range(8):
                        anchor_aggregate = Aggregate()
                        anchor_aggregate.update(
                            generated_ids[boundary + anchor, layer],
                            generated_weights[boundary + anchor, layer],
                            subsets[method],
                            (),
                        )
                        by_anchor[(method, anchor + 1)].merge(anchor_aggregate)
                    resident[method][layer] = subsets[method]
    aggregates = {method: totals[method].as_dict() for method in METHODS}
    strata: dict[str, object] = {
        "by_boundary": {
            method: {
                str(boundary): by_boundary[(method, boundary)].as_dict() for boundary in (0, 8)
            }
            for method in METHODS
        },
        "by_sample": {
            method: {
                str(reference["sample_id"]): by_sample[
                    (method, str(reference["sample_id"]))
                ].as_dict()
                for reference in protocol["rows"]
            }
            for method in METHODS
        },
        "by_eight_layer_block": {
            method: {str(block): by_layer_block[(method, block)].as_dict() for block in range(6)}
            for method in METHODS
        },
        "by_anchor": {
            method: {str(anchor): by_anchor[(method, anchor)].as_dict() for anchor in range(1, 9)}
            for method in METHODS
        },
    }
    return aggregates, strata, raw_rows


def _decision(
    protocol: dict[str, Any],
    aggregates: dict[str, dict[str, int | float]],
    audit_pass: bool,
) -> dict[str, object]:
    control = aggregates[CONTROL]
    rows = []
    thresholds = protocol["progress_reference"]
    for key in CANDIDATES:
        candidate = aggregates[key]
        hit_gain = float(candidate["mean_route_hit"]) - float(control["mean_route_hit"])
        mass_gain = float(candidate["mean_selected_mass"]) - float(control["mean_selected_mass"])
        checks = {
            "route_hit_improvement_pass": hit_gain >= thresholds["minimum_route_hit_improvement"],
            "selected_mass_improvement_pass": mass_gain
            >= thresholds["minimum_selected_mass_improvement"],
            "simulated_transfer_reduction_pass": float(candidate["estimated_transfer_reduction"])
            >= thresholds["minimum_simulated_transfer_reduction"],
            "cache_rng_information_audit_pass": audit_pass,
        }
        rows.append(
            {
                "candidate": key,
                "metrics": candidate,
                "route_hit_gain_over_control": hit_gain,
                "selected_mass_gain_over_control": mass_gain,
                "checks": checks,
                "eligible_for_new_frozen_development": all(checks.values()),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(cast(dict[str, Any], row["metrics"])["mean_selected_mass"]),
            -float(cast(dict[str, Any], row["metrics"])["mean_route_hit"]),
            str(row["candidate"]),
        ),
    )
    eligible = [row for row in ranked if row["eligible_for_new_frozen_development"]]
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": ("CANDIDATE_FOR_NEW_FROZEN_DEVELOPMENT" if eligible else "STOP/PIVOT"),
        "best_observed_candidate": ranked[0]["candidate"],
        "qualifying_candidate": eligible[0]["candidate"] if eligible else None,
        "candidates": rows,
        "accuracy_used": False,
        "result_role": "two_row_mechanism_hypothesis_only",
        "terminal_v1_decision_unchanged": True,
    }


def _git_revision() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_head": head, "worktree_clean": not dirty, "pid": os.getpid(), "ppid": os.getppid()}


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _report(
    aggregates: dict[str, dict[str, int | float]],
    decision: dict[str, object],
) -> str:
    control = aggregates[CONTROL]
    lines = [
        f"# {ANALYSIS_ID}",
        "",
        f"Mechanism decision: **{decision['decision']}**. The focused v1 terminal decision "
        "remains **STOP/PIVOT**.",
        "",
        "This is a two-row native-Qwen prompt-prefill route smoke plus open-loop saved-token "
        "scoring. It is not held-out evidence, task accuracy, hard closed-loop generation, or "
        "a runtime speedup claim.",
        "",
        "## Results",
        "",
        f"- `{CONTROL}` control: hit {float(control['mean_route_hit']):.6f}, mass "
        f"{float(control['mean_selected_mass']):.6f}, simulated transfer reduction "
        f"{float(control['estimated_transfer_reduction']):.6f}.",
    ]
    decisions = {
        str(row["candidate"]): row for row in cast(list[dict[str, Any]], decision["candidates"])
    }
    for key in CANDIDATES:
        row = aggregates[key]
        scored = decisions[key]
        lines.append(
            f"- `{key}`: hit {float(row['mean_route_hit']):.6f} "
            f"({float(scored['route_hit_gain_over_control']):+.6f}), mass "
            f"{float(row['mean_selected_mass']):.6f} "
            f"({float(scored['selected_mass_gain_over_control']):+.6f}), simulated transfer "
            f"reduction {float(row['estimated_transfer_reduction']):.6f}."
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "Every deployable score uses only the sampled next token, same-request native prompt "
            "routes, current-policy realized generated routes, and saved zero-contribution pseudo "
            "probabilities. No training, fitted coefficient, offline expert prior, transition "
            "table, "
            "default-vector value, future token, answer, correctness, or task accuracy is used. "
            "Prompt prefill capture latency is measured capture-inclusive runtime; it is not an "
            "isolated overhead measurement. Transfer is simulated.",
            "",
        )
    )
    return "\n".join(lines)


def _artifact_manifest(output: Path) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded:
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(output / "artifact_manifest.json", result)
    return result


def _validate_manifest(output: Path) -> dict[str, Any]:
    manifest = cast(
        dict[str, Any],
        json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8")),
    )
    if manifest.get("analysis_config_sha256") != CONFIG_SHA256:
        raise ValueError("prompt-route artifact manifest config changed")
    for row in manifest["artifacts"]:
        path = output / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"prompt-route artifact checksum mismatch: {path}")
    return manifest


def validate(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    protocol = _load_protocol(config_path)
    _source_audit(protocol)
    output = Path(protocol["artifact_root"])
    status = cast(
        dict[str, Any],
        json.loads((output / "pipeline_status.json").read_text(encoding="utf-8")),
    )
    if (
        status.get("state") != "complete"
        or status.get("stage") != "report_v1"
        or status.get("analysis_config_sha256") != CONFIG_SHA256
    ):
        raise ValueError("prompt-route pipeline is incomplete")
    for reference in protocol["rows"]:
        if (
            _valid_sample(
                output,
                int(reference["row_index"]),
                str(reference["sample_id"]),
            )
            is None
        ):
            raise ValueError("prompt-route sample is missing")
    manifest = _validate_manifest(output)
    return {
        "state": "valid",
        "decision": status["decision"],
        "samples": 2,
        "artifacts": manifest["artifact_count"],
    }


def run(config_path: Path, physical_gpu: int) -> dict[str, object]:
    started = time.perf_counter()
    protocol = _load_protocol(config_path)
    source_audit = _source_audit(protocol)
    output = Path(protocol["artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    suite, _ = load_focused_suite_and_manifest(
        Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml")
    )
    sources = _source_rows(suite)
    missing = []
    completed: list[dict[str, Any]] = []
    for reference in protocol["rows"]:
        existing = _valid_sample(
            output,
            int(reference["row_index"]),
            str(reference["sample_id"]),
        )
        if existing is None:
            missing.append(reference)
        else:
            completed.append(existing)
    if missing:
        accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
        model_config = _source_model(accuracy, physical_gpu)
        seed_everything(suite.decode.seed)
        torch.cuda.set_device(torch.device(model_config.device))
        model, tokenizer = _load_model(model_config, accuracy)
        ops = Qwen3MoePrefetchOps(model)
        if (ops.num_layers, ops.num_experts, ops.top_k) != (48, 128, 8):
            raise ValueError("runtime Qwen routed-layer facts changed")
        for reference in missing:
            row_index = int(reference["row_index"])
            source = sources[row_index]
            if source["sample_id"] != reference["sample_id"]:
                raise ValueError("prompt-route source row identity changed")
            json_path, tensor_path = _sample_paths(output, row_index)
            try:
                row, tensors = _capture_prompt_route(
                    model,
                    tokenizer,
                    ops,
                    model_config,
                    source,
                    physical_gpu,
                )
            except Exception as error:
                failure = json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")
                write_json_atomic(
                    failure,
                    {
                        "schema_version": 1,
                        "state": "failed",
                        "analysis_id": ANALYSIS_ID,
                        "analysis_config_sha256": CONFIG_SHA256,
                        "row_index": row_index,
                        "sample_id": reference["sample_id"],
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
            completed.append(cast(dict[str, Any], row))
            print(
                json.dumps(
                    {
                        "stage": "prompt_route_capture",
                        "sample_id": row["sample_id"],
                        "prompt_tokens": row["prompt_tokens"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del ops, model
        torch.cuda.empty_cache()
    completed.sort(key=lambda row: int(row["row_index"]))
    audit_pass = all(
        bool(row["production_cache_sequence_length_matches_prompt"])
        and bool(row["production_cache_signature_unchanged_after_extraction"])
        and bool(row["generation_rng_unchanged"])
        and not bool(row["future_generated_tokens_used"])
        for row in completed
    )
    if not audit_pass:
        raise RuntimeError("prompt-route cache/RNG/information audit failed")
    aggregates, strata, raw_rows = _aggregate(protocol, output)
    decision = _decision(protocol, aggregates, audit_pass)
    write_json_atomic(output / "resolved_config.json", protocol)
    write_json_atomic(output / "resolved_environment.json", _software_hardware())
    write_json_atomic(output / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(output / "source_audit.json", source_audit)
    write_json_atomic(output / "aggregates.json", aggregates)
    write_json_atomic(output / "stratified_metrics.json", strata)
    write_json_atomic(output / "decision.json", decision)
    write_json_atomic(
        output / "capture_costs.json",
        {
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "capture_prefill_latency_seconds_measured": row[
                        "capture_prefill_latency_seconds_measured"
                    ],
                    "capture_peak_temporary_cuda_bytes_measured": row[
                        "capture_peak_temporary_cuda_bytes_measured"
                    ],
                }
                for row in completed
            ],
            "runtime_definition": "capture_inclusive_native_prefill_not_isolated_hook_overhead",
        },
    )
    write_json_atomic(
        output / "cache_rng_information_audit.json",
        {
            "state": "pass" if audit_pass else "fail",
            "samples": completed,
            "prompt_routes_same_request_only": True,
            "future_token_leakage": False,
        },
    )
    raw = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in raw_rows
    ).encode()
    _write_bytes_atomic(output / "raw_metrics.jsonl.gz", gzip.compress(raw, mtime=0))
    _write_bytes_atomic(output / "report.md", _report(aggregates, decision).encode())
    failed = sorted(str(path.relative_to(output)) for path in output.rglob("*.FAILED.json"))
    write_json_atomic(
        output / "resume_audit.json",
        {
            "schema_version": 1,
            "state": "complete",
            "samples_expected": 2,
            "samples_validated": len(completed),
            "atomic_json_tensor_pairs": True,
            "checksum_resume": True,
            "failed_markers_preserved": failed,
        },
    )
    write_json_atomic(
        output / "provenance.json",
        {
            "schema_version": 1,
            "analysis_id": ANALYSIS_ID,
            "analysis_config_sha256": CONFIG_SHA256,
            "source_samples": ["test-786", "test-394"],
            "model": suite.model.model_id,
            "model_revision": suite.model.revision,
            "precision": suite.model.precision,
            "model_loaded_for_prompt_route_generation": True,
            "gpu_used_for_prompt_route_generation": True,
            "model_loaded_this_invocation": bool(missing),
            "gpu_used_this_invocation": bool(missing),
            "network_downloads": False,
            "task_accuracy_measured": False,
            "actual_closed_loop_generation": False,
            "future_true_tokens_used_by_candidates": False,
            "default_vector_values_loaded_or_used": False,
            "offline_calibration_statistics_used": False,
            "learned_or_fitted_parameters": False,
            "route_scoring": "saved_v17_teacher_forced_open_loop",
            "transfer_kind": "simulated",
            "analysis_elapsed_seconds_measured": time.perf_counter() - started,
        },
    )
    manifest = _artifact_manifest(output)
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": decision["decision"],
        "best_observed_candidate": decision["best_observed_candidate"],
        "qualifying_candidate": decision["qualifying_candidate"],
        "samples": len(completed),
        "artifact_count": manifest["artifact_count"],
    }
    write_json_atomic(output / "pipeline_status.json", status)
    validate(config_path)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.config, args.physical_gpu) if args.command == "run" else validate(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
