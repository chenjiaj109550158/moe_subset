"""Four-row calibration-free prompt-route development evaluation."""

from __future__ import annotations

import argparse
import csv
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
from torch import Tensor

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_calibration_free_analysis import Aggregate
from pseudoroute.benchmark.pseudo_embedding_prompt_route_smoke import (
    CANDIDATES,
    CONTROL,
    _capture_prompt_route,
    candidate_subsets,
)
from pseudoroute.benchmark.pseudo_embedding_prompt_route_smoke import (
    CONFIG_SHA256 as SMOKE_CONFIG_SHA256,
)
from pseudoroute.benchmark.pseudo_embedding_route import (
    _save_tensors_atomic,
    _source_model,
    _source_rows,
    load_focused_suite_and_manifest,
)
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

DEFAULT_CONFIG = Path(
    "configs/analysis/pseudo_embedding_calibration_free_prompt_route_development_v1.yaml"
)
CONFIG_SHA256 = "f1384397deba1e2cdb702dd4878af4092eaa64cd7625617900a22802d4813a25"
ANALYSIS_ID = "pseudo_embedding_calibration_free_prompt_route_development_v1"
SOURCE_ZERO = "sampled_next_independent_zero"
METHODS = (CONTROL, *CANDIDATES[:-1])


def _load_protocol(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("prompt-route development protocol bytes changed")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("prompt-route development protocol root must be a mapping")
    row = cast(dict[str, Any], raw)
    if (
        row.get("analysis_id") != ANALYSIS_ID
        or row.get("status") != "protocol_frozen_before_model_execution"
        or row.get("route_token_cap") != 128
        or tuple(row.get("candidates", ())) != CANDIDATES[:-1]
        or row.get("artifact_root")
        != "artifacts/pseudo_embedding_calibration_free_prompt_route_development_v1"
    ):
        raise ValueError("prompt-route development frozen scope changed")
    references = tuple((item["row_index"], item["sample_id"]) for item in row["rows"])
    if references != (
        (0, "test-0"),
        (439, "test-439"),
        (879, "test-879"),
        (1318, "test-1318"),
    ):
        raise ValueError("prompt-route development sample scope changed")
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
        raise ValueError("prompt-route development calibration-free boundary changed")
    return row


def _source_audit(protocol: dict[str, Any]) -> dict[str, object]:
    mechanism_root = Path(protocol["source_mechanism"]["artifact_root"])
    mechanism_manifest = mechanism_root / "artifact_manifest.json"
    if sha256_file(mechanism_manifest) != protocol["source_mechanism"]["artifact_manifest_sha256"]:
        raise ValueError("prompt-route mechanism artifact manifest changed")
    development_root = Path(protocol["source_development"]["artifact_root"])
    development_manifest = development_root / "artifact_manifest.json"
    if (
        sha256_file(development_manifest)
        != protocol["source_development"]["artifact_manifest_sha256"]
    ):
        raise ValueError("focused development artifact manifest changed")
    for reference in protocol["rows"]:
        path = (
            development_root
            / "route/development/samples"
            / f"{int(reference['row_index']):05d}.safetensors"
        )
        if sha256_file(path) != reference["development_tensor_sha256"]:
            raise ValueError(f"focused development tensor changed: {path}")
    return {
        "state": "valid",
        "mechanism_manifest_sha256": sha256_file(mechanism_manifest),
        "development_manifest_sha256": sha256_file(development_manifest),
        "development_tensors_validated": 4,
    }


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
        raise ValueError(f"partial development prompt-route sample: {json_path}")
    row = cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_id") != sample_id
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"corrupt development prompt-route sample: {json_path}")
    return row


def _single_aggregate(
    ids: Tensor,
    weights: Tensor,
    subset: tuple[int, ...],
    previous_subset: tuple[int, ...],
) -> Aggregate:
    result = Aggregate()
    result.update(ids, weights, subset, previous_subset)
    return result


def realized_window_end(boundary: int, token_count: int, horizon: int = 8) -> int:
    if boundary < 0 or boundary >= token_count:
        raise ValueError("route boundary is outside the realized token sequence")
    return min(boundary + horizon, token_count)


def _aggregate(
    protocol: dict[str, Any],
    output: Path,
) -> tuple[
    dict[str, dict[str, int | float]],
    dict[str, object],
    dict[str, dict[str, dict[str, object]]],
    list[dict[str, object]],
]:
    source_root = Path(protocol["source_development"]["artifact_root"])
    totals = {method: Aggregate() for method in METHODS}
    by_boundary: dict[tuple[str, int], Aggregate] = {}
    by_sample = {
        (method, str(reference["sample_id"])): Aggregate()
        for method in METHODS
        for reference in protocol["rows"]
    }
    by_block = {(method, block): Aggregate() for method in METHODS for block in range(6)}
    by_anchor = {(method, anchor): Aggregate() for method in METHODS for anchor in range(1, 9)}
    raw_rows: list[dict[str, object]] = []
    for reference in protocol["rows"]:
        row_index = int(reference["row_index"])
        sample_id = str(reference["sample_id"])
        source = load_file(
            str(source_root / "route/development/samples" / f"{row_index:05d}.safetensors")
        )
        prompt = load_file(str(_sample_paths(output, row_index)[1]))
        ids = source["router_topk_ids"]
        weights = source["router_topk_weights"]
        sampled = source[f"{SOURCE_ZERO}__pre_topk_probabilities"]
        control_subsets = source[f"{SOURCE_ZERO}__subsets"]
        boundaries = tuple(int(value) for value in source["boundaries"].tolist())
        if boundaries != tuple(range(0, 128, 8)):
            raise ValueError("focused development boundary grid changed")
        resident: dict[str, dict[int, tuple[int, ...]]] = {
            method: {layer: () for layer in range(48)} for method in METHODS
        }
        for boundary_index, boundary in enumerate(boundaries):
            end = realized_window_end(boundary, ids.shape[0])
            history_ids = (
                prompt["prompt_router_topk_ids"] if boundary == 0 else ids[boundary - 8 : boundary]
            )
            history_weights = (
                prompt["prompt_router_topk_weights"]
                if boundary == 0
                else weights[boundary - 8 : boundary]
            )
            for layer in range(48):
                candidate = candidate_subsets(
                    sampled[boundary_index, layer],
                    sampled[boundary_index, layer],
                    history_ids[:, layer],
                    history_weights[:, layer],
                )
                subsets = {
                    CONTROL: tuple(
                        int(value) for value in control_subsets[boundary_index, layer].tolist()
                    ),
                    **{key: candidate[key] for key in CANDIDATES[:-1]},
                }
                for method in METHODS:
                    aggregate = _single_aggregate(
                        ids[boundary:end, layer],
                        weights[boundary:end, layer],
                        subsets[method],
                        resident[method][layer],
                    )
                    totals[method].merge(aggregate)
                    by_boundary.setdefault((method, boundary), Aggregate()).merge(aggregate)
                    by_sample[(method, sample_id)].merge(aggregate)
                    by_block[(method, layer // 8)].merge(aggregate)
                    for anchor in range(end - boundary):
                        by_anchor[(method, anchor + 1)].merge(
                            _single_aggregate(
                                ids[boundary + anchor, layer],
                                weights[boundary + anchor, layer],
                                subsets[method],
                                (),
                            )
                        )
                    raw_rows.append(
                        {
                            "sample_id": sample_id,
                            "boundary": boundary,
                            "layer": layer,
                            "method": method,
                            "subset": list(subsets[method]),
                            **aggregate.as_dict(),
                        }
                    )
                    resident[method][layer] = subsets[method]
    aggregates = {method: totals[method].as_dict() for method in METHODS}
    strata: dict[str, object] = {
        "by_boundary": {
            method: {
                str(boundary): by_boundary[(method, boundary)].as_dict()
                for boundary in range(0, 128, 8)
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
            method: {str(block): by_block[(method, block)].as_dict() for block in range(6)}
            for method in METHODS
        },
        "by_anchor": {
            method: {str(anchor): by_anchor[(method, anchor)].as_dict() for anchor in range(1, 9)}
            for method in METHODS
        },
    }
    source_by_sample = _source_references_by_sample(protocol)
    leave_one_out: dict[str, dict[str, dict[str, object]]] = {}
    for method in CANDIDATES[:-1]:
        leave_one_out[method] = {}
        for excluded in protocol["rows"]:
            candidate_total = Aggregate()
            previous = _empty_metric_total()
            for included in protocol["rows"]:
                sample_id = str(included["sample_id"])
                if sample_id == excluded["sample_id"]:
                    continue
                candidate_total.merge(by_sample[(method, sample_id)])
                _merge_metric_total(
                    previous, source_by_sample["previous_route_commitment"][sample_id]
                )
            candidate_row = candidate_total.as_dict()
            previous_row = _metric_total_as_dict(previous)
            leave_one_out[method][str(excluded["sample_id"])] = {
                "candidate": candidate_row,
                "previous_route": previous_row,
                "route_hit_improvement": float(candidate_row["mean_route_hit"])
                - float(previous_row["mean_route_hit"]),
                "selected_mass_improvement": float(candidate_row["mean_selected_mass"])
                - float(previous_row["mean_selected_mass"]),
            }
    return aggregates, strata, leave_one_out, raw_rows


def _empty_metric_total() -> dict[str, float | int]:
    return {
        "route_hits": 0,
        "route_slots": 0,
        "selected_mass_hit": 0.0,
        "selected_mass_total": 0.0,
        "fallback_loads": 0,
        "prefetch_loads": 0,
    }


def _merge_metric_total(
    target: dict[str, float | int],
    source: dict[str, float | int],
) -> None:
    for key in target:
        target[key] = target[key] + source[key]


def _metric_total_as_dict(total: dict[str, float | int]) -> dict[str, float | int]:
    slots = int(total["route_slots"])
    mass_total = float(total["selected_mass_total"])
    return {
        **total,
        "mean_route_hit": int(total["route_hits"]) / slots,
        "mean_selected_mass": float(total["selected_mass_hit"]) / mass_total,
        "estimated_transfer_reduction": 1
        - (int(total["fallback_loads"]) + int(total["prefetch_loads"])) / slots,
    }


def _source_references_by_sample(
    protocol: dict[str, Any],
) -> dict[str, dict[str, dict[str, float | int]]]:
    path = (
        Path(protocol["source_development"]["artifact_root"])
        / "route/development/raw_metrics.csv.gz"
    )
    result = {
        method: {
            str(reference["sample_id"]): _empty_metric_total() for reference in protocol["rows"]
        }
        for method in ("previous_route_commitment", "static_frequency")
    }
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            method = str(row["method"])
            if method not in result:
                continue
            target = result[method][str(row["sample_id"])]
            for key in ("route_hits", "route_slots", "fallback_loads", "prefetch_loads"):
                target[key] = int(target[key]) + int(row[key])
            for key in ("selected_mass_hit", "selected_mass_total"):
                target[key] = float(target[key]) + float(row[key])
    return result


def _decision(
    protocol: dict[str, Any],
    aggregates: dict[str, dict[str, int | float]],
    audit_pass: bool,
) -> dict[str, object]:
    source = cast(
        dict[str, Any],
        json.loads(
            (
                Path(protocol["source_development"]["artifact_root"])
                / "route/development/aggregates.json"
            ).read_text(encoding="utf-8")
        ),
    )
    previous = cast(dict[str, Any], source["previous_route_commitment"])
    static = cast(dict[str, Any], source["static_frequency"])
    gate = protocol["progress_gate"]
    rows = []
    for method in CANDIDATES[:-1]:
        candidate = aggregates[method]
        hit_gain = float(candidate["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_gain = float(candidate["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        checks = {
            "route_hit_improvement_pass": hit_gain >= gate["minimum_route_hit_improvement"],
            "selected_mass_improvement_pass": mass_gain
            >= gate["minimum_selected_mass_improvement"],
            "not_worse_than_static_frequency_pass": (
                float(candidate["mean_route_hit"]) >= float(static["mean_route_hit"])
                and float(candidate["mean_selected_mass"]) >= float(static["mean_selected_mass"])
            ),
            "simulated_transfer_reduction_pass": float(candidate["estimated_transfer_reduction"])
            >= gate["minimum_simulated_transfer_reduction"],
            "cache_rng_information_audit_pass": audit_pass,
        }
        rows.append(
            {
                "candidate": method,
                "metrics": candidate,
                "route_hit_improvement_over_previous": hit_gain,
                "selected_mass_improvement_over_previous": mass_gain,
                "checks": checks,
                "eligible_for_held_out_protocol": all(checks.values()),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(cast(dict[str, Any], row["metrics"])["mean_selected_mass"]),
            -float(cast(dict[str, Any], row["metrics"])["mean_route_hit"]),
            -float(cast(dict[str, Any], row["metrics"])["estimated_transfer_reduction"]),
            str(row["candidate"]),
        ),
    )
    eligible = [row for row in ranked if row["eligible_for_held_out_protocol"]]
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": "CANDIDATE_FOR_NEW_FROZEN_HELD_OUT_ROUTE_PROTOCOL"
        if eligible
        else "STOP/PIVOT",
        "best_observed_candidate": ranked[0]["candidate"],
        "qualifying_candidate": eligible[0]["candidate"] if eligible else None,
        "candidates": rows,
        "references": {
            "previous_route_commitment": previous,
            "static_frequency": static,
        },
        "accuracy_used": False,
        "terminal_v1_decision_unchanged": True,
        "result_role": "four_row_development_route_evidence",
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
    leave_one_out: dict[str, dict[str, dict[str, object]]],
) -> str:
    references = cast(dict[str, dict[str, Any]], decision["references"])
    previous = references["previous_route_commitment"]
    lines = [
        f"# {ANALYSIS_ID}",
        "",
        f"Development decision: **{decision['decision']}**. The focused v1 terminal decision "
        "remains **STOP/PIVOT**.",
        "",
        "This is four-row open-loop route evidence with same-request native prompt route capture. "
        "It is not held-out evidence, task accuracy, hard closed-loop generation, or a "
        "speedup claim.",
        "",
        f"Frozen previous-route reference: hit {float(previous['mean_route_hit']):.6f}, mass "
        f"{float(previous['mean_selected_mass']):.6f}, simulated transfer reduction "
        f"{float(previous['estimated_transfer_reduction']):.6f}.",
        "",
    ]
    scored = {
        str(row["candidate"]): row for row in cast(list[dict[str, Any]], decision["candidates"])
    }
    for method in CANDIDATES[:-1]:
        row = aggregates[method]
        gate = scored[method]
        loo = leave_one_out[method].values()
        hit = [float(cast(Any, item["route_hit_improvement"])) for item in loo]
        mass = [float(cast(Any, item["selected_mass_improvement"])) for item in loo]
        lines.append(
            f"- `{method}`: hit {float(row['mean_route_hit']):.6f} "
            f"({float(gate['route_hit_improvement_over_previous']):+.6f}), mass "
            f"{float(row['mean_selected_mass']):.6f} "
            f"({float(gate['selected_mass_improvement_over_previous']):+.6f}), transfer "
            f"{float(row['estimated_transfer_reduction']):.6f}; LOO gains hit "
            f"[{min(hit):+.6f}, {max(hit):+.6f}], mass [{min(mass):+.6f}, {max(mass):+.6f}]."
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            "Candidates use only the sampled next token, current-request prompt routes, "
            "current-policy preceding generated routes, and zero-contribution pseudo "
            "probabilities. No learned/fitted value, offline prior, default vector, future "
            "token, answer, correctness, or accuracy is used. Transfer is simulated and "
            "prefill capture latency is capture-inclusive, not isolated "
            "hook overhead. Passing only justifies a separately frozen held-out route protocol.",
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
        raise ValueError("development artifact manifest config changed")
    for row in manifest["artifacts"]:
        path = output / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"development artifact checksum mismatch: {path}")
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
        raise ValueError("prompt-route development pipeline is incomplete")
    for reference in protocol["rows"]:
        if (
            _valid_sample(
                output,
                int(reference["row_index"]),
                str(reference["sample_id"]),
            )
            is None
        ):
            raise ValueError("development prompt-route sample is missing")
    manifest = _validate_manifest(output)
    return {
        "state": "valid",
        "decision": status["decision"],
        "samples": 4,
        "artifacts": manifest["artifact_count"],
    }


def run(config_path: Path, physical_gpu: int) -> dict[str, object]:
    started = time.perf_counter()
    protocol = _load_protocol(config_path)
    source_audit = _source_audit(protocol)
    output = Path(protocol["artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    suite, _ = load_focused_suite_and_manifest(Path(protocol["source_development"]["suite_config"]))
    if suite.fingerprint() != protocol["source_development"]["suite_fingerprint"]:
        raise ValueError("focused suite fingerprint changed")
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
                raise ValueError("development source row identity changed")
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
            row.update(
                {
                    "analysis_id": ANALYSIS_ID,
                    "analysis_config_sha256": CONFIG_SHA256,
                    "source_mechanism_config_sha256": SMOKE_CONFIG_SHA256,
                }
            )
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
                        "stage": "prompt_route_development_capture",
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
        raise RuntimeError("development cache/RNG/information audit failed")
    aggregates, strata, leave_one_out, raw_rows = _aggregate(protocol, output)
    decision = _decision(protocol, aggregates, audit_pass)
    write_json_atomic(output / "resolved_config.json", protocol)
    write_json_atomic(output / "resolved_environment.json", _software_hardware())
    write_json_atomic(output / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(output / "source_audit.json", source_audit)
    write_json_atomic(output / "aggregates.json", aggregates)
    write_json_atomic(output / "stratified_metrics.json", strata)
    write_json_atomic(output / "leave_one_out.json", leave_one_out)
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
            "same_request_prompt_routes_only": True,
            "future_token_leakage": False,
        },
    )
    raw = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in raw_rows
    ).encode()
    _write_bytes_atomic(output / "raw_metrics.jsonl.gz", gzip.compress(raw, mtime=0))
    _write_bytes_atomic(output / "report.md", _report(aggregates, decision, leave_one_out).encode())
    failed = sorted(str(path.relative_to(output)) for path in output.rglob("*.FAILED.json"))
    write_json_atomic(
        output / "resume_audit.json",
        {
            "schema_version": 1,
            "state": "complete",
            "samples_expected": 4,
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
            "source_samples": ["test-0", "test-439", "test-879", "test-1318"],
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
            "route_scoring": "checksum_locked_authoritative_development_tensors_open_loop",
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
