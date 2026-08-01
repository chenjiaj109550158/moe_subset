"""Disjoint held-out route evaluation for calibration-free prompt routing."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    Qwen3MoePrefetchOps,
    physical_expert_bytes,
)
from pseudoroute.benchmark.pseudo_embedding_content_smoke import SAMPLED, _aggregate_metrics
from pseudoroute.benchmark.pseudo_embedding_prompt_route_smoke import (
    EQUAL_SAMPLED,
    candidate_subsets,
)
from pseudoroute.benchmark.pseudo_embedding_route import (
    _route_metrics,
    _save_tensors_atomic,
    _source_model,
    _source_rows,
    _top_b,
    load_focused_suite_and_manifest,
    run_route_sample,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

DEFAULT_CONFIG = Path(
    "configs/analysis/pseudo_embedding_calibration_free_prompt_route_held_out_v1.yaml"
)
CONFIG_SHA256 = "1f6f5ab8372aa6f63796cfda49601745bf2736490ee21c8807fd9657b9a996ef"
ANALYSIS_ID = "pseudo_embedding_calibration_free_prompt_route_held_out_v1"
CANDIDATE = EQUAL_SAMPLED
ORACLE = "hard_oracle_commitment"
PREVIOUS = "previous_route_commitment"
STATIC = "static_frequency"
CONTROL = SAMPLED
METHODS = (ORACLE, PREVIOUS, STATIC, CONTROL, CANDIDATE)


def _load_protocol(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("held-out protocol bytes changed")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("held-out protocol root must be a mapping")
    row = cast(dict[str, Any], raw)
    if (
        row.get("analysis_id") != ANALYSIS_ID
        or row.get("status") != "protocol_frozen_before_model_execution"
        or row.get("route_token_cap") != 128
        or row.get("selected_candidate", {}).get("key") != CANDIDATE
        or row.get("artifact_root")
        != "artifacts/pseudo_embedding_calibration_free_prompt_route_held_out_v1"
    ):
        raise ValueError("held-out frozen scope changed")
    if any(
        row["calibration_free"][key] != "forbidden"
        for key in (
            "learned_parameters",
            "fitted_coefficients",
            "offline_expert_priors_in_candidate",
            "offline_route_transition_tables",
            "default_vector_values",
            "future_true_tokens_in_candidate",
        )
    ):
        raise ValueError("held-out calibration-free boundary changed")
    manifest = json.loads(Path(row["source_suite"]["sample_manifest"]).read_text())
    if row["rows"] != manifest["partitions"]["held_out_route"]["rows"]:
        raise ValueError("held-out deterministic row allocation changed")
    return row


def _source_audit(protocol: dict[str, Any]) -> dict[str, object]:
    development = Path(protocol["source_development"]["artifact_root"])
    development_manifest = development / "artifact_manifest.json"
    if (
        sha256_file(development_manifest)
        != protocol["source_development"]["artifact_manifest_sha256"]
    ):
        raise ValueError("development artifact manifest changed")
    sample_manifest = Path(protocol["source_suite"]["sample_manifest"])
    if sha256_file(sample_manifest) != protocol["source_suite"]["sample_manifest_sha256"]:
        raise ValueError("focused sample manifest changed")
    default_path = Path(
        "artifacts/speculating_experts_accuracy_v17/models/qwen3_30b_a3b/"
        "default_vectors/default_vectors.safetensors"
    )
    if sha256_file(default_path) != protocol["static_reference"]["tensor_sha256"]:
        raise ValueError("static reference count source changed")
    return {
        "state": "valid",
        "development_manifest_sha256": sha256_file(development_manifest),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "static_reference_tensor_sha256": sha256_file(default_path),
        "default_mean_tensor_accessed": False,
    }


def _static_subsets(protocol: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    path = Path(
        "artifacts/speculating_experts_accuracy_v17/models/qwen3_30b_a3b/"
        "default_vectors/default_vectors.safetensors"
    )
    if sha256_file(path) != protocol["static_reference"]["tensor_sha256"]:
        raise ValueError("static reference tensor changed")
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if tuple(stream.keys()) != ("count", "mean"):
            raise ValueError("default-vector tensor keys changed")
        counts = stream.get_tensor("count")
    if counts.shape != (48, 128):
        raise ValueError("static reference count tensor shape changed")
    return {layer: _top_b(counts[layer].double(), 32) for layer in range(48)}


def _append_candidate(
    row: dict[str, object],
    tensors: dict[str, Tensor],
    ops: Qwen3MoePrefetchOps,
) -> float:
    started = time.perf_counter()
    boundaries = tuple(int(value) for value in tensors["boundaries"].tolist())
    ids = tensors["router_topk_ids"]
    weights = tensors["router_topk_weights"]
    probabilities = tensors[f"{CONTROL}__pre_topk_probabilities"]
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(ops.num_layers)}
    candidate_subsets_by_boundary: list[list[tuple[int, ...]]] = []
    metrics = cast(list[dict[str, object]], row["metrics"])
    expert_bytes = physical_expert_bytes(ops, 0)
    for boundary_index, boundary in enumerate(boundaries):
        end = min(boundary + 8, ids.shape[0])
        history_ids = (
            tensors["prompt_router_topk_ids"]
            if boundary == 0
            else ids[max(0, boundary - 8) : boundary]
        )
        history_weights = (
            tensors["prompt_router_topk_weights"]
            if boundary == 0
            else weights[max(0, boundary - 8) : boundary]
        )
        by_layer = []
        for layer in range(ops.num_layers):
            subset = candidate_subsets(
                probabilities[boundary_index, layer],
                probabilities[boundary_index, layer],
                history_ids[:, layer],
                history_weights[:, layer],
            )[CANDIDATE]
            by_layer.append(subset)
            metrics.append(
                _route_metrics(
                    sample_id=str(row["sample_id"]),
                    boundary=boundary,
                    layer=layer,
                    method=CANDIDATE,
                    ids=ids[boundary:end, layer],
                    weights=weights[boundary:end, layer],
                    logits=tensors["router_logits"][boundary:end, layer],
                    subset=subset,
                    previous_subset=resident[layer],
                    expert_bytes=expert_bytes,
                )
            )
            resident[layer] = subset
        candidate_subsets_by_boundary.append(by_layer)
    tensors[f"{CANDIDATE}__subsets"] = torch.tensor(
        candidate_subsets_by_boundary,
        dtype=torch.int64,
    )
    row["selected_candidate"] = CANDIDATE
    row["selected_candidate_future_tokens_used"] = False
    return time.perf_counter() - started


def _sample_paths(output: Path, row_index: int) -> tuple[Path, Path]:
    root = output / "route/samples" / f"{row_index:05d}"
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
        raise ValueError(f"partial held-out sample artifact: {json_path}")
    row = cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_id") != sample_id
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"corrupt or incompatible held-out sample: {json_path}")
    return row


def paired_bootstrap(
    candidate: dict[str, dict[str, int | float]],
    previous: dict[str, dict[str, int | float]],
    *,
    draws: int = 10000,
    seed: int = 20260801,
) -> dict[str, object]:
    sample_ids = tuple(sorted(candidate))
    if sample_ids != tuple(sorted(previous)) or not sample_ids:
        raise ValueError("paired bootstrap sample IDs differ or are empty")
    rng = random.Random(seed)
    route_values = []
    mass_values = []
    for _ in range(draws):
        selected = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        route_values.append(
            sum(
                candidate[sample_id]["mean_route_hit"] - previous[sample_id]["mean_route_hit"]
                for sample_id in selected
            )
            / len(selected)
        )
        mass_values.append(
            sum(
                candidate[sample_id]["mean_selected_mass"]
                - previous[sample_id]["mean_selected_mass"]
                for sample_id in selected
            )
            / len(selected)
        )
    route_values.sort()
    mass_values.sort()
    lower = int(0.025 * (draws - 1))
    upper = int(0.975 * (draws - 1))
    return {
        "unit": "sample",
        "draws": draws,
        "seed": seed,
        "candidate_minus_previous": {
            "route_hit": {
                "paired_sample_mean": sum(
                    candidate[key]["mean_route_hit"] - previous[key]["mean_route_hit"]
                    for key in sample_ids
                )
                / len(sample_ids),
                "percentile_95": [route_values[lower], route_values[upper]],
            },
            "selected_mass": {
                "paired_sample_mean": sum(
                    candidate[key]["mean_selected_mass"] - previous[key]["mean_selected_mass"]
                    for key in sample_ids
                )
                / len(sample_ids),
                "percentile_95": [mass_values[lower], mass_values[upper]],
            },
        },
    }


def _group_aggregates(
    metrics: list[dict[str, object]],
    key: str,
) -> dict[str, dict[str, dict[str, int | float]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metrics:
        value = str(int(cast(Any, row[key])) // 8) if key == "layer" else str(row[key])
        grouped[value].append(row)
    return {value: _aggregate_metrics(rows) for value, rows in sorted(grouped.items())}


def _per_anchor(
    samples: list[tuple[dict[str, Any], dict[str, Tensor]]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    rows: dict[tuple[str, int], dict[str, float | int]] = defaultdict(
        lambda: {"hits": 0, "slots": 0, "mass_hit": 0.0, "mass_total": 0.0}
    )
    for _row, tensors in samples:
        boundaries = tuple(int(value) for value in tensors["boundaries"].tolist())
        ids = tensors["router_topk_ids"]
        weights = tensors["router_topk_weights"]
        for method in METHODS:
            subsets = tensors[f"{method}__subsets"]
            for boundary_index, boundary in enumerate(boundaries):
                end = min(boundary + 8, ids.shape[0])
                for anchor in range(end - boundary):
                    for layer in range(48):
                        allowed = set(subsets[boundary_index, layer].tolist())
                        token_ids = ids[boundary + anchor, layer]
                        token_weights = weights[boundary + anchor, layer].double()
                        mask = torch.tensor([int(value) in allowed for value in token_ids])
                        state = rows[(method, anchor + 1)]
                        state["hits"] = int(state["hits"]) + int(mask.sum())
                        state["slots"] = int(state["slots"]) + token_ids.numel()
                        state["mass_hit"] = float(state["mass_hit"]) + float(
                            token_weights[mask].sum()
                        )
                        state["mass_total"] = float(state["mass_total"]) + float(
                            token_weights.sum()
                        )
    return {
        method: {
            str(anchor): {
                **rows[(method, anchor)],
                "mean_route_hit": int(rows[(method, anchor)]["hits"])
                / int(rows[(method, anchor)]["slots"]),
                "mean_selected_mass": float(rows[(method, anchor)]["mass_hit"])
                / float(rows[(method, anchor)]["mass_total"]),
            }
            for anchor in range(1, 9)
        }
        for method in METHODS
    }


def _decision(
    protocol: dict[str, Any],
    aggregates: dict[str, dict[str, int | float]],
    audit_pass: bool,
) -> dict[str, object]:
    candidate = aggregates[CANDIDATE]
    previous = aggregates[PREVIOUS]
    oracle = aggregates[ORACLE]
    route_gain = float(candidate["mean_route_hit"]) - float(previous["mean_route_hit"])
    mass_gain = float(candidate["mean_selected_mass"]) - float(previous["mean_selected_mass"])
    route_gap = float(oracle["mean_route_hit"]) - float(previous["mean_route_hit"])
    mass_gap = float(oracle["mean_selected_mass"]) - float(previous["mean_selected_mass"])
    route_recovery = route_gain / route_gap if route_gap > 0 else 0.0
    mass_recovery = mass_gain / mass_gap if mass_gap > 0 else 0.0
    gate = protocol["strong_candidate_gate"]
    checks = {
        "route_hit_gap_recovery_pass": route_recovery
        >= gate["minimum_oracle_minus_previous_gap_recovery"],
        "selected_mass_gap_recovery_pass": mass_recovery
        >= gate["minimum_oracle_minus_previous_gap_recovery"],
        "absolute_route_hit_pass": float(candidate["mean_route_hit"])
        >= gate["minimum_mean_route_hit"],
        "absolute_selected_mass_pass": float(candidate["mean_selected_mass"])
        >= gate["minimum_mean_selected_mass"],
        "route_hit_improvement_pass": route_gain
        >= gate["minimum_route_hit_improvement_over_previous"],
        "selected_mass_improvement_pass": mass_gain
        >= gate["minimum_selected_mass_improvement_over_previous"],
        "simulated_transfer_reduction_pass": float(candidate["estimated_transfer_reduction"])
        >= gate["minimum_simulated_transfer_reduction"],
        "cache_rng_information_audit_pass": audit_pass,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": gate["pass_decision"] if passed else gate["failure_decision"],
        "selected_candidate": CANDIDATE,
        "metrics": candidate,
        "references": {
            method: aggregates[method] for method in (ORACLE, PREVIOUS, STATIC, CONTROL)
        },
        "route_hit_improvement_over_previous": route_gain,
        "selected_mass_improvement_over_previous": mass_gain,
        "route_hit_oracle_gap_recovery": route_recovery,
        "selected_mass_oracle_gap_recovery": mass_recovery,
        "checks": checks,
        "strong_candidate_gate_pass": passed,
        "accuracy_used": False,
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
    bootstrap: dict[str, object],
) -> str:
    lines = [
        f"# {ANALYSIS_ID}",
        "",
        f"Held-out route decision: **{decision['decision']}**. The focused v1 terminal decision "
        "remains **STOP/PIVOT**.",
        "",
        "This is eight-row held-out, teacher-forced open-loop route evidence. It is not task "
        "accuracy, actual hard closed-loop generation, or a runtime speedup claim.",
        "",
    ]
    for method in METHODS:
        row = aggregates[method]
        lines.append(
            f"- `{method}`: hit {float(row['mean_route_hit']):.6f}, mass "
            f"{float(row['mean_selected_mass']):.6f}, simulated transfer reduction "
            f"{float(row['estimated_transfer_reduction']):.6f}."
        )
    paired = cast(dict[str, Any], bootstrap["candidate_minus_previous"])
    lines.extend(
        (
            "",
            "## Strong-candidate checks",
            "",
            f"- Oracle-gap recovery: hit "
            f"{float(cast(Any, decision['route_hit_oracle_gap_recovery'])):.6f}, mass "
            f"{float(cast(Any, decision['selected_mass_oracle_gap_recovery'])):.6f}.",
            f"- Paired sample-bootstrap route-gain 95% interval: "
            f"{paired['route_hit']['percentile_95']}.",
            f"- Paired sample-bootstrap mass-gain 95% interval: "
            f"{paired['selected_mass']['percentile_95']}.",
            "",
            "## Claim boundary",
            "",
            "The candidate uses no learned/fitted value, offline prior, default-vector value, "
            "future token, answer, correctness, or accuracy. Static-frequency counts are "
            "reference-only and the default mean tensor is never accessed. Transfer is simulated; "
            "probe and total replay runtime are measured. Gate failure stops before task accuracy "
            "or closed-loop execution.",
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
        raise ValueError("held-out artifact manifest config changed")
    for row in manifest["artifacts"]:
        path = output / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"held-out artifact checksum mismatch: {path}")
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
        raise ValueError("held-out route pipeline is incomplete")
    for reference in protocol["rows"]:
        if (
            _valid_sample(
                output,
                int(reference["row_index"]),
                str(reference["sample_id"]),
            )
            is None
        ):
            raise ValueError("held-out route sample is missing")
    manifest = _validate_manifest(output)
    return {
        "state": "valid",
        "decision": status["decision"],
        "samples": 8,
        "artifacts": manifest["artifact_count"],
    }


def run(config_path: Path, physical_gpu: int) -> dict[str, object]:
    started = time.perf_counter()
    protocol = _load_protocol(config_path)
    source_audit = _source_audit(protocol)
    output = Path(protocol["artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    suite, _ = load_focused_suite_and_manifest(Path(protocol["source_suite"]["config"]))
    if suite.fingerprint() != protocol["source_suite"]["config_fingerprint"]:
        raise ValueError("focused suite fingerprint changed")
    sources = _source_rows(suite)
    completed: list[dict[str, Any]] = []
    missing = []
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
        zero_defaults = DefaultVectorArtifact(
            torch.zeros((48, 128), dtype=torch.int64),
            torch.zeros((48, 128, ops.hidden_size), dtype=torch.bfloat16),
            "calibration-free-synthetic-zero-interface-v1",
            "synthetic_zero_interface_not_an_expert_prior",
        )
        probe = QwenPseudoEmbeddingProbe(
            model,
            ops,
            zero_defaults,
            QwenPseudoVariant(CONTROL, "sampled_next_token", "independent", "zero"),
            anchors=tuple(range(1, 9)),
            budget=32,
        )
        static_subsets = _static_subsets(protocol)
        for reference in missing:
            row_index = int(reference["row_index"])
            source = sources[row_index]
            if source["sample_id"] != reference["sample_id"]:
                raise ValueError("held-out source row identity changed")
            json_path, tensor_path = _sample_paths(output, row_index)
            try:
                row, tensors = run_route_sample(
                    suite,
                    model,
                    tokenizer,
                    ops,
                    source,
                    (probe,),
                    static_subsets,
                    partition="held_out_route",
                    max_route_tokens=128,
                    physical_gpu=physical_gpu,
                    capture_prompt_route_window=8,
                )
                candidate_seconds = _append_candidate(row, tensors, ops)
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
                    "candidate_postprocess_seconds_measured": candidate_seconds,
                    "candidate_default_vector_values_used": False,
                    "static_reference_count_only": True,
                    "default_mean_tensor_accessed": False,
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
                        "stage": "prompt_route_held_out",
                        "sample_id": row["sample_id"],
                        "route_tokens": row["route_tokens"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del probe, zero_defaults, ops, model
        torch.cuda.empty_cache()
    completed.sort(key=lambda row: int(row["row_index"]))
    samples_with_tensors = []
    metrics: list[dict[str, object]] = []
    for row in completed:
        tensors = load_file(str(output / str(row["tensor_path"])))
        samples_with_tensors.append((row, tensors))
        metrics.extend(cast(list[dict[str, object]], row["metrics"]))
    aggregates = _aggregate_metrics(metrics)
    if set(aggregates) != set(METHODS):
        raise ValueError("held-out aggregate method set changed")
    by_sample_raw = _group_aggregates(metrics, "sample_id")
    per_sample = {
        method: {
            sample_id: cast(dict[str, dict[str, float]], rows)[method]
            for sample_id, rows in by_sample_raw.items()
        }
        for method in METHODS
    }
    bootstrap = paired_bootstrap(
        per_sample[CANDIDATE],
        per_sample[PREVIOUS],
        draws=protocol["bootstrap"]["samples"],
        seed=protocol["bootstrap"]["seed"],
    )
    prompt_audits = [row["prompt_route_capture_audit"] for row in completed]
    probe_audits = [audit for row in completed for audit in row["cache_rng_audits"]]
    audit_pass = (
        all(bool(audit["production_cache_signature_unchanged"]) for audit in probe_audits)
        and all(bool(audit["production_rng_unchanged"]) for audit in probe_audits)
        and all(bool(audit["shadow_cache_discarded"]) for audit in probe_audits)
        and all(
            bool(audit["production_cache_signature_unchanged_after_extraction"])
            for audit in prompt_audits
        )
        and all(bool(audit["generation_rng_unchanged"]) for audit in prompt_audits)
        and all(not bool(audit["future_generated_tokens_used"]) for audit in prompt_audits)
    )
    if not audit_pass:
        raise RuntimeError("held-out cache/RNG/information audit failed")
    decision = _decision(protocol, aggregates, audit_pass)
    strata = {
        "by_sample": by_sample_raw,
        "by_boundary": _group_aggregates(metrics, "boundary"),
        "by_eight_layer_block": _group_aggregates(metrics, "layer"),
        "by_anchor": _per_anchor(samples_with_tensors),
    }
    candidate_rows = [row for row in metrics if row["method"] == CANDIDATE]
    worst = sorted(
        candidate_rows,
        key=lambda row: (
            float(cast(Any, row["route_hit_rate"])),
            float(cast(Any, row["selected_mass_coverage"])),
            str(row["sample_id"]),
            int(cast(Any, row["boundary"])),
            int(cast(Any, row["layer"])),
        ),
    )[:100]
    write_json_atomic(output / "resolved_config.json", protocol)
    write_json_atomic(output / "resolved_environment.json", _software_hardware())
    write_json_atomic(output / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(output / "source_audit.json", source_audit)
    write_json_atomic(output / "aggregates.json", aggregates)
    write_json_atomic(output / "stratified_metrics.json", strata)
    write_json_atomic(output / "paired_bootstrap.json", bootstrap)
    write_json_atomic(output / "worst_cases.json", {"rows": worst})
    write_json_atomic(output / "decision.json", decision)
    write_json_atomic(
        output / "cache_rng_information_audit.json",
        {
            "state": "pass" if audit_pass else "fail",
            "probe_calls": len(probe_audits),
            "prompt_capture_calls": len(prompt_audits),
            "candidate_future_token_leakage": False,
            "candidate_default_vector_values_used": False,
            "static_reference_count_only": True,
            "default_mean_tensor_accessed": False,
            "prompt_audits": prompt_audits,
        },
    )
    write_json_atomic(
        output / "cost_report.json",
        {
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "route_tokens": row["route_tokens"],
                    "elapsed_seconds_measured": row["elapsed_seconds_measured"],
                    "candidate_postprocess_seconds_measured": row[
                        "candidate_postprocess_seconds_measured"
                    ],
                    "probe_costs": row["probe_costs"],
                }
                for row in completed
            ],
            "transfer_kind": "simulated",
        },
    )
    raw = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in metrics
    ).encode()
    _write_bytes_atomic(output / "raw_metrics.jsonl.gz", gzip.compress(raw, mtime=0))
    _write_bytes_atomic(output / "report.md", _report(aggregates, decision, bootstrap).encode())
    failed = sorted(str(path.relative_to(output)) for path in output.rglob("*.FAILED.json"))
    write_json_atomic(
        output / "resume_audit.json",
        {
            "schema_version": 1,
            "state": "complete",
            "samples_expected": 8,
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
            "source_samples": [str(row["sample_id"]) for row in protocol["rows"]],
            "model": suite.model.model_id,
            "model_revision": suite.model.revision,
            "precision": suite.model.precision,
            "model_loaded_for_route_generation": True,
            "gpu_used_for_route_generation": True,
            "model_loaded_this_invocation": bool(missing),
            "gpu_used_this_invocation": bool(missing),
            "network_downloads": False,
            "task_accuracy_measured": False,
            "actual_closed_loop_generation": False,
            "route_evaluation": "teacher_forced_saved_v17_trajectory_open_loop",
            "future_true_tokens_used_by_candidate": False,
            "default_vector_values_used_by_candidate": False,
            "offline_calibration_statistics_used_by_candidate": False,
            "learned_or_fitted_parameters": False,
            "static_reference_count_only": True,
            "default_mean_tensor_accessed": False,
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
        "strong_candidate_gate_pass": decision["strong_candidate_gate_pass"],
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
