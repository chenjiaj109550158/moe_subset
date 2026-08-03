"""Run the frozen calibration-free mass-preserving pseudo-residual analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from safetensors.torch import load_file

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    PolicySpec,
    _aggregate_policy,
    _paired_bootstrap,
    _per_sample,
    _save_tensors_atomic,
    run_policy_sample,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.qwen_pseudo import SubsetResidualExecution
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_mass_preserving_residual_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "6682aa83487c21924b0a80f755c36eb9caa0be72626a098339ae80b3ce18b5bb"
SAMPLES_SHA256 = "a0942180ef7cd8c4bf9b003069a96b93554653e5ca3b7c75a5babdc5e567d76d"
LAYERS = 48
EXPERTS = 128
TOP_K = 8
HORIZON = 8
BASELINE = "sampled_unigram_full_continuation"
BFLOAT16_ABSOLUTE_TOLERANCE = float(torch.finfo(torch.bfloat16).eps)
BASELINE_ROOT = Path("artifacts/pseudo_one_forward_context_continuation_v1")
PREVIOUS = "previous_route_commitment"
PREVIOUS_ROOT = Path("artifacts/pseudo_executed_embedding_composition_v1")
STATIC = "reference__static_frequency"
STATIC_ROOT = Path("artifacts/pseudo_embedding_qwen_gsm8k_residual_window_v1")


@dataclass(frozen=True)
class ResidualExecutionSpec:
    key: str
    strategy: SubsetResidualExecution

    def policy(self) -> PolicySpec:
        return PolicySpec(
            self.key,
            "pseudo",
            residual="zero",
            content="context_sampled_unigram_full_causal",
            selection="first_four_anchor_core_plus_history_fill",
        )

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


SPECS = (
    ResidualExecutionSpec(
        "natural_top8_intersection_zero_missing",
        "natural_top8_intersection_zero_missing",
    ),
    ResidualExecutionSpec(
        "rerouted_top8_scaled_by_captured_natural_mass",
        "rerouted_top8_scaled_by_captured_natural_mass",
    ),
    ResidualExecutionSpec(
        "captured_natural_plus_substitute_missing_mass",
        "captured_natural_plus_substitute_missing_mass",
    ),
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("mass-preserving residual protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("mass-preserving residual protocol identity changed")
    if tuple(config["variants"]["order"]) != tuple(spec.key for spec in SPECS):
        raise ValueError("mass-preserving residual variant order changed")
    source = config["source"]
    for path_key, hash_key in (
        ("diagnosis_config", "diagnosis_config_sha256"),
        ("diagnosis_sample_manifest", "diagnosis_sample_manifest_sha256"),
        ("diagnosis_artifact_manifest", "diagnosis_artifact_manifest_sha256"),
        ("context_config", "context_config_sha256"),
        ("context_sample_manifest", "context_sample_manifest_sha256"),
        ("context_artifact_manifest", "context_artifact_manifest_sha256"),
        ("previous_reference_config", "previous_reference_config_sha256"),
        ("previous_reference_artifact_manifest", "previous_reference_artifact_manifest_sha256"),
        ("static_reference_config", "static_reference_config_sha256"),
        ("static_reference_artifact_manifest", "static_reference_artifact_manifest_sha256"),
        ("focused_suite_config", "focused_suite_config_sha256"),
    ):
        path = Path(str(source[path_key]))
        if sha256_file(path) != source[hash_key]:
            raise ValueError(f"mass-preserving residual source checksum changed: {path}")
    return config, samples


def _paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "development" / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _reference_paths(root: Path, row_index: int, policy: str) -> tuple[Path, Path]:
    base = root / "development" / "samples" / f"{row_index:05d}" / policy
    return base.with_suffix(".json"), base.with_suffix(".safetensors")


def _reference_row(root: Path, reference: dict[str, Any], policy: str) -> dict[str, Any]:
    json_path, tensor_path = _reference_paths(root, int(reference["row_index"]), policy)
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"missing checksum-pinned reference row: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("sample_id") != reference["sample_id"]
        or row.get("policy") != policy
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible checksum-pinned reference row: {json_path}")
    return row


def _valid(reference: dict[str, Any], spec: ResidualExecutionSpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial mass-preserving residual row: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("residual_execution_spec_fingerprint") != spec.fingerprint()
        or row.get("subset_residual_execution") != spec.strategy
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible mass-preserving residual row: {json_path}")
    tensors = load_file(str(tensor_path))
    expected = {
        "shadow_executed_topk_weights": (8, HORIZON, LAYERS, TOP_K),
        "shadow_captured_natural_mass": (8, HORIZON, LAYERS),
        "pseudo_router_logits": (8, LAYERS, HORIZON, EXPERTS),
    }
    for key, shape in expected.items():
        if key not in tensors or tuple(tensors[key].shape) != shape:
            raise ValueError(f"mass-preserving residual tensor shape changed: {tensor_path}/{key}")
    return row


def run(*, physical_gpu: int, shard_index: int, shard_count: int) -> dict[str, object]:
    config, samples = _protocol()
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid mass-preserving residual shard")
    references = [
        row
        for index, row in enumerate(samples["partitions"]["development"])
        if index % shard_count == shard_index
    ]
    missing = [
        (reference, spec)
        for reference in references
        for spec in SPECS
        if _valid(reference, spec) is None
    ]
    if not missing:
        return {"state": "already_complete", "rows": len(references) * len(SPECS)}
    suite = load_pseudo_embedding_config(config["source"]["focused_suite_config"])
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    model_config = _source_model(accuracy, physical_gpu)
    seed = int(config["decode"]["seed"])
    seed_everything(seed)
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed facts changed")
    sources = _source_rows(suite)
    revision = _git_head()
    completed = 0
    for reference, spec in missing:
        row_index = int(reference["row_index"])
        json_path, tensor_path = _paths(row_index, spec.key)
        try:
            seed_everything(seed)
            row, tensors = run_policy_sample(
                config,
                suite,
                accuracy,
                model,
                tokenizer,
                ops,
                sources[row_index],
                spec.policy(),
                {},
                stage="development",
                route_token_cap=int(config["operating_point"]["route_token_cap"]),
                physical_gpu=physical_gpu,
                shadow_expert_execution=True,
                analysis_id=ANALYSIS_ID,
                analysis_config_sha256=CONFIG_SHA256,
                sample_manifest_sha256=SAMPLES_SHA256,
                subset_residual_execution=spec.strategy,
            )
            row["residual_execution_spec"] = asdict(spec)
            row["residual_execution_spec_fingerprint"] = spec.fingerprint()
            row["execution_git_head"] = revision
            _save_tensors_atomic(tensor_path, tensors)
            row["tensor_sha256"] = sha256_file(tensor_path)
            row["tensor_bytes"] = tensor_path.stat().st_size
            write_json_atomic(json_path, row)
            completed += 1
        except Exception as error:
            failure = json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")
            write_json_atomic(
                failure,
                {
                    "state": "failed",
                    "analysis_id": ANALYSIS_ID,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": spec.key,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "execution_git_head": revision,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return {"state": "complete", "shard_index": shard_index, "completed_now": completed}


def _anchor_one(rows: list[dict[str, Any]], paths: dict[str, Path]) -> tuple[float, float]:
    hits = slots = 0
    mass_hit = mass_total = 0.0
    for row in rows:
        tensors = load_file(str(paths[str(row["sample_id"])]))
        ids = tensors["natural_router_topk_ids"]
        weights = tensors["natural_router_topk_weights"].float()
        subsets = tensors["subsets"]
        boundaries = [int(value) for value in tensors["boundaries"].tolist()]
        for boundary_index, boundary in enumerate(boundaries):
            allowed = torch.isin(ids[boundary], subsets[boundary_index])
            hits += int(allowed.sum())
            slots += int(allowed.numel())
            mass_hit += float(weights[boundary][allowed].double().sum())
            mass_total += float(weights[boundary].double().sum())
    return hits / slots, mass_hit / mass_total


def _candidate_audit(
    row: dict[str, Any], spec: ResidualExecutionSpec, config: dict[str, Any]
) -> bool:
    expected = config["audits"]
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    for boundary_index, (cost, audit) in enumerate(
        zip(row["probe_costs"], row["cache_rng_audits"], strict=True)
    ):
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                audit.get("one_causal_forward_per_boundary", False),
                audit.get("shadow_expert_execution", False),
                audit.get("subset_residual_execution") == spec.strategy,
                audit.get("execution_weights_finite_nonnegative", False),
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                not audit.get("forbidden_inputs_present", True),
                audit.get("diagnostic_oracle_inputs_present") is False,
                audit.get("deployable") is True,
                cost["attention_calls"] == expected["expected_attention_calls_per_boundary"],
                cost["attention_queries"] == expected["expected_attention_queries_per_boundary"],
                cost["router_calls"] == expected["expected_router_calls_per_boundary"],
                cost["expert_calls"] == expected["expected_expert_calls_per_boundary"],
                cost["lm_head_calls"] == expected["expected_lm_head_calls_per_boundary"],
            )
        ):
            return False
        weight_sum = float(audit["execution_weight_sum_mean"])
        captured = float(audit["captured_natural_mass_mean"])
        if boundary_index == 0 and (
            abs(weight_sum - 1) > BFLOAT16_ABSOLUTE_TOLERANCE
            or abs(captured - 1) > BFLOAT16_ABSOLUTE_TOLERANCE
        ):
            return False
        if boundary_index > 0:
            if spec.strategy == "captured_natural_plus_substitute_missing_mass":
                if abs(weight_sum - 1) > BFLOAT16_ABSOLUTE_TOLERANCE:
                    return False
            elif abs(weight_sum - captured) > BFLOAT16_ABSOLUTE_TOLERANCE:
                return False
        continuation = audit.get("context_continuation") or {}
        if continuation.get("mode") != "sampled_unigram_full_continuation":
            return False
        if not continuation.get("anchor_one_is_sampled_next", False):
            return False
        if not continuation.get("copied_indices_in_known_context", False):
            return False
    return True


def _candidate_tensor_weight_audit(
    rows: list[dict[str, Any]],
    paths: dict[str, Path],
    spec: ResidualExecutionSpec,
) -> bool:
    for row in rows:
        tensors = load_file(str(paths[str(row["sample_id"])]))
        weights = tensors["shadow_executed_topk_weights"].float()
        captured = tensors["shadow_captured_natural_mass"].float()
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            return False
        weight_sums = weights.sum(dim=-1)
        target = (
            torch.ones_like(weight_sums)
            if spec.strategy == "captured_natural_plus_substitute_missing_mass"
            else captured
        )
        if not torch.allclose(weight_sums, target, rtol=0.0, atol=BFLOAT16_ABSOLUTE_TOLERANCE):
            return False
    return True


def aggregate() -> dict[str, object]:
    config, samples = _protocol()
    references = samples["partitions"]["development"]
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_paths: dict[str, dict[str, Path]] = defaultdict(dict)
    for reference in references:
        for spec in SPECS:
            row = _valid(reference, spec)
            if row is None:
                raise RuntimeError(f"missing residual row: {reference['sample_id']}/{spec.key}")
            candidates[spec.key].append(row)
            candidate_paths[spec.key][str(row["sample_id"])] = _paths(
                int(row["row_index"]), spec.key
            )[1]
    baseline_rows = [_reference_row(BASELINE_ROOT, row, BASELINE) for row in references]
    previous_rows = [_reference_row(PREVIOUS_ROOT, row, PREVIOUS) for row in references]
    static_rows = [_reference_row(STATIC_ROOT, row, STATIC) for row in references]
    policies = {BASELINE: _aggregate_policy(baseline_rows)}
    policies.update({key: _aggregate_policy(rows) for key, rows in candidates.items()})
    references_aggregated = {
        PREVIOUS: _aggregate_policy(previous_rows),
        STATIC: _aggregate_policy(static_rows),
    }
    baseline_paths = {
        str(row["sample_id"]): _reference_paths(BASELINE_ROOT, int(row["row_index"]), BASELINE)[1]
        for row in baseline_rows
    }
    baseline_anchor = _anchor_one(baseline_rows, baseline_paths)
    candidate_anchors = {
        key: _anchor_one(rows, candidate_paths[key]) for key, rows in candidates.items()
    }
    rule = config["selection_rule"]["eligible_requires"]
    comparisons = {}
    eligible = []
    for spec in SPECS:
        values = policies[spec.key]
        route_delta = float(values["mean_route_hit"]) - float(policies[BASELINE]["mean_route_hit"])
        mass_delta = float(values["mean_selected_mass"]) - float(
            policies[BASELINE]["mean_selected_mass"]
        )
        anchor_route_delta = candidate_anchors[spec.key][0] - baseline_anchor[0]
        anchor_mass_delta = candidate_anchors[spec.key][1] - baseline_anchor[1]
        audit_pass = all(
            _candidate_audit(row, spec, config) for row in candidates[spec.key]
        ) and _candidate_tensor_weight_audit(candidates[spec.key], candidate_paths[spec.key], spec)
        passes = all(
            (
                route_delta >= float(rule["minimum_mean_route_hit_delta_over_baseline"]),
                mass_delta >= float(rule["minimum_mean_selected_mass_delta_over_baseline"]),
                anchor_route_delta >= -float(rule["maximum_anchor_one_route_hit_regression"]),
                anchor_mass_delta >= -float(rule["maximum_anchor_one_selected_mass_regression"]),
                float(values["estimated_transfer_reduction"])
                >= float(rule["minimum_estimated_transfer_reduction"]),
                float(values["mean_probe_latency_seconds"])
                <= float(rule["maximum_mean_probe_latency_seconds"]),
                audit_pass,
            )
        )
        if passes:
            eligible.append(spec.key)
        comparisons[spec.key] = {
            "mean_route_hit_delta_over_baseline": route_delta,
            "mean_selected_mass_delta_over_baseline": mass_delta,
            "anchor_one_route_hit_delta_over_baseline": anchor_route_delta,
            "anchor_one_selected_mass_delta_over_baseline": anchor_mass_delta,
            "paired_bootstrap": _paired_bootstrap(
                {str(row["sample_id"]): _per_sample(row) for row in candidates[spec.key]},
                {str(row["sample_id"]): _per_sample(row) for row in baseline_rows},
                draws=int(config["analysis"]["bootstrap_samples"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
            "audit_pass": audit_pass,
            "eligible": passes,
        }
    selected = (
        sorted(
            eligible,
            key=lambda key: (
                -float(policies[key]["mean_selected_mass"]),
                -float(policies[key]["mean_route_hit"]),
                -float(policies[key]["estimated_transfer_reduction"]),
                float(policies[key]["mean_probe_latency_seconds"]),
                key,
            ),
        )[0]
        if eligible
        else None
    )
    progress = config["progress_gate"]
    progress_gate_pass = False
    if selected is not None:
        selected_values = policies[selected]
        progress_gate_pass = all(
            (
                float(selected_values["mean_route_hit"])
                - float(references_aggregated[PREVIOUS]["mean_route_hit"])
                >= float(progress["minimum_absolute_mean_route_hit_improvement"]),
                float(selected_values["mean_selected_mass"])
                - float(references_aggregated[PREVIOUS]["mean_selected_mass"])
                >= float(progress["minimum_absolute_mean_selected_mass_improvement"]),
                float(selected_values["mean_route_hit"])
                >= float(references_aggregated[STATIC]["mean_route_hit"]),
                float(selected_values["mean_selected_mass"])
                >= float(references_aggregated[STATIC]["mean_selected_mass"]),
                float(selected_values["estimated_transfer_reduction"])
                >= float(progress["minimum_estimated_transfer_reduction"]),
            )
        )
    weight_report = {}
    for spec in SPECS:
        weight_sums = []
        captured = []
        for row in candidates[spec.key]:
            tensors = load_file(str(candidate_paths[spec.key][str(row["sample_id"])]))
            weight_sums.append(tensors["shadow_executed_topk_weights"].float().sum(dim=-1))
            captured.append(tensors["shadow_captured_natural_mass"].float())
        weights = torch.cat(weight_sums)
        masses = torch.cat(captured)
        target = (
            torch.ones_like(weights)
            if spec.strategy == "captured_natural_plus_substitute_missing_mass"
            else masses
        )
        weight_report[spec.key] = {
            "bfloat16_absolute_tolerance": BFLOAT16_ABSOLUTE_TOLERANCE,
            "tensor_semantics_audit_pass": bool(
                torch.allclose(weights, target, rtol=0.0, atol=BFLOAT16_ABSOLUTE_TOLERANCE)
            ),
            "execution_weight_sum_mean": float(weights.mean()),
            "execution_weight_sum_min": float(weights.min()),
            "execution_weight_sum_max": float(weights.max()),
            "captured_natural_mass_mean": float(masses.mean()),
            "captured_natural_mass_min": float(masses.min()),
            "captured_natural_mass_max": float(masses.max()),
            "mean_absolute_weight_sum_minus_captured_mass": float((weights - masses).abs().mean()),
            "maximum_absolute_weight_sum_minus_target": float((weights - target).abs().max()),
        }
    decision = {
        "analysis_id": ANALYSIS_ID,
        "decision": "NARROW" if progress_gate_pass else "STOP/PIVOT",
        "selected_candidate": selected,
        "eligible_candidates": eligible,
        "development_progress_gate_pass": progress_gate_pass,
        "held_out_route_authorized": False,
        "held_out_route_requires_separate_frozen_protocol": True,
        "task_accuracy_authorized": False,
    }
    audit = {
        "all_pass": all(value["audit_pass"] for value in comparisons.values()),
        "new_sample_policy_rows": sum(len(rows) for rows in candidates.values()),
        "baseline_rows_reused": len(baseline_rows),
        "accuracy_or_correctness_used": False,
        "future_tokens_or_component_oracle_used": False,
        "learned_or_fitted_parameters": False,
        "bfloat16_absolute_weight_tolerance": BFLOAT16_ABSOLUTE_TOLERANCE,
    }
    root = OUTPUT / "development"
    write_json_atomic(
        root / "aggregates.json", {"policies": policies, "references": references_aggregated}
    )
    write_json_atomic(root / "comparisons.json", comparisons)
    write_json_atomic(
        root / "anchor_one.json", {"baseline": baseline_anchor, "candidates": candidate_anchors}
    )
    write_json_atomic(root / "weight_semantics.json", weight_report)
    write_json_atomic(root / "audit.json", audit)
    write_json_atomic(root / "decision.json", decision)
    write_json_atomic(
        root / "cost_report.json",
        {
            key: {
                "mean_probe_latency_seconds": values["mean_probe_latency_seconds"],
                "max_temporary_cuda_bytes": values["max_temporary_cuda_bytes"],
                "attention_calls": values["attention_calls"],
                "attention_queries": values["attention_queries"],
                "router_calls": values["router_calls"],
                "expert_calls": values["expert_calls"],
            }
            for key, values in policies.items()
        },
    )
    status: dict[str, object] = {
        "state": "complete",
        "new_sample_policy_rows": audit["new_sample_policy_rows"],
        "all_audits_pass": audit["all_pass"],
        **decision,
    }
    write_json_atomic(root / "pipeline_status.json", status)
    return status


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
        if relative not in excluded:
            artifacts.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    manifest = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(OUTPUT / "artifact_manifest.json", manifest)
    return manifest


def _report() -> str:
    aggregates = _json(OUTPUT / "development" / "aggregates.json")
    comparisons = _json(OUTPUT / "development" / "comparisons.json")
    decision = _json(OUTPUT / "development" / "decision.json")
    weights = _json(OUTPUT / "development" / "weight_semantics.json")
    audit = _json(OUTPUT / "development" / "audit.json")
    lines = [
        "# Mass-preserving previous-subset residual v1",
        "",
        "All new candidates are calibration-free and deployable at the boundary: sampled-unigram "
        "known-context content, one batched causal H=8 pseudo traversal, and only the previous "
        "realized layer-local B32 experts after boundary zero.",
        "",
        "| Policy | Route hit | Selected mass | Transfer reduction | Probe s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, values in aggregates["policies"].items():
        lines.append(
            f"| {key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(values['estimated_transfer_reduction']):.6f} | "
            f"{float(values['mean_probe_latency_seconds']):.4f} |"
        )
    lines.extend(["", "## Comparisons to checksum-pinned sampled-unigram baseline", ""])
    for key, values in comparisons.items():
        lines.append(
            f"- {key}: route/mass delta "
            f"{float(values['mean_route_hit_delta_over_baseline']):+.6f}/"
            f"{float(values['mean_selected_mass_delta_over_baseline']):+.6f}; anchor-1 "
            f"{float(values['anchor_one_route_hit_delta_over_baseline']):+.6f}/"
            f"{float(values['anchor_one_selected_mass_delta_over_baseline']):+.6f}; "
            f"eligible `{values['eligible']}`; paired 95% intervals "
            f"{values['paired_bootstrap']['route_hit_delta_percentile_95']} and "
            f"{values['paired_bootstrap']['selected_mass_delta_percentile_95']}."
        )
    lines.extend(["", "## Residual weight semantics", ""])
    for key, values in weights.items():
        lines.append(
            f"- {key}: execution-weight sum mean "
            f"{float(values['execution_weight_sum_mean']):.6f}; captured natural mass mean "
            f"{float(values['captured_natural_mass_mean']):.6f}."
        )
    lines.extend(
        [
            "",
            f"Development decision: **{decision['decision']}**; selected candidate: "
            f"`{decision['selected_candidate']}`; original development progress gate: "
            f"`{decision['development_progress_gate_pass']}`.",
            "",
            f"All candidate cache/RNG/information/weight audits pass: `{audit['all_pass']}`. "
            "The baseline, previous-route, and static rows are checksum-pinned prior measured "
            "rows and were not rerun.",
            "",
            "Route and task-state metrics plus probe cost are measured; transfer is simulated. "
            "No held-out route, task accuracy, free generation, exact-token identity, NLL, "
            "closed-loop runtime, or runtime speedup was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples = _protocol()
    development = _json(OUTPUT / "development" / "pipeline_status.json")
    if development.get("state") != "complete" or not development.get("all_audits_pass"):
        raise ValueError("mass-preserving residual aggregate is incomplete")
    rows = [
        _json(path)
        for path in sorted((OUTPUT / "development" / "samples").glob("*/*.json"))
        if ".FAILED." not in path.name
    ]
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    decision = _json(OUTPUT / "development" / "decision.json")
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "git_head_at_report": _git_head(),
            "row_execution_git_heads": sorted({str(row["execution_git_head"]) for row in rows}),
            "pid": os.getpid(),
            "ppid": os.getppid(),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit_and_selected_mass",
                "execution_weight_sums_and_captured_natural_mass",
                "native_one_forward_probe_latency_and_memory",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": [
                "held_out_route",
                "task_accuracy",
                "free_generation",
                "exact_token_identity",
                "nll_or_perplexity",
                "closed_loop_runtime",
                "runtime_speedup",
            ],
        },
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "new_sample_policy_rows_validated": len(rows),
            "baseline_rows_reused": 4,
            "atomic_json_tensor_pairs": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "analysis_id": ANALYSIS_ID,
            "model": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "development_sample_ids": [
                row["sample_id"] for row in samples["partitions"]["development"]
            ],
            "future_token_or_component_oracle_inputs": False,
            "learned_or_fitted_parameters": False,
            "default_vector_values_loaded": False,
            "task_accuracy_measured": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "analysis_id": ANALYSIS_ID,
            "decision": decision["decision"],
            "scope": "Qwen_GSM8K_H8_B32_four_row_mass_preserving_residual_development",
            "selected_candidate": decision["selected_candidate"],
            "development_progress_gate_pass": decision["development_progress_gate_pass"],
            "held_out_route_authorized": False,
            "task_accuracy_authorized": False,
            "runtime_speedup_claim": False,
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report())
    manifest = _manifest()
    final = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": decision["decision"],
        "selected_candidate": decision["selected_candidate"],
        "development_progress_gate_pass": decision["development_progress_gate_pass"],
        "new_sample_policy_rows": len(rows),
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": False,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", final)
    validate()
    return final


def validate() -> dict[str, object]:
    _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "report_v1":
        raise ValueError("mass-preserving residual pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("mass-preserving residual artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"mass-preserving residual checksum changed: {path}")
    if _json(OUTPUT / "resume_audit.json")["new_sample_policy_rows_validated"] != 12:
        raise ValueError("mass-preserving residual row count changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "selected_candidate": status["selected_candidate"],
        "new_sample_policy_rows": status["new_sample_policy_rows"],
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate", "finalize", "validate", "all"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    started = time.time()
    if args.command in {"run", "all"}:
        result = run(
            physical_gpu=args.gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        if args.command == "all":
            result = aggregate()
            result = finalize()
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps({**result, "command_elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
