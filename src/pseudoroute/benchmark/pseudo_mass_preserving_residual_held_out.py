"""Run the frozen corrected-reference and held-out mass-residual analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from safetensors.torch import load_file

from pseudoroute.benchmark import pseudo_mass_preserving_residual as development_analysis
from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    BUDGET,
    HORIZON,
    PolicySpec,
    _aggregate_policy,
    _paired_bootstrap,
    _per_sample,
    _save_tensors_atomic,
    _static_subsets,
    run_policy_sample,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_mass_preserving_residual_held_out_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "2effc6808f2637f64f60836a90175f664305f5874460a458922da26bfd96416f"
SAMPLES_SHA256 = "c83dbce41acb7bce2b0cf399c2632965d784ae289f0ca81bb6a6f105aa1762f3"
LAYERS = 48
EXPERTS = 128
TOP_K = 8
BFLOAT16_ABSOLUTE_TOLERANCE = float(torch.finfo(torch.bfloat16).eps)
Stage = Literal["development_reference", "held_out_route"]

CANDIDATE = PolicySpec(
    "natural_top8_intersection_zero_missing",
    "pseudo",
    residual="zero",
    content="context_sampled_unigram_full_causal",
    selection="first_four_anchor_core_plus_history_fill",
)
PREVIOUS = PolicySpec("previous_route_commitment", "previous")
ORACLE = PolicySpec("hard_oracle_commitment", "oracle")


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("held-out mass-residual protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("held-out mass-residual protocol identity changed")
    for section in (
        "source_development",
        "source_predeclared_partition",
        "source_historical_evaluator",
        "source_suite",
    ):
        for key, expected in config[section].items():
            if key.endswith("_sha256"):
                path = Path(str(config[section][key.removesuffix("_sha256")]))
                if sha256_file(path) != expected:
                    raise ValueError(f"held-out source checksum changed: {path}")
    source_samples = _json(Path(config["source_predeclared_partition"]["sample_manifest"]))
    if (
        samples["partitions"]["held_out_route"]
        != source_samples["partitions"]["held_out_route_new"]["rows"]
    ):
        raise ValueError("held-out ranked rows differ from the predeclared partition")
    development_decision = _json(Path(config["source_development"]["decision"]))
    if development_decision.get("selected_candidate") != CANDIDATE.key:
        raise ValueError("development-selected candidate changed")
    return config, samples


def _paths(stage: Stage, row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / stage / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _specs(stage: Stage) -> tuple[PolicySpec, ...]:
    return (PREVIOUS,) if stage == "development_reference" else (ORACLE, PREVIOUS, CANDIDATE)


def _valid(stage: Stage, reference: dict[str, Any], spec: PolicySpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(stage, int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial held-out mass-residual row: {json_path}")
    row = _json(json_path)
    expected_shadow = spec == CANDIDATE
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("policy_spec_fingerprint") != spec.fingerprint()
        or row.get("shadow_expert_execution") is not expected_shadow
        or row.get("previous_first_window_static_frequency") is not (spec == PREVIOUS)
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible held-out mass-residual row: {json_path}")
    tensors = load_file(str(tensor_path))
    boundaries = tensors.get("boundaries")
    if boundaries is None or boundaries.ndim != 1 or int(boundaries[0]) != 0:
        raise ValueError(f"invalid held-out boundaries: {tensor_path}")
    expected = {
        "subsets": (len(boundaries), LAYERS, BUDGET),
        "natural_router_topk_ids": (int(row["route_tokens"]), LAYERS, TOP_K),
        "natural_router_topk_weights": (int(row["route_tokens"]), LAYERS, TOP_K),
    }
    for key, shape in expected.items():
        if key not in tensors or tuple(tensors[key].shape) != shape:
            raise ValueError(f"held-out tensor shape changed: {tensor_path}/{key}")
    if spec == CANDIDATE:
        pseudo_expected: dict[str, tuple[int, ...]] = {
            "pseudo_router_logits": (len(boundaries), LAYERS, HORIZON, EXPERTS),
            "shadow_executed_topk_weights": (
                len(boundaries),
                HORIZON,
                LAYERS,
                TOP_K,
            ),
            "shadow_captured_natural_mass": (len(boundaries), HORIZON, LAYERS),
        }
        for key, pseudo_shape in pseudo_expected.items():
            if key not in tensors or tuple(tensors[key].shape) != pseudo_shape:
                raise ValueError(f"candidate tensor shape changed: {tensor_path}/{key}")
    return row


def _stage_inputs(stage: Stage, samples: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if stage == "development_reference":
        return (
            list(samples["partitions"][stage]),
            int(_protocol()[0]["operating_point"]["development_reference_route_token_cap"]),
        )
    return (
        list(samples["partitions"][stage]),
        int(_protocol()[0]["operating_point"]["held_out_route_token_cap"]),
    )


def run_stage(
    stage: Stage, *, physical_gpu: int, shard_index: int, shard_count: int
) -> dict[str, object]:
    config, samples = _protocol()
    if stage == "held_out_route":
        status_path = OUTPUT / "development_reference" / "pipeline_status.json"
        if not status_path.is_file() or not _json(status_path).get("progress_gate_pass"):
            raise RuntimeError("corrected development progress gate did not authorize held-out")
    references, route_token_cap = _stage_inputs(stage, samples)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid held-out mass-residual shard")
    references = [row for index, row in enumerate(references) if index % shard_count == shard_index]
    missing = [
        (reference, spec)
        for reference in references
        for spec in _specs(stage)
        if _valid(stage, reference, spec) is None
    ]
    if not missing:
        return {"state": "already_complete", "rows": len(references) * len(_specs(stage))}
    suite = load_pseudo_embedding_config(config["source_suite"]["config"])
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    model_config = _source_model(accuracy, physical_gpu)
    seed = int(config["decode"]["seed"])
    seed_everything(seed)
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed facts changed")
    static_subsets = _static_subsets(config)
    sources = _source_rows(suite)
    revision = _git_head()
    completed = 0
    for reference, spec in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        if source.get("sample_id") != reference["sample_id"]:
            raise ValueError(f"held-out source/sample mismatch at row {row_index}")
        json_path, tensor_path = _paths(stage, row_index, spec.key)
        try:
            seed_everything(seed)
            row, tensors = run_policy_sample(
                config,
                suite,
                accuracy,
                model,
                tokenizer,
                ops,
                source,
                spec,
                static_subsets,
                stage="development" if stage == "development_reference" else "held_out",
                route_token_cap=route_token_cap,
                physical_gpu=physical_gpu,
                shadow_expert_execution=spec == CANDIDATE,
                analysis_id=ANALYSIS_ID,
                analysis_config_sha256=CONFIG_SHA256,
                sample_manifest_sha256=SAMPLES_SHA256,
                subset_residual_execution=(
                    "natural_top8_intersection_zero_missing"
                    if spec == CANDIDATE
                    else "renormalized_reroute"
                ),
                previous_first_window_static_frequency=spec == PREVIOUS,
            )
            row["analysis_partition"] = stage
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
                    "analysis_partition": stage,
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
    del ops, model
    torch.cuda.empty_cache()
    return {"state": "complete", "stage": stage, "completed_now": completed}


def _top32(scores: torch.Tensor) -> tuple[int, ...]:
    ranking = sorted(range(scores.numel()), key=lambda expert: (-float(scores[expert]), expert))
    return tuple(sorted(ranking[:BUDGET]))


def _previous_audit(
    row: dict[str, Any], tensor_path: Path, static: dict[int, tuple[int, ...]]
) -> bool:
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    tensors = load_file(str(tensor_path))
    subsets = tensors["subsets"]
    boundaries = [int(value) for value in tensors["boundaries"].tolist()]
    ids = tensors["natural_router_topk_ids"]
    weights = tensors["natural_router_topk_weights"].double()
    expected_static = torch.tensor([static[layer] for layer in range(LAYERS)])
    if not torch.equal(subsets[0], expected_static):
        return False
    for boundary_index in range(1, len(boundaries)):
        start = boundaries[boundary_index - 1]
        end = boundaries[boundary_index]
        for layer in range(LAYERS):
            scores = torch.zeros(EXPERTS, dtype=torch.float64)
            scores.scatter_add_(
                0, ids[start:end, layer].reshape(-1), weights[start:end, layer].reshape(-1)
            )
            if tuple(int(value) for value in subsets[boundary_index, layer]) != _top32(scores):
                return False
    return True


def _candidate_audit(row: dict[str, Any], tensor_path: Path, config: dict[str, Any]) -> bool:
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
                audit.get("subset_residual_execution") == "natural_top8_intersection_zero_missing",
                audit.get("execution_weights_finite_nonnegative", False),
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                not audit.get("forbidden_inputs_present", True),
                audit.get("diagnostic_oracle_inputs_present") is False,
                audit.get("deployable") is True,
                cost["attention_calls"] == LAYERS,
                cost["attention_queries"] == LAYERS * HORIZON,
                cost["router_calls"] == LAYERS,
                cost["expert_calls"] == LAYERS,
                cost["lm_head_calls"] == 0,
            )
        ):
            return False
        continuation = audit.get("context_continuation") or {}
        if not all(
            (
                continuation.get("mode") == "sampled_unigram_full_continuation",
                continuation.get("anchor_one_is_sampled_next", False),
                continuation.get("copied_indices_in_known_context", False),
                continuation.get("future_token_accessed") is False,
            )
        ):
            return False
        weight_sum = float(audit["execution_weight_sum_mean"])
        captured_mean = float(audit["captured_natural_mass_mean"])
        if abs(weight_sum - captured_mean) > BFLOAT16_ABSOLUTE_TOLERANCE:
            return False
        if boundary_index == 0 and abs(captured_mean - 1) > BFLOAT16_ABSOLUTE_TOLERANCE:
            return False
    tensors = load_file(str(tensor_path))
    weights = tensors["shadow_executed_topk_weights"].float()
    captured_tensor = tensors["shadow_captured_natural_mass"].float()
    return bool(
        torch.isfinite(weights).all()
        and not (weights < 0).any()
        and torch.allclose(
            weights.sum(dim=-1),
            captured_tensor,
            rtol=0.0,
            atol=BFLOAT16_ABSOLUTE_TOLERANCE,
        )
    )


def _oracle_audit(row: dict[str, Any]) -> bool:
    return bool(
        row["prompt_capture_audit"]["production_rng_unchanged"]
        and len(row["cache_rng_audits"]) == len(row["boundaries"])
        and all(
            audit.get("production_cache_signature_unchanged", False)
            and audit.get("production_rng_unchanged", False)
            and audit.get("shadow_cache_discarded", False)
            and audit.get("future_saved_tokens_used", False)
            and audit.get("role") == "routing_information_oracle_only"
            for audit in row["cache_rng_audits"]
        )
    )


def _row_audit(
    stage: Stage,
    row: dict[str, Any],
    spec: PolicySpec,
    static: dict[int, tuple[int, ...]],
    config: dict[str, Any],
) -> bool:
    tensor_path = _paths(stage, int(row["row_index"]), spec.key)[1]
    if row.get("task_accuracy_measured") or row.get("identity_materialized"):
        return False
    if spec == PREVIOUS:
        return _previous_audit(row, tensor_path, static)
    if spec == CANDIDATE:
        return _candidate_audit(row, tensor_path, config)
    return _oracle_audit(row)


def _development_candidate_rows(samples: dict[str, Any]) -> list[dict[str, Any]]:
    source_spec = next(spec for spec in development_analysis.SPECS if spec.key == CANDIDATE.key)
    rows = []
    for reference in samples["partitions"]["development_reference"]:
        row = development_analysis._valid(reference, source_spec)
        if row is None:
            raise RuntimeError(f"missing pinned development candidate: {reference['sample_id']}")
        rows.append(row)
    return rows


def aggregate_development() -> dict[str, object]:
    config, samples = _protocol()
    references = samples["partitions"]["development_reference"]
    previous_rows = []
    static = _static_subsets(config)
    for reference in references:
        row = _valid("development_reference", reference, PREVIOUS)
        if row is None:
            raise RuntimeError(f"missing corrected previous row: {reference['sample_id']}")
        previous_rows.append(row)
    candidate_rows = _development_candidate_rows(samples)
    policies = {
        CANDIDATE.key: _aggregate_policy(candidate_rows),
        PREVIOUS.key: _aggregate_policy(previous_rows),
    }
    candidate = policies[CANDIDATE.key]
    previous = policies[PREVIOUS.key]
    route_delta = float(candidate["mean_route_hit"]) - float(previous["mean_route_hit"])
    mass_delta = float(candidate["mean_selected_mass"]) - float(previous["mean_selected_mass"])
    source_audit = _json(
        Path(config["source_development"]["artifact_manifest"]).parent
        / "development"
        / "audit.json"
    )
    previous_audit = all(
        _row_audit("development_reference", row, PREVIOUS, static, config) for row in previous_rows
    )
    gate = config["development_reference_repair"]["progress_gate"]
    progress_gate_pass = all(
        (
            route_delta >= float(gate["minimum_absolute_mean_route_hit_improvement"]),
            mass_delta >= float(gate["minimum_absolute_mean_selected_mass_improvement"]),
            float(candidate["estimated_transfer_reduction"])
            >= float(gate["minimum_estimated_transfer_reduction"]),
            source_audit.get("all_pass") is True,
            previous_audit,
        )
    )
    root = OUTPUT / "development_reference"
    write_json_atomic(root / "aggregates.json", {"policies": policies})
    write_json_atomic(
        root / "paired_bootstrap.json",
        _paired_bootstrap(
            {str(row["sample_id"]): _per_sample(row) for row in candidate_rows},
            {str(row["sample_id"]): _per_sample(row) for row in previous_rows},
            draws=10000,
            seed=int(config["strong_candidate_gate"]["bootstrap_seed"]),
        ),
    )
    write_json_atomic(
        root / "audit.json",
        {
            "all_pass": source_audit.get("all_pass") is True and previous_audit,
            "candidate_rows_checksum_reused": 4,
            "corrected_previous_rows": 4,
            "legacy_previous_rows_used": False,
            "accuracy_or_correctness_used": False,
        },
    )
    decision = {
        "decision": "PROCEED_TO_HELD_OUT" if progress_gate_pass else "STOP/PIVOT",
        "progress_gate_pass": progress_gate_pass,
        "candidate_minus_corrected_previous_route_hit": route_delta,
        "candidate_minus_corrected_previous_selected_mass": mass_delta,
    }
    write_json_atomic(root / "decision.json", decision)
    status: dict[str, object] = {"state": "complete", **decision}
    write_json_atomic(root / "pipeline_status.json", status)
    return status


def _gap_recovery(candidate: float, previous: float, oracle: float) -> float:
    gap = oracle - previous
    return (candidate - previous) / gap if gap > 0 else float("-inf")


def _strata_and_worst(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "route_hits": 0.0,
                "route_slots": 0.0,
                "selected_mass_hit": 0.0,
                "selected_mass_total": 0.0,
            }
        )
    )
    observations: list[dict[str, object]] = []
    for row in rows:
        policy = str(row["policy"])
        tensors = load_file(str(_paths("held_out_route", int(row["row_index"]), policy)[1]))
        ids = tensors["natural_router_topk_ids"]
        weights = tensors["natural_router_topk_weights"].float()
        subsets = tensors["subsets"]
        boundaries = [int(value) for value in tensors["boundaries"].tolist()]
        for boundary_index, boundary in enumerate(boundaries):
            end = (
                boundaries[boundary_index + 1]
                if boundary_index + 1 < len(boundaries)
                else ids.shape[0]
            )
            for token_index in range(boundary, end):
                anchor = token_index - boundary + 1
                for layer in range(LAYERS):
                    mask = torch.isin(ids[token_index, layer], subsets[boundary_index, layer])
                    observation: dict[str, object] = {
                        "policy": policy,
                        "sample_id": str(row["sample_id"]),
                        "boundary": boundary,
                        "anchor": anchor,
                        "layer": layer,
                        "route_hits": int(mask.sum()),
                        "route_slots": int(mask.numel()),
                        "selected_mass_hit": float(
                            weights[token_index, layer][mask].double().sum()
                        ),
                        "selected_mass_total": float(weights[token_index, layer].double().sum()),
                    }
                    observations.append(observation)
                    for axis, value in (
                        ("anchor", anchor),
                        ("layer", layer),
                        ("layer_block", layer // 8),
                    ):
                        aggregate = grouped[axis][f"{policy}/{value}"]
                        for field in aggregate:
                            aggregate[field] += float(cast(int | float, observation[field]))
    strata = {
        axis: {
            key: {
                **values,
                "mean_route_hit": values["route_hits"] / values["route_slots"],
                "mean_selected_mass": values["selected_mass_hit"] / values["selected_mass_total"],
            }
            for key, values in sorted(groups.items())
        }
        for axis, groups in grouped.items()
    }
    worst = sorted(
        observations,
        key=lambda row: (
            float(cast(float, row["selected_mass_hit"]))
            / float(cast(float, row["selected_mass_total"])),
            float(cast(int, row["route_hits"])) / float(cast(int, row["route_slots"])),
            str(row["policy"]),
            str(row["sample_id"]),
            cast(int, row["boundary"]),
            cast(int, row["anchor"]),
            cast(int, row["layer"]),
        ),
    )[:200]
    return cast(dict[str, object], strata), worst


def aggregate_held_out() -> dict[str, object]:
    config, samples = _protocol()
    static = _static_subsets(config)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit_pass = True
    for reference in samples["partitions"]["held_out_route"]:
        for spec in _specs("held_out_route"):
            row = _valid("held_out_route", reference, spec)
            if row is None:
                raise RuntimeError(f"missing held-out row: {reference['sample_id']}/{spec.key}")
            rows.append(row)
            grouped[spec.key].append(row)
            audit_pass = audit_pass and _row_audit("held_out_route", row, spec, static, config)
    policies = {key: _aggregate_policy(value) for key, value in grouped.items()}
    candidate = policies[CANDIDATE.key]
    previous = policies[PREVIOUS.key]
    oracle = policies[ORACLE.key]
    route_delta = float(candidate["mean_route_hit"]) - float(previous["mean_route_hit"])
    mass_delta = float(candidate["mean_selected_mass"]) - float(previous["mean_selected_mass"])
    route_recovery = _gap_recovery(
        float(candidate["mean_route_hit"]),
        float(previous["mean_route_hit"]),
        float(oracle["mean_route_hit"]),
    )
    mass_recovery = _gap_recovery(
        float(candidate["mean_selected_mass"]),
        float(previous["mean_selected_mass"]),
        float(oracle["mean_selected_mass"]),
    )
    gate = config["strong_candidate_gate"]
    gate_pass = all(
        (
            float(candidate["mean_route_hit"]) >= float(gate["minimum_mean_route_hit"]),
            float(candidate["mean_selected_mass"]) >= float(gate["minimum_mean_selected_mass"]),
            route_delta >= float(gate["minimum_route_hit_improvement_over_previous"]),
            mass_delta >= float(gate["minimum_selected_mass_improvement_over_previous"]),
            route_recovery >= float(gate["minimum_oracle_minus_previous_gap_recovery"]),
            mass_recovery >= float(gate["minimum_oracle_minus_previous_gap_recovery"]),
            float(candidate["estimated_transfer_reduction"])
            >= float(gate["minimum_estimated_transfer_reduction"]),
            audit_pass,
        )
    )
    root = OUTPUT / "held_out_route"
    write_json_atomic(root / "aggregates.json", {"policies": policies})
    write_json_atomic(
        root / "paired_bootstrap.json",
        {
            "candidate_vs_previous": _paired_bootstrap(
                {str(row["sample_id"]): _per_sample(row) for row in grouped[CANDIDATE.key]},
                {str(row["sample_id"]): _per_sample(row) for row in grouped[PREVIOUS.key]},
                draws=int(gate["bootstrap_samples"]),
                seed=int(gate["bootstrap_seed"]),
            ),
            "candidate_vs_oracle": _paired_bootstrap(
                {str(row["sample_id"]): _per_sample(row) for row in grouped[CANDIDATE.key]},
                {str(row["sample_id"]): _per_sample(row) for row in grouped[ORACLE.key]},
                draws=int(gate["bootstrap_samples"]),
                seed=int(gate["bootstrap_seed"]),
            ),
        },
    )
    strata, worst = _strata_and_worst(rows)
    write_json_atomic(root / "stratified_metrics.json", strata)
    write_json_atomic(root / "worst_cases.json", worst)
    write_json_atomic(
        root / "cost_report.json",
        {
            key: {
                field: values[field]
                for field in (
                    "mean_probe_latency_seconds",
                    "max_temporary_cuda_bytes",
                    "attention_calls",
                    "attention_queries",
                    "router_calls",
                    "expert_calls",
                    "mean_sample_policy_elapsed_seconds",
                )
            }
            for key, values in policies.items()
        },
    )
    write_json_atomic(
        root / "audit.json",
        {
            "all_pass": audit_pass,
            "sample_policy_rows": len(rows),
            "candidate_future_or_label_leakage": False,
            "legacy_previous_rows_used": False,
            "bfloat16_absolute_weight_tolerance": BFLOAT16_ABSOLUTE_TOLERANCE,
        },
    )
    decision = {
        "decision": (
            "CANDIDATE_FOR_SEPARATELY_FROZEN_CLOSED_LOOP_PILOT" if gate_pass else "STOP/PIVOT"
        ),
        "strong_candidate_gate_pass": gate_pass,
        "candidate_minus_previous_route_hit": route_delta,
        "candidate_minus_previous_selected_mass": mass_delta,
        "oracle_gap_recovery_route_hit": route_recovery,
        "oracle_gap_recovery_selected_mass": mass_recovery,
        "task_accuracy_authorized": False,
    }
    write_json_atomic(root / "decision.json", decision)
    status: dict[str, object] = {
        "state": "complete",
        "sample_policy_rows": len(rows),
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
    development = _json(OUTPUT / "development_reference" / "aggregates.json")["policies"]
    development_decision = _json(OUTPUT / "development_reference" / "decision.json")
    held_path = OUTPUT / "held_out_route" / "aggregates.json"
    held = _json(held_path)["policies"] if held_path.is_file() else {}
    decision_path = OUTPUT / "held_out_route" / "decision.json"
    decision = (
        _json(decision_path)
        if decision_path.is_file()
        else {
            "decision": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
            "oracle_gap_recovery_route_hit": None,
            "oracle_gap_recovery_selected_mass": None,
        }
    )
    lines = [
        "# Mass-preserving pseudo residual held-out v1",
        "",
        "The corrected static-first previous-route reference retained the development gate: "
        f"`{development_decision['progress_gate_pass']}`.",
        "",
        "| Stage/policy | Route hit | Selected mass | Transfer reduction | Probe s |",
        "|---|---:|---:|---:|---:|",
    ]
    for prefix, policies in (("development", development), ("held-out", held)):
        for key, values in policies.items():
            lines.append(
                f"| {prefix}/{key} | {float(values['mean_route_hit']):.6f} | "
                f"{float(values['mean_selected_mass']):.6f} | "
                f"{float(values['estimated_transfer_reduction']):.6f} | "
                f"{float(values['mean_probe_latency_seconds']):.4f} |"
            )
    lines.extend(
        [
            "",
            f"Held-out decision: **{decision['decision']}**. Oracle-gap recovery route/mass: "
            f"{decision['oracle_gap_recovery_route_hit']}/"
            f"{decision['oracle_gap_recovery_selected_mass']}.",
            "",
            "All route metrics are measured teacher-forced on each policy's own hard-subset "
            "state. Probe/replay cost is measured; transfer is simulated. No task accuracy, "
            "free generation, exact-token identity, NLL/perplexity, closed-loop runtime, or "
            "runtime speedup was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples = _protocol()
    development = _json(OUTPUT / "development_reference" / "pipeline_status.json")
    development_pass = bool(development.get("progress_gate_pass"))
    if development_pass:
        held = _json(OUTPUT / "held_out_route" / "pipeline_status.json")
        if held.get("state") != "complete":
            raise ValueError("held-out route stage is incomplete")
    row_paths = [path for path in OUTPUT.glob("*/samples/*/*.json") if ".FAILED." not in path.name]
    rows = [_json(path) for path in sorted(row_paths)]
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
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
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "new_sample_policy_rows_validated": len(rows),
            "development_candidate_rows_checksum_reused": 4,
            "atomic_json_tensor_pairs": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit_and_selected_mass",
                "native_probe_and_replay_latency_and_memory",
                "attention_router_and_expert_calls",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": [
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
        OUTPUT / "provenance.json",
        {
            "analysis_id": ANALYSIS_ID,
            "model": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "development_reference_sample_ids": [
                row["sample_id"] for row in samples["partitions"]["development_reference"]
            ],
            "held_out_sample_ids": [
                row["sample_id"] for row in samples["partitions"]["held_out_route"]
            ],
            "candidate_learned_or_fitted_parameters": False,
            "candidate_future_token_or_label_inputs": False,
            "default_mean_vectors_loaded": False,
            "task_accuracy_measured": False,
            "network_downloads": False,
        },
    )
    decision = (
        _json(OUTPUT / "held_out_route" / "decision.json")
        if development_pass
        else {
            "decision": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
            "strong_candidate_gate_pass": False,
            "candidate_minus_previous_route_hit": None,
            "candidate_minus_previous_selected_mass": None,
            "oracle_gap_recovery_route_hit": None,
            "oracle_gap_recovery_selected_mass": None,
        }
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "analysis_id": ANALYSIS_ID,
            "decision": "NARROW" if decision["strong_candidate_gate_pass"] else "STOP/PIVOT",
            "scope": "Qwen_GSM8K_H8_B32_eight_row_held_out_route_only",
            "held_out_gate_decision": decision["decision"],
            "strong_candidate_gate_pass": decision["strong_candidate_gate_pass"],
            "candidate_minus_previous_route_hit": decision["candidate_minus_previous_route_hit"],
            "candidate_minus_previous_selected_mass": decision[
                "candidate_minus_previous_selected_mass"
            ],
            "oracle_gap_recovery_route_hit": decision["oracle_gap_recovery_route_hit"],
            "oracle_gap_recovery_selected_mass": decision["oracle_gap_recovery_selected_mass"],
            "task_accuracy_authorized": False,
            "runtime_speedup_claim": False,
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report())
    manifest = _manifest()
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "decision": "NARROW" if decision["strong_candidate_gate_pass"] else "STOP/PIVOT",
        "strong_candidate_gate_pass": decision["strong_candidate_gate_pass"],
        "development_progress_gate_pass": development_pass,
        "new_sample_policy_rows": len(rows),
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": False,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "report_v1":
        raise ValueError("held-out mass-residual pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("held-out mass-residual artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"held-out artifact checksum changed: {path}")
    resume_rows = _json(OUTPUT / "resume_audit.json")["new_sample_policy_rows_validated"]
    if resume_rows != status["new_sample_policy_rows"] or resume_rows not in {4, 28}:
        raise ValueError("held-out mass-residual row count changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "strong_candidate_gate_pass": status["strong_candidate_gate_pass"],
        "new_sample_policy_rows": status["new_sample_policy_rows"],
        "artifacts": manifest["artifact_count"],
    }


def all_stages(*, physical_gpu: int) -> dict[str, object]:
    run_stage("development_reference", physical_gpu=physical_gpu, shard_index=0, shard_count=1)
    development = aggregate_development()
    if not development["progress_gate_pass"]:
        return finalize()
    run_stage("held_out_route", physical_gpu=physical_gpu, shard_index=0, shard_count=1)
    aggregate_held_out()
    return finalize()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "run-development-reference",
            "aggregate-development-reference",
            "run-held-out",
            "aggregate-held-out",
            "finalize",
            "validate",
            "all",
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    started = time.time()
    if args.command == "run-development-reference":
        result = run_stage(
            "development_reference",
            physical_gpu=args.gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.command == "aggregate-development-reference":
        result = aggregate_development()
    elif args.command == "run-held-out":
        result = run_stage(
            "held_out_route",
            physical_gpu=args.gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.command == "aggregate-held-out":
        result = aggregate_held_out()
    elif args.command == "finalize":
        result = finalize()
    elif args.command == "validate":
        result = validate()
    else:
        result = all_stages(physical_gpu=args.gpu)
    print(json.dumps({**result, "command_elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
