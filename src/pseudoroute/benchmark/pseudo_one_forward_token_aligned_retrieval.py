"""Run the frozen one-forward token-aligned state-retrieval analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from safetensors.torch import load_file

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    PolicySpec,
    StateRetrievalMode,
    _aggregate_policy,
    _paired_bootstrap,
    _per_sample,
    _save_tensors_atomic,
    run_policy_sample,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    CONFIG_SHA256 as SOURCE_CONFIG_SHA256,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    SAMPLES_SHA256 as SOURCE_SAMPLES_SHA256,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    SPECS as SOURCE_SPECS,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    _protocol as _source_protocol,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    _valid as _source_valid,
)
from pseudoroute.benchmark.qwen_pseudo import StateRetrievalMix, StateRetrievalTarget
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_one_forward_token_aligned_retrieval_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "19a368fe6a5e819cc0e5b3a6c3528ccf9baf94a66ba4733a278e9c0d1e5cadff"
SAMPLES_SHA256 = "dc18107f178100f41df1ffccdf23f07ddd3f611044f4b7f7ac5da1edbad36c69"
LAYERS = 48
EXPERTS = 128
TOP_K = 8
HORIZON = 8
MINIMUM_TRANSFER_REDUCTION = 0.30

ProtocolTarget = Literal["router_input", "fresh_moe_residual"]


@dataclass(frozen=True)
class RetrievalSpec:
    key: str
    retrieval_mode: StateRetrievalMode
    target: ProtocolTarget
    mixing: StateRetrievalMix

    @property
    def probe_target(self) -> StateRetrievalTarget:
        return "moe_residual" if self.target == "fresh_moe_residual" else "router_input"

    def policy(self) -> PolicySpec:
        return PolicySpec(
            self.key,
            "pseudo",
            residual="zero",
            content="recent_sequence_causal",
            selection="first_four_anchor_core_plus_history_fill",
        )

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


SPECS = (
    RetrievalSpec(
        "exact_recent_residual_additive_norm",
        "exact_token_most_recent",
        "fresh_moe_residual",
        "additive_norm",
    ),
    RetrievalSpec(
        "exact_recent_residual_replace_norm",
        "exact_token_most_recent",
        "fresh_moe_residual",
        "replace_norm",
    ),
    RetrievalSpec(
        "exact_then_embedding_nearest_residual_additive_norm",
        "exact_token_then_embedding_nearest",
        "fresh_moe_residual",
        "additive_norm",
    ),
    RetrievalSpec(
        "exact_recent_router_input_additive_norm",
        "exact_token_most_recent",
        "router_input",
        "additive_norm",
    ),
    RetrievalSpec(
        "exact_then_embedding_nearest_router_input_additive_norm",
        "exact_token_then_embedding_nearest",
        "router_input",
        "additive_norm",
    ),
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("token-aligned retrieval protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("token-aligned retrieval protocol identity changed")
    source = config["source"]
    if (
        sha256_file(Path(source["baseline_config"])) != SOURCE_CONFIG_SHA256
        or source["baseline_config_sha256"] != SOURCE_CONFIG_SHA256
    ):
        raise ValueError("token-aligned retrieval source config changed")
    if (
        sha256_file(Path(source["baseline_sample_manifest"])) != SOURCE_SAMPLES_SHA256
        or source["baseline_sample_manifest_sha256"] != SOURCE_SAMPLES_SHA256
    ):
        raise ValueError("token-aligned retrieval source samples changed")
    source_manifest = Path(source["baseline_artifact_root"]) / "artifact_manifest.json"
    if sha256_file(source_manifest) != source["baseline_artifact_manifest_sha256"]:
        raise ValueError("token-aligned retrieval source artifact changed")
    source_config, source_samples, composition = _source_protocol()
    if source_samples["partitions"]["development"] != samples["partitions"]["development"]:
        raise ValueError("token-aligned retrieval development IDs changed")
    if source_config["analysis_id"] != source["baseline_analysis_id"]:
        raise ValueError("token-aligned retrieval source identity changed")
    configured = {row["key"]: row for row in config["variants"]}
    for spec in SPECS:
        row = configured.get(spec.key)
        if row is None:
            raise ValueError(f"missing frozen retrieval spec: {spec.key}")
        if (
            row["retrieval"] != spec.retrieval_mode
            or row["target"] != spec.target
            or row["mixing"] != spec.mixing
        ):
            raise ValueError(f"frozen retrieval spec changed: {spec.key}")
    return config, samples, composition


def _paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "development" / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid(reference: dict[str, Any], spec: RetrievalSpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial token-aligned retrieval row: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("retrieval_spec_fingerprint") != spec.fingerprint()
        or row.get("state_retrieval_mode") != spec.retrieval_mode
        or row.get("state_retrieval_target") != spec.probe_target
        or row.get("state_retrieval_mix") != spec.mixing
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible token-aligned retrieval row: {json_path}")
    return row


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(*, physical_gpu: int, shard_index: int, shard_count: int) -> dict[str, object]:
    config, samples, composition = _protocol()
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid token-aligned retrieval shard")
    references = [
        reference
        for index, reference in enumerate(samples["partitions"]["development"])
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
    suite = load_pseudo_embedding_config(composition["source_suite"]["config"])
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
                state_retrieval_mode=spec.retrieval_mode,
                state_retrieval_target=spec.probe_target,
                state_retrieval_mix=spec.mixing,
            )
            row["retrieval_spec"] = asdict(spec)
            row["retrieval_spec_fingerprint"] = spec.fingerprint()
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


def _source_rows_checked(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for reference in references:
        row = _source_valid(reference, SOURCE_SPECS[0])
        if row is None:
            raise RuntimeError(f"missing uncorrected source baseline: {reference['sample_id']}")
        rows.append(row)
    return rows


def _tensor_path(row: dict[str, Any]) -> Path:
    if row["analysis_id"] == ANALYSIS_ID:
        return _paths(int(row["row_index"]), str(row["policy"]))[1]
    root = Path("artifacts/pseudo_one_forward_state_correction_v1/development/samples")
    return root / f"{int(row['row_index']):05d}" / f"{row['policy']}.safetensors"


def _anchor_strata(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    totals: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: {anchor: [0.0, 0.0, 0.0, 0.0] for anchor in range(1, HORIZON + 1)}
    )
    for row in rows:
        tensors = load_file(str(_tensor_path(row)))
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
                for layer in range(ids.shape[1]):
                    allowed = torch.isin(ids[token_index, layer], subsets[boundary_index, layer])
                    value = totals[str(row["policy"])][anchor]
                    value[0] += float(allowed.sum())
                    value[1] += float(allowed.numel())
                    value[2] += float(weights[token_index, layer][allowed].double().sum())
                    value[3] += float(weights[token_index, layer].double().sum())
    return {
        policy: {
            str(anchor): {
                "mean_route_hit": value[0] / value[1],
                "mean_selected_mass": value[2] / value[3],
                "route_slots": int(value[1]),
            }
            for anchor, value in anchors.items()
        }
        for policy, anchors in totals.items()
    }


def _retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    costs = [cost for row in rows for cost in row["retrieval_costs"]]
    kinds = [kind for cost in costs for kind in cost["match_kinds"]]
    similarities = [float(value) for cost in costs for value in cost["similarities"]]
    paired_latencies = [
        float(retrieval["latency_seconds_measured"]) + float(probe_cost["latency_seconds_measured"])
        for row in rows
        for retrieval, probe_cost in zip(row["retrieval_costs"], row["probe_costs"], strict=True)
    ]
    return {
        "retrieval_anchors": len(kinds),
        "retrieval_exact_fraction": kinds.count("exact_token_most_recent") / len(kinds),
        "retrieval_fallback_fraction": (
            kinds.count("embedding_nearest_most_recent_tie") / len(kinds)
        ),
        "retrieval_missing_fraction": kinds.count("missing_exact_zero_weight") / len(kinds),
        "mean_similarity": sum(similarities) / len(similarities),
        "mean_retrieval_latency_seconds": sum(
            float(cost["latency_seconds_measured"]) for cost in costs
        )
        / len(costs),
        "mean_total_planning_latency_seconds": sum(paired_latencies) / len(paired_latencies),
        "max_retrieval_temporary_cuda_bytes": max(
            int(cost["temporary_cuda_bytes_measured"]) for cost in costs
        ),
        "max_history_state_bank_bytes": max(
            int(cost["history_state_bank_bytes"]) for cost in costs
        ),
    }


def _audit_pass(row: dict[str, Any], spec: RetrievalSpec, config: dict[str, Any]) -> bool:
    prompt = row["prompt_capture_audit"]
    if not prompt["production_rng_unchanged"] or not prompt["full_prompt_tokens"]:
        return False
    expected = config["audits"]
    if not (len(row["probe_costs"]) == len(row["retrieval_costs"]) == len(row["cache_rng_audits"])):
        return False
    for cost, retrieval, audit in zip(
        row["probe_costs"],
        row["retrieval_costs"],
        row["cache_rng_audits"],
        strict=True,
    ):
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                audit.get("one_causal_forward_per_boundary", False),
                audit.get("shadow_expert_execution", False),
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                audit.get("retrieval_history_length_matches_realized_context", False),
                audit.get("retrieved_indices_precede_boundary", False),
                audit.get("retrieval_state_bank_policy_local", False),
                audit.get("retrieval_state_finite", False),
                not audit.get("forbidden_inputs_present", True),
                audit.get("state_retrieval_target") == spec.probe_target,
                audit.get("state_retrieval_mix") == spec.mixing,
                retrieval.get("retrieved_indices_precede_boundary", False),
                all(0 <= float(value) <= 1 for value in retrieval["similarities"]),
                cost["attention_calls"] == expected["expected_attention_calls_per_boundary"],
                cost["attention_queries"] == expected["expected_attention_queries_per_boundary"],
                cost["router_calls"] == expected["expected_router_calls_per_boundary"],
                cost["expert_calls"] == expected["expected_expert_calls_per_boundary"],
            )
        ):
            return False
    return True


def _retrieval_strata(rows: list[dict[str, Any]]) -> dict[str, object]:
    kinds: dict[str, list[float]] = defaultdict(list)
    anchors: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        for cost in row["retrieval_costs"]:
            for anchor, (kind, similarity) in enumerate(
                zip(cost["match_kinds"], cost["similarities"], strict=True), start=1
            ):
                kinds[str(kind)].append(float(similarity))
                anchors[anchor].append((str(kind), float(similarity)))
    return {
        "by_match_kind": {
            kind: {
                "anchors": len(values),
                "mean_similarity": sum(values) / len(values),
            }
            for kind, values in sorted(kinds.items())
        },
        "by_anchor": {
            str(anchor): {
                "observations": len(values),
                "exact_fraction": sum(kind == "exact_token_most_recent" for kind, _ in values)
                / len(values),
                "mean_similarity": sum(value for _, value in values) / len(values),
            }
            for anchor, values in sorted(anchors.items())
        },
    }


def aggregate() -> dict[str, object]:
    config, samples, _composition = _protocol()
    references = samples["partitions"]["development"]
    source_rows = _source_rows_checked(references)
    candidate_rows: list[dict[str, Any]] = []
    for reference in references:
        for spec in SPECS:
            row = _valid(reference, spec)
            if row is None:
                raise RuntimeError(f"missing retrieval row: {reference['sample_id']}/{spec.key}")
            candidate_rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_key = SOURCE_SPECS[0].key
    grouped[baseline_key] = source_rows
    for row in candidate_rows:
        grouped[str(row["policy"])].append(row)
    policies: dict[str, dict[str, Any]] = {
        policy: dict(_aggregate_policy(rows)) for policy, rows in sorted(grouped.items())
    }
    for spec in SPECS:
        policies[spec.key].update(_retrieval_summary(grouped[spec.key]))
    anchors = _anchor_strata([*source_rows, *candidate_rows])
    baseline = policies[baseline_key]
    baseline_anchor = anchors[baseline_key]["1"]
    comparisons: dict[str, object] = {}
    signals: list[str] = []
    all_audits = all(
        _audit_pass(row, next(spec for spec in SPECS if spec.key == row["policy"]), config)
        for row in candidate_rows
    )
    per_sample = {
        policy: {str(row["sample_id"]): _per_sample(row) for row in rows}
        for policy, rows in grouped.items()
    }
    worst_cases: dict[str, object] = {}
    for spec in SPECS:
        candidate = policies[spec.key]
        candidate_anchor = anchors[spec.key]["1"]
        route_delta = float(candidate["mean_route_hit"]) - float(baseline["mean_route_hit"])
        mass_delta = float(candidate["mean_selected_mass"]) - float(baseline["mean_selected_mass"])
        anchor_route_delta = float(candidate_anchor["mean_route_hit"]) - float(
            baseline_anchor["mean_route_hit"]
        )
        anchor_mass_delta = float(candidate_anchor["mean_selected_mass"]) - float(
            baseline_anchor["mean_selected_mass"]
        )
        signal = (
            route_delta
            >= float(
                config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"]
            )
            and mass_delta
            >= float(
                config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"]
            )
            and anchor_route_delta
            >= -float(config["analysis"]["maximum_anchor_one_absolute_regression"])
            and anchor_mass_delta
            >= -float(config["analysis"]["maximum_anchor_one_absolute_regression"])
            and float(candidate["estimated_transfer_reduction"]) >= MINIMUM_TRANSFER_REDUCTION
            and float(candidate["mean_total_planning_latency_seconds"])
            <= float(config["analysis"]["maximum_mean_total_planning_latency_seconds"])
            and all_audits
        )
        if signal:
            signals.append(spec.key)
        comparisons[spec.key] = {
            "mean_route_hit_delta": route_delta,
            "mean_selected_mass_delta": mass_delta,
            "anchor_one_route_hit_delta": anchor_route_delta,
            "anchor_one_selected_mass_delta": anchor_mass_delta,
            "estimated_transfer_reduction_delta": float(candidate["estimated_transfer_reduction"])
            - float(baseline["estimated_transfer_reduction"]),
            "paired_bootstrap": _paired_bootstrap(
                per_sample[spec.key],
                per_sample[baseline_key],
                draws=int(config["analysis"]["bootstrap_samples"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
            "route_signal": signal,
        }
        deltas = sorted(
            (
                {
                    "sample_id": sample_id,
                    "route_hit_delta": values[0] - per_sample[baseline_key][sample_id][0],
                    "selected_mass_delta": values[1] - per_sample[baseline_key][sample_id][1],
                }
                for sample_id, values in per_sample[spec.key].items()
            ),
            key=lambda row: (row["selected_mass_delta"], row["route_hit_delta"], row["sample_id"]),
        )
        worst_cases[spec.key] = deltas
    ranked = sorted(
        (spec.key for spec in SPECS),
        key=lambda key: (
            -float(policies[key]["mean_selected_mass"]),
            -float(policies[key]["mean_route_hit"]),
            float(policies[key]["mean_total_planning_latency_seconds"]),
            key,
        ),
    )
    root = OUTPUT / "development"
    write_json_atomic(root / "aggregates.json", {"analysis_id": ANALYSIS_ID, "policies": policies})
    write_json_atomic(root / "comparisons.json", comparisons)
    write_json_atomic(root / "anchor_strata.json", anchors)
    write_json_atomic(root / "retrieval_strata.json", _retrieval_strata(candidate_rows))
    write_json_atomic(root / "per_sample.json", per_sample)
    write_json_atomic(root / "worst_cases.json", worst_cases)
    write_json_atomic(
        root / "cost_report.json",
        {
            spec.key: {
                key: policies[spec.key][key]
                for key in (
                    "mean_probe_latency_seconds",
                    "mean_retrieval_latency_seconds",
                    "mean_total_planning_latency_seconds",
                    "max_temporary_cuda_bytes",
                    "max_retrieval_temporary_cuda_bytes",
                    "max_history_state_bank_bytes",
                    "attention_queries",
                    "attention_calls",
                    "router_calls",
                    "expert_calls",
                )
            }
            for spec in SPECS
        },
    )
    audit = {
        "all_pass": all_audits,
        "candidate_sample_policy_rows": len(candidate_rows),
        "source_reference_rows": len(source_rows),
        "one_causal_forward_per_boundary": True,
        "policy_local_pre_boundary_state_only": True,
        "accuracy_or_correctness_used": False,
    }
    write_json_atomic(root / "audit.json", audit)
    decision = {
        "analysis_id": ANALYSIS_ID,
        "decision": "DEVELOPMENT_ROUTE_SIGNAL" if signals else "STOP/PIVOT",
        "ranked_candidate_variants": ranked,
        "route_signal_variants": signals,
        "selected_route_signal_variant": next((key for key in ranked if key in signals), None),
        "held_out_route_authorized": False,
        "task_accuracy_authorized": False,
    }
    write_json_atomic(root / "decision.json", decision)
    status = {
        "state": "complete",
        "candidate_sample_policy_rows": len(candidate_rows),
        "source_reference_rows": len(source_rows),
        "all_audits_pass": all_audits,
        "decision": decision["decision"],
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
        if relative in excluded:
            continue
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
    policies = _json(OUTPUT / "development" / "aggregates.json")["policies"]
    comparisons = _json(OUTPUT / "development" / "comparisons.json")
    anchors = _json(OUTPUT / "development" / "anchor_strata.json")
    decision = _json(OUTPUT / "development" / "decision.json")
    baseline_key = SOURCE_SPECS[0].key
    lines = [
        "# One-forward token-aligned state retrieval v1",
        "",
        "Each candidate uses one native causal H=8 pseudo traversal and retrieves only "
        "same-policy prompt or already-realized token states.",
        "",
        "| Variant | Route hit | Selected mass | Anchor-1 hit | Anchor-1 mass | Total plan s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in (baseline_key, *(spec.key for spec in SPECS)):
        values = policies[key]
        anchor = anchors[key]["1"]
        total_latency = values.get(
            "mean_total_planning_latency_seconds",
            values["mean_probe_latency_seconds"],
        )
        lines.append(
            f"| {key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(anchor['mean_route_hit']):.6f} | "
            f"{float(anchor['mean_selected_mass']):.6f} | "
            f"{float(total_latency):.4f} |"
        )
    lines.extend(["", "## Comparisons to checksum-pinned uncorrected v1", ""])
    for spec in SPECS:
        values = comparisons[spec.key]
        lines.append(
            f"- {spec.key}: hit/mass delta "
            f"{float(values['mean_route_hit_delta']):+.6f}/"
            f"{float(values['mean_selected_mass_delta']):+.6f}; anchor-1 delta "
            f"{float(values['anchor_one_route_hit_delta']):+.6f}/"
            f"{float(values['anchor_one_selected_mass_delta']):+.6f}; signal "
            f"`{values['route_signal']}`."
        )
    lines.extend(
        [
            "",
            f"Development-only decision: **{decision['decision']}**. Selected route signal: "
            f"`{decision['selected_route_signal_variant']}`.",
            "",
            "Route values are teacher-forced on each policy's own hard-subset state. Probe and "
            "retrieval costs are measured; transfer is simulated. No held-out route, task "
            "accuracy, free generation, exact-token identity, closed-loop runtime, or runtime "
            "speedup was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples, _composition = _protocol()
    status = _json(OUTPUT / "development" / "pipeline_status.json")
    if status.get("state") != "complete" or not status.get("all_audits_pass"):
        raise ValueError("token-aligned retrieval development aggregate is incomplete")
    row_paths = sorted(
        path
        for path in (OUTPUT / "development" / "samples").glob("*/*.json")
        if ".FAILED." not in path.name
    )
    rows = [_json(path) for path in row_paths]
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
            "source_reference_manifest_sha256": config["source"][
                "baseline_artifact_manifest_sha256"
            ],
            "pid": os.getpid(),
            "ppid": os.getppid(),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit",
                "teacher_forced_policy_state_selected_mass",
                "native_one_forward_probe_latency_and_memory",
                "token_retrieval_latency_and_memory",
                "attention_router_expert_calls",
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
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "candidate_sample_policy_rows_validated": len(rows),
            "source_reference_rows_validated": 4,
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
            "source_reference_analysis": config["source"]["baseline_analysis_id"],
            "learned_or_fitted_parameters": False,
            "default_vector_values_loaded": False,
            "task_accuracy_measured": False,
            "future_token_or_answer_access": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "analysis_id": ANALYSIS_ID,
            "decision": decision["decision"],
            "selected_route_signal_variant": decision["selected_route_signal_variant"],
            "scope": "Qwen_GSM8K_H8_B32_one_forward_token_aligned_development",
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
        "candidate_sample_policy_rows": len(rows),
        "source_reference_rows": 4,
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
        raise ValueError("token-aligned retrieval pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("token-aligned retrieval artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"token-aligned retrieval checksum changed: {path}")
    resume = _json(OUTPUT / "resume_audit.json")
    if resume["candidate_sample_policy_rows_validated"] != 20:
        raise ValueError("token-aligned retrieval candidate row count changed")
    if resume["source_reference_rows_validated"] != 4:
        raise ValueError("token-aligned retrieval source row count changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "candidate_sample_policy_rows": status["candidate_sample_policy_rows"],
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate", "finalize", "validate"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.command == "run":
        result = run(
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
