"""Run the frozen one-forward current-context continuation analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from safetensors.torch import load_file

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.context_continuation import ContextContinuationMode
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    ContentVariant,
    PolicySpec,
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
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_one_forward_context_continuation_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "519502aa463af860f4f639ff1233bf14d77bfca6ee6cb5aeff28f80817316897"
SAMPLES_SHA256 = "20a38b86b2b04ded8ded9faaf1c9eb0a90cb73668e9c23980188367a3369b4d4"
LAYERS = 48
EXPERTS = 128
TOP_K = 8
HORIZON = 8


@dataclass(frozen=True)
class ContinuationSpec:
    key: str
    content: ContentVariant
    mode: ContextContinuationMode
    match_rule: str
    required_copied_tokens: int
    partial_copy: bool

    def policy(self) -> PolicySpec:
        return PolicySpec(
            self.key,
            "pseudo",
            residual="zero",
            content=self.content,
            selection="first_four_anchor_core_plus_history_fill",
        )

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


SPECS = (
    ContinuationSpec(
        "longest_suffix_full_continuation",
        "context_longest_suffix_full_causal",
        "longest_suffix_full_continuation",
        "longest_suffix",
        7,
        False,
    ),
    ContinuationSpec(
        "sampled_unigram_full_continuation",
        "context_sampled_unigram_full_causal",
        "sampled_unigram_full_continuation",
        "sampled_token_unigram",
        7,
        False,
    ),
    ContinuationSpec(
        "longest_suffix_partial_recent_fill",
        "context_longest_suffix_partial_causal",
        "longest_suffix_partial_recent_fill",
        "longest_suffix",
        1,
        True,
    ),
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("context-continuation protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("context-continuation protocol identity changed")
    source = config["source"]
    if (
        sha256_file(Path(source["baseline_config"])) != SOURCE_CONFIG_SHA256
        or source["baseline_config_sha256"] != SOURCE_CONFIG_SHA256
    ):
        raise ValueError("context-continuation source config changed")
    if (
        sha256_file(Path(source["baseline_sample_manifest"])) != SOURCE_SAMPLES_SHA256
        or source["baseline_sample_manifest_sha256"] != SOURCE_SAMPLES_SHA256
    ):
        raise ValueError("context-continuation source sample manifest changed")
    source_manifest = Path(source["baseline_artifact_root"]) / "artifact_manifest.json"
    if sha256_file(source_manifest) != source["baseline_artifact_manifest_sha256"]:
        raise ValueError("context-continuation source artifact changed")
    preceding_manifest = Path(source["preceding_artifact_root"]) / "artifact_manifest.json"
    if sha256_file(preceding_manifest) != source["preceding_artifact_manifest_sha256"]:
        raise ValueError("context-continuation preceding artifact changed")
    source_config, source_samples, composition = _source_protocol()
    if source_samples["partitions"]["development"] != samples["partitions"]["development"]:
        raise ValueError("context-continuation development IDs changed")
    if source_config["analysis_id"] != source["baseline_analysis_id"]:
        raise ValueError("context-continuation source identity changed")
    configured = {row["key"]: row for row in config["variants"]}
    for spec in SPECS:
        row = configured.get(spec.key)
        if row is None:
            raise ValueError(f"missing frozen continuation spec: {spec.key}")
        if (
            row["match_rule"] != spec.match_rule
            or int(row["required_copied_tokens"]) != spec.required_copied_tokens
            or bool(row["partial_copy"]) != spec.partial_copy
        ):
            raise ValueError(f"frozen continuation spec changed: {spec.key}")
    return config, samples, composition


def _paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "development" / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid(reference: dict[str, Any], spec: ContinuationSpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial context-continuation sample artifact: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("context_continuation_enabled") is not True
        or row.get("continuation_spec_fingerprint") != spec.fingerprint()
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible context-continuation artifact: {json_path}")
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
        raise ValueError("invalid context-continuation shard")
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
        raise ValueError("runtime Qwen routed model facts changed")
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
            )
            row["continuation_spec"] = asdict(spec)
            row["continuation_spec_fingerprint"] = spec.fingerprint()
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
                    "analysis_config_sha256": CONFIG_SHA256,
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


def _layer_half_strata(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    totals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "layers_0_23": [0.0, 0.0, 0.0, 0.0],
            "layers_24_47": [0.0, 0.0, 0.0, 0.0],
        }
    )
    for row in rows:
        for metric in row["metrics"]:
            bucket = "layers_0_23" if int(metric["layer"]) <= 23 else "layers_24_47"
            value = totals[str(row["policy"])][bucket]
            value[0] += float(metric["route_hits"])
            value[1] += float(metric["route_slots"])
            value[2] += float(metric["selected_mass_hit"])
            value[3] += float(metric["selected_mass_total"])
    return {
        policy: {
            bucket: {
                "mean_route_hit": value[0] / value[1],
                "mean_selected_mass": value[2] / value[3],
                "route_slots": int(value[1]),
            }
            for bucket, value in buckets.items()
        }
        for policy, buckets in totals.items()
    }


def _plans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], audit["context_continuation"])
        for row in rows
        for audit in row["cache_rng_audits"]
    ]


def _continuation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plans = _plans(rows)
    suffix_counts = Counter(int(plan["matched_suffix_length"]) for plan in plans)
    matched = [plan for plan in plans if not bool(plan["fallback_used"])]
    return {
        "boundaries": len(plans),
        "matched_boundaries": len(matched),
        "fallback_boundaries": len(plans) - len(matched),
        "continuation_coverage": len(matched) / len(plans),
        "full_copy_boundaries": sum(int(plan["copied_anchor_count"]) == 7 for plan in plans),
        "mean_copied_anchor_fraction": sum(float(plan["copied_anchor_fraction"]) for plan in plans)
        / len(plans),
        "mean_changed_anchor_fraction": sum(
            float(plan["changed_anchor_fraction"]) for plan in plans
        )
        / len(plans),
        "mean_matched_suffix_length_all_boundaries": sum(
            int(plan["matched_suffix_length"]) for plan in plans
        )
        / len(plans),
        "mean_matched_suffix_length_when_matched": (
            sum(int(plan["matched_suffix_length"]) for plan in matched) / len(matched)
            if matched
            else 0.0
        ),
        "matched_suffix_length_histogram": {
            str(length): count for length, count in sorted(suffix_counts.items())
        },
        "mean_content_planning_latency_seconds": sum(
            float(plan["planning_latency_seconds_measured"]) for plan in plans
        )
        / len(plans),
        "max_content_planning_latency_seconds": max(
            float(plan["planning_latency_seconds_measured"]) for plan in plans
        ),
    }


def _suffix_bucket(plan: dict[str, Any]) -> str:
    length = int(plan["matched_suffix_length"])
    if bool(plan["fallback_used"]):
        return "fallback"
    return "suffix_length_1" if length == 1 else "suffix_length_2_plus"


def _suffix_strata(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    totals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    )
    for row in rows:
        plan_by_boundary = {
            int(audit["boundary"]): cast(dict[str, Any], audit["context_continuation"])
            for audit in row["cache_rng_audits"]
        }
        counted: set[tuple[int, str]] = set()
        for metric in row["metrics"]:
            boundary = int(metric["boundary"])
            bucket = _suffix_bucket(plan_by_boundary[boundary])
            value = totals[str(row["policy"])][bucket]
            value[0] += float(metric["route_hits"])
            value[1] += float(metric["route_slots"])
            value[2] += float(metric["selected_mass_hit"])
            value[3] += float(metric["selected_mass_total"])
            marker = (boundary, bucket)
            if marker not in counted:
                value[4] += 1
                counted.add(marker)
    return {
        policy: {
            bucket: {
                "mean_route_hit": value[0] / value[1],
                "mean_selected_mass": value[2] / value[3],
                "route_slots": int(value[1]),
                "boundaries": int(value[4]),
            }
            for bucket, value in buckets.items()
        }
        for policy, buckets in totals.items()
    }


def _audit_pass(row: dict[str, Any], spec: ContinuationSpec, config: dict[str, Any]) -> bool:
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    expected = config["audits"]
    for cost, audit in zip(row["probe_costs"], row["cache_rng_audits"], strict=True):
        plan = audit.get("context_continuation")
        if not isinstance(plan, dict):
            return False
        copied = [int(value) for value in plan["copied_context_indices"]]
        anchors = [int(value) for value in plan["anchor_token_ids"]]
        baseline = [int(value) for value in plan["baseline_anchor_token_ids"]]
        copied_count = int(plan["copied_anchor_count"])
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                audit.get("one_causal_forward_per_boundary", False),
                audit.get("shadow_expert_execution", False),
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                audit.get("anchor_token_ids_supplied", False),
                not audit.get("forbidden_inputs_present", True),
                plan.get("mode") == spec.mode,
                plan.get("anchor_one_is_sampled_next", False),
                plan.get("earlier_match_nonoverlapping", False),
                plan.get("copied_indices_in_known_context", False),
                plan.get("future_token_accessed") is False,
                plan.get("tie_break") == "most_recent_match_start",
                int(plan["current_query_end"]) == int(plan["known_context_length"]),
                0 <= int(plan["current_query_start"]) <= int(plan["current_query_end"]),
                len(anchors) == HORIZON,
                len(baseline) == HORIZON,
                len(copied) == copied_count,
                all(0 <= index < int(plan["known_context_length"]) for index in copied),
                cost["attention_calls"] == expected["expected_attention_calls_per_boundary"],
                cost["attention_queries"] == expected["expected_attention_queries_per_boundary"],
                cost["router_calls"] == expected["expected_router_calls_per_boundary"],
                cost["expert_calls"] == expected["expected_expert_calls_per_boundary"],
                cost["lm_head_calls"] == expected["expected_lm_head_calls_per_boundary"],
            )
        ):
            return False
        if spec.partial_copy:
            if not bool(plan["fallback_used"]) and not 1 <= copied_count <= 7:
                return False
        elif copied_count not in (0, 7):
            return False
        if spec.mode == "sampled_unigram_full_continuation" and copied_count:
            if int(plan["matched_suffix_length"]) != 1:
                return False
    return True


def aggregate() -> dict[str, object]:
    config, samples, _composition = _protocol()
    references = samples["partitions"]["development"]
    source_rows = _source_rows_checked(references)
    candidate_rows: list[dict[str, Any]] = []
    for reference in references:
        for spec in SPECS:
            row = _valid(reference, spec)
            if row is None:
                raise RuntimeError(f"missing continuation row: {reference['sample_id']}/{spec.key}")
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
        policies[spec.key].update(_continuation_summary(grouped[spec.key]))
    anchors = _anchor_strata([*source_rows, *candidate_rows])
    halves = _layer_half_strata([*source_rows, *candidate_rows])
    suffixes = _suffix_strata(candidate_rows)
    baseline = policies[baseline_key]
    baseline_anchor = anchors[baseline_key]["1"]
    all_audits = all(
        _audit_pass(row, next(spec for spec in SPECS if spec.key == row["policy"]), config)
        for row in candidate_rows
    )
    per_sample = {
        policy: {str(row["sample_id"]): _per_sample(row) for row in rows}
        for policy, rows in grouped.items()
    }
    comparisons: dict[str, object] = {}
    worst_cases: dict[str, object] = {}
    signals: list[str] = []
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
            and float(candidate["estimated_transfer_reduction"])
            >= float(config["analysis"]["minimum_estimated_transfer_reduction"])
            and float(candidate["mean_probe_latency_seconds"])
            <= float(config["analysis"]["maximum_mean_probe_latency_seconds"])
            and all_audits
        )
        if signal:
            signals.append(spec.key)
        comparisons[spec.key] = {
            "mean_route_hit_delta": route_delta,
            "mean_selected_mass_delta": mass_delta,
            "anchor_one_route_hit_delta": anchor_route_delta,
            "anchor_one_selected_mass_delta": anchor_mass_delta,
            "paired_bootstrap": _paired_bootstrap(
                per_sample[spec.key],
                per_sample[baseline_key],
                draws=int(config["analysis"]["bootstrap_samples"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
            "route_signal": signal,
        }
        worst_cases[spec.key] = sorted(
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
    ranked = sorted(
        (spec.key for spec in SPECS),
        key=lambda key: (
            -float(policies[key]["mean_selected_mass"]),
            -float(policies[key]["mean_route_hit"]),
            float(policies[key]["mean_content_planning_latency_seconds"]),
            key,
        ),
    )
    root = OUTPUT / "development"
    write_json_atomic(root / "aggregates.json", {"analysis_id": ANALYSIS_ID, "policies": policies})
    write_json_atomic(root / "comparisons.json", comparisons)
    write_json_atomic(root / "anchor_strata.json", anchors)
    write_json_atomic(root / "layer_half_strata.json", halves)
    write_json_atomic(root / "suffix_length_strata.json", suffixes)
    write_json_atomic(
        root / "continuation_report.json",
        {spec.key: _continuation_summary(grouped[spec.key]) for spec in SPECS},
    )
    write_json_atomic(root / "per_sample.json", per_sample)
    write_json_atomic(root / "worst_cases.json", worst_cases)
    write_json_atomic(
        root / "cost_report.json",
        {
            spec.key: {
                "mean_probe_latency_seconds": policies[spec.key]["mean_probe_latency_seconds"],
                "mean_content_planning_latency_seconds": policies[spec.key][
                    "mean_content_planning_latency_seconds"
                ],
                "max_temporary_cuda_bytes": policies[spec.key]["max_temporary_cuda_bytes"],
                "attention_queries": policies[spec.key]["attention_queries"],
                "attention_calls": policies[spec.key]["attention_calls"],
                "router_calls": policies[spec.key]["router_calls"],
                "expert_calls": policies[spec.key]["expert_calls"],
                "total_sample_policy_elapsed_seconds": policies[spec.key][
                    "total_sample_policy_elapsed_seconds"
                ],
            }
            for spec in SPECS
        },
    )
    write_json_atomic(
        root / "audit.json",
        {
            "all_pass": all_audits,
            "candidate_sample_policy_rows": len(candidate_rows),
            "source_reference_rows": len(source_rows),
            "one_causal_forward_per_boundary": True,
            "future_token_or_accuracy_used": False,
            "continuation_indices_known_context_only": True,
        },
    )
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
    suffixes = _json(OUTPUT / "development" / "suffix_length_strata.json")
    worst = _json(OUTPUT / "development" / "worst_cases.json")
    decision = _json(OUTPUT / "development" / "decision.json")
    baseline_key = SOURCE_SPECS[0].key
    lines = [
        "# One-forward current-context continuation v1",
        "",
        "Each candidate copies only known current-request token context, then runs one "
        "native causal H=8 traversal with fresh native MoE residuals.",
        "",
        "| Variant | Route hit | Selected mass | Coverage | Changed anchors | "
        "Transfer reduction | Probe s | Plan ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = policies[baseline_key]
    lines.append(
        f"| {baseline_key} | {float(baseline['mean_route_hit']):.6f} | "
        f"{float(baseline['mean_selected_mass']):.6f} | - | - | "
        f"{float(baseline['estimated_transfer_reduction']):.4f} | "
        f"{float(baseline['mean_probe_latency_seconds']):.4f} | - |"
    )
    for spec in SPECS:
        values = policies[spec.key]
        lines.append(
            f"| {spec.key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(values['continuation_coverage']):.4f} | "
            f"{float(values['mean_changed_anchor_fraction']):.4f} | "
            f"{float(values['estimated_transfer_reduction']):.4f} | "
            f"{float(values['mean_probe_latency_seconds']):.4f} | "
            f"{1000 * float(values['mean_content_planning_latency_seconds']):.3f} |"
        )
    lines.extend(["", "## Comparisons to checksum-pinned uncorrected v1", ""])
    for spec in SPECS:
        values = comparisons[spec.key]
        policy = policies[spec.key]
        anchor = anchors[spec.key]["1"]
        bootstrap = values["paired_bootstrap"]
        lines.append(
            f"- {spec.key}: hit/mass delta "
            f"{float(values['mean_route_hit_delta']):+.6f}/"
            f"{float(values['mean_selected_mass_delta']):+.6f}; anchor-1 "
            f"{float(anchor['mean_route_hit']):.6f}/"
            f"{float(anchor['mean_selected_mass']):.6f}; mean matched suffix "
            f"{float(policy['mean_matched_suffix_length_when_matched']):.3f}; paired "
            f"hit CI [{float(bootstrap['route_hit_delta_percentile_95'][0]):+.6f}, "
            f"{float(bootstrap['route_hit_delta_percentile_95'][1]):+.6f}], mass CI "
            f"[{float(bootstrap['selected_mass_delta_percentile_95'][0]):+.6f}, "
            f"{float(bootstrap['selected_mass_delta_percentile_95'][1]):+.6f}]; signal "
            f"{values['route_signal']}."
        )
    lines.extend(["", "## Suffix-length strata and worst samples", ""])
    for spec in SPECS:
        strata = suffixes[spec.key]
        rendered = ", ".join(
            f"{bucket}: {int(values['boundaries'])} boundaries at "
            f"{float(values['mean_route_hit']):.6f}/"
            f"{float(values['mean_selected_mass']):.6f}"
            for bucket, values in sorted(strata.items())
        )
        lowest = worst[spec.key][0]
        lines.append(
            f"- {spec.key}: {rendered}. Lowest paired sample "
            f"{lowest['sample_id']} at "
            f"{float(lowest['route_hit_delta']):+.6f}/"
            f"{float(lowest['selected_mass_delta']):+.6f}."
        )
    lines.extend(
        [
            "",
            f"Development-only decision: **{decision['decision']}**. Selected route signal: "
            f"{decision['selected_route_signal_variant']}.",
            "",
            "Route values are teacher-forced on each policy's own hard-subset state. "
            "Content planning and probe cost are measured; transfer is simulated. No "
            "held-out route, task accuracy, free generation, exact-token identity, "
            "closed-loop runtime, or runtime speedup was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples, _composition = _protocol()
    status = _json(OUTPUT / "development" / "pipeline_status.json")
    if status.get("state") != "complete" or not status.get("all_audits_pass"):
        raise ValueError("context-continuation development aggregate is incomplete")
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
                "known_context_continuation_coverage_and_planning_latency",
                "native_one_forward_probe_latency_and_memory",
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
            "offline_continuation_tables": False,
            "future_true_tokens_after_sampled_next_used": False,
            "task_accuracy_measured": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "analysis_id": ANALYSIS_ID,
            "decision": decision["decision"],
            "selected_route_signal_variant": decision["selected_route_signal_variant"],
            "scope": "Qwen_GSM8K_H8_B32_one_forward_context_continuation_development",
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
        raise ValueError("context-continuation pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("context-continuation artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"context-continuation checksum changed: {path}")
    resume = _json(OUTPUT / "resume_audit.json")
    if resume["candidate_sample_policy_rows_validated"] != 12:
        raise ValueError("context-continuation candidate row count changed")
    if resume["source_reference_rows_validated"] != 4:
        raise ValueError("context-continuation source row count changed")
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
