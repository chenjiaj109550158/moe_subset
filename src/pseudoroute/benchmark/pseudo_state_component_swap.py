"""Run the frozen Qwen pseudo-state component-swap diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping
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
from pseudoroute.benchmark.qwen_pseudo import DiagnosticStatePatch
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_state_component_swap_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "6fc9c4cd92de0108c198de03de63badd38f0a5c9885aad33e15a893ea5efe53f"
SAMPLES_SHA256 = "2a75b1cb96f46189d469d1d5a431174351fa726f44a71909cd1708c5ed27bb17"
LAYERS = 48
EXPERTS = 128
TOP_K = 8
HORIZON = 8
BUDGET = 32


@dataclass(frozen=True)
class ComponentSwapSpec:
    key: str
    diagnostic_state_patch: DiagnosticStatePatch = "none"
    diagnostic_hidden_after_layer: int | None = None

    def policy(self) -> PolicySpec:
        return PolicySpec(
            self.key,
            "pseudo",
            residual="zero",
            content="exact_future_causal",
            selection="first_four_anchor_core_plus_history_fill",
            diagnostic=True,
        )

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


SPECS = (
    ComponentSwapSpec("exact_future_unpatched"),
    ComponentSwapSpec(
        "exact_future_attention_output_oracle",
        "oracle_attention_output",
    ),
    ComponentSwapSpec(
        "exact_future_moe_residual_oracle",
        "oracle_moe_residual",
    ),
    ComponentSwapSpec(
        "exact_future_attention_and_moe_oracle",
        "oracle_attention_and_moe",
    ),
    ComponentSwapSpec(
        "exact_future_hidden_after_layer_7_oracle",
        "oracle_hidden_after_layer",
        7,
    ),
    ComponentSwapSpec(
        "exact_future_hidden_after_layer_15_oracle",
        "oracle_hidden_after_layer",
        15,
    ),
    ComponentSwapSpec(
        "exact_future_hidden_after_layer_23_oracle",
        "oracle_hidden_after_layer",
        23,
    ),
    ComponentSwapSpec(
        "exact_future_hidden_after_layer_31_oracle",
        "oracle_hidden_after_layer",
        31,
    ),
    ComponentSwapSpec(
        "exact_future_hidden_after_layer_39_oracle",
        "oracle_hidden_after_layer",
        39,
    ),
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("aggregate metric is not numeric")
    return float(value)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("component-swap protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("component-swap protocol identity changed")
    if tuple(config["variants"]["order"]) != tuple(spec.key for spec in SPECS):
        raise ValueError("component-swap variant order changed")
    sources = config["source"]
    for path_key, hash_key in (
        ("parent_config", "parent_config_sha256"),
        ("parent_sample_manifest", "parent_sample_manifest_sha256"),
        ("parent_artifact_manifest", "parent_artifact_manifest_sha256"),
        ("composition_config", "composition_config_sha256"),
        ("composition_sample_manifest", "composition_sample_manifest_sha256"),
        ("composition_artifact_manifest", "composition_artifact_manifest_sha256"),
        ("focused_suite_config", "focused_suite_config_sha256"),
        ("focused_suite_sample_manifest", "focused_suite_sample_manifest_sha256"),
    ):
        path = Path(str(sources[path_key]))
        if sha256_file(path) != sources[hash_key]:
            raise ValueError(f"component-swap source checksum changed: {path}")
    return config, samples


def _paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "development" / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid(reference: dict[str, Any], spec: ComponentSwapSpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial component-swap row: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("component_swap_spec_fingerprint") != spec.fingerprint()
        or row.get("diagnostic_state_patch") != spec.diagnostic_state_patch
        or row.get("diagnostic_hidden_after_layer") != spec.diagnostic_hidden_after_layer
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible component-swap row: {json_path}")
    tensors = load_file(str(tensor_path))
    expected_shapes = {
        "pseudo_router_logits": (8, LAYERS, HORIZON, EXPERTS),
        "oracle_component_router_logits": (8, HORIZON, LAYERS, EXPERTS),
        "oracle_component_topk_ids": (8, HORIZON, LAYERS, TOP_K),
        "component_state_cosines": (8, 4, HORIZON, LAYERS),
        "centered_router_logit_cosines": (8, HORIZON, LAYERS),
        "pseudo_oracle_topk_overlaps": (8, HORIZON, LAYERS),
        "planning_subset_oracle_jaccards": (8, LAYERS),
    }
    for key, shape in expected_shapes.items():
        if key not in tensors or tuple(tensors[key].shape) != shape:
            raise ValueError(f"component-swap tensor shape changed: {tensor_path}/{key}")
    return row


def run(*, physical_gpu: int, shard_index: int, shard_count: int) -> dict[str, object]:
    config, samples = _protocol()
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid component-swap shard")
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
                capture_natural_component_state=True,
                diagnostic_state_patch=spec.diagnostic_state_patch,
                diagnostic_hidden_after_layer=spec.diagnostic_hidden_after_layer,
            )
            row["component_swap_spec"] = asdict(spec)
            row["component_swap_spec_fingerprint"] = spec.fingerprint()
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


def _audit_pass(row: dict[str, Any], config: dict[str, Any]) -> bool:
    expected = config["audits"]
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    if len(row["probe_costs"]) != 8 or len(row["oracle_component_capture_costs"]) != 8:
        return False
    for cost, capture, audit in zip(
        row["probe_costs"],
        row["oracle_component_capture_costs"],
        row["cache_rng_audits"],
        strict=True,
    ):
        component = audit.get("oracle_component_capture") or {}
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                audit.get("one_causal_forward_per_boundary", False),
                audit.get("shadow_expert_execution", False),
                audit.get("shadow_component_states_complete", False),
                audit.get("information_regime") == "diagnostic_future_state_oracle",
                audit.get("deployable") is False,
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                not audit.get("forbidden_inputs_present", True),
                cost["attention_calls"] == expected["expected_pseudo_attention_calls_per_boundary"],
                cost["attention_queries"]
                == expected["expected_pseudo_attention_queries_per_boundary"],
                cost["router_calls"] == expected["expected_pseudo_router_calls_per_boundary"],
                cost["expert_calls"] == expected["expected_pseudo_expert_calls_per_boundary"],
                capture["attention_calls"]
                == expected["expected_oracle_attention_calls_per_boundary"],
                capture["attention_queries"]
                == expected["expected_oracle_attention_queries_per_full_boundary"],
                capture["router_calls"] == expected["expected_oracle_router_calls_per_boundary"],
                capture["expert_calls"] == expected["expected_oracle_expert_calls_per_boundary"],
                component.get("production_cache_signature_unchanged", False),
                component.get("production_rng_unchanged", False),
                component.get("shadow_cache_discarded", False),
            )
        ):
            return False
    return True


def classify_attribution(
    policy: Mapping[str, Mapping[str, object]],
    config: dict[str, Any],
) -> dict[str, object]:
    rule = config["attribution_rule"]
    keys = {
        "baseline": SPECS[0].key,
        "attention": SPECS[1].key,
        "moe": SPECS[2].key,
        "joint": SPECS[3].key,
    }
    base = policy[keys["baseline"]]
    gains: dict[str, dict[str, float]] = {}
    for name in ("attention", "moe", "joint"):
        candidate = policy[keys[name]]
        gains[name] = {
            "route_hit": _number(candidate["mean_route_hit"]) - _number(base["mean_route_hit"]),
            "selected_mass": _number(candidate["mean_selected_mass"])
            - _number(base["mean_selected_mass"]),
        }
    joint = gains["joint"]
    if joint["route_hit"] < float(rule["minimum_joint_absolute_route_hit_gain"]) or joint[
        "selected_mass"
    ] < float(rule["minimum_joint_absolute_selected_mass_gain"]):
        classification = "component_path_unresolved"
    else:
        fractions = {
            name: {
                metric: gains[name][metric] / joint[metric]
                for metric in ("route_hit", "selected_mass")
            }
            for name in ("attention", "moe")
        }
        means = {name: sum(values.values()) / 2 for name, values in fractions.items()}
        minimum = float(rule["component_dominance_minimum_fraction_of_joint_gain"])
        lead = float(rule["component_dominance_minimum_mean_recovery_lead"])
        if min(fractions["attention"].values()) >= minimum and (
            means["attention"] - means["moe"] >= lead
        ):
            classification = "attention_dominant"
        elif min(fractions["moe"].values()) >= minimum and (
            means["moe"] - means["attention"] >= lead
        ):
            classification = "moe_residual_dominant"
        else:
            classification = "coupled_attention_and_residual"
    checkpoint_threshold = float(rule["checkpoint_recovery_threshold_fraction_of_joint_gain"])
    passing_checkpoints: list[int] = []
    if joint["route_hit"] > 0 and joint["selected_mass"] > 0:
        for spec in SPECS[4:]:
            values = policy[spec.key]
            route = _number(values["mean_route_hit"]) - _number(base["mean_route_hit"])
            mass = _number(values["mean_selected_mass"]) - _number(base["mean_selected_mass"])
            if (
                route >= checkpoint_threshold * joint["route_hit"]
                and mass >= checkpoint_threshold * joint["selected_mass"]
            ):
                if spec.diagnostic_hidden_after_layer is None:
                    raise AssertionError("hidden checkpoint spec is missing its layer")
                passing_checkpoints.append(spec.diagnostic_hidden_after_layer)
    mapping = config["next_candidate_mapping"]
    return {
        "classification": classification,
        "focused_decision": (
            "STOP/PIVOT" if classification == "component_path_unresolved" else "NARROW"
        ),
        "gains_over_unpatched": gains,
        "latest_checkpoint_meeting_recovery": (
            max(passing_checkpoints) if passing_checkpoints else None
        ),
        "passing_checkpoints": passing_checkpoints,
        "next_calibration_free_candidate": mapping[classification],
        "candidate_execution_requires_separate_frozen_protocol": True,
    }


def _alignment_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output: dict[str, object] = {}
    names = ("attention_output", "router_input", "moe_residual", "decoder_layer_output")
    for policy, policy_rows in sorted(grouped.items()):
        states = []
        logits = []
        overlaps = []
        jaccards = []
        margins = []
        for row in policy_rows:
            tensors = load_file(str(_paths(int(row["row_index"]), policy)[1]))
            states.append(tensors["component_state_cosines"].float())
            logits.append(tensors["centered_router_logit_cosines"].float())
            overlaps.append(tensors["pseudo_oracle_topk_overlaps"].float())
            jaccards.append(tensors["planning_subset_oracle_jaccards"].float())
            sorted_logits = (
                tensors["oracle_component_router_logits"]
                .float()
                .sort(dim=-1, descending=True)
                .values
            )
            margins.append(sorted_logits[..., 7] - sorted_logits[..., 8])
        state = torch.cat(states)
        logit = torch.cat(logits)
        overlap = torch.cat(overlaps)
        jaccard = torch.cat(jaccards)
        margin = torch.cat(margins)
        quantiles = torch.quantile(margin.reshape(-1), torch.tensor([0.25, 0.5, 0.75]))
        margin_strata = {}
        edges = (float("-inf"), *[float(value) for value in quantiles], float("inf"))
        for index in range(4):
            mask = (margin >= edges[index]) & (margin <= edges[index + 1])
            margin_strata[f"q{index + 1}"] = {
                "count": int(mask.sum()),
                "mean_centered_router_logit_cosine": float(logit[mask].mean()),
                "mean_pseudo_oracle_topk_overlap": float(overlap[mask].mean()),
            }
        output[policy] = {
            "mean_state_cosines": {
                name: float(state[:, index].mean()) for index, name in enumerate(names)
            },
            "mean_centered_router_logit_cosine": float(logit.mean()),
            "mean_pseudo_oracle_topk_overlap": float(overlap.mean()),
            "mean_planning_subset_oracle_jaccard": float(jaccard.mean()),
            "anchor_strata": {
                str(anchor + 1): {
                    "mean_centered_router_logit_cosine": float(logit[:, anchor].mean()),
                    "mean_pseudo_oracle_topk_overlap": float(overlap[:, anchor].mean()),
                }
                for anchor in range(HORIZON)
            },
            "layer_block_strata": {
                f"{start}-{start + 7}": {
                    "mean_centered_router_logit_cosine": float(
                        logit[..., start : start + 8].mean()
                    ),
                    "mean_pseudo_oracle_topk_overlap": float(
                        overlap[..., start : start + 8].mean()
                    ),
                    "mean_planning_subset_oracle_jaccard": float(
                        jaccard[..., start : start + 8].mean()
                    ),
                }
                for start in range(0, LAYERS, 8)
            },
            "natural_router_margin_quartiles": [float(value) for value in quantiles],
            "natural_router_margin_strata": margin_strata,
        }
    return output


def _worst_cases(rows: list[dict[str, Any]], *, limit: int = 32) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for row in rows:
        policy = str(row["policy"])
        tensors = load_file(str(_paths(int(row["row_index"]), policy)[1]))
        logit = tensors["centered_router_logit_cosines"].float()
        overlap = tensors["pseudo_oracle_topk_overlaps"].float()
        for flat_index in torch.topk(-logit.reshape(-1), min(limit, logit.numel())).indices:
            boundary, anchor, layer = torch.unravel_index(flat_index, logit.shape)
            cases.append(
                {
                    "sample_id": row["sample_id"],
                    "row_index": row["row_index"],
                    "policy": policy,
                    "boundary": int(boundary) * HORIZON,
                    "anchor": int(anchor) + 1,
                    "layer": int(layer),
                    "centered_router_logit_cosine": float(logit[boundary, anchor, layer]),
                    "pseudo_oracle_topk_overlap": float(overlap[boundary, anchor, layer]),
                }
            )
    cases.sort(key=lambda value: (_number(value["centered_router_logit_cosine"]), str(value)))
    return {"ordering": "lowest_centered_router_logit_cosine", "cases": cases[:limit]}


def aggregate() -> dict[str, object]:
    config, samples = _protocol()
    references = samples["partitions"]["development"]
    rows = []
    for reference in references:
        for spec in SPECS:
            row = _valid(reference, spec)
            if row is None:
                raise RuntimeError(f"missing component row: {reference['sample_id']}/{spec.key}")
            rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    policies = {
        policy: _aggregate_policy(policy_rows) for policy, policy_rows in sorted(grouped.items())
    }
    baseline = policies[SPECS[0].key]
    comparisons = {}
    for spec in SPECS[1:]:
        candidate = policies[spec.key]
        comparisons[spec.key] = {
            "mean_route_hit_delta": float(candidate["mean_route_hit"])
            - float(baseline["mean_route_hit"]),
            "mean_selected_mass_delta": float(candidate["mean_selected_mass"])
            - float(baseline["mean_selected_mass"]),
            "paired_bootstrap": _paired_bootstrap(
                {str(row["sample_id"]): _per_sample(row) for row in grouped[spec.key]},
                {str(row["sample_id"]): _per_sample(row) for row in grouped[SPECS[0].key]},
                draws=int(config["analysis"]["bootstrap_samples"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
        }
    attribution = classify_attribution(policies, config)
    audit = {
        "all_pass": all(_audit_pass(row, config) for row in rows),
        "sample_policy_rows": len(rows),
        "accuracy_or_correctness_used": False,
        "held_out_rows_used": False,
        "learned_or_fitted_parameters": False,
    }
    cost_report = {}
    for policy, policy_rows in grouped.items():
        captures = [value for row in policy_rows for value in row["oracle_component_capture_costs"]]
        cost_report[policy] = {
            "mean_pseudo_probe_latency_seconds": policies[policy]["mean_probe_latency_seconds"],
            "mean_oracle_capture_latency_seconds": sum(
                float(value["latency_seconds_measured"]) for value in captures
            )
            / len(captures),
            "max_pseudo_temporary_cuda_bytes": policies[policy]["max_temporary_cuda_bytes"],
            "max_oracle_capture_temporary_cuda_bytes": max(
                int(value["temporary_cuda_bytes_measured"]) for value in captures
            ),
            "pseudo_attention_calls": policies[policy]["attention_calls"],
            "oracle_attention_calls": sum(int(value["attention_calls"]) for value in captures),
            "pseudo_router_calls": policies[policy]["router_calls"],
            "oracle_router_calls": sum(int(value["router_calls"]) for value in captures),
            "pseudo_expert_calls": policies[policy]["expert_calls"],
            "oracle_expert_calls": sum(int(value["expert_calls"]) for value in captures),
        }
    root = OUTPUT / "development"
    write_json_atomic(root / "aggregates.json", {"analysis_id": ANALYSIS_ID, "policies": policies})
    write_json_atomic(root / "comparisons.json", comparisons)
    write_json_atomic(root / "attribution.json", attribution)
    write_json_atomic(root / "state_alignment.json", _alignment_summary(rows))
    write_json_atomic(root / "worst_cases.json", _worst_cases(rows))
    write_json_atomic(root / "cost_report.json", cost_report)
    write_json_atomic(root / "audit.json", audit)
    write_json_atomic(
        root / "decision.json",
        {
            "analysis_id": ANALYSIS_ID,
            **attribution,
            "held_out_route_authorized": False,
            "task_accuracy_authorized": False,
        },
    )
    status = {
        "state": "complete",
        "sample_policy_rows": len(rows),
        "all_audits_pass": audit["all_pass"],
        "classification": attribution["classification"],
        "focused_decision": attribution["focused_decision"],
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
    attribution = _json(OUTPUT / "development" / "attribution.json")
    alignment = _json(OUTPUT / "development" / "state_alignment.json")
    audit = _json(OUTPUT / "development" / "audit.json")
    resume = _json(OUTPUT / "resume_audit.json")
    lines = [
        "# Pseudo state component swap v1",
        "",
        "This is a non-deployable mechanism diagnostic on four frozen Qwen/GSM8K development "
        "rows at H=8, B=32. Every variant uses exact saved future tokens and one native causal "
        "pseudo traversal; the future component capture is a separate full-expert oracle pass.",
        "",
        "| Variant | Route hit | Selected mass | Transfer reduction | Pseudo probe s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, values in policies.items():
        lines.append(
            f"| {key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(values['estimated_transfer_reduction']):.6f} | "
            f"{float(values['mean_probe_latency_seconds']):.4f} |"
        )
    lines.extend(["", "## Component gains over unpatched", ""])
    for key, values in comparisons.items():
        lines.append(
            f"- {key}: route-hit {float(values['mean_route_hit_delta']):+.6f}; "
            f"selected-mass {float(values['mean_selected_mass_delta']):+.6f}; paired 95% "
            f"intervals {values['paired_bootstrap']['route_hit_delta_percentile_95']} and "
            f"{values['paired_bootstrap']['selected_mass_delta_percentile_95']}."
        )
    baseline_alignment = alignment[SPECS[0].key]
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            f"Frozen classification: **{attribution['classification']}**. Focused decision: "
            f"**{attribution['focused_decision']}**. Latest hidden checkpoint recovering the "
            f"frozen threshold: `{attribution['latest_checkpoint_meeting_recovery']}`.",
            "",
            f"Unpatched mean centered router-logit cosine is "
            f"{float(baseline_alignment['mean_centered_router_logit_cosine']):.6f}; mean natural "
            f"top-8 overlap is {float(baseline_alignment['mean_pseudo_oracle_topk_overlap']):.6f}; "
            f"mean B32 oracle Jaccard is "
            f"{float(baseline_alignment['mean_planning_subset_oracle_jaccard']):.6f}.",
            "",
            f"Next predeclared calibration-free family: "
            f"`{attribution['next_calibration_free_candidate']}`. It was not executed here; a "
            "separate frozen protocol is required before its first model output.",
            "",
            "## Integrity and evidence boundary",
            "",
            f"All cache/RNG/component/information audits pass: `{audit['all_pass']}`. Validated "
            f"atomic rows: {resume['sample_policy_rows_validated']}; checksum resume: "
            f"`{resume['checksum_resume_pass']}`; failed markers preserved: "
            f"{len(resume['failed_markers_preserved'])}.",
            "",
            "Route/state alignment, probe latency, capture latency, and memory are measured. "
            "Transfer is simulated. Task accuracy, free generation, exact-token identity, NLL, "
            "closed-loop runtime, and runtime speedup were not measured. Component swaps and "
            "exact future tokens are diagnostic oracle inputs, not deployable pseudo embeddings.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples = _protocol()
    development = _json(OUTPUT / "development" / "pipeline_status.json")
    if development.get("state") != "complete" or not development.get("all_audits_pass"):
        raise ValueError("component-swap development aggregate is incomplete")
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
            "pid": os.getpid(),
            "ppid": os.getppid(),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit_and_selected_mass",
                "pseudo_vs_full_native_component_state_alignment",
                "pseudo_and_oracle_capture_latency_and_memory",
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
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "sample_policy_rows_validated": len(rows),
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
            "future_token_and_component_oracle_inputs": True,
            "component_swaps_deployable": False,
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
            "decision": decision["focused_decision"],
            "classification": decision["classification"],
            "scope": "Qwen_GSM8K_H8_B32_four_row_component_diagnostic",
            "next_calibration_free_candidate": decision["next_calibration_free_candidate"],
            "candidate_executed": False,
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
        "decision": decision["focused_decision"],
        "classification": decision["classification"],
        "sample_policy_rows": len(rows),
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
        raise ValueError("component-swap pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("component-swap artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"component-swap checksum changed: {path}")
    if _json(OUTPUT / "resume_audit.json")["sample_policy_rows_validated"] != 36:
        raise ValueError("component-swap row count changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "classification": status["classification"],
        "sample_policy_rows": status["sample_policy_rows"],
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
