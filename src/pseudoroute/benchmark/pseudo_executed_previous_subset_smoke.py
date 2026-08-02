"""Run the frozen native pseudo-expert previous-subset mechanism smoke."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict
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
    Stage,
    _aggregate_policy,
    _save_tensors_atomic,
    run_policy_sample,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    write_json_atomic,
)
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_executed_previous_subset_smoke_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "117ce2f4f05d91760d36a7e20136120dbead5853d97af89b37d95faeefcd7f4c"
SAMPLES_SHA256 = "2f968509cb7fe6e3a97b803ffd002f64bfab847b12e818508959901d065b688a"
STAGE: Stage = "executed_subset_smoke"
ROUTE_TOKEN_CAP = 16
LAYERS = 48
EXPERTS = 128
TOP_K = 8
BUDGET = 32

EXECUTED = PolicySpec(
    "pseudo_executed_previous_subset",
    "pseudo",
    residual="zero",
    content="sampled_repeat_independent",
    selection="first_four_anchor_core_plus_history_fill",
)
CONTROL = PolicySpec(
    "provided_previous_residual_control",
    "pseudo",
    residual="previous_window_position_aligned",
    content="sampled_repeat_independent",
    selection="first_four_anchor_core_plus_history_fill",
)
PREVIOUS = PolicySpec("previous_route_commitment_reference", "previous")
SPECS = (EXECUTED, CONTROL, PREVIOUS)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("executed-subset smoke config checksum changed")
    if sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("executed-subset smoke sample manifest checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config.get("analysis_id") != ANALYSIS_ID or samples.get("analysis_id") != ANALYSIS_ID:
        raise ValueError("executed-subset smoke protocol identity changed")
    rows = samples.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("executed-subset smoke must retain exactly two frozen rows")
    if [(row["row_index"], row["sample_id"]) for row in rows] != [
        (786, "test-786"),
        (394, "test-394"),
    ]:
        raise ValueError("executed-subset smoke row IDs changed")
    if config["scope"] != {
        "partition": "mechanism_smoke",
        "samples": 2,
        "policies_per_sample": 3,
        "task_accuracy": "forbidden",
        "held_out_or_development_expansion": "forbidden",
    }:
        raise ValueError("executed-subset smoke scope changed")
    return config, samples


def _source_audit(config: dict[str, Any]) -> dict[str, object]:
    suite_manifest = Path(config["source_suite"]["artifact_root"]) / "artifact_manifest.json"
    residual_manifest = (
        Path("artifacts/pseudo_embedding_qwen_gsm8k_residual_window_v1") / "artifact_manifest.json"
    )
    suite_sha = sha256_file(suite_manifest)
    residual_sha = sha256_file(residual_manifest)
    if suite_sha != config["source_suite"]["artifact_manifest_sha256"]:
        raise ValueError("focused suite source artifact manifest changed")
    if residual_sha != config["source_residual_analysis"]["result_manifest_sha256"]:
        raise ValueError("residual-window source artifact manifest changed")
    return {
        "analysis_config_path": str(CONFIG),
        "analysis_config_sha256": CONFIG_SHA256,
        "sample_manifest_path": str(SAMPLES),
        "sample_manifest_sha256": SAMPLES_SHA256,
        "source_suite_artifact_manifest": str(suite_manifest),
        "source_suite_artifact_manifest_sha256": suite_sha,
        "source_residual_artifact_manifest": str(residual_manifest),
        "source_residual_artifact_manifest_sha256": residual_sha,
        "all_frozen_hashes_match": True,
        "new_rows_or_traces_added": False,
        "network_downloads": False,
    }


def _sample_paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _load_valid_sample(
    reference: dict[str, Any],
    spec: PolicySpec,
) -> dict[str, Any] | None:
    json_path, tensor_path = _sample_paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial executed-subset sample artifact: {json_path}")
    row = _json(json_path)
    expected_shadow = spec.key == EXECUTED.key
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("stage") != STAGE
        or row.get("row_index") != reference["row_index"]
        or row.get("sample_id") != reference["sample_id"]
        or row.get("policy") != spec.key
        or row.get("policy_spec_fingerprint") != spec.fingerprint()
        or row.get("shadow_expert_execution") is not expected_shadow
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"corrupt or incompatible executed-subset sample: {json_path}")
    return row


def _failed_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")


def run(*, physical_gpu: int) -> list[dict[str, Any]]:
    config, samples = _read_protocol()
    _source_audit(config)
    existing: list[dict[str, Any]] = []
    missing: list[tuple[dict[str, Any], PolicySpec]] = []
    for reference in samples["rows"]:
        for spec in SPECS:
            row = _load_valid_sample(reference, spec)
            if row is None:
                missing.append((reference, spec))
            else:
                existing.append(row)
    if not missing:
        return existing

    suite = load_pseudo_embedding_config(config["source_suite"]["config"])
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
    completed = list(existing)
    for reference, spec in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        if source["sample_id"] != reference["sample_id"]:
            raise ValueError(f"sample manifest/source mismatch at row {row_index}")
        json_path, tensor_path = _sample_paths(row_index, spec.key)
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
                {},
                stage=STAGE,
                route_token_cap=ROUTE_TOKEN_CAP,
                physical_gpu=physical_gpu,
                shadow_expert_execution=spec.key == EXECUTED.key,
                analysis_id=ANALYSIS_ID,
                analysis_config_sha256=CONFIG_SHA256,
                sample_manifest_sha256=SAMPLES_SHA256,
            )
            _save_tensors_atomic(tensor_path, tensors)
            row["tensor_sha256"] = sha256_file(tensor_path)
            row["tensor_bytes"] = tensor_path.stat().st_size
            write_json_atomic(json_path, row)
            completed.append(cast(dict[str, Any], row))
        except Exception as error:
            write_json_atomic(
                _failed_path(json_path),
                {
                    "state": "failed",
                    "analysis_id": ANALYSIS_ID,
                    "analysis_config_sha256": CONFIG_SHA256,
                    "sample_manifest_sha256": SAMPLES_SHA256,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": spec.key,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return completed


def _rows_by_policy(
    samples: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reference in samples["rows"]:
        for spec in SPECS:
            row = _load_valid_sample(reference, spec)
            if row is None:
                raise RuntimeError("sample row is missing before aggregation")
            grouped[spec.key].append(row)
    return dict(grouped)


def _all_probe_audits_pass(row: dict[str, Any]) -> bool:
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    for audit in row["cache_rng_audits"]:
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                audit.get("full_pre_mask_scores_all_experts", False),
                not audit.get("forbidden_inputs_present", True),
            )
        ):
            return False
    return True


def _tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        )[0]
    )


def _mechanism_audit(
    samples: dict[str, Any],
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    total_escape = 0
    for reference in samples["rows"]:
        row_index = int(reference["row_index"])
        executed_row = next(row for row in grouped[EXECUTED.key] if row["row_index"] == row_index)
        control_row = next(row for row in grouped[CONTROL.key] if row["row_index"] == row_index)
        executed = load_file(str(_sample_paths(row_index, EXECUTED.key)[1]))
        control = load_file(str(_sample_paths(row_index, CONTROL.key)[1]))
        required = {
            "subsets",
            "pseudo_router_logits",
            "planning_moe_residuals",
            "shadow_executed_topk_ids",
        }
        required_tensors_present = required <= set(executed)
        if not required_tensors_present:
            raise ValueError(f"executed row is missing audit tensors: {row_index}")
        subsets = executed["subsets"]
        logits = executed["pseudo_router_logits"]
        residuals = executed["planning_moe_residuals"]
        shadow_ids = executed["shadow_executed_topk_ids"]
        expected_shapes = (
            tuple(subsets.shape) == (2, LAYERS, BUDGET)
            and tuple(logits.shape) == (2, LAYERS, 8, EXPERTS)
            and tuple(shadow_ids.shape) == (2, 8, LAYERS, TOP_K)
            and tuple(residuals.shape[:3]) == (2, 8, LAYERS)
        )
        finite_nonzero = bool(torch.isfinite(residuals).all()) and bool(
            torch.count_nonzero(residuals)
        )
        boundary_zero = executed_row["cache_rng_audits"][0]
        later = executed_row["cache_rng_audits"][1]
        boundary_zero_full = (
            boundary_zero["execution_scope"] == "full_native_topk"
            and boundary_zero["execution_subsets_supplied"] is False
            and boundary_zero["shadow_expert_execution"] is True
            and boundary_zero["shadow_residual_finite"] is True
            and boundary_zero["shadow_residual_nonzero"] is True
        )
        later_scope = (
            later["execution_scope"] == "previous_realized_window_subset"
            and later["execution_subsets_supplied"] is True
            and later["executed_ids_within_supplied_subset"] is True
            and later["shadow_residual_finite"] is True
            and later["shadow_residual_nonzero"] is True
        )
        containment = True
        for anchor in range(8):
            for layer in range(LAYERS):
                containment = containment and (
                    set(int(value) for value in shadow_ids[1, anchor, layer].tolist())
                    <= set(int(value) for value in subsets[0, layer].tolist())
                )
        escape_by_layer = [
            len(
                set(int(value) for value in subsets[1, layer].tolist())
                - set(int(value) for value in subsets[0, layer].tolist())
            )
            for layer in range(LAYERS)
        ]
        row_escape = sum(escape_by_layer)
        total_escape += row_escape
        control_residuals = control["planning_moe_residuals"]
        mean_abs_delta = float((residuals.float() - control_residuals.float()).abs().mean())
        residual_cosine = _tensor_cosine(residuals, control_residuals)
        fresh_vs_control = (
            mean_abs_delta > 0
            and not torch.equal(residuals, control_residuals)
            and executed_row["residual_bank_sha256_by_boundary"]
            != control_row["residual_bank_sha256_by_boundary"]
        )
        audit_pass = (
            required_tensors_present
            and expected_shapes
            and finite_nonzero
            and boundary_zero_full
            and later_scope
            and containment
            and row_escape > 0
            and fresh_vs_control
            and _all_probe_audits_pass(executed_row)
            and _all_probe_audits_pass(control_row)
        )
        checks.append(
            {
                "row_index": row_index,
                "sample_id": reference["sample_id"],
                "pass": audit_pass,
                "required_tensors_present": required_tensors_present,
                "expected_tensor_shapes": expected_shapes,
                "full_pre_mask_scores_all_128_experts": logits.shape[-1] == EXPERTS,
                "pseudo_residual_finite_nonzero": finite_nonzero,
                "boundary_zero_full_native_top8": boundary_zero_full,
                "later_executed_ids_within_previous_subset": containment and later_scope,
                "next_subset_escape_expert_slots": row_escape,
                "layers_with_next_subset_escape": sum(value > 0 for value in escape_by_layer),
                "fresh_residual_differs_from_provided_control": fresh_vs_control,
                "fresh_vs_control_mean_absolute_delta": mean_abs_delta,
                "fresh_vs_control_cosine": residual_cosine,
                "cache_rng_shadow_information_audits_pass": (
                    _all_probe_audits_pass(executed_row) and _all_probe_audits_pass(control_row)
                ),
            }
        )
    all_rows_audited = all(
        _all_probe_audits_pass(row)
        for rows in grouped.values()
        for row in rows
        if row["probe_costs"]
    )
    all_pass = all(bool(check["pass"]) for check in checks) and total_escape > 0
    return {
        "state": "pass" if all_pass else "fail",
        "all_pass": all_pass,
        "sample_checks": checks,
        "native_expert_mlp_executed_on_pseudo_router_input": all_pass,
        "boundary_zero_full_native_top8": all(
            bool(check["boundary_zero_full_native_top8"]) for check in checks
        ),
        "later_boundaries_restricted_to_previous_realized_subset": all(
            bool(check["later_executed_ids_within_previous_subset"]) for check in checks
        ),
        "full_pre_mask_router_scores_all_128_experts": all(
            bool(check["full_pre_mask_scores_all_128_experts"]) for check in checks
        ),
        "total_next_subset_escape_expert_slots": total_escape,
        "production_cache_rng_shadow_information_audits_pass": all_rows_audited,
        "task_accuracy_measured": False,
        "future_true_tokens_used_by_pseudo_policy": False,
        "learned_or_fitted_parameters_used": False,
        "default_vector_values_used": False,
    }


def _write_gzip_metrics(rows: list[dict[str, Any]]) -> None:
    path = OUTPUT / "raw_route_metrics.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            for metric in row["metrics"]:
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _git_revision() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
    )
    return {
        "git_head_at_execution_report": head,
        "tracked_worktree_clean_at_execution_report": not tracked_dirty,
        "protocol_freeze_commit": "49406d6",
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }


def _report(
    aggregates: dict[str, dict[str, int | float]],
    audit: dict[str, object],
    decision: dict[str, object],
) -> str:
    executed = aggregates[EXECUTED.key]
    control = aggregates[CONTROL.key]
    previous = aggregates[PREVIOUS.key]
    checks = cast(list[dict[str, object]], audit["sample_checks"])
    lines = [
        "# Pseudo-executed previous-subset mechanism smoke v1",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "This is a two-row, 16-token teacher-forced mechanism smoke. It is not held-out "
        "evaluation, task accuracy, closed-loop free generation, or a runtime-speedup result.",
        "",
        "## Mechanism",
        "",
        "At boundary 0 the shadow rollout can execute its native top-8 experts because the "
        "prefill assumption exposes all experts. At later boundaries it recomputes each MoE "
        "residual on the pseudo hidden state while restricting execution to the policy's own "
        "previous realized B=32 subset. The next subset is selected only after the complete "
        "eight-anchor shadow rollout.",
        "",
        "## Descriptive route results",
        "",
        "| Policy | Route hit | Selected mass | Simulated transfer reduction | "
        "Mean probe s | Expert calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in (
        (EXECUTED.key, executed),
        (CONTROL.key, control),
        (PREVIOUS.key, previous),
    ):
        lines.append(
            f"| {key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(values['estimated_transfer_reduction']):.6f} | "
            f"{float(values['mean_probe_latency_seconds']):.4f} | "
            f"{int(values['expert_calls'])} |"
        )
    lines.extend(
        [
            "",
            "These route metrics replay the saved v17 token trajectory on each policy's own "
            "hard-subset state. Transfer is simulated; probe/replay time and CUDA memory "
            "are measured.",
            "",
            "## Feasibility audit",
            "",
        ]
    )
    for check in checks:
        lines.append(
            f"- {check['sample_id']}: pass={check['pass']}; "
            f"next-subset escape={check['next_subset_escape_expert_slots']} expert slots across "
            f"{check['layers_with_next_subset_escape']} layers; "
            f"fresh/control mean absolute residual delta="
            f"{cast(float, check['fresh_vs_control_mean_absolute_delta']):.6g}; "
            f"cosine={cast(float, check['fresh_vs_control_cosine']):.6f}."
        )
    lines.extend(
        [
            "",
            "All production cache identity/data-pointer/version/length and RNG invariants are "
            "checked at each pseudo boundary. Full 128-expert pre-mask scores are retained; "
            "later expert IDs are checked against the preceding layer-local subset; shadow caches "
            "are discarded.",
            "",
            "## Interpretation",
            "",
            (
                "The proposed residual path is mechanically viable and can now be compared in a "
                "larger predeclared route experiment."
                if audit["all_pass"]
                else "The proposed residual path failed at least one predeclared mechanism check."
            ),
            "This smoke does not establish that it improves route quality; the route numbers are "
            "descriptive and were not used as a selection gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_manifest() -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(value for value in OUTPUT.rglob("*") if value.is_file()):
        relative = str(path.relative_to(OUTPUT))
        if relative in excluded:
            continue
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
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


def _validate_manifest() -> dict[str, Any]:
    manifest = _json(OUTPUT / "artifact_manifest.json")
    if (
        manifest.get("analysis_id") != ANALYSIS_ID
        or manifest.get("analysis_config_sha256") != CONFIG_SHA256
        or manifest.get("sample_manifest_sha256") != SAMPLES_SHA256
    ):
        raise ValueError("executed-subset artifact manifest identity changed")
    actual_paths = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    if actual_paths != recorded_paths:
        raise ValueError("executed-subset artifact manifest file set changed")
    for artifact in manifest["artifacts"]:
        path = OUTPUT / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"executed-subset artifact checksum mismatch: {path}")
    return manifest


def finalize() -> dict[str, object]:
    config, samples = _read_protocol()
    source_audit = _source_audit(config)
    grouped = _rows_by_policy(samples)
    all_rows = [row for rows in grouped.values() for row in rows]
    aggregates = {policy: _aggregate_policy(rows) for policy, rows in sorted(grouped.items())}
    audit = _mechanism_audit(samples, grouped)
    decision = {
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": "MECHANISM_FEASIBLE" if audit["all_pass"] else "NOT_FEASIBLE",
        "mechanism_feasibility_pass": audit["all_pass"],
        "route_metrics_used_as_selection_gate": False,
        "maximum_claim": "mechanism_feasibility_on_two_fixed_smoke_rows",
        "task_accuracy_claim": False,
        "runtime_speedup_claim": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(OUTPUT / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(OUTPUT / "source_audit.json", source_audit)
    write_json_atomic(
        OUTPUT / "aggregates.json",
        {
            "analysis_id": ANALYSIS_ID,
            "analysis_config_sha256": CONFIG_SHA256,
            "policies": aggregates,
            "executed_minus_provided_control": {
                "mean_route_hit": (
                    float(aggregates[EXECUTED.key]["mean_route_hit"])
                    - float(aggregates[CONTROL.key]["mean_route_hit"])
                ),
                "mean_selected_mass": (
                    float(aggregates[EXECUTED.key]["mean_selected_mass"])
                    - float(aggregates[CONTROL.key]["mean_selected_mass"])
                ),
            },
        },
    )
    _write_gzip_metrics(all_rows)
    write_json_atomic(OUTPUT / "mechanism_audit.json", audit)
    write_json_atomic(
        OUTPUT / "probe_cost_report.json",
        {
            "kind": "measured",
            "policies": {
                policy: {
                    key: value
                    for key, value in aggregate.items()
                    if key
                    in {
                        "mean_probe_latency_seconds",
                        "max_temporary_cuda_bytes",
                        "attention_queries",
                        "attention_calls",
                        "router_calls",
                        "expert_calls",
                        "total_sample_policy_elapsed_seconds",
                        "mean_sample_policy_elapsed_seconds",
                    }
                }
                for policy, aggregate in aggregates.items()
            },
        },
    )
    write_json_atomic(OUTPUT / "decision.json", decision)
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit",
                "teacher_forced_policy_state_selected_mass",
                "native_pseudo_expert_residual",
                "probe_latency",
                "temporary_cuda_memory",
                "attention_router_expert_calls",
                "sample_policy_replay_elapsed_time",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": [
                "task_accuracy",
                "exact_token_identity",
                "free_generation_nll_or_perplexity",
                "closed_loop_generation_runtime",
                "runtime_speedup",
            ],
            "identity_materialized_rows": 0,
            "actual_closed_loop_generation_rows": 0,
        },
    )
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "schema_version": 1,
            "state": "complete",
            "sample_policy_rows_expected": 6,
            "sample_policy_rows_validated": len(all_rows),
            "row_count_pass": len(all_rows) == 6,
            "atomic_json_tensor_pairs": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "schema_version": 1,
            "analysis_id": ANALYSIS_ID,
            "model": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "precision": config["model"]["precision"],
            "sample_ids": [row["sample_id"] for row in samples["rows"]],
            "execution_pids": sorted({int(row["pid"]) for row in all_rows}),
            "execution_ppids": sorted({int(row["ppid"]) for row in all_rows}),
            "physical_gpus": sorted({int(row["physical_gpu"]) for row in all_rows}),
            "network_downloads": False,
            "new_natural_traces": False,
            "learned_or_fitted_parameters": False,
            "default_vector_values_loaded": False,
            "task_accuracy_measured": False,
            "route_evaluation": ("teacher_forced_saved_v17_tokens_on_policy_own_hard_subset_state"),
            "transfer_kind": "simulated",
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report(aggregates, audit, decision))
    manifest = _artifact_manifest()
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": decision["decision"],
        "mechanism_feasibility_pass": audit["all_pass"],
        "sample_policy_rows": len(all_rows),
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": False,
        "held_out_route_executed": False,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    config, samples = _read_protocol()
    _source_audit(config)
    grouped = _rows_by_policy(samples)
    rows = [row for policy_rows in grouped.values() for row in policy_rows]
    if len(rows) != 6:
        raise ValueError("executed-subset smoke row count is incomplete")
    audit = _mechanism_audit(samples, grouped)
    status = _json(OUTPUT / "pipeline_status.json")
    expected_decision = "MECHANISM_FEASIBLE" if audit["all_pass"] else "NOT_FEASIBLE"
    if (
        status.get("state") != "complete"
        or status.get("stage") != "report_v1"
        or status.get("sample_policy_rows") != 6
        or status.get("decision") != expected_decision
    ):
        raise ValueError("executed-subset smoke pipeline status is incomplete")
    resume = _json(OUTPUT / "resume_audit.json")
    if resume.get("sample_policy_rows_validated") != 6 or not resume.get("checksum_resume_pass"):
        raise ValueError("executed-subset smoke resume audit failed")
    manifest = _validate_manifest()
    return {
        "state": "valid",
        "decision": expected_decision,
        "sample_policy_rows": 6,
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    started = time.time()
    if args.command == "run":
        run(physical_gpu=args.gpu)
        result = finalize()
        result["elapsed_seconds_this_invocation"] = time.time() - started
    else:
        result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
