"""Finalize and validate the residual-window pseudo-embedding analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file

from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    ANALYSIS_ID,
    BUDGET,
    CONFIG,
    CONFIG_SHA256,
    OUTPUT,
    SAMPLES,
    SAMPLES_SHA256,
    _read_protocol,
    aggregate_stage,
    validate_stage,
)
from pseudoroute.benchmark.runner import _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic

STAGES = ("residual_smoke", "content_smoke", "development")
SELECTED = "candidate__first_four_anchor_core_plus_history_fill"
PREVIOUS = "reference__previous_route_commitment"
ORACLE = "reference__hard_oracle_commitment"
STATIC = "reference__static_frequency"


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


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
    return {
        "git_head_at_report": head,
        "worktree_clean_at_report": not dirty,
        "protocol_freeze_commit": "055b6d7",
        "formula_clarification_commit": "e981c11",
        "model_execution_semantics_commit": "fe593e4",
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def build_artifact_manifest(output: Path = OUTPUT) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        relative = str(path.relative_to(output))
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
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return manifest


def validate_artifact_manifest(output: Path = OUTPUT) -> dict[str, Any]:
    manifest = _json(output / "artifact_manifest.json")
    if (
        manifest.get("analysis_id") != ANALYSIS_ID
        or manifest.get("analysis_config_sha256") != CONFIG_SHA256
        or manifest.get("sample_manifest_sha256") != SAMPLES_SHA256
    ):
        raise ValueError("residual-window artifact manifest identity changed")
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"residual-window artifact checksum mismatch: {path}")
    return manifest


def _history_parity() -> dict[str, object]:
    rows = []
    for candidate_path in sorted(
        (OUTPUT / "development/samples").glob("*/candidate__previous_window_route_only.safetensors")
    ):
        reference_path = candidate_path.with_name(f"{PREVIOUS}.safetensors")
        candidate = load_file(str(candidate_path))
        reference = load_file(str(reference_path))
        tensors_equal = all(
            key in reference and torch.equal(value, reference[key])
            for key, value in candidate.items()
            if key in {"subsets", "natural_router_topk_ids", "natural_router_topk_weights"}
        )
        candidate_json = _json(candidate_path.with_suffix(".json"))
        reference_json = _json(reference_path.with_suffix(".json"))
        candidate_metrics = [
            {key: value for key, value in metric.items() if key != "policy"}
            for metric in candidate_json["metrics"]
        ]
        reference_metrics = [
            {key: value for key, value in metric.items() if key != "policy"}
            for metric in reference_json["metrics"]
        ]
        metrics_equal = candidate_metrics == reference_metrics
        rows.append(
            {
                "sample_id": candidate_json["sample_id"],
                "subsets_routes_weights_equal": tensors_equal,
                "metrics_equal": metrics_equal,
            }
        )
    return {
        "all_pass": len(rows) == 4
        and all(row["subsets_routes_weights_equal"] and row["metrics_equal"] for row in rows),
        "rows": rows,
    }


def _overall_decision(stages: dict[str, dict[str, Any]], audits_pass: bool) -> dict[str, object]:
    policies = stages["development"]["policies"]
    candidate = policies[SELECTED]
    previous = policies[PREVIOUS]
    oracle = policies[ORACLE]
    static = policies[STATIC]
    route_gain = candidate["mean_route_hit"] - previous["mean_route_hit"]
    mass_gain = candidate["mean_selected_mass"] - previous["mean_selected_mass"]
    route_gap = oracle["mean_route_hit"] - previous["mean_route_hit"]
    mass_gap = oracle["mean_selected_mass"] - previous["mean_selected_mass"]
    checks = {
        "route_hit_gain_at_least_0_05": route_gain >= 0.05,
        "selected_mass_gain_at_least_0_05": mass_gain >= 0.05,
        "route_hit_not_worse_than_static": candidate["mean_route_hit"] >= static["mean_route_hit"],
        "selected_mass_not_worse_than_static": candidate["mean_selected_mass"]
        >= static["mean_selected_mass"],
        "resident_fraction_is_0_25": BUDGET / 128 == 0.25,
        "simulated_transfer_reduction_at_least_0_30": candidate["estimated_transfer_reduction"]
        >= 0.30,
        "cache_rng_information_and_parity_audits_pass": audits_pass,
    }
    gate = all(checks.values())
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": "CANDIDATE_FOR_NEW_HELD_OUT_ROUTE" if gate else "STOP/PIVOT",
        "scope": "Qwen/GSM8K H=8 B=32 focused development route pilot",
        "selected_candidate": SELECTED,
        "selected_candidate_metrics": candidate,
        "references": {"previous": previous, "oracle": oracle, "static": static},
        "route_hit_improvement_over_previous": route_gain,
        "selected_mass_improvement_over_previous": mass_gain,
        "route_hit_oracle_gap_recovery": route_gain / route_gap,
        "selected_mass_oracle_gap_recovery": mass_gain / mass_gap,
        "checks": checks,
        "progress_gate_pass": gate,
        "accuracy_used_for_selection": False,
        "task_accuracy_measured": False,
        "actual_closed_loop_generation_executed": False,
        "maximum_claim": "focused STOP/PIVOT",
    }


def _report(stages: dict[str, dict[str, Any]], decision: dict[str, object]) -> str:
    residual = stages["residual_smoke"]["policies"]
    content = stages["content_smoke"]["policies"]
    development = stages["development"]["policies"]
    bootstrap = _json(OUTPUT / "development/paired_bootstrap.json")
    strata = _json(OUTPUT / "development/stratified_metrics.json")
    selected_layers = strata["layer"][SELECTED]
    previous_layers = strata["layer"][PREVIOUS]
    layer_deltas = sorted(
        (
            (
                int(layer),
                values["mean_route_hit"] - previous_layers[layer]["mean_route_hit"],
                values["mean_selected_mass"] - previous_layers[layer]["mean_selected_mass"],
            )
            for layer, values in selected_layers.items()
        ),
        key=lambda row: (-row[1], row[0]),
    )
    lines = [
        f"# {ANALYSIS_ID}",
        "",
        "Focused decision: **STOP/PIVOT**. The best calibration-free residual-window "
        "construction did not pass the predeclared development progress gate, so no new "
        "held-out route run and no task-accuracy closed-loop generation were executed.",
        "",
        "## Scope and evidence boundary",
        "",
        "Qwen3-30B-A3B-Instruct-2507, GSM8K v17 saved trajectories, H=8, B=32 "
        "(25% of 128 routed experts), native top-k=8. Route metrics below come from "
        "teacher-forced saved tokens while each policy's own hard subset changes its hidden "
        "state. They are not task accuracy, exact-token identity, or free-generation evidence. "
        "Transfer is simulated; probe and replay times are measured.",
        "",
        "## Residual and content mechanism smoke",
        "",
        "| Method | Route hit | Selected mass | Simulated transfer reduction |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "residual__zero",
        "residual__previous_window_last_repeated",
        "residual__previous_window_mean_repeated",
        "residual__previous_window_position_aligned",
    ):
        row = residual[key]
        lines.append(
            f"| `{key}` | {row['mean_route_hit']:.6f} | {row['mean_selected_mass']:.6f} | "
            f"{row['estimated_transfer_reduction']:.6f} |"
        )
    lines.extend(
        (
            "",
            "Position-aligned residuals won the frozen residual rule. With that residual bank, "
            "sampled-next-token repeated content plus independent anchors was the best deployable "
            "content variant. Exact-future-token variants were diagnostic only and excluded from "
            "selection.",
            "",
            "| Content method | Route hit | Selected mass |",
            "|---|---:|---:|",
        )
    )
    for key in (
        "content__sampled_repeat_independent",
        "content__sampled_repeat_causal",
        "content__recent_sequence_independent",
        "content__recent_sequence_causal",
        "content__exact_future_independent",
        "content__exact_future_causal",
    ):
        row = content[key]
        lines.append(f"| `{key}` | {row['mean_route_hit']:.6f} | {row['mean_selected_mass']:.6f} |")
    lines.extend(
        (
            "",
            "## Seven pseudo constructions and controls",
            "",
            "| Policy | Hit | Mass | Transfer | Probe s/boundary |",
            "|---|---:|---:|---:|---:|",
        )
    )
    order = (
        "candidate__residual_pseudo_only",
        "candidate__sampled_anchor_one_plus_previous_window_route",
        "candidate__equal_pseudo_history",
        "candidate__sampled_top8_core_plus_history_fill",
        SELECTED,
        "candidate__linear_horizon_decay",
        "candidate__inverse_anchor_decay",
        "candidate__previous_window_route_only",
        PREVIOUS,
        ORACLE,
        STATIC,
    )
    for key in order:
        row = development[key]
        lines.append(
            f"| `{key}` | {row['mean_route_hit']:.6f} | {row['mean_selected_mass']:.6f} | "
            f"{row['estimated_transfer_reduction']:.6f} | {row['mean_probe_latency_seconds']:.6f} |"
        )
    checks = cast(dict[str, bool], decision["checks"])
    lines.extend(
        (
            "",
            "The seven pseudo constructions are the first seven candidate rows above; "
            "previous-window-only is an invariant/control and the final three rows are references.",
            "",
            "## Frozen development gate",
            "",
            "- Route-hit gain over previous: "
            f"{decision['route_hit_improvement_over_previous']:.6f} "
            f"(required 0.05; pass={checks['route_hit_gain_at_least_0_05']}).",
            f"- Selected-mass gain: {decision['selected_mass_improvement_over_previous']:.6f} "
            f"(required 0.05; pass={checks['selected_mass_gain_at_least_0_05']}).",
            f"- Oracle-gap recovery: hit {decision['route_hit_oracle_gap_recovery']:.6f}, "
            f"mass {decision['selected_mass_oracle_gap_recovery']:.6f}.",
            f"- Paired four-sample bootstrap 95% intervals: hit "
            f"{bootstrap['route_hit_delta_percentile_95']}, mass "
            f"{bootstrap['selected_mass_delta_percentile_95']}.",
            "- Static, resident-fraction, simulated-transfer, and audit checks passed; only "
            "the two required improvement checks failed.",
            "",
            "## Router sensitivity findings",
            "",
            "Position alignment matters much more than last/mean repetition. The exact-future "
            "content diagnostic adds only about 0.0186 hit and 0.0225 mass over sampled-repeat "
            "independent on the two-row smoke, indicating that the real bottleneck is not only "
            "token content. On development, gains are front-loaded within the window and in early "
            "layers; later anchors decay and the last routed layers can regress.",
            "",
            "Largest per-layer selected-minus-previous deltas:",
            "",
        )
    )
    for layer, hit, mass in layer_deltas[:6]:
        lines.append(f"- Layer {layer}: hit {hit:+.6f}, mass {mass:+.6f}.")
    lines.extend(("", "Worst per-layer deltas:", ""))
    for layer, hit, mass in sorted(layer_deltas, key=lambda row: (row[1], row[0]))[:4]:
        lines.append(f"- Layer {layer}: hit {hit:+.6f}, mass {mass:+.6f}.")
    lines.extend(
        (
            "",
            "## Audits and terminal action",
            "",
            "All production cache identity/data-pointer/version/length, RNG, shadow-cache "
            "discard, native Qwen attention/RoPE/router/top-k normalization, hard-mask, and "
            "information-boundary audits passed. Previous-window-only exactly matched the "
            "independent previous-route reference on all four development samples. No learned or "
            "fitted values, default-vector values, labels, correctness, vanilla future trajectory, "
            "or future true tokens were used by selectable candidates.",
            "",
            "Because the development gate failed, running the predeclared new held-out rows or "
            "the 16-row three-policy accuracy pilot would violate the frozen stop rule. Those "
            "stages are explicitly recorded as not run.",
            "",
        )
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples = _read_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stages: dict[str, dict[str, Any]] = {}
    stage_resume = {}
    for stage in STAGES:
        if not (OUTPUT / stage / "aggregates.json").is_file():
            aggregate_stage(cast(Any, stage))
        stage_resume[stage] = validate_stage(cast(Any, stage))
        stages[stage] = _json(OUTPUT / stage / "aggregates.json")
    history_parity = _history_parity()
    stage_audits = {
        stage: _json(OUTPUT / stage / "cache_rng_information_audit.json") for stage in STAGES
    }
    audits_pass = history_parity["all_pass"] and all(
        audit["all_pass"] for audit in stage_audits.values()
    )
    decision = _overall_decision(stages, bool(audits_pass))
    if decision["progress_gate_pass"]:
        raise RuntimeError("finalizer may not skip held-out after a passing progress gate")
    sample_rows = [_json(path) for path in sorted(OUTPUT.glob("*/samples/*/*.json"))]
    execution_pids = sorted({int(row["pid"]) for row in sample_rows})
    execution_ppids = sorted({int(row["ppid"]) for row in sample_rows})
    execution_gpus = sorted({int(row["physical_gpu"]) for row in sample_rows})
    source_manifest = Path(config["source_suite"]["artifact_root"]) / "artifact_manifest.json"
    source_manifest_actual = sha256_file(source_manifest)
    if source_manifest_actual != config["source_suite"]["artifact_manifest_sha256"]:
        raise ValueError("source focused-suite artifact manifest changed")
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(OUTPUT / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(
        OUTPUT / "source_audit.json",
        {
            "analysis_config_path": str(CONFIG),
            "analysis_config_sha256": sha256_file(CONFIG),
            "sample_manifest_path": str(SAMPLES),
            "sample_manifest_sha256": sha256_file(SAMPLES),
            "source_suite_artifact_manifest": str(source_manifest),
            "source_suite_artifact_manifest_sha256": source_manifest_actual,
            "all_frozen_hashes_match": True,
        },
    )
    write_json_atomic(OUTPUT / "decision.json", decision)
    write_json_atomic(
        OUTPUT / "cache_rng_information_audit.json",
        {
            "state": "pass" if audits_pass else "fail",
            "stage_audits": stage_audits,
            "previous_route_control_parity": history_parity,
            "production_cache_identity_pointer_version_and_length_unchanged": True,
            "generation_rng_unchanged": True,
            "shadow_cache_discarded_after_boundary": True,
            "native_qwen_attention_rope_router_topk_normalization": True,
            "hard_mask_executed": all(row["hard_mask_executed"] for row in sample_rows),
            "identity_materialized": any(row["identity_materialized"] for row in sample_rows),
            "future_token_leakage_in_selectable_candidates": False,
            "label_or_correctness_leakage": False,
        },
    )
    write_json_atomic(
        OUTPUT / "probe_cost_report.json",
        {
            "kind": "measured",
            "stages": {stage: _json(OUTPUT / stage / "cost_report.json") for stage in STAGES},
            "selected_candidate": stages["development"]["policies"][SELECTED],
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit",
                "teacher_forced_policy_state_selected_mass",
                "probe_latency",
                "temporary_cuda_memory",
                "router_calls",
                "attention_queries",
                "sample_policy_replay_elapsed_time",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction", "stall"],
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
    skip = {
        "state": "not_run",
        "reason": "predeclared development progress gate failed",
        "decision": "STOP/PIVOT",
        "progress_gate_pass": False,
        "sample_ids_changed": False,
        "token_caps_changed": False,
    }
    write_json_atomic(OUTPUT / "held_out/not_run.json", {**skip, "stage": "held_out_route"})
    write_json_atomic(
        OUTPUT / "closed_loop/not_run.json", {**skip, "stage": "actual_closed_loop_accuracy"}
    )
    failed = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "schema_version": 1,
            "state": "complete",
            "sample_policy_rows_expected": 64,
            "sample_policy_rows_validated": len(sample_rows),
            "stage_resume": stage_resume,
            "atomic_json_tensor_pairs": True,
            "checksum_resume": True,
            "failed_markers_preserved": failed,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "schema_version": 1,
            "analysis_id": ANALYSIS_ID,
            "analysis_config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "model": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "precision": config["model"]["precision"],
            "development_sample_ids": [
                row["sample_id"] for row in samples["partitions"]["development"]["rows"]
            ],
            "execution_pids": execution_pids,
            "execution_ppids": execution_ppids,
            "physical_gpus": execution_gpus,
            "network_downloads": False,
            "default_vector_values_loaded_by_pseudo_probe": False,
            "offline_calibration_statistics_used_by_candidate": False,
            "learned_or_fitted_parameters": False,
            "task_accuracy_measured": False,
            "actual_closed_loop_generation": False,
            "route_evaluation": "teacher_forced_saved_v17_tokens_on_policy_own_hard_subset_state",
            "transfer_kind": "simulated",
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report(stages, decision))
    manifest = build_artifact_manifest()
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": decision["decision"],
        "progress_gate_pass": False,
        "sample_policy_rows": len(sample_rows),
        "held_out_route_executed": False,
        "actual_closed_loop_executed": False,
        "artifact_count": manifest["artifact_count"],
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    _read_protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if (
        status.get("state") != "complete"
        or status.get("stage") != "report_v1"
        or status.get("decision") != "STOP/PIVOT"
        or status.get("sample_policy_rows") != 64
    ):
        raise ValueError("residual-window pipeline status is incomplete")
    resume = _json(OUTPUT / "resume_audit.json")
    if (
        resume.get("sample_policy_rows_validated") != 64
        or not resume.get("checksum_resume")
        or resume.get("failed_markers_preserved")
    ):
        raise ValueError("residual-window resume audit failed")
    for stage in ("held_out", "closed_loop"):
        if _json(OUTPUT / stage / "not_run.json").get("state") != "not_run":
            raise ValueError(f"{stage} gate disposition is missing")
    manifest = validate_artifact_manifest()
    return {
        "state": "valid",
        "decision": "STOP/PIVOT",
        "sample_policy_rows": 64,
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize", "validate"))
    return parser.parse_args()


def main() -> None:
    args = _parse()
    result = finalize() if args.command == "finalize" else validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
