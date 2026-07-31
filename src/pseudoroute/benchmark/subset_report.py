"""Strict aggregation, paired gates, decisions, and provenance for subset oracle v1."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from pseudoroute.benchmark.config import AccuracySuiteConfig
from pseudoroute.benchmark.subset_config import (
    TASKS,
    SubsetModelConfig,
    SubsetOracleSuiteConfig,
    TaskKey,
)
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _source_rows(root: Path, model: str, task: str) -> list[dict[str, Any]]:
    path = root / "models" / model / "results" / "vanilla" / task / "samples.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    complete = [cast(dict[str, Any], row) for row in rows if row.get("state") == "complete"]
    return sorted(complete, key=lambda row: int(row["row_index"]))


def _selected(output: Path) -> dict[str, dict[str, Any]]:
    selection = _json(output / "selected_operating_points.json")
    return {str(row["model"]): row for row in selection["selected"]}


def _actual_path(
    output: Path, model: str, task: str, row_index: int, policy: str, stage: str = "full"
) -> Path:
    return (
        output
        / "models"
        / model
        / "closed_loop"
        / stage
        / task
        / f"{row_index:05d}"
        / f"{policy}.json"
    )


def _validate_actual(
    row: dict[str, Any],
    source: dict[str, Any],
    suite: SubsetOracleSuiteConfig,
    model: SubsetModelConfig,
    task: str,
    horizon: int,
    budget: int,
    policy: str,
) -> None:
    expected = {
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "model": model.key,
        "task": task,
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "policy": policy,
        "evaluation_mode": "actual_closed_loop_generation",
        "horizon": horizon,
        "budget": budget,
        "logical_shard": int(source["row_index"]) % suite.closed_loop.sample_shards,
        "source_v17_row_sha256": sha256_json(source),
    }
    observed = {key: row.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"closed-loop provenance mismatch for {model.key}/{task}/{source['row_index']}/"
            f"{policy}: {observed} != {expected}"
        )


def _paired_row(
    model: str,
    task: str,
    horizon: int,
    budget: int,
    policy: str,
    sources: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    allowed_drop: int,
) -> dict[str, object]:
    differences = [
        int(bool(row["correct"])) - int(bool(source["correct"]))
        for source, row in zip(sources, actual, strict=True)
    ]
    sample_count = len(differences)
    mean = fmean(differences)
    variance = (
        sum((value - mean) ** 2 for value in differences) / (sample_count - 1)
        if sample_count > 1
        else 0.0
    )
    radius = 1.96 * math.sqrt(variance / sample_count) if sample_count else float("nan")
    vanilla_successes = sum(bool(source["correct"]) for source in sources)
    policy_successes = sum(bool(row["correct"]) for row in actual)
    losses = sum(
        bool(source["correct"]) and not bool(row["correct"])
        for source, row in zip(sources, actual, strict=True)
    )
    gains = sum(
        not bool(source["correct"]) and bool(row["correct"])
        for source, row in zip(sources, actual, strict=True)
    )
    return {
        "model": model,
        "task": task,
        "horizon": horizon,
        "budget": budget,
        "policy": policy,
        "samples": sample_count,
        "vanilla_successes": vanilla_successes,
        "policy_successes": policy_successes,
        "vanilla_accuracy": vanilla_successes / sample_count,
        "policy_accuracy": policy_successes / sample_count,
        "paired_accuracy_delta": mean,
        "paired_delta_ci95_low": max(-1.0, mean - radius),
        "paired_delta_ci95_high": min(1.0, mean + radius),
        "paired_losses": losses,
        "paired_gains": gains,
        "allowed_drop_questions": allowed_drop,
        "minimum_successes": vanilla_successes - allowed_drop,
        "accuracy_gate_pass": policy_successes >= vanilla_successes - allowed_drop,
        "ci_method": "paired_difference_normal_95",
    }


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = fieldnames or list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_jsonl_gzip_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _natural_materialized(
    suite: SubsetOracleSuiteConfig,
    model: SubsetModelConfig,
    task: str,
    source: dict[str, Any],
    horizon: int,
    budget: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "model": model.key,
        "task": task,
        "row_index": source["row_index"],
        "sample_id": source["sample_id"],
        "policy": "natural_v17_reuse",
        "horizon": horizon,
        "budget": budget,
        "correct": source["correct"],
        "generated_tokens": source["generated_tokens"],
        "generated_token_ids": source["generated_token_ids"],
        "exact_token_agreement": 1.0,
        "evaluation_mode": "measured_v17_closed_loop_generation_reused_without_rerun",
        "identity_materialized": False,
        "source_reused": True,
        "source_v17_row_sha256": sha256_json(source),
    }


def _lossless_materialized(natural: dict[str, Any]) -> dict[str, Any]:
    row = dict(natural)
    row.update(
        {
            "policy": "lossless_oracle_residency",
            "evaluation_mode": (
                "identity_materialized_from_v17_after_every_task_actual_identity_smoke"
            ),
            "identity_materialized": True,
            "accuracy_improvement_claimed": False,
            "fallback_metrics_scope": "representative_natural_trace_grid_not_this_full_sample",
            "measured_runtime": None,
        }
    )
    return row


def _smoke_audit(
    suite: SubsetOracleSuiteConfig, output: Path
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    rows: list[dict[str, Any]] = []
    per_model_changed: dict[str, bool] = {}
    for model in suite.models:
        changed = False
        for task_value in TASKS:
            task = cast(TaskKey, task_value)
            source_id = suite.closed_loop.smoke_rows[task]
            for policy in ("lossless_oracle_residency", "hard_oracle_commitment"):
                root = output / "models" / model.key / "closed_loop" / "mechanism_smoke" / task
                matches = list(root.glob(f"*/{policy}.json"))
                if len(matches) != 1:
                    raise ValueError(
                        f"mechanism smoke row count mismatch: {model.key}/{task}/{policy}"
                    )
                row = _json(matches[0])
                if (
                    row.get("sample_id") != source_id
                    or row.get("horizon") != suite.closed_loop.smoke_horizon
                    or row.get("budget") != model.native_top_k
                    or row.get("evaluation_mode") != "actual_closed_loop_generation"
                ):
                    raise ValueError(f"mechanism smoke provenance mismatch: {matches[0]}")
                if policy == "lossless_oracle_residency":
                    if row.get("exact_token_agreement") != 1.0 or row.get("executed_route_changed"):
                        raise ValueError(f"lossless identity smoke failed: {matches[0]}")
                else:
                    changed = changed or bool(row.get("executed_route_changed"))
                rows.append(row)
        if not changed:
            raise ValueError(f"hard mechanism smoke changed no executed route for {model.key}")
        per_model_changed[model.key] = changed
    return rows, {
        "mechanism_smoke_rows": len(rows),
        "lossless_exact_identity": True,
        "hard_executed_route_changed_by_model": per_model_changed,
        "smoke_point_is_operating_candidate": False,
    }


def aggregate_and_validate(
    suite: SubsetOracleSuiteConfig,
    accuracy: AccuracySuiteConfig,
    output: Path,
    *,
    actual_policies: tuple[str, ...] = (
        "hard_oracle_commitment",
        "previous_route_commitment",
    ),
    tasks: tuple[TaskKey, ...] | None = None,
    execution_scope_id: str | None = None,
    execution_scope_fingerprint: str | None = None,
) -> dict[str, object]:
    """Require every selected full row, then emit paired reports and final decisions."""
    allowed_policies = {
        "hard_oracle_commitment",
        "previous_route_commitment",
    }
    if "hard_oracle_commitment" not in actual_policies or not set(actual_policies).issubset(
        allowed_policies
    ):
        raise ValueError(f"invalid required full policies: {actual_policies}")
    evaluated_tasks = cast(tuple[TaskKey, ...], TASKS) if tasks is None else tasks
    if (
        not evaluated_tasks
        or len(set(evaluated_tasks)) != len(evaluated_tasks)
        or not set(evaluated_tasks).issubset(TASKS)
    ):
        raise ValueError(f"invalid aggregate task scope: {evaluated_tasks}")
    scoped_datasets = [
        dataset for dataset in accuracy.datasets if dataset.key in set(evaluated_tasks)
    ]
    if tuple(dataset.key for dataset in scoped_datasets) != evaluated_tasks:
        raise ValueError("aggregate task scope must follow frozen dataset order")
    if (execution_scope_id is None) != (execution_scope_fingerprint is None):
        raise ValueError("execution-scope ID and fingerprint must be provided together")
    if execution_scope_fingerprint is not None and len(execution_scope_fingerprint) != 64:
        raise ValueError("execution-scope fingerprint must be a SHA-256 hex digest")
    selected = _selected(output)
    _, smoke_audit = _smoke_audit(suite, output)
    source_root = Path(suite.source_accuracy.artifact_root)
    paired: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, Any]] = []
    selected_policy_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    source_rows_in_scope = 0
    for model in suite.models:
        point = selected.get(model.key)
        if point is None:
            continue
        horizon = int(point["horizon"])
        budget = int(point["budget"])
        for dataset in scoped_datasets:
            task = dataset.key
            sources = _source_rows(source_root, model.key, task)
            if len(sources) != dataset.expected_samples:
                raise ValueError(f"source row count mismatch: {model.key}/{task}")
            source_rows_in_scope += len(sources)
            natural_materialized = [
                _natural_materialized(suite, model, task, source, horizon, budget)
                for source in sources
            ]
            all_rows.extend(natural_materialized)
            all_rows.extend(_lossless_materialized(row) for row in natural_materialized)
            natural_successes = sum(bool(source["correct"]) for source in sources)
            frozen = suite.paired_accuracy_gate.vanilla[model.key][task]
            if natural_successes != frozen.successes or len(sources) != frozen.samples:
                raise ValueError(f"frozen vanilla gate differs from source: {model.key}/{task}")
            accuracy_rows.append(
                {
                    "model": model.key,
                    "task": task,
                    "horizon": horizon,
                    "budget": budget,
                    "policy": "natural_v17_reuse",
                    "samples": len(sources),
                    "successes": natural_successes,
                    "accuracy": natural_successes / len(sources),
                    "measurement_kind": "measured_v17_closed_loop_generation_reused",
                    "mean_exact_token_agreement": 1.0,
                    "mean_measured_seconds": "",
                    "mean_vanilla_token_nll": "",
                    "mean_perplexity": "",
                }
            )
            accuracy_rows.append(
                {
                    "model": model.key,
                    "task": task,
                    "horizon": horizon,
                    "budget": budget,
                    "policy": "lossless_oracle_residency",
                    "samples": len(sources),
                    "successes": natural_successes,
                    "accuracy": natural_successes / len(sources),
                    "measurement_kind": "identity_materialized_after_actual_identity_smoke",
                    "mean_exact_token_agreement": 1.0,
                    "mean_measured_seconds": "",
                    "mean_vanilla_token_nll": "",
                    "mean_perplexity": "",
                }
            )
            for policy in actual_policies:
                actual = []
                for source in sources:
                    path = _actual_path(output, model.key, task, int(source["row_index"]), policy)
                    if not path.is_file():
                        raise ValueError(f"missing selected closed-loop sample: {path}")
                    row = _json(path)
                    _validate_actual(row, source, suite, model, task, horizon, budget, policy)
                    actual.append(row)
                selected_policy_rows[(model.key, task, policy)] = actual
                all_rows.extend(actual)
                successes = sum(bool(row["correct"]) for row in actual)
                accuracy_rows.append(
                    {
                        "model": model.key,
                        "task": task,
                        "horizon": horizon,
                        "budget": budget,
                        "policy": policy,
                        "samples": len(actual),
                        "successes": successes,
                        "accuracy": successes / len(actual),
                        "measurement_kind": "actual_closed_loop_generation",
                        "mean_exact_token_agreement": fmean(
                            float(row["exact_token_agreement"]) for row in actual
                        ),
                        "mean_measured_seconds": fmean(
                            float(row["elapsed_seconds_measured"]) for row in actual
                        ),
                        "mean_vanilla_token_nll": fmean(
                            float(row["vanilla_token_nll_on_policy_context"]) for row in actual
                        ),
                        "mean_perplexity": fmean(float(row["perplexity"]) for row in actual),
                    }
                )
                paired.append(
                    _paired_row(
                        model.key,
                        task,
                        horizon,
                        budget,
                        policy,
                        sources,
                        actual,
                        frozen.allowed_drop_questions,
                    )
                )
    closed_loop_path = output / "closed_loop" / "all_samples.jsonl.gz"
    row_count = _write_jsonl_gzip_atomic(closed_loop_path, all_rows)
    _write_csv_atomic(
        output / "accuracy_summary.csv",
        accuracy_rows,
        fieldnames=[
            "model",
            "task",
            "horizon",
            "budget",
            "policy",
            "samples",
            "successes",
            "accuracy",
            "measurement_kind",
            "mean_exact_token_agreement",
            "mean_measured_seconds",
            "mean_vanilla_token_nll",
            "mean_perplexity",
        ],
    )
    _write_csv_atomic(
        output / "paired_vanilla_delta.csv",
        paired,
        fieldnames=[
            "model",
            "task",
            "horizon",
            "budget",
            "policy",
            "samples",
            "vanilla_successes",
            "policy_successes",
            "vanilla_accuracy",
            "policy_accuracy",
            "paired_accuracy_delta",
            "paired_delta_ci95_low",
            "paired_delta_ci95_high",
            "paired_losses",
            "paired_gains",
            "allowed_drop_questions",
            "minimum_successes",
            "accuracy_gate_pass",
            "ci_method",
        ],
    )
    write_json_atomic(
        output / "accuracy_summary.json",
        {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "config_fingerprint": suite.fingerprint(),
            "required_full_actual_policies": list(actual_policies),
            "evaluated_tasks": list(evaluated_tasks),
            "execution_scope_id": execution_scope_id,
            "execution_scope_fingerprint": execution_scope_fingerprint,
            "rows": accuracy_rows,
            "paired_rows": paired,
            "closed_loop_materialized_rows": row_count,
            "policy_measurement_notes": {
                "natural_v17_reuse": "measured in v17 and reused here without rerun",
                "lossless_oracle_residency": (
                    "open-loop simulated fully; actual identity smoke only"
                ),
                "hard_oracle_commitment": "actual closed-loop generation",
                "previous_route_commitment": "actual non-oracle closed-loop generation",
            },
        },
    )
    operating_rows = _csv(output / "operating_points.csv")
    decision_rows: list[dict[str, object]] = []
    hard_paired = {
        (str(row["model"]), str(row["task"])): row
        for row in paired
        if row["policy"] == "hard_oracle_commitment"
    }
    full_task_scope = evaluated_tasks == TASKS
    model_scoped_accuracy_pass: dict[str, bool] = {
        model.key: model.key in selected
        and all(
            bool(hard_paired[(model.key, task)]["accuracy_gate_pass"]) for task in evaluated_tasks
        )
        for model in suite.models
    }
    for open_row in operating_rows:
        model_key = open_row["model"]
        task_name = open_row["task"]
        if model_key not in selected or task_name not in evaluated_tasks:
            continue
        horizon = int(open_row["horizon"])
        budget = int(open_row["budget"])
        selected_row = open_row["selected_for_closed_loop"].lower() == "true"
        open_pass = open_row["task_open_loop_pass"].lower() == "true"
        full_expert = open_row["below_all_expert"].lower() != "true"
        paired_row = hard_paired.get((model_key, task_name)) if selected_row else None
        accuracy_pass = bool(paired_row and paired_row["accuracy_gate_pass"])
        if full_expert:
            label, reason = "STOP/PIVOT", "all-expert reference is not a saving"
        elif not open_pass:
            label, reason = "STOP/PIVOT", "predeclared open-loop task gate failed"
        elif not selected_row:
            label, reason = "STOP/PIVOT", "not selected by predeclared all-task ordering"
        elif not accuracy_pass:
            label, reason = "STOP/PIVOT", "paired hard-commitment accuracy gate failed"
        elif full_task_scope and model_scoped_accuracy_pass[model_key]:
            label, reason = "GO", "all six open-loop and paired accuracy gates passed"
        elif model_scoped_accuracy_pass[model_key]:
            label, reason = (
                "NARROW",
                "evaluated task-scoped gates passed; unscheduled tasks were not evaluated",
            )
        else:
            label, reason = "NARROW", "task-scoped gates passed; another task failed"
        decision_rows.append(
            {
                "model": model_key,
                "task": task_name,
                "horizon": horizon,
                "budget": budget,
                "decision": label,
                "reason": reason,
                "selected_for_closed_loop": selected_row,
                "open_loop_task_pass": open_pass,
                "hard_accuracy_gate_pass": accuracy_pass if selected_row else "",
            }
        )
    _write_csv_atomic(output / "decisions_per_model_task_point.csv", decision_rows)
    labels = {row["decision"] for row in decision_rows if row["selected_for_closed_loop"]}
    if selected and labels == {"GO"}:
        overall = "GO"
        authorization = "all-task learned subset selector authorized"
    elif "NARROW" in labels:
        overall = "NARROW"
        authorization = "only explicitly NARROW task scopes are authorized"
    else:
        overall = "STOP/PIVOT"
        authorization = "learned subset selector training is not authorized"
    decision = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "evaluated_actual_policies": list(actual_policies),
        "evaluated_tasks": list(evaluated_tasks),
        "all_six_tasks_evaluated": full_task_scope,
        "execution_scope_id": execution_scope_id,
        "execution_scope_fingerprint": execution_scope_fingerprint,
        "overall_decision": overall,
        "predictor_authorization": authorization,
        "selected_operating_points": list(selected.values()),
        "model_scoped_task_accuracy_pass": model_scoped_accuracy_pass,
        "model_all_task_accuracy_pass": {
            model: passed and full_task_scope
            for model, passed in model_scoped_accuracy_pass.items()
        },
        "per_point_rows": len(decision_rows),
        "no_macro_average_override": True,
    }
    write_json_atomic(output / "decision.json", decision)
    pareto = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "execution_scope_id": execution_scope_id,
        "execution_scope_fingerprint": execution_scope_fingerprint,
        "measurement_partition": {
            "route_coverage_fallback_transfer_stall": "simulated from natural trace replay",
            "mechanism_identity": "actual closed-loop smoke",
            "hard_accuracy_runtime": "actual closed-loop full generation",
        },
        "selected_operating_points": list(selected.values()),
        "selected_open_loop_task_rows": [
            row
            for row in operating_rows
            if row["selected_for_closed_loop"].lower() == "true"
            and row["model"] in selected
            and row["task"] in evaluated_tasks
        ],
        "mechanism_smoke": smoke_audit,
    }
    write_json_atomic(output / "fallback_transfer_pareto.json", pareto)
    audit = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "source_v17_rows": 5216,
        "source_v17_rows_in_scope": source_rows_in_scope,
        "natural_trace_manifests": 2,
        "open_loop_envelopes": 2,
        "selected_models": len(selected),
        "required_full_actual_policies": list(actual_policies),
        "evaluated_tasks": list(evaluated_tasks),
        "execution_scope_id": execution_scope_id,
        "execution_scope_fingerprint": execution_scope_fingerprint,
        "actual_full_sample_rows": sum(len(rows) for rows in selected_policy_rows.values()),
        "materialized_closed_loop_rows": row_count,
        "mechanism_smoke": smoke_audit,
        "resume_atomic_unit": suite.closed_loop.atomic_unit,
        "provenance_validation": "passed",
    }
    write_json_atomic(output / "provenance_audit.json", audit)
    _write_report(
        output,
        suite,
        decision,
        paired,
        smoke_audit,
        actual_policies,
        evaluated_tasks,
        execution_scope_id,
        execution_scope_fingerprint,
    )
    manifest = build_artifact_manifest(suite, output)
    return {"decision": decision, "audit": audit, "manifest": manifest}


def _write_report(
    output: Path,
    suite: SubsetOracleSuiteConfig,
    decision: dict[str, object],
    paired: list[dict[str, object]],
    smoke_audit: dict[str, object],
    actual_policies: tuple[str, ...],
    evaluated_tasks: tuple[TaskKey, ...],
    execution_scope_id: str | None,
    execution_scope_fingerprint: str | None,
) -> None:
    actual_policy_text = ", ".join(actual_policies)
    lines = [
        "# Benchmark subset oracle v1 report",
        "",
        f"Config fingerprint: `{suite.fingerprint()}`",
        "",
        f"Execution scope: `{execution_scope_id}` (`{execution_scope_fingerprint}`).",
        "",
        f"Full accuracy task scope: `{', '.join(evaluated_tasks)}`.",
        "Unscheduled task artifacts are preserved as provenance and are not aggregated.",
        "",
        f"Overall decision: **{decision['overall_decision']}**",
        "",
        str(decision["predictor_authorization"]),
        "",
        "The natural reference is measured v17 generation reused without rerunning. Lossless ",
        "residency transfer/stall metrics are simulated from natural route replay and its ",
        "exact-token identity has actual smoke evidence. Required full actual policies: ",
        f"`{actual_policy_text}`. Their runtime is measured closed-loop generation; no hard ",
        "result is identity-materialized. Preserved unscheduled rows are provenance only.",
        "",
        f"Mechanism smoke rows: {smoke_audit['mechanism_smoke_rows']}; lossless identity passed.",
        "",
        "## Paired hard-commitment accuracy",
        "",
        "| Model | Task | H | B | Vanilla | Hard | Delta | 95% CI | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        if row["policy"] != "hard_oracle_commitment":
            continue
        lines.append(
            f"| {row['model']} | {row['task']} | {row['horizon']} | {row['budget']} | "
            f"{int(cast(Any, row['vanilla_successes']))}/{int(cast(Any, row['samples']))} | "
            f"{int(cast(Any, row['policy_successes']))}/{int(cast(Any, row['samples']))} | "
            f"{float(cast(Any, row['paired_accuracy_delta'])):+.4f} | "
            f"[{float(cast(Any, row['paired_delta_ci95_low'])):+.4f}, "
            f"{float(cast(Any, row['paired_delta_ci95_high'])):+.4f}] | "
            f"{'PASS' if row['accuracy_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "AIME retains integer question counts. Per-task/per-point decisions are in ",
            "`decisions_per_model_task_point.csv`; no cross-task macro average can override them.",
            "",
        ]
    )
    temporary = output / f".report.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, output / "report.md")


def build_artifact_manifest(suite: SubsetOracleSuiteConfig, output: Path) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded or relative.startswith("logs/"):
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return manifest


def validate_artifact_manifest(suite: SubsetOracleSuiteConfig, output: Path) -> None:
    manifest = _json(output / "artifact_manifest.json")
    if manifest.get("config_fingerprint") != suite.fingerprint():
        raise ValueError("artifact manifest config mismatch")
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact manifest checksum mismatch: {path}")
