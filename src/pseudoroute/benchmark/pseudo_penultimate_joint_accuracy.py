"""Run the frozen penultimate-joint Qwen/GSM8K wave-one accuracy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_closed_loop import paired_accuracy_bootstrap
from pseudoroute.benchmark.pseudo_embedding_route import _source_model
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    _check_reference,
    _write_checksummed,
    run_policy_sample,
)
from pseudoroute.benchmark.runner import (
    _load_model,
    _model_example,
    _read_jsonl,
    _software_hardware,
)
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.benchmark.tasks import load_examples
from pseudoroute.utils.determinism import seed_everything

Stage = Literal["smoke", "actual"]

PILOT_ID = "pseudo_penultimate_joint_qwen_gsm8k_wave1_v1"
POLICY = "penultimate_unigram_joint_equivalent"
ONLINE_BASELINE = "natural_top8_intersection_zero_missing"
CONFIG = Path("configs/benchmark/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1.yaml")
SAMPLES = Path("configs/benchmark/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1_samples.json")
CONFIG_SHA256 = "57242f91f204c4e765d893dd754e4e523f8fdce62c303827d2e375cec4cc654c"
SAMPLES_SHA256 = "0f4fd75760390fb8ea468af888c8dcd0b22483cdb2af0f96f51a9833c6614d10"
OUTPUT = Path("artifacts/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1")
HORIZON, BUDGET, LAYERS, EXPERTS, TOP_K, CAP = 8, 32, 48, 128, 8, 512
IDS = (
    "test-44",
    "test-632",
    "test-444",
    "test-519",
    "test-1311",
    "test-1264",
    "test-825",
    "test-252",
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_online_baseline(reference: dict[str, Any]) -> dict[str, Any]:
    baseline = cast(dict[str, Any], reference["online_baseline"])
    path = Path(str(baseline["path"]))
    if sha256_file(path) != baseline["file_sha256"]:
        raise ValueError(f"online baseline file checksum changed: {path}")
    row = _json(path)
    expected = row.pop("row_payload_sha256", None)
    if (
        expected != baseline["row_payload_sha256"]
        or expected != sha256_json(row)
        or row.get("policy") != ONLINE_BASELINE
        or row.get("evaluation_mode") != "actual_hard_closed_loop_generation"
        or row.get("sample_id") != reference["sample_id"]
        or int(row.get("row_index", -1)) != int(reference["row_index"])
        or bool(row.get("identity_materialized"))
    ):
        raise ValueError(f"online baseline payload changed: {path}")
    row["row_payload_sha256"] = expected
    row["external_reference_path"] = str(path)
    row["external_reference_file_sha256"] = baseline["file_sha256"]
    return row


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("penultimate-joint protocol fingerprint changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config.get("pilot_id") != PILOT_ID or samples.get("pilot_id") != PILOT_ID:
        raise ValueError("penultimate-joint pilot identity changed")
    point = config["operating_point"]
    model = config["model"]
    if (
        point["horizon"],
        point["budget_per_layer"],
        model["routed_layers"],
        model["routed_experts_per_layer"],
        model["native_top_k"],
    ) != (HORIZON, BUDGET, LAYERS, EXPERTS, TOP_K):
        raise ValueError("penultimate-joint operating point changed")
    if tuple(row["sample_id"] for row in samples["accuracy"]) != IDS:
        raise ValueError("penultimate-joint wave-one sample order changed")
    for key in (
        "focused_config",
        "focused_sample_manifest",
        "accuracy_config",
        "vanilla_rows",
        "online_baseline_config",
        "online_baseline_sample_manifest",
    ):
        path = Path(config["source"][key])
        if sha256_file(path) != config["source"][f"{key}_sha256"]:
            raise ValueError(f"penultimate-joint source changed: {path}")
    original = _json(Path(config["source"]["focused_sample_manifest"]))
    original_rows = original["partitions"]["closed_loop_wave_1"]["rows"]

    def triples(rows: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
        return [(row["row_index"], row["sample_id"], row["sha256_rank"]) for row in rows]

    if triples(samples["accuracy"]) != triples(original_rows):
        raise ValueError("penultimate-joint IDs differ from frozen wave one")
    if any(not bool(_load_online_baseline(row)["correct"]) for row in samples["accuracy"]):
        raise ValueError("frozen online baseline is no longer 8/8")
    return config, samples


def _source_rows(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["row_index"]): row
        for row in _read_jsonl(Path(config["source"]["vanilla_rows"]))
        if row.get("state") == "complete"
    }
    if set(rows) != set(range(1319)):
        raise ValueError("frozen Qwen/GSM8K vanilla source row set changed")
    return rows


def _sample_path(stage: Stage, row_index: int) -> Path:
    return OUTPUT / stage / f"{row_index:05d}" / f"{POLICY}.json"


def _failure_path(stage: Stage, row_index: int) -> Path:
    path = _sample_path(stage, row_index)
    return path.with_name(f"{POLICY}.{os.getpid()}.FAILED.json")


def _load_checksummed(
    path: Path,
    *,
    stage: Stage,
    reference: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = _json(path)
    expected = row.pop("row_payload_sha256", None)
    valid = (
        row.get("state") == "complete"
        and row.get("artifact_checksum_serialization") == "json_round_trip_v1"
        and row.get("pilot_id") == PILOT_ID
        and row.get("config_sha256") == CONFIG_SHA256
        and row.get("sample_manifest_sha256") == SAMPLES_SHA256
        and row.get("stage") == stage
        and row.get("policy") == POLICY
        and row.get("sample_id") == reference["sample_id"]
        and int(row.get("row_index", -1)) == int(reference["row_index"])
        and int(row.get("max_new_tokens", -1)) == max_new_tokens
        and expected == sha256_json(row)
    )
    if not valid:
        raise ValueError(f"incompatible or corrupt penultimate row: {path}")
    row["row_payload_sha256"] = expected
    return row


def _planning_row_audit(row: dict[str, Any]) -> bool:
    planning = cast(list[dict[str, Any]], row["planning_rows"])
    if not planning or planning[0].get("planning_timing") != "online_post_sample_bootstrap":
        return False
    later = planning[1:]
    if len(later) != int(row["penultimate_joint_boundaries"]):
        return False
    for boundary in later:
        content = cast(dict[str, Any], boundary["content"])
        probe = cast(dict[str, Any], boundary["probe_audit"])
        parity = cast(dict[str, Any], boundary["bridge_parity"])
        if (
            boundary.get("planning_timing") != "before_current_window_last_production_forward"
            or content.get("last_sampled_token_available_to_planner") is not False
            or content.get("disposable_hard_bridge_executed") is not True
            or content.get("production_cache_signature_unchanged") is not True
            or content.get("production_rng_unchanged") is not True
            or content.get("shadow_cache_discarded") is not True
            or parity.get("pass") is not True
            or probe.get("subset_residual_execution") != "natural_top8_intersection_zero_missing"
            or probe.get("execution_subsets_supplied") is not True
            or probe.get("executed_ids_within_supplied_subset") is not True
            or probe.get("full_pre_mask_scores_all_experts") is not True
            or probe.get("execution_weights_finite_nonnegative") is not True
        ):
            return False
    return True


def _decorate(row: dict[str, object], source: dict[str, Any]) -> dict[str, object]:
    row.update(
        {
            "pilot_id": PILOT_ID,
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "policy": POLICY,
            "policy_role": "deployable_calibration_free_timing_hypothesis",
            "information_regime": "pre_sample_penultimate_current_policy_known_context_only",
            "logical_accuracy_simulation": True,
            "actual_fused_or_overlapped_runtime": False,
            "actual_offload_engine_used": False,
            "offload_runtime_measured": False,
            "prefetch_overlap_measured": False,
            "joint_fused_runtime_measured": False,
            "result_authorizes_fused_runtime": False,
            "user_confirmation_required_for_fused_runtime": True,
            "source_v17_correct": bool(source["correct"]),
            "source_v17_generated_tokens": int(source["generated_tokens"]),
            "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
            "source_target_sha256": source["target_sha256"],
        }
    )
    row["penultimate_information_cache_rng_bridge_audit_pass"] = _planning_row_audit(
        cast(dict[str, Any], row)
    )
    if not bool(row["penultimate_information_cache_rng_bridge_audit_pass"]):
        raise RuntimeError("penultimate row audit failed")
    return row


def _work(samples: dict[str, Any], stage: Stage) -> list[dict[str, Any]]:
    if stage == "smoke":
        smoke = samples["mechanism_smoke"]
        return [next(row for row in samples["accuracy"] if row["sample_id"] == smoke["sample_id"])]
    return cast(list[dict[str, Any]], samples["accuracy"])


def run(*, stage: Stage, physical_gpu: int = 0) -> dict[str, object]:
    config, samples = _protocol()
    cap = int(config["execution"]["mechanism_smoke"]["max_new_tokens"]) if stage == "smoke" else CAP
    units = _work(samples, stage)
    missing = [
        reference
        for reference in units
        if _load_checksummed(
            _sample_path(stage, int(reference["row_index"])),
            stage=stage,
            reference=reference,
            max_new_tokens=cap,
        )
        is None
    ]
    if not missing:
        return {"state": "already_complete", "stage": stage, "rows": len(units)}
    if stage == "actual":
        audit_smoke()
    accuracy = load_accuracy_suite_config(config["source"]["accuracy_config"])
    model_config = _source_model(accuracy, physical_gpu)
    datasets = [dataset for dataset in accuracy.datasets if dataset.key == "gsm8k"]
    if len(datasets) != 1:
        raise ValueError("frozen accuracy suite is missing GSM8K")
    examples = load_examples(datasets[0], cache_dir=accuracy.dataset_cache_dir)
    sources = _source_rows(config)
    seed_everything(int(config["decode"]["seed"]))
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed-model facts changed")
    revision = _git_head()
    setup = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "execution_git_head": revision,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "physical_gpu": physical_gpu,
        "offload_engine_instantiated": False,
        "software_hardware": _software_hardware(),
    }
    setup["setup_payload_sha256"] = sha256_json(setup)
    write_json_atomic(OUTPUT / "setup" / f"{int(time.time())}.{os.getpid()}.json", setup)
    completed = 0
    for reference in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        _check_reference(reference, source)
        example = _model_example(examples[row_index], model_config)
        if example.sample_id != reference["sample_id"]:
            raise ValueError(f"dataset/source mismatch: {reference['sample_id']}")
        max_tokens = min(example.max_new_tokens, cap)
        path = _sample_path(stage, row_index)
        try:
            row = run_policy_sample(
                config,
                accuracy,
                model_config,
                model,
                tokenizer,
                ops,
                example,
                source,
                policy="sampled_unigram_full_continuation",
                stage=stage,
                max_new_tokens=max_tokens,
                physical_gpu=physical_gpu,
                execution_git_head=revision,
                subset_residual_execution="natural_top8_intersection_zero_missing",
                planning_timing="penultimate_joint_equivalent",
            )
            row = _decorate(row, source)
            _write_checksummed(path, row)
            completed += 1
        except Exception as error:
            failure: dict[str, object] = {
                "schema_version": 1,
                "state": "failed",
                "pilot_id": PILOT_ID,
                "config_sha256": CONFIG_SHA256,
                "sample_manifest_sha256": SAMPLES_SHA256,
                "stage": stage,
                "row_index": row_index,
                "sample_id": reference["sample_id"],
                "policy": POLICY,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "execution_git_head": revision,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failure["failure_payload_sha256"] = sha256_json(failure)
            write_json_atomic(_failure_path(stage, row_index), failure)
            raise
    return {"state": "complete", "stage": stage, "new_rows": completed}


def _rows(stage: Stage) -> list[dict[str, Any]]:
    config, samples = _protocol()
    cap = int(config["execution"]["mechanism_smoke"]["max_new_tokens"]) if stage == "smoke" else CAP
    rows = []
    for reference in _work(samples, stage):
        row = _load_checksummed(
            _sample_path(stage, int(reference["row_index"])),
            stage=stage,
            reference=reference,
            max_new_tokens=cap,
        )
        if row is None:
            raise ValueError(f"penultimate {stage} rows are incomplete")
        rows.append(row)
    return rows


def audit_smoke() -> dict[str, object]:
    config, _ = _protocol()
    row = _rows("smoke")[0]
    minimum = int(config["execution"]["mechanism_smoke"]["expected_joint_boundaries_minimum"])
    if (
        int(row["penultimate_joint_boundaries"]) < minimum
        or int(row["logical_simulator_disposable_bridge_calls"]) < minimum
        or not bool(row["all_bridge_parity_audits_pass"])
        or not bool(row["penultimate_information_cache_rng_bridge_audit_pass"])
        or bool(row["actual_offload_engine_used"])
        or row["actual_offload_metrics"] is not None
    ):
        raise ValueError("penultimate mechanism smoke audit failed")
    result = {
        "state": "valid",
        "sample_id": row["sample_id"],
        "generated_tokens": row["generated_tokens"],
        "boundaries": row["boundary_count"],
        "penultimate_joint_boundaries": row["penultimate_joint_boundaries"],
        "bridge_parity_passes": row["bridge_parity_passes"],
        "offload_engine_used": False,
    }
    write_json_atomic(OUTPUT / "smoke" / "audit.json", result)
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    successes = sum(bool(row["correct"]) for row in rows)
    route_hits = sum(int(row["route_hits"]) for row in rows)
    route_slots = sum(int(row["route_slots"]) for row in rows)
    mass_hit = sum(float(row["selected_mass_hit"]) for row in rows)
    mass_total = sum(float(row["selected_mass_total"]) for row in rows)
    exact_matches = sum(int(row["exact_token_matches"]) for row in rows)
    exact_total = sum(int(row["exact_token_comparison_tokens"]) for row in rows)
    nll_tokens = sum(int(row["nll_tokens"]) for row in rows)
    weighted_nll = (
        sum(float(row["v17_token_nll_on_policy_context"]) * int(row["nll_tokens"]) for row in rows)
        / nll_tokens
    )
    generated = sum(int(row["generated_tokens"]) for row in rows)
    elapsed = sum(float(row["elapsed_seconds_measured"]) for row in rows)
    planned = sum(int(row["simulated_planned_transfer_bytes"]) for row in rows)
    natural = sum(int(row["simulated_natural_reference_bytes"]) for row in rows)
    return {
        "samples": len(rows),
        "successes": successes,
        "accuracy_measured": successes / len(rows),
        "total_generated_tokens": generated,
        "exact_token_agreement_weighted": exact_matches / exact_total,
        "samples_with_any_token_divergence": sum(
            row["first_token_divergence"] is not None for row in rows
        ),
        "route_hit_rate_weighted": route_hits / route_slots,
        "selected_routing_mass_coverage_weighted": mass_hit / mass_total,
        "weighted_v17_token_nll_on_policy_context": weighted_nll,
        "weighted_perplexity": math.exp(weighted_nll) if weighted_nll < 700 else float("inf"),
        "total_logical_simulator_runtime_seconds_measured": elapsed,
        "aggregate_logical_simulator_tokens_per_second_measured": generated / elapsed,
        "probe_latency_seconds_measured": sum(
            float(row["probe_latency_seconds_measured"]) for row in rows
        ),
        "duplicate_bridge_latency_seconds_measured": sum(
            float(row.get("logical_simulator_bridge_latency_seconds_measured", 0.0)) for row in rows
        ),
        "simulated_planned_transfer_bytes": planned,
        "simulated_natural_reference_bytes": natural,
        "simulated_estimated_transfer_reduction": 1 - planned / natural,
        "all_cache_rng_bridge_audits_pass": all(
            bool(row.get("penultimate_information_cache_rng_bridge_audit_pass", True))
            for row in rows
        ),
    }


def aggregate() -> dict[str, object]:
    config, samples = _protocol()
    smoke = audit_smoke()
    candidate_rows = _rows("actual")
    baseline_rows = [_load_online_baseline(row) for row in samples["accuracy"]]
    sources = _source_rows(config)
    correctness = {
        "vanilla_v17": {
            row["sample_id"]: bool(sources[int(row["row_index"])]["correct"])
            for row in samples["accuracy"]
        },
        ONLINE_BASELINE: {row["sample_id"]: bool(row["correct"]) for row in baseline_rows},
        POLICY: {row["sample_id"]: bool(row["correct"]) for row in candidate_rows},
    }
    bootstrap = paired_accuracy_bootstrap(
        correctness,
        samples=int(config["accuracy_gate"]["paired_bootstrap_samples"]),
        seed=int(config["accuracy_gate"]["paired_bootstrap_seed"]),
    )
    candidate = _summary(candidate_rows)
    baseline = _summary(baseline_rows)
    successes = cast(int, candidate["successes"])
    if successes == 8:
        decision = "PILOT_NARROW_STRONG_PRESERVATION_AWAITING_USER_CONFIRMATION"
    elif successes == 7:
        decision = "PILOT_NARROW_ONE_ALLOWED_LOSS_AWAITING_USER_CONFIRMATION"
    else:
        decision = "STOP_PIVOT_AWAITING_USER_CONFIRMATION"
    ids = list(IDS)
    paired = {
        "baseline": ONLINE_BASELINE,
        "candidate": POLICY,
        "paired_gains": sum(
            correctness[POLICY][sample] and not correctness[ONLINE_BASELINE][sample]
            for sample in ids
        ),
        "paired_losses": sum(
            correctness[ONLINE_BASELINE][sample] and not correctness[POLICY][sample]
            for sample in ids
        ),
        "paired_equal": sum(
            correctness[ONLINE_BASELINE][sample] == correctness[POLICY][sample] for sample in ids
        ),
        "accuracy_difference": cast(float, candidate["accuracy_measured"])
        - cast(float, baseline["accuracy_measured"]),
    }
    per_sample = []
    for reference in samples["accuracy"]:
        sample_id = str(reference["sample_id"])
        candidate_row = next(row for row in candidate_rows if row["sample_id"] == sample_id)
        baseline_row = next(row for row in baseline_rows if row["sample_id"] == sample_id)
        per_sample.append(
            {
                "row_index": reference["row_index"],
                "sample_id": sample_id,
                "vanilla_v17_correct": correctness["vanilla_v17"][sample_id],
                "online_baseline": {
                    "correct": bool(baseline_row["correct"]),
                    "generated_tokens": int(baseline_row["generated_tokens"]),
                    "first_token_divergence": baseline_row["first_token_divergence"],
                },
                "penultimate_candidate": {
                    "correct": bool(candidate_row["correct"]),
                    "generated_tokens": int(candidate_row["generated_tokens"]),
                    "first_token_divergence": candidate_row["first_token_divergence"],
                    "penultimate_joint_boundaries": candidate_row["penultimate_joint_boundaries"],
                },
            }
        )
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "samples": 8,
        "new_actual_rows": 8,
        "external_online_baseline_rows": 8,
        "vanilla_v17": {
            "samples": 8,
            "successes": 8,
            "accuracy_measured_preexisting": 1.0,
            "regenerated": False,
        },
        "online_post_sample_baseline": baseline,
        "penultimate_joint_candidate": candidate,
        "candidate_vs_online_baseline": paired,
        "paired_accuracy_bootstrap": bootstrap,
        "per_sample": per_sample,
        "accuracy_gate": {
            "minimum_candidate_successes": 7,
            "candidate_pass": successes >= 7,
            "strong_preservation": successes == 8,
        },
        "smoke_audit": smoke,
        "decision": decision,
        "checkpoint": "awaiting_user_confirmation_before_fused_or_offload_implementation",
        "conclusion_ceiling": "PILOT_NARROW",
    }
    write_json_atomic(OUTPUT / "actual" / "summary.json", result)
    write_json_atomic(OUTPUT / "actual" / "per_sample.json", {"rows": per_sample})
    write_json_atomic(
        OUTPUT / "actual" / "paired_accuracy_bootstrap.json",
        {"rows": bootstrap},
    )
    return result


def _report(summary: dict[str, Any]) -> str:
    baseline = summary["online_post_sample_baseline"]
    candidate = summary["penultimate_joint_candidate"]
    paired = summary["candidate_vs_online_baseline"]
    return "\n".join(
        [
            "# Penultimate-joint Qwen/GSM8K wave-1 accuracy report",
            "",
            "Actual hard closed-loop generation uses the same eight frozen GSM8K rows at ",
            "H=8 and B=32. The online baseline is checksum-pinned and was not rerun.",
            "",
            "| Policy | Correct | Token agreement | Route hit | Selected mass |",
            "|---|---:|---:|---:|---:|",
            (
                f"| {ONLINE_BASELINE} | {baseline['successes']}/8 | "
                f"{float(baseline['exact_token_agreement_weighted']):.6f} | "
                f"{float(baseline['route_hit_rate_weighted']):.6f} | "
                f"{float(baseline['selected_routing_mass_coverage_weighted']):.6f} |"
            ),
            (
                f"| {POLICY} | {candidate['successes']}/8 | "
                f"{float(candidate['exact_token_agreement_weighted']):.6f} | "
                f"{float(candidate['route_hit_rate_weighted']):.6f} | "
                f"{float(candidate['selected_routing_mass_coverage_weighted']):.6f} |"
            ),
            "",
            (
                "Penultimate versus online paired gains/losses/equal: "
                f"{paired['paired_gains']}/{paired['paired_losses']}/{paired['paired_equal']}."
            ),
            "",
            "## Scope boundary",
            "",
            f"Focused decision: **{summary['decision']}**. The candidate simulator uses a ",
            "disposable duplicate bridge to preserve the intended state dependency. Its ",
            "latency is not joint-kernel runtime. Offloading, prefetch overlap, fused runtime, ",
            "and production speedup are not measured. The pipeline is stopped at the required ",
            "user-confirmation checkpoint; N=8 cannot establish full-dataset preservation.",
            "",
        ]
    )


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
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(OUTPUT / "artifact_manifest.json", manifest)
    return manifest


def finalize() -> dict[str, object]:
    config, samples = _protocol()
    summary = cast(dict[str, Any], aggregate())
    rows = _rows("actual")
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    external = {
        "schema_version": 1,
        "source_policy": ONLINE_BASELINE,
        "rows": [
            {
                "row_index": reference["row_index"],
                "sample_id": reference["sample_id"],
                **reference["online_baseline"],
            }
            for reference in samples["accuracy"]
        ],
    }
    write_json_atomic(OUTPUT / "resolved_external_references.json", external)
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "git_head_at_report": _git_head(),
            "row_execution_git_heads": sorted({str(row["execution_git_head"]) for row in rows}),
            "row_pids": sorted({int(row["pid"]) for row in rows}),
            "row_ppids": sorted({int(row["ppid"]) for row in rows}),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "actual_hard_closed_loop_gsm8k_accuracy",
                "token_agreement_and_divergence",
                "route_hit_and_selected_routing_mass",
                "v17_token_nll_and_perplexity_on_policy_context",
                "logical_simulator_probe_bridge_and_generation_runtime",
                "cache_rng_and_bridge_parity",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": [
                "actual_expert_offload_runtime",
                "prefetch_overlap",
                "joint_fused_kernel_runtime",
                "production_speedup",
                "full_dataset_accuracy",
            ],
            "identity_materialized_rows": 0,
            "actual_hard_closed_loop_rows": 8,
        },
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "smoke_rows_validated": 1,
            "actual_rows_validated": len(rows),
            "external_online_baseline_rows_checksum_validated": 8,
            "atomic_json_rows": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "pilot_id": PILOT_ID,
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "sample_ids": list(IDS),
            "current_policy_closed_loop_context_used": True,
            "candidate_calibration_free": True,
            "future_v17_tokens_used": False,
            "label_or_correctness_used_during_execution": False,
            "offload_engine_used": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "pilot_id": PILOT_ID,
            "decision": summary["decision"],
            "scope": "Qwen_GSM8K_H8_B32_wave_one_penultimate_joint_accuracy_checkpoint",
            "candidate_successes": summary["penultimate_joint_candidate"]["successes"],
            "accuracy_gate_minimum_successes": 7,
            "user_confirmation_required_for_fused_runtime": True,
            "fused_runtime_authorized": False,
            "offload_speedup_claim": False,
            "full_dataset_go": False,
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report(summary))
    manifest = _manifest()
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "wave_1_report_v1",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "decision": summary["decision"],
        "new_actual_rows": 8,
        "external_online_baseline_rows": 8,
        "artifact_count": manifest["artifact_count"],
        "checkpoint": "awaiting_user_confirmation_before_fused_or_offload_implementation",
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "wave_1_report_v1":
        raise ValueError("penultimate wave-one pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual_files = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual_files != recorded:
        raise ValueError("penultimate artifact file set changed")
    for artifact in manifest["artifacts"]:
        path = OUTPUT / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"penultimate artifact checksum changed: {path}")
    smoke = _rows("smoke")
    rows = _rows("actual")
    if len(smoke) != 1 or len(rows) != 8:
        raise ValueError("penultimate row count changed")
    for row in [*smoke, *rows]:
        if (
            not bool(row["hard_mask_executed"])
            or bool(row["identity_materialized"])
            or not bool(row["outside_subset_router_logits_masked"])
            or not bool(row["native_topk_and_normalization_after_mask"])
            or row["planning_timing"] != "penultimate_joint_equivalent"
            or not bool(row["all_planning_audits_pass"])
            or not bool(row["all_bridge_parity_audits_pass"])
            or not bool(row["penultimate_information_cache_rng_bridge_audit_pass"])
            or bool(row["actual_offload_engine_used"])
            or row["actual_offload_metrics"] is not None
            or bool(row["future_v17_tokens_used_by_policy"])
            or bool(row["label_or_correctness_used_during_policy_execution"])
        ):
            raise ValueError("penultimate execution audit changed")
    resume = _json(OUTPUT / "resume_audit.json")
    if (
        resume["smoke_rows_validated"] != 1
        or resume["actual_rows_validated"] != 8
        or resume["external_online_baseline_rows_checksum_validated"] != 8
        or not resume["checksum_resume_pass"]
    ):
        raise ValueError("penultimate resume audit changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "actual_rows": len(rows),
        "external_online_baseline_rows": 8,
        "artifacts": manifest["artifact_count"],
        "checkpoint": status["checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "protocol",
            "run-smoke",
            "smoke-audit",
            "run-wave1",
            "aggregate",
            "finalize",
            "validate",
            "all",
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.command == "protocol":
        config, samples = _protocol()
        result: dict[str, object] = {
            "state": "valid",
            "pilot_id": config["pilot_id"],
            "samples": len(samples["accuracy"]),
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
        }
    elif args.command == "run-smoke":
        result = run(stage="smoke", physical_gpu=args.gpu)
    elif args.command == "smoke-audit":
        result = audit_smoke()
    elif args.command == "run-wave1":
        result = run(stage="actual", physical_gpu=args.gpu)
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    elif args.command == "validate":
        result = validate()
    else:
        run(stage="smoke", physical_gpu=args.gpu)
        audit_smoke()
        run(stage="actual", physical_gpu=args.gpu)
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
