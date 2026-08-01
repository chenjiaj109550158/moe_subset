"""Versioned, resumable CLI for the focused Qwen/GSM8K pseudo pilot."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.pseudo_embedding_closed_loop import (
    _load_checksummed_row,
    aggregate_closed_loop,
    run_closed_loop_wave,
    wave_eta_checkpoint,
)
from pseudoroute.benchmark.pseudo_embedding_closed_loop import (
    _sample_path as closed_loop_sample_path,
)
from pseudoroute.benchmark.pseudo_embedding_config import (
    PseudoEmbeddingSuiteConfig,
    SampleManifest,
)
from pseudoroute.benchmark.pseudo_embedding_report import aggregate_route_partition
from pseudoroute.benchmark.pseudo_embedding_route import (
    _load_valid_sample,
    _sample_paths,
    load_focused_suite_and_manifest,
    run_route_partition,
)
from pseudoroute.benchmark.runner import _software_hardware
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)

DEFAULT_CONFIG = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml")


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _gpu_compute_processes() -> list[dict[str, object]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows: list[dict[str, object]] = []
    for line in query.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4:
            raise RuntimeError(f"unexpected nvidia-smi compute row: {line}")
        pid = int(values[0])
        ppid_result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        rows.append(
            {
                "pid": pid,
                "ppid": int(ppid_result) if ppid_result.isdigit() else None,
                "gpu_uuid": values[1],
                "used_memory_mib": int(values[2]),
                "process_name": values[3],
            }
        )
    return rows


def _assert_no_other_gpu_worker() -> list[dict[str, object]]:
    rows = _gpu_compute_processes()
    foreign = [row for row in rows if row["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"another GPU compute worker is active: {foreign}")
    return rows


def preflight(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    if accuracy.fingerprint() != suite.source_accuracy.config_fingerprint:
        raise ValueError("source v17 accuracy config fingerprint changed")
    source_status_path = Path(suite.source_accuracy.artifact_root) / "pipeline_status.json"
    source_status = _json(source_status_path)
    if (
        source_status.get("state") != suite.source_accuracy.required_pipeline_state
        or source_status.get("stage") != suite.source_accuracy.required_pipeline_stage
    ):
        raise ValueError("source v17 pipeline state changed")
    default_root = Path(suite.default_vectors.root)
    tensor_path = default_root / "default_vectors.safetensors"
    default_manifest_path = default_root / "manifest.json"
    if sha256_file(tensor_path) != suite.default_vectors.tensor_sha256:
        raise ValueError("default-vector tensor checksum changed")
    if sha256_file(default_manifest_path) != suite.default_vectors.manifest_sha256:
        raise ValueError("default-vector manifest checksum changed")
    default_manifest = _json(default_manifest_path)
    if (
        default_manifest.get("fingerprint") != suite.default_vectors.artifact_fingerprint
        or default_manifest.get("definition") != suite.default_vectors.definition
        or int(default_manifest.get("zero_count_experts", -1))
        != suite.default_vectors.unobserved_layer_expert_pairs
    ):
        raise ValueError("default-vector artifact semantics changed")
    head = _git("rev-parse", "HEAD")
    required_commits = ("cbb0e60", "94b8810", "c0a96e5", "15fb7ae", "04280a3")
    for commit in required_commits:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            check=True,
            capture_output=True,
            text=True,
        )
    dirty = _git("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise RuntimeError("focused preflight requires a clean committed worktree")
    gpu_processes = _assert_no_other_gpu_worker()
    resolved = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "config": suite.model_dump(mode="json"),
        "sample_manifest": manifest.model_dump(mode="json"),
        "sample_manifest_payload_sha256": sha256_json(manifest.model_dump(mode="json")),
    }
    execution = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "git_head": head,
        "required_commits": list(required_commits),
        "worktree_clean": True,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "network_downloads_authorized": False,
        "model_and_dataset_cache_policy": "preexisting_local_cache_only",
    }
    result = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_pipeline_status": source_status,
        "source_pipeline_status_sha256": sha256_file(source_status_path),
        "default_vector_tensor_sha256": sha256_file(tensor_path),
        "default_vector_manifest_sha256": sha256_file(default_manifest_path),
        "default_vector_unobserved_pairs": int(default_manifest["zero_count_experts"]),
        "gpu_compute_processes_before_run": gpu_processes,
        "other_gpu_worker_present": False,
        "authoritative_subset_artifact_reused_not_rerun": True,
        "old_failed_markers_preserved": True,
    }
    write_json_atomic(output / "resolved_config.json", resolved)
    write_json_atomic(output / "resolved_execution_revision.json", execution)
    write_json_atomic(output / "environment.json", _software_hardware())
    write_json_atomic(output / "preflight.json", result)
    return result


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _route_variant_keys(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
    partition: str,
) -> tuple[str, ...] | None:
    if partition != "held_out_route":
        return None
    selection = _json(output / "development_selection.json")
    selected = selection.get("selected_variant")
    if not isinstance(selected, str) or not selection.get("progress_gate_pass"):
        raise RuntimeError("held-out route execution requires an eligible development variant")
    return (selected,)


def run_route_stage(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
    partition: str,
    *,
    physical_gpu: int,
) -> dict[str, object]:
    _assert_no_other_gpu_worker()
    run_route_partition(
        suite,
        manifest,
        output,
        partition,
        physical_gpu=physical_gpu,
        variant_keys=_route_variant_keys(suite, output, partition),
    )
    _release_cuda()
    return aggregate_route_partition(suite, manifest, output, partition)


def _skipped_held_out_gate(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
    reason: str,
) -> dict[str, object]:
    row = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "selected_variant": None,
        "held_out_route_gate_pass": False,
        "actual_closed_loop_authorized": False,
        "state": "not_run_by_frozen_stop_rule",
        "reason": reason,
        "decision": "STOP/PIVOT",
    }
    write_json_atomic(output / "held_out_route_gate.json", row)
    return row


def _decision(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
) -> dict[str, object]:
    mechanism = _json(output / "mechanism_smoke_audit.json")
    development = (
        _json(output / "development_selection.json")
        if (output / "development_selection.json").is_file()
        else None
    )
    held_out = (
        _json(output / "held_out_route_gate.json")
        if (output / "held_out_route_gate.json").is_file()
        else None
    )
    actual = (
        _json(output / "closed_loop" / "summary.json")
        if (output / "closed_loop" / "summary.json").is_file()
        else None
    )
    reasons = []
    if not mechanism.get("mechanism_smoke_pass"):
        reasons.append("mechanism_smoke_failed")
    if development is None or not development.get("progress_gate_pass"):
        reasons.append("development_progress_gate_failed_or_not_reached")
    if held_out is None or not held_out.get("held_out_route_gate_pass"):
        reasons.append("held_out_strong_candidate_gate_failed_or_not_reached")
    if actual is not None:
        policies = cast(dict[str, dict[str, Any]], actual["policies"])
        failed = [key for key, row in policies.items() if not row["accuracy_gate_pass"]]
        if failed:
            reasons.append(f"actual_accuracy_gate_failed:{','.join(sorted(failed))}")
    elif held_out is not None and held_out.get("held_out_route_gate_pass"):
        reasons.append("authorized_actual_closed_loop_not_completed")
    route_pass = not reasons
    label = "NARROW" if route_pass and actual is not None else "STOP/PIVOT"
    if label == "NARROW" and suite.decision.maximum_positive_label != "NARROW":
        raise AssertionError("focused decision exceeded its frozen ceiling")
    return {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "decision": label,
        "reasons": reasons,
        "mechanism_smoke_pass": bool(mechanism.get("mechanism_smoke_pass")),
        "development_progress_gate_pass": bool(
            development and development.get("progress_gate_pass")
        ),
        "held_out_route_gate_pass": bool(held_out and held_out.get("held_out_route_gate_pass")),
        "actual_closed_loop_completed": actual is not None,
        "selected_variant": development.get("selected_variant") if development else None,
        "scope": {
            "model": suite.model.model_id,
            "revision": suite.model.revision,
            "task": suite.dataset.key,
            "horizon": suite.operating_point.horizon,
            "budget": suite.operating_point.budget_per_layer,
            "samples_maximum": suite.closed_loop.sample_count,
        },
        "claim_limits": {
            "full_dataset_go": False,
            "runtime_speedup": False,
            "transfer_is_simulated": True,
            "stall_not_estimated_without_frozen_latency_model": True,
            "identity_materialized_rows": 0,
            "default_vectors_are_complete_expert_prior": False,
            "unobserved_default_layer_expert_pairs_saved_as_zero": 444,
        },
    }


def _report_markdown(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
    decision: dict[str, object],
) -> str:
    lines = [
        f"# {suite.suite_id}",
        "",
        f"Focused decision: **{decision['decision']}**.",
        "",
        "This report is limited to Qwen/GSM8K at `H=8,B=32` (32/128 routed experts per "
        "layer, native top-8). It is not a full-dataset GO or a runtime-speedup claim.",
        "",
        "## Route evidence",
        "",
    ]
    for partition in ("mechanism_smoke", "development", "held_out_route"):
        path = output / "route" / partition / "aggregates.json"
        if not path.is_file():
            continue
        lines.extend((f"### {partition}", ""))
        aggregates = _json(path)
        for method, row in sorted(aggregates.items()):
            lines.append(
                f"- `{method}`: route hit {float(row['mean_route_hit']):.6f}, selected mass "
                f"{float(row['mean_selected_mass']):.6f}, simulated transfer reduction "
                f"{float(row['estimated_transfer_reduction']):.6f}."
            )
        lines.append("")
    expected_path = output / "expected_top_m_ablation.json"
    if expected_path.is_file():
        expected = _json(expected_path)
        lines.extend(
            (
                "## Expected top-8 auxiliary",
                "",
                "The protocol-authorized expected next-token embedding was run as an auxiliary "
                "ablation. It did not enter the frozen four-variant ranking.",
                "",
                f"Measured mean probe latency: "
                f"{float(expected['cost']['mean_probe_latency_seconds_measured']):.6f} s/boundary.",
                "",
            )
        )
    actual_path = output / "closed_loop" / "summary.json"
    if actual_path.is_file():
        actual = _json(actual_path)
        lines.extend(("## Actual closed-loop generation", ""))
        lines.append(
            f"All {int(actual['sample_policy_rows'])} rows are true hard closed-loop rows; "
            "identity-materialized rows: 0."
        )
        lines.append("")
        for policy, row in sorted(actual["policies"].items()):
            lines.append(
                f"- `{policy}`: {int(row['successes'])}/{int(row['samples'])} correct, "
                "mean exact-token "
                f"agreement {float(row['mean_exact_token_agreement']):.6f}, measured runtime "
                f"{float(row['total_runtime_seconds_measured']):.2f} s."
            )
        lines.append("")
    lines.extend(
        (
            "## Measurement boundary",
            "",
            "Task accuracy, token identity, NLL/perplexity, route divergence, probe cost, and "
            "generation runtime are measured where present. Route replay is open-loop. Expert "
            "transfer bytes are simulated. Stall is not estimated because focused v1 froze no "
            "latency model. The default-vector artifact has 444 unobserved layer/expert pairs "
            "stored as zero and is not a complete expert prior.",
            "",
        )
    )
    return "\n".join(lines)


def _write_markdown_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _resume_audit(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
) -> dict[str, object]:
    route_rows = 0
    route_partitions = []
    for partition in ("mechanism_smoke", "development", "held_out_route"):
        root = output / "route" / partition
        if not root.is_dir():
            continue
        count = 0
        for reference in manifest.partitions[partition].rows:
            json_path, tensor_path = _sample_paths(output, partition, reference.row_index)
            row = _load_valid_sample(suite, json_path, tensor_path)
            if row is None or row["sample_id"] != reference.sample_id:
                raise ValueError(f"route resume audit failed: {json_path}")
            count += 1
        route_rows += count
        route_partitions.append({"partition": partition, "samples": count})
    actual_rows = 0
    for wave in (1, 2):
        for reference in manifest.partitions[f"closed_loop_wave_{wave}"].rows:
            for policy in suite.policies:
                path = closed_loop_sample_path(output, wave, reference.row_index, policy)
                row = _load_checksummed_row(path, suite)
                if row is not None:
                    if row["sample_id"] != reference.sample_id:
                        raise ValueError(f"closed-loop resume audit failed: {path}")
                    actual_rows += 1
    failed_markers = sorted(str(path.relative_to(output)) for path in output.rglob("*.FAILED.json"))
    result = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "route_sample_rows_validated": route_rows,
        "route_partitions": route_partitions,
        "actual_sample_policy_rows_validated": actual_rows,
        "failed_markers_preserved": failed_markers,
        "atomic_route_tensor_plus_json": True,
        "atomic_actual_checksummed_json": True,
        "resume_validation_pass": True,
    }
    write_json_atomic(output / "resume_audit.json", result)
    return result


def build_artifact_manifest(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded or relative.startswith("logs/"):
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return manifest


def validate_artifact_manifest(
    suite: PseudoEmbeddingSuiteConfig,
    output: Path,
) -> dict[str, object]:
    manifest = _json(output / "artifact_manifest.json")
    if manifest.get("config_fingerprint") != suite.fingerprint():
        raise ValueError("artifact manifest config mismatch")
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"artifact manifest checksum mismatch: {path}")
    return manifest


def finalize(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
) -> dict[str, object]:
    decision = _decision(suite, output)
    write_json_atomic(output / "decision.json", decision)
    _write_markdown_atomic(output / "report.md", _report_markdown(suite, output, decision))
    resume = _resume_audit(suite, manifest, output)
    artifact_manifest = build_artifact_manifest(suite, output)
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "decision": decision["decision"],
        "report": str((output / "report.md").resolve()),
        "artifact_count": artifact_manifest["artifact_count"],
        "resume_audit": resume,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(output / "pipeline_status.json", status)
    validate_artifact_manifest(suite, output)
    return status


def validate(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
) -> dict[str, object]:
    status = _json(output / "pipeline_status.json")
    if (
        status.get("state") != "complete"
        or status.get("stage") != "report_v1"
        or status.get("config_fingerprint") != suite.fingerprint()
    ):
        raise ValueError("focused pipeline is not complete/report_v1")
    resume = _resume_audit(suite, manifest, output)
    artifact_manifest = build_artifact_manifest(suite, output)
    validate_artifact_manifest(suite, output)
    return {
        "state": "valid",
        "decision": status["decision"],
        "route_rows": resume["route_sample_rows_validated"],
        "actual_rows": resume["actual_sample_policy_rows_validated"],
        "artifacts": artifact_manifest["artifact_count"],
    }


def pipeline(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
    *,
    physical_gpu: int,
    allow_second_wave_after_eta: bool,
) -> dict[str, object]:
    preflight(suite, manifest, output)
    mechanism = run_route_stage(
        suite,
        manifest,
        output,
        "mechanism_smoke",
        physical_gpu=physical_gpu,
    )
    if not cast(dict[str, Any], mechanism["gate"])["mechanism_smoke_pass"]:
        _skipped_held_out_gate(suite, output, "mechanism_smoke_failed")
        return finalize(suite, manifest, output)
    development = run_route_stage(
        suite,
        manifest,
        output,
        "development",
        physical_gpu=physical_gpu,
    )
    if not cast(dict[str, Any], development["gate"])["progress_gate_pass"]:
        _skipped_held_out_gate(suite, output, "development_progress_gate_failed")
        return finalize(suite, manifest, output)
    held_out = run_route_stage(
        suite,
        manifest,
        output,
        "held_out_route",
        physical_gpu=physical_gpu,
    )
    if not cast(dict[str, Any], held_out["gate"])["held_out_route_gate_pass"]:
        return finalize(suite, manifest, output)
    _assert_no_other_gpu_worker()
    run_closed_loop_wave(suite, manifest, output, wave=1, physical_gpu=physical_gpu)
    _release_cuda()
    eta = wave_eta_checkpoint(suite, manifest, output)
    if eta["second_wave_requires_explicit_user_confirmation"] and not allow_second_wave_after_eta:
        status = {
            "schema_version": 1,
            "state": "awaiting_user_confirmation",
            "stage": "closed_loop_wave_1_complete",
            "suite_id": suite.suite_id,
            "config_fingerprint": suite.fingerprint(),
            "eta": eta,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        write_json_atomic(output / "pipeline_status.json", status)
        return status
    if eta["second_wave_requires_explicit_user_confirmation"]:
        write_json_atomic(
            output / "closed_loop" / "second_wave_authorization.json",
            {
                "schema_version": 1,
                "suite_id": suite.suite_id,
                "config_fingerprint": suite.fingerprint(),
                "authorization_source": "explicit_cli_flag_after_human_confirmation",
                "projected_total_hours": eta["projected_16_row_three_policy_total_hours"],
                "authorized_at": datetime.now(UTC).isoformat(),
            },
        )
    _assert_no_other_gpu_worker()
    run_closed_loop_wave(suite, manifest, output, wave=2, physical_gpu=physical_gpu)
    _release_cuda()
    aggregate_closed_loop(suite, manifest, output)
    return finalize(suite, manifest, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    route = subparsers.add_parser("route")
    route.add_argument(
        "partition",
        choices=("mechanism_smoke", "development", "held_out_route"),
    )
    route.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    aggregate = subparsers.add_parser("aggregate-route")
    aggregate.add_argument(
        "partition",
        choices=("mechanism_smoke", "development", "held_out_route"),
    )
    closed = subparsers.add_parser("closed-loop")
    closed.add_argument("--wave", type=int, choices=(1, 2), required=True)
    closed.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    subparsers.add_parser("eta")
    subparsers.add_parser("aggregate-closed-loop")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    run = subparsers.add_parser("run")
    run.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    run.add_argument("--allow-second-wave-after-eta", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite, manifest = load_focused_suite_and_manifest(args.config)
    output = Path(suite.artifact_root)
    if args.command == "preflight":
        result = preflight(suite, manifest, output)
    elif args.command == "route":
        result = run_route_stage(
            suite,
            manifest,
            output,
            args.partition,
            physical_gpu=args.gpu,
        )
    elif args.command == "aggregate-route":
        result = aggregate_route_partition(suite, manifest, output, args.partition)
    elif args.command == "closed-loop":
        result = {
            "rows": len(
                run_closed_loop_wave(
                    suite,
                    manifest,
                    output,
                    wave=args.wave,
                    physical_gpu=args.gpu,
                )
            )
        }
    elif args.command == "eta":
        result = wave_eta_checkpoint(suite, manifest, output)
    elif args.command == "aggregate-closed-loop":
        result = aggregate_closed_loop(suite, manifest, output)
    elif args.command == "finalize":
        result = finalize(suite, manifest, output)
    elif args.command == "validate":
        result = validate(suite, manifest, output)
    else:
        result = pipeline(
            suite,
            manifest,
            output,
            physical_gpu=args.gpu,
            allow_second_wave_after_eta=args.allow_second_wave_after_eta,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 2 if result.get("state") == "awaiting_user_confirmation" else 0


if __name__ == "__main__":
    sys.exit(main())
