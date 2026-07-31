"""CLI for the versioned, resumable benchmark subset-oracle suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import torch
import transformers
from huggingface_hub.constants import HF_HUB_CACHE

from pseudoroute.benchmark.config import (
    AccuracySuiteConfig,
    load_accuracy_suite_config,
)
from pseudoroute.benchmark.subset_closed_loop import (
    ActualPolicy,
    run_model_closed_loop,
)
from pseudoroute.benchmark.subset_config import (
    SubsetModelConfig,
    SubsetOracleSuiteConfig,
    TaskKey,
    load_subset_oracle_config,
)
from pseudoroute.benchmark.subset_grid import (
    run_open_loop_grid,
    select_operating_points,
)
from pseudoroute.benchmark.subset_report import (
    aggregate_and_validate,
    validate_artifact_manifest,
)
from pseudoroute.benchmark.subset_trace import (
    audit_source_accuracy,
    collect_model_traces,
    sha256_file,
    validate_trace_manifest,
    write_json_atomic,
)

GPT_KERNEL_REPOSITORY = "kernels-community/gpt-oss-triton-kernels"
GPT_KERNEL_REVISION = "9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0"


def _configure_cached_gpt_kernel(suite: SubsetOracleSuiteConfig, output: Path) -> dict[str, object]:
    """Use the already-cached CUDA variant without resolving a Hub version."""
    snapshot = (
        Path(HF_HUB_CACHE)
        / "kernels--kernels-community--gpt-oss-triton-kernels"
        / "snapshots"
        / GPT_KERNEL_REVISION
    )
    variant = snapshot / "build" / "torch-cuda"
    if not (variant / "__init__.py").is_file():
        raise FileNotFoundError(
            "the pinned GPT-OSS Triton kernel CUDA variant is not present in local cache; "
            "downloads are prohibited for this run"
        )
    override = f"{GPT_KERNEL_REPOSITORY}={snapshot}"
    existing = os.environ.get("LOCAL_KERNELS")
    os.environ["LOCAL_KERNELS"] = f"{existing}:{override}" if existing else override
    files = [path for path in sorted(variant.rglob("*")) if path.is_file()]
    provenance: dict[str, object] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "repository": GPT_KERNEL_REPOSITORY,
        "revision": GPT_KERNEL_REVISION,
        "snapshot": str(snapshot),
        "variant": "torch-cuda",
        "source": "preexisting_local_cache",
        "network_download": False,
        "files": [
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    write_json_atomic(output / "gpt_mxfp4_kernel_provenance.json", provenance)
    return provenance


DEFAULT_CONFIG = Path("configs/benchmark/benchmark_subset_oracle_v1.yaml")
DEFAULT_OUTPUT = Path("artifacts/benchmark_subset_oracle_v1_r2_authoritative")


def _load(
    config_path: Path,
) -> tuple[SubsetOracleSuiteConfig, AccuracySuiteConfig]:
    suite = load_subset_oracle_config(config_path)
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    if accuracy.suite_id != suite.source_accuracy.suite_id:
        raise ValueError("source accuracy suite ID differs from the frozen subset config")
    if accuracy.fingerprint() != suite.source_accuracy.config_fingerprint:
        raise ValueError("source accuracy config fingerprint differs from the frozen value")
    return suite, accuracy


def _model(suite: SubsetOracleSuiteConfig, key: str) -> SubsetModelConfig:
    matches = [model for model in suite.models if model.key == key]
    if len(matches) != 1:
        raise ValueError(f"unknown model key: {key}")
    return matches[0]


def _write_resolved(
    config_path: Path,
    output: Path,
    suite: SubsetOracleSuiteConfig,
    accuracy: AccuracySuiteConfig,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.json"
    resolved = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_path": str(config_path.resolve()),
        "config_fingerprint": suite.fingerprint(),
        "config": suite.model_dump(mode="json"),
        "source_accuracy_fingerprint": accuracy.fingerprint(),
        "source_accuracy_resolved": accuracy.model_dump(mode="json"),
    }
    if resolved_path.is_file():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise ValueError("resolved config artifact differs from the current frozen config")
    else:
        write_json_atomic(resolved_path, resolved)
    environment_path = output / "environment.json"
    if environment_path.is_file():
        return
    repository = Path(__file__).resolve().parents[3]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout
    write_json_atomic(
        environment_path,
        {
            "schema_version": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                {
                    "logical": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "git_head": git_head,
            "git_worktree_porcelain": git_status,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit")

    for command in ("trace", "grid", "validate-trace"):
        child = subparsers.add_parser(command)
        child.add_argument("--model-key", required=True)

    subparsers.add_parser("select")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument(
        "--actual-policies",
        default="hard_oracle_commitment,previous_route_commitment",
        help="comma-separated full-stage policies required by aggregation",
    )
    aggregate.add_argument(
        "--tasks",
        help="comma-separated frozen task keys required by this execution scope",
    )
    aggregate.add_argument("--execution-scope-id")
    aggregate.add_argument("--execution-scope-fingerprint")
    subparsers.add_parser("validate")

    closed = subparsers.add_parser("closed-loop")
    closed.add_argument("--model-key", required=True)
    closed.add_argument("--horizon", type=int, required=True)
    closed.add_argument("--budget", type=int, required=True)
    closed.add_argument(
        "--policies",
        required=True,
        help="comma-separated actual policies (natural_v17_reuse is not executable)",
    )
    closed.add_argument("--shard-index", type=int, required=True)
    closed.add_argument(
        "--stage",
        required=True,
        choices=("mechanism_smoke", "selected_smoke", "full"),
    )
    closed.add_argument("--physical-gpu", type=int, choices=(0, 1))
    closed.add_argument(
        "--tasks",
        help="comma-separated frozen task keys; omitted means every frozen task",
    )
    return parser.parse_args()


def _task_filter(raw: str | None, accuracy: AccuracySuiteConfig) -> tuple[TaskKey, ...] | None:
    if raw is None:
        return None
    values = tuple(value for value in raw.split(",") if value)
    available = {dataset.key for dataset in accuracy.datasets}
    if not values or len(set(values)) != len(values) or not set(values).issubset(available):
        raise ValueError(f"invalid task filter: {values}")
    return cast(tuple[TaskKey, ...], values)


def main() -> None:
    args = parse_args()
    config_path = cast(Path, args.config)
    output = cast(Path, args.output_dir)
    suite, accuracy = _load(config_path)
    command = cast(str, args.command)
    if command in {"trace", "closed-loop"} and cast(str, args.model_key) == "gpt_oss_20b":
        _configure_cached_gpt_kernel(suite, output)
    _write_resolved(config_path, output, suite, accuracy)
    result: object
    if command == "audit":
        result = audit_source_accuracy(suite, accuracy, output)
    elif command == "trace":
        result = collect_model_traces(
            suite, accuracy, _model(suite, cast(str, args.model_key)), output
        )
    elif command == "validate-trace":
        result = validate_trace_manifest(suite, _model(suite, cast(str, args.model_key)), output)
    elif command == "grid":
        result = run_open_loop_grid(suite, _model(suite, cast(str, args.model_key)), output)
    elif command == "select":
        result = select_operating_points(suite, output)
    elif command == "aggregate":
        for model in suite.models:
            validate_trace_manifest(suite, model, output)
            run_open_loop_grid(suite, model, output)
        allowed_aggregate = {
            "hard_oracle_commitment",
            "previous_route_commitment",
        }
        aggregate_policies = tuple(value for value in str(args.actual_policies).split(",") if value)
        if "hard_oracle_commitment" not in aggregate_policies or not set(
            aggregate_policies
        ).issubset(allowed_aggregate):
            raise ValueError(f"invalid aggregate policies: {aggregate_policies}")
        aggregate_tasks = _task_filter(cast(str | None, args.tasks), accuracy)
        result = aggregate_and_validate(
            suite,
            accuracy,
            output,
            actual_policies=aggregate_policies,
            tasks=aggregate_tasks,
            execution_scope_id=cast(str | None, args.execution_scope_id),
            execution_scope_fingerprint=cast(str | None, args.execution_scope_fingerprint),
        )
    elif command == "validate":
        for model in suite.models:
            validate_trace_manifest(suite, model, output)
            run_open_loop_grid(suite, model, output)
        validate_artifact_manifest(suite, output)
        result = {"validation": "passed"}
    elif command == "closed-loop":
        model = _model(suite, cast(str, args.model_key))
        horizon = int(args.horizon)
        budget = int(args.budget)
        if horizon not in suite.trace.horizons:
            raise ValueError(f"horizon is outside frozen grid: {horizon}")
        if budget not in model.budgets:
            raise ValueError(f"budget is outside frozen grid for {model.key}: {budget}")
        allowed = {
            "lossless_oracle_residency",
            "hard_oracle_commitment",
            "previous_route_commitment",
        }
        raw_policies = tuple(value for value in str(args.policies).split(",") if value)
        if not raw_policies or not set(raw_policies).issubset(allowed):
            raise ValueError(f"invalid actual closed-loop policies: {raw_policies}")
        stage = cast(Literal["mechanism_smoke", "selected_smoke", "full"], str(args.stage))
        tasks = _task_filter(cast(str | None, args.tasks), accuracy)
        result = run_model_closed_loop(
            suite,
            accuracy,
            model,
            output,
            horizon=horizon,
            budget=budget,
            policies=cast(tuple[ActualPolicy, ...], raw_policies),
            shard_index=int(args.shard_index),
            stage=stage,
            physical_gpu=cast(int | None, args.physical_gpu),
            tasks=tasks,
        )
    else:
        raise AssertionError(f"unhandled command: {command}")
    print(
        json.dumps(
            {
                "state": "complete",
                "command": command,
                "suite_id": suite.suite_id,
                "config_fingerprint": suite.fingerprint(),
                "result_type": type(result).__name__,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
