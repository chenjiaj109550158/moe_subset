#!/usr/bin/env python3
"""Complete the frozen benchmark_subset_oracle_v1 pipeline with safe resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pseudoroute.benchmark.subset_config import load_subset_oracle_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/benchmark/benchmark_subset_oracle_v1.yaml"
OUTPUT = REPO / "artifacts/benchmark_subset_oracle_v1_r2_authoritative"
STATUS = OUTPUT / "pipeline_status.json"
LOGS = OUTPUT / "logs"
POLL_SECONDS = 30


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _status(state: str, stage: str, **detail: object) -> None:
    suite = load_subset_oracle_config(CONFIG)
    _atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "config_fingerprint": suite.fingerprint(),
            "state": state,
            "stage": stage,
            "updated_at": datetime.now(UTC).isoformat(),
            "orchestrator_pid": os.getpid(),
            "orchestrator_ppid": os.getppid(),
            **detail,
        },
    )


def _alive(pid: int, fragment: str) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return False
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return fragment in command


def _wait_for_resumed_workers() -> None:
    if not STATUS.is_file():
        return
    previous = json.loads(STATUS.read_text(encoding="utf-8"))
    previous_orchestrator = int(previous.get("orchestrator_pid", -1))
    if previous_orchestrator != os.getpid() and _alive(
        previous_orchestrator, "complete_subset_oracle_v1.py"
    ):
        raise RuntimeError(
            f"another subset-oracle orchestrator is still active: PID {previous_orchestrator}"
        )
    if previous.get("state") != "running":
        return
    workers = [
        worker
        for worker in previous.get("active_workers", [])
        if _alive(int(worker["pid"]), "pseudoroute.benchmark.subset_runner")
    ]
    while workers:
        _status("waiting", "resume_existing_workers", active_workers=workers)
        time.sleep(POLL_SECONDS)
        workers = [
            worker
            for worker in workers
            if _alive(int(worker["pid"]), "pseudoroute.benchmark.subset_runner")
        ]


def _gpu_processes() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid, gpu_uuid, memory = [value.strip() for value in line.split(",")]
        rows.append({"pid": int(pid), "gpu_uuid": gpu_uuid, "used_memory_mib": int(memory)})
    return rows


def _preflight() -> None:
    gpu_rows = _gpu_processes()
    if gpu_rows:
        raise RuntimeError(
            "refusing to start with pre-existing GPU compute processes; ownership is unknown: "
            f"{gpu_rows}"
        )
    process_snapshot = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _atomic_json(
        OUTPUT / "preflight.json",
        {
            "schema_version": 1,
            "checked_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "gpu_compute_processes": gpu_rows,
            "process_snapshot_sha256": __import__("hashlib")
            .sha256(process_snapshot.encode())
            .hexdigest(),
            "gpu_ownership_check": "passed_no_preexisting_compute_processes",
        },
    )


def _record_execution_revision() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    payload = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "git_head": head,
        "git_worktree_porcelain": status,
        "base_environment": "../environment.json",
    }
    path = OUTPUT / "execution_revisions" / f"{head}.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("git_head") != head or existing.get("git_worktree_porcelain") != status:
            raise RuntimeError(f"execution revision provenance changed: {path}")
        return
    _atomic_json(path, payload)


def _command(*arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pseudoroute.benchmark.subset_runner",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        *arguments,
    ]


def _offline_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _progress() -> dict[str, object]:
    counts: dict[str, int] = {}
    for stage in ("mechanism_smoke", "selected_smoke", "full"):
        counts[stage] = len(list((OUTPUT / "models").glob(f"*/closed_loop/{stage}/**/*.json")))
    trace_shards = len(list((OUTPUT / "natural_traces").glob("*/shards/**/*.safetensors")))
    return {"closed_loop_json_rows": counts, "natural_trace_shards": trace_shards}


def _run_parallel(commands: Sequence[tuple[str, int | None, list[str]]], stage: str) -> None:
    if not commands:
        _status("running", stage, active_workers=[], note="no commands required")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    started: list[tuple[subprocess.Popen[str], Any, dict[str, object]]] = []
    for label, physical_gpu, command in commands:
        log_path = LOGS / f"{stage}.{label}.log"
        stream = log_path.open("a", encoding="utf-8")
        stream.write(f"\n[{datetime.now(UTC).isoformat()}] {' '.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=_offline_environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        metadata = {
            "label": label,
            "pid": process.pid,
            "ppid": os.getpid(),
            "physical_gpu": physical_gpu,
            "command": command,
            "log": str(log_path.relative_to(OUTPUT)),
        }
        started.append((process, stream, metadata))
    try:
        while True:
            active = [metadata for process, _, metadata in started if process.poll() is None]
            _status(
                "running",
                stage,
                active_workers=active,
                workers=[metadata for _, _, metadata in started],
                progress=_progress(),
                gpu_processes=_gpu_processes(),
            )
            if not active:
                break
            time.sleep(POLL_SECONDS)
        returncodes = {
            str(metadata["label"]): process.returncode for process, _, metadata in started
        }
        if any(code != 0 for code in returncodes.values()):
            raise RuntimeError(f"{stage} worker failure: {returncodes}")
    finally:
        for _, stream, _ in started:
            stream.close()


def _run_one(stage: str, *arguments: str) -> None:
    _run_parallel([(stage, None, _command(*arguments))], stage)


def _selection() -> dict[str, dict[str, Any]]:
    path = OUTPUT / "selected_operating_points.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["model"]): row for row in result["selected"]}


def _validate_smoke(stage: str, require_lossless: bool) -> None:
    suite = load_subset_oracle_config(CONFIG)
    for model in suite.models:
        hard_changed = False
        for task in suite.trace.sample_rows:
            root = OUTPUT / "models" / model.key / "closed_loop" / stage / task
            hard = list(root.glob("*/hard_oracle_commitment.json"))
            if len(hard) != 1:
                raise RuntimeError(f"missing hard smoke: {stage}/{model.key}/{task}")
            hard_row = json.loads(hard[0].read_text(encoding="utf-8"))
            hard_changed = hard_changed or bool(hard_row["executed_route_changed"])
            if require_lossless:
                lossless = list(root.glob("*/lossless_oracle_residency.json"))
                if len(lossless) != 1:
                    raise RuntimeError(f"missing lossless smoke: {stage}/{model.key}/{task}")
                row = json.loads(lossless[0].read_text(encoding="utf-8"))
                if row["exact_token_agreement"] != 1.0 or row["executed_route_changed"]:
                    raise RuntimeError(f"lossless smoke identity failed: {lossless[0]}")
        if stage == "mechanism_smoke" and not hard_changed:
            raise RuntimeError(f"hard mechanism smoke changed no route for {model.key}")


def main() -> None:
    suite = load_subset_oracle_config(CONFIG)
    try:
        _wait_for_resumed_workers()
        _preflight()
        _record_execution_revision()
        _run_one("audit_source_v17", "audit")
        trace_commands = [
            (
                model.key,
                model.physical_gpu,
                _command("trace", "--model-key", model.key),
            )
            for model in suite.models
        ]
        _run_parallel(trace_commands, "natural_trace")
        for model in suite.models:
            _run_one(f"validate_trace_{model.key}", "validate-trace", "--model-key", model.key)
        grid_commands = [
            (model.key, None, _command("grid", "--model-key", model.key)) for model in suite.models
        ]
        _run_parallel(grid_commands, "open_loop_grid")
        _run_one("select_operating_points", "select")
        mechanism = [
            (
                model.key,
                model.physical_gpu,
                _command(
                    "closed-loop",
                    "--model-key",
                    model.key,
                    "--horizon",
                    str(suite.closed_loop.smoke_horizon),
                    "--budget",
                    str(model.native_top_k),
                    "--policies",
                    "lossless_oracle_residency,hard_oracle_commitment",
                    "--shard-index",
                    "0",
                    "--stage",
                    "mechanism_smoke",
                ),
            )
            for model in suite.models
        ]
        _run_parallel(mechanism, "mechanism_smoke")
        _validate_smoke("mechanism_smoke", require_lossless=True)
        selected = _selection()
        selected_smoke = []
        for model in suite.models:
            point = selected.get(model.key)
            if point is None:
                continue
            selected_smoke.append(
                (
                    model.key,
                    model.physical_gpu,
                    _command(
                        "closed-loop",
                        "--model-key",
                        model.key,
                        "--horizon",
                        str(point["horizon"]),
                        "--budget",
                        str(point["budget"]),
                        "--policies",
                        "lossless_oracle_residency,hard_oracle_commitment,previous_route_commitment",
                        "--shard-index",
                        "0",
                        "--stage",
                        "selected_smoke",
                    ),
                )
            )
        _run_parallel(selected_smoke, "selected_point_smoke")
        if selected_smoke:
            _validate_smoke("selected_smoke", require_lossless=True)
        for shard in range(suite.closed_loop.sample_shards):
            wave = []
            for model in suite.models:
                point = selected.get(model.key)
                if point is None:
                    continue
                wave.append(
                    (
                        model.key,
                        model.physical_gpu,
                        _command(
                            "closed-loop",
                            "--model-key",
                            model.key,
                            "--horizon",
                            str(point["horizon"]),
                            "--budget",
                            str(point["budget"]),
                            "--policies",
                            "hard_oracle_commitment,previous_route_commitment",
                            "--shard-index",
                            str(shard),
                            "--stage",
                            "full",
                        ),
                    )
                )
            _run_parallel(wave, f"full_shard_wave_{shard}")
        _run_one("aggregate", "aggregate")
        _run_one("validate", "validate")
        decision = json.loads((OUTPUT / "decision.json").read_text(encoding="utf-8"))
        _status(
            "complete",
            "report_v1",
            decision=decision["overall_decision"],
            predictor_authorization=decision["predictor_authorization"],
            progress=_progress(),
        )
    except BaseException as error:
        _status(
            "blocked",
            "failed",
            exception_type=type(error).__name__,
            message=str(error),
            progress=_progress(),
        )
        raise


if __name__ == "__main__":
    main()
