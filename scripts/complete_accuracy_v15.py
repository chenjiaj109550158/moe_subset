#!/usr/bin/env python3
"""Finish vanilla v15, enforce alignment, then materialize oracle results."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/benchmark/speculating_experts_accuracy_v15.yaml"
QWEN_SOURCE = REPO / "artifacts/speculating_experts_accuracy_v10"
GPT_SOURCE = REPO / "artifacts/speculating_experts_accuracy_v12"
OUTPUT = REPO / "artifacts/speculating_experts_accuracy_v15"
STATUS = OUTPUT / "pipeline_status.json"
EXPECTED = {
    "humaneval": 164,
    "mbpp_plus": 378,
    "gsm8k": 1319,
    "aime24": 30,
    "aime25": 30,
    "strategyqa": 687,
}
SOURCE_WORKERS = {
    516481: "qwen-v10",
    526879: "gpt-v12-shard-0",
    547581: "gpt-v12-shard-2",
    549348: "gpt-v12-shard-3",
    555003: "gpt-v12-shard-1",
}


def write_status(state: str, stage: str, **detail: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "stage": stage,
        "updated_at": datetime.now(UTC).isoformat(),
        **detail,
    }
    temporary = STATUS.with_name(f".{STATUS.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATUS)


def benchmark_worker_alive(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return False
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "pseudoroute.benchmark.runner" in command


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing result file: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_task_complete(root: Path, model: str, task: str) -> None:
    task_root = root / "models" / model / "results" / "vanilla" / task
    if not (task_root / "DONE").is_file():
        raise RuntimeError(f"missing DONE marker: {model}/{task}")
    rows = [
        row for row in jsonl_rows(task_root / "samples.jsonl") if row.get("state") == "complete"
    ]
    expected = EXPECTED[task]
    indices = [int(row["row_index"]) for row in rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if (
        len(rows) != expected
        or set(indices) != set(range(expected))
        or len(set(sample_ids)) != expected
    ):
        raise RuntimeError(f"invalid completed rows: {model}/{task}: {len(rows)}/{expected}")


def run_command(*arguments: str, stage: str) -> None:
    command = [sys.executable, "-m", "pseudoroute.benchmark.runner", *arguments]
    write_status("running", stage, command=command)
    subprocess.run(command, cwd=REPO, check=True)


def copy_model_support(source: Path, model: str) -> None:
    source_model = source / "models" / model
    target_model = OUTPUT / "models" / model
    target_model.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_model / "default_vectors",
        target_model / "default_vectors",
        dirs_exist_ok=True,
    )
    shutil.copy2(source_model / "validation.json", target_model / "validation.json")


def wait_for_sources() -> None:
    while True:
        active = [label for pid, label in SOURCE_WORKERS.items() if benchmark_worker_alive(pid)]
        if not active:
            return
        write_status("waiting", "source_vanilla", active_workers=active)
        time.sleep(30)


def validate_sources() -> None:
    failures = sorted([*QWEN_SOURCE.rglob("FAILED*.json"), *GPT_SOURCE.rglob("FAILED*.json")])
    if failures:
        raise RuntimeError(f"source workers failed: {[str(path) for path in failures]}")
    for task in EXPECTED:
        assert_task_complete(QWEN_SOURCE, "qwen3_30b_a3b", task)
        assert_task_complete(GPT_SOURCE, "gpt_oss_20b", task)


def import_vanilla() -> None:
    common = ("--config", str(CONFIG), "--output-dir", str(OUTPUT))
    run_command(
        *common,
        "--model-key",
        "qwen3_30b_a3b",
        "--tasks",
        ",".join(EXPECTED),
        "--import-compatible-vanilla-from",
        str(QWEN_SOURCE),
        "--import-only",
        stage="import_qwen_v15",
    )
    gpt_reusable = ("mbpp_plus", "gsm8k", "aime24", "aime25", "strategyqa")
    run_command(
        *common,
        "--model-key",
        "gpt_oss_20b",
        "--tasks",
        ",".join(gpt_reusable),
        "--import-compatible-vanilla-from",
        str(GPT_SOURCE),
        "--import-only",
        stage="import_gpt_non_humaneval_v15",
    )


def run_gpt_humaneval() -> None:
    common = [
        sys.executable,
        "-m",
        "pseudoroute.benchmark.runner",
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--model-key",
        "gpt_oss_20b",
        "--policies",
        "vanilla",
        "--tasks",
        "humaneval",
        "--shard-count",
        "4",
    ]
    commands = [common + ["--shard-index", str(index)] for index in range(4)]
    write_status("running", "gpt_humaneval_v15", commands=commands)
    workers = [subprocess.Popen(command, cwd=REPO) for command in commands]
    returncodes = [worker.wait() for worker in workers]
    if any(returncodes):
        raise RuntimeError(f"GPT HumanEval v15 shard failures: {returncodes}")
    assert_task_complete(OUTPUT, "gpt_oss_20b", "humaneval")


def enforce_alignment_gate() -> dict[str, Any]:
    run_command(
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--aggregate-only",
        stage="aggregate_vanilla_v15",
    )
    summary = json.loads((OUTPUT / "summary.json").read_text())
    alignment = summary["model_vanilla_alignment"]
    if alignment != {"gpt_oss_20b": True, "qwen3_30b_a3b": True}:
        failed = [
            {
                "model": row["model"],
                "task": row["task"],
                "accuracy": row["accuracy"],
                "paper": row["paper_reference_accuracy"],
                "tolerance": row.get("alignment_tolerance"),
            }
            for row in summary["rows"]
            if row["policy"] == "vanilla" and not row.get("vanilla_aligned")
        ]
        write_status("blocked", "vanilla_alignment_gate", alignment=alignment, failed=failed)
        raise RuntimeError(f"vanilla alignment gate failed: {failed}")
    return summary


def materialize_and_validate_oracle() -> None:
    run_command(
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--policies",
        "oracle_pf",
        "--materialize-oracle-only",
        stage="materialize_oracle_v15",
    )
    summary = json.loads((OUTPUT / "summary.json").read_text())
    rows = summary["rows"]
    vanilla = {(row["model"], row["task"]): row for row in rows if row["policy"] == "vanilla"}
    oracle = {(row["model"], row["task"]): row for row in rows if row["policy"] == "oracle_pf"}
    if set(vanilla) != set(oracle) or len(oracle) != 12:
        raise RuntimeError("oracle materialization does not cover all 12 model/task pairs")
    for key in vanilla:
        if vanilla[key]["accuracy"] != oracle[key]["accuracy"]:
            raise RuntimeError(f"oracle accuracy differs from vanilla: {key}")
    exact = summary["oracle_vanilla_exact_token_agreement"]
    if any(value != 1.0 for model in exact.values() for value in model.values()):
        raise RuntimeError(f"oracle token identity check failed: {exact}")
    report = {
        "schema_version": 1,
        "suite_id": summary["suite_id"],
        "config_fingerprint": summary["config_fingerprint"],
        "comparison": "oracle_pf_minus_paper_router_pf",
        "rows": [
            {
                "model": row["model"],
                "task": row["task"],
                "samples": row["samples"],
                "oracle_accuracy": row["accuracy"],
                "paper_router_pf_accuracy": row["paper_reference_accuracy"],
                "oracle_minus_paper_router_pf": row["local_minus_paper"],
            }
            for row in rows
            if row["policy"] == "oracle_pf"
        ],
    }
    (OUTPUT / "oracle_vs_paper_router_pf.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    try:
        wait_for_sources()
        write_status("running", "validate_source_artifacts")
        validate_sources()
        copy_model_support(QWEN_SOURCE, "qwen3_30b_a3b")
        copy_model_support(GPT_SOURCE, "gpt_oss_20b")
        import_vanilla()
        run_gpt_humaneval()
        enforce_alignment_gate()
        materialize_and_validate_oracle()
        write_status(
            "complete",
            "oracle_v15",
            summary=str(OUTPUT / "summary.json"),
            report=str(OUTPUT / "oracle_vs_paper_router_pf.json"),
        )
        return 0
    except BaseException as error:
        if not STATUS.is_file() or json.loads(STATUS.read_text()).get("state") != "blocked":
            write_status(
                "failed",
                "pipeline_exception",
                exception_type=type(error).__name__,
                message=str(error),
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
