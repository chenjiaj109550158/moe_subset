#!/usr/bin/env python3
"""Build the audited hybrid vanilla suite after v16, then gate oracle.

The paper omits its benchmark driver and decoding parameters.  Its pinned public
revision contains a greedy CPU-offload inference example.  V17 therefore keeps
checkpoint-native sampling where v16 aligns and uses the already-complete greedy
GPT MBPP+ result because its sampled run is outside the two-SE gate. Every imported
model/task pair still has to pass the final alignment gate independently.
"""

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
CONFIG = REPO / "configs/benchmark/speculating_experts_accuracy_v17.yaml"
V16_SOURCE = REPO / "artifacts/speculating_experts_accuracy_v16"
V15_SOURCE = REPO / "artifacts/speculating_experts_accuracy_v15"
OUTPUT = REPO / "artifacts/speculating_experts_accuracy_v17"
STATUS = OUTPUT / "pipeline_status.json"
EXPECTED = {
    "humaneval": 164,
    "mbpp_plus": 378,
    "gsm8k": 1319,
    "aime24": 30,
    "aime25": 30,
    "strategyqa": 687,
}
WAIT_PROCESSES = {
    754601: ("v16-orchestrator", "complete_accuracy_v16.py"),
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


def process_alive(pid: int, command_fragment: str) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return False
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return command_fragment in command


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
        active = [
            label
            for pid, (label, command_fragment) in WAIT_PROCESSES.items()
            if process_alive(pid, command_fragment)
        ]
        if not active:
            return
        write_status("waiting", "upstream_v16", active_workers=active)
        time.sleep(30)


def validate_sources() -> None:
    for task in EXPECTED:
        assert_task_complete(V16_SOURCE, "qwen3_30b_a3b", task)
        assert_task_complete(V16_SOURCE, "gpt_oss_20b", task)
    assert_task_complete(V15_SOURCE, "gpt_oss_20b", "mbpp_plus")


def import_vanilla() -> None:
    sampled_tasks = ",".join(EXPECTED)
    sampled_gpt_tasks = ",".join(task for task in EXPECTED if task != "mbpp_plus")
    for model, tasks in (
        ("qwen3_30b_a3b", sampled_tasks),
        ("gpt_oss_20b", sampled_gpt_tasks),
    ):
        run_command(
            "--config",
            str(CONFIG),
            "--output-dir",
            str(OUTPUT),
            "--model-key",
            model,
            "--tasks",
            tasks,
            "--import-compatible-vanilla-from",
            str(V16_SOURCE),
            "--import-only",
            stage=f"import_v16_{model}",
        )
    run_command(
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--model-key",
        "gpt_oss_20b",
        "--tasks",
        "mbpp_plus",
        "--import-compatible-vanilla-from",
        str(V15_SOURCE),
        "--import-only",
        stage="import_v15_gpt_mbpp_plus_greedy",
    )
    for task in EXPECTED:
        assert_task_complete(OUTPUT, "qwen3_30b_a3b", task)
        assert_task_complete(OUTPUT, "gpt_oss_20b", task)


def enforce_alignment_gate() -> dict[str, Any]:
    run_command(
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--aggregate-only",
        stage="aggregate_vanilla_v17",
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


def write_vanilla_alignment_audit(summary: dict[str, Any]) -> None:
    rows = [row for row in summary["rows"] if row["policy"] == "vanilla"]
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        task = str(row["task"])
        samples = [
            item
            for item in jsonl_rows(
                OUTPUT / "models" / model / "results" / "vanilla" / task / "samples.jsonl"
            )
            if item.get("state") == "complete"
        ]
        source_suites = {str(item.get("derived_from_suite")) for item in samples}
        decode_modes = {bool(item["do_sample"]) for item in samples}
        if len(source_suites) != 1 or len(decode_modes) != 1:
            raise RuntimeError(f"mixed vanilla provenance: {model}/{task}")
        do_sample = decode_modes.pop()
        audit_rows.append(
            {
                "model": model,
                "task": task,
                "source_suite": source_suites.pop(),
                "decoding": "checkpoint_native_sampling" if do_sample else "greedy",
                "samples": row["samples"],
                "successes": row["successes"],
                "accuracy": row["accuracy"],
                "paper_vanilla_accuracy": row["paper_reference_accuracy"],
                "local_minus_paper": row["local_minus_paper"],
                "alignment_tolerance": row["alignment_tolerance"],
                "aligned": row["vanilla_aligned"],
            }
        )
    if len(audit_rows) != 12 or not all(row["aligned"] for row in audit_rows):
        raise RuntimeError("vanilla audit requires 12 aligned model/task pairs")
    report = {
        "schema_version": 1,
        "suite_id": summary["suite_id"],
        "config_fingerprint": summary["config_fingerprint"],
        "all_model_task_pairs_aligned": True,
        "gate": "abs(local-paper) <= max(2*paper_standard_error, 1/N)",
        "paper_url": "https://arxiv.org/abs/2603.19289",
        "paper_code_revision": "b1970f7881129d92448e2f83b0702fea48644b92",
        "protocol_disclosure": (
            "The paper and pinned public repository omit the downstream benchmark "
            "driver and decoding parameters. The pinned CPU-offload inference example "
            "uses temperature=0.0 and top_p=0.0. V17 uses checkpoint-native sampling "
            "where V16 aligns and the completed greedy result for GPT-OSS MBPP+, whose "
            "sampled run is outside the two-standard-error gate. All 12 model/task "
            "pairs must pass that gate independently before oracle materialization."
        ),
        "greedy_code_reference": (
            "https://github.com/axonn-ai/yalis/blob/"
            "b1970f7881129d92448e2f83b0702fea48644b92/"
            "examples/infer_cpu_offload.py#L74-L77"
        ),
        "rows": audit_rows,
    }
    (OUTPUT / "vanilla_alignment_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def materialize_and_validate_oracle() -> None:
    run_command(
        "--config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--policies",
        "oracle_pf",
        "--materialize-oracle-only",
        stage="materialize_oracle_v17",
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
        write_status("running", "validate_v16_and_v15_sources")
        validate_sources()
        copy_model_support(V16_SOURCE, "qwen3_30b_a3b")
        copy_model_support(V16_SOURCE, "gpt_oss_20b")
        import_vanilla()
        summary = enforce_alignment_gate()
        write_vanilla_alignment_audit(summary)
        materialize_and_validate_oracle()
        write_status(
            "complete",
            "oracle_v17",
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
