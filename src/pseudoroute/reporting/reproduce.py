"""Clean orchestration of the primary M11 result suite."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pseudoroute.reporting.aggregate import AggregationSummary, aggregate_runs
from pseudoroute.reporting.manifest import finalize_run_manifest, load_and_validate_manifest


@dataclass(frozen=True)
class SuiteTask:
    name: str
    command: str
    result_kind: str
    config: Path
    path_overrides: dict[str, str]


def load_suite(config_root: Path, suite: str) -> tuple[SuiteTask, ...]:
    path = config_root / f"{suite}.yaml"
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("reproduction suite requires schema_version 1")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("reproduction suite requires a non-empty tasks list")
    result = []
    names: set[str] = set()
    for item in tasks:
        if not isinstance(item, dict) or set(item) - {
            "name",
            "command",
            "result_kind",
            "config",
            "path_overrides",
        }:
            raise ValueError("invalid reproduction task fields")
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate reproduction task name: {name}")
        names.add(name)
        result.append(
            SuiteTask(
                name,
                str(item["command"]),
                str(item["result_kind"]),
                Path(str(item["config"])),
                {str(key): str(value) for key, value in item.get("path_overrides", {}).items()},
            )
        )
    return tuple(result)


def _set_path(config: dict[str, Any], dotted: str, value: str) -> None:
    parts = dotted.split(".")
    target = config
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"override path is not a mapping: {dotted}")
        target = child
    target[parts[-1]] = value


def reproduce_primary(
    *, suite: str, config_root: Path, output_root: Path, dry_run: bool
) -> AggregationSummary | None:
    tasks = load_suite(config_root, suite)
    estimate = {
        "suite": suite,
        "tasks": [task.name for task in tasks],
        "commands": [task.command for task in tasks],
        "output_root": str(output_root),
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return None
    runs_root = output_root / "runs"
    paper_root = output_root / "paper"
    runs_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task in tasks:
        run_root = runs_root / task.name
        if (run_root / "DONE").is_file() and (run_root / "run_manifest.json").is_file():
            manifest = load_and_validate_manifest(run_root)
            if not manifest.complete or manifest.command != task.command:
                raise ValueError(f"incompatible completed task during resume: {run_root}")
            continue
        run_root.mkdir(parents=True, exist_ok=True)
        raw: Any = yaml.safe_load(task.config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"task config root is not a mapping: {task.config}")
        for dotted, dependency in task.path_overrides.items():
            if dependency not in {candidate.name for candidate in tasks}:
                raise ValueError(f"unknown task dependency: {dependency}")
            _set_path(raw, dotted, str((runs_root / dependency).resolve()))
        staged = output_root / f".{task.name}.resolved.yaml"
        staged.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pseudoroute.cli",
                task.command,
                "--config",
                str(staged),
                "--output-dir",
                str(run_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        (run_root / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (run_root / "stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode:
            failures += 1
            (run_root / "resolved_config.json").write_text(
                json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            failure = {
                "command": task.command,
                "returncode": process.returncode,
                "stderr": process.stderr,
            }
            (run_root / "FAILED.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (run_root / "metrics.json").write_text(
                json.dumps({"failed": True, **failure}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        finalize_run_manifest(run_root, command=task.command, result_kind=task.result_kind)
        staged.unlink()
    summary = aggregate_runs(runs_root, paper_root)
    reproduction = {
        "schema_version": 1,
        **estimate,
        "included_run_ids": summary.included_run_ids,
        "incomplete_runs": summary.incomplete_runs,
        "failures": failures,
    }
    (output_root / "reproduction_manifest.json").write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"reproduction suite retained {failures} failed task(s)")
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    return summary
