"""Idempotent preflight for a fresh run directory; all downloads are explicitly scoped."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from pseudoroute.restart.data import prepare_data
from pseudoroute.restart.state import write_json


def prepare(out: Path) -> None:
    for name in ("tests", "profiles", "runs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    if not (out / "source_manifest.json").exists():
        write_json(
            out / "source_manifest.json",
            {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "run_id": out.name,
                "workspace": str(Path.cwd()),
                "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "remote": subprocess.check_output(
                    ["git", "remote", "get-url", "origin"], text=True
                ).strip(),
            },
        )
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"])
    lock = out / "environment.lock.txt"
    if lock.exists() and lock.read_bytes() != environment:
        raise ValueError("installed environment no longer matches its lock")
    if not lock.exists():
        lock.write_bytes(environment)
    child_env = os.environ | {
        "USE_HUB_KERNELS": "0",
        "HF_HUB_DISABLE_XET": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }

    def command(label: str, args: list[str]) -> None:
        with (out / "tests" / f"{label}.log").open("w") as log:
            log.write("COMMAND: " + repr(args) + "\n")
            log.flush()
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=child_env, check=True)

    if not (out / "hardware.json").exists():
        command(
            "hardware_probe", [sys.executable, "scripts/restart/probe.py", "--run-dir", str(out)]
        )
    if not (out / "samples_manifest.json").exists():
        prepare_data(out, Path(".cache/restart/datasets"))
    manifest = out / "model_manifest.json"
    if not manifest.exists() or not json.loads(manifest.read_text()).get("weights_verified"):
        command(
            "checkpoint_acquire",
            [
                sys.executable,
                "scripts/restart/acquire.py",
                "--run-dir",
                str(out),
                "--cache",
                ".cache/restart/hub",
                "--download",
            ],
        )
    if not (out / "backend_selection.json").exists():
        command(
            "resident_microbench",
            [sys.executable, "-m", "pseudoroute.restart.microbench", "--run-dir", str(out)],
        )
