import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pseudoroute.restart.state import guarded_run_dir


def test_restart_clis_help_and_blocked_resume():
    root = Path.cwd()
    for module in ("pseudoroute.restart.run_all", "pseudoroute.restart.verify"):
        p = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
        assert p.returncode == 0 and "--run-dir" in p.stdout
    from pseudoroute.restart.run_all import PHASES

    (root / "artifacts/restart_v1").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=root / "artifacts/restart_v1", prefix="test-resume-"
    ) as temporary:
        out = Path(temporary)
        fixtures = {
            "hardware.json": {"test_fixture": True},
            "source_manifest.json": {"commit": "test_fixture_not_measurement"},
            "samples_manifest.json": {"development_ids": [], "confirmation_ids": []},
            "prompts.json": [],
            "evaluator_targets.json": [],
            "model_manifest.json": {"weights_verified": False, "state": "TEST_FIXTURE"},
            "STATUS.json": {
                "run_id": out.name,
                "terminal_state": "BLOCKED",
                "phase": "P0_BOOTSTRAP",
                "wall_seconds_used": 0,
                "gpu_seconds_used": 0,
                "last_error": None,
                "phases": {p: {"status": "PENDING", "attempts": 0, "outputs": []} for p in PHASES},
            },
        }
        for name, value in fixtures.items():
            (out / name).write_text(json.dumps(value))
        (out / "environment.lock.txt").write_text("test_fixture\n")
        command = [
            sys.executable,
            "-m",
            "pseudoroute.restart.run_all",
            "--config",
            "configs/restart/restart_v1.yaml",
            "--run-dir",
            str(out),
            "--resume",
        ]
        for attempt in (1, 2):
            p = subprocess.run(
                command, capture_output=True, text=True, env=os.environ | {"USE_HUB_KERNELS": "0"}
            )
            assert p.returncode == 2, p.stdout + p.stderr
            status = json.loads((out / "STATUS.json").read_text())
            assert status["phases"]["P0_BOOTSTRAP"]["attempts"] == attempt
            decision = json.loads((out / "DECISION.json").read_text())
            assert decision["overall"] == "BLOCKED"
            assert decision["primary_metrics"]["decode_speedup"] is None
        p = subprocess.run(
            [sys.executable, "-m", "pseudoroute.restart.verify", "--run-dir", str(out)],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stdout + p.stderr


def test_historical_output_root_rejected():
    import pytest

    with pytest.raises(ValueError):
        guarded_run_dir(Path("artifacts/qwen_penultimate_joint_offload_speed_v2"), Path.cwd())
