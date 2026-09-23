"""Bounded restart driver. Every measured row is atomic and tied to frozen identities."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from pseudoroute.benchmark.scoring import _score_gsm8k
from pseudoroute.restart.backend import MoEComputeBackend
from pseudoroute.restart.diagnostics import real_model_checks
from pseudoroute.restart.inference import generate
from pseudoroute.restart.model import PolicyRuntime, load_cpu_first
from pseudoroute.restart.state import (
    digest,
    guarded_run_dir,
    load_row,
    object_hash,
    run_lock,
    save_row,
    write_json,
)

PHASES = (
    "P0_BOOTSTRAP",
    "P1_AUDIT_FIX",
    "P2_GPU_BACKEND",
    "P3_RUNTIME_DIAGNOSIS",
    "P4_POLICY_DEV",
    "P5_CONFIRM",
    "P6_FINAL_AUDIT",
)


def utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def source_hashes() -> dict[str, str]:
    return {
        str(p): digest(p)
        for parent in ("src", "configs/restart", "scripts/restart", "tests/restart", "docs/restart")
        for p in sorted(Path(parent).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml", ".sh", ".json", ".txt", ".md")
    }


class Runner:
    def __init__(self, config: Path, out: Path, resume: bool) -> None:
        self.config_path = config
        self.protocol = yaml.safe_load(config.read_text())
        self.out = out
        self.start = time.time()
        self.gpu_start: float | None = None
        self.status: dict[str, Any] = {
            "run_id": out.name,
            "started_at_utc": utc(),
            "terminal_state": None,
            "phase": "P0_BOOTSTRAP",
            "phases": {p: {"status": "PENDING", "attempts": 0, "outputs": []} for p in PHASES},
            "last_error": None,
            "wall_seconds_used": 0.0,
            "gpu_seconds_used": 0.0,
            "next_command": None,
        }
        old = out / "STATUS.json"
        if old.exists():
            if not resume:
                raise FileExistsError("run exists; use --resume")
            self.status = json.loads(old.read_text())
            self.status["terminal_state"] = None
        self.prior_wall = float(self.status.get("wall_seconds_used", 0))
        if not old.exists():
            origin = json.loads((out / "source_manifest.json").read_text()).get("started_at_utc")
            if origin:
                self.prior_wall = max(
                    0.0, time.time() - dt.datetime.fromisoformat(origin).timestamp()
                )
        self.prior_gpu = float(self.status.get("gpu_seconds_used", 0))
        # Historical preflight worker start/stop was not centrally logged. Charge its
        # entire elapsed wall interval conservatively, separately from measured main worker time.
        self.status.setdefault("preflight_gpu_seconds_budget_upper_bound", self.prior_wall)
        self.hashes = source_hashes()
        self.identity = object_hash(
            {
                "source": self.hashes,
                "config": digest(config),
                "env": digest(out / "environment.lock.txt"),
                "hardware": digest(out / "hardware.json"),
                "model": self.protocol["model"],
                "samples": digest(out / "samples_manifest.json"),
                "correctness_protocol": digest(out / "correctness_protocol.json")
                if (out / "correctness_protocol.json").exists()
                else None,
            }
        )
        prior = self.status.get("identity")
        if prior is not None and prior != self.identity:
            raise ValueError(
                "source/config/environment identity changed: create a new run or explicitly invalidate affected rows; refusing mixed resume"  # noqa: E501
            )
        self.status["last_error"] = None
        self.status["identity"] = self.identity
        self.status["hashes"] = {
            "source": object_hash(self.hashes),
            "config": digest(config),
            "environment": digest(out / "environment.lock.txt"),
            "samples": digest(out / "samples_manifest.json"),
        }
        write_json(out / "execution_source_hashes.json", self.hashes)
        self.samples = json.loads((out / "samples_manifest.json").read_text())
        self.prompts = {
            x["sample_id"]: x["user_prompt"] for x in json.loads((out / "prompts.json").read_text())
        }
        # Evaluator targets never cross the generate()/runtime/selector API boundary.
        self.targets = {
            x["sample_id"]: x["target"]
            for x in json.loads((out / "evaluator_targets.json").read_text())
        }
        self.rows: list[dict[str, Any]] = []
        self.model: Any = None
        self.tokenizer: Any = None
        self.runtime: PolicyRuntime
        self.cap = 512
        self.persist()

    def persist(self) -> None:
        self.status["wall_seconds_used"] = self.prior_wall + time.time() - self.start
        self.status["gpu_seconds_used"] = self.prior_gpu + (
            time.time() - self.gpu_start if self.gpu_start else 0
        )
        write_json(self.out / "STATUS.json", self.status)
        (self.out / "STATUS.md").write_text(
            "\n".join(
                [
                    f"Run: {self.out.name}",
                    f"State: {self.status['terminal_state'] or 'RUNNING'}",
                    f"Phase: {self.status['phase']}",
                    *[f"- {p}: {v['status']}" for p, v in self.status["phases"].items()],
                ]
            )
        )

    def phase(self, name: str, state: str, outputs: list[str] | None = None) -> None:
        self.status["phase"] = name
        p = self.status["phases"][name]
        p["status"] = state
        if state == "RUNNING":
            p["attempts"] += 1
            p["started_at_utc"] = utc()
        else:
            p["completed_at_utc"] = utc()
        if state == "PASS":
            row_paths = sorted((self.out / "runs").rglob("*.json"))
            evidence_name = f"{name}_evidence.json"
            write_json(
                self.out / evidence_name,
                {
                    "phase": name,
                    "rows": [
                        {"path": str(path.relative_to(self.out)), "sha256": digest(path)}
                        for path in row_paths
                    ],
                },
            )
            outputs = [*(outputs or []), evidence_name]
        if outputs:
            p["outputs"] = [{"path": x, "sha256": digest(self.out / x)} for x in outputs]
        self.persist()
        print(f"{utc()} {name}: {state}", flush=True)

    def budget(self, estimate: float = 0) -> None:
        self.persist()
        r = self.protocol["resources"]
        fraction = 1 - r["reserve_final_fraction"]
        if (
            self.status["wall_seconds_used"] + estimate > r["max_wall_hours"] * 3600 * fraction
            or self.status["gpu_seconds_used"]
            + self.status["preflight_gpu_seconds_budget_upper_bound"]
            + estimate
            > r["max_gpu_hours"] * 3600 * fraction
        ):
            raise TimeoutError("new work would enter the reserved finalization budget")

    def encoded(self, sid: str) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self.prompts[sid]}],
                tokenize=True,
                return_tensors="pt",
                return_dict=False,
                add_generation_prompt=True,
                current_date="2026-03-09",
            ).to("cuda:0"),
        )

    def row(
        self,
        stage: str,
        sid: str,
        policy: str,
        rep: int = 0,
        fixed: int | None = None,
        backend: str = "vllm_fused",
    ) -> dict[str, Any]:
        key = f"{stage}/{sid}/{policy}_{backend}_{rep}.json"
        path = self.out / "runs" / key
        previous = load_row(path, self.identity)
        if previous is not None:
            self.rows.append(previous)
            return previous
        self.budget(300)
        import shutil

        r = self.protocol["resources"]
        if shutil.disk_usage(self.out).free < r["minimum_free_disk_headroom_gib"] * 2**30:
            raise OSError("disk free space entered required headroom")
        if (
            sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
            > r["maximum_artifacts_gib"] * 2**30
        ):
            raise OSError("run artifact capacity limit reached")
        self.status["next_command"] = (
            f"{sys.executable} -m pseudoroute.restart.run_all --config {self.config_path} --run-dir {self.out} --resume"  # noqa: E501
        )
        self.status["step"] = key
        self.persist()
        self.runtime.manager.backend = MoEComputeBackend(backend)
        started = time.time()
        value = generate(
            self.model,
            self.tokenizer,
            self.runtime,
            self.encoded(sid),
            policy=policy,
            fixed_forwards=fixed,
            max_new_tokens=self.cap,
        )
        if fixed is None:
            score = _score_gsm8k(value["text"], self.targets[sid])
            value.update(correct=score.correct, parsed_answer=score.parsed_answer)
        value.update(
            identity=self.identity,
            state="COMPLETE",
            stage=stage,
            sample_id=sid,
            repetition=rep,
            backend=backend,
            started_at_utc=dt.datetime.fromtimestamp(started, dt.UTC).isoformat(),
            completed_at_utc=utc(),
            dtype="bfloat16",
            slots_per_layer=32,
            model_id=self.protocol["model"]["id"],
            model_revision=self.protocol["model"]["revision"],
            attention="sdpa",
            max_new_tokens=self.cap,
        )
        save_row(path, value)
        self.rows.append(value)
        self.persist()
        print(
            f"{utc()} {key}: {value['production_forwards_per_second']:.3f} fwd/s; correct={value.get('correct')}",  # noqa: E501
            flush=True,
        )
        return value

    def matrix(
        self, stage: str, ids: list[str], policies: list[str], reps: int, fixed: int | None
    ) -> list[dict[str, Any]]:
        rows = []
        rng = random.Random(20260922 + sum(map(ord, stage)))
        schedule = []
        for rep in range(reps):
            for i, sid in enumerate(ids):
                order = list(policies)
                offset = (i + rep) % len(order)
                order = order[offset:] + order[:offset]
                if rep % 2:
                    order.reverse()
                schedule.append([(sid, p, rep) for p in order])
        rng.shuffle(schedule)
        write_json(self.out / f"{stage}_order.json", schedule)
        for block in schedule:
            for sid, policy, rep in block:
                rows.append(self.row(stage, sid, policy, rep, fixed))
        return rows

    def run(self) -> None:
        self.phase("P0_BOOTSTRAP", "RUNNING")
        model_manifest = json.loads((self.out / "model_manifest.json").read_text())
        if not model_manifest.get("weights_verified"):
            raise FileNotFoundError(
                "specified checkpoint download/hash verification incomplete; see model_download.log"
            )
        if not torch.cuda.is_available():
            raise OSError("requested CUDA unavailable")
        installed = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"])
        if installed != (self.out / "environment.lock.txt").read_bytes():
            raise RuntimeError(
                "installed environment changed from the frozen measurement environment"
            )
        self.phase(
            "P0_BOOTSTRAP",
            "PASS",
            [
                "hardware.json",
                "environment.lock.txt",
                "model_manifest.json",
                "samples_manifest.json",
            ],
        )
        write_json(
            self.out / "cost_estimate_P0.json",
            {
                "kind": "SCHEDULING_ESTIMATE_NOT_MEASUREMENT",
                "main_runtime_estimate": None,
                "reason": "model throughput not yet observed",
                "resource_envelope": json.loads(
                    (self.out / "resource_resolution.json").read_text()
                ),
                "preflight_gpu_seconds_budget_upper_bound": self.status[
                    "preflight_gpu_seconds_budget_upper_bound"
                ],
            },
        )
        self.phase("P1_AUDIT_FIX", "RUNNING")
        self.gpu_start = time.time()
        self.model, self.tokenizer, self.runtime = load_cpu_first(
            model_manifest["snapshot_path"], manifest_path=self.out / "loader_manifest.json"
        )
        resources = self.protocol["resources"]
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        if free_gpu < max(
            resources["minimum_gpu_headroom_gib"] * 2**30,
            resources["minimum_gpu_headroom_fraction"] * total_gpu,
        ):
            raise OSError("model setup exceeds required common GPU headroom")
        mem = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
        }
        if mem["MemAvailable"] < resources["minimum_host_headroom_gib"] * 2**30:
            raise OSError("model setup exceeds required host memory headroom")
        frozen = {
            **self.protocol,
            "resolved": {
                "slots": 32,
                "d": 8,
                "backend": "vllm_fused",
                "host_mode": self.runtime.manager.host_mode,
                "baseline_scope": "exact_demand_LRU_only",
                "candidate_configs": [
                    "S",
                    "H",
                    "HC",
                    "P8_diagnostic",
                    "P4",
                    "PW4_conditional",
                    "PM4_conditional",
                ],
                "identity": self.identity,
                "confirmation_claims": "resolve after development, before confirmation",
                "sample_manifest_sha256": digest(self.out / "samples_manifest.json"),
                "correctness_protocol_sha256": digest(self.out / "correctness_protocol.json"),
            },
        }
        if not (self.out / "protocol_frozen_1.yaml").exists():
            (self.out / "protocol_frozen_1.yaml").write_text(
                yaml.safe_dump(frozen, sort_keys=False)
            )
        write_json(
            self.out / "development_plan.json",
            {
                "max_candidates": 10,
                "actual_registered_configs": 7,
                "required": ["S", "H", "HC"],
                "diagnostic": ["P8", "P4"],
                "conditional": ["PW4", "PM4"],
                "pseudo_promotion": "strict first-eight accuracy gain over best H/HC; PW4 screened only after P4 passes; before confirmation",  # noqa: E501
                "not_implemented": ["EP: exact_demand_LRU_only baseline scope"],
                "no_parameter_search": True,
            },
        )
        first = self.samples["development_ids"][0]
        if not (self.out / "tests/real_smoke_P4.json").exists():
            real_model_checks(
                self.model,
                self.tokenizer,
                self.runtime,
                self.encoded(first),
                self.out,
                additional_prompts=[
                    self.encoded(sid) for sid in self.samples["development_ids"][1:3]
                ],
            )
        if json.loads((self.out / "real_model_correctness.json").read_text())["state"] != "PASS":
            raise RuntimeError("unresolved model correctness gate")
        resource = json.loads((self.out / "resource_resolution.json").read_text())
        resource.update(
            real_model_ready=True,
            host_mode=self.runtime.manager.host_mode,
            reason="checkpoint, CPU-first load, fixed-route numerical and window smokes passed",
        )
        write_json(self.out / "resource_resolution.json", resource)
        self.phase("P1_AUDIT_FIX", "PASS", ["real_model_correctness.json", "loader_manifest.json"])
        self.phase("P2_GPU_BACKEND", "RUNNING")
        if not json.loads((self.out / "backend_selection.json").read_text()).get(
            "correctness_pass"
        ):
            raise RuntimeError("optimized backend numerical gate not passed")
        self.phase(
            "P2_GPU_BACKEND",
            "PASS",
            ["backend_selection.json", "backend_correctness.json", "moe_microbench.csv"],
        )
        smoke = [json.loads(p.read_text()) for p in (self.out / "tests").glob("real_smoke_*.json")]
        write_json(
            self.out / "cost_estimate_P2.json",
            {
                "kind": "SCHEDULING_ESTIMATE_NOT_MEASUREMENT",
                "smoke_rows": len(smoke),
                "observed_min_smoke_tps": min(r["production_forwards_per_second"] for r in smoke),
                "upper_token_workload_3_policy_N256_seconds": (32 + 256)
                * 3
                * 512
                / min(r["production_forwards_per_second"] for r in smoke),
                "confirmation_n_not_yet_selected": True,
            },
        )
        self.phase("P3_RUNTIME_DIAGNOSIS", "RUNNING")
        for backend in ("python_reference", "vllm_fused"):
            for policy in ("E", "S"):
                self.row("runtime_grid", first, policy, fixed=16, backend=backend)
        self.row("runtime_grid", first, "H", fixed=16)
        from pseudoroute.restart.diagnostics import profile_runtime

        profile_runtime(self.model, self.tokenizer, self.runtime, self.encoded(first), self.out)
        self.phase("P3_RUNTIME_DIAGNOSIS", "PASS", ["profiles/runtime_summary.json"])

        self.phase("P4_POLICY_DEV", "RUNNING")
        dev = self.samples["development_ids"]
        self.matrix("development_performance", dev[:8], ["E", "S", "H", "HC"], 3, 128)
        dev_quality = self.matrix("development_quality", dev, ["E", "S", "H", "HC"], 1, None)
        trunc = sum(r["truncated"] for r in dev_quality if r["policy"] == "E") / 32
        if trunc > 0.05:
            self.cap = 1024
            dev_quality = self.matrix(
                "development_quality_cap1024", dev, ["E", "S", "H", "HC"], 1, None
            )
        # P8 and P4 are diagnostic; no unbounded pseudo sweep.
        self.matrix("pseudo_diagnostic_performance", dev[:2], ["P8", "P4"], 1, 128)
        self.matrix("pseudo_diagnostic_quality", dev[:2], ["P8", "P4"], 1, None)
        pseudo_quality = self.matrix("pseudo_screen_quality", dev[:8], ["P4"], 1, None)
        correct = {
            p: sum(r["correct"] for r in dev_quality if r["policy"] == p)
            for p in ["E", "S", "H", "HC"]
        }
        p4_correct = sum(r["correct"] for r in pseudo_quality)
        history8 = max(
            sum(r["correct"] for r in dev_quality if r["policy"] == p and r["sample_id"] in dev[:8])
            for p in ["H", "HC"]
        )
        candidates = ["H", "HC"]
        if p4_correct > history8:
            promoted = self.matrix("pseudo_development_quality", dev, ["P4"], 1, None)
            self.matrix("pseudo_development_performance", dev[:8], ["P4"], 3, 128)
            correct["P4"] = sum(r["correct"] for r in promoted)
            candidates.append("P4")
            weighted_screen = self.matrix("weighted_pseudo_screen", dev[:8], ["PW4"], 1, None)
            if sum(r["correct"] for r in weighted_screen) >= p4_correct:
                weighted = self.matrix("pseudo_development_quality", dev, ["PW4"], 1, None)
                self.matrix("pseudo_development_performance", dev[:8], ["PW4"], 3, 128)
                correct["PW4"] = sum(r["correct"] for r in weighted)
                candidates.append("PW4")
            self.matrix("pseudo_semantics_ablation", dev[:2], ["PM4"], 1, 128)
            self.matrix("pseudo_semantics_quality", dev[:2], ["PM4"], 1, None)
        write_json(
            self.out / "development_selection.json",
            {
                "P4_first_eight_correct": p4_correct,
                "best_history_first_eight_correct": history8,
                "pseudo_promotion_pass": p4_correct > history8,
                "eligible_candidates": candidates,
                "PW4_PM4_status": "CONDITIONAL_BRANCH_EXECUTED"
                if p4_correct > history8
                else "SKIPPED_BY_GATE",
                "gate": "P4 strict accuracy gain over best H/HC on first eight dev samples",
            },
        )
        means = {
            p: sum(
                r["production_forwards_per_second"]
                for r in self.rows
                if r["policy"] == p
                and r["stage"] in ["development_performance", "pseudo_development_performance"]
            )
            / max(
                1,
                sum(
                    1
                    for r in self.rows
                    if r["policy"] == p
                    and r["stage"] in ["development_performance", "pseudo_development_performance"]
                ),
            )
            for p in candidates
        }
        candidate = max(candidates, key=lambda p: (correct[p], means[p], p == "H"))
        confirm_policies = ["E", "S", candidate]
        if candidate.startswith("P"):
            confirm_policies.append(max(["H", "HC"], key=lambda p: (correct[p], means[p])))
        remaining = min(
            self.protocol["resources"]["max_wall_hours"] * 3600 * 0.9
            - self.status["wall_seconds_used"],
            self.protocol["resources"]["max_gpu_hours"] * 3600 * 0.9
            - self.status["gpu_seconds_used"]
            - self.status["preflight_gpu_seconds_budget_upper_bound"],
        )
        avg = sum(r["request_wall_seconds"] for r in dev_quality) / len(dev_quality)
        chosen = next(
            (
                n
                for n in [256, 128, 64]
                if avg * 1.5 * n * len(confirm_policies) + 120 * 128 / min(means.values())
                < remaining
            ),
            None,
        )
        if chosen is None:
            raise TimeoutError("insufficient budget for the smallest predeclared N=64")
        freeze2 = {
            "candidate": candidate,
            "policies": confirm_policies,
            "quality_n": chosen,
            "max_new_tokens": self.cap,
            "claims": 4 if candidate.startswith("P") else 3,
            "identity": self.identity,
            "environment_sha256": digest(self.out / "environment.lock.txt"),
            "sample_ids": self.samples["confirmation_ids"][:chosen],
            "performance_ids": self.samples["confirmation_performance_ids"],
            "frozen_at_utc": utc(),
            "selection_quality": correct,
            "selection_tps": means,
            "estimated_confirmation_seconds": avg * 1.5 * chosen * len(confirm_policies),
            "PW4_PM4": "see development_selection.json",  # noqa: E501
        }
        if (self.out / "protocol_frozen_2.yaml").exists():
            old = yaml.safe_load((self.out / "protocol_frozen_2.yaml").read_text())
            if old["identity"] != self.identity:
                raise ValueError("freeze2 source mismatch")
            freeze2 = old
            self.cap = old["max_new_tokens"]
        else:
            (self.out / "protocol_frozen_2.yaml").write_text(
                yaml.safe_dump(freeze2, sort_keys=False)
            )
        write_json(
            self.out / "cost_estimate_P4.json",
            {
                "kind": "SCHEDULING_ESTIMATE_NOT_MEASUREMENT",
                "remaining_seconds": remaining,
                "selected_n": freeze2["quality_n"],
                "mean_development_request_seconds": avg,
                "estimated_confirmation_seconds": freeze2["estimated_confirmation_seconds"],
                "dev_p95_generated_tokens": sorted(r["generated_tokens"] for r in dev_quality)[
                    int(0.95 * (len(dev_quality) - 1))
                ],
                "resource_only_n_choice_before_confirmation": True,
            },
        )
        self.phase("P4_POLICY_DEV", "PASS", ["protocol_frozen_2.yaml"])
        self.phase("P5_CONFIRM", "RUNNING")
        self.matrix(
            "confirmation_performance", freeze2["performance_ids"], freeze2["policies"], 5, 128
        )
        self.matrix("confirmation_quality", freeze2["sample_ids"], freeze2["policies"], 1, None)
        self.phase("P5_CONFIRM", "PASS")
        self.phase("P6_FINAL_AUDIT", "RUNNING")
        checks = [
            ("final_pip_check", [sys.executable, "-m", "pip", "check"]),
            (
                "final_regressions",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/restart",
                    f"--junitxml={self.out}/tests/final_regressions.xml",
                ],
            ),
            (
                "final_ruff",
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "src/pseudoroute/restart",
                    "tests/restart",
                    "scripts/restart",
                ],
            ),
            (
                "final_mypy",
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--follow-imports=silent",
                    "src/pseudoroute/restart",
                ],
            ),
            (
                "legacy_integrity",
                ["git", "diff", "--exit-code", "HEAD", "--", "artifacts", "configs/benchmark"],
            ),
        ]
        for label, command in checks:
            with (self.out / "tests" / f"{label}.log").open("w") as log:
                log.write("COMMAND: " + repr(command) + "\n")
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        self.status["terminal_state"] = "COMPLETE"
        self.persist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    out = guarded_run_dir(args.run_dir, root)
    out.mkdir(parents=True, exist_ok=True)
    # A resumed run that is blocked before model readiness performs CPU reporting only.
    # It must not claim a GPU (including when the final audit runs its CLI regression).
    needs_gpu = not (out / "STATUS.json").exists() or json.loads(
        (out / "model_manifest.json").read_text()
    ).get("weights_verified", False)
    gpu_lock = (
        run_lock(root / ".cache/restart/gpu-0.lock", out.name)
        if needs_gpu
        else contextlib.nullcontext()
    )
    with run_lock(out / "run.lock", out.name), gpu_lock:
        if (out / "STATUS.json").exists() and args.resume:
            prior = json.loads((out / "STATUS.json").read_text())
            if prior.get("terminal_state") == "COMPLETE":
                from pseudoroute.restart.verify import verify

                print(json.dumps(verify(out)))
                return
        if not (out / "STATUS.json").exists():
            from pseudoroute.restart.prepare import prepare

            prepare(out)
        runner = Runner(args.config, out, args.resume)
        try:
            runner.run()
        except Exception as exc:
            runner.status["last_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "at_utc": utc(),
            }
            phase = runner.status["phase"]
            state = (
                "BUDGET_EXHAUSTED"
                if isinstance(exc, TimeoutError)
                else "BLOCKED"
                if isinstance(exc, (OSError, FileNotFoundError))
                else "INTERRUPTED"
            )
            runner.status["terminal_state"] = state
            runner.phase(phase, "BLOCKED" if state == "BLOCKED" else "FAIL")
            write_json(out / "last_failure.json", runner.status["last_error"])
            traceback.print_exc()
        finally:
            runner.persist()
            from pseudoroute.restart.report import finalize

            finalize(out)
            from pseudoroute.restart.verify import verify

            verification = verify(out)
            write_json(out / "verification.json", verification)
            if runner.status["terminal_state"] == "COMPLETE":
                runner.phase("P6_FINAL_AUDIT", "PASS", ["verification.json"])
                finalize(out)
                verify(out)
            write_json(
                out / "artifact_manifest.json",
                {
                    str(p.relative_to(out)): digest(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file() and p.name not in ("artifact_manifest.json", "run.lock")
                },
            )
        if runner.status["terminal_state"] != "COMPLETE":
            sys.exit(2)


if __name__ == "__main__":
    main()
