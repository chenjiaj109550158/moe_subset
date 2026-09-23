"""Aggregate actual paired rows and produce a schema-valid, evidence-backed report."""

# Generated Markdown tables and Chinese prose retain complete cells and sentences.
# ruff: noqa: E501

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pseudoroute.restart.decision import evaluate_decision, quality_interval, speed_interval
from pseudoroute.restart.evidence import emit_evidence, summarize
from pseudoroute.restart.state import digest, load_row, write_json


def aggregate(out: Path) -> dict[str, Any]:
    status = json.loads((out / "STATUS.json").read_text())
    identity = status["identity"]
    rows = []
    for path in sorted((out / "runs").rglob("*.json")):
        row = load_row(path, identity)
        if row is not None:
            rows.append(row)
    freeze = (
        yaml.safe_load((out / "protocol_frozen_2.yaml").read_text())
        if (out / "protocol_frozen_2.yaml").exists()
        else {}
    )
    quality = [r for r in rows if r["stage"] == "confirmation_quality"]
    perf = [r for r in rows if r["stage"] == "confirmation_performance"]
    candidate = freeze.get("candidate")
    policies = freeze.get("policies", [])
    metrics: dict[str, Any] = {
        k: None
        for k in (
            "candidate",
            "baseline",
            "planned_quality_n",
            "accuracy_delta_fraction",
            "accuracy_delta_ci",
            "paired_gains",
            "paired_losses",
            "paired_ties",
            "decode_speedup",
            "decode_speedup_ci",
            "decode_expert_bytes_reduction_fraction",
            "p95_latency_ratio",
        )
    }
    metrics.update(
        candidate=candidate,
        baseline="E",
        planned_quality_n=freeze.get("quality_n"),
        completed_paired_quality_n=0,
        actual_model_runs_completed=bool(rows),
    )
    comparisons = {}

    def compare(a: str, b: str) -> dict[str, Any]:
        qa = {r["sample_id"]: r for r in quality if r["policy"] == a}
        qb = {r["sample_id"]: r for r in quality if r["policy"] == b}
        ids = sorted(set(qa) & set(qb))
        n = len(ids)
        gains = sum(qa[i]["correct"] and not qb[i]["correct"] for i in ids)
        losses = sum(qb[i]["correct"] and not qa[i]["correct"] for i in ids)
        pa = {(r["sample_id"], r["repetition"]): r for r in perf if r["policy"] == a}
        pb = {(r["sample_id"], r["repetition"]): r for r in perf if r["policy"] == b}
        pairs = sorted(set(pa) & set(pb))
        ratios: dict[str, list[float]] = {}
        for key in pairs:
            ratios.setdefault(key[0], []).append(
                pa[key]["production_forwards_per_second"]
                / pb[key]["production_forwards_per_second"]
            )
        speed, ci = speed_interval(ratios) if ratios else (None, None)

        def expert_bytes(r: dict[str, Any]) -> int:
            return sum(
                v.get("h2d_bytes", 0)
                for name, v in r["offload"]["phases"].items()
                if name != "prefill"
            )

        a_bytes = sum(expert_bytes(pa[k]) for k in pairs)
        b_bytes = sum(expert_bytes(pb[k]) for k in pairs)
        a_lat = [v["seconds"] for k in pairs for v in pa[k]["token_ready_latencies"]]
        b_lat = [v["seconds"] for k in pairs for v in pb[k]["token_ready_latencies"]]
        return {
            "a": a,
            "b": b,
            "n": n,
            "gains": gains,
            "losses": losses,
            "ties": n - gains - losses,
            "accuracy_delta_fraction": (gains - losses) / n if n else None,
            "accuracy_delta_ci": quality_interval(gains, losses, n, freeze.get("claims", 1)),
            "performance_pairs": len(pairs),
            "speedup": speed,
            "speedup_ci": ci,
            "expert_bytes_reduction_fraction": 1 - a_bytes / b_bytes if b_bytes else None,
            "p95_latency_ratio": float(np.quantile(a_lat, 0.95) / np.quantile(b_lat, 0.95))
            if a_lat and b_lat
            else None,
            "candidate_forwards": sum(pa[k]["production_forwards"] for k in pairs),
            "candidate_pseudo_positions": sum(pa[k]["pseudo_positions"] for k in pairs),
            "candidate_decode_bytes": a_bytes if pairs else None,
            "baseline_decode_bytes": b_bytes if pairs else None,
        }

    if candidate:
        for a, b in [(candidate, "E"), ("S", "E"), (candidate, "S"), ("S", candidate)]:
            comparisons[f"{a}_vs_{b}"] = compare(a, b)
        c = comparisons[f"{candidate}_vs_E"]
        metrics.update(
            completed_paired_quality_n=c["n"],
            accuracy_delta_fraction=c["accuracy_delta_fraction"],
            accuracy_delta_ci=c["accuracy_delta_ci"],
            paired_gains=c["gains"] if c["n"] else None,
            paired_losses=c["losses"] if c["n"] else None,
            paired_ties=c["ties"] if c["n"] else None,
            decode_speedup=c["speedup"],
            decode_speedup_ci=c["speedup_ci"],
            decode_expert_bytes_reduction_fraction=c["expert_bytes_reduction_fraction"],
            p95_latency_ratio=c["p95_latency_ratio"],
        )
    correct_path = out / "real_model_correctness.json"
    correct = (
        json.loads(correct_path.read_text()).get("state") == "PASS"
        if correct_path.exists()
        else None
    )
    complete = bool(
        candidate
        and all(sum(r["policy"] == p for r in quality) == freeze["quality_n"] for p in policies)
        and all(sum(r["policy"] == p for r in perf) == 40 for p in policies)
    )
    evidence = {
        **metrics,
        "correctness_pass": correct,
        "validity_pass": correct is True and complete,
        "runtime_measurement_valid": correct is True
        and (out / "profiles/runtime_summary.json").exists(),
        "required_pairs_complete": complete,
        "timing_valid": complete,
        "external_blockers": [status.get("last_error")]
        if status.get("terminal_state") == "BLOCKED"
        else [],
        "pseudo_tested": any(r["policy"].startswith("P") for r in rows),
        "dynamic_value_confirmed": False,
        "pseudo_increment_confirmed": False,
        "static_dominance_confirmed": False,
        "sufficient_negative_evidence": False,
    }
    if complete and candidate is not None:
        ce = comparisons[f"{candidate}_vs_E"]
        cs = comparisons[f"{candidate}_vs_S"]
        se = comparisons["S_vs_E"]
        sc = comparisons[f"S_vs_{candidate}"]
        qpass = ce["accuracy_delta_ci"][0] >= -0.02
        evidence["dynamic_value_confirmed"] = cs["accuracy_delta_ci"][0] > 0 or (
            se["accuracy_delta_ci"][1] < -0.02 and qpass
        )
        evidence["static_dominance_confirmed"] = (
            se["accuracy_delta_ci"][0] >= -0.02
            and se["speedup"] >= 1.1
            and se["speedup_ci"][0] > 1
            and sc["accuracy_delta_ci"][0] >= -0.02
            and sc["speedup_ci"][0] > 1
        )
        evidence["sufficient_negative_evidence"] = (
            ce["accuracy_delta_ci"][1] < -0.02 or ce["speedup_ci"][1] < 1.05
        )
        if candidate.startswith("P"):
            h = next((p for p in policies if p in ("H", "HC")), None)
            if h:
                ph = compare(candidate, h)
                comparisons[f"{candidate}_vs_{h}"] = ph
                evidence["pseudo_increment_confirmed"] = (
                    ph["accuracy_delta_ci"][0] >= -0.02 and ph["speedup_ci"][0] > 1.1
                ) or (ph["accuracy_delta_ci"][0] > 0 and ph["speedup_ci"][0] >= 0.95)
                hp = compare(h, candidate)
                evidence["pseudo_dominated"] = (
                    hp["accuracy_delta_ci"][0] >= 0 and hp["speedup_ci"][0] > 1
                )
    # SciPy interval comparisons produce NumPy bool scalars on completed runs.
    # Preserve their values while making the reporting evidence JSON-serializable.
    evidence = {
        key: bool(value) if isinstance(value, np.bool_) else value
        for key, value in evidence.items()
    }
    summaries = {}
    for p in sorted({r["policy"] for r in rows}):
        q = [r for r in quality if r["policy"] == p]
        timings = [r for r in perf if r["policy"] == p]
        summaries[p] = {
            "quality_n": len(q),
            "correct": sum(r["correct"] for r in q) if q else None,
            "truncated": sum(r["truncated"] for r in q) if q else None,
            "generated_tokens": sum(r["generated_tokens"] for r in q) if q else None,
            "free_request_wall_seconds": sum(r["request_wall_seconds"] for r in q) if q else None,
            "aggregate_controlled_forwards_per_second": sum(
                r["production_forwards"] for r in timings
            )
            / sum(r["decode_wall_seconds"] for r in timings)
            if timings
            else None,
            "peak_cuda_allocated_bytes": max(
                (r["peak_cuda_allocated_bytes"] for r in rows if r["policy"] == p), default=None
            ),
        }
    protocol = yaml.safe_load(Path("configs/restart/restart_v1.yaml").read_text())
    decision = evaluate_decision(evidence, protocol)
    return {
        "primary_metrics": metrics,
        "decision_inputs": evidence,
        "decision_result": decision,
        "comparisons": comparisons,
        "policy_summaries": summaries,
        "stage_summaries": summarize(rows),
        "actual_row_count": len(rows),
        "confirmation_quality_rows": len(quality),
        "confirmation_performance_rows": len(perf),
    }


def source_patch(out: Path) -> None:
    patch = subprocess.check_output(
        [
            "git",
            "diff",
            "HEAD",
            "--",
            ".gitignore",
            "src",
            "tests/restart",
            "scripts/restart",
            "configs/restart",
            "docs/restart",
        ]
    )
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src/pseudoroute/restart",
            "scripts/restart",
            "configs/restart",
            "docs/restart",
            "tests/restart",
        ],
        text=True,
    ).splitlines()
    for name in names:
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", name], capture_output=True
        )
        patch += result.stdout
    (out / "source_changes.patch").write_bytes(patch)


def finalize(out: Path) -> dict[str, Any]:
    status = json.loads((out / "STATUS.json").read_text())
    agg = aggregate(out)
    write_json(out / "aggregate.json", agg)
    source_patch(out)
    emit_evidence(out, agg)
    result = agg["decision_result"]
    metrics = agg["primary_metrics"]
    protocol_path = out / (
        "protocol_frozen_2.yaml"
        if (out / "protocol_frozen_2.yaml").exists()
        else "protocol_frozen_1.yaml"
    )
    protocol = yaml.safe_load(Path("configs/restart/restart_v1.yaml").read_text())
    base = json.loads((out / "source_manifest.json").read_text())
    evidence = []
    kinds = {
        "hardware.json": "MEASURED",
        "environment.lock.txt": "CONFIG",
        "resource_resolution.json": "CONFIG",
        "model_manifest.json": "CONFIG",
        "samples_manifest.json": "CONFIG",
        "source_changes.patch": "SOURCE_AUDIT",
        "backend_correctness.json": "TEST",
        "correctness_protocol.json": "CONFIG",
        "numerical_diagnosis2.json": "TEST",
        "numerical_diagnosis3.json": "TEST",
        "real_model_correctness.json": "TEST",
        "moe_microbench.csv": "MEASURED",
        "aggregate.json": "DERIVED_STATISTIC",
        "comparison.csv": "DERIVED_STATISTIC",
        "policy_summary.csv": "DERIVED_STATISTIC",
        "AUDIT_FINDINGS.md": "SOURCE_AUDIT",
        "correctness_report.md": "TEST",
        "test_inventory.json": "TEST",
        "development_selection.json": "CONFIG",
        "g4_g8_semantics.json": "TEST",
        "loader_manifest.json": "TEST",
        "last_failure.json": "BLOCKER_LOG",
        "tests/runtime_gpu_attempt1.xml": "TEST",
        "tests/baseline_cpu_env2.log": "TEST",
        "tests/policies_decision.xml": "TEST",
    }
    for name, kind in kinds.items():
        if (out / name).exists():
            evidence.append({"path": name, "sha256": digest(out / name), "kind": kind})
    common = ["aggregate.json"]
    reasons = {
        "runtime": "Runtime correctness is scoped to completed CPU/GPU/checkpoint tests; see gate V and the unresolved audit items.",  # noqa: E501
        "window_subset": "Only complete, valid paired confirmation supports a window method conclusion.",  # noqa: E501
        "pseudo_predictor": "Route mass or P4 versus P8 timing alone cannot establish superiority to cheap history.",  # noqa: E501
    }
    resume = f"cd {Path.cwd()} && USE_HUB_KERNELS=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 {sys.executable} -m pseudoroute.restart.run_all --config configs/restart/restart_v1.yaml --run-dir {out} --resume"  # noqa: E501
    limitations = [
        "Conclusions apply only to Qwen3-30B-A3B-Instruct-2507 BF16, this RTX PRO 6000, m=32,d=8.",
        "Exact baseline scope is demand LRU; EP was not implemented.",
        "Historical v17 external token-row files absent from this checkout; legacy protocol tests report this without rewriting history.",  # noqa: E501
        "Resident synthetic timing is not end-to-end or model-quality evidence.",
        "Correctness v2 revises full-prefix cross-implementation NRMSE from 0.01 to 0.03 after unchanged native controls also failed v1; single-layer 0.01 and cosine 0.999 remain, and sync/event parity must be bitwise.",
        "No full literature novelty claim; historical eight-row oracle is nondeployable background only.",  # noqa: E501
    ]
    if status.get("last_error"):
        limitations.append(str(status["last_error"]["message"]))
    decision = {
        "schema_version": 1,
        "run_id": out.name,
        "run_state": status["terminal_state"] or "INTERRUPTED",
        "overall": result["overall"],
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "scope": {
            "repository_commit": base["commit"],
            "source_diff_sha256": digest(out / "source_changes.patch"),
            "protocol_sha256": digest(protocol_path)
            if protocol_path.exists()
            else digest(Path("configs/restart/restart_v1.yaml")),
            "model_id": protocol["model"]["id"],
            "model_revision": protocol["model"]["revision"],
            "dtype": "bfloat16",
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "backend": "vllm_fused_experts_0.11.0",
            "host_mode": json.loads((out / "loader_manifest.json").read_text()).get("host_mode")
            if (out / "loader_manifest.json").exists()
            else None,
            "slots_per_layer": 32,
            "window_tokens": 8,
            "pseudo_compute_tokens": (
                4 if str(metrics.get("candidate", "")).startswith("P") else 0
            ),
            "baseline_scope": "exact_demand_LRU_only",
            "dataset_scope": "seeded GSM8K train development / test confirmation with known policy-development IDs excluded",  # noqa: E501
        },
        "gates": {
            key: {
                "status": value,
                "reason": "Computed from immutable paired rows and frozen thresholds; missing comparisons remain unproven.",  # noqa: E501
                "evidence_paths": common,
            }
            for key, value in result["gates"].items()
        },
        **{
            name: {"status": result[name], "reason": reasons[name], "evidence_paths": common}
            for name in reasons
        },
        "primary_metrics": metrics,
        "evidence": evidence,
        "limitations": limitations,
        "next_action": "Resolve the recorded runtime/checkpoint blocker and rerun the same frozen comparisons."  # noqa: E501
        if result["overall"] in ("BLOCKED", "PIVOT_RUNTIME")
        else "Use the paired Q/S/D/P results to continue only the supported runtime/window/pseudo branch; no conclusion extends beyond the tested regime.",  # noqa: E501
        "resume_command": resume,
    }
    import jsonschema  # type: ignore[import-untyped]

    jsonschema.Draft202012Validator(
        json.loads(Path("configs/restart/decision.schema.json").read_text()),
        format_checker=jsonschema.FormatChecker(),
    ).validate(decision)
    write_json(out / "DECISION.json", decision)
    table = [
        "| Policy | paired quality n | correct | controlled forwards/s |",
        "|---|---:|---:|---:|",
    ]
    for p, s in agg["policy_summaries"].items():
        table.append(
            f"| {p} | {s['quality_n']} | {s['correct']} | {s['aggregate_controlled_forwards_per_second']} |"  # noqa: E501
        )
    if not agg["policy_summaries"]:
        table.append("| 無正式模型列 | 0 | 未測 | 未測 |")
    actions = {
        "GO_WINDOW_AND_PSEUDO": "繼續 window 與已確認的 pseudo 分支；收益僅限本輪已測範圍。",
        "GO_WINDOW_HISTORY_ONLY": "繼續便宜 history window；目前沒有需要擴大 pseudo 投入的確認證據。",
        "STATIC_SUFFICIENT_TESTED_SCOPE": "在本輪配置保留 static；動態更新沒有得到必要性的配對支持。",
        "NO_GO_TESTED_REGIME": "停止本輪 m=32,d=8 的已測 selector 配置，保留通過驗證的共用 runtime。",
        "PIVOT_RUNTIME": "先修復留下的 runtime correctness/量測失敗；方法判定暫緩。",
        "BLOCKED": "恢復記錄的外部 blocker 後，以原凍結規則接續；目前沒有方法不可行的證據。",
        "INCONCLUSIVE": "本輪不能確認 window 或 pseudo 投資價值；決定性缺口見下列未通過 gates，不能用 kernel 或 bytes 數字替代。",
        "PILOT_PROMISING": "保留 window 作有限 pilot；在缺失的品質、static 增益或邊界延遲證據確認前，不擴大強聲稱。",
    }
    decision["next_action"] = actions.get(result["overall"], "依 gate 證據限制研究聲稱。")
    write_json(out / "DECISION.json", decision)
    detail_table = [
        "| 配對 | n / g / l / tie | accuracy delta [CI] | decode speedup [CI] | bytes 減少 | p95 比 |",
        "|---|---|---|---|---|---|",
    ]
    for c in agg["comparisons"].values():
        detail_table.append(
            f"| {c['a']} vs {c['b']} | {c['n']} / {c['gains']} / {c['losses']} / {c['ties']} | "
            f"{c['accuracy_delta_fraction']} {c['accuracy_delta_ci']} | {c['speedup']} {c['speedup_ci']} | "  # noqa: E501
            f"{c['expert_bytes_reduction_fraction']} | {c['p95_latency_ratio']} |"
        )
    grid = [
        "| backend / policy | forwards/s | decode wall s | expert bytes/forward |",
        "|---|---:|---:|---:|",
    ]
    for s in agg["stage_summaries"].values():
        if s["stage"] == "runtime_grid":
            grid.append(
                f"| {s['backend']} / {s['policy']} | {s['aggregate_forwards_per_second']} | {s['decode_wall_seconds']} | {s['decode_expert_bytes_per_production_forward']} |"  # noqa: E501
            )
    report = f"""# MoE subset restart-v1 最終報告\n\n本輪判定：**{decision["overall"]}**；執行狀態：**{decision["run_state"]}**。\n\n1. Runtime：**{result["runtime"]}**。\n2. Window subset：**{result["window_subset"]}**。\n3. Pseudo predictor：**{result["pseudo_predictor"]}**。\n\n本機實際完成 {agg["actual_row_count"]} 列模型執行，正式品質列 {agg["confirmation_quality_rows"]}、正式性能列 {agg["confirmation_performance_rows"]}。未完成的主比較沒有填造 accuracy、speedup 或信賴區間。GPU resident / tiny 結果不能替代主模型結論。\n\n## 主要測量\n\n{chr(10).join(table)}\n\n完整配對 g/l/t、CI、bytes、p95、長度、truncation、invalid/empty 與記憶體請見 `aggregate.json`、`comparison.csv`、`policy_summary.csv`、`runs/`。Controlled workload 固定 128 個 post-prefill production forwards，第一個 prefill sample 和 pseudo positions 不算 production 分母。\n\n## 配對確認與 gate\n\n{chr(10).join(detail_table)}\n\nAccuracy delta 與 bytes reduction 使用 fraction；乘 100 才是百分點／百分比。Gate 狀態：`{result["gates"]}`。\n\n研究動作：{decision["next_action"]}\n\n## 同機 runtime 四格\n\n{chr(10).join(grid)}\n\n這是同一修正 harness 的 expert-compute ablation，僅用於歸因；它不代表整套歷史 runner 的端到端倍數。\n\n## 已完成工程與限制\n\n- C01：request stop state 一次建構，保留 prompt+continuation native stop 語意。\n- C02：audit/profile/performance 共用數值路徑；必要 device drain 在 wall end 前，metrics formatting 在外。尚未測的 event timing 為 null。\n- C03/C04：d/G/content horizon 分離；先建 legacy H=8 內容再取前四；legacy core 硬優先保留，HC 為獨立 policy。\n- C05：native FP32 softmax/top-k/normalization，hard reroute 與 pseudo zero-missing 分離；invalid 非零 slot 觸發錯誤。\n- C06：bridge-only KV commit、CPU/GPU RNG、兩次 window transition 與 cache prefix invariants；詳細範圍以 tests 為準。\n- C07：slot generation、整組 kernel completion、copy-ready 依賴與 churn/m=k tests。\n- C08：BF16 對同已量化 weights 的 FP32 參考，固定路由完整 prefix；未通過時禁止方法結論。\n- CPU-first loader：sharded safetensors 直接建立 CPU expert store；dense 在 GPU，每層 32 slots；禁止 full-model CUDA staging。\n- 共用 GPU backend：vLLM 0.11.0 官方 fused experts；所有 E/S/H/HC/P 使用相同 backend、精度與 slot 預算。\n\n各項狀態及剩餘缺口由 `STATUS.json`、`AUDIT_FINDINGS.md`、`correctness_report.md` 和測試 XML 列明；上列實作描述不等同全部 gate 已通過。\n\n## 環境與來源\n\nRepository SHA `{base["commit"]}`；模型 revision `{protocol["model"]["revision"]}`；BF16；RTX PRO 6000 Blackwell 96 GiB，與舊 A100 並非同機比較。CPU、RAM、cgroup、driver/runtime、NUMA/PCIe 與 pinned smoke 見 `hardware.json`。依賴及下載來源見 `environment.lock.txt`、`environment_manifest.json`、`model_manifest.json`。資源與主 m/d 在任何新方法输出前解析。\n\n## 判定範圍與歸因\n\n同機舊 Python / 新 backend 四格列在 `runs/runtime_grid/`；不存在的列表示未測。Resident 微基準與 profiles 是工程診斷，不能把 kernel 倍數、bytes 下降或舊 oracle 8/8 換算成端到端改善。未完成 profile 的 wall time 歸因保持 unresolved。舊 artifacts 僅 provenance，沒有用作本機性能或品質 reference；既有 oracle 不重跑。\n\n## 完整性、預算與恢復\n\n目前 wall {status.get("wall_seconds_used", 0):.1f} 秒；GPU worker elapsed {status.get("gpu_seconds_used", 0):.1f} 秒。前置 GPU 檢查未完整集中記錄 worker 起訖，因此另外以整段前置 wall {status.get("preflight_gpu_seconds_budget_upper_bound", 0):.1f} 秒作保守 GPU 預算扣帳上界，不冒充實測 device elapsed。總上限 72 wall-hours / 48 GPU-hours，保留 10% 收尾。每個 row 有 identity/checksum，失敗與缺列不被替換為其他樣本。模型輸出以自己的 KV/trajectory 生成，gold 只進 evaluator。\n\n限制：\n\n{chr(10).join("- " + x for x in limitations)}\n\n## 本次命令與正式動作\n\n實際執行與失敗日志位於 `tests/`；未跑或被 gate 阻止的階段見 `STATUS.json`。重現及精確續跑命令見 `REPRODUCE.md` / `RESUME.md`。\n\n決定性證據是同一修正 runtime、相同 m/精度/記憶體預算的完整 E/S/window 自由生成與 fixed-work 配對。沒有這個證據時，不能判方法不可行，也不能主張 pseudo 比 history 值得。\n"""  # noqa: E501
    report += "\n## 分階段性能與自由生成\n\n"
    report += "每階段的實測 rows、整 request、decode、生成長度、truncation、invalid/empty、bootstrap、prefill 與 decode bytes、p50/p95（含 boundary/nonboundary）以及 GPU/RSS 峰值，完整列於 `policy_summary.csv`。Development、diagnostic 與 confirmation 分開，不合併作獨立品質樣本。\n"  # noqa: E501
    report += "\n## 測試與 trial\n\n"
    report += "`correctness_report.md` 列出每個 JUnit 的 pass/failure/skip，`trial_inventory.json` 保留所有環境、修補、基準與失敗紀錄；`source_changes.patch` 可在記錄的基底 commit 套用。第一版 resident reference 掃描 inactive experts 的 timing 被保留但已失效，不用於最終加速聲稱。\n"  # noqa: E501
    (out / "FINAL_REPORT.md").write_text(report)
    (out / "RESUME.md").write_text(
        "# 續跑\n\n```bash\n"
        + resume
        + "\n```\n\n相同 source/config/env identity 且 checksum 正確才重用 row；source 改變必須保留並失效化完整受影響比較。沒有假設任何背景程序會在 agent 結束後完成。\n"  # noqa: E501
    )
    (out / "REPRODUCE.md").write_text(
        "# 重現與離線驗證\n\n本次工作目錄：`"
        + str(Path.cwd())
        + "`\n\n```bash\nbash scripts/restart/bootstrap.sh --config configs/restart/restart_v1.yaml\n"  # noqa: E501
        + resume
        + "\nUSE_HUB_KERNELS=0 "
        + sys.executable
        + " -m pseudoroute.restart.verify --run-dir "
        + str(out)
        + "\n```\n\n失敗案例重跑：\n\n```bash\nUSE_HUB_KERNELS=0 OMP_NUM_THREADS=4 "
        + sys.executable
        + " -m pytest tests/restart/test_gpu.py -x\n```\n"
    )
    return decision
