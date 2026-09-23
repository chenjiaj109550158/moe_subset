"""Readable audit, measured summaries and trial inventory derived from saved evidence."""

# Markdown table cells are intentionally kept together in generated evidence.
# ruff: noqa: E501

from __future__ import annotations

import csv
import io
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for stage, policy, backend in sorted({(r["stage"], r["policy"], r["backend"]) for r in rows}):
        selected = [
            r for r in rows if (r["stage"], r["policy"], r["backend"]) == (stage, policy, backend)
        ]
        quality = [r for r in selected if r["mode"] == "free_generation"]
        forwards = sum(r["production_forwards"] for r in selected)
        decode = sum(r["decode_wall_seconds"] for r in selected)
        latency = [v["seconds"] for r in selected for v in r["token_ready_latencies"]]
        boundary = [
            v["seconds"] for r in selected for v in r["token_ready_latencies"] if v["boundary"]
        ]
        ordinary = [
            v["seconds"] for r in selected for v in r["token_ready_latencies"] if not v["boundary"]
        ]
        decode_bytes = sum(
            v.get("h2d_bytes", 0)
            for r in selected
            for k, v in r["offload"]["phases"].items()
            if k != "prefill"
        )
        prefill_bytes = sum(
            r["offload"]["phases"].get("prefill", {}).get("h2d_bytes", 0) for r in selected
        )
        result[f"{stage}/{policy}/{backend}"] = {
            "stage": stage,
            "policy": policy,
            "backend": backend,
            "rows": len(selected),
            "quality_n": len(quality),
            "correct": sum(r["correct"] for r in quality) if quality else None,
            "truncated": sum(r["truncated"] for r in quality) if quality else None,
            "invalid_answers": sum(r.get("parsed_answer") is None for r in quality)
            if quality
            else None,
            "empty_answers": sum(not r["text"].strip() for r in quality) if quality else None,
            "generated_tokens": sum(r["generated_tokens"] for r in selected),
            "production_forwards": forwards,
            "pseudo_positions": sum(r["pseudo_positions"] for r in selected),
            "aggregate_forwards_per_second": forwards / decode if decode else None,
            "decode_wall_seconds": decode,
            "request_wall_seconds": sum(r["request_wall_seconds"] for r in selected),
            "mean_request_seconds": statistics.mean(r["request_wall_seconds"] for r in selected),
            "mean_prefill_seconds": statistics.mean(r["prefill_wall_seconds"] for r in selected),
            "mean_bootstrap_seconds": statistics.mean(
                r["bootstrap_wall_seconds"] for r in selected
            ),
            "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in selected),
            "p95_generated_tokens": float(
                np.quantile([r["generated_tokens"] for r in selected], 0.95)
            ),
            "decode_expert_bytes": decode_bytes,
            "prefill_expert_bytes": prefill_bytes,
            "decode_expert_bytes_per_production_forward": decode_bytes / forwards
            if forwards
            else None,
            "p50_token_ready_seconds": float(np.quantile(latency, 0.5)) if latency else None,
            "p95_token_ready_seconds": float(np.quantile(latency, 0.95)) if latency else None,
            "p95_boundary_seconds": float(np.quantile(boundary, 0.95)) if boundary else None,
            "p95_nonboundary_seconds": float(np.quantile(ordinary, 0.95)) if ordinary else None,
            "peak_cuda_allocated_bytes": max(r["peak_cuda_allocated_bytes"] for r in selected),
            "peak_cuda_reserved_bytes": max(r["peak_cuda_reserved_bytes"] for r in selected),
            "peak_host_rss_bytes": max(
                (r.get("host_peak_rss_bytes", 0) for r in selected), default=0
            )
            or None,
        }
    return result


def emit_evidence(out: Path, agg: dict[str, Any]) -> None:
    comparisons = list(agg["comparisons"].values())
    for name, records in [
        ("comparison.csv", comparisons),
        ("policy_summary.csv", list(agg["stage_summaries"].values())),
    ]:
        if records:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            (out / name).write_text(buffer.getvalue())
    suites = []
    for path in sorted((out / "tests").glob("*.xml")):
        root = ET.parse(path).getroot()
        entries = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        suites.append(
            {
                "file": str(path.relative_to(out)),
                **{
                    key: sum(int(x.attrib.get(key, "0")) for x in entries)
                    for key in ("tests", "failures", "errors", "skipped")
                },
            }
        )
    (out / "test_inventory.json").write_text(json.dumps(suites, indent=2))
    table = ["| Test evidence | tests | failures | errors | skipped |", "|---|---:|---:|---:|---:|"]
    table += [
        f"| {s['file']} | {s['tests']} | {s['failures']} | {s['errors']} | {s['skipped']} |"
        for s in suites
    ]
    model_file = out / "real_model_correctness.json"
    model = json.loads(model_file.read_text()) if model_file.exists() else {"state": "NOT_RUN"}
    g4_file = out / "g4_g8_semantics.json"
    g4 = json.loads(g4_file.read_text()) if g4_file.exists() else {"state": "NOT_RUN"}
    (out / "correctness_report.md").write_text(
        "# Correctness evidence\n\nCorrectness protocol v2 retains single-layer NRMSE <= 0.01 and cosine >= 0.999. Full-prefix cross-implementation NRMSE is <= 0.03 after the original 0.01 also failed unchanged native controls. Same-backend synchronous/event-ordered execution requires bitwise equality. Original failures and calibration evidence are preserved; see docs/restart/numerical_protocol_v2.md.\n\n"
        + "\n".join(table)
        + "\n\nCheckpoint checks:\n\n```json\n"
        + json.dumps(model, indent=2)
        + "\n```\n\nG4/G8 fixed-prefix bootstrap and joint semantics:\n\n```json\n"
        + json.dumps(g4, indent=2)
        + "\n```\n\n"
        "Initial environment collection/import errors are preserved. Six remaining legacy failures are missing historical v17 token-row files, not passing tests. Tiny/native tests and resident synthetic checks do not substitute for full-checkpoint checks. Full-resident native checkpoint execution was not used; the reference is CPU-first synchronous Python expert compute at fixed routes, plus native tiny and real single-layer FP32 checks. G4/G8 shape-dependent BF16 differences are measured, not assumed bitwise equal.\n"
    )
    audit = """# Source audit and fixes

| Item | Source / finding | Change and evidence | Remaining scope |
|---|---|---|---|
| C01 | `benchmark/subset_closed_loop.py::_finished` constructs StopStringCriteria each token | New `restart/stopping.py::RequestStopState`, native prompt+continuation semantics; `tests/restart/test_stopping.py`; CPU profile in `profiles/stop_cpu.json` when completed | Historical helper retained for legacy reproduction; new measurements exclusively use request state. Host stop check remains. |
| C02 | old speed runner formatted metrics before end timestamp; legacy per-expert events | Timestamp fixed in old runner; new shared harness drains then timestamps before formatting. Persistent hooks, reusable per-layer dependency events, profile-only detailed timing | Profile overhead is excluded from official performance rows. Durations can overlap and cannot be summed as wall. |
| C03 | legacy horizon and compute count coupled | `restart/policies.py`: generate legacy horizon 8 then truncate G4; first-four tokens, positions, RNG and prefix checked in `g4_g8_semantics.json`; own-trajectory diagnostics saved | BF16 shape differences may change routes/answers; only measured differences are reported. |
| C04 | first-four disjoint top-8 union occupies m32 and prevents history displacement | Regression retains this legacy behavior; HC/PW4 are separately named selectors with deterministic ID tie break | GPU FP32 history may differ from old CPU double at near ties. No tie-breaking epsilon added. |
| C05 | full router weighting and slot safety required | Native FP32 softmax/top-k and normalization; hard production reroute separated from pseudo natural intersection; invalid zero slot sanitized before dereference, nonzero invalid raises at drain | Natural stats describe each policy's current hidden state, not an exact reference trajectory. |
| C06 | joint cache and generation/planning RNG must remain isolated | Bridge-only fork/commit, prefix invariance, two transitions, partial window, EOS/stop boundary, G0/4/8, stochastic RNG fixtures in GPU tests; CPU RNG helper repaired | Pseudo legacy content explicitly rejects fewer than seven preceding known tokens; H/S allow short prompts. |
| C07 | slots can be overwritten before old kernels finish | Group completion event after reads; copy stream waits; next compute waits ready; ordered GPU map mutations; generation trace and high-churn m=k/reset tests | Sanitizer availability recorded separately. End-to-end performance includes exact demand CPU scheduling. |
| C08 | fused accumulation is not bitwise identical to Python | Fixed BF16 weights promoted to FP32 for single-layer reference; actual shape resident and full-prefix fixed-route comparisons | No precision/model substitution. An unresolved numerical failure blocks method conclusions. |
| CPU-first | old QwenExpertOffloadEngine requires full CUDA model before extraction | Meta model + verified sharded CPU expert backing; only dense and m-slot experts on CUDA; all checkpoint keys/shapes checked | No full-model CUDA staging. See `loader_manifest.json` for actual readiness and memory. |
| Backend | old active-expert Python linear loop is costly | Shared official vLLM fused experts, m-slot logical mapping, same optimized backend in E/S/H/HC/P | EP not implemented; baseline scope exact demand LRU. Four-grid is a compute ablation in the updated common harness, not replay of the whole old instrumented harness. |

Failed environment resolver attempts, the initial resident reference that scanned inactive slots, repaired tests and final active-only reference timing are preserved in tests and `*.attempt1.*`. Only the current fair active-expert reference is used for backend speed claims. No custom kernel search was performed. No historical frozen config or artifact is rewritten. Runtime validation and research disposition remain separate.
"""
    (out / "AUDIT_FINDINGS.md").write_text(audit)
    trial_files = sorted(
        str(p.relative_to(out))
        for p in out.rglob("*")
        if p.is_file()
        and ("attempt" in p.name or "failure" in p.name or p.suffix in (".log", ".xml"))
    )
    (out / "trial_inventory.json").write_text(
        json.dumps(
            {"logs_and_trials": trial_files, "simulation_used_as_model_evidence": False}, indent=2
        )
    )
