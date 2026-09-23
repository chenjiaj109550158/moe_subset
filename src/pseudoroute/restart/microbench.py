"""Measured resident compute correctness, cache diagnostics and timing blocks."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from pseudoroute.restart.backend import MoEComputeBackend, numerical_metrics
from pseudoroute.restart.state import write_json


def fp32_reference(
    x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, gu: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in range(gu.shape[0]):
        t, r = torch.where((ids == e) & (weights != 0))
        if t.numel() == 0:
            continue
        g, u = F.linear(x[t].float(), gu[e].float()).chunk(2, -1)
        y = F.linear(F.silu(g) * u, down[e].float())
        out.index_add_(0, t, y * weights[t, r, None].float())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.run_dir
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    fused = MoEComputeBackend("vllm_fused", audit=True)
    reference = MoEComputeBackend("python_reference", audit=True)
    import vllm

    write_json(
        out / "backend_candidates.json",
        {
            "official_candidates": [
                {
                    "name": "vllm_fused_experts",
                    "version": vllm.__version__,
                    "signature": str(inspect.signature(fused.fused)),
                    "state": "IMPORTED",
                    "license": "Apache-2.0; installed wheel retains LICENSE/NOTICE",
                }
            ],
            "custom_kernel_configs": 0,
        },
    )
    gu = torch.randn(32, 1536, 2048, device="cuda", dtype=torch.bfloat16) / math.sqrt(2048)
    down = torch.randn(32, 2048, 768, device="cuda", dtype=torch.bfloat16) / math.sqrt(768)
    checks = []
    rows = []
    for t in (1, 5, 9, 16, 32, 128):
        for distribution in ("uniform", "skew"):
            x = torch.randn(t, 2048, device="cuda", dtype=torch.bfloat16)
            ids = (
                torch.stack([torch.randperm(32, device="cuda")[:8] for _ in range(t)])
                if distribution == "uniform"
                else torch.arange(8, device="cuda")[None].expand(t, -1).contiguous()
            )
            weights = torch.softmax(torch.randn(t, 8, device="cuda"), -1).bfloat16()
            started = time.perf_counter()
            y = fused.forward(x, ids, weights, gu, down, {})
            torch.cuda.synchronize()
            cold = time.perf_counter() - started
            expected = fp32_reference(x, ids, weights, gu, down)
            metrics = numerical_metrics(y, expected)
            metrics.update(
                tokens=t,
                distribution=distribution,
                cold_compile_or_first_call_seconds=cold,
                passed=metrics["finite"]
                and metrics["nrmse"] <= 0.01
                and metrics["cosine"] >= 0.999,
            )
            checks.append(metrics)
            write_json(out / "backend_correctness.json", {"state": "RUNNING", "checks": checks})
            # Rotating weights exceed L2; two full layers, no selected-matrix materialization.
            layers = [(gu, down), (gu.clone(), down.clone())]
            for backend in (reference, fused):
                backend.audit = False
                for cache in ("hot", "rotating"):
                    for _ in range(2):
                        backend.forward(x, ids, weights, gu, down, {})
                    torch.cuda.synchronize()
                    pilot = time.perf_counter()
                    backend.forward(x, ids, weights, gu, down, {})
                    torch.cuda.synchronize()
                    repetitions = max(
                        2, min(30, math.ceil(0.03 / max(time.perf_counter() - pilot, 1e-6)))
                    )
                    for block in range(5):
                        start_event, end_event = (
                            torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
                            torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
                        )
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        start_event.record()  # type: ignore[no-untyped-call]
                        for rep in range(repetitions):
                            g, w = layers[rep % 2] if cache == "rotating" else layers[0]
                            backend.forward(x, ids, weights, g, w, {})
                        end_event.record()  # type: ignore[no-untyped-call]
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - start
                        rows.append(
                            {
                                "tokens": t,
                                "distribution": distribution,
                                "backend": backend.name,
                                "cache": cache,
                                "block": block,
                                "repetitions": repetitions,
                                "host_wall_ms_per_forward": wall * 1000 / repetitions,
                                "gpu_elapsed_ms_per_forward": start_event.elapsed_time(end_event)  # type: ignore[no-untyped-call]
                                / repetitions,
                                "synthetic": True,
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            }
                        )
            print(json.dumps(metrics), flush=True)
    with (out / "moe_microbench.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(c["passed"] for c in checks)
    write_json(
        out / "backend_correctness.json", {"state": "PASS" if passed else "FAIL", "checks": checks}
    )
    write_json(
        out / "backend_selection.json",
        {
            "selected": "vllm_fused" if passed else None,
            "correctness_pass": passed,
            "all_policies_share": True,
            "selection_scope": "full forward including dispatch/MLP/combine at real H/I/k; includes T=1",  # noqa: E501
            "performance_rows": len(rows),
        },
    )
    if not passed:
        raise RuntimeError(
            "frozen numerical gate failed; no method performance conclusions allowed"
        )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            fused.forward(x[:1].contiguous(), ids[:1], weights[:1], gu, down, {})
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out / "profiles" / "resident_t1.json"))
    (out / "profiles" / "resident_t1.txt").write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    )


if __name__ == "__main__":
    main()
