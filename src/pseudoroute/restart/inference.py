"""Own-trajectory generation; fixed-work and task-quality paths are explicitly separate."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from pseudoroute.benchmark.qwen_penultimate_joint import (
    commit_bridge_only_cache,
    fork_bridge_only_cache,
)
from pseudoroute.benchmark.subset_closed_loop import _fork_cache_copy_on_write
from pseudoroute.restart.model import PolicyRuntime
from pseudoroute.restart.policies import WindowConfig, anchor_ids
from pseudoroute.restart.stopping import RequestStopState


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    runtime: PolicyRuntime,
    prompt_ids: torch.Tensor,
    *,
    policy: str,
    max_new_tokens: int = 512,
    fixed_forwards: int | None = None,
    stop_strings: tuple[str, ...] = ("Q:", "</s>", "<|im_end|>"),
    d: int = 8,
    mode: str = "performance",
) -> dict[str, Any]:
    if mode != runtime.mode or mode != runtime.manager.mode:
        raise ValueError("instrumentation mode must be fixed at model initialization")
    manager = runtime.manager
    manager.reset()
    runtime.reset(policy)
    runtime.d = d
    torch.cuda.reset_peak_memory_stats(manager.device)
    g = 8 if policy == "P8" else (4 if policy.startswith("P") else 0)
    config = WindowConfig(d, g)
    prompt = tuple(prompt_ids[0].tolist())
    cap = max_new_tokens if fixed_forwards is None else fixed_forwards + 1
    request_start = time.perf_counter()
    stopper = RequestStopState(
        tokenizer,
        prompt_ids,
        model.generation_config.eos_token_id,
        stop_strings if fixed_forwards is None else (),
        cap,
    )
    prefill_start = time.perf_counter()
    prefill = model(input_ids=prompt_ids, use_cache=True, return_dict=True, logits_to_keep=1)
    manager.drain()
    prefill_end = time.perf_counter()
    cache = prefill.past_key_values
    scores = prefill.logits[:, -1]
    decode_start = prefill_end
    first = int(scores.argmax(-1).item())
    first_sample_ready = time.perf_counter()
    generated = [first]
    stopped = stopper.append(first, scores) if fixed_forwards is None else False
    production = 0
    pseudo_positions = 0
    joint_calls = 0
    bootstrap_wall = 0.0
    token_ready = []
    last_ready = time.perf_counter()
    previous_cache_length = int(cache.get_seq_length())
    if g and not stopped and cap > 1:
        start = time.perf_counter()
        runtime.kind = "bootstrap"
        manager.phase = "bootstrap_probe"
        anchors = anchor_ids(prompt, tuple(generated), config)
        fork = _fork_cache_copy_on_write(cache)
        model(
            input_ids=torch.tensor([anchors], device=prompt_ids.device),
            past_key_values=fork,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        if int(cache.get_seq_length()) != previous_cache_length:
            raise RuntimeError("bootstrap modified production KV")
        pseudo_positions += g
        runtime.initialize()
        manager.drain()
        bootstrap_wall = time.perf_counter() - start
    elif policy != "E" and not stopped and cap > 1:
        start = time.perf_counter()
        runtime.initialize()
        manager.drain()
        bootstrap_wall = time.perf_counter() - start
    while len(generated) < cap and (fixed_forwards is not None or not stopped):
        production += 1
        boundary = production % d == 0
        need_next = production < (
            fixed_forwards if fixed_forwards is not None else max_new_tokens - 1
        )
        joint = bool(g and boundary and need_next)
        manager.phase = "joint_compute" if joint else "production"
        runtime.kind = "joint" if joint else "production"
        current = torch.tensor([[generated[-1]]], device=prompt_ids.device)
        before = int(cache.get_seq_length())
        if joint:
            anchors = anchor_ids(prompt, tuple(generated), config)
            state = fork_bridge_only_cache(cache, joint_tokens=g + 1)
            ids = torch.tensor([[generated[-1], *anchors]], device=prompt_ids.device)
            output = model(
                input_ids=ids, past_key_values=state.cache, use_cache=True, return_dict=True
            )
            scores = output.logits[:, 0]
            commit_bridge_only_cache(cache, state)
            pseudo_positions += g
            joint_calls += 1
        else:
            output = model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            scores = output.logits[:, -1]
        if int(cache.get_seq_length()) != before + 1:
            raise RuntimeError("production must commit exactly one KV position")
        token = int(scores.argmax(-1).item())
        generated.append(token)
        stopped = stopper.append(token, scores) if fixed_forwards is None else False
        # argmax.item() is an actual token-ready synchronization, already needed for host stopping/content.  # noqa: E501
        ready = time.perf_counter()
        token_ready.append(
            {"forward": production, "boundary": boundary, "seconds": ready - last_ready}
        )
        last_ready = ready
        if boundary and need_next and not stopped:
            runtime.after_boundary()
    manager.drain()
    decode_end = time.perf_counter()
    # Formatting, metric traversal and answer evaluation are deliberately outside inference wall.
    reporting_start = time.perf_counter()
    metrics = manager.metrics()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    proc_status = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    }
    row = {
        "policy": policy,
        "mode": "controlled_fixed_length" if fixed_forwards is not None else "free_generation",
        "instrumentation": mode,
        "generated_token_ids": generated,
        "text": text,
        "generated_tokens": len(generated),
        "first_sampled_from_prefill": 1,
        "production_forwards": production,
        "pseudo_positions": pseudo_positions,
        "joint_calls": joint_calls,
        "final_sample_not_forwarded": 1,
        "residency_window_tokens": d,
        "pseudo_compute_tokens": g,
        "pseudo_content_horizon": 8,
        "selector_core_anchors": 4,
        "request_setup_seconds": prefill_start - request_start,
        "prefill_wall_seconds": prefill_end - prefill_start,
        "bootstrap_wall_seconds": bootstrap_wall,
        "decode_wall_seconds": decode_end - decode_start,
        "request_wall_seconds": decode_end - request_start,
        "time_to_first_token_seconds": first_sample_ready - request_start,
        "decode_seconds_per_production_forward": (decode_end - decode_start) / production
        if production
        else None,
        "host_rss_bytes": int(proc_status["VmRSS"].split()[0]) * 1024,
        "host_peak_rss_bytes": int(proc_status["VmHWM"].split()[0]) * 1024,
        "host_pinned_expert_bytes": metrics["host_expert_bytes"]
        if manager.host_mode == "full_pinned_cpu"
        else 0,
        "production_forwards_per_second": production / (decode_end - decode_start),
        "truncated": fixed_forwards is None and len(generated) >= cap and not stopped,
        "stopped": stopped,
        "cache_length": int(cache.get_seq_length()),
        "prompt_tokens": len(prompt),
        "token_ready_latencies": token_ready,
        "offload": metrics,
        "natural_route_residency_profile": {
            "scope": "before current layer demand load; production positions only",
            "columns": [
                "sum_topk_fraction_resident",
                "sum_natural_mass_resident",
                "production_positions",
            ],
            "per_layer": runtime.routing_diagnostics.cpu().tolist(),
        }
        if mode != "performance"
        else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "measured": True,
        "reference_tokens_used_for_generation": False,
    }
    row["metrics_reporting_seconds"] = time.perf_counter() - reporting_start
    return row
