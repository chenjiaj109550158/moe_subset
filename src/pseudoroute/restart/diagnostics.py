"""Correctness checks on actual checkpoint weights, before any method claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pseudoroute.restart.backend import MoEComputeBackend, numerical_metrics
from pseudoroute.restart.inference import generate
from pseudoroute.restart.microbench import fp32_reference
from pseudoroute.restart.model import PolicyRuntime
from pseudoroute.restart.state import write_json


@torch.inference_mode()
def real_model_checks(
    model: Any,
    tokenizer: Any,
    runtime: PolicyRuntime,
    prompt: torch.Tensor,
    out: Path,
    additional_prompts: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = []
    manager = runtime.manager
    # Real checkpoint MLP weights; quantized BF16 values promoted to FP32 reference.
    manager.phase = "correctness"
    manager.load(0, tuple(range(manager.slots)))
    gu, dw = manager.weights[0]
    generator = torch.Generator(device="cuda").manual_seed(20260922)
    x = (
        torch.randn(
            5, model.config.hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.05
    )
    ids = torch.arange(8, device="cuda")[None].expand(5, -1).contiguous()
    weights = torch.full((5, 8), 0.125, device="cuda", dtype=torch.bfloat16)
    actual = manager.backend.forward(x, ids, weights, gu, dw, {})
    expected = fp32_reference(x, ids, weights, gu, dw)
    m = numerical_metrics(actual, expected)
    m["name"] = "real_layer_fixed_routes_fp32_reference"
    checks.append(m)
    # Cross-implementation controls are calibrated separately from exact lifecycle parity.
    # v1's global 1% bound also failed the unchanged Transformers native control.
    import json

    protocol = json.loads((out / "correctness_protocol.json").read_text())
    for prefix_index, case in enumerate([prompt, *(additional_prompts or [])]):
        manager.reset()
        runtime.reset("E")
        runtime.capture_fixed_routes = True
        manager.backend = MoEComputeBackend("python_reference")
        inputs: dict[int, torch.Tensor] = {}
        handles = []
        if prefix_index == 0:
            for layer_index, layer in enumerate(model.model.layers):

                def capture(
                    module: Any,
                    args: tuple[Any, ...],
                    i: int = layer_index,
                    target: dict[int, torch.Tensor] = inputs,
                ) -> None:
                    target[i] = args[0].reshape(-1, args[0].shape[-1])[:4].clone()

                handles.append(layer.mlp.register_forward_pre_hook(capture))
        reference = model(input_ids=case, use_cache=True, return_dict=True)
        manager.drain()
        for handle in handles:
            handle.remove()
        routes = runtime.captured_routes
        reference_logits = reference.logits.clone()
        del reference
        manager.reset()
        runtime.reset("E")
        runtime.fixed_routes = routes
        manager.backend = MoEComputeBackend("vllm_fused")
        actual = model(input_ids=case, use_cache=True, return_dict=True)
        manager.drain()
        m = numerical_metrics(actual.logits, reference_logits)
        m.update(
            name=f"full_prefix_{prefix_index}_fixed_routes_reference_vs_fused",
            acceptance_max_nrmse=protocol["full_prefix_cross_implementation"]["max_nrmse"],
        )
        checks.append(m)
        # Same backend/math: explicitly synchronous copies must match event ordering bitwise.
        asynchronous_logits = actual.logits.clone()
        del actual, reference_logits
        original_load = manager.load

        def synchronous_load(
            layer: int,
            experts: tuple[int, ...],
            *,
            asynchronous: bool = False,
            loader: Any = original_load,
        ) -> None:
            loader(layer, experts, asynchronous=asynchronous)
            torch.cuda.synchronize()

        manager.load = synchronous_load  # type: ignore[method-assign]
        try:
            manager.reset()
            runtime.reset("E")
            runtime.fixed_routes = routes
            sync = model(input_ids=case, use_cache=True, return_dict=True)
            manager.drain()
            m = numerical_metrics(sync.logits, asynchronous_logits)
            m.update(
                name=f"full_prefix_{prefix_index}_synchronous_vs_event_ordered",
                bitwise_required=True,
                bitwise_equal=torch.equal(sync.logits, asynchronous_logits),
            )
            checks.append(m)
            del sync, asynchronous_logits
        finally:
            manager.load = original_load  # type: ignore[method-assign]
        for layer, x in inputs.items():
            ids, weights = (tensor[:4] for tensor in routes[layer])
            manager.reset()
            y = manager.execute(layer, x.contiguous(), ids, weights, resident=False)
            high = fp32_reference(x, manager.maps[layer][ids], weights, *manager.weights[layer])
            m = numerical_metrics(y, high)
            m.update(name=f"actual_input_layer_{layer}_fp32_reference", acceptance_max_nrmse=0.01)
            checks.append(m)
        del inputs, routes
    runtime.reset("E")
    passed = all(
        m["finite"]
        and m["nrmse"] <= m.get("acceptance_max_nrmse", 0.01)
        and m["cosine"] >= 0.999
        and (not m.get("bitwise_required") or m["bitwise_equal"])
        for m in checks
    )
    result = {
        "state": "PASS" if passed else "FAIL",
        "checks": checks,
        "model_id": model.config._name_or_path,
        "full_resident_reference": False,
        "correctness_protocol_revision": 2,
        "cross_implementation_threshold_is_revised": True,
        "calibration_evidence": "docs/restart/numerical_protocol_v2.md",
        "reference_scope": "CPU-first synchronous reference compute plus real-layer FP32 and tiny native checks",  # noqa: E501
    }
    write_json(out / "real_model_correctness.json", result)
    if not passed:
        raise RuntimeError("full-checkpoint numerical gate failed")
    compare_g4_g8(model, runtime, prompt, out)
    # Exercise at least two complete window transitions and partial final window.
    rows = []
    for policy in ("E", "S", "H", "HC", "P8", "P4"):
        row = generate(model, tokenizer, runtime, prompt, policy=policy, fixed_forwards=19)
        rows.append(row)
        write_json(out / "tests" / f"real_smoke_{policy}.json", row)
        assert row["production_forwards"] == 19 and row["generated_tokens"] == 20
        assert row["cache_length"] == row["prompt_tokens"] + 19
        if policy.startswith("P"):
            assert row["joint_calls"] == 2
        print(
            f"real smoke {policy}: {row['production_forwards_per_second']:.3f} forwards/s",
            flush=True,
        )
    return result


def profile_runtime(
    model: Any, tokenizer: Any, runtime: PolicyRuntime, prompt: torch.Tensor, out: Path
) -> None:
    """Separate profile runs include true H2D waits; durations may overlap."""
    summaries = []
    runtime.mode = runtime.manager.mode = "profile"
    for policy in ("E", "S", "H"):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            row = generate(
                model, tokenizer, runtime, prompt, policy=policy, fixed_forwards=16, mode="profile"
            )
        prof.export_chrome_trace(str(out / "profiles" / f"runtime_{policy}.json"))
        (out / "profiles" / f"runtime_{policy}.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        )
        summaries.append(
            {
                "policy": policy,
                "instrumentation": "profile_not_formal_throughput",
                "row": row,
                "overlapping_durations_must_not_be_summed_as_wall": True,
            }
        )
    write_json(out / "profiles/runtime_summary.json", summaries)
    runtime.mode = runtime.manager.mode = "performance"


@torch.inference_mode()
def compare_g4_g8(model: Any, runtime: PolicyRuntime, prompt: torch.Tensor, out: Path) -> None:
    from pseudoroute.benchmark.qwen_penultimate_joint import fork_bridge_only_cache
    from pseudoroute.benchmark.subset_closed_loop import _fork_cache_copy_on_write
    from pseudoroute.restart.policies import WindowConfig, anchor_ids

    manager = runtime.manager
    manager.reset()
    runtime.reset("P8")
    prefix = model(input_ids=prompt, use_cache=True, return_dict=True, logits_to_keep=1)
    cache = prefix.past_key_values
    sampled = int(prefix.logits[:, -1].argmax(-1).item())
    tokens = tuple(prompt[0].tolist())
    anchors = {
        g: anchor_ids(tokens, (sampled,), WindowConfig(pseudo_compute_tokens=g)) for g in (4, 8)
    }
    assert anchors[4] == anchors[8][:4]
    prefix_tensors = [layer.keys.clone() for layer in cache.layers]
    results = {}
    bootstrap_sets = None
    for kind in ("bootstrap", "joint"):
        records = {}
        for g in (8, 4):
            manager.reset()
            runtime.policy = f"P{g}"
            runtime.kind = kind
            if kind == "joint":
                assert bootstrap_sets is not None
                for layer, subset in bootstrap_sets.items():
                    runtime.set_subset(layer, subset)
                    runtime.window[layer].zero_()
                fork = fork_bridge_only_cache(cache, joint_tokens=g + 1).cache
                input_ids = torch.tensor([[sampled, *anchors[g]]], device=prompt.device)
            else:
                fork = _fork_cache_copy_on_write(cache)
                input_ids = torch.tensor([anchors[g]], device=prompt.device)
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            output = model(
                input_ids=input_ids, past_key_values=fork, use_cache=True, return_dict=True
            )
            manager.drain()
            assert torch.equal(cpu_rng, torch.random.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            assert all(
                torch.equal(saved, layer.keys)
                for saved, layer in zip(prefix_tensors, cache.layers, strict=True)
            )
            subsets = (
                {
                    layer: runtime.select(layer, bootstrap=kind == "bootstrap")
                    for layer in range(len(runtime.history))
                }
                if kind == "bootstrap"
                else dict(runtime.next_subsets)
            )
            if kind == "bootstrap" and g == 8:
                bootstrap_sets = subsets
            records[g] = {
                "logits": output.logits[:, : 4 + int(kind == "joint")].clone(),
                "routes": [
                    runtime.pseudo[layer][1][:4].clone() for layer in range(len(runtime.history))
                ],
                "subsets": subsets,
            }
        a, b = records[4], records[8]
        metric = numerical_metrics(a["logits"], b["logits"])
        results[kind] = {
            "same_first_four_token_ids": True,
            "same_prefix_and_positions": True,
            "production_prefix_unmodified": True,
            "rng_unchanged": True,
            "logits_metrics": metric,
            "first_four_route_id_agreement": sum(
                float((x == y).float().mean())
                for x, y in zip(a["routes"], b["routes"], strict=True)
            )
            / len(a["routes"]),
            "subset_layers_exact": sum(
                a["subsets"][layer] == b["subsets"][layer] for layer in a["subsets"]
            ),
            "layers": len(a["subsets"]),
            "bitwise_identity_assumed": False,
        }
    runtime.reset("E")
    write_json(out / "g4_g8_semantics.json", results)
