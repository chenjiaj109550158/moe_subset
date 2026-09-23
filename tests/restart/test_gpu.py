import copy
import math

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.qwen_penultimate_joint import (
    commit_bridge_only_cache,
    fork_bridge_only_cache,
)
from pseudoroute.restart.backend import MoEComputeBackend, numerical_metrics
from pseudoroute.restart.model import PolicyRuntime, SlotMlp
from pseudoroute.restart.residency import HostLayer, ResidencyManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no fallback")


def layers(e=8, h=64, i=32):
    return HostLayer(
        torch.randn(e, 2 * i, h, dtype=torch.bfloat16).pin_memory() / math.sqrt(h),
        torch.randn(e, h, i, dtype=torch.bfloat16).pin_memory() / math.sqrt(i),
    )


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_slot_permutation_churn_streams_zero_missing_and_prefill_union(slots):
    torch.manual_seed(19)
    host = layers(e=16)
    # Arithmetic may return an ordinary CPU tensor; explicitly pin backing weights.
    host = HostLayer(host.gate_up.pin_memory(), host.down.pin_memory())
    backend = MoEComputeBackend(audit=True)
    manager = ResidencyManager([host], slots, backend, mode="audit")
    x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    gu, dw = host.gate_up.cuda(), host.down.cuda()
    reference = MoEComputeBackend("python_reference", audit=True)
    for cycle in range(5):
        if cycle == 3:
            manager.reset()
        ids = torch.stack([torch.randperm(16, device="cuda")[:2] for _ in range(7)])
        weights = torch.full((7, 2), 0.5, device="cuda", dtype=torch.bfloat16)
        actual = manager.execute(0, x, ids, weights)
        expected = reference.forward(x, ids, weights, gu, dw, {})
        manager.drain()
        metric = numerical_metrics(actual, expected)
        assert metric["nrmse"] < 0.01 and metric["cosine"] > 0.999
        target = tuple(range((cycle % 2) * slots, (cycle % 2 + 1) * slots))
        manager.load(0, target, asynchronous=True)
        resident_ids = torch.tensor([target[:2]] * 7, device="cuda")
        y = manager.execute(0, x, resident_ids, weights, resident=True)
        manager.drain()
        assert (
            numerical_metrics(y, reference.forward(x, resident_ids, weights, gu, dw, {}))["nrmse"]
            < 0.01
        )
    zero_ids = torch.full((7, 2), -1, device="cuda")
    assert torch.equal(
        backend.forward(x, zero_ids, torch.zeros_like(weights), gu, dw, {}), torch.zeros_like(x)
    )
    with pytest.raises(ValueError):
        backend.forward(x, zero_ids, weights, gu, dw, {})
    with pytest.raises(ValueError):
        backend.forward(x[:0], zero_ids[:0], weights[:0], gu, dw, {})
    with pytest.raises(ValueError):
        backend.forward(x[:, ::2], resident_ids, weights, gu, dw, {})


def tiny_runtime(mode="audit"):
    torch.manual_seed(27)
    c = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=256,
    )
    model = Qwen3MoeForCausalLM(c).to("cuda", dtype=torch.bfloat16).eval()
    native = copy.deepcopy(model)
    stores = [
        HostLayer(
            layer.mlp.experts.gate_up_proj.detach().cpu().pin_memory(),
            layer.mlp.experts.down_proj.detach().cpu().pin_memory(),
        )
        for layer in model.model.layers
    ]
    manager = ResidencyManager(stores, 4, MoEComputeBackend(audit=True), mode=mode)
    runtime = PolicyRuntime(manager, 8, 2, mode=mode)
    for i, layer in enumerate(model.model.layers):
        layer.mlp = SlotMlp(layer.mlp.gate, runtime, i)
    return model, native, runtime


@torch.inference_mode()
def test_tiny_full_forward_native_audit_performance_and_joint_cache():
    model, native, runtime = tiny_runtime()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    expected = native(input_ids=ids, use_cache=True)
    output = model(input_ids=ids, use_cache=True)
    runtime.manager.drain()
    metric = numerical_metrics(output.logits, expected.logits)
    assert metric["nrmse"] < 0.01 and metric["cosine"] > 0.999
    audit = output.logits.clone()
    runtime.manager.reset()
    runtime.reset("E")
    runtime.mode = runtime.manager.mode = "performance"
    output = model(input_ids=ids, use_cache=True)
    assert torch.equal(output.logits, audit)
    runtime.manager.drain()
    cache = output.past_key_values
    for g in (4, 8):
        prefix = [layer.keys.clone() for layer in cache.layers]
        length = cache.get_seq_length()
        state = fork_bridge_only_cache(cache, joint_tokens=g + 1)
        runtime.kind = "production"
        seq = torch.tensor([[9] * (g + 1)], device="cuda")
        rng = torch.random.get_rng_state().clone()
        crng = torch.cuda.get_rng_state().clone()
        joint = model(input_ids=seq, past_key_values=state.cache, use_cache=True)
        assert cache.get_seq_length() == length
        assert all(
            torch.equal(p, layer.keys) for p, layer in zip(prefix, cache.layers, strict=True)
        )
        assert torch.equal(rng, torch.random.get_rng_state()) and torch.equal(
            crng, torch.cuda.get_rng_state()
        )
        commit_bridge_only_cache(cache, state)
        assert cache.get_seq_length() == length + 1
        with pytest.raises(RuntimeError):
            commit_bridge_only_cache(cache, state)
        assert torch.isfinite(joint.logits).all()


@pytest.mark.parametrize("policy,g", [("H", 0), ("P4", 4), ("P8", 8)])
@pytest.mark.parametrize("termination", ["eos", "stop"])
@torch.inference_mode()
def test_short_prompt_eos_stop_boundary_and_production_counts(policy, g, termination):
    from pathlib import Path

    from transformers import AutoTokenizer

    from pseudoroute.restart.inference import generate

    model, _, runtime = tiny_runtime(mode="performance")
    tokenizer = AutoTokenizer.from_pretrained(
        Path(
            ".cache/restart/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        ),
        local_files_only=True,
    )
    model.generation_config.eos_token_id = 99 if termination == "eos" else None
    original = model.forward
    production = 0

    def scripted_logits(*args, **kwargs):
        nonlocal production
        result = original(*args, **kwargs)
        if runtime.kind in ("production", "joint"):
            production += 1
        token = 3
        if production == 8:
            token = (
                99 if termination == "eos" else tokenizer.encode(":", add_special_tokens=False)[0]
            )
        elif production == 7 and termination == "stop":
            token = tokenizer.encode("Q", add_special_tokens=False)[0]
        result.logits.fill_(-100)
        result.logits[..., token] = 100
        return result

    model.forward = scripted_logits
    rng = torch.random.get_rng_state().clone()
    generation_rng = torch.Generator(device="cuda").manual_seed(101)
    before = generation_rng.get_state().clone()
    row = generate(
        model,
        tokenizer,
        runtime,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device="cuda"),
        policy=policy,
        max_new_tokens=20,
        stop_strings=("Q:",) if termination == "stop" else (),
    )
    assert row["production_forwards"] == 8 and row["generated_tokens"] == 9
    assert row["stopped"] and not row["truncated"]
    assert row["cache_length"] == 15
    assert row["pseudo_compute_tokens"] == g
    assert torch.equal(before, generation_rng.get_state())
    assert torch.equal(rng, torch.random.get_rng_state())
    # An independent stochastic generation generator sees no planning draws.
    reference = torch.Generator(device="cuda").manual_seed(101)
    probabilities = torch.ones(128, device="cuda") / 128
    assert torch.equal(
        torch.multinomial(probabilities, 16, generator=generation_rng),
        torch.multinomial(probabilities, 16, generator=reference),
    )
