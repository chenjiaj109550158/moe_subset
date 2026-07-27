from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from pseudoroute.execution.routing_policy import MaskedSubstitutionPolicy, NaturalRoutingPolicy
from pseudoroute.models.adapters.hf_deepseek_v2 import (
    DEEPSEEK_V2_LITE_REVISION,
    HFDeepseekV2Adapter,
    transformers_cache_compatibility_shim,
)
from pseudoroute.models.adapters.hf_gpt_oss import HFGptOssAdapter
from pseudoroute.models.adapters.hf_qwen2_moe import HFQwen2MoeAdapter
from pseudoroute.models.base import TraceLevel, TraceRequest
from pseudoroute.types import ExpertKey


def test_qwen_native_router_shared_expert_and_commitment_parity() -> None:
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    config = Qwen2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
    )
    config.architectures = ["Qwen2MoeForCausalLM"]
    torch.manual_seed(11)
    subject = HFQwen2MoeAdapter(
        Qwen2MoeForCausalLM(config).eval(),  # type: ignore[no-untyped-call]
        model_id="literal",
        revision="0" * 40,
    )
    tokens = torch.tensor([[1, 2, 3]])
    base = subject.run_base_forward(tokens)
    traced = subject.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    natural = subject.forward_with_policy(tokens, NaturalRoutingPolicy())
    assert torch.equal(base.logits, traced.logits)
    assert torch.equal(base.logits, natural.logits)
    assert all(handle.shared_expert_count == 1 for handle in subject.iter_moe_layers())
    first = traced.traces[0]
    routed = subject.route_from_state(0, torch.randn(1, 16))
    assert torch.equal(routed.pre_topk_scores, routed.raw_logits.float().softmax(dim=-1))
    assert first.topk_weights.sum() < 1.0
    allowed = frozenset(
        ExpertKey(layer, expert) for layer in subject.spec.moe_layer_indices for expert in (0, 1)
    )
    committed = subject.forward_with_policy(
        tokens, MaskedSubstitutionPolicy(), allowed_experts=allowed
    )
    assert torch.isfinite(committed.logits).all()
    assert all(
        set(record.executed_topk_ids.reshape(-1).tolist()).issubset({0, 1})
        for record in committed.executed_routes
    )


class FakeDeepGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 8))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        assert hidden_states.ndim == 3
        logits = torch.nn.functional.linear(
            hidden_states.reshape(-1, 8).float(), self.weight.float()
        )
        scores = logits.softmax(dim=-1)
        weights, ids = scores.topk(2, dim=-1, sorted=False)
        return ids, weights * 0.75, None


class FakeDeepLayer(nn.Module):
    def __init__(self, sparse: bool) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        if sparse:
            self.mlp = nn.Module()
            self.mlp.gate = FakeDeepGate()
            self.mlp.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            self.mlp.shared_experts = nn.Linear(8, 8)
        else:
            self.mlp = nn.Linear(8, 8)


class FakeDeepBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList(
            [FakeDeepLayer(False), FakeDeepLayer(True), FakeDeepLayer(True)]
        )


class FakeDeepModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["DeepseekV2ForCausalLM"],
            num_hidden_layers=3,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            n_routed_experts=4,
            n_shared_experts=2,
            num_experts_per_tok=2,
            hidden_size=8,
            norm_topk_prob=False,
            routed_scaling_factor=0.75,
            scoring_func="softmax",
            topk_method="greedy",
        )
        self.model = FakeDeepBase()
        self.lm_head = nn.Linear(8, 32)

    def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> SimpleNamespace:
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers[1:]:
            gate = cast(Any, layer.mlp).gate
            ids, weights, _ = gate(hidden)
            hidden = hidden + weights.mean(dim=-1).reshape(*hidden.shape[:2], 1)
            assert ids.shape[-1] == 2
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=None)


def test_deepseek_sparse_indices_scaling_and_shared_semantics() -> None:
    subject = HFDeepseekV2Adapter(
        FakeDeepModel(), model_id="literal", revision=DEEPSEEK_V2_LITE_REVISION
    )
    assert subject.spec.moe_layer_indices == (1, 2)
    assert subject.spec.shared_experts_by_layer == {1: 2, 2: 2}
    tokens = torch.tensor([[1, 2, 3]])
    base = subject.run_base_forward(tokens)
    traced = subject.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    natural = subject.forward_with_policy(tokens, NaturalRoutingPolicy())
    assert torch.equal(base.logits, traced.logits)
    assert torch.equal(base.logits, natural.logits)
    assert len(traced.traces) == 6
    first = traced.traces[0]
    assert first.layer_idx == 1
    assert torch.allclose(first.pre_topk_scores.sum(dim=-1), torch.ones(1))
    assert torch.allclose(
        first.topk_weights.sum(dim=-1), first.pre_topk_scores.topk(2).values.sum(dim=-1) * 0.75
    )
    routed = subject.route_from_state(1, torch.randn(1, 8))
    assert routed.raw_logits.shape == (1, 4)
    assert routed.topk_ids.shape == (1, 2)
    allowed = frozenset(
        ExpertKey(layer, expert) for layer in subject.spec.moe_layer_indices for expert in (0, 1)
    )
    committed = subject.forward_with_policy(
        tokens, MaskedSubstitutionPolicy(), allowed_experts=allowed
    )
    assert torch.isfinite(committed.logits).all()
    assert all(
        set(record.executed_topk_ids.reshape(-1).tolist()).issubset({0, 1})
        for record in committed.executed_routes
    )


def test_deepseek_cache_shim_is_minimal_and_reversible() -> None:
    from transformers import DynamicCache

    existing = getattr(DynamicCache, "get_usable_length", None)
    with transformers_cache_compatibility_shim():
        cache = DynamicCache()
        get_usable_length = cast(Any, cache).get_usable_length
        assert get_usable_length(1) == 0
    assert getattr(DynamicCache, "get_usable_length", None) is existing


def test_gpt_oss_router_uses_topk_logits_then_selected_softmax() -> None:
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(  # type: ignore[call-arg]
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        quantization_config={"quant_method": "mxfp4"},
    )
    config.architectures = ["GptOssForCausalLM"]
    torch.manual_seed(17)
    subject = HFGptOssAdapter(
        GptOssForCausalLM(config).eval(),  # type: ignore[no-untyped-call]
        model_id="literal",
        revision="0" * 40,
    )
    tokens = torch.tensor([[1, 2]])
    traced = subject.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    first = traced.traces[0]
    assert torch.equal(first.raw_logits, first.pre_topk_scores)
    expected = first.raw_logits.gather(-1, first.topk_ids).softmax(dim=-1)
    assert torch.equal(first.topk_weights, expected)
    assert all(handle.has_router_bias for handle in subject.iter_moe_layers())
