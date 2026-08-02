from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.prefetch import NativeRoute, Qwen3MoePrefetchOps
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.subset_closed_loop import _cache_mutation_signature


def _model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(17)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=128,
    )
    return Qwen3MoeForCausalLM(config).eval()


def _prefill(model: Qwen3MoeForCausalLM) -> object:
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            use_cache=True,
            return_dict=True,
        )
    assert output.past_key_values is not None
    return output.past_key_values


def _probe(
    model: Qwen3MoeForCausalLM,
    ops: Qwen3MoePrefetchOps,
) -> QwenPseudoEmbeddingProbe:
    return QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            "sampled_repeat_native_previous_subset",
            "sampled_next_token",
            "independent",
            "native_expert_execution",
        ),
        anchors=(1, 2),
        budget=3,
    )


def test_native_expert_execution_uses_supplied_subset_and_fresh_residual() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    subsets = {layer: (0, 1, 2) for layer in range(ops.num_layers)}
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(ops.num_layers)}
    original = ops.experts

    def experts(layer: int, hidden: torch.Tensor, route: NativeRoute) -> torch.Tensor:
        value = original(layer, hidden, route)
        captured[layer].append(value.detach().clone())
        return value

    ops.experts = experts  # type: ignore[method-assign]
    cache = _prefill(model)
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    result = _probe(model, ops).predict(
        cache,
        sampled_next_token_id=4,
        current_token_id=3,
        execution_subsets=subsets,
        execution_subset_source="previous_realized_window_subset",
    )
    for layer in range(ops.num_layers):
        assert result.raw_router_logits[layer].shape == (2, ops.num_experts)
        assert set(result.shadow_executed_topk_ids[layer].reshape(-1).tolist()) <= set(
            subsets[layer]
        )
        assert torch.equal(result.shadow_moe_residuals[layer], torch.cat(captured[layer]))
        assert torch.isfinite(result.shadow_moe_residuals[layer]).all()
        assert torch.count_nonzero(result.shadow_moe_residuals[layer])
    assert result.cost.expert_calls == ops.num_layers * 2
    assert result.cost.router_calls == ops.num_layers * 2
    assert result.audit["shadow_expert_execution"] is True
    assert result.audit["full_pre_mask_scores_all_experts"] is True
    assert result.audit["executed_ids_within_supplied_subset"] is True
    assert result.audit["shadow_residual_finite"] is True
    assert result.audit["shadow_residual_nonzero"] is True
    assert result.audit["production_cache_signature_unchanged"] is True
    assert result.audit["production_rng_unchanged"] is True
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_native_expert_execution_without_subset_uses_full_natural_topk() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    result = _probe(model, ops).predict(
        _prefill(model),
        sampled_next_token_id=4,
        current_token_id=3,
        execution_subset_source="full_native_topk_prefill_access",
    )
    for layer in range(ops.num_layers):
        assert torch.equal(
            result.shadow_executed_topk_ids[layer],
            result.pseudo_topk_ids[layer],
        )
    assert result.audit["execution_scope"] == "full_native_topk"
    assert result.audit["executed_ids_within_supplied_subset"] is None
