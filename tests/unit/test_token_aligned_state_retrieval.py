from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    token_aligned_state_retrieval,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
    _retrieval_mix_norm_match,
)
from pseudoroute.benchmark.subset_closed_loop import _cache_mutation_signature


def _model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(29)
    config = Qwen3MoeConfig(
        vocab_size=16,
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


def test_token_retrieval_uses_most_recent_exact_and_zero_weight_missing() -> None:
    router = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    residual = -router
    embeddings = torch.eye(6, 3)
    result = token_aligned_state_retrieval(
        (1, 4),
        (1, 2, 1, 3),
        router,
        residual,
        embeddings,
        "exact_token_most_recent",
    )

    assert result.history_indices == (2, -1)
    assert result.match_kinds == (
        "exact_token_most_recent",
        "missing_exact_zero_weight",
    )
    assert result.similarities.tolist() == [1.0, 0.0]
    assert torch.equal(result.router_inputs[0], router[2])
    assert torch.count_nonzero(result.router_inputs[1]) == 0


def test_embedding_fallback_breaks_equal_cosine_by_most_recent_history() -> None:
    router = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    embeddings = torch.zeros(6, 3)
    embeddings[1] = torch.tensor([1.0, 0.0, 0.0])
    embeddings[2] = torch.tensor([0.0, 1.0, 0.0])
    embeddings[3] = torch.tensor([0.0, 1.0, 0.0])
    embeddings[4] = torch.tensor([0.0, 1.0, 0.0])
    result = token_aligned_state_retrieval(
        (1, 4),
        (1, 2, 1, 3),
        router,
        -router,
        embeddings,
        "exact_token_then_embedding_nearest",
    )

    assert result.history_indices == (2, 3)
    assert result.match_kinds == (
        "exact_token_most_recent",
        "embedding_nearest_most_recent_tie",
    )
    assert result.similarities.tolist() == [1.0, 1.0]
    assert torch.equal(result.moe_outputs[1], -router[3])


def test_retrieval_mixing_preserves_fresh_anchor_norms() -> None:
    fresh = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    retrieved = torch.tensor([[0.0, 4.0], [3.0, 0.0]])
    similarities = torch.tensor([1.0, 0.5])
    additive = _retrieval_mix_norm_match(
        fresh,
        retrieved,
        similarities,
        mixing="additive_norm",
    )
    replaced = _retrieval_mix_norm_match(
        fresh,
        retrieved,
        similarities,
        mixing="replace_norm",
    )

    assert torch.allclose(additive.norm(dim=-1), fresh.norm(dim=-1))
    assert torch.allclose(replaced.norm(dim=-1), fresh.norm(dim=-1))
    assert not torch.equal(additive, replaced)


def test_native_retrieval_targets_preserve_cache_rng_and_one_forward_calls() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    cache = _prefill(model)
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    subsets = {layer: (0, 1, 2) for layer in range(ops.num_layers)}
    retrieved = torch.linspace(
        -2,
        2,
        2 * ops.num_layers * ops.hidden_size,
    ).reshape(2, ops.num_layers, ops.hidden_size)
    similarities = torch.ones(2)

    def run(target: str) -> object:
        return QwenPseudoEmbeddingProbe(
            model,
            ops,
            None,
            QwenPseudoVariant(
                f"retrieval_{target}",
                "provided_sequence",
                "causal",
                "native_expert_execution",
                state_retrieval_target=target,  # type: ignore[arg-type]
                state_retrieval_mix="additive_norm",
            ),
            anchors=(1, 2),
            budget=3,
        ).predict(
            cache,
            sampled_next_token_id=4,
            current_token_id=3,
            anchor_token_ids=(4, 3),
            execution_subsets=subsets,
            execution_subset_source="previous_realized_window_subset",
            retrieved_router_inputs=retrieved if target == "router_input" else None,
            retrieved_moe_residuals=retrieved if target == "moe_residual" else None,
            retrieval_similarities=similarities,
            retrieval_state_source="current_policy_pre_boundary_state_bank",
        )

    baseline = QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            "retrieval_baseline",
            "provided_sequence",
            "causal",
            "native_expert_execution",
        ),
        anchors=(1, 2),
        budget=3,
    ).predict(
        cache,
        sampled_next_token_id=4,
        current_token_id=3,
        anchor_token_ids=(4, 3),
        execution_subsets=subsets,
        execution_subset_source="previous_realized_window_subset",
    )
    router = run("router_input")
    residual = run("moe_residual")

    assert not torch.allclose(router.raw_router_logits[0], baseline.raw_router_logits[0])
    assert torch.equal(residual.raw_router_logits[0], baseline.raw_router_logits[0])
    assert not torch.allclose(residual.raw_router_logits[1], baseline.raw_router_logits[1])
    for result in (router, residual):
        assert result.cost.attention_calls == ops.num_layers
        assert result.cost.attention_queries == ops.num_layers * 2
        assert result.cost.router_calls == ops.num_layers
        assert result.cost.expert_calls == ops.num_layers
        assert result.audit["production_cache_signature_unchanged"] is True
        assert result.audit["production_rng_unchanged"] is True
        assert result.audit["one_causal_forward_per_boundary"] is True
        assert result.audit["retrieval_state_finite"] is True
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)
