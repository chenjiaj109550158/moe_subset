from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.qwen_pseudo import (
    MidlayerSelfConditioning,
    QwenPseudoEmbeddingProbe,
    QwenPseudoProbeResult,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.subset_closed_loop import _cache_mutation_signature


def _model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(41)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
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
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            use_cache=True,
            return_dict=True,
        )
    assert output.past_key_values is not None
    return output.past_key_values


def _run(
    model: Qwen3MoeForCausalLM,
    cache: object,
    mode: MidlayerSelfConditioning,
    cap: float | None = None,
) -> QwenPseudoProbeResult:
    ops = Qwen3MoePrefetchOps(model)
    subsets = {layer: (0, 1, 2) for layer in range(ops.num_layers)}
    return QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            f"midlayer_{mode}_{cap}",
            "provided_sequence",
            "causal",
            "native_expert_execution",
            midlayer_self_conditioning=mode,
            midlayer_refresh_after_layer=(1 if mode != "none" else None),
            midlayer_max_relative_embedding_delta_norm=cap,
        ),
        anchors=(1, 2, 3),
        budget=3,
    ).predict(
        cache,
        sampled_next_token_id=5,
        current_token_id=4,
        anchor_token_ids=(5, 3, 4),
        execution_subsets=subsets,
        execution_subset_source="previous_realized_window_subset",
    )


def test_midlayer_greedy_refresh_changes_only_later_nonfirst_anchor_routes() -> None:
    model = _model()
    cache = _prefill(model)
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    baseline = _run(model, cache, "none")
    refreshed = _run(model, cache, "greedy_shift")

    for layer in (0, 1):
        assert torch.equal(refreshed.raw_router_logits[layer], baseline.raw_router_logits[layer])
    for layer in range(4):
        assert torch.equal(
            refreshed.raw_router_logits[layer][0],
            baseline.raw_router_logits[layer][0],
        )
    assert any(
        not torch.allclose(
            refreshed.raw_router_logits[layer][1:],
            baseline.raw_router_logits[layer][1:],
        )
        for layer in (2, 3)
    )
    assert refreshed.cost.attention_calls == 4
    assert refreshed.cost.attention_queries == 12
    assert refreshed.cost.router_calls == 4
    assert refreshed.cost.expert_calls == 4
    assert refreshed.cost.lm_head_calls == 1
    assert refreshed.cost.lm_head_queries == 2
    assert refreshed.cost.midlayer_refreshes == 1
    assert refreshed.audit["midlayer_anchor_one_hidden_bitwise_unchanged"] is True
    assert refreshed.audit["midlayer_refresh_finite"] is True
    assert refreshed.audit["midlayer_hidden_norm_preserved"] is True
    assert refreshed.audit["midlayer_lm_head_logits_shape"] == [1, 2, 64]
    assert refreshed.audit["production_cache_signature_unchanged"] is True
    assert refreshed.audit["production_rng_unchanged"] is True
    assert refreshed.audit["one_causal_forward_per_boundary"] is True
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_expected_top8_and_capped_greedy_refresh_are_finite_and_audited() -> None:
    model = _model()
    cache = _prefill(model)
    expected = _run(model, cache, "expected_top8_shift")
    capped = _run(model, cache, "greedy_shift", 0.25)

    assert expected.audit["midlayer_self_conditioning"] == "expected_top8_shift"
    assert len(expected.audit["midlayer_expected_top8_token_ids"]) == 2
    assert all(len(values) == 8 for values in expected.audit["midlayer_expected_top8_token_ids"])
    assert all(
        abs(sum(values) - 1) < 1e-5 for values in expected.audit["midlayer_expected_top8_weights"]
    )
    assert capped.audit["midlayer_max_relative_embedding_delta_norm"] == 0.25
    assert max(capped.audit["midlayer_applied_embedding_delta_norm_ratio"]) <= 0.2501
    assert capped.audit["midlayer_anchor_one_hidden_bitwise_unchanged"] is True
    assert capped.audit["midlayer_refresh_finite"] is True
