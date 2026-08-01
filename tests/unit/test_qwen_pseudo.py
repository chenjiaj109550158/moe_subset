from __future__ import annotations

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_mutation_signature,
    _fork_cache_copy_on_write,
)


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


def _defaults() -> DefaultVectorArtifact:
    count = torch.ones((2, 4), dtype=torch.int64)
    count[1, 3] = 0
    mean = torch.arange(2 * 4 * 32, dtype=torch.float32).reshape(2, 4, 32)
    mean = mean / mean.abs().max()
    mean[1, 3] = 0
    return DefaultVectorArtifact(count, mean, "synthetic-default-fingerprint")


def _prefill(model: Qwen3MoeForCausalLM) -> object:
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            use_cache=True,
            return_dict=True,
        )
    assert output.past_key_values is not None
    return output.past_key_values


def test_native_qwen_probe_preserves_cache_rng_and_information_boundary() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    cache = _prefill(model)
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    probe = QwenPseudoEmbeddingProbe(
        model,
        ops,
        _defaults(),
        QwenPseudoVariant(
            "sampled_next_independent_default_topk",
            "sampled_next_token",
            "independent",
            "default_vector_selected_topk_mixture",
        ),
        anchors=(1, 2),
        budget=3,
    )
    result = probe.predict(
        cache,
        sampled_next_token_id=4,
        current_token_id=3,
    )
    assert result.absolute_positions == (3, 4)
    assert result.audit["production_cache_signature_unchanged"]
    assert result.audit["production_rng_unchanged"]
    assert result.audit["shadow_cache_discarded"]
    assert not result.audit["forbidden_inputs_present"]
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result.cost.attention_queries == 4
    assert result.cost.attention_calls == 4
    for layer in range(2):
        assert result.raw_router_logits[layer].shape == (2, 4)
        assert result.pre_topk_probabilities[layer].shape == (2, 4)
        assert torch.allclose(
            result.pre_topk_probabilities[layer].sum(-1),
            torch.ones(2),
        )
        assert len(result.subsets[layer]) == 3
        assert len(set(result.subsets[layer])) == 3


def test_first_anchor_first_layer_matches_native_qwen_forward() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    cache = _prefill(model)
    probe = QwenPseudoEmbeddingProbe(
        model,
        ops,
        _defaults(),
        QwenPseudoVariant(
            "sampled_next_independent_zero",
            "sampled_next_token",
            "independent",
            "zero",
        ),
        anchors=(1,),
        budget=3,
    )
    result = probe.predict(
        cache,
        sampled_next_token_id=4,
        current_token_id=3,
    )
    fork = _fork_cache_copy_on_write(cache)
    with NativeRouteCaptureContext(ops) as capture, torch.inference_mode():
        model(
            input_ids=torch.tensor([[4]]),
            past_key_values=fork,
            use_cache=True,
            return_dict=True,
        )
        records = capture.drain()
    assert len(records) == 2
    assert torch.equal(result.raw_router_logits[0][0], records[0].natural.logits[0])
    assert torch.equal(result.pseudo_topk_ids[0][0], records[0].natural.ids[0])
    assert torch.equal(
        result.pseudo_topk_weights[0][0],
        records[0].natural.weights[0].float(),
    )


def test_causal_and_current_content_ablation_use_native_qwen_paths() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    defaults = _defaults()
    causal = QwenPseudoEmbeddingProbe(
        model,
        ops,
        defaults,
        QwenPseudoVariant(
            "sampled_next_causal_default_topk",
            "sampled_next_token",
            "causal",
            "default_vector_selected_topk_mixture",
        ),
        anchors=(1, 2),
        budget=3,
    ).predict(_prefill(model), sampled_next_token_id=4, current_token_id=3)
    current = QwenPseudoEmbeddingProbe(
        model,
        ops,
        defaults,
        QwenPseudoVariant(
            "current_token_independent_default_topk",
            "current_token",
            "independent",
            "default_vector_selected_topk_mixture",
        ),
        anchors=(1, 2),
        budget=3,
    ).predict(_prefill(model), sampled_next_token_id=4, current_token_id=3)
    assert causal.cost.attention_calls == 2
    assert causal.cost.router_calls == 2
    assert not torch.equal(
        causal.raw_router_logits[0],
        current.raw_router_logits[0],
    )


def test_expected_top_m_uses_sampling_logits_without_rng_or_cache_mutation() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    probe = QwenPseudoEmbeddingProbe(
        model,
        ops,
        _defaults(),
        QwenPseudoVariant(
            "expected_top8_independent_default_topk",
            "expected_top_m",
            "independent",
            "default_vector_selected_topk_mixture",
        ),
        anchors=(1, 2),
        budget=3,
    )
    cache = _prefill(model)
    logits = torch.full((1, model.config.vocab_size), -10.0)
    logits[0, 3] = 2.0
    logits[0, 4] = 1.0
    result = probe.predict(
        cache,
        sampled_next_token_id=3,
        current_token_id=2,
        next_token_logits=logits,
        expected_top_m=2,
    )
    assert result.raw_router_logits[0].shape == (2, 4)
    assert result.audit["expected_top_m"] == 2
    assert result.audit["production_cache_signature_unchanged"] is True
    assert result.audit["production_rng_unchanged"] is True
