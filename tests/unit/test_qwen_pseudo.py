from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    MoeOutputCaptureContext,
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
    SubsetExecutionContext,
    moe_output_bank,
    moe_router_input_bank,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_mutation_signature,
    _fork_cache_copy_on_write,
    natural_token_lookahead_copy_on_write,
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


def test_natural_token_lookahead_is_copy_on_write_and_rng_read_only() -> None:
    model = _model()
    cache = _prefill(model)
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    result = natural_token_lookahead_copy_on_write(
        model,
        torch.tensor([[4]]),
        cache,
        horizon=4,
    )
    assert result.anchor_token_ids[0] == 4
    assert len(result.anchor_token_ids) == 4
    assert result.natural_forward_calls == 3
    assert result.production_cache_signature_unchanged
    assert result.production_rng_unchanged
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)


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


def test_sampled_then_expected_and_norm_matched_contents_are_audited() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    logits = torch.full((1, model.config.vocab_size), -10.0)
    logits[0, 3] = 2.0
    logits[0, 4] = 1.0
    result = QwenPseudoEmbeddingProbe(
        model,
        ops,
        _defaults(),
        QwenPseudoVariant(
            "sampled_then_expected",
            "sampled_then_expected_top_m",
            "causal",
            "default_vector_selected_topk_mixture",
            "sampled_token",
        ),
        anchors=(1, 2),
        budget=3,
    ).predict(
        _prefill(model),
        sampled_next_token_id=3,
        current_token_id=2,
        next_token_logits=logits,
        expected_top_m=2,
    )
    assert result.audit["expected_top_m"] == 2
    assert result.audit["expected_embedding_norm"] == "sampled_token"
    assert result.audit["production_cache_signature_unchanged"] is True


def test_provided_anchor_sequence_uses_each_native_token_without_state_mutation() -> None:
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
            "recent_window_independent_zero",
            "provided_sequence",
            "independent",
            "zero",
        ),
        anchors=(1, 2),
        budget=3,
    )
    result = probe.predict(
        cache,
        sampled_next_token_id=4,
        current_token_id=3,
        anchor_token_ids=(4, 5),
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
    assert torch.equal(result.raw_router_logits[0][0], records[0].natural.logits[0])
    assert not torch.equal(result.raw_router_logits[0][0], result.raw_router_logits[0][1])
    assert result.audit["anchor_token_ids_supplied"] is True
    assert result.audit["production_cache_signature_unchanged"] is True
    assert result.audit["production_rng_unchanged"] is True
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_provided_anchor_sequence_rejects_missing_or_misplaced_ids() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    cache = _prefill(model)
    provided = QwenPseudoEmbeddingProbe(
        model,
        ops,
        _defaults(),
        QwenPseudoVariant(
            "recent_window_causal_zero",
            "provided_sequence",
            "causal",
            "zero",
        ),
        anchors=(1, 2),
        budget=3,
    )
    with pytest.raises(ValueError, match="one token ID per pseudo anchor"):
        provided.predict(
            cache,
            sampled_next_token_id=4,
            current_token_id=3,
            anchor_token_ids=(4,),
        )
    sampled = QwenPseudoEmbeddingProbe(
        model,
        ops,
        _defaults(),
        QwenPseudoVariant(
            "sampled_next_independent_zero",
            "sampled_next_token",
            "independent",
            "zero",
        ),
        anchors=(1, 2),
        budget=3,
    )
    with pytest.raises(ValueError, match="only valid for provided_sequence"):
        sampled.predict(
            cache,
            sampled_next_token_id=4,
            current_token_id=3,
            anchor_token_ids=(4, 5),
        )


def test_native_moe_output_capture_is_exact_and_tail_ordered() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    mlp_inputs: dict[int, torch.Tensor] = {}

    def capture_input(_module: object, inputs: tuple[object, ...], *, layer: int) -> None:
        value = inputs[0]
        assert isinstance(value, torch.Tensor)
        mlp_inputs[layer] = value.detach().clone()

    handles = [
        ops.mlp(layer).register_forward_pre_hook(
            lambda module, inputs, index=layer: capture_input(module, inputs, layer=index)
        )
        for layer in range(ops.num_layers)
    ]
    try:
        with MoeOutputCaptureContext(ops, tail_tokens=2) as capture, torch.inference_mode():
            model(input_ids=torch.tensor([[1, 2, 3, 4]]), return_dict=True)
            records = capture.drain()
    finally:
        for handle in handles:
            handle.remove()
    bank = moe_output_bank(records, expected_layers=ops.num_layers)
    router_inputs = moe_router_input_bank(records, expected_layers=ops.num_layers)
    assert bank.shape == (2, 2, 32)
    assert router_inputs.shape == bank.shape
    for layer, record in enumerate(records):
        hidden = mlp_inputs[layer]
        flat = hidden.reshape(-1, ops.hidden_size)
        with torch.inference_mode():
            route = ops.route(layer, flat)
            expected = ops.experts(layer, flat, route).reshape_as(hidden)[:, -2:]
        assert torch.equal(record.value, expected)
        assert torch.equal(bank[:, layer], expected[0])
        assert torch.equal(record.router_input, hidden[:, -2:])
        assert torch.equal(router_inputs[:, layer], hidden[0, -2:])


def test_hard_subset_capture_is_the_policy_executed_moe_output() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    token = torch.tensor([[4]])
    with NativeRouteCaptureContext(ops) as native_capture, torch.inference_mode():
        model(input_ids=token, return_dict=True)
        native = native_capture.drain()
    allowed = {}
    for record in native:
        first_natural = int(record.natural.ids[0, 0])
        allowed[record.layer] = tuple(
            expert for expert in range(ops.num_experts) if expert != first_natural
        )[: ops.top_k]
    mlp_inputs: dict[int, torch.Tensor] = {}

    def capture_input(_module: object, inputs: tuple[object, ...], *, layer: int) -> None:
        value = inputs[0]
        assert isinstance(value, torch.Tensor)
        mlp_inputs[layer] = value.detach().clone()

    handles = [
        ops.mlp(layer).register_forward_pre_hook(
            lambda module, inputs, index=layer: capture_input(module, inputs, layer=index)
        )
        for layer in range(ops.num_layers)
    ]
    try:
        with (
            SubsetExecutionContext(ops, "hard", allowed) as subset,
            MoeOutputCaptureContext(ops) as output_capture,
            torch.inference_mode(),
        ):
            model(input_ids=token, return_dict=True)
            routes = subset.drain()
            outputs = output_capture.drain()
    finally:
        for handle in handles:
            handle.remove()
    assert any(not torch.equal(record.natural.ids, record.executed.ids) for record in routes)
    for layer, (route, output) in enumerate(zip(routes, outputs, strict=True)):
        hidden = mlp_inputs[layer]
        with torch.inference_mode():
            expected = ops.experts(
                layer,
                hidden.reshape(-1, ops.hidden_size),
                route.executed,
            ).reshape_as(hidden)
        assert torch.equal(output.value, expected)
        assert set(route.executed.ids.reshape(-1).tolist()) <= set(allowed[layer])


def test_provided_residual_probe_is_calibration_free_and_changes_later_router() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    cache = _prefill(model)
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    residuals = torch.zeros((2, 2, 32), dtype=next(model.parameters()).dtype)
    residuals[:, 0] = torch.arange(32, dtype=residuals.dtype)[None] / 3
    provided = QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            "sampled_repeat_position_residual",
            "sampled_next_token",
            "independent",
            "provided_residual",
        ),
        anchors=(1, 2),
        budget=3,
    ).predict(
        cache,
        sampled_next_token_id=4,
        current_token_id=3,
        provided_residuals=residuals,
        provided_residual_source="current_policy_previous_window_position_aligned",
    )
    zero = QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            "sampled_repeat_zero",
            "sampled_next_token",
            "independent",
            "zero",
        ),
        anchors=(1, 2),
        budget=3,
    ).predict(cache, sampled_next_token_id=4, current_token_id=3)
    assert torch.equal(provided.raw_router_logits[0], zero.raw_router_logits[0])
    assert not torch.equal(provided.raw_router_logits[1], zero.raw_router_logits[1])
    assert provided.cost.persistent_default_vector_bytes == 0
    assert provided.cost.residual_bank_input_bytes == residuals.numel() * residuals.element_size()
    assert provided.audit["provided_residual_bank"] is True
    assert provided.audit["default_fingerprint"] is None
    assert provided.audit["production_cache_signature_unchanged"] is True
    assert provided.audit["production_rng_unchanged"] is True
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_provided_residual_probe_rejects_wrong_shape_or_missing_source() -> None:
    model = _model()
    ops = Qwen3MoePrefetchOps(model)
    probe = QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            "sampled_repeat_position_residual",
            "sampled_next_token",
            "causal",
            "provided_residual",
        ),
        anchors=(1, 2),
        budget=3,
    )
    with pytest.raises(ValueError, match="anchors, layers, hidden"):
        probe.predict(
            _prefill(model),
            sampled_next_token_id=4,
            current_token_id=3,
            provided_residuals=torch.zeros((1, 2, 32)),
            provided_residual_source="previous_window",
        )
    with pytest.raises(ValueError, match="information-source label"):
        probe.predict(
            _prefill(model),
            sampled_next_token_id=4,
            current_token_id=3,
            provided_residuals=torch.zeros((2, 2, 32)),
        )
