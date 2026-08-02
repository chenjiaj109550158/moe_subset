from __future__ import annotations

from dataclasses import asdict

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from pseudoroute.benchmark.context_continuation import build_context_continuation
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.qwen_pseudo import QwenPseudoEmbeddingProbe, QwenPseudoVariant
from pseudoroute.benchmark.subset_closed_loop import _cache_mutation_signature


def test_longest_suffix_prefers_length_then_most_recent_without_future_leakage() -> None:
    prompt = (90, 1, 2, 7, 8, 1, 2, 6, 6, 1, 2, 7, 8, 9, 10, 11)
    realized = (1, 2, 99, 100, 101, 102, 103, 104, 777, 778)
    plan = build_context_continuation(
        1,
        prompt,
        realized,
        "longest_suffix_full_continuation",
    )
    changed_future = build_context_continuation(
        1,
        prompt,
        (*realized[:2], 500, 501, 502, 503, 504, 505, 506, 507),
        "longest_suffix_full_continuation",
    )

    assert plan.anchor_token_ids == (2, 7, 8, 9, 10, 11, 1, 2)
    assert plan.matched_suffix_length == 2
    assert plan.match_start == 9
    assert plan.copied_context_indices == (11, 12, 13, 14, 15, 16, 17)
    assert plan.copied_anchor_fraction == 1.0
    assert plan.anchor_token_ids == changed_future.anchor_token_ids
    assert plan.copied_context_indices == changed_future.copied_context_indices
    assert plan.anchor_one_is_sampled_next
    assert plan.earlier_match_nonoverlapping
    assert plan.copied_indices_in_known_context
    assert not plan.future_token_accessed


def test_unigram_uses_most_recent_eligible_occurrence() -> None:
    prompt = (3, 4, 5, 6, 7, 8, 9, 3, 10, 11, 12, 13, 14, 15, 16)
    realized = (3, 20, 21, 22, 23, 24, 25, 26)
    plan = build_context_continuation(
        0,
        prompt,
        realized,
        "sampled_unigram_full_continuation",
    )

    assert plan.match_start == 7
    assert plan.matched_suffix_length == 1
    assert plan.anchor_token_ids == (3, 10, 11, 12, 13, 14, 15, 16)
    assert plan.eligible_match_count_at_selected_length == 2


def test_partial_copy_fills_same_anchor_positions_and_full_mode_falls_back() -> None:
    prompt = (50, 51, 52, 53, 54, 55, 56, 8, 9, 8)
    realized = (9, 70, 71, 72, 73, 74, 75, 76)
    partial = build_context_continuation(
        0,
        prompt,
        realized,
        "longest_suffix_partial_recent_fill",
    )
    full = build_context_continuation(
        0,
        prompt,
        realized,
        "longest_suffix_full_continuation",
    )

    assert partial.matched_suffix_length == 2
    assert partial.copied_anchor_count == 2
    assert partial.anchor_token_ids == (9, 8, 9, 55, 56, 8, 9, 8)
    assert partial.partial_recent_fill_count == 5
    assert not partial.fallback_used
    assert full.anchor_token_ids == (9, 53, 54, 55, 56, 8, 9, 8)
    assert full.fallback_used
    assert full.copied_anchor_count == 0
    comparable = asdict(full)
    comparable.pop("planning_latency_seconds_measured")
    assert comparable["future_token_accessed"] is False


def test_context_continuation_runs_one_native_causal_forward_read_only() -> None:
    torch.manual_seed(53)
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
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
    ).eval()
    prompt = (1, 2, 7, 8, 1, 2, 9)
    realized = (1, 20, 21)
    plan = build_context_continuation(
        0,
        prompt,
        realized,
        "longest_suffix_full_continuation",
        horizon=3,
    )
    with torch.inference_mode():
        prefill = model(
            input_ids=torch.tensor([prompt]),
            use_cache=True,
            return_dict=True,
        )
    assert prefill.past_key_values is not None
    cache = prefill.past_key_values
    signature = _cache_mutation_signature(cache)
    rng = torch.random.get_rng_state().clone()
    ops = Qwen3MoePrefetchOps(model)
    subsets = {layer: (0, 1, 2) for layer in range(ops.num_layers)}
    result = QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            "context_continuation_native_smoke",
            "provided_sequence",
            "causal",
            "native_expert_execution",
        ),
        anchors=(1, 2, 3),
        budget=3,
    ).predict(
        cache,
        sampled_next_token_id=realized[0],
        current_token_id=prompt[-1],
        anchor_token_ids=plan.anchor_token_ids,
        execution_subsets=subsets,
        execution_subset_source="previous_realized_window_subset",
    )

    assert result.cost.attention_calls == 4
    assert result.cost.attention_queries == 12
    assert result.cost.router_calls == 4
    assert result.cost.expert_calls == 4
    assert result.cost.lm_head_calls == 0
    assert result.audit["anchor_token_ids_supplied"] is True
    assert result.audit["one_causal_forward_per_boundary"] is True
    assert result.audit["production_cache_signature_unchanged"] is True
    assert result.audit["production_rng_unchanged"] is True
    assert result.audit["shadow_cache_discarded"] is True
    assert result.audit["executed_ids_within_supplied_subset"] is True
    assert _cache_mutation_signature(cache) == signature
    assert torch.equal(torch.random.get_rng_state(), rng)
