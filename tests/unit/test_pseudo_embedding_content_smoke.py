from __future__ import annotations

import torch

from pseudoroute.benchmark.pseudo_embedding_content_smoke import (
    RECENT_CAUSAL,
    RECENT_INDEPENDENT,
    SAMPLED,
    TRUE_INDEPENDENT,
    anchor_token_provider,
    blend_subsets,
    recent_anchor_token_ids,
    true_future_anchor_token_ids,
)
from pseudoroute.benchmark.qwen_pseudo import QwenPseudoVariant


def test_recent_anchor_contents_use_no_unrealized_source_token() -> None:
    prompt = tuple(range(100, 110))
    source = tuple(range(20))
    assert recent_anchor_token_ids(0, prompt, source) == (0, 103, 104, 105, 106, 107, 108, 109)
    assert recent_anchor_token_ids(8, prompt, source) == (8, 1, 2, 3, 4, 5, 6, 7)
    assert true_future_anchor_token_ids(8, source) == tuple(range(8, 16))


def test_anchor_provider_separates_deployable_history_from_future_diagnostic() -> None:
    prompt = tuple(range(100, 110))
    source = tuple(range(20))
    sampled = QwenPseudoVariant(SAMPLED, "sampled_next_token", "independent", "zero")
    recent = QwenPseudoVariant(RECENT_INDEPENDENT, "provided_sequence", "independent", "zero")
    recent_causal = QwenPseudoVariant(RECENT_CAUSAL, "provided_sequence", "causal", "zero")
    oracle = QwenPseudoVariant(TRUE_INDEPENDENT, "provided_sequence", "independent", "zero")
    assert anchor_token_provider(sampled, 8, prompt, source) is None
    assert anchor_token_provider(recent, 8, prompt, source) == (8, 1, 2, 3, 4, 5, 6, 7)
    assert anchor_token_provider(recent_causal, 8, prompt, source) == (
        8,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )
    assert anchor_token_provider(oracle, 8, prompt, source) == tuple(range(8, 16))


def test_equal_history_blend_uses_pseudo_only_at_first_boundary() -> None:
    probabilities = torch.zeros((2, 1, 2, 4), dtype=torch.float32)
    probabilities[0, 0, :, 2] = 1
    probabilities[1, 0, 0, 2] = 1
    probabilities[1, 0, 1, 3] = 1
    ids = torch.tensor([[[0]], [[0]], [[1]], [[1]]])
    weights = torch.ones_like(ids, dtype=torch.float32)
    subsets = blend_subsets(
        probabilities,
        ids,
        weights,
        (0, 2),
        horizon=2,
        budget=2,
        experts=4,
    )
    assert tuple(subsets.shape) == (2, 1, 2)
    assert 2 in subsets[0, 0].tolist()
    assert 0 in subsets[1, 0].tolist()
