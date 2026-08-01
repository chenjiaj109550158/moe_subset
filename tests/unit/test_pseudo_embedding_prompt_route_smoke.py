from __future__ import annotations

import torch

from pseudoroute.benchmark.pseudo_embedding_prompt_route_smoke import (
    CORE_HISTORY,
    EQUAL_RECENT,
    EQUAL_SAMPLED,
    KNOWN_HISTORY,
    PROMPT_PRIOR,
    candidate_subsets,
    native_weight_normalization_audit,
)


def test_prompt_route_candidates_are_deterministic_and_calibration_free() -> None:
    sampled = torch.zeros((8, 8))
    recent = torch.zeros((8, 8))
    sampled[:, 4] = 1
    recent[:, 5] = 1
    history_ids = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 2]])
    history_weights = torch.full_like(history_ids, 0.5, dtype=torch.float32)
    result = candidate_subsets(
        sampled,
        recent,
        history_ids,
        history_weights,
        budget=4,
        top_k=2,
        experts=8,
    )
    assert set(result) == {
        PROMPT_PRIOR,
        KNOWN_HISTORY,
        EQUAL_SAMPLED,
        CORE_HISTORY,
        EQUAL_RECENT,
    }
    assert all(len(value) == 4 and value == tuple(sorted(value)) for value in result.values())
    assert 0 in result[PROMPT_PRIOR]
    assert 4 in result[KNOWN_HISTORY]
    assert 5 in result[EQUAL_RECENT]


def test_prompt_route_tie_break_uses_ascending_expert_id() -> None:
    probabilities = torch.full((8, 8), 1 / 8)
    history_ids = torch.tensor([[7, 6], [5, 4], [3, 2], [1, 0]])
    history_weights = torch.ones_like(history_ids, dtype=torch.float32)
    result = candidate_subsets(
        probabilities,
        probabilities,
        history_ids,
        history_weights,
        budget=4,
        top_k=2,
        experts=8,
    )
    assert result[PROMPT_PRIOR] == (0, 1, 2, 3)


def test_native_weight_normalization_uses_bfloat16_precision_bound() -> None:
    compatible = torch.tensor([[0.5009765625, 0.5009765625]])
    error, tolerance = native_weight_normalization_audit(compatible, torch.bfloat16)
    assert error == 0.001953125
    assert tolerance == torch.finfo(torch.bfloat16).eps
    incompatible = torch.tensor([[0.51, 0.51]])
    error, tolerance = native_weight_normalization_audit(incompatible, torch.bfloat16)
    assert error > tolerance
