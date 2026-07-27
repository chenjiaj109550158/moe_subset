import math

import torch

from pseudoroute.oracle.metrics import (
    expert_union_sizes,
    full_mass_coverage,
    segment_hit_rate,
    selected_mass_coverage,
    union_coverage,
)
from pseudoroute.oracle.windows import OracleSample, OracleWindow, iter_windows


def sample(sample_id: str, offset: int, length: int) -> OracleSample:
    ids = torch.tensor([[[offset % 4, (offset + 1) % 4]] for _ in range(length)], dtype=torch.int64)
    weights = torch.full((length, 1, 2), 0.5)
    logits = torch.zeros(length, 1, 4)
    return OracleSample(sample_id, torch.arange(offset, offset + length), ids, weights, logits)


def test_windows_never_cross_sample_boundaries() -> None:
    samples = (sample("a", 0, 3), sample("b", 100, 4))
    windows = list(iter_windows(samples, (2, 3)))
    assert [(item.sample_id, item.start, item.end) for item in windows] == [
        ("a", 0, 2),
        ("a", 1, 3),
        ("a", 0, 3),
        ("b", 0, 2),
        ("b", 1, 3),
        ("b", 2, 4),
        ("b", 0, 3),
        ("b", 1, 4),
    ]
    assert all(item.end <= (3 if item.sample_id == "a" else 4) for item in windows)


def test_metrics_match_hand_calculation() -> None:
    probabilities = torch.tensor([[[0.4, 0.3, 0.2, 0.1]], [[0.1, 0.2, 0.3, 0.4]]])
    window = OracleWindow(
        "hand",
        0,
        2,
        torch.tensor([[[0, 1]], [[1, 2]]]),
        torch.tensor([[[0.6, 0.4]], [[0.25, 0.75]]]),
        probabilities.log(),
    )
    subset = {0: (1, 2)}
    assert segment_hit_rate(window, subset) == 0.75
    assert math.isclose(selected_mass_coverage(window, subset), 0.7, abs_tol=1e-6)
    assert math.isclose(full_mass_coverage(window, subset), 0.5, abs_tol=1e-7)
    assert expert_union_sizes(window) == {0: 3}
    assert math.isclose(union_coverage(window, subset), 2 / 3)
