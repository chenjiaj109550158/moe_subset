from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pseudoroute.benchmark.pseudo_state_component_swap import (
    CONFIG,
    SPECS,
    classify_attribution,
)

ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _policies(
    *,
    attention: tuple[float, float],
    moe: tuple[float, float],
    joint: tuple[float, float],
    checkpoint: tuple[float, float] = (0.50, 0.50),
) -> dict[str, dict[str, object]]:
    values = {spec.key: {"mean_route_hit": 0.50, "mean_selected_mass": 0.50} for spec in SPECS}
    for spec, pair in zip(SPECS[1:4], (attention, moe, joint), strict=True):
        values[spec.key] = {"mean_route_hit": pair[0], "mean_selected_mass": pair[1]}
    for spec in SPECS[4:]:
        values[spec.key] = {
            "mean_route_hit": checkpoint[0],
            "mean_selected_mass": checkpoint[1],
        }
    return values


@pytest.mark.parametrize(
    ("attention", "moe", "joint", "expected"),
    [
        ((0.51, 0.51), (0.51, 0.51), (0.51, 0.51), "component_path_unresolved"),
        ((0.59, 0.59), (0.52, 0.52), (0.60, 0.60), "attention_dominant"),
        ((0.52, 0.52), (0.59, 0.59), (0.60, 0.60), "moe_residual_dominant"),
        ((0.56, 0.56), (0.56, 0.56), (0.60, 0.60), "coupled_attention_and_residual"),
    ],
)
def test_frozen_component_attribution_rule(
    attention: tuple[float, float],
    moe: tuple[float, float],
    joint: tuple[float, float],
    expected: str,
) -> None:
    result = classify_attribution(
        _policies(attention=attention, moe=moe, joint=joint),
        _config(),
    )
    assert result["classification"] == expected
    assert result["focused_decision"] == (
        "STOP/PIVOT" if expected == "component_path_unresolved" else "NARROW"
    )
    assert result["candidate_execution_requires_separate_frozen_protocol"] is True


def test_latest_passing_hidden_checkpoint_is_selected() -> None:
    policies = _policies(
        attention=(0.56, 0.56),
        moe=(0.56, 0.56),
        joint=(0.60, 0.60),
        checkpoint=(0.56, 0.56),
    )
    result = classify_attribution(policies, _config())
    assert result["passing_checkpoints"] == [7, 15, 23, 31, 39]
    assert result["latest_checkpoint_meeting_recovery"] == 39
