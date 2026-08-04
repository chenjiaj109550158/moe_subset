from __future__ import annotations

import pytest
import torch

from pseudoroute.benchmark.prefetch import NativeRoute, SubsetRouteRecord
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    _bridge_parity,
    _penultimate_joint_trigger,
)
from pseudoroute.benchmark.pseudo_penultimate_joint_accuracy import (
    _planning_row_audit,
    _protocol,
)


def _record(layer: int, *, weight: float = 1.0) -> SubsetRouteRecord:
    route = NativeRoute(
        logits=torch.tensor([[2.0, 1.0]]),
        weights=torch.tensor([[weight]]),
        ids=torch.tensor([[0]]),
    )
    return SubsetRouteRecord(layer=layer, natural=route, executed=route, allowed=(0,))


def test_penultimate_trigger_is_the_last_forward_of_each_h8_window() -> None:
    assert [_penultimate_joint_trigger(step) for step in range(17)] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        _penultimate_joint_trigger(-1)


def test_bridge_parity_requires_exact_shadow_and_real_routes_and_logits() -> None:
    records = (_record(0), _record(1))
    logits = torch.tensor([[3.0, 2.0]])
    assert _bridge_parity(records, records, logits, logits)["pass"] is True

    changed = (_record(0), _record(1, weight=0.5))
    audit = _bridge_parity(records, changed, logits, logits)
    assert audit["pass"] is False
    assert audit["natural_weights_bitwise_equal"] is False


def test_planning_audit_separates_bootstrap_from_penultimate_boundaries() -> None:
    row = {
        "penultimate_joint_boundaries": 1,
        "planning_rows": [
            {"planning_timing": "online_post_sample_bootstrap"},
            {
                "planning_timing": "before_current_window_last_production_forward",
                "content": {
                    "last_sampled_token_available_to_planner": False,
                    "disposable_hard_bridge_executed": True,
                    "production_cache_signature_unchanged": True,
                    "production_rng_unchanged": True,
                    "shadow_cache_discarded": True,
                },
                "probe_audit": {
                    "subset_residual_execution": ("natural_top8_intersection_zero_missing"),
                    "execution_subsets_supplied": True,
                    "executed_ids_within_supplied_subset": True,
                    "full_pre_mask_scores_all_experts": True,
                    "execution_weights_finite_nonnegative": True,
                },
                "bridge_parity": {"pass": True},
            },
        ],
    }
    assert _planning_row_audit(row)
    row["planning_rows"][1]["content"]["last_sampled_token_available_to_planner"] = True
    assert not _planning_row_audit(row)


def test_runtime_protocol_loader_validates_pinned_external_rows() -> None:
    config, samples = _protocol()
    assert config["pilot_id"] == "pseudo_penultimate_joint_qwen_gsm8k_wave1_v1"
    assert len(samples["accuracy"]) == 8
