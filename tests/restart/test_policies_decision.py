from pathlib import Path

import pytest
import torch
import yaml

from pseudoroute.restart.decision import evaluate_decision, quality_interval
from pseudoroute.restart.policies import (
    WindowConfig,
    anchor_ids,
    cost_aware_subset,
    legacy_subset,
    top_ids,
)
from pseudoroute.restart.state import load_row, save_row


@pytest.fixture
def protocol():
    return yaml.safe_load(Path("configs/restart/restart_v1.yaml").read_text())


def complete():
    return dict(
        actual_model_runs_completed=True,
        correctness_pass=True,
        validity_pass=True,
        planned_quality_n=256,
        completed_paired_quality_n=256,
        required_pairs_complete=True,
        accuracy_delta_ci=[-0.01, 0.02],
        decode_speedup=1.2,
        decode_speedup_ci=[1.15, 1.25],
        timing_valid=True,
        p95_latency_ratio=1.1,
        dynamic_value_confirmed=True,
    )


def test_all_ties_do_not_prove_quality():
    ci = quality_interval(0, 0, 8)
    assert ci[0] < -0.02 and ci[1] > 0.02
    assert quality_interval(0, 0, 0) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"decode_speedup_ci": [0.99, 1.3]},
        {"decode_speedup": None, "decode_speedup_ci": None, "bytes_reduction": 0.95},
        {"completed_paired_quality_n": 255},
        {"required_pairs_complete": False},
        {"planned_quality_n": 2, "completed_paired_quality_n": 2},
    ],
)
def test_insufficient_evidence_cannot_go(protocol, changes):
    assert not evaluate_decision(complete() | changes, protocol)["overall"].startswith("GO")


def test_gpu_blocker_is_not_method_failure(protocol):
    result = evaluate_decision({"external_blockers": ["no CUDA"]}, protocol)
    assert result["overall"] == "BLOCKED" and result["window_subset"] == "BLOCKED"


def test_correctness_failure_is_runtime_problem(protocol):
    assert (
        evaluate_decision(complete() | {"correctness_pass": False}, protocol)["overall"]
        == "PIVOT_RUNTIME"
    )


def test_static_dominance(protocol):
    assert (
        evaluate_decision(complete() | {"static_dominance_confirmed": True}, protocol)["overall"]
        == "STATIC_SUFFICIENT_TESTED_SCOPE"
    )


def test_history_can_survive_pseudo_failure(protocol):
    result = evaluate_decision(complete() | {"pseudo_dominated": True}, protocol)
    assert result["overall"] == "GO_WINDOW_HISTORY_ONLY"
    assert result["pseudo_predictor"] == "STOP_CURRENT_VARIANT"


def test_resume_rejects_hash_change(tmp_path):
    p = tmp_path / "row.json"
    save_row(p, {"identity": "frozen", "state": "COMPLETE"})
    assert load_row(p, "frozen")
    with pytest.raises(ValueError):
        load_row(p, "changed")


def test_tail_anchors_do_not_affect_legacy_selection():
    ids = torch.arange(64).reshape(8, 8)
    p = torch.ones(8, 128)
    h = torch.ones(128)
    h[100] = 1e12
    expected = tuple(range(32))
    assert legacy_subset(p, ids, h, 32) == expected
    p[4:] = 1e9
    ids[4:] = 100
    assert legacy_subset(p, ids, h, 32) == expected
    assert legacy_subset(p[:4], ids[:4], h, 32) == expected


def test_g4_content_is_prefix_of_legacy_g8_bootstrap_and_later():
    prompt = (1, 4, 9, 8, 7, 6, 5, 4, 3, 2)
    for known in [(4,), (4, 2, 3, 4, 9, 8, 7, 6, 5)]:
        assert (
            anchor_ids(prompt, known, WindowConfig(pseudo_compute_tokens=4))
            == anchor_ids(prompt, known, WindowConfig(pseudo_compute_tokens=8))[:4]
        )
    assert anchor_ids((), (), WindowConfig(pseudo_compute_tokens=0)) == ()
    with pytest.raises(ValueError):
        WindowConfig(pseudo_compute_tokens=3)


def test_cost_aware_ties_no_epsilon_and_zero_fallback():
    assert top_ids(torch.tensor([1.0, 1.0, 1.0001]), 2) == (0, 2)
    h = torch.zeros(8)
    assert cost_aware_subset(h, (6, 7), torch.ones(8), budget=2) == (6, 7)


def test_runtime_validation_survives_incomplete_method_comparison(protocol):
    result = evaluate_decision(
        {
            "correctness_pass": True,
            "runtime_measurement_valid": True,
            "actual_model_runs_completed": True,
            "validity_pass": False,
        },
        protocol,
    )
    assert result["runtime"] == "VALIDATED"
    assert result["overall"] == "INCONCLUSIVE"
    assert result["window_subset"] == "INCONCLUSIVE"
