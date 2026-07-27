import json

import torch

from pseudoroute.config import load_config
from pseudoroute.execution.closed_loop import evaluate_closed_loop, plan_oracle_window
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE


def adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=config.experiment.seed, device="cpu"))


def test_plan_budget_and_transfer_bytes_are_exact() -> None:
    subject = adapter()
    plan = plan_oracle_window(
        subject,
        torch.tensor([[1, 2, 3]]),
        horizon=2,
        budget_ratio=1.0,
        gamma=0.9,
        resident=frozenset(),
    )
    assert all(len(experts) == 2 for experts in plan.subset_by_layer.values())
    assert len(plan.load_delta) == 4
    assert plan.transfer_bytes == 4 * 4096


def test_lossless_closed_loop_matches_base_and_replans_from_generated_prefix() -> None:
    summary, routes, windows = evaluate_closed_loop(
        adapter(),
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=4,
        horizon=2,
        budget_ratio=1.0,
        gamma=0.9,
        policy_name="lossless_fallback",
    )
    assert summary.exact_sequence_match
    assert summary.first_divergence is None
    assert summary.mean_next_token_kl == 0.0
    assert routes and windows
    generated = json.loads(summary.generated_tokens)
    second_prefix = json.loads(windows[1].planning_prefix)
    assert second_prefix == [1, 2, 3, *generated[:2]]
    assert windows[1].boundary == len(second_prefix)


def test_diverged_run_replans_from_its_own_authoritative_trajectory() -> None:
    summary, _, windows = evaluate_closed_loop(
        adapter(),
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=12,
        horizon=4,
        budget_ratio=1.0,
        gamma=0.9,
        policy_name="masked_substitution",
    )
    assert summary.first_divergence is not None
    generated = json.loads(summary.generated_tokens)
    base = json.loads(summary.base_tokens)
    third_prefix = json.loads(windows[2].planning_prefix)
    assert third_prefix == [1, 2, 3, *generated[:8]]
    assert third_prefix != [1, 2, 3, *base[:8]]
