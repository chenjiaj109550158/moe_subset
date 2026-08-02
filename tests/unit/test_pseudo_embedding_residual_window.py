from __future__ import annotations

import torch
from safetensors.torch import save_file

from pseudoroute.benchmark import pseudo_embedding_residual_window as residual_window
from pseudoroute.benchmark.prefetch import NativeRoute, SubsetRouteRecord
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    BUDGET,
    HORIZON,
    _aggregate_policy,
    _candidate_specs,
    _paired_bootstrap,
    _ranked_selection,
    _residual_specs,
    _stratified_outputs,
    candidate_subsets,
    residual_bank_variant,
    route_scores,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenProbeCost,
    QwenPseudoProbeResult,
)


def _probe_result() -> QwenPseudoProbeResult:
    experts = 40
    probabilities = torch.zeros(HORIZON, experts)
    for anchor in range(HORIZON):
        probabilities[anchor, anchor : anchor + 8] = torch.arange(8, 0, -1)
        probabilities[anchor] /= probabilities[anchor].sum()
    topk = probabilities.topk(8, dim=-1)
    logits = probabilities.clamp_min(1e-8).log()
    return QwenPseudoProbeResult(
        "synthetic",
        tuple(range(1, 9)),
        tuple(range(8)),
        {0: logits},
        {0: probabilities},
        {0: topk.indices},
        {0: topk.values},
        {0: probabilities.double().sum(dim=0)},
        {0: tuple(range(BUDGET))},
        QwenProbeCost(0.1, 10, 20, 0, 30, 8, 8, 8, 0),
        {},
    )


def test_four_residual_bank_variants_are_exact_and_calibration_free() -> None:
    bank = torch.arange(HORIZON * 2 * 3, dtype=torch.float32).reshape(HORIZON, 2, 3)
    zero = residual_bank_variant(bank, "zero")
    aligned = residual_bank_variant(bank, "previous_window_position_aligned")
    last = residual_bank_variant(bank, "previous_window_last_repeated")
    mean = residual_bank_variant(bank, "previous_window_mean_repeated")
    assert torch.count_nonzero(zero) == 0
    assert aligned.data_ptr() == bank.data_ptr()
    assert torch.equal(last[0], bank[-1]) and torch.equal(last[-1], bank[-1])
    assert torch.equal(mean[0], bank.mean(dim=0))
    assert torch.equal(mean[0], mean[-1])


def test_pre_mask_route_history_scatter_uses_natural_not_executed() -> None:
    logits = torch.zeros(2, 4)
    natural = NativeRoute(
        logits,
        torch.tensor([[0.75, 0.25], [0.4, 0.6]]),
        torch.tensor([[0, 1], [1, 2]]),
    )
    executed = NativeRoute(
        logits,
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([[2, 3], [2, 3]]),
    )
    scores = route_scores((SubsetRouteRecord(0, natural, executed, (2, 3)),), experts=4)
    assert torch.allclose(scores[0], torch.tensor([0.75, 0.65, 0.6, 0.0], dtype=torch.float64))


def test_all_frozen_candidate_formulas_return_deterministic_top32() -> None:
    result = _probe_result()
    history = {0: torch.arange(40, dtype=torch.float64)}
    methods = [
        spec.selection
        for spec in _candidate_specs(
            "previous_window_position_aligned", "sampled_repeat_independent"
        )
        if spec.role == "pseudo"
    ]
    assert None not in methods
    for method in methods:
        subsets = candidate_subsets(result, history, str(method))
        assert len(subsets[0]) == BUDGET
        assert len(set(subsets[0])) == BUDGET
        assert subsets[0] == tuple(sorted(subsets[0]))
    history_only = candidate_subsets(result, history, "previous_window_route_only")[0]
    assert history_only == tuple(range(8, 40))
    sampled_core = candidate_subsets(result, history, "sampled_top8_core_plus_history_fill")[0]
    assert set(range(8)) <= set(sampled_core)


def test_frozen_policy_sets_have_four_residuals_and_eight_candidates() -> None:
    assert len(_residual_specs()) == 4
    specs = _candidate_specs("previous_window_position_aligned", "sampled_repeat_independent")
    assert sum(spec.role == "pseudo" for spec in specs) == 8
    assert {spec.role for spec in specs if spec.role != "pseudo"} == {
        "oracle",
        "previous",
        "static",
    }


def _sample_row(policy: str, hit: int, mass: float, latency: float) -> dict[str, object]:
    return {
        "sample_id": "test-0",
        "boundaries": [0],
        "metrics": [
            {
                "route_hits": hit,
                "route_slots": 10,
                "selected_mass_hit": mass,
                "selected_mass_total": 10.0,
                "simulated_lossless_transfer_bytes": 40,
                "natural_reference_bytes": 100,
            }
        ],
        "probe_costs": [
            {
                "latency_seconds_measured": latency,
                "temporary_cuda_bytes_measured": 100,
                "attention_queries": 384,
                "router_calls": 384,
            }
        ],
        "policy": policy,
        "elapsed_seconds_measured": latency + 1,
    }


def test_aggregate_ranking_and_paired_bootstrap_are_deterministic() -> None:
    a = _aggregate_policy([_sample_row("a", 7, 7.0, 0.2)])
    b = _aggregate_policy([_sample_row("b", 7, 8.0, 0.3)])
    assert a["mean_route_hit"] == 0.7
    assert a["estimated_transfer_reduction"] == 0.6
    assert _ranked_selection({"a": a, "b": b}, {"a", "b"}) == "b"
    candidate = {"test-0": (0.8, 0.9), "test-1": (0.6, 0.7)}
    previous = {"test-0": (0.7, 0.8), "test-1": (0.5, 0.6)}
    first = _paired_bootstrap(candidate, previous, draws=100, seed=17)
    second = _paired_bootstrap(candidate, previous, draws=100, seed=17)
    assert first == second


def test_development_ranking_uses_transfer_before_latency() -> None:
    slow_high_transfer = _aggregate_policy([_sample_row("a", 7, 7.0, 0.9)])
    fast_low_transfer = _aggregate_policy([_sample_row("b", 7, 7.0, 0.1)])
    slow_high_transfer["estimated_transfer_reduction"] = 0.5
    fast_low_transfer["estimated_transfer_reduction"] = 0.4
    values = {"a": slow_high_transfer, "b": fast_low_transfer}
    assert _ranked_selection(values, {"a", "b"}) == "b"
    assert _ranked_selection(values, {"a", "b"}, include_transfer=True) == "a"


def test_strata_reconstruct_anchor_margin_and_residual_buckets(
    tmp_path: object,
    monkeypatch: object,
) -> None:
    # pytest fixtures are left untyped here to keep this tensor-schema test compact.
    root = tmp_path  # type: ignore[assignment]
    monkeypatch.setattr(residual_window, "OUTPUT", root)  # type: ignore[attr-defined]
    json_path, tensor_path = residual_window._sample_paths("residual_smoke", 0, "residual__zero")
    json_path.parent.mkdir(parents=True)
    ids = torch.arange(8).reshape(1, 1, 8)
    weights = torch.full((1, 1, 8), 0.125)
    save_file(
        {
            "natural_router_topk_ids": ids,
            "natural_router_topk_weights": weights,
            "natural_router_logits": torch.arange(10, dtype=torch.float32).reshape(1, 1, 10),
            "subsets": torch.arange(32).reshape(1, 1, 32),
            "boundaries": torch.tensor([0]),
            "planning_residual_norms": torch.ones(1, 8, 1),
            "planning_residual_router_input_cosine": torch.zeros(1, 8, 1),
        },
        str(tensor_path),
    )
    row = {
        "stage": "residual_smoke",
        "row_index": 0,
        "policy": "residual__zero",
        "sample_id": "test-0",
    }
    strata, worst = _stratified_outputs([row])
    assert strata["anchor"]["residual__zero"]["1"]["mean_route_hit"] == 1.0  # type: ignore[index]
    assert (
        strata["router_margin_tertile"]["residual__zero"]["low"][  # type: ignore[index]
            "token_layer_observations"
        ]
        == 1
    )
    assert len(worst) == 1
