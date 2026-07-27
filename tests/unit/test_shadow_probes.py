from pathlib import Path

import pytest
import torch

from pseudoroute.config import load_config
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.probes.default_vectors import (
    DefaultVectorShadowRolloutProbe,
    StreamingDefaultVectorCollector,
    calibrate_default_vectors,
    load_default_vectors,
    same_token_next_layer_router_error,
    save_default_vectors,
)
from pseudoroute.probes.pseudo import (
    FuturePositionRephasedProbe,
    PseudoTokenProbe,
    UncertaintyEnsembleProbe,
)
from pseudoroute.probes.shadow_cache import build_read_only_cache
from pseudoroute.types import DeployableDecodeState, InformationRegime


def _adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=17, device="cpu"))


def _state(
    regime: InformationRegime = InformationRegime.ONLINE_PRE_SAMPLE,
) -> DeployableDecodeState:
    next_token = 4 if regime is InformationRegime.ONLINE_POST_SAMPLE else None
    return DeployableDecodeState(
        regime,
        torch.tensor([[1, 2, 3]]),
        2,
        (torch.zeros(4), torch.zeros(4)),
        frozenset(),
        next_token,
    )


def _documents() -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        ("a", torch.tensor([[1, 2, 3, 4, 5]])),
        ("b", torch.tensor([[6, 7, 8, 9, 10]])),
    )


def test_shadow_cache_is_immutable_and_future_positions_are_exact() -> None:
    adapter = _adapter()
    cache = build_read_only_cache(adapter, _state().prefix_token_ids)
    before = cache.fingerprint
    output = FuturePositionRephasedProbe(adapter).predict(_state(), (1, 3))
    cache.assert_unchanged()
    assert cache.fingerprint == before
    assert output.metadata["positions"] == [3, 5]
    no_rephase = FuturePositionRephasedProbe(adapter, "no_rephase").predict(_state(), (1, 3))
    wrong = FuturePositionRephasedProbe(adapter, "wrong_position", 2).predict(_state(), (1, 3))
    assert not torch.equal(output.aggregate_utility[0], no_rephase.aggregate_utility[0])
    assert not torch.equal(output.aggregate_utility[0], wrong.aggregate_utility[0])


def test_pre_and_post_sample_content_boundary() -> None:
    adapter = _adapter()
    probe = PseudoTokenProbe(adapter, "independent", "sampled_next_token")
    with pytest.raises(ValueError, match="post_sample"):
        probe.predict(_state(), (1,))
    output = probe.predict(_state(InformationRegime.ONLINE_POST_SAMPLE), (1, 2))
    assert output.metadata["information_regime"] == "online_post_sample"
    with pytest.raises(ValueError, match="cannot contain the next token"):
        DeployableDecodeState(
            InformationRegime.ONLINE_PRE_SAMPLE,
            torch.tensor([[1]]),
            0,
            (torch.zeros(4),),
            frozenset(),
            2,
        )


def test_probe_rng_and_generation_are_unchanged() -> None:
    adapter = _adapter()
    prompt = torch.tensor([[1, 2, 3]])
    expected_generation = adapter.model.generate(prompt, max_new_tokens=3)
    before = torch.random.get_rng_state().clone()
    FuturePositionRephasedProbe(adapter, random_content=True, seed=99).predict(_state(), (1, 2))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(expected_generation, adapter.model.generate(prompt, max_new_tokens=3))


def test_pseudo_modes_return_valid_probabilities() -> None:
    adapter = _adapter()
    for mode in ("independent", "causal"):
        output = PseudoTokenProbe(adapter, mode, "current_token").predict(_state(), (1, 2))
        assert output.per_horizon_probs is not None
        for probabilities in output.per_horizon_probs.values():
            assert probabilities.shape == (2, 4)
            assert torch.allclose(probabilities.sum(-1), torch.ones(2))


def test_uncertainty_ensemble_reports_branch_spread() -> None:
    adapter = _adapter()
    probe = UncertaintyEnsembleProbe(
        (
            FuturePositionRephasedProbe(adapter, "correct"),
            FuturePositionRephasedProbe(adapter, "wrong_position", 2),
        )
    )
    output = probe.predict(_state(), (1, 2))
    assert output.uncertainty is not None
    assert all((value >= 0).all() for value in output.uncertainty.values())


def test_streaming_defaults_match_brute_force_and_round_trip(tmp_path: Path) -> None:
    adapter = _adapter()
    store = calibrate_default_vectors(adapter, _documents())
    values: list[list[list[torch.Tensor]]] = [[[] for _ in range(4)] for _ in range(2)]
    with torch.inference_mode():
        for _, tokens in _documents():
            output = adapter.model(tokens, capture_activations=True)
            for activation in output.activations:
                assert activation.router_input is not None
                for expert in activation.topk_ids.reshape(-1).tolist():
                    values[activation.layer_idx][expert].append(
                        adapter._layer(activation.layer_idx)
                        .experts[expert](  # noqa: SLF001
                            activation.router_input
                        )
                        .reshape(-1)
                        .double()
                    )
    for layer in range(2):
        for expert in range(4):
            assert int(store.count[layer, expert]) == len(values[layer][expert])
            if values[layer][expert]:
                assert torch.allclose(
                    store.mean[layer, expert], torch.stack(values[layer][expert]).mean(0)
                )
    save_default_vectors(tmp_path, store)
    loaded = load_default_vectors(tmp_path)
    assert torch.equal(loaded.count, store.count)
    assert torch.equal(loaded.mean, store.mean)
    errors = same_token_next_layer_router_error(adapter, _documents()[0][1], loaded)
    assert len(errors) == 1 and errors[0] >= 0


def test_default_rollout_has_uncertainty_and_valid_routes() -> None:
    adapter = _adapter()
    store = calibrate_default_vectors(adapter, _documents())
    output = DefaultVectorShadowRolloutProbe(adapter, store).predict(_state(), (1, 2, 4))
    assert output.uncertainty is not None
    assert output.metadata["anchors"] == [1, 2, 4]
    for layer, probabilities in output.per_horizon_probs.items():  # type: ignore[union-attr]
        assert probabilities.shape == (3, 4)
        assert output.uncertainty[layer].shape == (3,)


def test_streaming_collector_matches_scalar_welford() -> None:
    collector = StreamingDefaultVectorCollector(1, 1, 2)
    collector.update(0, 0, torch.tensor([1.0, 3.0]))
    collector.update(0, 0, torch.tensor([3.0, 7.0]))
    store = collector.finalize()
    assert torch.equal(store.mean[0, 0], torch.tensor([2.0, 5.0], dtype=torch.float64))
    assert torch.equal(store.variance[0, 0], torch.tensor([2.0, 8.0], dtype=torch.float64))
