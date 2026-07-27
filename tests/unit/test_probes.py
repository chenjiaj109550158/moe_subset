import json

import pytest
import torch

from pseudoroute.config import load_config
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.probes import CurrentRouteProbe, RollingFrequencyProbe, RollingMassProbe
from pseudoroute.probes.heuristics import MarkovTransitionProbe
from pseudoroute.probes.learned import load_learned_probe, save_learned_probe
from pseudoroute.training.dataset import build_predictor_examples
from pseudoroute.training.train import train_linear
from pseudoroute.types import OfflineFutureTrace


def _examples():
    config = load_config("configs/model/tiny_moe.yaml")
    adapter = TinyMoEAdapter(TinyMoE(config.model, seed=7, device="cpu"))
    documents = tuple(
        (f"doc-{index}", torch.tensor([[index + 1, 2, 3, 4, 5, 6, 7]])) for index in range(6)
    )
    return build_predictor_examples(
        adapter,
        documents,
        horizons=(1, 2),
        history_length=2,
        train_fraction=0.5,
        validation_fraction=1 / 6,
        seed=9,
    )


def test_grouped_split_has_no_document_overlap() -> None:
    dataset, _ = _examples()
    groups = {
        split: {row.sample_id for row in dataset.rows if row.split == split}
        for split in ("train", "validation", "test")
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_online_probes_accept_only_deployable_state() -> None:
    future = OfflineFutureTrace("doc", 0, torch.tensor([1]), torch.zeros(1, 2))
    with pytest.raises(TypeError, match="DeployableDecodeState"):
        CurrentRouteProbe().predict(future, (1,))  # type: ignore[arg-type]


def test_heuristics_are_deterministic_and_well_shaped() -> None:
    dataset, states = _examples()
    expected = (dataset.num_experts,)
    for probe in (CurrentRouteProbe(), RollingFrequencyProbe(2), RollingMassProbe(2)):
        first = probe.predict(states[0], (1, 2))
        second = probe.predict(states[0], (1, 2))
        assert first.aggregate_utility.keys() == second.aggregate_utility.keys()
        for layer, values in first.aggregate_utility.items():
            assert values.shape == expected
            assert torch.isfinite(values).all() and (values >= 0).all()
            assert torch.equal(values, second.aggregate_utility[layer])


def test_safe_linear_serialization_round_trip_and_rejects_pickle(tmp_path) -> None:
    dataset, states = _examples()
    probe, _ = train_linear(dataset, ridge_lambda=0.01)
    root = tmp_path / "linear"
    save_learned_probe(root, probe)
    loaded = load_learned_probe(root)
    assert torch.equal(
        probe.predict_flat(dataset.features[0]), loaded.predict_flat(dataset.features[0])
    )
    output = loaded.predict(states[0], (1, 2))
    assert output.per_horizon_probs is not None
    for probabilities in output.per_horizon_probs.values():
        assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
        assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["safe_format"] = "pickle"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        load_learned_probe(root)


def test_markov_probabilities_are_valid_and_deterministic() -> None:
    _, states = _examples()
    transitions = torch.full((2, 4, 4), 0.25, dtype=torch.float64)
    probe = MarkovTransitionProbe(transitions)
    first = probe.predict(states[0], (2,))
    second = probe.predict(states[0], (2,))
    for layer, utility in first.aggregate_utility.items():
        assert utility.shape == (4,)
        assert torch.allclose(utility.sum(), torch.tensor(2.0, dtype=torch.float64))
        assert torch.equal(utility, second.aggregate_utility[layer])
