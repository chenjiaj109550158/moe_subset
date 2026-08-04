from __future__ import annotations

from types import SimpleNamespace

import torch

from pseudoroute.benchmark.qwen_penultimate_joint import (
    commit_bridge_only_cache,
    first_four_history_subset,
    fork_bridge_only_cache,
)


class _FakeDynamicLayer:
    def __init__(self, length: int) -> None:
        self.keys = torch.arange(length * 2, dtype=torch.float32).reshape(1, 1, length, 2)
        self.values = self.keys + 100
        self.sliding_window = None

    def get_seq_length(self) -> int:
        return int(self.keys.shape[-2])


class _FakeDynamicCache:
    def __init__(self, layers: int, length: int) -> None:
        self.layers = [_FakeDynamicLayer(length) for _ in range(layers)]

    def get_seq_length(self) -> int:
        return self.layers[0].get_seq_length()


def test_first_four_history_subset_has_ascending_layer_scoped_ties() -> None:
    probabilities = torch.zeros(8, 10)
    ids = torch.tensor([[0, 1]] * 8)
    history = torch.zeros(10)

    selected = first_four_history_subset(
        probabilities,
        ids,
        history,
        budget=8,
    )

    assert selected == tuple(range(8))


def test_bridge_only_cache_returns_joint_kv_but_commits_one_position() -> None:
    production = _FakeDynamicCache(layers=2, length=3)
    production_layer_ids = tuple(id(layer) for layer in production.layers)
    original_keys = tuple(layer.keys.clone() for layer in production.layers)
    state = fork_bridge_only_cache(production, joint_tokens=9)

    for layer_index, layer in enumerate(state.cache.layers):
        new_keys = torch.full((1, 1, 9, 2), 10.0 + layer_index)
        new_values = new_keys + 1000
        full_keys, full_values = layer.update(new_keys, new_values)
        assert full_keys.shape[-2] == 12
        assert full_values.shape[-2] == 12
        assert layer.keys.shape[-2] == 4
        assert layer.values.shape[-2] == 4
        assert torch.equal(layer.keys[..., :3, :], original_keys[layer_index])
        assert torch.equal(layer.keys[..., 3:, :], new_keys[..., :1, :])

    assert production.get_seq_length() == 3
    commit_bridge_only_cache(production, state)

    assert production.get_seq_length() == 4
    assert tuple(id(layer) for layer in production.layers) == production_layer_ids
    assert state.layer_update_counts == [1, 1]
    assert state.layer_query_lengths == [9, 9]
    assert state.committed
    assert all(layer.keys.shape[-2] == 4 for layer in production.layers)


def test_bridge_only_cache_rejects_sliding_layers() -> None:
    production = _FakeDynamicCache(layers=1, length=3)
    production.layers[0].sliding_window = 128

    try:
        fork_bridge_only_cache(production, joint_tokens=9)
    except ValueError as error:
        assert "sliding" in str(error)
    else:
        raise AssertionError("sliding cache unexpectedly accepted")


def test_fake_cache_shape_matches_dynamic_cache_contract() -> None:
    cache = _FakeDynamicCache(layers=1, length=3)
    assert isinstance(SimpleNamespace(layers=cache.layers).layers, list)
    assert cache.get_seq_length() == 3
