from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pseudoroute.runtime.qwen_offload import (
    QwenExpertOffloadEngine,
    deterministic_lru_slot,
    nonzero_expert_ids,
)


class _FakeExperts(nn.Module):
    def __init__(self, *, device: str) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(1234)
        self.gate_up_proj = nn.Parameter(
            torch.randn(4, 4, 3, generator=generator, device=device, dtype=torch.bfloat16)
        )
        self.down_proj = nn.Parameter(
            torch.randn(4, 3, 2, generator=generator, device=device, dtype=torch.bfloat16)
        )
        self.act_fn = F.silu

    def forward(self, hidden: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        result = torch.zeros_like(hidden)
        for expert in ids.unique().tolist():
            positions = (ids == int(expert)).nonzero(as_tuple=False)
            token_indices = positions[:, 0]
            topk_positions = positions[:, 1]
            gate, up = F.linear(hidden[token_indices], self.gate_up_proj[expert]).chunk(2, dim=-1)
            value = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            value = value * weights[token_indices, topk_positions, None]
            result.index_add_(0, token_indices, value.to(result.dtype))
        return result


class _FakeMlp(nn.Module):
    def __init__(self, *, device: str) -> None:
        super().__init__()
        self.gate = nn.Linear(3, 4, bias=False, device=device, dtype=torch.bfloat16)
        self.experts = _FakeExperts(device=device)


class _FakeLayer(nn.Module):
    def __init__(self, *, device: str) -> None:
        super().__init__()
        self.mlp = _FakeMlp(device=device)


class _FakeQwen(nn.Module):
    def __init__(self, *, device: str) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer(device=device)])
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=1,
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=3,
        )


def test_deterministic_lru_prefers_empty_then_oldest_with_expert_tie_break() -> None:
    assert deterministic_lru_slot([2, None, 1], {1: 3, 2: 3}, frozenset()) == 1
    assert deterministic_lru_slot([2, 3, 1], {1: 3, 2: 3, 3: 4}, frozenset()) == 2
    assert deterministic_lru_slot([2, 3, 1], {1: 1, 2: 2, 3: 3}, frozenset({1})) == 0


def test_nonzero_expert_ids_ignores_zero_weight_fallback_ids() -> None:
    ids = torch.tensor([[3, 1, 0], [2, 3, 1]])
    weights = torch.tensor([[0.0, 0.2, 0.0], [0.4, 0.0, 0.1]])
    assert nonzero_expert_ids(ids, weights) == (1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_offload_round_trip_is_exact_and_measures_real_h2d() -> None:
    model = _FakeQwen(device="cuda")
    experts = model.model.layers[0].mlp.experts
    hidden = torch.tensor(
        [[0.5, -0.25, 1.0], [-0.5, 0.75, 0.125]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    ids = torch.tensor([[0, 1], [2, 3]], device="cuda")
    weights = torch.tensor([[0.6, 0.4], [0.55, 0.45]], device="cuda", dtype=torch.bfloat16)
    expected = experts(hidden, ids, weights)

    engine = QwenExpertOffloadEngine(model, slots_per_layer=2)
    engine.reset_cache()
    engine.reset_metrics()
    with engine.phase("decode"):
        actual = experts(hidden, ids, weights)
    metrics = engine.finish_metrics()

    assert torch.equal(expected, actual)
    assert engine.no_full_expert_parameter_on_cuda
    assert engine.pinned_cpu_expert_bytes == engine.full_cpu_expert_bytes
    assert engine.gpu_expert_slot_capacity_bytes * 2 == engine.full_cpu_expert_bytes
    assert metrics.h2d_bytes == engine.full_cpu_expert_bytes
    assert metrics.cache_misses == 4
    assert metrics.phase_metrics["decode"].transfer_batches == 2
    assert engine.resident_experts(0) == frozenset({2, 3})
    assert engine.audit()["all_cpu_sources_pinned"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_resident_only_guard_and_exact_subset_preload() -> None:
    model = _FakeQwen(device="cuda")
    engine = QwenExpertOffloadEngine(model, slots_per_layer=2)
    experts = model.model.layers[0].mlp.experts
    hidden = torch.ones(1, 3, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1]], device="cuda")
    weights = torch.tensor([[0.5, 0.5]], device="cuda", dtype=torch.bfloat16)

    engine.reset_cache()
    with engine.resident_only(), pytest.raises(RuntimeError, match="resident-only"):
        experts(hidden, ids, weights)

    with engine.phase("prefetch"):
        engine.preload_subsets({0: (0, 1)})
    with engine.resident_only(), engine.phase("production"):
        output = experts(hidden, ids, weights)
    assert torch.isfinite(output).all()
    assert engine.resident_experts(0) == frozenset({0, 1})
