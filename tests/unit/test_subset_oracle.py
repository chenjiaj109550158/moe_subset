from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from pseudoroute.benchmark.prefetch import (
    GptOssPrefetchOps,
    NativeRoute,
    Qwen3MoePrefetchOps,
    SubsetExecutionContext,
    SubsetRouteRecord,
    masked_route,
)
from pseudoroute.benchmark.subset_closed_loop import (
    _rewind_cache,
    _token_agreement,
    subsets_from_route_steps,
)
from pseudoroute.benchmark.subset_config import (
    SubsetOracleSuiteConfig,
    load_subset_oracle_config,
)
from pseudoroute.benchmark.subset_grid import _row_metrics
from pseudoroute.benchmark.subset_report import _paired_row


class TinyQwenGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))
        self.top_k = 2
        self.norm_topk_prob = True

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = nn.functional.linear(hidden, self.weight)
        probabilities = logits.softmax(dim=-1, dtype=torch.float32)
        weights, ids = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(logits.dtype), ids


class TinyQwenExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0, 2.0, 4.0]))
        self.gate_up_proj = nn.Parameter(torch.ones(3, 2, 2))
        self.down_proj = nn.Parameter(torch.ones(3, 2, 2))

    def forward(self, hidden: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        scale = (weights * self.scale[ids]).sum(dim=-1, keepdim=True)
        return hidden * scale


class TinyQwenMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = TinyQwenGate()
        self.experts = TinyQwenExperts()

    def forward(self, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        _, weights, ids = self.gate(flat)
        return cast(Tensor, self.experts(flat, ids, weights)).reshape(shape)


class TinyQwenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = TinyQwenMlp()

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, self.mlp(hidden))


class TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=1,
            num_experts=3,
            num_experts_per_tok=2,
            hidden_size=2,
            norm_topk_prob=True,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyQwenLayer()])

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, cast(Any, self.model).layers[0](hidden))


class TinyGptRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))
        self.bias = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))
        self.top_k = 2

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = nn.functional.linear(hidden, self.weight, self.bias)
        selected, ids = logits.topk(self.top_k, dim=-1)
        return logits, selected.softmax(dim=-1, dtype=selected.dtype), ids


class TinyGptMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = TinyGptRouter()


class TinyGptLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = TinyGptMlp()


class TinyGpt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="gpt_oss",
            num_hidden_layers=1,
            num_local_experts=3,
            num_experts_per_tok=2,
            hidden_size=2,
            quantization_config={"quant_method": "mxfp4"},
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyGptLayer()])


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        self.length = length


@pytest.fixture
def suite() -> SubsetOracleSuiteConfig:
    return load_subset_oracle_config(Path("configs/benchmark/benchmark_subset_oracle_v1.yaml"))


def test_subset_config_grid_and_fingerprint_are_frozen(
    suite: SubsetOracleSuiteConfig,
) -> None:
    assert suite.fingerprint() == "0463596e816da5db528061b50c3b06e81b11bf6d012aa106963ba1c55e712227"
    assert suite.trace.horizons == (1, 2, 4, 8, 16)
    assert suite.models[0].budgets == (8, 16, 32, 64, 128)
    assert suite.models[1].budgets == (4, 8, 16, 24, 32)
    assert suite.closed_loop.aime_gpt_max_new_tokens == 32768
    assert suite.closed_loop.smoke_point_is_not_an_operating_candidate


def test_qwen_lossless_is_exact_and_hard_mask_changes_executed_route() -> None:
    model = TinyQwen().eval()
    ops = Qwen3MoePrefetchOps(model)
    hidden = torch.tensor([[[0.2, 1.0], [1.0, 0.2]]])
    natural = model(hidden)
    with SubsetExecutionContext(ops, "lossless", {0: (0, 2)}) as context:
        lossless = model(hidden)
        records = context.drain()
    assert torch.equal(lossless, natural)
    assert torch.equal(records[0].natural.ids, records[0].executed.ids)
    with SubsetExecutionContext(ops, "hard", {0: (0, 2)}) as context:
        hard = model(hidden)
        hard_records = context.drain()
    assert not torch.equal(hard, natural)
    assert set(hard_records[0].executed.ids.reshape(-1).tolist()) <= {0, 2}
    assert not torch.equal(hard_records[0].natural.ids, hard_records[0].executed.ids)
    assert torch.allclose(hard_records[0].executed.weights.sum(dim=-1), torch.ones(2))


def test_gpt_mask_preserves_biased_logit_topk_softmax_semantics() -> None:
    ops = GptOssPrefetchOps(TinyGpt())
    hidden = torch.tensor([[0.2, 1.0], [1.0, 0.2]])
    route = masked_route(ops, 0, hidden, (0, 2))
    assert set(route.ids.reshape(-1).tolist()) <= {0, 2}
    assert torch.isneginf(route.logits[:, 1]).all()
    selected = route.logits.gather(1, route.ids)
    assert torch.allclose(route.weights, selected.softmax(dim=-1))


def test_future_mass_subset_and_transfer_row_use_frozen_semantics(
    suite: SubsetOracleSuiteConfig,
) -> None:
    ops = Qwen3MoePrefetchOps(TinyQwen())
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    route_a = NativeRoute(logits, torch.tensor([[0.75, 0.25]]), torch.tensor([[0, 1]]))
    route_b = NativeRoute(logits, torch.tensor([[0.10, 0.90]]), torch.tensor([[0, 2]]))
    steps: list[tuple[SubsetRouteRecord, ...]] = [
        (SubsetRouteRecord(0, route_a, route_a, ()),),
        (SubsetRouteRecord(0, route_b, route_b, ()),),
    ]
    assert subsets_from_route_steps(steps, ops, 2) == {0: (0, 2)}
    model = suite.models[0]
    row = _row_metrics(
        model=model,
        task="gsm8k",
        sample_id="test-0",
        start=0,
        horizon=2,
        layer=0,
        budget=8,
        method="future_selected_routing_mass",
        ids=torch.arange(16).reshape(2, 8),
        weights=torch.full((2, 8), 0.125),
        logits=torch.arange(256, dtype=torch.float32).reshape(2, 128),
        subset=tuple(range(8)),
        previous_subset=(),
        expert_bytes=1024,
        margin_threshold=(0.1, 0.2),
        bandwidth_gib_per_second=suite.transfer_model.bandwidth_gib_per_second,
        fixed_latency_microseconds_per_load=(
            suite.transfer_model.fixed_latency_microseconds_per_load
        ),
    )
    assert row["route_hit_rate"] == 0.5
    assert row["fallback_frequency"] == 0.5
    assert row["timing_kind"] == "simulated_not_measured_runtime"


def test_paired_gate_cache_rewind_and_token_divergence() -> None:
    sources = [{"correct": value} for value in (True, True, False, False)]
    actual = [{"correct": value} for value in (True, False, True, False)]
    paired = _paired_row("model", "task", 4, 8, "hard", sources, actual, 1)
    assert paired["paired_accuracy_delta"] == 0
    assert paired["paired_losses"] == 1
    assert paired["paired_gains"] == 1
    assert paired["accuracy_gate_pass"]
    cache = FakeCache(12)
    _rewind_cache(cache, 7)
    assert cache.length == 7
    agreement, first = _token_agreement([1, 2, 3], [1, 4, 3])
    assert agreement == pytest.approx(2 / 3)
    assert first == 1
