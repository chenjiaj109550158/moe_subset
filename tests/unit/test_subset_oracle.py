from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from transformers.cache_utils import DynamicSlidingWindowLayer

from pseudoroute.benchmark.prefetch import (
    GptOssPrefetchOps,
    NativeRoute,
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
    SubsetExecutionContext,
    SubsetRouteRecord,
    masked_route,
)
from pseudoroute.benchmark.runner import _encode_saved_rendered_prompt
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_has_sliding_layers,
    _cache_mutation_signature,
    _fork_cache_copy_on_write,
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
from pseudoroute.benchmark.subset_scope import load_subset_execution_scope
from pseudoroute.benchmark.subset_trace import _ForcedTrajectoryProcessor


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

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits, _, _ = self.router(flat)
        return hidden, logits


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

    def forward(self, hidden: Tensor) -> object:
        return cast(Any, self.model).layers[0].mlp(hidden)


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        self.length = length


class FakeLayeredCache:
    def __init__(self, layer: DynamicSlidingWindowLayer) -> None:
        self.layers = [layer]

    def get_seq_length(self) -> int:
        return int(self.layers[0].get_seq_length())


class FakeSavedPromptTokenizer:
    def __init__(self, decoded: str) -> None:
        self.decoded = decoded

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_tensors: str
    ) -> dict[str, Tensor]:
        assert text == "saved prompt"
        assert not add_special_tokens
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, _ids: Tensor, *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return self.decoded


@pytest.fixture
def suite() -> SubsetOracleSuiteConfig:
    return load_subset_oracle_config(Path("configs/benchmark/benchmark_subset_oracle_v1.yaml"))


def _load_subset_orchestrator() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "complete_subset_oracle_v1.py"
    spec = importlib.util.spec_from_file_location("complete_subset_oracle_v1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subset_config_grid_and_fingerprint_are_frozen(
    suite: SubsetOracleSuiteConfig,
) -> None:
    assert suite.fingerprint() == "b68e18f45d373b31c45d99e8555e25fbf2816d7da8a68946536d6402668767a2"
    assert suite.protocol_revision == 2
    assert suite.trace.replay_chunk_tokens == 1
    assert suite.trace.horizons == (1, 2, 4, 8, 16)
    assert suite.models[0].budgets == (8, 16, 32, 64, 128)
    assert suite.models[1].budgets == (4, 8, 16, 24, 32)
    assert suite.closed_loop.aime_gpt_max_new_tokens == 32768
    assert suite.closed_loop.smoke_point_is_not_an_operating_candidate


def test_hard_only_scope_and_worker_waves_are_frozen(
    suite: SubsetOracleSuiteConfig,
) -> None:
    scope = load_subset_execution_scope(
        Path("configs/benchmark/benchmark_subset_oracle_v1_hard_only_full_v1.yaml")
    )
    assert scope.fingerprint() == (
        "54040188320e7d794c2f37a9b778d0c2bee318d606b178bb177184465bf71626"
    )
    assert scope.base_suite.config_fingerprint == suite.fingerprint()
    assert scope.full_stage.actual_policies == ("hard_oracle_commitment",)
    assert scope.full_stage.expected_actual_rows == 2608
    assert scope.focused_next_stage.model == "qwen3_30b_a3b"
    assert scope.focused_next_stage.task == "gsm8k"
    subset_orchestrator = _load_subset_orchestrator()
    selected = {"gpt_oss_20b": {"horizon": 1, "budget": 4}}
    waves = subset_orchestrator._full_worker_waves(suite, selected, scope)
    assert waves == [
        [("gpt_oss_20b", 0, 0), ("gpt_oss_20b", 1, 1)],
        [("gpt_oss_20b", 2, 0), ("gpt_oss_20b", 3, 1)],
    ]
    assert all(len({job[2] for job in wave}) == len(wave) for wave in waves)


def test_gpt_gsm8k_scope_and_gpu_queues_are_frozen(
    suite: SubsetOracleSuiteConfig,
) -> None:
    scope = load_subset_execution_scope(
        Path("configs/benchmark/benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2.yaml")
    )
    assert scope.fingerprint() == (
        "a5a908ad0124d2041b589a681e7102d3ea5d3f7df90752075799bb9875871d7c"
    )
    assert scope.base_suite.config_fingerprint == suite.fingerprint()
    assert scope.full_stage.tasks == ("gsm8k",)
    assert scope.full_stage.actual_policies == ("hard_oracle_commitment",)
    assert scope.full_stage.expected_actual_rows == 1319
    subset_orchestrator = _load_subset_orchestrator()
    selected = {"gpt_oss_20b": {"horizon": 1, "budget": 4}}
    queues = subset_orchestrator._full_worker_queues(suite, selected, scope)
    assert queues == {
        0: [("gpt_oss_20b", 0, 0), ("gpt_oss_20b", 2, 0)],
        1: [("gpt_oss_20b", 1, 1), ("gpt_oss_20b", 3, 1)],
    }


def test_forced_trajectory_processor_follows_decode_position() -> None:
    processor = _ForcedTrajectoryProcessor(prompt_tokens=3, trajectory=[2, 1])
    scores = torch.zeros((1, 4))
    first = processor(torch.tensor([[7, 8, 9]]), scores)
    second = processor(torch.tensor([[7, 8, 9, 2]]), scores)
    exhausted = processor(torch.tensor([[7, 8, 9, 2, 1]]), scores)
    assert first.argmax(dim=-1).item() == 2
    assert second.argmax(dim=-1).item() == 1
    assert torch.isneginf(first[0, [0, 1, 3]]).all()
    assert torch.isneginf(second[0, [0, 2, 3]]).all()
    assert exhausted is scores


def test_saved_rendered_prompt_requires_exact_tokenizer_roundtrip() -> None:
    config = cast(Any, SimpleNamespace(device="cpu"))
    inputs = _encode_saved_rendered_prompt(
        FakeSavedPromptTokenizer("saved prompt"), config, "saved prompt"
    )
    assert inputs["input_ids"].tolist() == [[3, 4]]
    assert inputs["input_ids"].device.type == "cpu"
    with pytest.raises(RuntimeError, match="exact tokenizer round-trip"):
        _encode_saved_rendered_prompt(
            FakeSavedPromptTokenizer("drifted prompt"), config, "saved prompt"
        )


def test_sliding_cache_fork_is_copy_on_write_and_preserves_original() -> None:
    layer = DynamicSlidingWindowLayer(sliding_window=4)
    key = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1)
    value = key + 10
    layer.update(key, value)
    cache = FakeLayeredCache(layer)
    signature = _cache_mutation_signature(cache)
    original_keys = layer.keys
    fork = cast(FakeLayeredCache, _fork_cache_copy_on_write(cache))
    fork_layer = fork.layers[0]
    assert _cache_has_sliding_layers(cache)
    assert fork_layer is not layer
    assert fork_layer.keys is original_keys
    next_key = torch.tensor([[[[3.0]]]])
    fork_layer.update(next_key, next_key + 10)
    assert layer.keys is original_keys
    assert layer.cumulative_length == 3
    assert _cache_mutation_signature(cache) == signature
    assert fork_layer.cumulative_length == 4


def test_selected_smoke_validation_skips_unselected_models(
    suite: SubsetOracleSuiteConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subset_orchestrator = _load_subset_orchestrator()
    selected_model = suite.models[1]
    monkeypatch.setattr(subset_orchestrator, "OUTPUT", tmp_path)
    monkeypatch.setattr(subset_orchestrator, "load_subset_oracle_config", lambda _: suite)
    for task in suite.trace.sample_rows:
        root = (
            tmp_path
            / "models"
            / selected_model.key
            / "closed_loop"
            / "selected_smoke"
            / task
            / "00000"
        )
        root.mkdir(parents=True)
        (root / "hard_oracle_commitment.json").write_text(
            json.dumps({"executed_route_changed": True}), encoding="utf-8"
        )
        (root / "lossless_oracle_residency.json").write_text(
            json.dumps({"exact_token_agreement": 1.0, "executed_route_changed": False}),
            encoding="utf-8",
        )
    subset_orchestrator._validate_smoke(
        "selected_smoke", require_lossless=True, model_keys={selected_model.key}
    )


def test_qwen_lossless_is_exact_and_hard_mask_changes_executed_route() -> None:
    model = TinyQwen().eval()
    ops = Qwen3MoePrefetchOps(model)
    hidden = torch.tensor([[[0.2, 1.0], [1.0, 0.2]]])
    natural = model(hidden)
    with SubsetExecutionContext(ops, "lossless", {0: (0, 2)}) as context:
        lossless = model(hidden)
        records = context.drain()
    with NativeRouteCaptureContext(ops, {0: (0, 2)}) as native_context:
        native_captured = model(hidden)
        native_records = native_context.drain()
    assert torch.equal(native_captured, natural)
    assert torch.equal(native_records[0].natural.ids, native_records[0].executed.ids)
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

    with NativeRouteCaptureContext(ops) as context:
        ops.model(hidden[None])
        records = context.drain()
    native = records[0].natural
    expected = ops.route(0, hidden)
    assert torch.equal(native.logits, expected.logits)
    assert torch.equal(native.ids, expected.ids)
    assert torch.equal(native.weights, expected.weights)


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
