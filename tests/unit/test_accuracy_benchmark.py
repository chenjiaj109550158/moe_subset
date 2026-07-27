from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    NativeRoute,
    PrefetchModelOps,
    StreamingExpertMeans,
    load_default_vectors,
    policy_context,
    save_default_vectors,
)
from pseudoroute.benchmark.runner import (
    _materialize_oracle_task,
    _paired_policy_comparisons,
    _sha256_file,
    validate_accuracy_envelope,
)
from pseudoroute.benchmark.scoring import score_response, wilson_interval
from pseudoroute.benchmark.tasks import BenchmarkExample


class FakeMlp(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.expert_scale = nn.Parameter(torch.arange(1, experts + 1, dtype=torch.float32))
        self.top_k = top_k

    def route(self, hidden: Tensor) -> NativeRoute:
        logits = self.gate(hidden)
        weights, ids = logits.softmax(dim=-1).topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return NativeRoute(logits, weights, ids)

    def experts(self, hidden: Tensor, route: NativeRoute) -> Tensor:
        scale = (route.weights * self.expert_scale[route.ids]).sum(dim=-1, keepdim=True)
        return hidden * scale

    def forward(self, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        return self.experts(flat, self.route(flat)).reshape(shape)


class FakeLayer(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = FakeMlp(hidden, experts, top_k)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + cast(Tensor, self.mlp(self.post_attention_layernorm(hidden)))


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([FakeLayer(4, 3, 2), FakeLayer(4, 3, 2)])
        self.head = nn.Linear(4, 16)

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embedding(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return cast(Tensor, self.head(hidden))


class FakeOps(PrefetchModelOps):
    @property
    def layers(self) -> tuple[nn.Module, ...]:
        return tuple(cast(FakeModel, self.model).layers)

    @property
    def num_experts(self) -> int:
        return 3

    @property
    def top_k(self) -> int:
        return 2

    @property
    def hidden_size(self) -> int:
        return 4

    def validate(self) -> None:
        if len(self.layers) != 2:
            raise ValueError

    def mlp(self, layer: int) -> nn.Module:
        return cast(FakeLayer, self.layers[layer]).mlp

    def post_attention_norm(self, layer: int) -> nn.Module:
        return cast(FakeLayer, self.layers[layer]).post_attention_layernorm

    def route(self, layer: int, hidden: Tensor) -> NativeRoute:
        return cast(FakeMlp, self.mlp(layer)).route(hidden)

    def experts(self, layer: int, hidden: Tensor, route: NativeRoute) -> Tensor:
        return cast(FakeMlp, self.mlp(layer)).experts(hidden, route)

    def format_mlp_output(self, value: Tensor, route: NativeRoute) -> object:
        return value

    def experts_and_collect(
        self,
        layer: int,
        hidden: Tensor,
        route: NativeRoute,
        collector: StreamingExpertMeans,
    ) -> Tensor:
        mlp = cast(FakeMlp, self.mlp(layer))
        result = torch.zeros_like(hidden)
        for expert in route.ids.unique().tolist():  # type: ignore[no-untyped-call]
            positions = (route.ids == expert).nonzero()
            token_indices = positions[:, 0]
            topk_positions = positions[:, 1]
            values = hidden[token_indices] * mlp.expert_scale[expert]
            collector.update(layer, expert, values)
            result.index_add_(
                0,
                token_indices,
                values * route.weights[token_indices, topk_positions, None],
            )
        return result


def test_common_prefetch_policy_preserves_natural_and_executes_prediction() -> None:
    torch.manual_seed(4)
    model = FakeModel().eval()
    ops = FakeOps(model)
    tokens = torch.tensor([[1, 2, 3]])
    native = model(tokens)
    with policy_context(ops, "vanilla"):
        explicit = model(tokens)
    assert torch.equal(native, explicit)
    defaults = DefaultVectorArtifact(
        count=torch.ones(2, 3, dtype=torch.int64),
        mean=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 20,
        fingerprint="literal",
    )
    with policy_context(ops, "router_pf", defaults) as context:
        predicted = model(tokens)
    assert torch.isfinite(predicted).all()
    assert context.stats.compared_tokens == 3
    assert context.stats.selected_slots == 6


def test_default_vector_safe_round_trip(tmp_path: Path) -> None:
    collector = StreamingExpertMeans(1, 2, 2)
    collector.update(0, 0, torch.tensor([[1.0, 3.0], [3.0, 5.0]]))
    collector.update(0, 1, torch.tensor([[2.0, 4.0]]))
    artifact = collector.finalize()
    save_default_vectors(tmp_path, artifact, {"source": "literal"})
    loaded = load_default_vectors(tmp_path)
    assert torch.equal(loaded.count, artifact.count)
    assert torch.equal(loaded.mean, artifact.mean)
    assert loaded.fingerprint == artifact.fingerprint


def _example(task: str, target: str) -> BenchmarkExample:
    return BenchmarkExample(task, "0", "prompt", None, target, {}, (), 10)


def test_answer_scorers_use_final_explicit_answers() -> None:
    gsm = score_response(_example("gsm8k", "42"), "work 11. The answer is 42.")
    aime = score_response(_example("aime24", "123"), r"work \boxed{123}")
    strategy = score_response(_example("strategyqa", "yes"), "Maybe no. Final answer: yes")
    harmony = score_response(_example("strategyqa", "yes"), "analysisMaybe no.assistantfinalYes")
    truncated = score_response(_example("strategyqa", "yes"), "analysisMaybe yes")
    assert gsm.correct and gsm.parsed_answer == "42"
    assert aime.correct and aime.parsed_answer == "123"
    assert not score_response(_example("aime24", "123"), "reasoning 5; final 123").correct
    assert strategy.correct and strategy.parsed_answer == "yes"
    assert harmony.correct and harmony.parsed_answer == "yes"
    assert not truncated.correct and truncated.parsed_answer == ""
    lower, upper = wilson_interval(8, 10)
    assert lower < 0.8 < upper


def test_accuracy_suite_protocol_is_frozen_and_complete() -> None:
    suite = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v4.yaml")
    assert suite.policies == ("vanilla", "router_pf", "oracle_pf")
    assert {dataset.expected_samples for dataset in suite.datasets} >= {30, 164, 378, 687, 1319}
    assert suite.fingerprint() == "225a55988d1236086f3cf93e08002424f1292890f5954ec11b28138c6e9457ba"


def test_oracle_materialization_and_paired_comparison(tmp_path: Path) -> None:
    suite = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v4.yaml")
    model = suite.models[0]
    model_root = tmp_path / "models" / model.key
    vanilla_path = model_root / "results" / "vanilla" / "humaneval" / "samples.jsonl"
    vanilla_path.parent.mkdir(parents=True)
    vanilla_path.write_text(
        '{"state":"complete","sample_id":"0","correct":true,'
        '"generated_token_ids":[1,2],"generated_text":"ok"}\n'
    )
    _materialize_oracle_task(model_root, model, suite, "humaneval", 1)
    oracle_path = model_root / "results" / "oracle_pf" / "humaneval" / "samples.jsonl"
    oracle = __import__("json").loads(oracle_path.read_text())
    assert oracle["derived"] is True
    assert oracle["generated_token_ids"] == [1, 2]
    assert oracle["elapsed_seconds_measured"] is None

    paired: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {
        (model.key, "humaneval", "0"): {
            "vanilla": oracle,
            "oracle_pf": oracle,
            "router_pf": {**oracle, "correct": False},
        }
    }
    comparisons = _paired_policy_comparisons(suite, paired)
    comparison = comparisons[0]
    assert comparison["oracle_minus_local_router_pf"] == 1.0
    assert comparison["oracle_only_correct"] == 1
    assert comparison["router_pf_only_correct"] == 0


def test_accuracy_envelope_validation(tmp_path: Path) -> None:
    import json

    artifact = tmp_path / "summary.csv"
    artifact.write_text("accuracy\n1.0\n")
    envelope = {
        "schema_version": 1,
        "state": "complete",
        "artifacts": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        ],
    }
    (tmp_path / "artifact_envelope.json").write_text(json.dumps(envelope))
    assert validate_accuracy_envelope(tmp_path)["state"] == "complete"
    artifact.write_text("changed\n")
    try:
        validate_accuracy_envelope(tmp_path)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("modified artifact must fail checksum validation")


def test_code_sandbox_allows_benign_doctest_import() -> None:
    example = BenchmarkExample(
        task="humaneval",
        sample_id="literal",
        user_prompt="prompt",
        assistant_prefix=None,
        target="assert identity(3) == 3",
        row={"prompt": "def identity(value):\n", "entry_point": "identity"},
        stop_strings=(),
        max_new_tokens=32,
    )
    generated = (
        "    return value\n\n"
        "if __name__ == '__main__':\n"
        "    import doctest\n"
        "    doctest.testmod()\n"
    )
    assert score_response(example, generated).correct


def test_code_extractor_accepts_complete_fenced_rewrite_after_prefix_close() -> None:
    example = BenchmarkExample(
        task="humaneval",
        sample_id="rewrite",
        user_prompt="prompt",
        assistant_prefix="prefix",
        target="assert identity(3) == 3",
        row={
            "prompt": 'def identity(value):\n    """Return the input."""\n',
            "entry_point": "identity",
        },
        stop_strings=(),
        max_new_tokens=32,
    )
    generated = (
        "```\nExplanation before the replacement.\n"
        "```python\ndef identity(value):\n    return value\n```"
    )
    assert score_response(example, generated).correct
