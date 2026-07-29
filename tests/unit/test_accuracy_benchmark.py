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
    _can_reuse_imported_score,
    _generate,
    _generation_compatibility_payload,
    _materialize_oracle_task,
    _merge_task_shards,
    _model_example,
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
    gsm_final = score_response(
        _example("gsm8k", "45"),
        "work 180. Final Answer: John is 45 miles away after 4 hours.",
    )
    gsm_decimal = score_response(_example("gsm8k", "26"), "Final Answer: **$26.00**")
    gsm_boxed = score_response(
        _example("gsm8k", "6"),
        r"work 12 and 4. Final Answer: \boxed{6}. Remaining 6 at 4 mph.",
    )
    gsm_verbose = score_response(
        _example("gsm8k", "230"),
        (
            "assistantfinalDay 1: 80 miles. Day 2: 150 miles. "
            "So answer: Each train covers **230 miles**."
        ),
    )
    gsm_emphasized = score_response(
        _example("gsm8k", "70"),
        (
            "assistantfinalTotal time is 30 + 40 = 70 minutes. "
            "So Carmen spent **70 minutes** (1 hour and 10 minutes)."
        ),
    )
    gsm_latex_comma = score_response(
        _example("gsm8k", "8000"),
        r"assistantfinalThus \boxed{8{,}000}.",
    )
    gsm_labeled_sentence = score_response(
        _example("gsm8k", "3"),
        "assistantfinal**Answer:** A 42 kg bag lasts **3 weeks**.",
    )
    gsm_word_answer = score_response(
        _example("gsm8k", "2"),
        (
            "assistantfinal**Step 3: Determine truckloads** "
            "4500 / 2250 = 2 trips. The farmer needs **two trips**."
        ),
    )
    gsm_converted_units = score_response(
        _example("gsm8k", "216"),
        (
            "assistantfinal**Final Answer:** It takes **216 seconds**, "
            "or **3 minutes and 36 seconds**."
        ),
    )
    gsm_answer_before_explanation = score_response(
        _example("gsm8k", "25000"),
        (
            "assistantfinal**Answer: $25,000 per year**\n\n**Explanation** "
            "After 30 years, 50% of $50,000 is **$25,000**."
        ),
    )
    gsm_answer_before_steps = score_response(
        _example("gsm8k", "48"),
        (
            "assistantfinal**Answer: 48 french fries**\n\n"
            "**Step-by-step reasoning** Step 1 starts with one example."
        ),
    )
    gsm_wrong_conclusion = score_response(
        _example("gsm8k", "6"),
        (
            "assistantfinalThey read **3 distinct books**. "
            "Counting reading events would instead give 6."
        ),
    )
    aime = score_response(_example("aime24", "123"), r"work \boxed{123}")
    strategy = score_response(_example("strategyqa", "yes"), "Maybe no. Final answer: yes")
    harmony = score_response(_example("strategyqa", "yes"), "analysisMaybe no.assistantfinalYes")
    truncated = score_response(_example("strategyqa", "yes"), "analysisMaybe yes")
    assert gsm.correct and gsm.parsed_answer == "42"
    assert gsm_final.correct and gsm_final.parsed_answer == "45"
    assert gsm_decimal.correct and gsm_decimal.parsed_answer == "26.00"
    assert gsm_boxed.correct and gsm_boxed.parsed_answer == "6"
    assert gsm_verbose.correct and gsm_verbose.parsed_answer == "230"
    assert gsm_emphasized.correct and gsm_emphasized.parsed_answer == "70"
    assert gsm_latex_comma.correct and gsm_latex_comma.parsed_answer == "8000"
    assert gsm_labeled_sentence.correct and gsm_labeled_sentence.parsed_answer == "3"
    assert gsm_word_answer.correct and gsm_word_answer.parsed_answer == "2"
    assert gsm_converted_units.correct and gsm_converted_units.parsed_answer == "216"
    assert gsm_answer_before_explanation.correct
    assert gsm_answer_before_explanation.parsed_answer == "25000"
    assert gsm_answer_before_steps.correct
    assert gsm_answer_before_steps.parsed_answer == "48"
    assert not gsm_wrong_conclusion.correct and gsm_wrong_conclusion.parsed_answer == "3"
    assert aime.correct and aime.parsed_answer == "123"
    assert not score_response(_example("aime24", "123"), "reasoning 5; final 123").correct
    assert strategy.correct and strategy.parsed_answer == "yes"
    assert harmony.correct and harmony.parsed_answer == "yes"
    assert not truncated.correct and truncated.parsed_answer == ""
    lower, upper = wilson_interval(8, 10)
    assert lower < 0.8 < upper


def test_accuracy_suite_protocol_is_frozen_and_complete() -> None:
    suite = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v5.yaml")
    assert suite.policies == ("vanilla", "router_pf", "oracle_pf")
    assert {dataset.expected_samples for dataset in suite.datasets} >= {30, 164, 378, 687, 1319}
    assert suite.fingerprint() == "6533c5cd17b77427ddbe280b921aab4b3ef68a104b402ad1761c440fda226e8f"


def test_accuracy_v6_raises_gpt_reasoning_budgets_only() -> None:
    v5 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v5.yaml")
    v6 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v6.yaml")
    assert v6.protocol_revision == 6
    assert v6.models[0] == v5.models[0]
    assert v6.models[1].max_new_tokens_overrides == {
        "gsm8k": 4096,
        "strategyqa": 4096,
    }


def test_accuracy_v7_changes_only_scorer_protocol_identity() -> None:
    v6 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v6.yaml")
    v7 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v7.yaml")
    assert v7.protocol_revision == 7
    assert v7.models == v6.models
    assert v7.datasets == v6.datasets
    assert v7.decode == v6.decode


def test_accuracy_v8_selects_medium_gpt_and_finite_aime_budget() -> None:
    v7 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v7.yaml")
    v8 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v8.yaml")
    assert v8.protocol_revision == v7.protocol_revision == 7
    assert v8.models[0] == v7.models[0]
    assert v8.models[1].reasoning_effort == "medium"
    assert v8.models[1].max_new_tokens_overrides == {
        "gsm8k": 4096,
        "strategyqa": 4096,
        "aime24": 4096,
        "aime25": 4096,
    }


def test_accuracy_v9_preserves_gpt_reasoning_channel_for_code_only() -> None:
    v8 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v8.yaml")
    v9 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v9.yaml")
    assert v9.protocol_revision == 8
    assert v9.models[0].code_assistant_prefill
    assert not v9.models[1].code_assistant_prefill
    assert _generation_compatibility_payload(
        v8.model_dump(mode="json"), "gpt_oss_20b", "gsm8k"
    ) == _generation_compatibility_payload(v9.model_dump(mode="json"), "gpt_oss_20b", "gsm8k")
    assert _generation_compatibility_payload(
        v8.model_dump(mode="json"), "gpt_oss_20b", "humaneval"
    ) != _generation_compatibility_payload(v9.model_dump(mode="json"), "gpt_oss_20b", "humaneval")
    assert _can_reuse_imported_score(7, 8, "gsm8k")
    assert v9.fingerprint() == "8cf30430b17b2e82a23adb573872d2fb440c0bf31a22eb2d2e5d5264f6974d1d"


def test_accuracy_v10_raises_only_gpt_code_caps() -> None:
    v9 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v9.yaml")
    v10 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v10.yaml")
    assert v10.protocol_revision == 9
    assert v10.models[0] == v9.models[0]
    assert v10.models[1].max_new_tokens_overrides == {
        **v9.models[1].max_new_tokens_overrides,
        "humaneval": 4096,
        "mbpp_plus": 4096,
    }
    assert _generation_compatibility_payload(
        v9.model_dump(mode="json"), "gpt_oss_20b", "gsm8k"
    ) == _generation_compatibility_payload(v10.model_dump(mode="json"), "gpt_oss_20b", "gsm8k")
    assert _generation_compatibility_payload(
        v9.model_dump(mode="json"), "gpt_oss_20b", "humaneval"
    ) != _generation_compatibility_payload(v10.model_dump(mode="json"), "gpt_oss_20b", "humaneval")
    assert _can_reuse_imported_score(8, 9, "gsm8k")
    assert v10.fingerprint() == "0d4fbb451a55f8fab39dabb8ab55c4e798f24f1489f8c94a10d717aced612fdc"


def test_accuracy_v11_is_a_scorer_only_revision() -> None:
    v10 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v10.yaml")
    v11 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v11.yaml")
    assert v11.protocol_revision == 10
    assert v11.models == v10.models
    assert v11.datasets == v10.datasets
    assert v11.decode == v10.decode
    for model in v11.models:
        for dataset in v11.datasets:
            assert _generation_compatibility_payload(
                v10.model_dump(mode="json"), model.key, dataset.key
            ) == _generation_compatibility_payload(
                v11.model_dump(mode="json"), model.key, dataset.key
            )
    assert not _can_reuse_imported_score(9, 10, "humaneval")
    assert not _can_reuse_imported_score(9, 10, "mbpp_plus")
    assert _can_reuse_imported_score(9, 10, "gsm8k")
    assert v11.fingerprint() == "4c2b60dbb4cc6cc2aa1ebd6b3f8bad936b6ca6ee423166229fc12aed4cc5576d"


def test_accuracy_v12_is_a_scorer_only_revision() -> None:
    v11 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v11.yaml")
    v12 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v12.yaml")
    assert v12.protocol_revision == 11
    assert v12.models == v11.models
    assert v12.datasets == v11.datasets
    assert v12.decode == v11.decode
    for model in v12.models:
        for dataset in v12.datasets:
            assert _generation_compatibility_payload(
                v11.model_dump(mode="json"), model.key, dataset.key
            ) == _generation_compatibility_payload(
                v12.model_dump(mode="json"), model.key, dataset.key
            )
    assert not _can_reuse_imported_score(10, 11, "humaneval")
    assert not _can_reuse_imported_score(10, 11, "mbpp_plus")
    assert _can_reuse_imported_score(10, 11, "gsm8k")
    assert not _can_reuse_imported_score(9, 11, "humaneval")
    assert _can_reuse_imported_score(9, 11, "gsm8k")
    assert v12.fingerprint() == "beaf89e1a035fc093358433ce3a01758f03919f291e3af00542805e552dc29fc"


def test_accuracy_v13_is_a_gsm_scorer_only_revision() -> None:
    v12 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v12.yaml")
    v13 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v13.yaml")
    assert v13.protocol_revision == 12
    assert v13.models == v12.models
    assert v13.datasets == v12.datasets
    assert v13.decode == v12.decode
    for model in v13.models:
        for dataset in v13.datasets:
            assert _generation_compatibility_payload(
                v12.model_dump(mode="json"), model.key, dataset.key
            ) == _generation_compatibility_payload(
                v13.model_dump(mode="json"), model.key, dataset.key
            )
    assert not _can_reuse_imported_score(11, 12, "gsm8k")
    assert _can_reuse_imported_score(11, 12, "humaneval")
    assert not _can_reuse_imported_score(9, 12, "gsm8k")
    assert not _can_reuse_imported_score(9, 12, "humaneval")
    assert _can_reuse_imported_score(9, 12, "strategyqa")
    assert v13.fingerprint() == "1a11f813076b50c4ef8b47d635da578f5ab2237c68f9964ceb08e67bbbb6ade1"


def test_accuracy_v14_is_an_mbpp_scorer_only_revision() -> None:
    v13 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v13.yaml")
    v14 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v14.yaml")
    assert v14.protocol_revision == 13
    assert v14.models == v13.models
    assert v14.datasets == v13.datasets
    assert v14.decode == v13.decode
    for model in v14.models:
        for dataset in v14.datasets:
            assert _generation_compatibility_payload(
                v13.model_dump(mode="json"), model.key, dataset.key
            ) == _generation_compatibility_payload(
                v14.model_dump(mode="json"), model.key, dataset.key
            )
    assert not _can_reuse_imported_score(12, 13, "mbpp_plus")
    assert _can_reuse_imported_score(12, 13, "humaneval")
    assert _can_reuse_imported_score(12, 13, "gsm8k")
    assert not _can_reuse_imported_score(11, 13, "gsm8k")
    assert not _can_reuse_imported_score(9, 13, "humaneval")
    assert _can_reuse_imported_score(9, 13, "strategyqa")
    assert v14.fingerprint() == "e85cf403a078ad416ded894fda93ada30c498e28bd734a752c12c8879f191fe2"


def test_accuracy_v15_only_extends_gpt_humaneval_cap() -> None:
    v14 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v14.yaml")
    v15 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v15.yaml")
    assert v15.protocol_revision == 14
    assert v15.models[0] == v14.models[0]
    assert v15.models[1].max_new_tokens_overrides == {
        **v14.models[1].max_new_tokens_overrides,
        "humaneval": 16384,
    }
    assert v15.datasets == v14.datasets
    assert v15.decode == v14.decode
    assert _generation_compatibility_payload(
        v14.model_dump(mode="json"), "gpt_oss_20b", "gsm8k"
    ) == _generation_compatibility_payload(v15.model_dump(mode="json"), "gpt_oss_20b", "gsm8k")
    assert _generation_compatibility_payload(
        v14.model_dump(mode="json"), "gpt_oss_20b", "humaneval"
    ) != _generation_compatibility_payload(v15.model_dump(mode="json"), "gpt_oss_20b", "humaneval")
    assert _can_reuse_imported_score(13, 14, "gsm8k")
    assert _can_reuse_imported_score(13, 14, "humaneval")
    assert v15.fingerprint() == "dac2637dcce7c6c9cbbff44204a011969c66017b552194090206254457cb1c01"


def test_accuracy_v16_adopts_checkpoint_sampling_and_schedules_models() -> None:
    v15 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v15.yaml")
    v16 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v16.yaml")
    assert v16.protocol_revision == 15
    for before, after in zip(v15.models, v16.models, strict=True):
        assert after.model_copy(update={"device": before.device}) == before
    assert [(model.key, model.device) for model in v16.models] == [
        ("qwen3_30b_a3b", "cuda:1"),
        ("gpt_oss_20b", "cuda:0"),
    ]
    assert v16.datasets == v15.datasets
    assert v16.decode.model_copy(update={"do_sample": False}) == v15.decode
    assert v16.decode.do_sample
    for model in v16.models:
        for dataset in v16.datasets:
            assert _generation_compatibility_payload(
                v15.model_dump(mode="json"), model.key, dataset.key
            ) != _generation_compatibility_payload(
                v16.model_dump(mode="json"), model.key, dataset.key
            )
    smoke = load_accuracy_suite_config(
        "configs/benchmark/speculating_experts_accuracy_v16_gpu1_smoke.yaml"
    )
    assert _generation_compatibility_payload(
        v16.model_dump(mode="json"), "qwen3_30b_a3b", "aime24"
    ) == _generation_compatibility_payload(
        smoke.model_dump(mode="json"), "qwen3_30b_a3b", "aime24"
    )
    assert v16.fingerprint() == "46f764f29c0ff97b9300c3d5f80dc0d60c37f9bef0678d3914d3c2c8b67a3698"


def test_accuracy_v17_uses_greedy_unless_complete_v15_fails_gate() -> None:
    v15 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v15.yaml")
    v16 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v16.yaml")
    v17 = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v17.yaml")
    sampled = {
        ("qwen3_30b_a3b", "aime24"),
        ("qwen3_30b_a3b", "aime25"),
        ("gpt_oss_20b", "humaneval"),
        ("gpt_oss_20b", "aime24"),
        ("gpt_oss_20b", "aime25"),
    }

    assert v17.protocol_revision == 16
    assert not v17.decode.do_sample
    assert v17.datasets == v16.datasets
    for before, after in zip(v16.models, v17.models, strict=True):
        assert after.model_copy(update={"do_sample_overrides": {}}) == before
    for model in v17.models:
        for dataset in v17.datasets:
            key = (model.key, dataset.key)
            assert v17.do_sample_for(model, dataset.key) == (key in sampled)
            source = v16 if key in sampled else v15
            assert _generation_compatibility_payload(
                v17.model_dump(mode="json"), model.key, dataset.key
            ) == _generation_compatibility_payload(
                source.model_dump(mode="json"), model.key, dataset.key
            )
    assert v17.fingerprint() == "1d40653aaa2d7c70fd4eb88e87b93b787663197bb3f6817bc58ced225991fbf9"


def test_generate_forwards_protocol_sampling_without_overriding_checkpoint_params() -> None:
    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs: dict[str, Any] = {}

        def generate(self, **kwargs: Any) -> Tensor:
            self.kwargs = kwargs
            return torch.tensor([[10, 11, 12]])

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        @staticmethod
        def decode(tokens: Tensor, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return "answer"

    suite = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v16.yaml")
    model = RecordingModel()
    example = BenchmarkExample(
        task="aime24",
        sample_id="sample",
        user_prompt="Question: sample\nAnswer:",
        assistant_prefix=None,
        target="12",
        row={},
        stop_strings=(),
        max_new_tokens=32,
    )
    ids, text = _generate(
        model,
        Tokenizer(),
        {"input_ids": torch.tensor([[10, 11]])},
        example,
        suite.models[0],
        do_sample=suite.decode.do_sample,
    )
    assert ids == [12] and text == "answer"
    assert model.kwargs["do_sample"] is True
    assert not {"temperature", "top_p", "top_k"} & model.kwargs.keys()


def test_v7_import_score_reuse_matrix_is_explicit() -> None:
    assert _can_reuse_imported_score(5, 7, "humaneval")
    assert _can_reuse_imported_score(6, 7, "strategyqa")
    assert not _can_reuse_imported_score(5, 7, "gsm8k")
    assert not _can_reuse_imported_score(4, 7, "humaneval")
    assert _can_reuse_imported_score(7, 7, "gsm8k")


def test_generation_compatibility_is_model_and_task_scoped() -> None:
    v5 = load_accuracy_suite_config(
        "configs/benchmark/speculating_experts_accuracy_v5.yaml"
    ).model_dump(mode="json")
    v7 = load_accuracy_suite_config(
        "configs/benchmark/speculating_experts_accuracy_v7.yaml"
    ).model_dump(mode="json")
    assert _generation_compatibility_payload(
        v5, "qwen3_30b_a3b", "gsm8k"
    ) == _generation_compatibility_payload(v7, "qwen3_30b_a3b", "gsm8k")
    assert _generation_compatibility_payload(
        v5, "gpt_oss_20b", "humaneval"
    ) == _generation_compatibility_payload(v7, "gpt_oss_20b", "humaneval")
    assert _generation_compatibility_payload(
        v5, "gpt_oss_20b", "aime24"
    ) == _generation_compatibility_payload(v7, "gpt_oss_20b", "aime24")
    assert _generation_compatibility_payload(
        v5, "gpt_oss_20b", "gsm8k"
    ) != _generation_compatibility_payload(v7, "gpt_oss_20b", "gsm8k")
    assert _generation_compatibility_payload(
        v5, "gpt_oss_20b", "strategyqa"
    ) != _generation_compatibility_payload(v7, "gpt_oss_20b", "strategyqa")


def test_task_shards_merge_in_original_row_order(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    shards = task_root / "shards"
    shards.mkdir(parents=True)
    payloads = {
        0: [
            {"state": "complete", "row_index": 0, "sample_id": "a"},
            {"state": "complete", "row_index": 2, "sample_id": "c"},
        ],
        1: [
            {"state": "complete", "row_index": 1, "sample_id": "b"},
            {"state": "complete", "row_index": 3, "sample_id": "d"},
        ],
    }
    json_module = __import__("json")
    for shard_index, rows in payloads.items():
        stem = f"{shard_index:05d}-of-00002"
        (shards / f"{stem}.jsonl").write_text(
            "".join(json_module.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        (shards / f"{stem}.DONE").write_text("complete\n", encoding="utf-8")
    assert _merge_task_shards(task_root, 2, 4)
    merged = [
        json_module.loads(line)
        for line in (task_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["row_index"] for row in merged] == [0, 1, 2, 3]
    assert (task_root / "DONE").read_text(encoding="utf-8") == "complete\n"


def test_oracle_materialization_and_paired_comparison(tmp_path: Path) -> None:
    suite = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v5.yaml")
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


def test_gpt_code_without_assistant_prefill_scores_standalone_fence() -> None:
    suite = load_accuracy_suite_config("configs/benchmark/speculating_experts_accuracy_v9.yaml")
    example = BenchmarkExample(
        task="humaneval",
        sample_id="standalone",
        user_prompt="prompt",
        assistant_prefix="prefilled",
        target="assert identity(3) == 3",
        row={
            "prompt": 'def identity(value):\n    """Return the input."""\n',
            "entry_point": "identity",
        },
        stop_strings=(),
        max_new_tokens=32,
    )
    rendered_example = _model_example(example, suite.models[1])
    assert rendered_example.assistant_prefix is None
    assert _model_example(example, suite.models[0]).assistant_prefix == "prefilled"
    generated = (
        "analysisUse a direct return.assistantfinal"
        "Here is the implementation:\n"
        "```python\ndef identity(value):\n    return value\n```"
    )
    assert score_response(rendered_example, generated).correct


def test_gpt_mbpp_prefers_relevant_python_fence_after_math_fences() -> None:
    example = BenchmarkExample(
        task="mbpp_plus",
        sample_id="standalone-multiple-fences",
        user_prompt="prompt",
        assistant_prefix=None,
        target="assert square(3) == 9",
        row={"code": "def square(value):\n    return value * value\n"},
        stop_strings=(),
        max_new_tokens=32,
    )
    generated = (
        "analysisDerive the formula.assistantfinal"
        "The recurrence is:\n```\nP[k] = P[k-1] + 1\n```\n"
        "```python\ndef square(value):\n    return value * value\n```"
    )
    assert score_response(example, generated).correct


def test_standalone_fenced_program_is_a_separate_source_unit() -> None:
    example = BenchmarkExample(
        task="humaneval",
        sample_id="standalone-program",
        user_prompt="prompt",
        assistant_prefix=None,
        target="assert identity(3) == 3",
        row={
            "prompt": "def identity(value):\n",
            "entry_point": "identity",
        },
        stop_strings=(),
        max_new_tokens=32,
    )
    generated = (
        "analysisUse a direct return.assistantfinal"
        "```python\n"
        "from __future__ import annotations\n\n"
        "def identity(value):\n    return value\n\n"
        'if __name__ == "__main__":\n'
        '    raise AssertionError("candidate-only test block ran")\n'
        "```"
    )
    assert score_response(example, generated).correct


def test_standalone_humaneval_program_retains_prompt_helpers() -> None:
    example = BenchmarkExample(
        task="humaneval",
        sample_id="prompt-helper",
        user_prompt="prompt",
        assistant_prefix=None,
        target="assert decode_shift(encode_shift('abc')) == 'abc'",
        row={
            "prompt": (
                "def encode_shift(value):\n"
                "    return ''.join(chr((ord(char) - 92) % 26 + 97) for char in value)\n\n"
                "def decode_shift(value):\n"
            ),
            "entry_point": "decode_shift",
        },
        stop_strings=(),
        max_new_tokens=32,
    )
    generated = (
        "analysisInvert the shift.assistantfinal"
        "```python\n"
        "def decode_shift(value):\n"
        "    return ''.join(chr((ord(char) - 102) % 26 + 97) for char in value)\n"
        "```"
    )
    assert score_response(example, generated).correct


def test_code_extractor_prefers_nonempty_prefix_continuation() -> None:
    example = BenchmarkExample(
        task="mbpp_plus",
        sample_id="continuation",
        user_prompt="prompt",
        assistant_prefix="prefix",
        target="assert square(3) == 9",
        row={},
        stop_strings=(),
        max_new_tokens=32,
    )
    generated = (
        "def square(value):\n    return value * value\n```\n"
        "Explanation with a later fenced example.\n```python\nnot code\n```"
    )
    assert score_response(example, generated).correct
