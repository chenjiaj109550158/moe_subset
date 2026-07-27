import pytest
import torch

from pseudoroute.config import load_config, resolve_device
from pseudoroute.models.tiny_moe import TinyMoE


def build_model() -> TinyMoE:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoE(config.model, seed=config.experiment.seed).to(
        resolve_device(config.model.device)
    )


def tokens(values: list[list[int]]) -> torch.Tensor:
    config = load_config("configs/model/tiny_moe.yaml")
    return torch.tensor(values, device=resolve_device(config.model.device))


def test_forward_and_generation_are_deterministic() -> None:
    input_ids = tokens([[1, 2, 3]])
    first, second = build_model(), build_model()
    assert torch.equal(first(input_ids).logits, second(input_ids).logits)
    assert torch.equal(
        first.generate(input_ids, max_new_tokens=4),
        second.generate(input_ids, max_new_tokens=4),
    )


def test_router_topk_is_exact_and_weights_are_normalized() -> None:
    output = build_model()(tokens([[1, 2, 3]]))
    assert output.traces
    for trace in output.traces:
        expected = trace.pre_topk_scores.topk(2, dim=-1).indices
        assert torch.equal(trace.topk_ids, expected)
        assert torch.allclose(
            trace.topk_weights.sum(dim=-1), torch.ones(1, device=trace.topk_weights.device)
        )


def test_trace_capture_does_not_change_logits() -> None:
    model = build_model()
    input_ids = tokens([[1, 2, 3]])
    assert torch.equal(
        model(input_ids, capture_trace=True).logits,
        model(input_ids, capture_trace=False).logits,
    )


def test_expert_ids_are_layer_scoped() -> None:
    traces = build_model()(tokens([[1]])).traces
    same_numeric_id = [trace.expert_keys[0] for trace in traces]
    assert len({key.layer_idx for key in same_numeric_id}) == 2
    assert same_numeric_id[0] != same_numeric_id[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_smoke() -> None:
    config = load_config("configs/model/tiny_moe.yaml")
    model = TinyMoE(config.model, seed=config.experiment.seed, device="cuda")
    output = model(torch.tensor([[1, 2, 3]], device="cuda"))
    assert output.logits.device.type == "cuda"


def test_model_spec_is_typed_and_layer_scoped() -> None:
    spec = build_model().spec
    assert spec.model_id == "tiny_moe"
    assert spec.moe_layer_indices == (0, 1)
    assert spec.num_experts_by_layer == {0: 4, 1: 4}
    assert spec.top_k_by_layer == {0: 2, 1: 2}
    assert len(spec.expert_bytes) == 8
    assert {key.layer_idx for key in spec.expert_bytes} == {0, 1}


def test_seed_1234_trace_matches_literal_regression_values() -> None:
    trace = build_model()(tokens([[1, 2, 3]])).traces[0]
    expected_logits = torch.tensor(
        [0.02891925, 0.26115829, -0.17370431, -0.81302875],
        device=trace.raw_logits.device,
    )
    expected_scores = torch.tensor(
        [0.28499147, 0.35949427, 0.23271990, 0.12279437],
        device=trace.pre_topk_scores.device,
    )
    expected_weights = torch.tensor([0.55780023, 0.44219983], device=trace.topk_weights.device)
    assert trace.token_position == 0
    assert trace.layer_idx == 0
    assert trace.topk_ids.tolist() == [[1, 0]]
    assert torch.allclose(trace.raw_logits.flatten(), expected_logits, atol=1e-6, rtol=0)
    assert torch.allclose(trace.pre_topk_scores.flatten(), expected_scores, atol=1e-6, rtol=0)
    assert torch.allclose(trace.topk_weights.flatten(), expected_weights, atol=1e-6, rtol=0)
