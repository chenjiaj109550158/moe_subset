import torch

from pseudoroute.models.adapters.hf_olmoe import HFOlmoeAdapter
from pseudoroute.models.base import TraceLevel, TraceRequest


def _adapter() -> HFOlmoeAdapter:
    from transformers import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        eos_token_id=1,
    )
    config.architectures = ["OlmoeForCausalLM"]
    torch.manual_seed(7)
    return HFOlmoeAdapter(OlmoeForCausalLM(config).eval(), model_id="literal", revision="0" * 40)


def test_native_olmoe_route_semantics_and_trace_parity() -> None:
    subject = _adapter()
    tokens = torch.tensor([[2, 3, 5, 7]])
    traced = subject.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    with torch.inference_mode():
        native = subject.model(
            input_ids=tokens, output_router_logits=True, use_cache=False, return_dict=True
        )
    logits = native.router_logits[0].reshape(1, tokens.shape[1], -1)
    probabilities = logits.float().softmax(dim=-1)
    weights, ids = probabilities.topk(2, dim=-1)
    first = traced.traces[0]
    assert torch.equal(first.raw_logits, logits[:, 0])
    assert torch.equal(first.pre_topk_scores, probabilities[:, 0])
    assert torch.equal(first.topk_ids, ids[:, 0])
    assert torch.equal(first.topk_weights, weights[:, 0].to(logits.dtype))
    assert torch.equal(traced.logits, native.logits)
    assert subject.spec.num_experts_by_layer == {0: 4, 1: 4}
    handles = subject.iter_moe_layers()
    assert all(handle.shared_expert is None for handle in handles)
    assert all(handle.combine_semantics.endswith("not_renormalized") for handle in handles)


def test_olmoe_structure_validation_rejects_wrong_architecture() -> None:
    subject = _adapter()
    subject.model.config.architectures = ["WrongForCausalLM"]
    try:
        subject.validate_structure()
    except ValueError as error:
        assert "requires OlmoeForCausalLM" in str(error)
    else:
        raise AssertionError("wrong architecture was accepted")
