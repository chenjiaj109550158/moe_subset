import torch

from pseudoroute.config import load_config
from pseudoroute.execution import (
    LosslessFallbackPolicy,
    MaskedSubstitutionPolicy,
    MaskedTruncationPolicy,
    NaturalRoutingPolicy,
    TokenRoutingContext,
)
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.types import ExpertKey


def adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=config.experiment.seed, device="cpu"))


def test_natural_and_lossless_fallback_are_exact() -> None:
    subject = adapter()
    tokens = torch.tensor([[1, 2, 3]])
    base = subject.run_base_forward(tokens)
    natural = subject.forward_with_policy(tokens, NaturalRoutingPolicy())
    allowed = frozenset(ExpertKey(layer, 0) for layer in subject.spec.moe_layer_indices)
    lossless = subject.forward_with_policy(
        tokens, LosslessFallbackPolicy(), allowed_experts=allowed
    )
    assert torch.equal(natural.logits, base.logits)
    assert torch.equal(lossless.logits, base.logits)
    assert any(record.fallback_used for record in lossless.executed_routes)


def test_substitution_never_executes_unavailable_expert_and_preserves_topk() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    natural = torch.tensor([[0, 1]])
    allowed = frozenset({ExpertKey(0, 2), ExpertKey(0, 3)})
    route = MaskedSubstitutionPolicy().choose(0, logits, natural, TokenRoutingContext(5, allowed))
    assert route.executed_topk_ids.tolist() == [[2, 3]]
    assert route.executed_topk_ids.shape[-1] == natural.shape[-1]
    assert set(route.executed_topk_ids.flatten().tolist()) <= {2, 3}
    assert torch.allclose(route.executed_topk_weights.sum(-1), torch.ones(1))


def test_truncation_preserved_and_renormalized_semantics() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    natural = torch.tensor([[0, 1]])
    context = TokenRoutingContext(0, frozenset({ExpertKey(0, 1), ExpertKey(0, 3)}))
    preserved = MaskedTruncationPolicy(False).choose(0, logits, natural, context)
    normalized = MaskedTruncationPolicy(True).choose(0, logits, natural, context)
    assert preserved.executed_topk_ids.tolist() == [[1]]
    assert float(preserved.executed_topk_weights.sum()) < 1.0
    assert normalized.executed_topk_ids.tolist() == [[1]]
    assert torch.equal(
        normalized.executed_topk_weights, torch.ones_like(normalized.executed_topk_weights)
    )
