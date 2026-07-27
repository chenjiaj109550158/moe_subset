import torch

from pseudoroute.analysis.expert_criticality import (
    run_intervention,
    sample_active_targets,
)
from pseudoroute.config import load_config
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE


def adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=config.experiment.seed, device="cpu"))


def test_intervention_records_errors_and_restores_original_model_state() -> None:
    subject = adapter()
    tokens = torch.tensor([[1, 2, 3, 4]])
    target = sample_active_targets(subject, (("sample", tokens),), max_targets=1, seed=2)[0]
    before = {name: value.detach().clone() for name, value in subject.model.state_dict().items()}
    row = run_intervention(
        subject, "sample", tokens, target, intervention="zero_selected_contribution"
    )
    after = subject.model.state_dict()
    assert row.immediate_output_l2 > 0
    assert row.next_token_kl >= 0
    assert row.downstream_topk_flip_count <= row.downstream_positions
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_substitution_intervention_is_sampled_and_finite() -> None:
    subject = adapter()
    tokens = torch.tensor([[1, 2, 3, 4]])
    target = sample_active_targets(subject, (("sample", tokens),), max_targets=1, seed=9)[0]
    row = run_intervention(
        subject, "sample", tokens, target, intervention="substitute_next_available"
    )
    assert torch.isfinite(torch.tensor(row.immediate_output_relative_l2))
    assert torch.isfinite(torch.tensor(row.downstream_router_kl))
