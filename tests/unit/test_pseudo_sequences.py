import pytest
import torch

from pseudoroute.analysis.pseudo_sequences import (
    FactorialCondition,
    OfflineDocument,
    OfflinePseudoSequence,
    build_factorial_sequences,
)
from pseudoroute.types import require_deployable_state


def examples() -> tuple[OfflinePseudoSequence, ...]:
    documents = (
        OfflineDocument("a", torch.tensor([1, 2, 3, 4, 5, 6])),
        OfflineDocument("b", torch.tensor([7, 8, 9, 10, 11, 12])),
    )
    return build_factorial_sequences(
        documents,
        boundary=3,
        horizon=2,
        vocab_size=32,
        content_seed=9,
        position_offset=7,
        include_context_swap=True,
    )


def test_factorial_conditions_change_only_intended_axes() -> None:
    by_condition = {
        example.condition: example
        for example in examples()
        if example.sample_id == "a" and not example.context_swapped
    }
    scsp = by_condition[FactorialCondition.SC_SP]
    dcsp = by_condition[FactorialCondition.DC_SP]
    scdp = by_condition[FactorialCondition.SC_DP]
    dcdp = by_condition[FactorialCondition.DC_DP]
    assert torch.equal(scsp.input_ids[:, 3:], scdp.input_ids[:, 3:])
    assert torch.equal(dcsp.input_ids[:, 3:], dcdp.input_ids[:, 3:])
    assert not torch.equal(scsp.input_ids[:, 3:], dcsp.input_ids[:, 3:])
    assert torch.equal(scsp.position_ids, dcsp.position_ids)
    assert torch.equal(scdp.position_ids, dcdp.position_ids)
    assert not torch.equal(scsp.position_ids[:, 3:], scdp.position_ids[:, 3:])


def test_context_swap_changes_prefix_only_and_documents_never_cross() -> None:
    base = next(
        item
        for item in examples()
        if item.sample_id == "a"
        and item.condition is FactorialCondition.SC_SP
        and not item.context_swapped
    )
    swapped = next(
        item
        for item in examples()
        if item.sample_id == "a"
        and item.condition is FactorialCondition.SC_SP
        and item.context_swapped
    )
    assert not torch.equal(base.input_ids[:, :3], swapped.input_ids[:, :3])
    assert torch.equal(base.input_ids[:, 3:], swapped.input_ids[:, 3:])
    assert torch.equal(base.position_ids, swapped.position_ids)
    with pytest.raises(ValueError, match="document boundary"):
        build_factorial_sequences(
            (OfflineDocument("short", torch.tensor([1, 2, 3])),),
            boundary=2,
            horizon=2,
            vocab_size=8,
            content_seed=1,
            position_offset=1,
            include_context_swap=False,
        )


def test_offline_pseudo_sequence_is_rejected_by_online_boundary() -> None:
    with pytest.raises(TypeError, match="DeployableDecodeState"):
        require_deployable_state(examples()[0])
