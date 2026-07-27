import pytest
import torch

from pseudoroute.types import DeployableDecodeState, InformationRegime, OfflineFutureTrace


def test_pre_sample_rejects_next_token() -> None:
    with pytest.raises(ValueError, match="cannot contain"):
        DeployableDecodeState(
            information_regime=InformationRegime.ONLINE_PRE_SAMPLE,
            prefix_token_ids=torch.tensor([1, 2]),
            absolute_position=2,
            current_router_logits=(),
            resident_experts=frozenset(),
            next_token_id=3,
        )


def test_offline_trace_is_not_a_deployable_state() -> None:
    offline = OfflineFutureTrace("sample", 0, torch.tensor([1]), torch.zeros(1, 1))
    assert not isinstance(offline, DeployableDecodeState)
