import pytest
import torch

from pseudoroute.config import load_config
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.runtime.offload_engine import ExpertState, TinyMoEOffloadEngine
from pseudoroute.types import ExpertKey

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for M10")


def _model() -> TinyMoE:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoE(config.model, seed=1234, device="cuda")


def test_lossless_synchronous_offload_matches_base_and_obeys_budget() -> None:
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    base = _model()
    expected = base(tokens, capture_trace=False).logits
    engine = TinyMoEOffloadEngine(
        _model(), slots_per_layer=2, pinned_memory=True, asynchronous=False
    )
    engine.reset_metrics()
    actual = engine.forward(tokens)
    metrics = engine.finish_metrics()
    assert torch.equal(expected, actual)
    assert metrics.h2d_bytes > 0
    assert metrics.resident_expert_bytes < sum(
        handle.byte_size for handle in engine.handles.values()
    )
    assert all(handle.up_weight.device.type == "cpu" for handle in engine.handles.values())
    assert metrics.pinned_cpu_bytes == sum(handle.byte_size for handle in engine.handles.values())


def test_asynchronous_matches_synchronous_with_slot_replacement_dependencies() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
    synchronous = TinyMoEOffloadEngine(
        _model(), slots_per_layer=1, pinned_memory=True, asynchronous=False
    )
    expected = synchronous.forward(tokens)
    asynchronous = TinyMoEOffloadEngine(
        _model(), slots_per_layer=1, pinned_memory=True, asynchronous=True
    )
    asynchronous.reset_metrics()
    actual = asynchronous.forward(tokens)
    metrics = asynchronous.finish_metrics()
    assert torch.equal(expected, actual)
    assert len(metrics.transfer_records) > 2
    assert all(record.asynchronous for record in metrics.transfer_records)
    assert all(slot.compute_complete is not None for slots in asynchronous.slots for slot in slots)
    assert all(
        handle.state in {ExpertState.CPU, ExpertState.RESIDENT}
        for handle in asynchronous.handles.values()
    )


def test_subset_plan_preloads_fixed_slots_and_hard_commit_stays_in_subset() -> None:
    engine = TinyMoEOffloadEngine(
        _model(), slots_per_layer=2, pinned_memory=True, asynchronous=True
    )
    subset = frozenset(
        {
            ExpertKey(0, 0),
            ExpertKey(0, 1),
            ExpertKey(1, 0),
            ExpertKey(1, 1),
        }
    )
    engine.reset_metrics()
    logits = engine.forward(
        torch.tensor([[1, 2, 3]], device="cuda"),
        subset_plan=subset,
        miss_policy="hard_commit",
    )
    metrics = engine.finish_metrics()
    assert torch.isfinite(logits).all()
    assert {slot.logical_expert for slots in engine.slots for slot in slots} == subset
    assert metrics.h2d_bytes == sum(engine.handles[key].byte_size for key in subset)
