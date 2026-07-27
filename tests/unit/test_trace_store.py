import json
from pathlib import Path

import pytest
import torch

from pseudoroute.config import load_config
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.base import TraceLevel
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.tracing import TraceStore, TraceValidationError, collect_sample, validate_trace


def tiny_adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=config.experiment.seed, device="cpu"))


def store(
    root: Path, *, resume: bool = False, level: TraceLevel = TraceLevel.ROUTER_LOGITS
) -> TraceStore:
    return TraceStore.create(
        root,
        spec=tiny_adapter().spec,
        model_revision="in-repository",
        dataset_id="synthetic",
        dataset_revision="v1",
        dataset_split="test",
        trace_level=level,
        resume=resume,
    )


def test_chunked_store_random_access_resume_and_validation(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    first = store(root)
    tokens = torch.tensor([[1, 2, 3]])
    assert collect_sample(
        tiny_adapter(),
        first,
        sample_id="sample-a",
        token_ids=tokens,
        trace_level=TraceLevel.ROUTER_LOGITS,
    )
    resumed = store(root, resume=True)
    assert not collect_sample(
        tiny_adapter(),
        resumed,
        sample_id="sample-a",
        token_ids=tokens,
        trace_level=TraceLevel.ROUTER_LOGITS,
    )
    assert collect_sample(
        tiny_adapter(),
        resumed,
        sample_id="sample-b",
        token_ids=torch.tensor([[4, 5]]),
        trace_level=TraceLevel.ROUTER_LOGITS,
    )
    resumed.mark_complete()
    manifest = validate_trace(root)
    assert manifest.num_samples == 2
    assert manifest.num_tokens == 5
    assert [record.start_offset for record in manifest.shards] == [0, 3]
    loaded = resumed.load_sample("sample-a")
    assert loaded["router_logits"].shape == (3, 2, 4)
    assert loaded["router_pre_topk_scores"].shape == (3, 2, 4)
    assert loaded["router_topk_ids"].shape == (3, 2, 2)
    assert loaded["layer_ids"].tolist() == list(tiny_adapter().spec.moe_layer_indices)
    assert manifest.schema_version == 2
    assert manifest.moe_layer_indices == tiny_adapter().spec.moe_layer_indices
    assert manifest.shared_experts_by_layer == tiny_adapter().spec.shared_experts_by_layer
    assert manifest.routing_semantics_by_layer == tiny_adapter().spec.routing_semantics_by_layer


def test_routes_only_omits_full_logits(tmp_path: Path) -> None:
    subject = store(tmp_path / "routes", level=TraceLevel.ROUTES_ONLY)
    collect_sample(
        tiny_adapter(),
        subject,
        sample_id="one",
        token_ids=torch.tensor([[1, 2]]),
        trace_level=TraceLevel.ROUTES_ONLY,
    )
    subject.mark_complete()
    assert "router_logits" not in subject.load_sample("one")
    validate_trace(tmp_path / "routes")


def test_validation_detects_corruption(tmp_path: Path) -> None:
    root = tmp_path / "corrupt"
    subject = store(root)
    collect_sample(
        tiny_adapter(),
        subject,
        sample_id="one",
        token_ids=torch.tensor([[1]]),
        trace_level=TraceLevel.ROUTER_LOGITS,
    )
    subject.mark_complete()
    manifest = json.loads((root / "manifest.json").read_text())
    shard = root / manifest["shards"][0]["path"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(TraceValidationError, match="checksum mismatch"):
        validate_trace(root)


def test_resume_rejects_incompatible_manifest(tmp_path: Path) -> None:
    root = tmp_path / "incompatible"
    store(root)
    with pytest.raises(ValueError, match="incompatible"):
        TraceStore.create(
            root,
            spec=tiny_adapter().spec,
            model_revision="different",
            dataset_id="synthetic",
            dataset_revision="v1",
            dataset_split="test",
            trace_level=TraceLevel.ROUTER_LOGITS,
            resume=True,
        )
