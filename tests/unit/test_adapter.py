from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pseudoroute.config import load_config
from pseudoroute.execution import NaturalRoutingPolicy
from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.base import TraceLevel, TraceRequest
from pseudoroute.models.introspection import export_model_manifest
from pseudoroute.models.registry import create_adapter, registered_adapters
from pseudoroute.models.tiny_moe import TinyMoE


def adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    model = TinyMoE(config.model, seed=config.experiment.seed, device="cpu")
    return TinyMoEAdapter(model)


def test_tiny_adapter_implements_common_contract() -> None:
    subject = adapter()
    assert subject.spec.model_id == "tiny_moe"
    assert len(subject.iter_moe_layers()) == 2
    assert all(len(layer.experts) == 4 for layer in subject.iter_moe_layers())
    subject.validate_structure()


def test_trace_capture_does_not_change_tiny_adapter_logits() -> None:
    subject = adapter()
    tokens = torch.tensor([[1, 2, 3]])
    base = subject.run_base_forward(tokens)
    traced = subject.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    assert torch.equal(base.logits, traced.logits)
    assert len(traced.traces) == 6


def test_natural_policy_does_not_change_tiny_adapter_outputs() -> None:
    subject = adapter()
    tokens = torch.tensor([[1, 2, 3]])
    base = subject.run_base_forward(tokens)
    policy = subject.forward_with_policy(tokens, NaturalRoutingPolicy())
    assert torch.equal(base.logits, policy.logits)


def test_registry_and_unsupported_adapter_error() -> None:
    assert "tiny" in registered_adapters()
    config = load_config("configs/model/tiny_moe.yaml")
    created = create_adapter(
        "tiny", model=TinyMoE(config.model, seed=config.experiment.seed, device="cpu")
    )
    assert isinstance(created, TinyMoEAdapter)
    with pytest.raises(ValueError, match="registered adapters"):
        create_adapter("unknown")


class UnsupportedHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(architectures=("UnsupportedForCausalLM",))
        self.model = SimpleNamespace(layers=())


def test_unsupported_hf_architecture_fails_actionably() -> None:
    with pytest.raises(ValueError, match="requires MixtralForCausalLM"):
        HFMixtralAdapter(UnsupportedHFModel(), model_id="unsupported", revision="0" * 40)


def test_model_manifest_export_is_stable(tmp_path) -> None:
    destination = tmp_path / "model_manifest.json"
    export_model_manifest(destination, adapter().spec, revision="in-repository")
    first = destination.read_text()
    export_model_manifest(destination, adapter().spec, revision="in-repository")
    assert destination.read_text() == first
    assert '"fingerprint"' in first
