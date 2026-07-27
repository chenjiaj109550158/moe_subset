"""Safely serializable learned M6 deployable probes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from pseudoroute.probes.base import FutureRoutingProbe, ProbeOutput
from pseudoroute.training.dataset import state_features
from pseudoroute.types import DeployableDecodeState


@dataclass(frozen=True)
class FeatureSpec:
    num_layers: int
    num_experts: int
    history_length: int
    position_scale: float
    horizon_scale: float


class LearnedProbe(FutureRoutingProbe):
    feature_spec: FeatureSpec

    @property
    def requires_training(self) -> bool:
        return True

    def predict_flat(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        predictions = []
        for horizon in horizons:
            features = state_features(
                state,
                horizon,
                num_experts=self.feature_spec.num_experts,
                history_length=self.feature_spec.history_length,
                position_scale=self.feature_spec.position_scale,
                horizon_scale=self.feature_spec.horizon_scale,
            )
            predictions.append(self.predict_flat(features).clamp_min(0))
        stacked = torch.stack(predictions)
        aggregate = {
            layer: stacked[
                -1,
                layer * self.feature_spec.num_experts : (layer + 1) * self.feature_spec.num_experts,
            ]
            for layer in range(self.feature_spec.num_layers)
        }
        per_horizon = {}
        for layer in range(self.feature_spec.num_layers):
            values = stacked[
                :,
                layer * self.feature_spec.num_experts : (layer + 1) * self.feature_spec.num_experts,
            ]
            per_horizon[layer] = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return ProbeOutput(
            horizons,
            None,
            per_horizon,
            aggregate,
            None,
            0.0,
            {"probe": self.name, "online_only": True},
        )


@dataclass(frozen=True)
class DirectLinearProbe(LearnedProbe):
    weight: Tensor
    bias: Tensor
    feature_spec: FeatureSpec
    ridge_lambda: float = 0.0

    @property
    def name(self) -> str:
        return "direct_ridge" if self.ridge_lambda > 0 else "direct_linear"

    def predict_flat(self, features: Tensor) -> Tensor:
        return features.double() @ self.weight.double() + self.bias.double()


@dataclass(frozen=True)
class DirectMLPProbe(LearnedProbe):
    input_weight: Tensor
    input_bias: Tensor
    output_weight: Tensor
    output_bias: Tensor
    feature_spec: FeatureSpec

    @property
    def name(self) -> str:
        return "direct_mlp"

    def predict_flat(self, features: Tensor) -> Tensor:
        hidden = torch.relu(features.float() @ self.input_weight.float() + self.input_bias.float())
        return (hidden @ self.output_weight.float() + self.output_bias.float()).double()


@dataclass(frozen=True)
class ADEPTStyleRFProbe(LearnedProbe):
    feature_indices: Tensor
    thresholds: Tensor
    left_values: Tensor
    right_values: Tensor
    feature_spec: FeatureSpec

    @property
    def name(self) -> str:
        return "adept_style_rf"

    def predict_flat(self, features: Tensor) -> Tensor:
        selected = features.double()[self.feature_indices.long()]
        choose_left = selected <= self.thresholds.double()
        values = torch.where(
            choose_left[:, None], self.left_values.double(), self.right_values.double()
        )
        return values.mean(dim=0)


def save_learned_probe(root: Path, probe: LearnedProbe) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "safe_format": "safetensors",
        "probe_name": probe.name,
        "information_regime": "online_pre_sample",
        "feature_spec": probe.feature_spec.__dict__,
    }
    if isinstance(probe, DirectLinearProbe):
        manifest["ridge_lambda"] = probe.ridge_lambda
        tensors = {"weight": probe.weight, "bias": probe.bias}
    elif isinstance(probe, DirectMLPProbe):
        tensors = {
            "input_weight": probe.input_weight,
            "input_bias": probe.input_bias,
            "output_weight": probe.output_weight,
            "output_bias": probe.output_bias,
        }
    elif isinstance(probe, ADEPTStyleRFProbe):
        tensors = {
            "feature_indices": probe.feature_indices,
            "thresholds": probe.thresholds,
            "left_values": probe.left_values,
            "right_values": probe.right_values,
        }
    else:
        raise TypeError(f"unsupported learned probe: {type(probe).__name__}")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(root / "model.safetensors"),
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_learned_probe(root: Path) -> LearnedProbe:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("safe_format") != "safetensors":
        raise ValueError("unsafe or unsupported predictor format")
    tensors = load_file(str(root / "model.safetensors"))
    spec = FeatureSpec(**manifest["feature_spec"])
    name = manifest["probe_name"]
    if name in {"direct_linear", "direct_ridge"}:
        return DirectLinearProbe(
            tensors["weight"], tensors["bias"], spec, float(manifest["ridge_lambda"])
        )
    if name == "direct_mlp":
        return DirectMLPProbe(
            tensors["input_weight"],
            tensors["input_bias"],
            tensors["output_weight"],
            tensors["output_bias"],
            spec,
        )
    if name == "adept_style_rf":
        return ADEPTStyleRFProbe(
            tensors["feature_indices"],
            tensors["thresholds"],
            tensors["left_values"],
            tensors["right_values"],
            spec,
        )
    raise ValueError(f"unknown predictor type: {name}")
