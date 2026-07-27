"""Streaming expert-default calibration and M7 shadow rollout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.probes.base import FutureRoutingProbe, ProbeOutput
from pseudoroute.probes.pseudo import _attend
from pseudoroute.probes.shadow_cache import ShadowKVBuffer, build_read_only_cache
from pseudoroute.types import DeployableDecodeState, InformationRegime


@dataclass(frozen=True)
class DefaultVectorStore:
    count: Tensor  # [layers, experts]
    mean: Tensor  # [layers, experts, hidden]
    variance: Tensor  # [layers, experts, hidden]
    definition: str
    calibration_fingerprint: str


class StreamingDefaultVectorCollector:
    def __init__(self, num_layers: int, num_experts: int, hidden_size: int) -> None:
        self.count = torch.zeros(num_layers, num_experts, dtype=torch.int64)
        self.mean = torch.zeros(num_layers, num_experts, hidden_size, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)
        self._fingerprint = hashlib.sha256()

    def update(self, layer_idx: int, expert_idx: int, value: Tensor) -> None:
        vector = value.detach().cpu().double().reshape(-1)
        self._fingerprint.update(vector.contiguous().numpy().tobytes())
        count = int(self.count[layer_idx, expert_idx]) + 1
        self.count[layer_idx, expert_idx] = count
        delta = vector - self.mean[layer_idx, expert_idx]
        self.mean[layer_idx, expert_idx] += delta / count
        delta2 = vector - self.mean[layer_idx, expert_idx]
        self.m2[layer_idx, expert_idx] += delta * delta2

    def finalize(self) -> DefaultVectorStore:
        denominator = (self.count - 1).clamp_min(1).unsqueeze(-1)
        variance = self.m2 / denominator
        variance[self.count < 2] = 0
        return DefaultVectorStore(
            self.count.clone(),
            self.mean.clone(),
            variance,
            "expert_output",
            self._fingerprint.hexdigest(),
        )


def calibrate_default_vectors(
    adapter: TinyMoEAdapter, documents: tuple[tuple[str, Tensor], ...]
) -> DefaultVectorStore:
    collector = StreamingDefaultVectorCollector(
        len(adapter.spec.moe_layer_indices),
        next(iter(adapter.spec.num_experts_by_layer.values())),
        adapter.spec.hidden_size,
    )
    with torch.inference_mode():
        for _, token_ids in documents:
            output = adapter.model(token_ids, capture_trace=True, capture_activations=True)
            for activation in output.activations:
                if activation.router_input is None:
                    raise ValueError("default calibration requires router input capture")
                layer = adapter._layer(activation.layer_idx)  # noqa: SLF001
                for expert_idx in activation.topk_ids.reshape(-1).tolist():
                    value = layer.experts[int(expert_idx)](activation.router_input)
                    collector.update(activation.layer_idx, int(expert_idx), value)
    return collector.finalize()


def save_default_vectors(root: Path, store: DefaultVectorStore) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "count": store.count.contiguous(),
            "mean": store.mean.contiguous(),
            "variance": store.variance.contiguous(),
        },
        str(root / "default_vectors.safetensors"),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "safe_format": "safetensors",
                "definition": store.definition,
                "calibration_fingerprint": store.calibration_fingerprint,
                "counts": store.count.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_default_vectors(root: Path) -> DefaultVectorStore:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("safe_format") != "safetensors"
        or manifest.get("definition") != "expert_output"
    ):
        raise ValueError("unsafe or unsupported default-vector artifact")
    tensors = load_file(str(root / "default_vectors.safetensors"))
    return DefaultVectorStore(
        tensors["count"],
        tensors["mean"].double(),
        tensors["variance"].double(),
        str(manifest["definition"]),
        str(manifest["calibration_fingerprint"]),
    )


@dataclass(frozen=True)
class DefaultVectorShadowRolloutProbe(FutureRoutingProbe):
    adapter: TinyMoEAdapter
    defaults: DefaultVectorStore
    pseudo_content: str = "current_token"
    attention_mode: str = "independent"

    @property
    def name(self) -> str:
        return f"default_vector_shadow_rollout_{self.attention_mode}_{self.pseudo_content}"

    @property
    def requires_training(self) -> bool:
        return False

    def _content_id(self, state: DeployableDecodeState) -> int:
        if self.pseudo_content == "current_token":
            return int(state.prefix_token_ids[0, -1])
        if self.pseudo_content == "sampled_next_token":
            if state.information_regime is not InformationRegime.ONLINE_POST_SAMPLE:
                raise ValueError("sampled_next_token requires online_post_sample")
            if state.next_token_id is None:
                raise ValueError("post-sample state is missing its sampled next token")
            return state.next_token_id
        raise ValueError(f"unsupported pseudo content: {self.pseudo_content}")

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        cache = build_read_only_cache(self.adapter, state.prefix_token_ids)
        content_id = self._content_id(state)
        hidden_by_anchor = {
            horizon: self.adapter.model.embedding.weight[content_id].unsqueeze(0)
            for horizon in horizons
        }
        probabilities: dict[int, list[Tensor]] = {
            layer: [] for layer in self.adapter.spec.moe_layer_indices
        }
        uncertainties: dict[int, list[Tensor]] = {
            layer: [] for layer in self.adapter.spec.moe_layer_indices
        }
        with torch.inference_mode():
            for layer_idx, production in enumerate(cache.layers):
                layer = self.adapter._layer(layer_idx)  # noqa: SLF001
                shadow = ShadowKVBuffer.empty()
                for horizon in horizons:
                    hidden = hidden_by_anchor[horizon]
                    normalized = layer.attention_norm(hidden)
                    qkv = layer.attention.qkv(normalized).view(
                        1, 1, 3, layer.attention.num_heads, layer.attention.head_dim
                    )
                    query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
                    position = state.absolute_position + horizon
                    if layer.attention.rope is not None:
                        key = layer.attention.rope(key, key.new_tensor([position]))
                    attention = _attend(
                        self.adapter,
                        layer_idx,
                        query[:, :, 0],
                        production,
                        position=position,
                        shadow=shadow if self.attention_mode == "causal" else None,
                        pseudo_key=key,
                        pseudo_value=value,
                    )
                    post_attention = hidden + attention
                    router_input = layer.router_norm(post_attention)
                    route = self.adapter.route_from_state(layer_idx, router_input)
                    probabilities[layer_idx].append(route.pre_topk_scores.squeeze(0))
                    mixture = torch.zeros_like(hidden, dtype=torch.float64)
                    variance = torch.zeros_like(mixture)
                    for expert_idx in range(self.defaults.mean.shape[1]):
                        weight = (
                            torch.where(
                                route.topk_ids == expert_idx,
                                route.topk_weights,
                                torch.zeros_like(route.topk_weights),
                            )
                            .sum(dim=-1, keepdim=True)
                            .double()
                        )
                        mixture += weight * self.defaults.mean[layer_idx, expert_idx].to(
                            hidden.device
                        )
                        variance += weight.square() * self.defaults.variance[
                            layer_idx, expert_idx
                        ].to(hidden.device)
                    hidden_by_anchor[horizon] = post_attention + mixture.to(post_attention.dtype)
                    uncertainties[layer_idx].append(variance.mean().sqrt().reshape(1))
        cache.assert_unchanged()
        per_horizon = {layer: torch.stack(values) for layer, values in probabilities.items()}
        uncertainty = {layer: torch.cat(values) for layer, values in uncertainties.items()}
        interval_weights = torch.tensor(
            [
                horizons[0],
                *(right - left for left, right in zip(horizons, horizons[1:], strict=False)),
            ],
            dtype=torch.float64,
        )
        temporary_bytes = sum(
            value.numel() * value.element_size() for value in per_horizon.values()
        )
        temporary_bytes += len(horizons) * self.adapter.spec.hidden_size * 8
        if self.attention_mode == "causal":
            temporary_bytes += 2 * len(horizons) * self.adapter.spec.hidden_size * 4
        return ProbeOutput(
            horizons,
            None,
            per_horizon,
            {
                layer: (value.double() * interval_weights.to(value.device)[:, None]).sum(0)
                for layer, value in per_horizon.items()
            },
            uncertainty,
            0.0,
            {
                "probe": self.name,
                "information_regime": state.information_regime.value,
                "anchors": list(horizons),
                "pseudo_content": self.pseudo_content,
                "pseudo_attention": self.attention_mode,
                "default_definition": self.defaults.definition,
                "calibration_fingerprint": self.defaults.calibration_fingerprint,
                "temporary_bytes": temporary_bytes,
                "production_cache_unchanged": True,
            },
        )


def same_token_next_layer_router_error(
    adapter: TinyMoEAdapter, token_ids: Tensor, defaults: DefaultVectorStore
) -> tuple[float, ...]:
    """Replace one layer's expert outputs by defaults and score the next router."""
    model = adapter.model
    hidden = model.embedding(token_ids.to(next(model.parameters()).device))
    errors = []
    with torch.inference_mode():
        for layer_idx in adapter.spec.moe_layer_indices:
            layer = adapter._layer(layer_idx)  # noqa: SLF001
            attention, _, _ = layer.attention(layer.attention_norm(hidden))
            post_attention = hidden + attention
            router_input = layer.router_norm(post_attention)
            route = adapter.route_from_state(layer_idx, router_input)
            exact = torch.zeros_like(hidden)
            approximate = torch.zeros_like(hidden, dtype=torch.float64)
            for expert_idx, expert in enumerate(layer.experts):
                weights = torch.where(
                    route.topk_ids == expert_idx,
                    route.topk_weights,
                    torch.zeros_like(route.topk_weights),
                ).sum(dim=-1, keepdim=True)
                exact += weights * expert(router_input)
                approximate += weights.double() * defaults.mean[layer_idx, expert_idx].to(
                    hidden.device
                )
            if layer_idx + 1 < len(adapter.spec.moe_layer_indices):
                next_layer = adapter._layer(layer_idx + 1)  # noqa: SLF001
                exact_hidden = post_attention + exact
                approx_hidden = post_attention + approximate.to(hidden.dtype)
                exact_attention, _, _ = next_layer.attention(
                    next_layer.attention_norm(exact_hidden)
                )
                approx_attention, _, _ = next_layer.attention(
                    next_layer.attention_norm(approx_hidden)
                )
                exact_logits = next_layer.router(
                    next_layer.router_norm(exact_hidden + exact_attention)
                )
                approx_logits = next_layer.router(
                    next_layer.router_norm(approx_hidden + approx_attention)
                )
                errors.append(float((exact_logits - approx_logits).square().mean().sqrt()))
            hidden = post_attention + exact
    return tuple(errors)
