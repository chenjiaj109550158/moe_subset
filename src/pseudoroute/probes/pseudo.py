"""M7 future-position and pseudo-token routing probes for the tiny adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor

from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.probes.base import FutureRoutingProbe, ProbeOutput
from pseudoroute.probes.shadow_cache import (
    ReadOnlyLayerCache,
    ShadowKVBuffer,
    build_read_only_cache,
)
from pseudoroute.types import DeployableDecodeState, InformationRegime


def _attend(
    adapter: TinyMoEAdapter,
    layer_idx: int,
    query: Tensor,
    production: ReadOnlyLayerCache,
    *,
    position: int,
    shadow: ShadowKVBuffer | None = None,
    pseudo_key: Tensor | None = None,
    pseudo_value: Tensor | None = None,
) -> Tensor:
    layer = adapter._layer(layer_idx)  # noqa: SLF001 - adapter reference shadow path
    if layer.attention.rope is not None:
        query = layer.attention.rope(query.unsqueeze(2), query.new_tensor([position])).squeeze(2)
    keys = production.keys
    values = production.values
    if shadow is not None and shadow.keys:
        keys = torch.cat((keys, *shadow.keys), dim=2)
        values = torch.cat((values, *shadow.values), dim=2)
    scores = (query.unsqueeze(2) @ keys.transpose(-1, -2)) / math.sqrt(layer.attention.head_dim)
    attended = scores.softmax(dim=-1) @ values
    hidden_size = layer.attention.num_heads * layer.attention.head_dim
    output = layer.attention.output(attended.transpose(1, 2).reshape(1, 1, hidden_size))[:, 0]
    if shadow is not None and pseudo_key is not None and pseudo_value is not None:
        shadow.keys.append(pseudo_key)
        shadow.values.append(pseudo_value)
    return cast(Tensor, output)


def _route_output(
    name: str,
    horizons: tuple[int, ...],
    probabilities: dict[int, Tensor],
    *,
    metadata: dict[str, object],
) -> ProbeOutput:
    interval_weights = torch.tensor(
        [
            horizons[0],
            *(right - left for left, right in zip(horizons, horizons[1:], strict=False)),
        ],
        dtype=torch.float64,
    )
    aggregate = {
        layer: (values.double() * interval_weights.to(values.device)[:, None]).sum(dim=0)
        for layer, values in probabilities.items()
    }
    temporary_bytes = sum(value.numel() * value.element_size() for value in probabilities.values())
    return ProbeOutput(
        horizons,
        None,
        probabilities,
        aggregate,
        None,
        0.0,
        {"probe": name, "temporary_bytes": temporary_bytes, **metadata},
    )


@dataclass(frozen=True)
class FuturePositionRephasedProbe(FutureRoutingProbe):
    adapter: TinyMoEAdapter
    position_mode: Literal["correct", "no_rephase", "wrong_position"] = "correct"
    wrong_offset: int = -1
    random_content: bool = False
    seed: int = 0

    @property
    def name(self) -> str:
        suffix = "random_content" if self.random_content else self.position_mode
        return f"future_position_rephased_{suffix}"

    @property
    def requires_training(self) -> bool:
        return False

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        cache = build_read_only_cache(self.adapter, state.prefix_token_ids)
        per_layer: dict[int, list[Tensor]] = {
            layer: [] for layer in self.adapter.spec.moe_layer_indices
        }
        devices = (
            [next(self.adapter.model.parameters()).device.index or 0]
            if next(self.adapter.model.parameters()).is_cuda
            else []
        )
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(self.seed)
            for horizon in horizons:
                for layer_idx, production in enumerate(cache.layers):
                    query = production.current_pre_rope_query
                    if self.random_content:
                        query = torch.randn_like(query)
                    position = state.absolute_position + horizon
                    if self.position_mode == "no_rephase":
                        position = state.absolute_position
                    elif self.position_mode == "wrong_position":
                        position += self.wrong_offset
                    attention = _attend(
                        self.adapter, layer_idx, query, production, position=position
                    )
                    layer = self.adapter._layer(layer_idx)  # noqa: SLF001
                    router_input = layer.router_norm(production.current_hidden + attention)
                    per_layer[layer_idx].append(
                        layer.router(router_input).softmax(dim=-1).squeeze(0)
                    )
        cache.assert_unchanged()
        probabilities = {layer: torch.stack(values) for layer, values in per_layer.items()}
        head_elements = self.adapter.model.config.hidden_size
        score_elements = self.adapter.model.config.num_heads * cache.prefix_length
        scratch_bytes = (head_elements + score_elements + self.adapter.spec.hidden_size) * 4
        return _route_output(
            self.name,
            horizons,
            probabilities,
            metadata={
                "information_regime": state.information_regime.value,
                "position_mode": self.position_mode,
                "random_content": self.random_content,
                "positions": [state.absolute_position + horizon for horizon in horizons],
                "production_cache_unchanged": True,
                "temporary_bytes": scratch_bytes
                + sum(value.numel() * value.element_size() for value in probabilities.values()),
            },
        )


@dataclass(frozen=True)
class PseudoTokenProbe(FutureRoutingProbe):
    adapter: TinyMoEAdapter
    attention_mode: Literal["independent", "causal"]
    pseudo_content: Literal["current_token", "sampled_next_token", "fixed_token"]
    fixed_token_id: int = 0

    @property
    def name(self) -> str:
        return f"pseudo_token_{self.attention_mode}_{self.pseudo_content}"

    @property
    def requires_training(self) -> bool:
        return False

    def _content_id(self, state: DeployableDecodeState) -> int:
        if self.pseudo_content == "current_token":
            return int(state.prefix_token_ids[0, -1])
        if self.pseudo_content == "fixed_token":
            return self.fixed_token_id
        if state.information_regime is not InformationRegime.ONLINE_POST_SAMPLE:
            raise ValueError("sampled_next_token content requires online_post_sample")
        if state.next_token_id is None:
            raise ValueError("post-sample state must contain exactly one sampled next token")
        return state.next_token_id

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        content_id = self._content_id(state)
        cache = build_read_only_cache(self.adapter, state.prefix_token_ids)
        hidden_by_anchor = {
            horizon: self.adapter.model.embedding.weight[content_id].unsqueeze(0)
            for horizon in horizons
        }
        per_layer: dict[int, list[Tensor]] = {
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
                    per_layer[layer_idx].append(route.pre_topk_scores.squeeze(0))
                    combined = torch.zeros_like(hidden)
                    for expert_idx, expert in enumerate(layer.experts):
                        weight = torch.where(
                            route.topk_ids == expert_idx,
                            route.topk_weights,
                            torch.zeros_like(route.topk_weights),
                        ).sum(dim=-1, keepdim=True)
                        combined += weight * expert(router_input)
                    if layer.shared_expert is not None:
                        combined += layer.shared_expert(router_input)
                    hidden_by_anchor[horizon] = post_attention + combined
        cache.assert_unchanged()
        probabilities = {layer: torch.stack(values) for layer, values in per_layer.items()}
        hidden_bytes = len(horizons) * self.adapter.spec.hidden_size * 4
        shadow_bytes = (
            2 * len(horizons) * self.adapter.spec.hidden_size * 4
            if self.attention_mode == "causal"
            else 0
        )
        return _route_output(
            self.name,
            horizons,
            probabilities,
            metadata={
                "information_regime": state.information_regime.value,
                "pseudo_content": self.pseudo_content,
                "pseudo_attention": self.attention_mode,
                "positions": [state.absolute_position + horizon for horizon in horizons],
                "production_cache_unchanged": True,
                "temporary_bytes": hidden_bytes
                + shadow_bytes
                + sum(value.numel() * value.element_size() for value in probabilities.values()),
            },
        )


@dataclass(frozen=True)
class UncertaintyEnsembleProbe(FutureRoutingProbe):
    branches: tuple[FutureRoutingProbe, ...]

    @property
    def name(self) -> str:
        return "uncertainty_ensemble"

    @property
    def requires_training(self) -> bool:
        return any(branch.requires_training for branch in self.branches)

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        if len(self.branches) < 2:
            raise ValueError("uncertainty ensemble requires at least two branches")
        outputs = tuple(branch.predict(state, horizons) for branch in self.branches)
        aggregate = {}
        uncertainty = {}
        per_horizon = {}
        for layer in outputs[0].aggregate_utility:
            utilities = torch.stack([output.aggregate_utility[layer] for output in outputs])
            aggregate[layer] = utilities.mean(0)
            uncertainty[layer] = utilities.std(0, unbiased=False)
            branch_probabilities = [output.per_horizon_probs for output in outputs]
            if all(values is not None for values in branch_probabilities):
                per_horizon[layer] = torch.stack(
                    [values[layer] for values in branch_probabilities if values is not None]
                ).mean(0)
        temporary_bytes = sum(int(output.metadata.get("temporary_bytes", 0)) for output in outputs)
        return ProbeOutput(
            horizons,
            None,
            per_horizon or None,
            aggregate,
            uncertainty,
            0.0,
            {
                "probe": self.name,
                "branches": [branch.name for branch in self.branches],
                "temporary_bytes": temporary_bytes,
                "information_regime": state.information_regime.value,
            },
        )
