"""Tiny model implementation of the common adapter contract."""

from __future__ import annotations

import copy
from typing import cast

from torch import Tensor, nn

from pseudoroute.analysis.pseudo_sequences import OfflinePseudoSequence
from pseudoroute.execution.routing_policy import RoutingPolicy
from pseudoroute.models.base import (
    ForwardResult,
    MoELayerHandle,
    MoEModelAdapter,
    RouteResult,
    TraceRequest,
)
from pseudoroute.models.registry import register_adapter
from pseudoroute.models.tiny_moe import LayerActivationTrace, TinyMoE, TinyMoELayer
from pseudoroute.types import ExpertKey, ModelSpec


@register_adapter("tiny")
class TinyMoEAdapter(MoEModelAdapter):
    def __init__(self, model: TinyMoE) -> None:
        self.model = model
        self.validate_structure()

    @property
    def spec(self) -> ModelSpec:
        return self.model.spec

    def _layer(self, layer_idx: int) -> TinyMoELayer:
        return cast(TinyMoELayer, self.model.layers[layer_idx])

    def validate_structure(self) -> None:
        if len(self.model.layers) != self.model.config.num_layers:
            raise ValueError("tiny adapter: configured layer count does not match model")
        for layer_idx in range(self.model.config.num_layers):
            layer = self._layer(layer_idx)
            if layer.layer_idx != layer_idx:
                raise ValueError(f"tiny adapter: layer index mismatch at {layer_idx}")
            if len(layer.experts) != self.model.config.num_experts:
                raise ValueError(f"tiny adapter: expert count mismatch at layer {layer_idx}")
            if layer.router.out_features != self.model.config.num_experts:
                raise ValueError(f"tiny adapter: router shape mismatch at layer {layer_idx}")
            if layer.top_k != self.model.config.top_k:
                raise ValueError(f"tiny adapter: top-k mismatch at layer {layer_idx}")

    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]:
        return tuple(
            MoELayerHandle(
                layer_idx=layer.layer_idx,
                router=layer.router,
                experts=tuple(layer.experts),
                top_k=layer.top_k,
                normalization=layer.router_norm,
                combine_semantics="softmax_topk_renormalized_weighted_sum",
                shared_expert=layer.shared_expert,
                score_function="softmax",
                has_router_bias=layer.router.bias is not None,
            )
            for layer in (self._layer(index) for index in range(self.model.config.num_layers))
        )

    def run_base_forward(
        self,
        input_ids: Tensor,
        *,
        position_ids: Tensor | None = None,
        kv_cache: object | None = None,
        use_cache: bool = False,
        trace_request: TraceRequest | None = None,
    ) -> ForwardResult:
        if position_ids is not None:
            expected = input_ids.new_tensor(range(input_ids.shape[1])).expand_as(input_ids)
            if not bool((position_ids == expected).all()):
                raise ValueError("tiny adapter supports canonical positions only in M1")
        if kv_cache is not None or use_cache:
            raise NotImplementedError("tiny adapter KV caching begins in a later milestone")
        output = self.model(input_ids, capture_trace=trace_request is not None)
        return ForwardResult(logits=output.logits, traces=output.traces)

    def route_from_state(self, layer_idx: int, router_input: Tensor) -> RouteResult:
        layer = self._layer(layer_idx)
        raw_logits = layer.router(router_input)
        scores = raw_logits.softmax(dim=-1)
        topk_scores, topk_ids = scores.topk(layer.top_k, dim=-1)
        weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        return RouteResult(raw_logits, scores, topk_ids, weights)

    def forward_with_policy(
        self,
        input_ids: Tensor,
        policy: RoutingPolicy,
        *,
        allowed_experts: frozenset[ExpertKey] | None = None,
        allowed_experts_by_position: dict[int, frozenset[ExpertKey]] | None = None,
        kv_cache: object | None = None,
        use_cache: bool = False,
    ) -> ForwardResult:
        if kv_cache is not None or use_cache:
            raise NotImplementedError("tiny adapter KV caching begins in a later milestone")
        output = self.model(
            input_ids,
            capture_trace=True,
            policy=policy,
            allowed_experts=allowed_experts,
            allowed_experts_by_position=allowed_experts_by_position,
        )
        return ForwardResult(
            logits=output.logits,
            traces=output.traces,
            executed_routes=output.executed_routes,
        )

    def capture_offline_factorial(
        self, sequence: OfflinePseudoSequence
    ) -> tuple[LayerActivationTrace, ...]:
        output = self.model(
            sequence.input_ids.to(next(self.model.parameters()).device),
            position_ids=sequence.position_ids.to(next(self.model.parameters()).device),
            capture_trace=True,
            capture_activations=True,
        )
        return tuple(
            activation
            for activation in output.activations
            if activation.token_position >= sequence.boundary
        )

    def clone_kv_cache_for_shadow(self, kv_cache: object) -> object:
        return copy.deepcopy(kv_cache)

    def get_expert_handle(self, key: ExpertKey) -> nn.Module:
        if key.layer_idx not in self.spec.moe_layer_indices:
            raise KeyError(key)
        experts = self._layer(key.layer_idx).experts
        if key.expert_idx >= len(experts):
            raise KeyError(key)
        return experts[key.expert_idx]
