"""Exact Qwen2-MoE adapter preserving routed and shared-expert semantics."""

from __future__ import annotations

import copy
from functools import cached_property
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.execution.routing_policy import ExecutedRoute, RoutingPolicy
from pseudoroute.models.adapters.hf_mixtral import PackedMixtralExpert
from pseudoroute.models.adapters.subset_hooks import (
    NativeRouteSemantics,
    install_tuple_router_hooks,
)
from pseudoroute.models.base import (
    ForwardResult,
    MoELayerHandle,
    MoEModelAdapter,
    RouteResult,
    TraceRequest,
)
from pseudoroute.models.registry import register_adapter
from pseudoroute.types import ExpertKey, ModelSpec, RouterTrace


@register_adapter("hf_qwen2_moe")
class HFQwen2MoeAdapter(MoEModelAdapter):
    architecture = "Qwen2MoeForCausalLM"

    def __init__(self, model: nn.Module, *, model_id: str, revision: str) -> None:
        self.model = model
        self.model_id = model_id
        self.revision = revision
        self.validate_structure()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        device: str = "cpu",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        device_map: object | None = None,
        max_memory: dict[object, object] | None = None,
    ) -> HFQwen2MoeAdapter:
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, object] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
            "dtype": "auto",
        }
        if device_map is not None:
            kwargs.update(device_map=device_map, max_memory=max_memory)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        module = cast(nn.Module, model)
        if device_map is None:
            module.to(torch.device(device))
        module.eval()
        return cls(module, model_id=model_id, revision=revision)

    def _config(self) -> Any:
        return cast(Any, self.model.config)

    def _layers(self) -> Any:
        return cast(Any, cast(Any, self.model).model.layers)

    @property
    def input_device(self) -> torch.device:
        device = cast(Any, self.model).model.embed_tokens.weight.device
        return cast(torch.device, device)

    @cached_property
    def spec(self) -> ModelSpec:
        config = self._config()
        indices = tuple(range(int(config.num_hidden_layers)))
        expert_bytes: dict[ExpertKey, int] = {}
        for layer_idx, layer in enumerate(self._layers()):
            for expert_idx in range(int(config.num_experts)):
                expert_bytes[ExpertKey(layer_idx, expert_idx)] = sum(
                    int(parameter[expert_idx].numel() * parameter.element_size())
                    for parameter in layer.mlp.experts.parameters()
                    if parameter.ndim > 0 and parameter.shape[0] == config.num_experts
                )
        return ModelSpec(
            model_id=self.model_id,
            architecture=self.architecture,
            num_layers=int(config.num_hidden_layers),
            moe_layer_indices=indices,
            num_experts_by_layer={i: int(config.num_experts) for i in indices},
            top_k_by_layer={i: int(config.num_experts_per_tok) for i in indices},
            hidden_size=int(config.hidden_size),
            uses_rope=True,
            pre_norm=True,
            expert_bytes=expert_bytes,
            shared_experts_by_layer={i: 1 for i in indices},
            routing_semantics_by_layer={
                i: "softmax_float32_topk_"
                + ("renormalized" if config.norm_topk_prob else "not_renormalized")
                + "_plus_sigmoid_gated_shared_expert"
                for i in indices
            },
        )

    def validate_structure(self) -> None:
        config = self._config()
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if self.architecture not in architectures:
            raise ValueError(f"hf_qwen2_moe requires {self.architecture}; got {architectures}")
        if len(self._layers()) != int(config.num_hidden_layers):
            raise ValueError("hf_qwen2_moe: decoder layer count does not match config")
        for layer_idx, layer in enumerate(self._layers()):
            if tuple(layer.mlp.gate.weight.shape) != (
                int(config.num_experts),
                int(config.hidden_size),
            ):
                raise ValueError(f"hf_qwen2_moe: router shape mismatch at layer {layer_idx}")
            if not hasattr(layer.mlp, "shared_expert") or not hasattr(
                layer.mlp, "shared_expert_gate"
            ):
                raise ValueError(f"hf_qwen2_moe: missing shared expert at layer {layer_idx}")

    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]:
        config = self._config()
        return tuple(
            MoELayerHandle(
                layer_idx=i,
                router=cast(nn.Module, layer.mlp.gate),
                experts=tuple(
                    PackedMixtralExpert(layer.mlp.experts, e)
                    for e in range(int(config.num_experts))
                ),
                top_k=int(config.num_experts_per_tok),
                normalization=cast(nn.Module, layer.post_attention_layernorm),
                combine_semantics=self.spec.routing_semantics_by_layer[i],
                shared_expert=cast(nn.Module, layer.mlp.shared_expert),
                score_function="softmax_float32",
                has_router_bias=False,
                shared_expert_count=1,
            )
            for i, layer in enumerate(self._layers())
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
        with torch.inference_mode():
            output = cast(
                Any,
                self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=kv_cache,
                    use_cache=use_cache,
                    output_router_logits=trace_request is not None,
                    return_dict=True,
                ),
            )
        traces: list[RouterTrace] = []
        if trace_request is not None:
            batch, sequence = input_ids.shape
            if batch != 1:
                raise ValueError("trained trace capture supports batch size 1")
            for layer_idx, native_logits in enumerate(output.router_logits):
                logits = native_logits.reshape(batch, sequence, -1)
                scores = logits.float().softmax(dim=-1)
                weights, ids = scores.topk(self.spec.top_k_by_layer[layer_idx], dim=-1)
                if bool(self._config().norm_topk_prob):
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                weights = weights.to(logits.dtype)
                for position in range(sequence):
                    traces.append(
                        RouterTrace(
                            position,
                            layer_idx,
                            logits[:, position].detach().cpu(),
                            scores[:, position].detach().cpu(),
                            ids[:, position].detach().cpu(),
                            weights[:, position].detach().cpu(),
                        )
                    )
        return ForwardResult(
            logits=cast(Tensor, output.logits),
            traces=tuple(traces),
            kv_cache=getattr(output, "past_key_values", None) if use_cache else None,
        )

    def route_from_state(self, layer_idx: int, router_input: Tensor) -> RouteResult:
        logits, weights, ids = self._layers()[layer_idx].mlp.gate(router_input)
        return RouteResult(logits, logits.float().softmax(dim=-1), ids, weights)

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
        if allowed_experts_by_position is not None:
            raise NotImplementedError("trained cached execution accepts the current global subset")
        records: list[ExecutedRoute] = []
        handles = install_tuple_router_hooks(
            tuple((i, cast(nn.Module, layer.mlp.gate)) for i, layer in enumerate(self._layers())),
            policy=policy,
            allowed_experts=allowed_experts,
            top_k_by_layer=self.spec.top_k_by_layer,
            semantics_by_layer={
                i: NativeRouteSemantics("softmax_float32", bool(self._config().norm_topk_prob))
                for i in self.spec.moe_layer_indices
            },
            records=records,
        )
        try:
            result = self.run_base_forward(input_ids, kv_cache=kv_cache, use_cache=use_cache)
        finally:
            for handle in handles:
                handle.remove()
        return ForwardResult(result.logits, result.traces, result.kv_cache, tuple(records))

    def clone_kv_cache_for_shadow(self, kv_cache: object) -> object:
        return copy.deepcopy(kv_cache)

    def get_expert_handle(self, key: ExpertKey) -> nn.Module:
        if (
            key.layer_idx not in self.spec.moe_layer_indices
            or key.expert_idx >= self.spec.num_experts_by_layer.get(key.layer_idx, 0)
        ):
            raise KeyError(key)
        return PackedMixtralExpert(self._layers()[key.layer_idx].mlp.experts, key.expert_idx)
