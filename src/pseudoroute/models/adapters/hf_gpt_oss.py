"""gpt-oss adapter for native routing with explicitly separate MXFP4 semantics."""

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


@register_adapter("hf_gpt_oss")
class HFGptOssAdapter(MoEModelAdapter):
    """Native router adapter; expert residency bytes remain MXFP4-specific."""

    architecture = "GptOssForCausalLM"

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
    ) -> HFGptOssAdapter:
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
        count = int(config.num_local_experts)
        for layer_idx, layer in enumerate(self._layers()):
            experts = layer.mlp.experts
            physical = 0
            for projection in ("gate_up_proj", "down_proj"):
                tensor = getattr(experts, projection)
                storage = getattr(tensor, "storage", None)
                if storage is not None and not callable(storage):
                    data = storage.data
                    physical += int(data.numel() * data.element_size())
                    precision = getattr(experts, f"{projection}_precision_config")
                    scale_data = precision.weight_scale.storage.data
                    physical += int(scale_data.numel() * scale_data.element_size())
                else:
                    physical += int(tensor.numel() * tensor.element_size())
                bias = getattr(experts, f"{projection}_bias")
                physical += int(bias.numel() * bias.element_size())
            if physical % count:
                raise ValueError("gpt-oss physical MXFP4 expert storage is not expert-divisible")
            for expert_idx in range(count):
                expert_bytes[ExpertKey(layer_idx, expert_idx)] = physical // count
        semantics = "biased_logit_topk_then_selected_softmax_mxfp4_expert_weights"
        return ModelSpec(
            model_id=self.model_id,
            architecture=self.architecture,
            num_layers=int(config.num_hidden_layers),
            moe_layer_indices=indices,
            num_experts_by_layer={i: int(config.num_local_experts) for i in indices},
            top_k_by_layer={i: int(config.num_experts_per_tok) for i in indices},
            hidden_size=int(config.hidden_size),
            uses_rope=True,
            pre_norm=True,
            expert_bytes=expert_bytes,
            shared_experts_by_layer={i: 0 for i in indices},
            routing_semantics_by_layer={i: semantics for i in indices},
        )

    def validate_structure(self) -> None:
        config = self._config()
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if self.architecture not in architectures:
            raise ValueError(f"hf_gpt_oss requires {self.architecture}; got {architectures}")
        quantization = getattr(config, "quantization_config", None)
        quant_method = (
            quantization.get("quant_method")
            if isinstance(quantization, dict)
            else getattr(quantization, "quant_method", None)
        )
        if quant_method != "mxfp4":
            raise ValueError("hf_gpt_oss requires the checkpoint native MXFP4 representation")
        for layer_idx, layer in enumerate(self._layers()):
            router = layer.mlp.router
            expected = (int(config.num_local_experts), int(config.hidden_size))
            if tuple(router.weight.shape) != expected or tuple(router.bias.shape) != (
                int(config.num_local_experts),
            ):
                raise ValueError(f"hf_gpt_oss: router shape mismatch at layer {layer_idx}")

    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]:
        config = self._config()
        return tuple(
            MoELayerHandle(
                layer_idx=i,
                router=cast(nn.Module, layer.mlp.router),
                experts=tuple(
                    PackedMixtralExpert(layer.mlp.experts, e)
                    for e in range(int(config.num_local_experts))
                ),
                top_k=int(config.num_experts_per_tok),
                normalization=cast(nn.Module, layer.post_attention_layernorm),
                combine_semantics=self.spec.routing_semantics_by_layer[i],
                shared_expert=None,
                score_function="topk_logits_then_selected_softmax",
                has_router_bias=True,
                shared_expert_count=0,
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
        captured: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        if trace_request is not None:
            for layer_idx, layer in enumerate(self._layers()):

                def hook(
                    _module: nn.Module,
                    _inputs: tuple[object, ...],
                    output: object,
                    *,
                    index: int = layer_idx,
                ) -> None:
                    if not isinstance(output, tuple) or len(output) != 3:
                        raise RuntimeError("gpt-oss router did not expose its native tuple")
                    logits, weights, ids = output
                    if not all(isinstance(value, Tensor) for value in output):
                        raise RuntimeError("gpt-oss route tuple contains a non-tensor")
                    captured[index] = (logits, weights, ids)

                handles.append(layer.mlp.router.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                output = cast(
                    Any,
                    self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=kv_cache,
                        use_cache=use_cache,
                        return_dict=True,
                    ),
                )
        finally:
            for handle in handles:
                handle.remove()
        traces: list[RouterTrace] = []
        if trace_request is not None:
            batch, sequence = input_ids.shape
            if batch != 1 or tuple(sorted(captured)) != self.spec.moe_layer_indices:
                raise RuntimeError("gpt-oss trace capture requires batch 1 and every MoE layer")
            for layer_idx in self.spec.moe_layer_indices:
                logits, weights, ids = captured[layer_idx]
                logits = logits.reshape(batch, sequence, -1)
                weights = weights.reshape(batch, sequence, -1)
                ids = ids.reshape(batch, sequence, -1)
                for position in range(sequence):
                    traces.append(
                        RouterTrace(
                            position,
                            layer_idx,
                            logits[:, position].detach().cpu(),
                            logits[:, position].detach().cpu(),
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
        logits, weights, ids = self._layers()[layer_idx].mlp.router(router_input)
        return RouteResult(logits, logits, ids, weights)

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
            tuple((i, cast(nn.Module, layer.mlp.router)) for i, layer in enumerate(self._layers())),
            policy=policy,
            allowed_experts=allowed_experts,
            top_k_by_layer=self.spec.top_k_by_layer,
            semantics_by_layer={
                i: NativeRouteSemantics("selected_logits_softmax", True)
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
