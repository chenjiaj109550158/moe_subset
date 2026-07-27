"""Pinned DeepSeek-V2 adapter with an isolated Transformers compatibility shim."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.execution.routing_policy import ExecutedRoute, RoutingPolicy
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

DEEPSEEK_V2_LITE_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"


@contextmanager
def transformers_fx_compatibility_shim() -> Iterator[None]:
    """Temporarily restore only the removed query used by reviewed pinned code."""
    import transformers.utils as utils
    import transformers.utils.import_utils as import_utils

    sentinel = object()
    previous = getattr(utils, "is_torch_fx_available", sentinel)
    previous_import = getattr(import_utils, "is_torch_fx_available", sentinel)
    if previous is sentinel:
        utils.__dict__["is_torch_fx_available"] = lambda: False
    if previous_import is sentinel:
        import_utils.__dict__["is_torch_fx_available"] = lambda: False
    try:
        yield
    finally:
        if previous is sentinel:
            del utils.__dict__["is_torch_fx_available"]
        else:
            utils.__dict__["is_torch_fx_available"] = previous
        if previous_import is sentinel:
            del import_utils.__dict__["is_torch_fx_available"]
        else:
            import_utils.__dict__["is_torch_fx_available"] = previous_import


@contextmanager
def transformers_cache_compatibility_shim() -> Iterator[None]:
    """Temporarily restore the unbounded-cache query used by pinned remote code."""
    from transformers import DynamicCache

    sentinel = object()
    previous = getattr(DynamicCache, "get_usable_length", sentinel)
    if previous is sentinel:

        def get_usable_length(cache: Any, _new_seq_length: int, layer_idx: int = 0) -> int:
            return int(cache.get_seq_length(layer_idx))

        type.__setattr__(DynamicCache, "get_usable_length", get_usable_length)
    try:
        yield
    finally:
        if previous is sentinel:
            type.__delattr__(DynamicCache, "get_usable_length")
        else:
            type.__setattr__(DynamicCache, "get_usable_length", previous)


@register_adapter("hf_deepseek_v2")
class HFDeepseekV2Adapter(MoEModelAdapter):
    architecture = "DeepseekV2ForCausalLM"

    def __init__(self, model: nn.Module, *, model_id: str, revision: str) -> None:
        if revision != DEEPSEEK_V2_LITE_REVISION:
            raise ValueError("DeepSeek remote code is allowed only at the reviewed pinned revision")
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
    ) -> HFDeepseekV2Adapter:
        if revision != DEEPSEEK_V2_LITE_REVISION:
            raise ValueError("DeepSeek remote code is allowed only at the reviewed pinned revision")
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, object] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
            "dtype": "auto",
        }
        if device_map is not None:
            kwargs.update(device_map=device_map, max_memory=max_memory)
        with transformers_fx_compatibility_shim():
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

    def _moe_layers(self) -> tuple[tuple[int, Any], ...]:
        config = self._config()
        first = int(config.first_k_dense_replace)
        frequency = int(config.moe_layer_freq)
        return tuple(
            (i, layer)
            for i, layer in enumerate(self._layers())
            if i >= first and i % frequency == 0 and hasattr(layer.mlp, "gate")
        )

    @property
    def input_device(self) -> torch.device:
        device = cast(Any, self.model).model.embed_tokens.weight.device
        return cast(torch.device, device)

    @cached_property
    def spec(self) -> ModelSpec:
        config = self._config()
        indices = tuple(i for i, _ in self._moe_layers())
        expert_bytes: dict[ExpertKey, int] = {}
        for layer_idx, layer in self._moe_layers():
            for expert_idx, expert in enumerate(layer.mlp.experts):
                expert_bytes[ExpertKey(layer_idx, expert_idx)] = sum(
                    int(parameter.numel() * parameter.element_size())
                    for parameter in expert.parameters()
                )
        normalized = bool(config.norm_topk_prob)
        scale = float(config.routed_scaling_factor)
        semantics = (
            "softmax_float32_topk_"
            + ("renormalized" if normalized else f"scaled_{scale:g}")
            + "_plus_always_active_shared_experts"
        )
        return ModelSpec(
            model_id=self.model_id,
            architecture=self.architecture,
            num_layers=int(config.num_hidden_layers),
            moe_layer_indices=indices,
            num_experts_by_layer={i: int(config.n_routed_experts) for i in indices},
            top_k_by_layer={i: int(config.num_experts_per_tok) for i in indices},
            hidden_size=int(config.hidden_size),
            uses_rope=True,
            pre_norm=True,
            expert_bytes=expert_bytes,
            shared_experts_by_layer={i: int(config.n_shared_experts or 0) for i in indices},
            routing_semantics_by_layer={i: semantics for i in indices},
        )

    def validate_structure(self) -> None:
        config = self._config()
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if self.architecture not in architectures:
            raise ValueError(f"hf_deepseek_v2 requires {self.architecture}; got {architectures}")
        expected_indices = tuple(
            i
            for i in range(int(config.num_hidden_layers))
            if i >= int(config.first_k_dense_replace) and i % int(config.moe_layer_freq) == 0
        )
        actual_indices = tuple(i for i, _ in self._moe_layers())
        if actual_indices != expected_indices:
            raise ValueError(
                f"hf_deepseek_v2: MoE layer indices {actual_indices} != {expected_indices}"
            )
        for layer_idx, layer in self._moe_layers():
            if tuple(layer.mlp.gate.weight.shape) != (
                int(config.n_routed_experts),
                int(config.hidden_size),
            ):
                raise ValueError(f"hf_deepseek_v2: router shape mismatch at layer {layer_idx}")
            if len(layer.mlp.experts) != int(config.n_routed_experts):
                raise ValueError(f"hf_deepseek_v2: routed expert count mismatch at {layer_idx}")
            if int(config.n_shared_experts or 0) and not hasattr(layer.mlp, "shared_experts"):
                raise ValueError(f"hf_deepseek_v2: missing shared experts at layer {layer_idx}")
        if str(config.scoring_func) != "softmax" or str(config.topk_method) != "greedy":
            raise ValueError(
                "hf_deepseek_v2 adapter currently validates softmax greedy routing only"
            )

    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]:
        config = self._config()
        return tuple(
            MoELayerHandle(
                layer_idx=i,
                router=cast(nn.Module, layer.mlp.gate),
                experts=tuple(cast(nn.Module, expert) for expert in layer.mlp.experts),
                top_k=int(config.num_experts_per_tok),
                normalization=cast(nn.Module, layer.post_attention_layernorm),
                combine_semantics=self.spec.routing_semantics_by_layer[i],
                shared_expert=cast(nn.Module, getattr(layer.mlp, "shared_experts", None)),
                score_function="softmax_float32",
                has_router_bias=False,
                shared_expert_count=int(config.n_shared_experts or 0),
            )
            for i, layer in self._moe_layers()
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
        captured: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        if trace_request is not None:
            for layer_idx, layer in self._moe_layers():

                def hook(
                    module: nn.Module,
                    inputs: tuple[object, ...],
                    output: object,
                    *,
                    index: int = layer_idx,
                ) -> None:
                    if not inputs or not isinstance(inputs[0], Tensor):
                        raise RuntimeError("DeepSeek gate did not expose router input")
                    if not isinstance(output, tuple) or len(output) != 3:
                        raise RuntimeError("DeepSeek gate did not expose native route tuple")
                    ids, weights, _aux = output
                    if not isinstance(ids, Tensor) or not isinstance(weights, Tensor):
                        raise RuntimeError("DeepSeek native route values are not tensors")
                    state = inputs[0].reshape(-1, inputs[0].shape[-1])
                    weight = cast(Tensor, module.weight)
                    logits = torch.nn.functional.linear(state.float(), weight.float())
                    scores = logits.softmax(dim=-1, dtype=torch.float32)
                    captured[index] = (logits, scores, ids, weights)

                handles.append(layer.mlp.gate.register_forward_hook(hook))
        try:
            past_length = 0
            if kv_cache is not None:
                try:
                    past_length = int(cast(Any, kv_cache)[0][0].shape[-2])
                except (IndexError, TypeError, AttributeError):
                    getter = getattr(kv_cache, "get_seq_length", None)
                    past_length = int(getter()) if getter is not None else 0
            attention_mask = torch.ones(
                (input_ids.shape[0], past_length + input_ids.shape[1]),
                dtype=torch.long,
                device=input_ids.device,
            )
            actual_cache = kv_cache
            if use_cache and actual_cache is None:
                from transformers import DynamicCache

                actual_cache = DynamicCache()
            with transformers_cache_compatibility_shim(), torch.inference_mode():
                output = cast(
                    Any,
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=actual_cache,
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
            if batch != 1:
                raise ValueError("trained trace capture supports batch size 1")
            if tuple(sorted(captured)) != self.spec.moe_layer_indices:
                raise RuntimeError("DeepSeek trace did not capture every MoE layer")
            for layer_idx in self.spec.moe_layer_indices:
                logits, scores, ids, weights = captured[layer_idx]
                logits = logits.reshape(batch, sequence, -1)
                scores = scores.reshape(batch, sequence, -1)
                ids = ids.reshape(batch, sequence, -1)
                weights = weights.reshape(batch, sequence, -1)
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
        layer = dict(self._moe_layers()).get(layer_idx)
        if layer is None:
            raise KeyError(layer_idx)
        native_input = router_input.unsqueeze(1) if router_input.ndim == 2 else router_input
        ids, weights, _aux = layer.mlp.gate(native_input)
        state = router_input.reshape(-1, router_input.shape[-1])
        logits = torch.nn.functional.linear(state.float(), layer.mlp.gate.weight.float())
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        return RouteResult(logits, scores, ids, weights)

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
        config = self._config()
        records: list[ExecutedRoute] = []
        handles = install_tuple_router_hooks(
            tuple((i, cast(nn.Module, layer.mlp.gate)) for i, layer in self._moe_layers()),
            policy=policy,
            allowed_experts=allowed_experts,
            top_k_by_layer=self.spec.top_k_by_layer,
            semantics_by_layer={
                i: NativeRouteSemantics(
                    "softmax_float32",
                    bool(config.norm_topk_prob),
                    float(config.routed_scaling_factor),
                )
                for i in self.spec.moe_layer_indices
            },
            records=records,
            output_order="ids_weights_aux",
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
        layer = dict(self._moe_layers()).get(key.layer_idx)
        if layer is None or key.expert_idx >= self.spec.num_experts_by_layer.get(key.layer_idx, 0):
            raise KeyError(key)
        return cast(nn.Module, layer.mlp.experts[key.expert_idx])
