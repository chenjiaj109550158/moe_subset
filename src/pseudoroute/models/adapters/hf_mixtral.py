"""Exact Hugging Face Mixtral adapter using native router outputs."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.analysis.pseudo_sequences import OfflinePseudoSequence
from pseudoroute.execution.routing_policy import NaturalRoutingPolicy, RoutingPolicy
from pseudoroute.models.base import (
    ForwardResult,
    MoELayerHandle,
    MoEModelAdapter,
    RouteResult,
    TraceLevel,
    TraceRequest,
)
from pseudoroute.models.registry import register_adapter
from pseudoroute.models.tiny_moe import LayerActivationTrace
from pseudoroute.types import ExpertKey, ModelSpec, RouterTrace


class PackedMixtralExpert(nn.Module):
    """Logical view of one expert in Transformers' packed expert tensors."""

    def __init__(self, container: nn.Module, expert_idx: int) -> None:
        super().__init__()
        object.__setattr__(self, "container", container)
        self.expert_idx = expert_idx

    def extra_repr(self) -> str:
        return f"expert_idx={self.expert_idx}, packed=True"


@register_adapter("hf_mixtral")
class HFMixtralAdapter(MoEModelAdapter):
    architecture = "MixtralForCausalLM"

    def __init__(
        self,
        model: nn.Module,
        *,
        model_id: str,
        revision: str,
    ) -> None:
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
    ) -> HFMixtralAdapter:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("install pseudoroute-moe[hf] for the Mixtral adapter") from error
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        module = cast(nn.Module, model)
        module.to(torch.device(device))
        module.eval()
        return cls(module, model_id=model_id, revision=revision)

    def _config(self) -> Any:
        return cast(Any, self.model.config)

    def _layers(self) -> Any:
        base = cast(Any, self.model.model)
        return cast(Any, base.layers)

    @property
    def spec(self) -> ModelSpec:
        config = self._config()
        expert_bytes: dict[ExpertKey, int] = {}
        for layer_idx, layer in enumerate(self._layers()):
            container = layer.mlp.experts
            for expert_idx in range(int(config.num_local_experts)):
                size = sum(
                    int(parameter[expert_idx].numel() * parameter.element_size())
                    for parameter in container.parameters()
                    if parameter.ndim > 0 and parameter.shape[0] == config.num_local_experts
                )
                expert_bytes[ExpertKey(layer_idx, expert_idx)] = size
        return ModelSpec(
            model_id=self.model_id,
            architecture=self.architecture,
            num_layers=int(config.num_hidden_layers),
            moe_layer_indices=tuple(range(int(config.num_hidden_layers))),
            num_experts_by_layer={
                layer: int(config.num_local_experts)
                for layer in range(int(config.num_hidden_layers))
            },
            top_k_by_layer={
                layer: int(config.num_experts_per_tok)
                for layer in range(int(config.num_hidden_layers))
            },
            hidden_size=int(config.hidden_size),
            uses_rope=True,
            pre_norm=True,
            expert_bytes=expert_bytes,
        )

    def validate_structure(self) -> None:
        config = self._config()
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if self.architecture not in architectures:
            raise ValueError(
                f"hf_mixtral adapter requires {self.architecture}; got {architectures or 'unknown'}"
            )
        layers = self._layers()
        if len(layers) != int(config.num_hidden_layers):
            raise ValueError("hf_mixtral: decoder layer count does not match config")
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
                raise ValueError(f"hf_mixtral: missing native router at layer {layer_idx}")
            weight = layer.mlp.gate.weight
            expected = (int(config.num_local_experts), int(config.hidden_size))
            if tuple(weight.shape) != expected:
                raise ValueError(
                    f"hf_mixtral: router shape at layer {layer_idx} is {tuple(weight.shape)}, "
                    f"expected {expected}"
                )
            if int(layer.mlp.experts.num_experts) != int(config.num_local_experts):
                raise ValueError(f"hf_mixtral: expert count mismatch at layer {layer_idx}")
        if int(config.num_experts_per_tok) > int(config.num_local_experts):
            raise ValueError("hf_mixtral: top-k exceeds local expert count")

    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]:
        config = self._config()
        return tuple(
            MoELayerHandle(
                layer_idx=layer_idx,
                router=cast(nn.Module, layer.mlp.gate),
                experts=tuple(
                    PackedMixtralExpert(layer.mlp.experts, expert_idx)
                    for expert_idx in range(int(config.num_local_experts))
                ),
                top_k=int(config.num_experts_per_tok),
                normalization=cast(nn.Module, layer.post_attention_layernorm),
                combine_semantics="native_mixtral_softmax_topk_renormalized_packed",
                shared_expert=None,
                score_function="softmax",
                has_router_bias=False,
            )
            for layer_idx, layer in enumerate(self._layers())
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
        kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
            "use_cache": use_cache,
            "output_router_logits": trace_request is not None,
            "return_dict": True,
        }
        with torch.inference_mode():
            output = cast(Any, self.model(**kwargs))
        traces: list[RouterTrace] = []
        if trace_request is not None:
            router_logits = cast(tuple[Tensor, ...], output.router_logits)
            batch, sequence = input_ids.shape
            if batch != 1:
                raise ValueError("M1 trace capture supports batch size 1")
            for layer_idx, layer_logits in enumerate(router_logits):
                reshaped = layer_logits.reshape(batch, sequence, -1)
                scores = reshaped.float().softmax(dim=-1)
                topk_scores, topk_ids = scores.topk(self.spec.top_k_by_layer[layer_idx], dim=-1)
                weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
                for position in range(sequence):
                    traces.append(
                        RouterTrace(
                            token_position=position,
                            layer_idx=layer_idx,
                            raw_logits=reshaped[:, position].detach().cpu(),
                            pre_topk_scores=scores[:, position].detach().cpu(),
                            topk_ids=topk_ids[:, position].detach().cpu(),
                            topk_weights=weights[:, position].detach().cpu(),
                        )
                    )
        cache = getattr(output, "past_key_values", None) if use_cache else None
        return ForwardResult(
            logits=cast(Tensor, output.logits), traces=tuple(traces), kv_cache=cache
        )

    def capture_offline_factorial(
        self, sequence: OfflinePseudoSequence
    ) -> tuple[LayerActivationTrace, ...]:
        """Capture faithful native points; post-RoPE query is not exposed by Transformers."""
        pre_queries: dict[int, Tensor] = {}
        post_attention: dict[int, Tensor] = {}
        router_inputs: dict[int, Tensor] = {}
        handles = []
        for layer_idx, layer in enumerate(self._layers()):
            handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_idx: pre_queries.__setitem__(
                        index, cast(Tensor, output).detach().cpu()
                    )
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda _module, inputs, index=layer_idx: post_attention.__setitem__(
                        index, cast(Tensor, inputs[0]).detach().cpu()
                    )
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_idx: router_inputs.__setitem__(
                        index, cast(Tensor, output).detach().cpu()
                    )
                )
            )
        device = next(self.model.parameters()).device
        try:
            result = self.run_base_forward(
                sequence.input_ids.to(device),
                position_ids=sequence.position_ids.to(device),
                trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS),
            )
        finally:
            for handle in handles:
                handle.remove()
        traces = {(trace.token_position, trace.layer_idx): trace for trace in result.traces}
        records = []
        for position in range(sequence.boundary, sequence.boundary + sequence.horizon):
            for layer_idx in self.spec.moe_layer_indices:
                trace = traces[(position, layer_idx)]
                records.append(
                    LayerActivationTrace(
                        token_position=position,
                        layer_idx=layer_idx,
                        pre_rope_query=pre_queries[layer_idx][:, position].clone(),
                        post_rope_query=None,
                        post_attention_state=post_attention[layer_idx][:, position].clone(),
                        post_moe_state=None,
                        router_input=router_inputs[layer_idx][:, position].clone(),
                        router_logits=trace.raw_logits.clone(),
                        topk_ids=trace.topk_ids.clone(),
                        topk_weights=trace.topk_weights.clone(),
                    )
                )
        return tuple(records)

    def route_from_state(self, layer_idx: int, router_input: Tensor) -> RouteResult:
        gate = self._layers()[layer_idx].mlp.gate
        raw_logits, topk_weights, topk_ids = gate(router_input)
        scores = raw_logits.float().softmax(dim=-1)
        return RouteResult(raw_logits, scores, topk_ids, topk_weights)

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
        if allowed_experts is not None or allowed_experts_by_position is not None:
            raise NotImplementedError("constrained Hugging Face execution is not part of M3")
        if not isinstance(policy, NaturalRoutingPolicy) or not policy.authorize_natural_route():
            raise NotImplementedError("constrained routing policies begin in M3")
        return self.run_base_forward(input_ids, kv_cache=kv_cache, use_cache=use_cache)

    def clone_kv_cache_for_shadow(self, kv_cache: object) -> object:
        return copy.deepcopy(kv_cache)

    def get_expert_handle(self, key: ExpertKey) -> nn.Module:
        if key.layer_idx not in self.spec.moe_layer_indices:
            raise KeyError(key)
        count = self.spec.num_experts_by_layer[key.layer_idx]
        if key.expert_idx >= count:
            raise KeyError(key)
        return PackedMixtralExpert(self._layers()[key.layer_idx].mlp.experts, key.expert_idx)
