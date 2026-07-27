"""Exact Hugging Face OLMoE adapter using native router outputs."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.execution.routing_policy import NaturalRoutingPolicy, RoutingPolicy
from pseudoroute.models.adapters.hf_mixtral import PackedMixtralExpert
from pseudoroute.models.base import (
    ForwardResult,
    MoELayerHandle,
    MoEModelAdapter,
    RouteResult,
    TraceRequest,
)
from pseudoroute.models.registry import register_adapter
from pseudoroute.types import ExpertKey, ModelSpec, RouterTrace


@register_adapter("hf_olmoe")
class HFOlmoeAdapter(MoEModelAdapter):
    """Native-semantic adapter for Transformers ``OlmoeForCausalLM``."""

    architecture = "OlmoeForCausalLM"

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
    ) -> HFOlmoeAdapter:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("install pseudoroute-moe[hf] for the OLMoE adapter") from error
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
        return cast(Any, cast(Any, self.model).model.layers)

    @property
    def spec(self) -> ModelSpec:
        config = self._config()
        expert_bytes: dict[ExpertKey, int] = {}
        for layer_idx, layer in enumerate(self._layers()):
            for expert_idx in range(int(config.num_experts)):
                size = sum(
                    int(parameter[expert_idx].numel() * parameter.element_size())
                    for parameter in layer.mlp.experts.parameters()
                    if parameter.ndim > 0 and parameter.shape[0] == config.num_experts
                )
                expert_bytes[ExpertKey(layer_idx, expert_idx)] = size
        return ModelSpec(
            model_id=self.model_id,
            architecture=self.architecture,
            num_layers=int(config.num_hidden_layers),
            moe_layer_indices=tuple(range(int(config.num_hidden_layers))),
            num_experts_by_layer={
                layer: int(config.num_experts) for layer in range(int(config.num_hidden_layers))
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
                f"hf_olmoe adapter requires {self.architecture}; got {architectures or 'unknown'}"
            )
        if len(self._layers()) != int(config.num_hidden_layers):
            raise ValueError("hf_olmoe: decoder layer count does not match config")
        if int(config.num_experts_per_tok) > int(config.num_experts):
            raise ValueError("hf_olmoe: top-k exceeds local expert count")
        for layer_idx, layer in enumerate(self._layers()):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                raise ValueError(f"hf_olmoe: missing native router at layer {layer_idx}")
            expected = (int(config.num_experts), int(config.hidden_size))
            if tuple(gate.weight.shape) != expected:
                raise ValueError(
                    f"hf_olmoe: router shape at layer {layer_idx} is "
                    f"{tuple(gate.weight.shape)}, expected {expected}"
                )
            experts = layer.mlp.experts
            if int(experts.num_experts) != int(config.num_experts):
                raise ValueError(f"hf_olmoe: expert count mismatch at layer {layer_idx}")

    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]:
        config = self._config()
        normalized = bool(config.norm_topk_prob)
        return tuple(
            MoELayerHandle(
                layer_idx=layer_idx,
                router=cast(nn.Module, layer.mlp.gate),
                experts=tuple(
                    PackedMixtralExpert(layer.mlp.experts, expert_idx)
                    for expert_idx in range(int(config.num_experts))
                ),
                top_k=int(config.num_experts_per_tok),
                normalization=cast(nn.Module, layer.post_attention_layernorm),
                combine_semantics=(
                    "native_olmoe_float32_softmax_topk_"
                    + ("renormalized" if normalized else "not_renormalized")
                ),
                shared_expert=None,
                score_function="softmax_float32",
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
            for layer_idx, layer_logits in enumerate(
                cast(tuple[Tensor, ...], output.router_logits)
            ):
                logits = layer_logits.reshape(batch, sequence, -1)
                scores = logits.float().softmax(dim=-1)
                weights, ids = scores.topk(self.spec.top_k_by_layer[layer_idx], dim=-1)
                if bool(self._config().norm_topk_prob):
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                weights = weights.to(logits.dtype)
                for position in range(sequence):
                    traces.append(
                        RouterTrace(
                            token_position=position,
                            layer_idx=layer_idx,
                            raw_logits=logits[:, position].detach().cpu(),
                            pre_topk_scores=scores[:, position].detach().cpu(),
                            topk_ids=ids[:, position].detach().cpu(),
                            topk_weights=weights[:, position].detach().cpu(),
                        )
                    )
        return ForwardResult(
            logits=cast(Tensor, output.logits),
            traces=tuple(traces),
            kv_cache=getattr(output, "past_key_values", None) if use_cache else None,
        )

    def route_from_state(self, layer_idx: int, router_input: Tensor) -> RouteResult:
        raw_logits, weights, ids = self._layers()[layer_idx].mlp.gate(router_input)
        scores = raw_logits.float().softmax(dim=-1)
        return RouteResult(raw_logits, scores, ids, weights)

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
            raise NotImplementedError(
                "constrained OLMoE execution requires a validated native hook"
            )
        if not isinstance(policy, NaturalRoutingPolicy) or not policy.authorize_natural_route():
            raise NotImplementedError("only natural OLMoE routing is currently validated")
        return self.run_base_forward(input_ids, kv_cache=kv_cache, use_cache=use_cache)

    def clone_kv_cache_for_shadow(self, kv_cache: object) -> object:
        return copy.deepcopy(kv_cache)

    def get_expert_handle(self, key: ExpertKey) -> nn.Module:
        if key.layer_idx not in self.spec.moe_layer_indices:
            raise KeyError(key)
        if key.expert_idx >= self.spec.num_experts_by_layer[key.layer_idx]:
            raise KeyError(key)
        return PackedMixtralExpert(self._layers()[key.layer_idx].mlp.experts, key.expert_idx)
