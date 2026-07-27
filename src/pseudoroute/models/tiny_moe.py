"""Deterministic decoder-only tiny MoE runnable on CPU or an accelerator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from pseudoroute.config import TinyModelConfig
from pseudoroute.execution.routing_policy import (
    ExecutedRoute,
    RoutingPolicy,
    TokenRoutingContext,
)
from pseudoroute.types import ExpertKey, ModelSpec, RouterTrace


@dataclass(frozen=True)
class LayerActivationTrace:
    token_position: int
    layer_idx: int
    pre_rope_query: Tensor | None
    post_rope_query: Tensor | None
    post_attention_state: Tensor | None
    post_moe_state: Tensor | None
    router_input: Tensor | None
    router_logits: Tensor
    topk_ids: Tensor
    topk_weights: Tensor


@dataclass(frozen=True)
class TinyMoEOutput:
    logits: Tensor
    traces: tuple[RouterTrace, ...]
    executed_routes: tuple[ExecutedRoute, ...] = ()
    activations: tuple[LayerActivationTrace, ...] = ()


class RotaryEmbedding(nn.Module):
    def forward(self, value: Tensor, positions: Tensor) -> Tensor:
        dimension = value.shape[-1]
        frequencies = torch.arange(0, dimension, 2, device=value.device, dtype=value.dtype)
        frequencies = 1.0 / (10000 ** (frequencies / dimension))
        angles = positions.to(value.dtype).view(1, 1, -1, 1) * frequencies.view(1, 1, 1, -1)
        even, odd = value[..., 0::2], value[..., 1::2]
        rotated = torch.stack(
            (even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()),
            dim=-1,
        )
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, use_rope: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RotaryEmbedding() if use_rope else None

    def forward(
        self, hidden: Tensor, *, position_ids: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, sequence, hidden_size = hidden.shape
        qkv = self.qkv(hidden).view(batch, sequence, 3, self.num_heads, self.head_dim)
        query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        pre_rope_query = query
        if self.rope is not None:
            positions = (
                torch.arange(sequence, device=hidden.device)
                if position_ids is None
                else position_ids[0]
            )
            query, key = self.rope(query, positions), self.rope(key, positions)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal = torch.ones(sequence, sequence, device=hidden.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        attended = scores.softmax(dim=-1) @ value
        output = cast(
            Tensor, self.output(attended.transpose(1, 2).reshape(batch, sequence, hidden_size))
        )
        return output, pre_rope_query, query


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down = nn.Linear(expert_hidden_size, hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return cast(Tensor, self.down(torch.nn.functional.silu(self.up(hidden))))


class TinyMoELayer(nn.Module):
    def __init__(self, config: TinyModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.top_k = config.top_k
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config.hidden_size, config.num_heads, config.use_rope)
        self.router_norm = nn.LayerNorm(config.hidden_size)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=True)
        self.experts = nn.ModuleList(
            ExpertMLP(config.hidden_size, config.expert_hidden_size)
            for _ in range(config.num_experts)
        )
        self.shared_expert = (
            ExpertMLP(config.hidden_size, config.expert_hidden_size)
            if config.shared_expert
            else None
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        capture_trace: bool,
        capture_activations: bool = False,
        position_ids: Tensor | None = None,
        policy: RoutingPolicy | None = None,
        allowed_experts: frozenset[ExpertKey] | None = None,
        allowed_experts_by_position: dict[int, frozenset[ExpertKey]] | None = None,
    ) -> tuple[
        Tensor, RouterTrace | None, tuple[ExecutedRoute, ...], tuple[LayerActivationTrace, ...]
    ]:
        attention_output, pre_rope_query, post_rope_query = self.attention(
            self.attention_norm(hidden), position_ids=position_ids
        )
        hidden = hidden + attention_output
        post_attention_state = hidden
        router_input = self.router_norm(hidden)
        raw_logits = self.router(router_input)
        scores = raw_logits.softmax(dim=-1)
        topk_scores, topk_ids = scores.topk(self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        records: list[ExecutedRoute] = []
        effective_weights: Tensor | None = None
        if policy is not None:
            if hidden.shape[0] != 1:
                raise ValueError("M3 policy execution supports batch size 1")
            effective_weights = torch.zeros_like(scores)
            for position in range(hidden.shape[1]):
                record = policy.choose(
                    self.layer_idx,
                    raw_logits[:, position],
                    topk_ids[:, position],
                    TokenRoutingContext(
                        position,
                        (allowed_experts_by_position or {}).get(position, allowed_experts),
                    ),
                )
                records.append(record)
                if record.executed_topk_ids.numel():
                    effective_weights[:, position].scatter_add_(
                        -1, record.executed_topk_ids, record.executed_topk_weights
                    )
        combined = torch.zeros_like(hidden)
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(router_input)
            if effective_weights is None:
                weights = torch.where(
                    topk_ids == expert_idx, topk_weights, torch.zeros_like(topk_weights)
                ).sum(dim=-1, keepdim=True)
            else:
                weights = effective_weights[..., expert_idx : expert_idx + 1]
            combined = combined + weights * expert_output
        if self.shared_expert is not None:
            combined = combined + self.shared_expert(router_input)
        trace = None
        if capture_trace:
            trace = RouterTrace(
                token_position=-1,
                layer_idx=self.layer_idx,
                raw_logits=raw_logits.detach().clone(),
                pre_topk_scores=scores.detach().clone(),
                topk_ids=topk_ids.detach().clone(),
                topk_weights=topk_weights.detach().clone(),
            )
        activations: tuple[LayerActivationTrace, ...] = ()
        if capture_activations:
            activations = tuple(
                LayerActivationTrace(
                    token_position=position,
                    layer_idx=self.layer_idx,
                    pre_rope_query=pre_rope_query[:, :, position].detach().clone(),
                    post_rope_query=post_rope_query[:, :, position].detach().clone(),
                    post_attention_state=post_attention_state[:, position].detach().clone(),
                    post_moe_state=None,
                    router_input=router_input[:, position].detach().clone(),
                    router_logits=raw_logits[:, position].detach().clone(),
                    topk_ids=topk_ids[:, position].detach().clone(),
                    topk_weights=topk_weights[:, position].detach().clone(),
                )
                for position in range(hidden.shape[1])
            )
        layer_output = hidden + combined
        if capture_activations:
            activations = tuple(
                LayerActivationTrace(
                    token_position=record.token_position,
                    layer_idx=record.layer_idx,
                    pre_rope_query=record.pre_rope_query,
                    post_rope_query=record.post_rope_query,
                    post_attention_state=record.post_attention_state,
                    post_moe_state=layer_output[:, record.token_position].detach().clone(),
                    router_input=record.router_input,
                    router_logits=record.router_logits,
                    topk_ids=record.topk_ids,
                    topk_weights=record.topk_weights,
                )
                for record in activations
            )
        return layer_output, trace, tuple(records), activations


class TinyMoE(nn.Module):
    """A deliberately small causal model with exact, explicit top-k routing."""

    def __init__(
        self, config: TinyModelConfig, *, seed: int, device: torch.device | str = "cpu"
    ) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyMoELayer(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.to(device)
        self.eval()

    def forward(
        self,
        input_ids: Tensor,
        *,
        capture_trace: bool = True,
        capture_activations: bool = False,
        position_ids: Tensor | None = None,
        policy: RoutingPolicy | None = None,
        allowed_experts: frozenset[ExpertKey] | None = None,
        allowed_experts_by_position: dict[int, frozenset[ExpertKey]] | None = None,
    ) -> TinyMoEOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids")
        hidden = self.embedding(input_ids)
        traces: list[RouterTrace] = []
        executed_routes: list[ExecutedRoute] = []
        activations: list[LayerActivationTrace] = []
        for layer in self.layers:
            hidden, trace, layer_records, layer_activations = layer(
                hidden,
                capture_trace=capture_trace,
                capture_activations=capture_activations,
                position_ids=position_ids,
                policy=policy,
                allowed_experts=allowed_experts,
                allowed_experts_by_position=allowed_experts_by_position,
            )
            executed_routes.extend(layer_records)
            activations.extend(layer_activations)
            if trace is not None:
                for position in range(input_ids.shape[1]):
                    traces.append(
                        RouterTrace(
                            token_position=position,
                            layer_idx=trace.layer_idx,
                            raw_logits=trace.raw_logits[:, position].clone(),
                            pre_topk_scores=trace.pre_topk_scores[:, position].clone(),
                            topk_ids=trace.topk_ids[:, position].clone(),
                            topk_weights=trace.topk_weights[:, position].clone(),
                        )
                    )
        return TinyMoEOutput(
            self.lm_head(self.final_norm(hidden)),
            tuple(traces),
            tuple(executed_routes),
            tuple(activations),
        )

    @torch.inference_mode()
    def generate(
        self, input_ids: Tensor, *, max_new_tokens: int, eos_token_id: int | None = None
    ) -> Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            next_token = (
                self(generated, capture_trace=False).logits[:, -1].argmax(dim=-1, keepdim=True)
            )
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
        return generated

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="tiny_moe",
            architecture="deterministic_tiny_moe",
            num_layers=self.config.num_layers,
            moe_layer_indices=tuple(range(self.config.num_layers)),
            num_experts_by_layer={
                layer: self.config.num_experts for layer in range(self.config.num_layers)
            },
            top_k_by_layer={layer: self.config.top_k for layer in range(self.config.num_layers)},
            hidden_size=self.config.hidden_size,
            uses_rope=self.config.use_rope,
            pre_norm=True,
            expert_bytes={
                ExpertKey(layer, expert): self.config.synthetic_expert_bytes
                for layer in range(self.config.num_layers)
                for expert in range(self.config.num_experts)
            },
        )

    def manifest(self) -> dict[str, object]:
        spec = self.spec
        return {
            "model_id": spec.model_id,
            "architecture": spec.architecture,
            "num_layers": spec.num_layers,
            "moe_layer_indices": list(spec.moe_layer_indices),
            "num_experts_by_layer": {str(k): v for k, v in spec.num_experts_by_layer.items()},
            "top_k_by_layer": {str(k): v for k, v in spec.top_k_by_layer.items()},
            "hidden_size": spec.hidden_size,
            "uses_rope": spec.uses_rope,
            "pre_norm": spec.pre_norm,
            "expert_bytes": {
                f"{key.layer_idx}:{key.expert_idx}": size for key, size in spec.expert_bytes.items()
            },
        }
