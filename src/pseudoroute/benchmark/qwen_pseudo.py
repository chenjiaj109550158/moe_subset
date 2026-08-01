"""Native Qwen3-MoE pseudo-embedding shadow rollout for the focused pilot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask

from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    NativeRoute,
    Qwen3MoePrefetchOps,
)
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_length,
    _cache_mutation_signature,
    _fork_cache_copy_on_write,
)

PseudoContent = Literal[
    "sampled_next_token",
    "current_token",
    "expected_top_m",
    "provided_sequence",
]
PseudoAttention = Literal["independent", "causal"]
ExpertContribution = Literal["default_vector_selected_topk_mixture", "zero"]


@dataclass(frozen=True)
class QwenPseudoVariant:
    key: str
    content: PseudoContent
    attention: PseudoAttention
    expert_contribution: ExpertContribution


@dataclass(frozen=True)
class QwenProbeCost:
    latency_seconds_measured: float
    temporary_cuda_bytes_measured: int
    output_tensor_bytes: int
    persistent_default_vector_bytes: int
    attention_queries: int
    attention_calls: int
    router_calls: int
    cpu_gpu_synchronizations: int


@dataclass(frozen=True)
class QwenPseudoProbeResult:
    variant: str
    anchors: tuple[int, ...]
    absolute_positions: tuple[int, ...]
    raw_router_logits: dict[int, Tensor]
    pre_topk_probabilities: dict[int, Tensor]
    pseudo_topk_ids: dict[int, Tensor]
    pseudo_topk_weights: dict[int, Tensor]
    aggregate_utility: dict[int, Tensor]
    subsets: dict[int, tuple[int, ...]]
    cost: QwenProbeCost
    audit: dict[str, object]


def _top_b(scores: Tensor, budget: int) -> tuple[int, ...]:
    ranking = sorted(
        range(scores.numel()),
        key=lambda expert: (-float(scores[expert]), expert),
    )
    return tuple(sorted(ranking[:budget]))


def _tensor_bytes(tensors: dict[int, Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in tensors.values())


def _rng_snapshot(device: torch.device) -> tuple[Tensor, Tensor | None]:
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu, cuda


def _rng_equal(
    before: tuple[Tensor, Tensor | None],
    after: tuple[Tensor, Tensor | None],
) -> bool:
    return torch.equal(before[0], after[0]) and (
        before[1] is None or (after[1] is not None and torch.equal(before[1], after[1]))
    )


def _restore_rng(device: torch.device, state: tuple[Tensor, Tensor | None]) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


class QwenPseudoEmbeddingProbe:
    """Run exact Qwen attention/RoPE/router on disposable shadow residuals."""

    def __init__(
        self,
        model: nn.Module,
        ops: Qwen3MoePrefetchOps,
        defaults: DefaultVectorArtifact,
        variant: QwenPseudoVariant,
        *,
        anchors: tuple[int, ...] = tuple(range(1, 9)),
        budget: int = 32,
    ) -> None:
        if ops.model is not model:
            raise ValueError("Qwen probe ops/model identity mismatch")
        if anchors != tuple(range(1, len(anchors) + 1)) or not anchors:
            raise ValueError("Qwen pseudo anchors must be contiguous from one")
        if budget < ops.top_k or budget > ops.num_experts:
            raise ValueError("Qwen pseudo budget is outside native expert bounds")
        if tuple(defaults.mean.shape) != (
            ops.num_layers,
            ops.num_experts,
            ops.hidden_size,
        ):
            raise ValueError("Qwen default-vector shape mismatch")
        if tuple(defaults.count.shape) != (ops.num_layers, ops.num_experts):
            raise ValueError("Qwen default-vector count shape mismatch")
        if variant.content == "current_token" and variant.attention == "causal":
            raise ValueError("the frozen pilot has no current-token causal variant")
        self.model = model
        self.ops = ops
        self.variant = variant
        self.anchors = anchors
        self.budget = budget
        parameter = next(model.parameters())
        self.defaults = defaults.mean.to(device=parameter.device, dtype=parameter.dtype)
        self.default_counts = defaults.count
        self.default_fingerprint = defaults.fingerprint
        self.default_definition = defaults.definition

    @property
    def _base(self) -> Any:
        return cast(Any, self.model).model

    def _initial_hidden(
        self,
        sampled_next_token_id: int,
        current_token_id: int,
        sequence: int,
        expected_embedding: Tensor | None,
        anchor_token_ids: tuple[int, ...] | None,
    ) -> Tensor:
        if self.variant.content == "expected_top_m":
            if expected_embedding is None:
                raise ValueError("expected_top_m requires a deployable sampling-step embedding")
            return expected_embedding[:, None, :].expand(-1, sequence, -1)
        if self.variant.content == "provided_sequence":
            if anchor_token_ids is None or len(anchor_token_ids) != sequence:
                raise ValueError("provided_sequence requires one token ID per pseudo anchor")
            ids = torch.tensor(
                [anchor_token_ids],
                dtype=torch.long,
                device=next(self.model.parameters()).device,
            )
            return cast(Tensor, self._base.embed_tokens(ids))
        token_id = (
            sampled_next_token_id
            if self.variant.content == "sampled_next_token"
            else current_token_id
        )
        ids = torch.full(
            (1, sequence),
            token_id,
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        return cast(Tensor, self._base.embed_tokens(ids))

    def _position_inputs(
        self,
        hidden: Tensor,
        positions: tuple[int, ...],
        shadow_cache: Cache,
    ) -> tuple[Any, tuple[Tensor, Tensor]]:
        position_ids = torch.tensor(
            positions,
            dtype=torch.long,
            device=hidden.device,
        ).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=cast(Any, self.model).config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=shadow_cache,
            position_ids=position_ids,
        )
        position_embeddings = cast(
            tuple[Tensor, Tensor],
            self._base.rotary_emb(hidden, position_ids=position_ids),
        )
        return attention_mask, position_embeddings

    def _layer_step(
        self,
        layer_idx: int,
        hidden: Tensor,
        shadow_cache: Cache,
        attention_mask: Any,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, NativeRoute, Tensor]:
        layer = cast(Any, self.ops.layers[layer_idx])
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention, _ = layer.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            past_key_values=shadow_cache,
            position_embeddings=position_embeddings,
        )
        post_attention = residual + attention
        router_input = layer.post_attention_layernorm(post_attention)
        route = self.ops.route(layer_idx, router_input.reshape(-1, self.ops.hidden_size))
        probabilities = route.logits.float().softmax(dim=-1)
        if self.variant.expert_contribution == "zero":
            contribution = torch.zeros_like(router_input)
        else:
            selected_defaults = self.defaults[layer_idx][route.ids]
            contribution = (
                (route.weights.unsqueeze(-1) * selected_defaults)
                .sum(dim=1)
                .reshape_as(router_input)
            )
        return post_attention + contribution, route, probabilities

    def _independent_rollout(
        self,
        production_cache: object,
        positions: tuple[int, ...],
        sampled_next_token_id: int,
        current_token_id: int,
        expected_embedding: Tensor | None,
        anchor_token_ids: tuple[int, ...] | None,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        caches = [_fork_cache_copy_on_write(production_cache) for _ in self.anchors]
        hidden = [
            self._initial_hidden(
                sampled_next_token_id,
                current_token_id,
                1,
                expected_embedding,
                ((anchor_token_ids[anchor_idx],) if anchor_token_ids is not None else None),
            )
            for anchor_idx in range(len(self.anchors))
        ]
        masks_and_positions = [
            self._position_inputs(value, (position,), cast(Cache, cache))
            for value, position, cache in zip(hidden, positions, caches, strict=True)
        ]
        logits: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        probabilities: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        topk_ids: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        topk_weights: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        for layer_idx in range(self.ops.num_layers):
            for anchor_idx in range(len(self.anchors)):
                mask, position_embeddings = masks_and_positions[anchor_idx]
                hidden[anchor_idx], route, full_probs = self._layer_step(
                    layer_idx,
                    hidden[anchor_idx],
                    cast(Cache, caches[anchor_idx]),
                    mask,
                    position_embeddings,
                )
                logits[layer_idx].append(route.logits[0].detach().float().cpu())
                probabilities[layer_idx].append(full_probs[0].detach().cpu())
                topk_ids[layer_idx].append(route.ids[0].detach().cpu())
                topk_weights[layer_idx].append(route.weights[0].detach().float().cpu())
        return (
            {layer: torch.stack(values) for layer, values in logits.items()},
            {layer: torch.stack(values) for layer, values in probabilities.items()},
            {layer: torch.stack(values) for layer, values in topk_ids.items()},
            {layer: torch.stack(values) for layer, values in topk_weights.items()},
        )

    def _causal_rollout(
        self,
        production_cache: object,
        positions: tuple[int, ...],
        sampled_next_token_id: int,
        current_token_id: int,
        expected_embedding: Tensor | None,
        anchor_token_ids: tuple[int, ...] | None,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        cache = _fork_cache_copy_on_write(production_cache)
        hidden = self._initial_hidden(
            sampled_next_token_id,
            current_token_id,
            len(self.anchors),
            expected_embedding,
            anchor_token_ids,
        )
        attention_mask, position_embeddings = self._position_inputs(
            hidden,
            positions,
            cast(Cache, cache),
        )
        logits: dict[int, Tensor] = {}
        probabilities: dict[int, Tensor] = {}
        topk_ids: dict[int, Tensor] = {}
        topk_weights: dict[int, Tensor] = {}
        for layer_idx in range(self.ops.num_layers):
            hidden, route, full_probs = self._layer_step(
                layer_idx,
                hidden,
                cast(Cache, cache),
                attention_mask,
                position_embeddings,
            )
            logits[layer_idx] = route.logits.detach().float().cpu()
            probabilities[layer_idx] = full_probs.detach().cpu()
            topk_ids[layer_idx] = route.ids.detach().cpu()
            topk_weights[layer_idx] = route.weights.detach().float().cpu()
        return logits, probabilities, topk_ids, topk_weights

    def predict(
        self,
        production_cache: object,
        *,
        sampled_next_token_id: int,
        current_token_id: int,
        next_token_logits: Tensor | None = None,
        expected_top_m: int = 8,
        anchor_token_ids: tuple[int, ...] | None = None,
    ) -> QwenPseudoProbeResult:
        """Predict one H-window subset without mutating production state or RNG."""
        device = next(self.model.parameters()).device
        boundary_length = _cache_length(production_cache)
        positions = tuple(boundary_length + anchor - 1 for anchor in self.anchors)
        cache_before = _cache_mutation_signature(production_cache)
        rng_before = _rng_snapshot(device)
        expected_embedding = None
        if self.variant.content == "expected_top_m":
            if next_token_logits is None or tuple(next_token_logits.shape[:1]) != (1,):
                raise ValueError("expected_top_m requires one row of next-token logits")
            if expected_top_m < 1 or expected_top_m > next_token_logits.shape[-1]:
                raise ValueError("expected_top_m is outside the vocabulary")
        elif next_token_logits is not None:
            raise ValueError("next_token_logits are only valid for expected_top_m")
        if self.variant.content == "provided_sequence":
            if anchor_token_ids is None or len(anchor_token_ids) != len(self.anchors):
                raise ValueError("provided_sequence requires one token ID per pseudo anchor")
        elif anchor_token_ids is not None:
            raise ValueError("anchor_token_ids are only valid for provided_sequence")
        sync_count = 0
        allocated_before = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            sync_count += 1
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        rng_unchanged = False
        cache_unchanged = False
        try:
            with torch.inference_mode():
                if next_token_logits is not None:
                    selected_logits, selected_ids = next_token_logits.float().topk(
                        expected_top_m,
                        dim=-1,
                    )
                    weights = selected_logits.softmax(dim=-1).to(
                        dtype=next(self.model.parameters()).dtype
                    )
                    embeddings = self._base.embed_tokens(selected_ids.to(device))
                    expected_embedding = (weights.to(device).unsqueeze(-1) * embeddings).sum(dim=1)
                if self.variant.attention == "independent":
                    logits, probabilities, topk_ids, topk_weights = self._independent_rollout(
                        production_cache,
                        positions,
                        sampled_next_token_id,
                        current_token_id,
                        expected_embedding,
                        anchor_token_ids,
                    )
                else:
                    logits, probabilities, topk_ids, topk_weights = self._causal_rollout(
                        production_cache,
                        positions,
                        sampled_next_token_id,
                        current_token_id,
                        expected_embedding,
                        anchor_token_ids,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                sync_count += 1
            elapsed = time.perf_counter() - started
            cache_unchanged = (
                _cache_length(production_cache) == boundary_length
                and _cache_mutation_signature(production_cache) == cache_before
            )
            rng_unchanged = _rng_equal(rng_before, _rng_snapshot(device))
        finally:
            _restore_rng(device, rng_before)
        if not cache_unchanged:
            raise RuntimeError("Qwen pseudo probe mutated the production cache")
        if not rng_unchanged:
            raise RuntimeError("Qwen pseudo probe changed generation RNG")
        utility = {layer: values.double().sum(dim=0) for layer, values in probabilities.items()}
        subsets = {layer: _top_b(values, self.budget) for layer, values in utility.items()}
        output_bytes = sum(
            _tensor_bytes(values) for values in (logits, probabilities, topk_ids, topk_weights)
        )
        temporary_cuda = (
            max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
            if device.type == "cuda"
            else 0
        )
        attention_calls = (
            self.ops.num_layers * len(self.anchors)
            if self.variant.attention == "independent"
            else self.ops.num_layers
        )
        return QwenPseudoProbeResult(
            self.variant.key,
            self.anchors,
            positions,
            logits,
            probabilities,
            topk_ids,
            topk_weights,
            utility,
            subsets,
            QwenProbeCost(
                latency_seconds_measured=elapsed,
                temporary_cuda_bytes_measured=temporary_cuda,
                output_tensor_bytes=output_bytes,
                persistent_default_vector_bytes=(
                    self.defaults.numel() * self.defaults.element_size()
                ),
                attention_queries=self.ops.num_layers * len(self.anchors),
                attention_calls=attention_calls,
                router_calls=attention_calls,
                cpu_gpu_synchronizations=sync_count,
            ),
            {
                "information_regime": "online_post_sample",
                "deployable_inputs": (
                    "sampling_step_logits_and_current_policy_production_cache_only"
                    if self.variant.content == "expected_top_m"
                    else (
                        "caller_supplied_anchor_token_ids_and_current_policy_production_cache"
                        if self.variant.content == "provided_sequence"
                        else "sampled_next_token_and_current_policy_production_cache_only"
                    )
                ),
                "anchor_token_ids_supplied": anchor_token_ids is not None,
                "expected_top_m": (
                    expected_top_m if self.variant.content == "expected_top_m" else None
                ),
                "forbidden_inputs_present": False,
                "production_cache_sequence_length_before": boundary_length,
                "production_cache_sequence_length_after": _cache_length(production_cache),
                "production_cache_signature_unchanged": True,
                "production_rng_unchanged": True,
                "shadow_cache_discarded": True,
                "native_attention": True,
                "native_rope": True,
                "native_router": True,
                "native_top_k": self.ops.top_k,
                "native_norm_topk_prob": bool(cast(Any, self.model).config.norm_topk_prob),
                "shared_experts": 0,
                "layer_scoped_expert_ids": True,
                "default_definition": self.default_definition,
                "default_fingerprint": self.default_fingerprint,
                "unobserved_default_pairs": int((self.default_counts == 0).sum()),
            },
        )
