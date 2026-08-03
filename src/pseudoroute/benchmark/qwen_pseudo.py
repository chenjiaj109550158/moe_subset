"""Native Qwen3-MoE pseudo-embedding shadow rollout for the focused pilot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask

from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    MoeOutputCaptureContext,
    NativeRoute,
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
    masked_route_from_natural,
    moe_output_bank,
    moe_router_input_bank,
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
    "sampled_then_expected_top_m",
    "self_greedy",
    "self_expected_top_m",
    "self_topk_particles",
    "provided_sequence",
]
PseudoAttention = Literal["independent", "causal"]
ExpectedEmbeddingNorm = Literal["raw", "sampled_token"]
ExpertContribution = Literal[
    "default_vector_selected_topk_mixture",
    "native_expert_execution",
    "provided_residual",
    "zero",
]
StateCorrection = Literal["none", "recent_linear_norm"]
StateRetrievalTarget = Literal["none", "router_input", "moe_residual"]
StateRetrievalMix = Literal["none", "additive_norm", "replace_norm"]
MidlayerSelfConditioning = Literal["none", "greedy_shift", "expected_top8_shift"]
DiagnosticStatePatch = Literal[
    "none",
    "oracle_attention_output",
    "oracle_moe_residual",
    "oracle_attention_and_moe",
    "oracle_hidden_after_layer",
]


@dataclass(frozen=True)
class QwenPseudoVariant:
    key: str
    content: PseudoContent
    attention: PseudoAttention
    expert_contribution: ExpertContribution
    expected_embedding_norm: ExpectedEmbeddingNorm = "raw"
    particle_count: int = 1
    hidden_state_correction: StateCorrection = "none"
    residual_correction: StateCorrection = "none"
    correction_anchor_coefficients: tuple[float, ...] | None = None
    correction_max_relative_delta_norm: float | None = None
    state_retrieval_target: StateRetrievalTarget = "none"
    state_retrieval_mix: StateRetrievalMix = "none"
    midlayer_self_conditioning: MidlayerSelfConditioning = "none"
    midlayer_refresh_after_layer: int | None = None
    midlayer_max_relative_embedding_delta_norm: float | None = None
    diagnostic_state_patch: DiagnosticStatePatch = "none"
    diagnostic_hidden_after_layer: int | None = None


@dataclass(frozen=True)
class QwenNaturalComponentState:
    """Full-expert future states captured on a disposable native Qwen cache."""

    attention_outputs: Tensor
    router_inputs: Tensor
    moe_residuals: Tensor
    layer_outputs: Tensor
    router_logits: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
    latency_seconds_measured: float
    temporary_cuda_bytes_measured: int
    audit: dict[str, object]


@dataclass(frozen=True)
class QwenProbeCost:
    latency_seconds_measured: float
    temporary_cuda_bytes_measured: int
    output_tensor_bytes: int
    persistent_default_vector_bytes: int
    residual_bank_input_bytes: int
    attention_queries: int
    attention_calls: int
    router_calls: int
    cpu_gpu_synchronizations: int
    expert_calls: int = 0
    history_state_input_bytes: int = 0
    retrieval_state_input_bytes: int = 0
    lm_head_calls: int = 0
    lm_head_queries: int = 0
    midlayer_refreshes: int = 0
    diagnostic_state_input_bytes: int = 0


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
    shadow_executed_topk_ids: dict[int, Tensor] = field(default_factory=dict)
    shadow_moe_residuals: dict[int, Tensor] = field(default_factory=dict)
    shadow_attention_outputs: dict[int, Tensor] = field(default_factory=dict)
    shadow_router_inputs: dict[int, Tensor] = field(default_factory=dict)
    shadow_layer_outputs: dict[int, Tensor] = field(default_factory=dict)


def _top_b(scores: Tensor, budget: int) -> tuple[int, ...]:
    ranking = sorted(
        range(scores.numel()),
        key=lambda expert: (-float(scores[expert]), expert),
    )
    return tuple(sorted(ranking[:budget]))


def _tensor_bytes(tensors: dict[int, Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in tensors.values())


def _linear_velocity_norm_match(
    reference: Tensor,
    delta: Tensor,
    *,
    coefficients: tuple[float, ...] | None = None,
    max_relative_delta_norm: float | None = None,
) -> Tensor:
    if reference.ndim != 3 or delta.ndim != 1 or reference.shape[-1] != delta.shape[0]:
        raise ValueError("state correction requires [batch, anchors, hidden] plus [hidden]")
    values = (
        tuple(float(index) for index in range(1, reference.shape[1] + 1))
        if coefficients is None
        else coefficients
    )
    if len(values) != reference.shape[1] or not all(
        value >= 0 and torch.isfinite(torch.tensor(value)) for value in values
    ):
        raise ValueError("state correction coefficients must be finite and match anchors")
    if max_relative_delta_norm is not None and (
        max_relative_delta_norm <= 0
        or not bool(torch.isfinite(torch.tensor(max_relative_delta_norm)))
    ):
        raise ValueError("state correction norm cap must be finite and positive")
    coefficient_tensor = torch.tensor(
        values,
        device=reference.device,
        dtype=torch.float32,
    ).reshape(1, -1, 1)
    reference_f = reference.float()
    correction = coefficient_tensor * delta.to(reference.device).float()[None, None]
    reference_norm = reference_f.norm(dim=-1, keepdim=True)
    if max_relative_delta_norm is not None:
        correction_norm = correction.norm(dim=-1, keepdim=True)
        correction_scale = torch.where(
            correction_norm > 0,
            (max_relative_delta_norm * reference_norm / correction_norm).clamp(max=1),
            torch.ones_like(correction_norm),
        )
        correction = correction * correction_scale
    candidate = reference_f + correction
    candidate_norm = candidate.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    corrected = candidate * (reference_norm / candidate_norm)
    if not bool(torch.isfinite(corrected).all()):
        raise RuntimeError("state correction produced a non-finite tensor")
    return cast(Tensor, corrected.to(reference.dtype))


def _retrieval_mix_norm_match(
    reference: Tensor,
    retrieved: Tensor,
    similarities: Tensor,
    *,
    mixing: StateRetrievalMix,
) -> Tensor:
    """Mix anchor-aligned policy-local state while preserving each fresh norm."""
    if (
        reference.ndim != 3
        or retrieved.ndim != 2
        or reference.shape[1:] != retrieved.shape
        or similarities.ndim != 1
        or similarities.shape[0] != reference.shape[1]
    ):
        raise ValueError(
            "state retrieval requires [batch, anchors, hidden], [anchors, hidden], and [anchors]"
        )
    if mixing not in {"additive_norm", "replace_norm"}:
        raise ValueError("state retrieval mixing must be additive_norm or replace_norm")
    similarity = similarities.to(reference.device, dtype=torch.float32).reshape(1, -1, 1)
    if not bool(torch.isfinite(similarity).all()) or bool(
        ((similarity < 0) | (similarity > 1)).any()
    ):
        raise ValueError("state retrieval similarities must be finite in [0,1]")
    fresh = reference.float()
    recalled = retrieved.to(reference.device).float().unsqueeze(0)
    if not bool(torch.isfinite(recalled).all()):
        raise ValueError("retrieved state is non-finite")
    candidate = (
        fresh + similarity * recalled
        if mixing == "additive_norm"
        else (1 - similarity) * fresh + similarity * recalled
    )
    fresh_norm = fresh.norm(dim=-1, keepdim=True)
    candidate_norm = candidate.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    mixed = candidate * (fresh_norm / candidate_norm)
    if not bool(torch.isfinite(mixed).all()):
        raise RuntimeError("state retrieval produced a non-finite tensor")
    return cast(Tensor, mixed.to(reference.dtype))


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


def _hidden_tensor(output: object, *, source: str) -> Tensor:
    if isinstance(output, Tensor):
        value = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        value = output[0]
    else:
        raise RuntimeError(f"{source} did not expose a leading hidden-state tensor")
    if value.ndim != 3 or value.shape[0] != 1:
        raise RuntimeError(f"{source} hidden state must be [1, tokens, hidden]")
    return value


def capture_natural_qwen_components(
    model: nn.Module,
    ops: Qwen3MoePrefetchOps,
    production_cache: object,
    token_ids: tuple[int, ...],
) -> QwenNaturalComponentState:
    """Capture exact native future components without mutating production state."""
    if ops.model is not model:
        raise ValueError("natural component capture ops/model identity mismatch")
    if not 1 <= len(token_ids) <= 8:
        raise ValueError("natural component capture requires one to eight future tokens")
    device = next(model.parameters()).device
    boundary_length = _cache_length(production_cache)
    cache_before = _cache_mutation_signature(production_cache)
    rng_before = _rng_snapshot(device)
    shadow_cache = _fork_cache_copy_on_write(production_cache)
    attention_values: dict[int, Tensor] = {}
    layer_values: dict[int, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture_attention(layer: int, output: object) -> None:
        if layer in attention_values:
            raise RuntimeError("native attention component captured more than once")
        attention_values[layer] = (
            _hidden_tensor(
                output,
                source="native attention",
            )
            .detach()
            .clone()
        )

    def capture_layer(layer: int, output: object) -> None:
        if layer in layer_values:
            raise RuntimeError("native layer output captured more than once")
        layer_values[layer] = (
            _hidden_tensor(
                output,
                source="native decoder layer",
            )
            .detach()
            .clone()
        )

    allocated_before = 0
    sync_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        sync_count += 1
        allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cache_unchanged = False
    rng_unchanged = False
    fork_identity_unchanged = False
    try:
        for layer_idx, layer in enumerate(ops.layers):
            handles.append(
                cast(Any, layer).self_attn.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_idx: capture_attention(
                        index, output
                    )
                )
            )
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, index=layer_idx: capture_layer(index, output)
                )
            )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with (
            NativeRouteCaptureContext(ops) as route_capture,
            MoeOutputCaptureContext(ops) as moe_capture,
            torch.inference_mode(),
        ):
            output = cast(
                Any,
                model(
                    input_ids=input_ids,
                    past_key_values=shadow_cache,
                    use_cache=True,
                    return_dict=True,
                ),
            )
            routes = route_capture.drain()
            moe_records = moe_capture.drain()
        fork_identity_unchanged = output.past_key_values is shadow_cache
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
        for handle in handles:
            handle.remove()
        _restore_rng(device, rng_before)
    if not cache_unchanged:
        raise RuntimeError("natural component capture mutated the production cache")
    if not rng_unchanged:
        raise RuntimeError("natural component capture changed generation RNG")
    if not fork_identity_unchanged:
        raise RuntimeError("natural component capture changed shadow cache identity")
    expected_layers = tuple(range(ops.num_layers))
    if tuple(sorted(attention_values)) != expected_layers:
        raise RuntimeError("natural attention capture missed a routed layer")
    if tuple(sorted(layer_values)) != expected_layers:
        raise RuntimeError("natural layer-output capture missed a routed layer")
    if tuple(record.layer for record in routes) != expected_layers:
        raise RuntimeError("natural router capture missed or reordered a routed layer")
    attention_outputs = torch.stack(
        [attention_values[layer][0] for layer in expected_layers], dim=1
    )
    layer_outputs = torch.stack([layer_values[layer][0] for layer in expected_layers], dim=1)
    router_inputs = moe_router_input_bank(moe_records, expected_layers=ops.num_layers)
    moe_residuals = moe_output_bank(moe_records, expected_layers=ops.num_layers)
    router_logits = torch.stack([record.natural.logits for record in routes], dim=1)
    topk_ids = torch.stack([record.natural.ids for record in routes], dim=1)
    topk_weights = torch.stack([record.natural.weights for record in routes], dim=1)
    expected_state_shape = (len(token_ids), ops.num_layers, ops.hidden_size)
    if any(
        tuple(value.shape) != expected_state_shape
        for value in (attention_outputs, router_inputs, moe_residuals, layer_outputs)
    ):
        raise RuntimeError("natural component state bank shape mismatch")
    temporary_cuda = (
        max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
        if device.type == "cuda"
        else 0
    )
    return QwenNaturalComponentState(
        attention_outputs=attention_outputs,
        router_inputs=router_inputs,
        moe_residuals=moe_residuals,
        layer_outputs=layer_outputs,
        router_logits=router_logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        latency_seconds_measured=elapsed,
        temporary_cuda_bytes_measured=temporary_cuda,
        audit={
            "information_regime": "diagnostic_future_state_oracle",
            "deployable": False,
            "token_count": len(token_ids),
            "production_cache_sequence_length_before": boundary_length,
            "production_cache_sequence_length_after": _cache_length(production_cache),
            "production_cache_signature_unchanged": True,
            "production_rng_unchanged": True,
            "shadow_cache_identity_unchanged": True,
            "shadow_cache_discarded": True,
            "native_attention": True,
            "native_rope": True,
            "native_router": True,
            "full_native_expert_access": True,
            "attention_calls": ops.num_layers,
            "attention_queries": ops.num_layers * len(token_ids),
            "router_calls": ops.num_layers,
            "expert_calls": ops.num_layers,
            "cpu_gpu_synchronizations": sync_count,
        },
    )


class QwenPseudoEmbeddingProbe:
    """Run exact Qwen attention/RoPE/router on disposable shadow residuals."""

    def __init__(
        self,
        model: nn.Module,
        ops: Qwen3MoePrefetchOps,
        defaults: DefaultVectorArtifact | None,
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
        if variant.expert_contribution == "default_vector_selected_topk_mixture":
            if defaults is None:
                raise ValueError("default-vector contribution requires a default artifact")
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
        if variant.content.startswith("self_") and variant.attention != "causal":
            raise ValueError("autoregressive pseudo content requires causal attention")
        if variant.content == "self_topk_particles":
            if variant.particle_count not in (2, 4):
                raise ValueError("particle rollout requires a fixed count of two or four")
        elif variant.particle_count != 1:
            raise ValueError("particle count is only valid for particle rollout")
        if variant.expected_embedding_norm != "raw" and variant.content not in {
            "expected_top_m",
            "sampled_then_expected_top_m",
        }:
            raise ValueError("expected-embedding norm mode is invalid for this content")
        correction_enabled = (
            variant.hidden_state_correction != "none" or variant.residual_correction != "none"
        )
        if correction_enabled and (
            variant.content != "provided_sequence"
            or variant.attention != "causal"
            or variant.expert_contribution != "native_expert_execution"
        ):
            raise ValueError(
                "state corrections require causal provided content and native expert execution"
            )
        retrieval_enabled = variant.state_retrieval_target != "none"
        if retrieval_enabled != (variant.state_retrieval_mix != "none"):
            raise ValueError("state retrieval target and mixing must be enabled together")
        if retrieval_enabled and (
            variant.content != "provided_sequence"
            or variant.attention != "causal"
            or variant.expert_contribution != "native_expert_execution"
        ):
            raise ValueError(
                "state retrieval requires causal provided content and native expert execution"
            )
        if retrieval_enabled and correction_enabled:
            raise ValueError("state retrieval and velocity correction cannot be combined")
        midlayer_enabled = variant.midlayer_self_conditioning != "none"
        if midlayer_enabled and (
            variant.content != "provided_sequence"
            or variant.attention != "causal"
            or variant.expert_contribution != "native_expert_execution"
        ):
            raise ValueError(
                "midlayer self-conditioning requires causal provided content and native "
                "expert execution"
            )
        if midlayer_enabled and (correction_enabled or retrieval_enabled):
            raise ValueError(
                "midlayer self-conditioning cannot be combined with correction or retrieval"
            )
        if midlayer_enabled:
            refresh_layer = variant.midlayer_refresh_after_layer
            if refresh_layer is None or not 0 <= refresh_layer < ops.num_layers - 1:
                raise ValueError("midlayer refresh must precede the final routed layer")
            if len(anchors) < 2:
                raise ValueError("shifted midlayer refresh requires at least two anchors")
            cap = variant.midlayer_max_relative_embedding_delta_norm
            if cap is not None and (cap <= 0 or not bool(torch.isfinite(torch.tensor(cap)))):
                raise ValueError("midlayer embedding-delta cap must be finite and positive")
        elif (
            variant.midlayer_refresh_after_layer is not None
            or variant.midlayer_max_relative_embedding_delta_norm is not None
        ):
            raise ValueError("midlayer refresh settings require self-conditioning")
        diagnostic_enabled = variant.diagnostic_state_patch != "none"
        if diagnostic_enabled and (
            variant.content != "provided_sequence"
            or variant.attention != "causal"
            or variant.expert_contribution != "native_expert_execution"
        ):
            raise ValueError(
                "diagnostic state patches require causal provided content and native expert "
                "execution"
            )
        if diagnostic_enabled and (correction_enabled or retrieval_enabled or midlayer_enabled):
            raise ValueError(
                "diagnostic state patches cannot be combined with correction, retrieval, or "
                "midlayer refresh"
            )
        if variant.diagnostic_state_patch == "oracle_hidden_after_layer":
            checkpoint = variant.diagnostic_hidden_after_layer
            if checkpoint is None or not 0 <= checkpoint < ops.num_layers - 1:
                raise ValueError("diagnostic hidden checkpoint must precede the final layer")
        elif variant.diagnostic_hidden_after_layer is not None:
            raise ValueError("diagnostic hidden checkpoint requires the hidden patch variant")
        self.model = model
        self.ops = ops
        self.variant = variant
        self.anchors = anchors
        self.budget = budget
        parameter = next(model.parameters())
        self.defaults = (
            defaults.mean.to(device=parameter.device, dtype=parameter.dtype)
            if defaults is not None
            else None
        )
        self.default_counts = defaults.count if defaults is not None else None
        self.default_fingerprint = defaults.fingerprint if defaults is not None else None
        self.default_definition = defaults.definition if defaults is not None else None

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
        anchor_start: int = 0,
    ) -> Tensor:
        if self.variant.content == "expected_top_m":
            if expected_embedding is None:
                raise ValueError("expected_top_m requires a deployable sampling-step embedding")
            return expected_embedding[:, None, :].expand(-1, sequence, -1)
        if self.variant.content == "sampled_then_expected_top_m":
            if expected_embedding is None:
                raise ValueError("sampled_then_expected_top_m requires sampling-step logits")
            hidden = expected_embedding[:, None, :].expand(-1, sequence, -1).clone()
            if anchor_start == 0:
                sampled = torch.tensor(
                    [[sampled_next_token_id]],
                    dtype=torch.long,
                    device=hidden.device,
                )
                hidden[:, 0] = self._base.embed_tokens(sampled)[:, 0]
            return hidden
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

    def _expected_embedding(
        self,
        logits: Tensor,
        top_m: int,
        sampled_next_token_id: int,
    ) -> Tensor:
        selected_logits, selected_ids = logits.float().topk(top_m, dim=-1)
        weights = selected_logits.softmax(dim=-1).to(dtype=next(self.model.parameters()).dtype)
        embeddings = self._base.embed_tokens(selected_ids.to(logits.device))
        expected = (weights.to(logits.device).unsqueeze(-1) * embeddings).sum(dim=1)
        if self.variant.expected_embedding_norm == "sampled_token":
            sampled = torch.tensor(
                [[sampled_next_token_id]],
                dtype=torch.long,
                device=logits.device,
            )
            reference = self._base.embed_tokens(sampled)[:, 0]
            expected = expected * (
                reference.float().norm(dim=-1, keepdim=True)
                / expected.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(expected.dtype)
        return cast(Tensor, expected)

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

    def _shifted_midlayer_refresh(
        self,
        hidden: Tensor,
        anchor_token_ids: tuple[int, ...] | None,
    ) -> tuple[Tensor, dict[str, object]]:
        if anchor_token_ids is None or len(anchor_token_ids) != hidden.shape[1]:
            raise ValueError("midlayer refresh requires one initial token ID per anchor")
        if hidden.shape[0] != 1 or hidden.shape[1] < 2:
            raise ValueError("midlayer refresh requires batch one and at least two anchors")
        anchor_one = hidden[:, :1].clone()
        source_hidden = hidden[:, :-1]
        normalized = self._base.norm(source_hidden)
        vocabulary_logits = cast(Any, self.model).lm_head(normalized).float()
        predicted_top1 = vocabulary_logits.argmax(dim=-1)
        expected_top8_ids: Tensor | None = None
        expected_top8_weights: Tensor | None = None
        if self.variant.midlayer_self_conditioning == "greedy_shift":
            predicted_embedding = self._base.embed_tokens(predicted_top1)
        elif self.variant.midlayer_self_conditioning == "expected_top8_shift":
            selected_logits, expected_top8_ids = vocabulary_logits.topk(8, dim=-1)
            expected_top8_weights = selected_logits.softmax(dim=-1)
            selected_embeddings = self._base.embed_tokens(expected_top8_ids)
            predicted_embedding = (
                expected_top8_weights.to(selected_embeddings.dtype).unsqueeze(-1)
                * selected_embeddings
            ).sum(dim=-2)
        else:
            raise AssertionError("midlayer refresh mode is disabled")
        target_ids = torch.tensor(
            [anchor_token_ids[1:]],
            dtype=torch.long,
            device=hidden.device,
        )
        initial_embedding = self._base.embed_tokens(target_ids)
        raw_delta = predicted_embedding - initial_embedding
        target = hidden[:, 1:]
        target_f = target.float()
        target_norm = target_f.norm(dim=-1, keepdim=True)
        applied_delta = raw_delta.float()
        cap = self.variant.midlayer_max_relative_embedding_delta_norm
        if cap is not None:
            delta_norm = applied_delta.norm(dim=-1, keepdim=True)
            scale = torch.where(
                delta_norm > 0,
                (cap * target_norm / delta_norm).clamp(max=1),
                torch.ones_like(delta_norm),
            )
            applied_delta = applied_delta * scale
        candidate = target_f + applied_delta
        candidate_norm = candidate.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        refreshed_target = (candidate * (target_norm / candidate_norm)).to(target.dtype)
        refreshed = torch.cat((anchor_one, refreshed_target), dim=1)
        if not bool(torch.isfinite(refreshed).all()):
            raise RuntimeError("midlayer self-conditioning produced non-finite hidden state")
        raw_ratio = raw_delta.float().norm(dim=-1) / target_norm.squeeze(-1).clamp_min(1e-12)
        applied_ratio = applied_delta.norm(dim=-1) / target_norm.squeeze(-1).clamp_min(1e-12)
        return refreshed, {
            "midlayer_self_conditioning": self.variant.midlayer_self_conditioning,
            "midlayer_refresh_after_layer": self.variant.midlayer_refresh_after_layer,
            "midlayer_anchor_one_hidden_bitwise_unchanged": torch.equal(
                refreshed[:, :1], anchor_one
            ),
            "midlayer_refresh_finite": True,
            "midlayer_source_anchor_indices_one_based": list(range(1, hidden.shape[1])),
            "midlayer_target_anchor_indices_one_based": list(range(2, hidden.shape[1] + 1)),
            "midlayer_predicted_top1_token_ids": predicted_top1[0].detach().cpu().tolist(),
            "midlayer_initial_target_token_ids": list(anchor_token_ids[1:]),
            "midlayer_predicted_top1_change_fraction": float(
                (predicted_top1 != target_ids).float().mean()
            ),
            "midlayer_expected_top8_token_ids": (
                expected_top8_ids[0].detach().cpu().tolist()
                if expected_top8_ids is not None
                else None
            ),
            "midlayer_expected_top8_weights": (
                expected_top8_weights[0].detach().cpu().tolist()
                if expected_top8_weights is not None
                else None
            ),
            "midlayer_raw_embedding_delta_norm_ratio": raw_ratio[0].detach().cpu().tolist(),
            "midlayer_applied_embedding_delta_norm_ratio": (
                applied_ratio[0].detach().cpu().tolist()
            ),
            "midlayer_max_relative_embedding_delta_norm": cap,
            "midlayer_hidden_norm_preserved": bool(
                torch.allclose(
                    refreshed_target.float().norm(dim=-1),
                    target_norm.squeeze(-1),
                    rtol=1e-3,
                    atol=1e-3,
                )
            ),
            "midlayer_lm_head_logits_shape": list(vocabulary_logits.shape),
        }

    def _layer_step(
        self,
        layer_idx: int,
        hidden: Tensor,
        shadow_cache: Cache,
        attention_mask: Any,
        position_embeddings: tuple[Tensor, Tensor],
        provided_residual: Tensor | None,
        execution_subset: tuple[int, ...] | None,
        router_input_delta: Tensor | None = None,
        moe_residual_delta: Tensor | None = None,
        retrieved_router_input: Tensor | None = None,
        retrieved_moe_residual: Tensor | None = None,
        retrieval_similarities: Tensor | None = None,
        attention_output_override: Tensor | None = None,
        moe_residual_override: Tensor | None = None,
        shadow_state_sink: dict[str, dict[int, Tensor]] | None = None,
    ) -> tuple[Tensor, NativeRoute, Tensor, NativeRoute | None, Tensor]:
        layer = cast(Any, self.ops.layers[layer_idx])
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention, _ = layer.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            past_key_values=shadow_cache,
            position_embeddings=position_embeddings,
        )
        if attention_output_override is not None:
            if attention_output_override.shape != attention.shape:
                raise ValueError("oracle attention output shape does not match pseudo attention")
            attention = attention_output_override.to(
                device=attention.device,
                dtype=attention.dtype,
            )
        post_attention = residual + attention
        router_input = layer.post_attention_layernorm(post_attention)
        if router_input_delta is not None:
            router_input = _linear_velocity_norm_match(
                router_input,
                router_input_delta,
                coefficients=self.variant.correction_anchor_coefficients,
                max_relative_delta_norm=self.variant.correction_max_relative_delta_norm,
            )
        if retrieved_router_input is not None:
            if retrieval_similarities is None:
                raise AssertionError("retrieval similarities are missing")
            router_input = _retrieval_mix_norm_match(
                router_input,
                retrieved_router_input,
                retrieval_similarities,
                mixing=self.variant.state_retrieval_mix,
            )
        route = self.ops.route(layer_idx, router_input.reshape(-1, self.ops.hidden_size))
        probabilities = route.logits.float().softmax(dim=-1)
        if self.variant.expert_contribution == "zero":
            contribution = torch.zeros_like(router_input)
            executed_route = None
        elif self.variant.expert_contribution == "default_vector_selected_topk_mixture":
            if self.defaults is None:
                raise AssertionError("default-vector tensor missing")
            selected_defaults = self.defaults[layer_idx][route.ids]
            contribution = (
                (route.weights.unsqueeze(-1) * selected_defaults)
                .sum(dim=1)
                .reshape_as(router_input)
            )
            executed_route = None
        elif self.variant.expert_contribution == "native_expert_execution":
            flat = router_input.reshape(-1, self.ops.hidden_size)
            executed_route = (
                route
                if execution_subset is None
                else masked_route_from_natural(self.ops, route, execution_subset)
            )
            contribution = self.ops.experts(layer_idx, flat, executed_route).reshape_as(
                router_input
            )
        else:
            if provided_residual is None or provided_residual.shape != router_input.shape:
                raise RuntimeError("provided MoE residual does not match pseudo router input")
            contribution = provided_residual.to(
                device=router_input.device,
                dtype=router_input.dtype,
            )
            executed_route = None
        if moe_residual_delta is not None:
            contribution = _linear_velocity_norm_match(
                contribution,
                moe_residual_delta,
                coefficients=self.variant.correction_anchor_coefficients,
                max_relative_delta_norm=self.variant.correction_max_relative_delta_norm,
            )
        if retrieved_moe_residual is not None:
            if retrieval_similarities is None:
                raise AssertionError("retrieval similarities are missing")
            contribution = _retrieval_mix_norm_match(
                contribution,
                retrieved_moe_residual,
                retrieval_similarities,
                mixing=self.variant.state_retrieval_mix,
            )
        if moe_residual_override is not None:
            if moe_residual_override.shape != contribution.shape:
                raise ValueError("oracle MoE residual shape does not match pseudo residual")
            contribution = moe_residual_override.to(
                device=contribution.device,
                dtype=contribution.dtype,
            )
        layer_output = post_attention + contribution
        if shadow_state_sink is not None:
            shadow_state_sink["attention_outputs"][layer_idx] = attention[0].detach().cpu()
            shadow_state_sink["router_inputs"][layer_idx] = router_input[0].detach().cpu()
            shadow_state_sink["moe_residuals"][layer_idx] = contribution[0].detach().cpu()
            shadow_state_sink["layer_outputs"][layer_idx] = layer_output[0].detach().cpu()
        return (
            layer_output,
            route,
            probabilities,
            executed_route,
            contribution,
        )

    def _independent_rollout(
        self,
        production_cache: object,
        positions: tuple[int, ...],
        sampled_next_token_id: int,
        current_token_id: int,
        expected_embedding: Tensor | None,
        anchor_token_ids: tuple[int, ...] | None,
        provided_residuals: Tensor | None,
        execution_subsets: dict[int, tuple[int, ...]] | None,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
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
                anchor_idx,
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
        executed_ids: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        moe_residuals: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        for layer_idx in range(self.ops.num_layers):
            for anchor_idx in range(len(self.anchors)):
                mask, position_embeddings = masks_and_positions[anchor_idx]
                (
                    hidden[anchor_idx],
                    route,
                    full_probs,
                    executed_route,
                    moe_residual,
                ) = self._layer_step(
                    layer_idx,
                    hidden[anchor_idx],
                    cast(Cache, caches[anchor_idx]),
                    mask,
                    position_embeddings,
                    (
                        provided_residuals[anchor_idx, layer_idx][None, None]
                        if provided_residuals is not None
                        else None
                    ),
                    (execution_subsets[layer_idx] if execution_subsets is not None else None),
                )
                logits[layer_idx].append(route.logits[0].detach().float().cpu())
                probabilities[layer_idx].append(full_probs[0].detach().cpu())
                topk_ids[layer_idx].append(route.ids[0].detach().cpu())
                topk_weights[layer_idx].append(route.weights[0].detach().float().cpu())
                if executed_route is not None:
                    executed_ids[layer_idx].append(executed_route.ids[0].detach().cpu())
                    moe_residuals[layer_idx].append(moe_residual[0, 0].detach().cpu())
        return (
            {layer: torch.stack(values) for layer, values in logits.items()},
            {layer: torch.stack(values) for layer, values in probabilities.items()},
            {layer: torch.stack(values) for layer, values in topk_ids.items()},
            {layer: torch.stack(values) for layer, values in topk_weights.items()},
            {layer: torch.stack(values) for layer, values in executed_ids.items() if values},
            {layer: torch.stack(values) for layer, values in moe_residuals.items() if values},
        )

    def _causal_rollout(
        self,
        production_cache: object,
        positions: tuple[int, ...],
        sampled_next_token_id: int,
        current_token_id: int,
        expected_embedding: Tensor | None,
        anchor_token_ids: tuple[int, ...] | None,
        provided_residuals: Tensor | None,
        execution_subsets: dict[int, tuple[int, ...]] | None,
        router_input_deltas: Tensor | None,
        moe_residual_deltas: Tensor | None,
        retrieved_router_inputs: Tensor | None,
        retrieved_moe_residuals: Tensor | None,
        retrieval_similarities: Tensor | None,
        oracle_component_state: QwenNaturalComponentState | None,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[str, object],
    ]:
        cache = _fork_cache_copy_on_write(production_cache)
        hidden = self._initial_hidden(
            sampled_next_token_id,
            current_token_id,
            len(self.anchors),
            expected_embedding,
            anchor_token_ids,
            0,
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
        executed_ids: dict[int, Tensor] = {}
        moe_residuals: dict[int, Tensor] = {}
        shadow_state_sink: dict[str, dict[int, Tensor]] | None = (
            {
                "attention_outputs": {},
                "router_inputs": {},
                "moe_residuals": {},
                "layer_outputs": {},
            }
            if oracle_component_state is not None
            else None
        )
        midlayer_audit: dict[str, object] = {}
        for layer_idx in range(self.ops.num_layers):
            patch = self.variant.diagnostic_state_patch
            attention_override = (
                oracle_component_state.attention_outputs[:, layer_idx][None]
                if oracle_component_state is not None
                and patch in {"oracle_attention_output", "oracle_attention_and_moe"}
                else None
            )
            moe_override = (
                oracle_component_state.moe_residuals[:, layer_idx][None]
                if oracle_component_state is not None
                and patch in {"oracle_moe_residual", "oracle_attention_and_moe"}
                else None
            )
            (
                hidden,
                route,
                full_probs,
                executed_route,
                moe_residual,
            ) = self._layer_step(
                layer_idx,
                hidden,
                cast(Cache, cache),
                attention_mask,
                position_embeddings,
                (
                    provided_residuals[:, layer_idx][None]
                    if provided_residuals is not None
                    else None
                ),
                (execution_subsets[layer_idx] if execution_subsets is not None else None),
                (router_input_deltas[layer_idx] if router_input_deltas is not None else None),
                (moe_residual_deltas[layer_idx] if moe_residual_deltas is not None else None),
                (
                    retrieved_router_inputs[:, layer_idx]
                    if retrieved_router_inputs is not None
                    else None
                ),
                (
                    retrieved_moe_residuals[:, layer_idx]
                    if retrieved_moe_residuals is not None
                    else None
                ),
                retrieval_similarities,
                attention_override,
                moe_override,
                shadow_state_sink,
            )
            logits[layer_idx] = route.logits.detach().float().cpu()
            probabilities[layer_idx] = full_probs.detach().cpu()
            topk_ids[layer_idx] = route.ids.detach().cpu()
            topk_weights[layer_idx] = route.weights.detach().float().cpu()
            if executed_route is not None:
                executed_ids[layer_idx] = executed_route.ids.detach().cpu()
                moe_residuals[layer_idx] = moe_residual[0].detach().cpu()
            if (
                oracle_component_state is not None
                and patch == "oracle_hidden_after_layer"
                and layer_idx == self.variant.diagnostic_hidden_after_layer
            ):
                hidden = oracle_component_state.layer_outputs[:, layer_idx][None].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                if shadow_state_sink is None:
                    raise AssertionError("diagnostic shadow-state sink is missing")
                shadow_state_sink["layer_outputs"][layer_idx] = hidden[0].detach().cpu()
            if (
                self.variant.midlayer_self_conditioning != "none"
                and layer_idx == self.variant.midlayer_refresh_after_layer
            ):
                hidden, midlayer_audit = self._shifted_midlayer_refresh(
                    hidden,
                    anchor_token_ids,
                )
        return (
            logits,
            probabilities,
            topk_ids,
            topk_weights,
            executed_ids,
            moe_residuals,
            (shadow_state_sink["attention_outputs"] if shadow_state_sink is not None else {}),
            shadow_state_sink["router_inputs"] if shadow_state_sink is not None else {},
            shadow_state_sink["layer_outputs"] if shadow_state_sink is not None else {},
            midlayer_audit,
        )

    def _autoregressive_rollout(
        self,
        production_cache: object,
        positions: tuple[int, ...],
        sampled_next_token_id: int,
        expected_top_m: int,
        execution_subsets: dict[int, tuple[int, ...]] | None,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        cache = _fork_cache_copy_on_write(production_cache)
        token = torch.tensor(
            [[sampled_next_token_id]],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        hidden = cast(Tensor, self._base.embed_tokens(token))
        logits: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        probabilities: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        topk_ids: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        topk_weights: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        executed_ids: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        moe_residuals: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        for position in positions:
            attention_mask, position_embeddings = self._position_inputs(
                hidden,
                (position,),
                cast(Cache, cache),
            )
            for layer_idx in range(self.ops.num_layers):
                hidden, route, full_probs, executed_route, moe_residual = self._layer_step(
                    layer_idx,
                    hidden,
                    cast(Cache, cache),
                    attention_mask,
                    position_embeddings,
                    None,
                    (execution_subsets[layer_idx] if execution_subsets is not None else None),
                )
                logits[layer_idx].append(route.logits[0].detach().float().cpu())
                probabilities[layer_idx].append(full_probs[0].detach().cpu())
                topk_ids[layer_idx].append(route.ids[0].detach().cpu())
                topk_weights[layer_idx].append(route.weights[0].detach().float().cpu())
                if executed_route is not None:
                    executed_ids[layer_idx].append(executed_route.ids[0].detach().cpu())
                    moe_residuals[layer_idx].append(moe_residual[0, 0].detach().cpu())
            final_hidden = self._base.norm(hidden)
            next_logits = cast(Any, self.model).lm_head(final_hidden[:, -1])
            if self.variant.content == "self_greedy":
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                hidden = cast(Tensor, self._base.embed_tokens(next_token))
            else:
                hidden = self._expected_embedding(
                    next_logits,
                    expected_top_m,
                    int(next_logits.argmax(dim=-1)[0]),
                )[:, None, :]
        return (
            {layer: torch.stack(values) for layer, values in logits.items()},
            {layer: torch.stack(values) for layer, values in probabilities.items()},
            {layer: torch.stack(values) for layer, values in topk_ids.items()},
            {layer: torch.stack(values) for layer, values in topk_weights.items()},
            {layer: torch.stack(values) for layer, values in executed_ids.items() if values},
            {layer: torch.stack(values) for layer, values in moe_residuals.items() if values},
        )

    def _particle_rollout(
        self,
        production_cache: object,
        positions: tuple[int, ...],
        sampled_next_token_id: int,
        execution_subsets: dict[int, tuple[int, ...]] | None,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        token = torch.tensor(
            [[sampled_next_token_id]],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        particles: list[tuple[object, Tensor, float]] = [
            (
                _fork_cache_copy_on_write(production_cache),
                cast(Tensor, self._base.embed_tokens(token)),
                0.0,
            )
        ]
        logits: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        probabilities: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        topk_ids: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        topk_weights: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        executed_ids: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        moe_residuals: dict[int, list[Tensor]] = {layer: [] for layer in range(self.ops.num_layers)}
        for anchor_index, position in enumerate(positions):
            realized: list[dict[str, Any]] = []
            for cache, particle_hidden, log_weight in particles:
                attention_mask, position_embeddings = self._position_inputs(
                    particle_hidden,
                    (position,),
                    cast(Cache, cache),
                )
                layer_values: dict[int, dict[str, Tensor]] = {}
                hidden = particle_hidden
                for layer_idx in range(self.ops.num_layers):
                    hidden, route, full_probs, executed_route, moe_residual = self._layer_step(
                        layer_idx,
                        hidden,
                        cast(Cache, cache),
                        attention_mask,
                        position_embeddings,
                        None,
                        (execution_subsets[layer_idx] if execution_subsets is not None else None),
                    )
                    values = {
                        "logits": route.logits[0].detach().float().cpu(),
                        "probabilities": full_probs[0].detach().cpu(),
                        "topk_ids": route.ids[0].detach().cpu(),
                        "topk_weights": route.weights[0].detach().float().cpu(),
                        "residual": moe_residual[0, 0].detach().cpu(),
                    }
                    if executed_route is not None:
                        values["executed_ids"] = executed_route.ids[0].detach().cpu()
                    layer_values[layer_idx] = values
                final_hidden = self._base.norm(hidden)
                next_logits = cast(Any, self.model).lm_head(final_hidden[:, -1])
                realized.append(
                    {
                        "cache": cache,
                        "log_weight": log_weight,
                        "next_logits": next_logits,
                        "layers": layer_values,
                    }
                )
            mixture = torch.tensor(
                [float(value["log_weight"]) for value in realized],
                dtype=torch.float64,
            ).softmax(dim=0)
            representative = int(mixture.argmax())
            for layer_idx in range(self.ops.num_layers):
                layer_logits = torch.stack(
                    [
                        cast(dict[str, Tensor], value["layers"][layer_idx])["logits"]
                        for value in realized
                    ]
                )
                layer_probabilities = torch.stack(
                    [
                        cast(dict[str, Tensor], value["layers"][layer_idx])["probabilities"]
                        for value in realized
                    ]
                )
                layer_residuals = torch.stack(
                    [
                        cast(dict[str, Tensor], value["layers"][layer_idx])["residual"]
                        for value in realized
                    ]
                )
                mixed_logits = (mixture[:, None] * layer_logits.double()).sum(dim=0)
                mixed_probabilities = (mixture[:, None] * layer_probabilities.double()).sum(dim=0)
                selected_weights, selected_ids = mixed_probabilities.topk(self.ops.top_k)
                if bool(cast(Any, self.model).config.norm_topk_prob):
                    selected_weights = selected_weights / selected_weights.sum().clamp_min(1e-12)
                logits[layer_idx].append(mixed_logits.float())
                probabilities[layer_idx].append(mixed_probabilities.float())
                topk_ids[layer_idx].append(selected_ids)
                topk_weights[layer_idx].append(selected_weights.float())
                representative_values = cast(
                    dict[str, Tensor],
                    realized[representative]["layers"][layer_idx],
                )
                if "executed_ids" in representative_values:
                    executed_ids[layer_idx].append(representative_values["executed_ids"])
                    moe_residuals[layer_idx].append(
                        (mixture[:, None] * layer_residuals.double())
                        .sum(dim=0)
                        .to(layer_residuals.dtype)
                    )
            if anchor_index + 1 == len(positions):
                continue
            branches: list[tuple[float, int, int, object, Tensor]] = []
            for parent_index, value in enumerate(realized):
                next_logits = cast(Tensor, value["next_logits"])
                log_probabilities = next_logits.float().log_softmax(dim=-1)
                selected_log_probs, selected_ids = log_probabilities.topk(
                    self.variant.particle_count,
                    dim=-1,
                )
                for rank in range(self.variant.particle_count):
                    token_id = int(selected_ids[0, rank])
                    branch_token = selected_ids[:, rank : rank + 1]
                    branch_hidden = cast(Tensor, self._base.embed_tokens(branch_token))
                    branches.append(
                        (
                            float(value["log_weight"]) + float(selected_log_probs[0, rank]),
                            token_id,
                            parent_index,
                            _fork_cache_copy_on_write(value["cache"]),
                            branch_hidden,
                        )
                    )
            branches.sort(key=lambda value: (-value[0], value[1], value[2]))
            particles = [
                (cache, hidden, log_weight)
                for log_weight, _token_id, _parent, cache, hidden in branches[
                    : self.variant.particle_count
                ]
            ]
        return (
            {layer: torch.stack(values) for layer, values in logits.items()},
            {layer: torch.stack(values) for layer, values in probabilities.items()},
            {layer: torch.stack(values) for layer, values in topk_ids.items()},
            {layer: torch.stack(values) for layer, values in topk_weights.items()},
            {layer: torch.stack(values) for layer, values in executed_ids.items() if values},
            {layer: torch.stack(values) for layer, values in moe_residuals.items() if values},
        )

    def predict(
        self,
        production_cache: object,
        *,
        sampled_next_token_id: int,
        current_token_id: int,
        next_token_logits: Tensor | None = None,
        expected_top_m: int = 8,
        anchor_token_ids: tuple[int, ...] | None = None,
        provided_residuals: Tensor | None = None,
        provided_residual_source: str | None = None,
        execution_subsets: dict[int, tuple[int, ...]] | None = None,
        execution_subset_source: str | None = None,
        recent_router_inputs: Tensor | None = None,
        recent_moe_outputs: Tensor | None = None,
        correction_history_source: str | None = None,
        retrieved_router_inputs: Tensor | None = None,
        retrieved_moe_residuals: Tensor | None = None,
        retrieval_similarities: Tensor | None = None,
        retrieval_state_source: str | None = None,
        oracle_component_state: QwenNaturalComponentState | None = None,
        oracle_component_source: str | None = None,
    ) -> QwenPseudoProbeResult:
        """Predict one H-window subset without mutating production state or RNG."""
        device = next(self.model.parameters()).device
        boundary_length = _cache_length(production_cache)
        positions = tuple(boundary_length + anchor - 1 for anchor in self.anchors)
        cache_before = _cache_mutation_signature(production_cache)
        rng_before = _rng_snapshot(device)
        expected_embedding = None
        boundary_expected = self.variant.content in {
            "expected_top_m",
            "sampled_then_expected_top_m",
        }
        if boundary_expected:
            if next_token_logits is None or tuple(next_token_logits.shape[:1]) != (1,):
                raise ValueError("expected content requires one row of next-token logits")
            if expected_top_m < 1 or expected_top_m > next_token_logits.shape[-1]:
                raise ValueError("expected_top_m is outside the vocabulary")
        elif next_token_logits is not None:
            raise ValueError("next_token_logits are only valid for boundary-expected content")
        if self.variant.content == "self_expected_top_m" and (
            expected_top_m < 1 or expected_top_m > cast(Any, self.model).config.vocab_size
        ):
            raise ValueError("self expected_top_m is outside the vocabulary")
        if self.variant.content == "provided_sequence":
            if anchor_token_ids is None or len(anchor_token_ids) != len(self.anchors):
                raise ValueError("provided_sequence requires one token ID per pseudo anchor")
        elif anchor_token_ids is not None:
            raise ValueError("anchor_token_ids are only valid for provided_sequence")
        expected_residual_shape = (
            len(self.anchors),
            self.ops.num_layers,
            self.ops.hidden_size,
        )
        if self.variant.expert_contribution == "provided_residual":
            if provided_residuals is None or tuple(provided_residuals.shape) != (
                expected_residual_shape
            ):
                raise ValueError("provided_residual requires [anchors, layers, hidden] residuals")
            if not provided_residual_source:
                raise ValueError("provided_residual requires an information-source label")
        elif provided_residuals is not None or provided_residual_source is not None:
            raise ValueError("provided residuals are only valid for provided_residual")
        if self.variant.expert_contribution == "native_expert_execution":
            if not execution_subset_source:
                raise ValueError("native expert execution requires a subset-source label")
            if execution_subsets is not None:
                if set(execution_subsets) != set(range(self.ops.num_layers)):
                    raise ValueError("execution subsets must cover every routed layer")
                for subset in execution_subsets.values():
                    if (
                        len(subset) < self.ops.top_k
                        or len(subset) > self.ops.num_experts
                        or len(set(subset)) != len(subset)
                        or min(subset) < 0
                        or max(subset) >= self.ops.num_experts
                    ):
                        raise ValueError("invalid layer-local execution subset")
        elif execution_subsets is not None or execution_subset_source is not None:
            raise ValueError("execution subsets are only valid for native expert execution")
        expected_history_shape = (2, self.ops.num_layers, self.ops.hidden_size)
        hidden_correction = self.variant.hidden_state_correction != "none"
        residual_correction = self.variant.residual_correction != "none"
        if hidden_correction:
            if recent_router_inputs is None or tuple(recent_router_inputs.shape) != (
                expected_history_shape
            ):
                raise ValueError("hidden correction requires two policy-local router-input rows")
        elif recent_router_inputs is not None:
            raise ValueError("router-input history is only valid for hidden correction")
        if residual_correction:
            if recent_moe_outputs is None or tuple(recent_moe_outputs.shape) != (
                expected_history_shape
            ):
                raise ValueError("residual correction requires two policy-local MoE-output rows")
        elif recent_moe_outputs is not None:
            raise ValueError("MoE-output history is only valid for residual correction")
        if hidden_correction or residual_correction:
            if not correction_history_source:
                raise ValueError("state correction requires a policy-local history source")
            if self.variant.attention != "causal":
                raise ValueError("state correction requires one causal pseudo sequence")
        elif correction_history_source is not None:
            raise ValueError("correction history source requires an enabled correction")
        retrieval_enabled = self.variant.state_retrieval_target != "none"
        expected_retrieval_shape = (
            len(self.anchors),
            self.ops.num_layers,
            self.ops.hidden_size,
        )
        target_tensor = (
            retrieved_router_inputs
            if self.variant.state_retrieval_target == "router_input"
            else retrieved_moe_residuals
            if self.variant.state_retrieval_target == "moe_residual"
            else None
        )
        other_tensor = (
            retrieved_moe_residuals
            if self.variant.state_retrieval_target == "router_input"
            else retrieved_router_inputs
            if self.variant.state_retrieval_target == "moe_residual"
            else None
        )
        if retrieval_enabled:
            if target_tensor is None or tuple(target_tensor.shape) != expected_retrieval_shape:
                raise ValueError(
                    "state retrieval requires one [anchors, layers, hidden] target bank"
                )
            if other_tensor is not None:
                raise ValueError("state retrieval received a tensor for the wrong target")
            if retrieval_similarities is None or tuple(retrieval_similarities.shape) != (
                len(self.anchors),
            ):
                raise ValueError("state retrieval requires one similarity per anchor")
            if not retrieval_state_source:
                raise ValueError("state retrieval requires a policy-local source label")
            if not bool(torch.isfinite(target_tensor).all()):
                raise ValueError("retrieved state bank is non-finite")
            if not bool(torch.isfinite(retrieval_similarities).all()) or bool(
                ((retrieval_similarities < 0) | (retrieval_similarities > 1)).any()
            ):
                raise ValueError("state retrieval similarities must be finite in [0,1]")
        elif any(
            value is not None
            for value in (
                retrieved_router_inputs,
                retrieved_moe_residuals,
                retrieval_similarities,
                retrieval_state_source,
            )
        ):
            raise ValueError("retrieval inputs require an enabled state-retrieval variant")
        diagnostic_patch = self.variant.diagnostic_state_patch
        diagnostic_state_supplied = oracle_component_state is not None
        if diagnostic_patch != "none" and not diagnostic_state_supplied:
            raise ValueError("diagnostic state patch requires natural component state")
        if oracle_component_state is not None:
            if not oracle_component_source:
                raise ValueError("natural component state requires a diagnostic source label")
            if (
                self.variant.content != "provided_sequence"
                or self.variant.attention != "causal"
                or self.variant.expert_contribution != "native_expert_execution"
            ):
                raise ValueError(
                    "natural component diagnostics require causal provided content and native "
                    "expert execution"
                )
            expected_component_shape = (
                len(self.anchors),
                self.ops.num_layers,
                self.ops.hidden_size,
            )
            component_values = (
                oracle_component_state.attention_outputs,
                oracle_component_state.router_inputs,
                oracle_component_state.moe_residuals,
                oracle_component_state.layer_outputs,
            )
            if any(tuple(value.shape) != expected_component_shape for value in component_values):
                raise ValueError("natural component state bank shape mismatch")
            if tuple(oracle_component_state.router_logits.shape) != (
                len(self.anchors),
                self.ops.num_layers,
                self.ops.num_experts,
            ):
                raise ValueError("natural component router-logit shape mismatch")
            expected_topk_shape = (
                len(self.anchors),
                self.ops.num_layers,
                self.ops.top_k,
            )
            if any(
                tuple(value.shape) != expected_topk_shape
                for value in (
                    oracle_component_state.topk_ids,
                    oracle_component_state.topk_weights,
                )
            ):
                raise ValueError("natural component top-k shape mismatch")
            if not all(bool(torch.isfinite(value).all()) for value in component_values):
                raise ValueError("natural component state contains non-finite values")
            if not bool(torch.isfinite(oracle_component_state.router_logits).all()) or not bool(
                torch.isfinite(oracle_component_state.topk_weights).all()
            ):
                raise ValueError("natural component routes contain non-finite values")
            if not all(
                bool(oracle_component_state.audit.get(key, False))
                for key in (
                    "production_cache_signature_unchanged",
                    "production_rng_unchanged",
                    "shadow_cache_discarded",
                    "full_native_expert_access",
                )
            ):
                raise ValueError("natural component capture audit is incomplete")
        elif oracle_component_source is not None:
            raise ValueError("diagnostic component source requires natural component state")
        correction_coefficients = (
            self.variant.correction_anchor_coefficients
            if self.variant.correction_anchor_coefficients is not None
            else tuple(float(index) for index in range(1, len(self.anchors) + 1))
        )
        if hidden_correction or residual_correction:
            if len(correction_coefficients) != len(self.anchors) or not all(
                value >= 0 and bool(torch.isfinite(torch.tensor(value)))
                for value in correction_coefficients
            ):
                raise ValueError("state correction coefficients must be finite and match anchors")
            cap = self.variant.correction_max_relative_delta_norm
            if cap is not None and (cap <= 0 or not bool(torch.isfinite(torch.tensor(cap)))):
                raise ValueError("state correction norm cap must be finite and positive")
        elif (
            self.variant.correction_anchor_coefficients is not None
            or self.variant.correction_max_relative_delta_norm is not None
        ):
            raise ValueError("state correction schedule requires an enabled correction")
        router_input_deltas = (
            recent_router_inputs[1] - recent_router_inputs[0]
            if recent_router_inputs is not None
            else None
        )
        moe_residual_deltas = (
            recent_moe_outputs[1] - recent_moe_outputs[0]
            if recent_moe_outputs is not None
            else None
        )
        if router_input_deltas is not None and not bool(torch.isfinite(router_input_deltas).all()):
            raise ValueError("router-input correction history is non-finite")
        if moe_residual_deltas is not None and not bool(torch.isfinite(moe_residual_deltas).all()):
            raise ValueError("MoE-output correction history is non-finite")
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
                midlayer_audit: dict[str, object] = {}
                shadow_attention_outputs: dict[int, Tensor] = {}
                shadow_router_inputs: dict[int, Tensor] = {}
                shadow_layer_outputs: dict[int, Tensor] = {}
                if next_token_logits is not None:
                    expected_embedding = self._expected_embedding(
                        next_token_logits.to(device),
                        expected_top_m,
                        sampled_next_token_id,
                    )
                if self.variant.content == "self_topk_particles":
                    (
                        logits,
                        probabilities,
                        topk_ids,
                        topk_weights,
                        shadow_executed_topk_ids,
                        shadow_moe_residuals,
                    ) = self._particle_rollout(
                        production_cache,
                        positions,
                        sampled_next_token_id,
                        execution_subsets,
                    )
                elif self.variant.content.startswith("self_"):
                    (
                        logits,
                        probabilities,
                        topk_ids,
                        topk_weights,
                        shadow_executed_topk_ids,
                        shadow_moe_residuals,
                    ) = self._autoregressive_rollout(
                        production_cache,
                        positions,
                        sampled_next_token_id,
                        expected_top_m,
                        execution_subsets,
                    )
                elif self.variant.attention == "independent":
                    (
                        logits,
                        probabilities,
                        topk_ids,
                        topk_weights,
                        shadow_executed_topk_ids,
                        shadow_moe_residuals,
                    ) = self._independent_rollout(
                        production_cache,
                        positions,
                        sampled_next_token_id,
                        current_token_id,
                        expected_embedding,
                        anchor_token_ids,
                        provided_residuals,
                        execution_subsets,
                    )
                else:
                    (
                        logits,
                        probabilities,
                        topk_ids,
                        topk_weights,
                        shadow_executed_topk_ids,
                        shadow_moe_residuals,
                        shadow_attention_outputs,
                        shadow_router_inputs,
                        shadow_layer_outputs,
                        midlayer_audit,
                    ) = self._causal_rollout(
                        production_cache,
                        positions,
                        sampled_next_token_id,
                        current_token_id,
                        expected_embedding,
                        anchor_token_ids,
                        provided_residuals,
                        execution_subsets,
                        router_input_deltas,
                        moe_residual_deltas,
                        retrieved_router_inputs,
                        retrieved_moe_residuals,
                        retrieval_similarities,
                        oracle_component_state,
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
        native_execution = self.variant.expert_contribution == "native_expert_execution"
        shadow_values = tuple(shadow_moe_residuals.values())
        shadow_residual_finite = (
            all(bool(torch.isfinite(value).all()) for value in shadow_values)
            if native_execution
            else None
        )
        shadow_residual_nonzero = (
            any(bool(torch.count_nonzero(value)) for value in shadow_values)
            if native_execution
            else None
        )
        escape_counts = (
            {
                layer: len(set(subsets[layer]) - set(execution_subsets[layer]))
                for layer in range(self.ops.num_layers)
            }
            if execution_subsets is not None
            else None
        )
        executed_ids_within_subset = (
            all(
                set(shadow_executed_topk_ids[layer].reshape(-1).tolist())
                <= set(execution_subsets[layer])
                for layer in range(self.ops.num_layers)
            )
            if execution_subsets is not None
            else None
        )
        output_bytes = sum(
            _tensor_bytes(values)
            for values in (
                logits,
                probabilities,
                topk_ids,
                topk_weights,
                shadow_executed_topk_ids,
                shadow_moe_residuals,
                shadow_attention_outputs,
                shadow_router_inputs,
                shadow_layer_outputs,
            )
        )
        temporary_cuda = (
            max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
            if device.type == "cuda"
            else 0
        )
        query_calls = self.ops.num_layers * (
            1 + (len(self.anchors) - 1) * self.variant.particle_count
            if self.variant.content == "self_topk_particles"
            else len(self.anchors)
        )
        attention_calls = (
            query_calls
            if self.variant.attention == "independent" or self.variant.content.startswith("self_")
            else self.ops.num_layers
        )
        return QwenPseudoProbeResult(
            variant=self.variant.key,
            anchors=self.anchors,
            absolute_positions=positions,
            raw_router_logits=logits,
            pre_topk_probabilities=probabilities,
            pseudo_topk_ids=topk_ids,
            pseudo_topk_weights=topk_weights,
            aggregate_utility=utility,
            subsets=subsets,
            cost=QwenProbeCost(
                latency_seconds_measured=elapsed,
                temporary_cuda_bytes_measured=temporary_cuda,
                output_tensor_bytes=output_bytes,
                persistent_default_vector_bytes=(
                    self.defaults.numel() * self.defaults.element_size()
                    if self.defaults is not None
                    else 0
                ),
                residual_bank_input_bytes=(
                    provided_residuals.numel() * provided_residuals.element_size()
                    if provided_residuals is not None
                    else 0
                ),
                attention_queries=query_calls,
                attention_calls=attention_calls,
                router_calls=attention_calls,
                expert_calls=attention_calls if native_execution else 0,
                cpu_gpu_synchronizations=sync_count,
                history_state_input_bytes=sum(
                    value.numel() * value.element_size()
                    for value in (recent_router_inputs, recent_moe_outputs)
                    if value is not None
                ),
                retrieval_state_input_bytes=sum(
                    value.numel() * value.element_size()
                    for value in (
                        retrieved_router_inputs,
                        retrieved_moe_residuals,
                        retrieval_similarities,
                    )
                    if value is not None
                ),
                lm_head_calls=(1 if self.variant.midlayer_self_conditioning != "none" else 0),
                lm_head_queries=(
                    len(self.anchors) - 1
                    if self.variant.midlayer_self_conditioning != "none"
                    else 0
                ),
                midlayer_refreshes=(1 if self.variant.midlayer_self_conditioning != "none" else 0),
                diagnostic_state_input_bytes=(
                    sum(
                        value.numel() * value.element_size()
                        for value in (
                            oracle_component_state.attention_outputs,
                            oracle_component_state.router_inputs,
                            oracle_component_state.moe_residuals,
                            oracle_component_state.layer_outputs,
                            oracle_component_state.router_logits,
                            oracle_component_state.topk_ids,
                            oracle_component_state.topk_weights,
                        )
                    )
                    if oracle_component_state is not None
                    else 0
                ),
            ),
            audit={
                "information_regime": (
                    "diagnostic_future_state_oracle"
                    if diagnostic_state_supplied
                    else "online_post_sample"
                ),
                "deployable": not diagnostic_state_supplied,
                "deployable_inputs": (
                    "diagnostic_exact_future_tokens_and_full_native_component_state"
                    if diagnostic_state_supplied
                    else "sampling_step_logits_and_current_policy_production_cache_only"
                    if boundary_expected
                    else (
                        "caller_supplied_anchor_token_ids_and_current_policy_production_cache"
                        if self.variant.content == "provided_sequence"
                        else (
                            "sampled_next_token_current_policy_production_cache_and_"
                            "current_policy_previous_realized_subset"
                            if native_execution and execution_subsets is not None
                            else (
                                "sampled_next_token_current_policy_production_cache_and_"
                                "full_native_expert_access"
                                if native_execution
                                else (
                                    "sampled_next_token_current_policy_production_cache_and_"
                                    "current_policy_previous_window_residuals"
                                    if provided_residuals is not None
                                    else (
                                        "sampled_next_token_and_current_policy_"
                                        "production_cache_only"
                                    )
                                )
                            )
                        )
                    )
                ),
                "anchor_token_ids_supplied": anchor_token_ids is not None,
                "diagnostic_state_patch": diagnostic_patch,
                "diagnostic_hidden_after_layer": self.variant.diagnostic_hidden_after_layer,
                "diagnostic_component_state_supplied": diagnostic_state_supplied,
                "diagnostic_component_source": oracle_component_source,
                "diagnostic_component_shape": (
                    list(oracle_component_state.layer_outputs.shape)
                    if oracle_component_state is not None
                    else None
                ),
                "diagnostic_oracle_inputs_present": diagnostic_state_supplied,
                "shadow_component_states_complete": (
                    all(
                        set(values) == set(range(self.ops.num_layers))
                        for values in (
                            shadow_attention_outputs,
                            shadow_router_inputs,
                            shadow_moe_residuals,
                            shadow_layer_outputs,
                        )
                    )
                    if diagnostic_state_supplied
                    else None
                ),
                "provided_residual_bank": provided_residuals is not None,
                "provided_residual_source": provided_residual_source,
                "provided_residual_shape": (
                    list(provided_residuals.shape) if provided_residuals is not None else None
                ),
                "hidden_state_correction": self.variant.hidden_state_correction,
                "residual_correction": self.variant.residual_correction,
                "correction_history_source": correction_history_source,
                "recent_router_inputs_shape": (
                    list(recent_router_inputs.shape) if recent_router_inputs is not None else None
                ),
                "recent_moe_outputs_shape": (
                    list(recent_moe_outputs.shape) if recent_moe_outputs is not None else None
                ),
                "correction_history_finite": True,
                "anchor_correction_coefficients": (
                    list(correction_coefficients)
                    if hidden_correction or residual_correction
                    else None
                ),
                "anchor_one_correction_protected": (
                    correction_coefficients[0] == 0
                    if hidden_correction or residual_correction
                    else None
                ),
                "correction_max_relative_delta_norm": (
                    self.variant.correction_max_relative_delta_norm
                    if hidden_correction or residual_correction
                    else None
                ),
                "state_retrieval_target": self.variant.state_retrieval_target,
                "state_retrieval_mix": self.variant.state_retrieval_mix,
                "retrieval_state_source": retrieval_state_source,
                "retrieved_router_inputs_shape": (
                    list(retrieved_router_inputs.shape)
                    if retrieved_router_inputs is not None
                    else None
                ),
                "retrieved_moe_residuals_shape": (
                    list(retrieved_moe_residuals.shape)
                    if retrieved_moe_residuals is not None
                    else None
                ),
                "retrieval_similarities": (
                    retrieval_similarities.detach().float().cpu().tolist()
                    if retrieval_similarities is not None
                    else None
                ),
                "retrieval_state_finite": (
                    bool(torch.isfinite(target_tensor).all()) if target_tensor is not None else None
                ),
                **midlayer_audit,
                "one_causal_forward_per_boundary": (
                    self.variant.attention == "causal"
                    and not self.variant.content.startswith("self_")
                ),
                "shadow_expert_execution": native_execution,
                "execution_subset_source": execution_subset_source,
                "execution_subsets_supplied": execution_subsets is not None,
                "execution_scope": (
                    "previous_realized_window_subset"
                    if execution_subsets is not None
                    else "full_native_topk"
                    if native_execution
                    else None
                ),
                "full_pre_mask_scores_all_experts": all(
                    value.shape[-1] == self.ops.num_experts for value in logits.values()
                ),
                "shadow_residual_finite": shadow_residual_finite,
                "shadow_residual_nonzero": shadow_residual_nonzero,
                "executed_ids_within_supplied_subset": executed_ids_within_subset,
                "next_subset_escape_by_layer": escape_counts,
                "next_subset_escape_expert_slots": (
                    sum(escape_counts.values()) if escape_counts is not None else None
                ),
                "expected_top_m": (
                    expected_top_m
                    if boundary_expected or self.variant.content == "self_expected_top_m"
                    else None
                ),
                "expected_embedding_norm": self.variant.expected_embedding_norm,
                "autoregressive_shadow_content": self.variant.content.startswith("self_"),
                "particle_count": self.variant.particle_count,
                "particle_utility_aggregation": (
                    "normalized_path_probability_weighted"
                    if self.variant.content == "self_topk_particles"
                    else None
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
                "unobserved_default_pairs": (
                    int((self.default_counts == 0).sum())
                    if self.default_counts is not None
                    else None
                ),
            },
            shadow_executed_topk_ids=shadow_executed_topk_ids,
            shadow_moe_residuals=shadow_moe_residuals,
            shadow_attention_outputs=shadow_attention_outputs,
            shadow_router_inputs=shadow_router_inputs,
            shadow_layer_outputs=shadow_layer_outputs,
        )
