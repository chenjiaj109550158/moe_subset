"""Policy-state previous-window MoE-residual pseudo-embedding experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from pseudoroute.benchmark.config import AccuracySuiteConfig, load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import (
    MoeOutputCaptureContext,
    NativeRouteCaptureContext,
    Qwen3MoePrefetchOps,
    SubsetExecutionContext,
    SubsetRouteRecord,
    moe_output_bank,
    moe_router_input_bank,
    physical_expert_bytes,
)
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_content_smoke import (
    recent_anchor_token_ids,
    true_future_anchor_token_ids,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.qwen_pseudo import (
    ExpertContribution,
    PseudoAttention,
    PseudoContent,
    QwenPseudoEmbeddingProbe,
    QwenPseudoProbeResult,
    QwenPseudoVariant,
    StateCorrection,
    StateRetrievalMix,
    StateRetrievalTarget,
    _restore_rng,
    _rng_equal,
    _rng_snapshot,
)
from pseudoroute.benchmark.runner import (
    _encode_saved_rendered_prompt,
    _load_model,
)
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_length,
    _cache_mutation_signature,
    _fork_cache_copy_on_write,
)
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_embedding_qwen_gsm8k_residual_window_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "361ba85b995b1c2d8573816eadf73861ea0f9f89bed5158221ae6e26f87caf7b"
SAMPLES_SHA256 = "cd6957e34f72f69290ebd9cb147be54ccee3a3dcfe6ef5358f6d34dc850d1255"
HORIZON = 8
BUDGET = 32
LAYERS = 48
EXPERTS = 128
TOP_K = 8

Stage = Literal[
    "residual_smoke",
    "content_smoke",
    "development",
    "held_out",
    "executed_subset_smoke",
]
ResidualVariant = Literal[
    "zero",
    "previous_window_position_aligned",
    "previous_window_last_repeated",
    "previous_window_mean_repeated",
]
ContentVariant = Literal[
    "sampled_repeat_independent",
    "sampled_repeat_causal",
    "current_repeat_independent",
    "recent_sequence_independent",
    "recent_sequence_causal",
    "exact_future_independent",
    "exact_future_causal",
    "expected_top4_repeat_independent",
    "expected_top8_repeat_independent",
    "expected_top16_repeat_independent",
    "expected_top8_norm_matched_independent",
    "sampled_then_expected_top8_causal",
    "self_greedy_causal",
    "self_expected_top8_causal",
    "self_top2_particle_probability_weighted",
    "self_top4_particle_probability_weighted",
]
StateRetrievalMode = Literal[
    "none",
    "exact_token_most_recent",
    "exact_token_then_embedding_nearest",
]


@dataclass(frozen=True)
class PolicySpec:
    key: str
    role: Literal["pseudo", "oracle", "previous", "static"]
    residual: ResidualVariant | None = None
    content: ContentVariant | None = None
    selection: str | None = None
    diagnostic: bool = False

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class TokenAlignedStateRetrieval:
    router_inputs: Tensor
    moe_outputs: Tensor
    history_indices: tuple[int, ...]
    similarities: Tensor
    match_kinds: tuple[str, ...]
    latency_seconds_measured: float
    temporary_cuda_bytes_measured: int
    history_tokens: int
    history_state_bank_bytes: int


def _read_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("residual-window analysis config checksum changed")
    if sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("residual-window sample manifest checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = cast(dict[str, Any], json.loads(SAMPLES.read_text(encoding="utf-8")))
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("residual-window protocol identity changed")
    return config, samples


def _top_n(scores: Tensor, count: int, *, exclude: set[int] | None = None) -> tuple[int, ...]:
    blocked = exclude or set()
    ranking = sorted(
        (expert for expert in range(scores.numel()) if expert not in blocked),
        key=lambda expert: (-float(scores[expert]), expert),
    )
    if len(ranking) < count:
        raise ValueError("expert ranking is smaller than the requested selection")
    return tuple(ranking[:count])


def residual_bank_variant(bank: Tensor, variant: ResidualVariant) -> Tensor:
    """Return one frozen H-position residual transform without fitting."""
    if bank.ndim != 3 or bank.shape[0] != HORIZON:
        raise ValueError("previous-window residual bank must be [8, layers, hidden]")
    if variant == "zero":
        return torch.zeros_like(bank)
    if variant == "previous_window_position_aligned":
        return bank
    if variant == "previous_window_last_repeated":
        return bank[-1:].expand_as(bank)
    if variant == "previous_window_mean_repeated":
        return bank.mean(dim=0, keepdim=True).expand_as(bank)
    raise ValueError(f"unknown residual variant: {variant}")


def token_aligned_state_retrieval(
    anchor_token_ids: tuple[int, ...],
    history_token_ids: tuple[int, ...],
    history_router_inputs: Tensor,
    history_moe_outputs: Tensor,
    embedding_weight: Tensor,
    mode: StateRetrievalMode,
) -> TokenAlignedStateRetrieval:
    """Retrieve only current-policy, pre-boundary states for each pseudo anchor."""
    if mode == "none":
        raise ValueError("token-aligned retrieval requires an enabled mode")
    if not anchor_token_ids or not history_token_ids:
        raise ValueError("token-aligned retrieval requires non-empty anchors and history")
    if (
        history_router_inputs.ndim != 3
        or history_moe_outputs.shape != history_router_inputs.shape
        or history_router_inputs.shape[0] != len(history_token_ids)
    ):
        raise ValueError("history states must be aligned [tokens, layers, hidden] banks")
    if history_router_inputs.shape[-1] != embedding_weight.shape[-1]:
        raise ValueError("history state and input embedding hidden sizes differ")
    if not bool(torch.isfinite(history_router_inputs).all()) or not bool(
        torch.isfinite(history_moe_outputs).all()
    ):
        raise ValueError("history state bank is non-finite")

    device = history_router_inputs.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        allocated_before = 0
    started = time.perf_counter()
    candidate_embeddings: Tensor | None = None
    if mode == "exact_token_then_embedding_nearest":
        history_ids = torch.tensor(history_token_ids, device=embedding_weight.device)
        candidate_embeddings = embedding_weight.detach()[history_ids].float()

    indices: list[int] = []
    similarity_values: list[float] = []
    kinds: list[str] = []
    router_values: list[Tensor] = []
    residual_values: list[Tensor] = []
    for anchor_id in anchor_token_ids:
        matches = [
            index for index, token_id in enumerate(history_token_ids) if token_id == anchor_id
        ]
        if matches:
            index = matches[-1]
            similarity = 1.0
            kind = "exact_token_most_recent"
        elif mode == "exact_token_then_embedding_nearest":
            if candidate_embeddings is None:
                raise AssertionError("embedding-nearest candidates are missing")
            query = embedding_weight.detach()[int(anchor_id)].float()[None]
            scores = torch.nn.functional.cosine_similarity(
                candidate_embeddings,
                query.expand_as(candidate_embeddings),
                dim=-1,
            )
            maximum = scores.max()
            tied = torch.nonzero(scores == maximum, as_tuple=False).reshape(-1)
            index = int(tied[-1])
            similarity = float(maximum.clamp(0, 1))
            kind = "embedding_nearest_most_recent_tie"
        else:
            index = -1
            similarity = 0.0
            kind = "missing_exact_zero_weight"
        indices.append(index)
        similarity_values.append(similarity)
        kinds.append(kind)
        if index >= 0:
            router_values.append(history_router_inputs[index])
            residual_values.append(history_moe_outputs[index])
        else:
            router_values.append(torch.zeros_like(history_router_inputs[0]))
            residual_values.append(torch.zeros_like(history_moe_outputs[0]))
    similarities = torch.tensor(similarity_values, device=device, dtype=torch.float32)
    router_inputs = torch.stack(router_values)
    moe_outputs = torch.stack(residual_values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    temporary_cuda = (
        max(0, torch.cuda.max_memory_allocated(device) - allocated_before)
        if device.type == "cuda"
        else 0
    )
    return TokenAlignedStateRetrieval(
        router_inputs=router_inputs,
        moe_outputs=moe_outputs,
        history_indices=tuple(indices),
        similarities=similarities,
        match_kinds=tuple(kinds),
        latency_seconds_measured=elapsed,
        temporary_cuda_bytes_measured=temporary_cuda,
        history_tokens=len(history_token_ids),
        history_state_bank_bytes=(
            history_router_inputs.numel() * history_router_inputs.element_size()
            + history_moe_outputs.numel() * history_moe_outputs.element_size()
        ),
    )


def route_scores(
    steps: list[tuple[SubsetRouteRecord, ...]] | tuple[SubsetRouteRecord, ...],
    *,
    tail_tokens: int | None = None,
    experts: int = EXPERTS,
) -> dict[int, Tensor]:
    """Scatter pre-mask natural selected weights into layer-local expert scores."""
    if not steps:
        raise ValueError("route score source is empty")
    if isinstance(steps[0], SubsetRouteRecord):
        forwards = [steps]
    else:
        forwards = steps
    layers = len(forwards[0])
    scores = {layer: torch.zeros(experts, dtype=torch.float64) for layer in range(layers)}
    for forward in forwards:
        if tuple(record.layer for record in forward) != tuple(range(layers)):
            raise RuntimeError("route records missed or reordered a layer")
        for record in forward:
            ids = record.natural.ids
            weights = record.natural.weights
            if tail_tokens is not None:
                ids = ids[-tail_tokens:]
                weights = weights[-tail_tokens:]
            scores[record.layer].scatter_add_(
                0,
                ids.reshape(-1).long(),
                weights.reshape(-1).double(),
            )
    return scores


def candidate_subsets(
    result: QwenPseudoProbeResult,
    history: dict[int, Tensor],
    method: str,
    *,
    budget: int = BUDGET,
) -> dict[int, tuple[int, ...]]:
    """Apply one predeclared analytic pseudo/history formula."""
    subsets: dict[int, tuple[int, ...]] = {}
    inverse_denominator = sum(1 / anchor for anchor in range(1, HORIZON + 1))
    for layer in range(len(result.pre_topk_probabilities)):
        probabilities = result.pre_topk_probabilities[layer].double()
        if probabilities.shape[0] != HORIZON:
            raise ValueError("candidate formula requires eight pseudo anchors")
        pseudo = probabilities.sum(dim=0)
        previous = history[layer].double()
        if method == "residual_pseudo_only":
            chosen = _top_n(pseudo, budget)
        elif method == "previous_window_route_only":
            chosen = _top_n(previous, budget)
        elif method == "sampled_anchor_one_plus_previous_window_route":
            chosen = _top_n(HORIZON * probabilities[0] + previous, budget)
        elif method == "equal_pseudo_history":
            chosen = _top_n(pseudo + previous, budget)
        elif method == "sampled_top8_core_plus_history_fill":
            core = _top_n(probabilities[0], TOP_K)
            fill = _top_n(previous, budget - len(core), exclude=set(core))
            chosen = (*core, *fill)
        elif method == "first_four_anchor_core_plus_history_fill":
            union = {
                int(expert)
                for anchor in range(4)
                for expert in result.pseudo_topk_ids[layer][anchor].tolist()
            }
            first_four = probabilities[:4].sum(dim=0)
            core = tuple(
                sorted(union, key=lambda expert: (-float(first_four[expert]), expert))[:budget]
            )
            fill = _top_n(previous, budget - len(core), exclude=set(core))
            chosen = (*core, *fill)
        elif method == "anchor_one_top8_plus_later_corrected_utility_fill":
            core = _top_n(probabilities[0], TOP_K)
            later = probabilities[1:].sum(dim=0)
            fill = _top_n(later, budget - len(core), exclude=set(core))
            chosen = (*core, *fill)
        elif method == "linear_horizon_decay":
            weights = torch.arange(HORIZON, 0, -1, dtype=torch.float64) * (HORIZON / 36)
            chosen = _top_n((weights[:, None] * probabilities).sum(dim=0) + previous, budget)
        elif method == "inverse_anchor_decay":
            weights = torch.tensor(
                [HORIZON / (anchor * inverse_denominator) for anchor in range(1, 9)],
                dtype=torch.float64,
            )
            chosen = _top_n((weights[:, None] * probabilities).sum(dim=0) + previous, budget)
        else:
            raise ValueError(f"unknown candidate formula: {method}")
        subsets[layer] = tuple(sorted(chosen))
    return subsets


def _residual_specs() -> tuple[PolicySpec, ...]:
    variants: tuple[ResidualVariant, ...] = (
        "zero",
        "previous_window_position_aligned",
        "previous_window_last_repeated",
        "previous_window_mean_repeated",
    )
    return tuple(
        PolicySpec(
            f"residual__{variant}",
            "pseudo",
            residual=variant,
            content="sampled_repeat_independent",
            selection="residual_pseudo_only",
        )
        for variant in variants
    )


def _content_specs(residual: ResidualVariant) -> tuple[PolicySpec, ...]:
    variants: tuple[ContentVariant, ...] = (
        "sampled_repeat_independent",
        "sampled_repeat_causal",
        "recent_sequence_independent",
        "recent_sequence_causal",
        "exact_future_independent",
        "exact_future_causal",
    )
    return tuple(
        PolicySpec(
            f"content__{variant}",
            "pseudo",
            residual=residual,
            content=variant,
            selection="residual_pseudo_only",
            diagnostic=variant.startswith("exact_future"),
        )
        for variant in variants
    )


def _candidate_specs(
    residual: ResidualVariant,
    content: ContentVariant,
) -> tuple[PolicySpec, ...]:
    methods = (
        "residual_pseudo_only",
        "previous_window_route_only",
        "sampled_anchor_one_plus_previous_window_route",
        "equal_pseudo_history",
        "sampled_top8_core_plus_history_fill",
        "first_four_anchor_core_plus_history_fill",
        "linear_horizon_decay",
        "inverse_anchor_decay",
    )
    candidates = tuple(
        PolicySpec(
            f"candidate__{method}",
            "pseudo",
            residual=residual,
            content=content,
            selection=method,
        )
        for method in methods
    )
    references = (
        PolicySpec("reference__hard_oracle_commitment", "oracle"),
        PolicySpec("reference__previous_route_commitment", "previous"),
        PolicySpec("reference__static_frequency", "static"),
    )
    return (*candidates, *references)


def _tensor_sha256(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _static_subsets(config: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    suite = load_pseudo_embedding_config(config["source_suite"]["config"])
    source = Path(suite.default_vectors.root) / "default_vectors.safetensors"
    if sha256_file(source) != suite.default_vectors.tensor_sha256:
        raise ValueError("static-reference count tensor source checksum changed")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        count = handle.get_tensor("count")
    return {
        layer: tuple(sorted(_top_n(count[layer].double(), BUDGET)))
        for layer in range(count.shape[0])
    }


def _attention_for_spec_content(
    spec_content: str,
    content: PseudoContent,
) -> PseudoAttention:
    return (
        "causal"
        if spec_content.endswith("causal") or content.startswith("self_")
        else "independent"
    )


def _probe_for_spec(
    model: nn.Module,
    ops: Qwen3MoePrefetchOps,
    spec: PolicySpec,
    *,
    shadow_expert_execution: bool = False,
    hidden_state_correction: StateCorrection = "none",
    residual_correction: StateCorrection = "none",
    correction_anchor_coefficients: tuple[float, ...] | None = None,
    correction_max_relative_delta_norm: float | None = None,
    state_retrieval_target: StateRetrievalTarget = "none",
    state_retrieval_mix: StateRetrievalMix = "none",
) -> QwenPseudoEmbeddingProbe | None:
    if spec.role != "pseudo":
        return None
    if spec.content is None or spec.residual is None:
        raise ValueError("pseudo policy spec is incomplete")
    if spec.content.startswith(("recent", "exact")):
        content: PseudoContent = "provided_sequence"
    elif spec.content.startswith("current"):
        content = "current_token"
    elif spec.content.startswith("expected"):
        content = "expected_top_m"
    elif spec.content.startswith("sampled_then_expected"):
        content = "sampled_then_expected_top_m"
    elif spec.content.startswith("self_greedy"):
        content = "self_greedy"
    elif spec.content.startswith("self_expected"):
        content = "self_expected_top_m"
    elif spec.content.startswith("self_top"):
        content = "self_topk_particles"
    else:
        content = "sampled_next_token"
    attention = _attention_for_spec_content(spec.content, content)
    contribution: ExpertContribution = (
        "native_expert_execution"
        if shadow_expert_execution
        else "zero"
        if spec.residual == "zero"
        else "provided_residual"
    )
    return QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(
            spec.key,
            content,
            attention,
            contribution,
            (
                "sampled_token"
                if spec.content == "expected_top8_norm_matched_independent"
                else "raw"
            ),
            (
                2
                if spec.content == "self_top2_particle_probability_weighted"
                else 4
                if spec.content == "self_top4_particle_probability_weighted"
                else 1
            ),
            hidden_state_correction,
            residual_correction,
            correction_anchor_coefficients,
            correction_max_relative_delta_norm,
            state_retrieval_target,
            state_retrieval_mix,
        ),
        anchors=tuple(range(1, HORIZON + 1)),
        budget=BUDGET,
    )


def _anchor_ids(
    spec: PolicySpec,
    boundary: int,
    prompt_tokens: tuple[int, ...],
    source_tokens: tuple[int, ...],
) -> tuple[int, ...] | None:
    if spec.content is None or spec.content.startswith(("sampled", "current", "expected", "self")):
        return None
    if spec.content.startswith("recent_sequence"):
        return recent_anchor_token_ids(boundary, prompt_tokens, source_tokens)
    if spec.content.startswith("exact_future"):
        return true_future_anchor_token_ids(boundary, source_tokens)
    raise ValueError(f"unknown content variant: {spec.content}")


def _natural_lookahead(
    model: nn.Module,
    ops: Qwen3MoePrefetchOps,
    cache: object,
    token_ids: tuple[int, ...],
    device: torch.device,
) -> tuple[list[tuple[SubsetRouteRecord, ...]], dict[str, object]]:
    signature = _cache_mutation_signature(cache)
    length = _cache_length(cache)
    rng = _rng_snapshot(device)
    fork = _fork_cache_copy_on_write(cache)
    steps: list[tuple[SubsetRouteRecord, ...]] = []
    try:
        with torch.inference_mode():
            for token_id in token_ids:
                token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                with NativeRouteCaptureContext(ops) as capture:
                    output = cast(
                        Any,
                        model(
                            input_ids=token,
                            past_key_values=fork,
                            use_cache=True,
                            return_dict=True,
                        ),
                    )
                    if output.past_key_values is not fork:
                        raise RuntimeError("oracle fork cache identity changed")
                    steps.append(capture.drain())
        rng_unchanged = _rng_equal(rng, _rng_snapshot(device))
    finally:
        _restore_rng(device, rng)
    cache_unchanged = (
        _cache_length(cache) == length and _cache_mutation_signature(cache) == signature
    )
    if not cache_unchanged or not rng_unchanged:
        raise RuntimeError("teacher-forced oracle lookahead mutated production state")
    return steps, {
        "production_cache_signature_unchanged": cache_unchanged,
        "production_rng_unchanged": rng_unchanged,
        "shadow_cache_discarded": True,
        "future_saved_tokens_used": True,
        "role": "routing_information_oracle_only",
    }


def _hard_forward(
    model: nn.Module,
    ops: Qwen3MoePrefetchOps,
    token_id: int,
    cache: object,
    subset: dict[int, tuple[int, ...]],
    device: torch.device,
) -> tuple[Any, tuple[SubsetRouteRecord, ...], Tensor, Tensor]:
    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with (
        SubsetExecutionContext(ops, "hard", subset) as routes,
        MoeOutputCaptureContext(ops) as residuals,
        torch.inference_mode(),
    ):
        output = cast(
            Any,
            model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            ),
        )
        route_records = routes.drain()
        residual_records = residuals.drain()
    if output.past_key_values is not cache:
        raise RuntimeError("policy replay changed production cache identity")
    return (
        output,
        route_records,
        moe_output_bank(residual_records, expected_layers=ops.num_layers),
        moe_router_input_bank(residual_records, expected_layers=ops.num_layers),
    )


def _route_metric(
    *,
    sample_id: str,
    policy: str,
    boundary: int,
    layer: int,
    records: list[tuple[SubsetRouteRecord, ...]],
    subset: tuple[int, ...],
    previous_subset: tuple[int, ...],
    expert_bytes: int,
) -> dict[str, object]:
    ids = torch.cat([step[layer].natural.ids for step in records], dim=0)
    weights = torch.cat([step[layer].natural.weights for step in records], dim=0).float()
    logits = torch.cat([step[layer].natural.logits for step in records], dim=0).float()
    allowed = torch.tensor(subset, dtype=torch.long)
    mask = torch.isin(ids, allowed)
    slots = int(ids.numel())
    hits = int(mask.sum())
    selected_hit = float(weights.masked_select(mask).double().sum())
    selected_total = float(weights.double().sum())
    full = logits.double().softmax(dim=-1)
    full_hit = float(full[:, list(subset)].sum())
    full_total = float(full.sum())
    prefetch_loads = len(set(subset) - set(previous_subset))
    fallback_loads = slots - hits
    reference_bytes = slots * expert_bytes
    simulated_lossless_bytes = (prefetch_loads + fallback_loads) * expert_bytes
    margin = logits.topk(TOP_K + 1, dim=-1).values
    executed_changed = sum(
        not torch.equal(step[layer].natural.ids, step[layer].executed.ids) for step in records
    )
    return {
        "sample_id": sample_id,
        "policy": policy,
        "boundary": boundary,
        "layer": layer,
        "layer_block": layer // 8,
        "realized_tokens": len(records),
        "context_bucket": "0-31" if boundary < 32 else "32-63" if boundary < 64 else "64-127",
        "route_hits": hits,
        "route_slots": slots,
        "route_hit_rate": hits / slots,
        "selected_mass_hit": selected_hit,
        "selected_mass_total": selected_total,
        "selected_mass_coverage": selected_hit / selected_total,
        "full_mass_hit": full_hit,
        "full_mass_total": full_total,
        "full_mass_coverage": full_hit / full_total,
        "mean_router_top8_top9_margin": float((margin[:, 7] - margin[:, 8]).mean()),
        "fallback_loads_simulated": fallback_loads,
        "fallback_frequency": 1 - hits / slots,
        "prefetch_loads": prefetch_loads,
        "subset_churn": len(set(subset).symmetric_difference(previous_subset)),
        "simulated_lossless_transfer_bytes": simulated_lossless_bytes,
        "natural_reference_bytes": reference_bytes,
        "estimated_transfer_reduction": (
            1 - simulated_lossless_bytes / reference_bytes if reference_bytes else 0.0
        ),
        "hard_executed_route_changed_calls": executed_changed,
        "subset": list(subset),
    }


def _residual_statistics(residual: Tensor, router_input: Tensor) -> tuple[Tensor, Tensor]:
    residual_f = residual.float()
    input_f = router_input.float()
    if input_f.device != residual_f.device:
        input_f = input_f.to(residual_f.device)
    norms = residual_f.norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(residual_f, input_f, dim=-1)
    return norms.cpu(), cosine.cpu()


def run_policy_sample(
    config: dict[str, Any],
    suite: Any,
    accuracy: AccuracySuiteConfig,
    model: nn.Module,
    tokenizer: Any,
    ops: Qwen3MoePrefetchOps,
    source: dict[str, Any],
    spec: PolicySpec,
    static_subsets: dict[int, tuple[int, ...]],
    *,
    stage: Stage,
    route_token_cap: int,
    physical_gpu: int,
    shadow_expert_execution: bool = False,
    analysis_id: str = ANALYSIS_ID,
    analysis_config_sha256: str = CONFIG_SHA256,
    sample_manifest_sha256: str = SAMPLES_SHA256,
    hidden_state_correction: StateCorrection = "none",
    residual_correction: StateCorrection = "none",
    correction_anchor_coefficients: tuple[float, ...] | None = None,
    correction_max_relative_delta_norm: float | None = None,
    state_retrieval_mode: StateRetrievalMode = "none",
    state_retrieval_target: StateRetrievalTarget = "none",
    state_retrieval_mix: StateRetrievalMix = "none",
) -> tuple[dict[str, object], dict[str, Tensor]]:
    started = time.time()
    model_config = _source_model(accuracy, physical_gpu)
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, str(source["rendered_prompt"]))
    source_tokens = tuple(int(value) for value in source["generated_token_ids"])
    route_tokens = min(route_token_cap, len(source_tokens))
    if route_tokens < HORIZON or inputs["input_ids"].shape[1] < HORIZON:
        raise ValueError("residual-window replay requires at least eight prompt and output tokens")
    retrieval_enabled = state_retrieval_mode != "none"
    if retrieval_enabled != (state_retrieval_target != "none") or retrieval_enabled != (
        state_retrieval_mix != "none"
    ):
        raise ValueError("state retrieval mode, target, and mixing must be enabled together")
    if retrieval_enabled and (
        not shadow_expert_execution
        or spec.content != "recent_sequence_causal"
        or hidden_state_correction != "none"
        or residual_correction != "none"
    ):
        raise ValueError(
            "token-aligned retrieval requires recent causal native execution without velocity"
        )
    device = inputs["input_ids"].device
    prompt_rng = _rng_snapshot(device)
    with (
        NativeRouteCaptureContext(ops) as prompt_routes,
        MoeOutputCaptureContext(
            ops,
            tail_tokens=None if retrieval_enabled else HORIZON,
        ) as prompt_residuals,
        torch.inference_mode(),
    ):
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
        prompt_route_records = prompt_routes.drain()
        prompt_residual_records = prompt_residuals.drain()
    prompt_rng_unchanged = _rng_equal(prompt_rng, _rng_snapshot(device))
    if not prompt_rng_unchanged:
        raise RuntimeError("prompt capture changed RNG")
    cache = prefill.past_key_values
    if cache is None:
        raise RuntimeError("policy-state replay prefill returned no cache")
    captured_prompt_residuals = moe_output_bank(
        prompt_residual_records, expected_layers=ops.num_layers
    )
    captured_prompt_router_inputs = moe_router_input_bank(
        prompt_residual_records, expected_layers=ops.num_layers
    )
    residual_bank = captured_prompt_residuals[-HORIZON:]
    router_input_bank = captured_prompt_router_inputs[-HORIZON:]
    history = route_scores(prompt_route_records, tail_tokens=HORIZON, experts=ops.num_experts)
    prompt_tokens = tuple(int(value) for value in inputs["input_ids"][0].tolist())
    state_history_token_ids = prompt_tokens if retrieval_enabled else prompt_tokens[-HORIZON:]
    state_history_residuals = captured_prompt_residuals
    state_history_router_inputs = captured_prompt_router_inputs
    if len(state_history_token_ids) != state_history_residuals.shape[0]:
        raise RuntimeError("prompt token and captured state history lengths differ")
    planning_logits = cast(Tensor, prefill.logits[:, -1]).detach()
    probe = _probe_for_spec(
        model,
        ops,
        spec,
        shadow_expert_execution=shadow_expert_execution,
        hidden_state_correction=hidden_state_correction,
        residual_correction=residual_correction,
        correction_anchor_coefficients=correction_anchor_coefficients,
        correction_max_relative_delta_norm=correction_max_relative_delta_norm,
        state_retrieval_target=state_retrieval_target,
        state_retrieval_mix=state_retrieval_mix,
    )
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(ops.num_layers)}
    metrics: list[dict[str, object]] = []
    probe_costs: list[dict[str, object]] = []
    retrieval_costs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    boundaries: list[int] = []
    subsets: list[Tensor] = []
    pseudo_logits: list[Tensor] = []
    pseudo_probabilities: list[Tensor] = []
    residual_norms: list[Tensor] = []
    residual_cosines: list[Tensor] = []
    residual_digests: list[str] = []
    planning_residuals: list[Tensor] = []
    shadow_executed_ids: list[Tensor] = []
    natural_logits: list[Tensor] = []
    natural_ids: list[Tensor] = []
    natural_weights: list[Tensor] = []
    executed_ids: list[Tensor] = []
    executed_weights: list[Tensor] = []
    argmax_matches = 0
    argmax_compared = 0
    first_argmax_divergence: int | None = None
    first_route_divergence: int | None = None
    expert_bytes = {layer: physical_expert_bytes(ops, layer) for layer in range(ops.num_layers)}
    for boundary in range(0, route_tokens, HORIZON):
        end = min(boundary + HORIZON, route_tokens)
        boundaries.append(boundary)
        transformed = residual_bank_variant(residual_bank, spec.residual or "zero")
        planning_residual = transformed
        result: QwenPseudoProbeResult | None = None
        if spec.role == "oracle":
            future, audit = _natural_lookahead(
                model,
                ops,
                cache,
                source_tokens[boundary:end],
                device,
            )
            future_scores = route_scores(future, experts=ops.num_experts)
            active = {
                layer: tuple(sorted(_top_n(future_scores[layer], BUDGET)))
                for layer in range(ops.num_layers)
            }
            audits.append({"boundary": boundary, **audit})
        elif spec.role == "previous":
            active = {
                layer: tuple(sorted(_top_n(history[layer], BUDGET)))
                for layer in range(ops.num_layers)
            }
        elif spec.role == "static":
            active = static_subsets
        else:
            if (
                probe is None
                or spec.selection is None
                or spec.residual is None
                or spec.content is None
            ):
                raise AssertionError("pseudo policy construction failed")
            anchor_ids = _anchor_ids(spec, boundary, prompt_tokens, source_tokens)
            retrieval: TokenAlignedStateRetrieval | None = None
            if retrieval_enabled:
                if anchor_ids is None:
                    raise AssertionError("token-aligned retrieval requires explicit anchors")
                if len(state_history_token_ids) != len(prompt_tokens) + boundary:
                    raise RuntimeError("policy state history length is not boundary-aligned")
                retrieval = token_aligned_state_retrieval(
                    anchor_ids,
                    state_history_token_ids,
                    state_history_router_inputs,
                    state_history_residuals,
                    cast(Any, model).model.embed_tokens.weight,
                    state_retrieval_mode,
                )
                retrieval_costs.append(
                    {
                        "boundary": boundary,
                        "latency_seconds_measured": retrieval.latency_seconds_measured,
                        "temporary_cuda_bytes_measured": (retrieval.temporary_cuda_bytes_measured),
                        "history_tokens": retrieval.history_tokens,
                        "history_state_bank_bytes": retrieval.history_state_bank_bytes,
                        "anchor_token_ids": list(anchor_ids),
                        "history_indices": list(retrieval.history_indices),
                        "similarities": retrieval.similarities.detach().cpu().tolist(),
                        "match_kinds": list(retrieval.match_kinds),
                        "retrieved_indices_precede_boundary": all(
                            index < retrieval.history_tokens
                            for index in retrieval.history_indices
                            if index >= 0
                        ),
                    }
                )
            result = probe.predict(
                cache,
                sampled_next_token_id=source_tokens[boundary],
                current_token_id=(
                    prompt_tokens[-1] if boundary == 0 else source_tokens[boundary - 1]
                ),
                anchor_token_ids=anchor_ids,
                next_token_logits=(
                    planning_logits
                    if spec.content.startswith(("expected", "sampled_then_expected"))
                    else None
                ),
                expected_top_m=(
                    4
                    if spec.content == "expected_top4_repeat_independent"
                    else 16
                    if spec.content == "expected_top16_repeat_independent"
                    else 8
                ),
                provided_residuals=(
                    transformed if spec.residual != "zero" and not shadow_expert_execution else None
                ),
                provided_residual_source=(
                    "current_policy_previous_window_" + spec.residual
                    if spec.residual != "zero" and not shadow_expert_execution
                    else None
                ),
                execution_subsets=(
                    None if not shadow_expert_execution or boundary == 0 else resident
                ),
                execution_subset_source=(
                    "full_native_topk_prefill_access"
                    if shadow_expert_execution and boundary == 0
                    else (
                        "current_policy_previous_realized_window_subset"
                        if shadow_expert_execution
                        else None
                    )
                ),
                recent_router_inputs=(
                    router_input_bank[-2:] if hidden_state_correction != "none" else None
                ),
                recent_moe_outputs=(residual_bank[-2:] if residual_correction != "none" else None),
                correction_history_source=(
                    "current_policy_last_two_prompt_or_realized_mlp_states"
                    if hidden_state_correction != "none" or residual_correction != "none"
                    else None
                ),
                retrieved_router_inputs=(
                    retrieval.router_inputs
                    if retrieval is not None and state_retrieval_target == "router_input"
                    else None
                ),
                retrieved_moe_residuals=(
                    retrieval.moe_outputs
                    if retrieval is not None and state_retrieval_target == "moe_residual"
                    else None
                ),
                retrieval_similarities=(retrieval.similarities if retrieval is not None else None),
                retrieval_state_source=(
                    "current_policy_prompt_and_already_realized_token_aligned_states"
                    if retrieval is not None
                    else None
                ),
            )
            active = candidate_subsets(result, history, spec.selection)
            if shadow_expert_execution:
                planning_residual = torch.stack(
                    [result.shadow_moe_residuals[layer] for layer in range(ops.num_layers)],
                    dim=1,
                )
                shadow_executed_ids.append(
                    torch.stack(
                        [result.shadow_executed_topk_ids[layer] for layer in range(ops.num_layers)],
                        dim=1,
                    )
                )
            probe_costs.append({"boundary": boundary, **asdict(result.cost)})
            audits.append(
                {
                    "boundary": boundary,
                    **result.audit,
                    "retrieval_history_length_matches_realized_context": (
                        retrieval is not None
                        and retrieval.history_tokens == len(prompt_tokens) + boundary
                        if retrieval_enabled
                        else None
                    ),
                    "retrieved_indices_precede_boundary": (
                        all(
                            index < retrieval.history_tokens
                            for index in retrieval.history_indices
                            if index >= 0
                        )
                        if retrieval is not None
                        else None
                    ),
                    "retrieval_state_bank_policy_local": retrieval is not None
                    if retrieval_enabled
                    else None,
                }
            )
            pseudo_logits.append(
                torch.stack([result.raw_router_logits[layer] for layer in range(ops.num_layers)])
            )
            pseudo_probabilities.append(
                torch.stack(
                    [result.pre_topk_probabilities[layer] for layer in range(ops.num_layers)]
                )
            )
        planning_residuals.append(planning_residual.detach().cpu())
        norms, cosine = _residual_statistics(planning_residual, router_input_bank)
        residual_norms.append(norms)
        residual_cosines.append(cosine)
        residual_digests.append(_tensor_sha256(planning_residual))
        subsets.append(
            torch.tensor([active[layer] for layer in range(ops.num_layers)], dtype=torch.int64)
        )
        window_routes: list[tuple[SubsetRouteRecord, ...]] = []
        window_residuals: list[Tensor] = []
        window_router_inputs: list[Tensor] = []
        for position in range(boundary, end):
            output, records, output_bank, input_bank = _hard_forward(
                model,
                ops,
                source_tokens[position],
                cache,
                active,
                device,
            )
            window_routes.append(records)
            window_residuals.append(output_bank)
            window_router_inputs.append(input_bank)
            planning_logits = cast(Tensor, output.logits[:, -1]).detach()
            natural_logits.append(torch.stack([record.natural.logits[0] for record in records]))
            natural_ids.append(torch.stack([record.natural.ids[0] for record in records]))
            natural_weights.append(torch.stack([record.natural.weights[0] for record in records]))
            executed_ids.append(torch.stack([record.executed.ids[0] for record in records]))
            executed_weights.append(torch.stack([record.executed.weights[0] for record in records]))
            if first_route_divergence is None and any(
                not torch.equal(record.natural.ids, record.executed.ids) for record in records
            ):
                first_route_divergence = position
            if position + 1 < len(source_tokens):
                predicted = int(output.logits[:, -1].argmax())
                argmax_compared += 1
                if predicted == source_tokens[position + 1]:
                    argmax_matches += 1
                elif first_argmax_divergence is None:
                    first_argmax_divergence = position + 1
        for layer in range(ops.num_layers):
            metrics.append(
                _route_metric(
                    sample_id=str(source["sample_id"]),
                    policy=spec.key,
                    boundary=boundary,
                    layer=layer,
                    records=window_routes,
                    subset=active[layer],
                    previous_subset=resident[layer],
                    expert_bytes=expert_bytes[layer],
                )
            )
            resident[layer] = active[layer]
        history = route_scores(window_routes, experts=ops.num_experts)
        if retrieval_enabled:
            realized_residuals = torch.cat(window_residuals, dim=0)
            realized_router_inputs = torch.cat(window_router_inputs, dim=0)
            state_history_token_ids = (
                *state_history_token_ids,
                *source_tokens[boundary:end],
            )
            state_history_residuals = torch.cat(
                (state_history_residuals, realized_residuals), dim=0
            )
            state_history_router_inputs = torch.cat(
                (state_history_router_inputs, realized_router_inputs), dim=0
            )
            if not (
                len(state_history_token_ids)
                == state_history_residuals.shape[0]
                == state_history_router_inputs.shape[0]
            ):
                raise RuntimeError("realized token and policy state history lengths differ")
        if end < route_tokens:
            if len(window_residuals) != HORIZON:
                raise RuntimeError("non-final policy window did not realize eight tokens")
            residual_bank = torch.cat(window_residuals, dim=0)
            router_input_bank = torch.cat(window_router_inputs, dim=0)
    tensors: dict[str, Tensor] = {
        "token_ids": torch.tensor(source_tokens[:route_tokens], dtype=torch.int64),
        "boundaries": torch.tensor(boundaries, dtype=torch.int64),
        "subsets": torch.stack(subsets),
        "natural_router_logits": torch.stack(natural_logits),
        "natural_router_topk_ids": torch.stack(natural_ids),
        "natural_router_topk_weights": torch.stack(natural_weights),
        "executed_router_topk_ids": torch.stack(executed_ids),
        "executed_router_topk_weights": torch.stack(executed_weights),
        "planning_residual_norms": torch.stack(residual_norms),
        "planning_residual_router_input_cosine": torch.stack(residual_cosines),
        "planning_moe_residuals": torch.stack(planning_residuals),
    }
    if pseudo_logits:
        tensors["pseudo_router_logits"] = torch.stack(pseudo_logits)
        tensors["pseudo_pre_topk_probabilities"] = torch.stack(pseudo_probabilities)
    if shadow_executed_ids:
        tensors["shadow_executed_topk_ids"] = torch.stack(shadow_executed_ids)
    row: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": analysis_id,
        "analysis_config_sha256": analysis_config_sha256,
        "sample_manifest_sha256": sample_manifest_sha256,
        "stage": stage,
        "policy": spec.key,
        "policy_spec": asdict(spec),
        "policy_spec_fingerprint": spec.fingerprint(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "precision": config["model"]["precision"],
        "row_index": int(source["row_index"]),
        "sample_id": str(source["sample_id"]),
        "evaluation_mode": "teacher_forced_saved_v17_tokens_on_policy_own_hard_subset_state",
        "task_accuracy_measured": False,
        "identity_materialized": False,
        "hard_mask_executed": True,
        "shadow_expert_execution": shadow_expert_execution,
        "hidden_state_correction": hidden_state_correction,
        "residual_correction": residual_correction,
        "correction_anchor_coefficients": (
            list(correction_anchor_coefficients)
            if correction_anchor_coefficients is not None
            else None
        ),
        "correction_max_relative_delta_norm": correction_max_relative_delta_norm,
        "route_tokens": route_tokens,
        "boundaries": boundaries,
        "metrics": metrics,
        "probe_costs": probe_costs,
        "retrieval_costs": retrieval_costs,
        "cache_rng_audits": audits,
        "prompt_capture_audit": {
            "last_eight_prompt_tokens": not retrieval_enabled,
            "full_prompt_tokens": retrieval_enabled,
            "native_full_expert_prefill": True,
            "production_rng_unchanged": prompt_rng_unchanged,
            "residual_bank_shape": list(residual_bank.shape),
            "captured_prompt_state_bank_shape": list(captured_prompt_residuals.shape),
        },
        "residual_bank_sha256_by_boundary": residual_digests,
        "argmax_matches": argmax_matches,
        "argmax_compared": argmax_compared,
        "argmax_agreement": argmax_matches / argmax_compared if argmax_compared else 1.0,
        "first_argmax_divergence": first_argmax_divergence,
        "first_route_divergence": first_route_divergence,
        "state_retrieval_mode": state_retrieval_mode,
        "state_retrieval_target": state_retrieval_target,
        "state_retrieval_mix": state_retrieval_mix,
        "final_state_history_tokens": len(state_history_token_ids),
        "source_v17_row_sha256": sha256_json(source),
        "source_token_ids_sha256": sha256_json(source_tokens),
        "elapsed_seconds_measured": time.time() - started,
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }
    return row, tensors


def _sample_paths(stage: Stage, row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / stage / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _save_tensors_atomic(path: Path, tensors: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(temporary),
    )
    os.replace(temporary, path)


def _load_valid_sample(
    stage: Stage,
    row_index: int,
    sample_id: str,
    spec: PolicySpec,
) -> dict[str, Any] | None:
    json_path, tensor_path = _sample_paths(stage, row_index, spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial residual-window sample artifact: {json_path}")
    row = cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))
    if (
        row.get("state") != "complete"
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("stage") != stage
        or row.get("sample_id") != sample_id
        or row.get("policy_spec_fingerprint") != spec.fingerprint()
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"corrupt or incompatible residual-window sample: {json_path}")
    return row


def _selection(path: Path, field: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"required prior-stage selection is missing: {path}")
    row = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if row.get("analysis_config_sha256") != CONFIG_SHA256:
        raise ValueError(f"prior-stage selection config changed: {path}")
    return str(row[field])


def stage_specs(stage: Stage) -> tuple[PolicySpec, ...]:
    if stage == "residual_smoke":
        return _residual_specs()
    residual = cast(
        ResidualVariant,
        _selection(OUTPUT / "residual_smoke" / "selection.json", "selected_residual"),
    )
    if stage == "content_smoke":
        return _content_specs(residual)
    content = cast(
        ContentVariant,
        _selection(OUTPUT / "content_smoke" / "selection.json", "selected_content"),
    )
    if stage == "development":
        return _candidate_specs(residual, content)
    decision_path = OUTPUT / "development" / "decision.json"
    decision = cast(dict[str, Any], json.loads(decision_path.read_text(encoding="utf-8")))
    if not decision.get("progress_gate_pass"):
        raise RuntimeError("development gate failed; held-out route execution is forbidden")
    selected = _selection(OUTPUT / "development" / "selection.json", "selected_candidate")
    return tuple(
        spec
        for spec in _candidate_specs(residual, content)
        if spec.key == selected or spec.role in {"oracle", "previous", "static"}
    )


def _stage_partition(stage: Stage) -> tuple[str, int]:
    if stage in {"residual_smoke", "content_smoke"}:
        return "mechanism_smoke", 16
    if stage == "development":
        return "development", 128
    return "held_out_route_new", 128


def run_stage(stage: Stage, *, physical_gpu: int) -> list[dict[str, object]]:
    config, manifest = _read_protocol()
    partition, token_cap = _stage_partition(stage)
    references = manifest["partitions"][partition]["rows"]
    specs = stage_specs(stage)
    existing: list[dict[str, object]] = []
    missing: list[tuple[dict[str, Any], PolicySpec]] = []
    for reference in references:
        for spec in specs:
            row = _load_valid_sample(
                stage,
                int(reference["row_index"]),
                str(reference["sample_id"]),
                spec,
            )
            if row is None:
                missing.append((reference, spec))
            else:
                existing.append(cast(dict[str, object], row))
    if not missing:
        return existing
    suite = load_pseudo_embedding_config(config["source_suite"]["config"])
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    model_config = _source_model(accuracy, physical_gpu)
    seed_everything(int(config["decode"]["seed"]))
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed model facts changed")
    static = _static_subsets(config)
    sources = _source_rows(suite)
    completed = list(existing)
    for reference, spec in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        if source["sample_id"] != reference["sample_id"]:
            raise ValueError(f"sample manifest/source mismatch at row {row_index}")
        json_path, tensor_path = _sample_paths(stage, row_index, spec.key)
        try:
            seed_everything(int(config["decode"]["seed"]))
            row, tensors = run_policy_sample(
                config,
                suite,
                accuracy,
                model,
                tokenizer,
                ops,
                source,
                spec,
                static,
                stage=stage,
                route_token_cap=token_cap,
                physical_gpu=physical_gpu,
            )
            _save_tensors_atomic(tensor_path, tensors)
            row["tensor_sha256"] = sha256_file(tensor_path)
            row["tensor_bytes"] = tensor_path.stat().st_size
            write_json_atomic(json_path, row)
            completed.append(row)
        except Exception as error:
            failure = json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")
            write_json_atomic(
                failure,
                {
                    "state": "failed",
                    "analysis_id": ANALYSIS_ID,
                    "analysis_config_sha256": CONFIG_SHA256,
                    "stage": stage,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": spec.key,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return completed


def _aggregate_policy(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    metrics = [metric for row in rows for metric in row["metrics"]]
    hits = sum(int(metric["route_hits"]) for metric in metrics)
    slots = sum(int(metric["route_slots"]) for metric in metrics)
    mass_hit = sum(float(metric["selected_mass_hit"]) for metric in metrics)
    mass_total = sum(float(metric["selected_mass_total"]) for metric in metrics)
    transfer = sum(int(metric["simulated_lossless_transfer_bytes"]) for metric in metrics)
    reference = sum(int(metric["natural_reference_bytes"]) for metric in metrics)
    costs = [cost for row in rows for cost in row["probe_costs"]]
    return {
        "samples": len(rows),
        "boundaries": sum(len(row["boundaries"]) for row in rows),
        "route_hits": hits,
        "route_slots": slots,
        "mean_route_hit": hits / slots,
        "selected_mass_hit": mass_hit,
        "selected_mass_total": mass_total,
        "mean_selected_mass": mass_hit / mass_total,
        "simulated_lossless_transfer_bytes": transfer,
        "natural_reference_bytes": reference,
        "estimated_transfer_reduction": 1 - transfer / reference,
        "mean_probe_latency_seconds": (
            sum(float(cost["latency_seconds_measured"]) for cost in costs) / len(costs)
            if costs
            else 0.0
        ),
        "max_temporary_cuda_bytes": max(
            (int(cost["temporary_cuda_bytes_measured"]) for cost in costs), default=0
        ),
        "attention_queries": sum(int(cost["attention_queries"]) for cost in costs),
        "attention_calls": sum(int(cost.get("attention_calls", 0)) for cost in costs),
        "router_calls": sum(int(cost["router_calls"]) for cost in costs),
        "expert_calls": sum(int(cost.get("expert_calls", 0)) for cost in costs),
        "total_sample_policy_elapsed_seconds": sum(
            float(row["elapsed_seconds_measured"]) for row in rows
        ),
        "mean_sample_policy_elapsed_seconds": sum(
            float(row["elapsed_seconds_measured"]) for row in rows
        )
        / len(rows),
    }


def _ranked_selection(
    aggregates: dict[str, dict[str, int | float]],
    eligible: set[str],
    *,
    include_transfer: bool = False,
) -> str:
    return sorted(
        eligible,
        key=lambda key: (
            -float(aggregates[key]["mean_selected_mass"]),
            -float(aggregates[key]["mean_route_hit"]),
            (-float(aggregates[key]["estimated_transfer_reduction"]) if include_transfer else 0.0),
            float(aggregates[key]["mean_probe_latency_seconds"]),
            key,
        ),
    )[0]


def _audit_pass(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if not row["prompt_capture_audit"]["production_rng_unchanged"]:
            return False
        for audit in row["cache_rng_audits"]:
            if not audit.get("production_cache_signature_unchanged", False):
                return False
            if not audit.get("production_rng_unchanged", False):
                return False
            if not audit.get("shadow_cache_discarded", False):
                return False
    return True


def _paired_bootstrap(
    candidate: dict[str, tuple[float, float]],
    previous: dict[str, tuple[float, float]],
    *,
    draws: int = 10000,
    seed: int = 20260802,
) -> dict[str, object]:
    sample_ids = tuple(sorted(candidate))
    if sample_ids != tuple(sorted(previous)) or not sample_ids:
        raise ValueError("paired bootstrap sample IDs differ or are empty")
    rng = random.Random(seed)
    route: list[float] = []
    mass: list[float] = []
    for _ in range(draws):
        selected = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        route.append(sum(candidate[key][0] - previous[key][0] for key in selected) / len(selected))
        mass.append(sum(candidate[key][1] - previous[key][1] for key in selected) / len(selected))
    route.sort()
    mass.sort()
    lo = int(0.025 * (draws - 1))
    hi = int(0.975 * (draws - 1))
    return {
        "draws": draws,
        "seed": seed,
        "unit": "sample",
        "paired_sample_mean_route_hit_delta": sum(
            candidate[key][0] - previous[key][0] for key in sample_ids
        )
        / len(sample_ids),
        "paired_sample_mean_selected_mass_delta": sum(
            candidate[key][1] - previous[key][1] for key in sample_ids
        )
        / len(sample_ids),
        "route_hit_delta_percentile_95": [route[lo], route[hi]],
        "selected_mass_delta_percentile_95": [mass[lo], mass[hi]],
    }


def _per_sample(row: dict[str, Any]) -> tuple[float, float]:
    hits = sum(int(metric["route_hits"]) for metric in row["metrics"])
    slots = sum(int(metric["route_slots"]) for metric in row["metrics"])
    mass_hit = sum(float(metric["selected_mass_hit"]) for metric in row["metrics"])
    mass_total = sum(float(metric["selected_mass_total"]) for metric in row["metrics"])
    return hits / slots, mass_hit / mass_total


def _observation_aggregate(
    observations: list[dict[str, int | float | str]],
) -> dict[str, int | float]:
    hits = sum(int(row["route_hits"]) for row in observations)
    slots = sum(int(row["route_slots"]) for row in observations)
    mass_hit = sum(float(row["selected_mass_hit"]) for row in observations)
    mass_total = sum(float(row["selected_mass_total"]) for row in observations)
    return {
        "token_layer_observations": len(observations),
        "route_hits": hits,
        "route_slots": slots,
        "mean_route_hit": hits / slots,
        "selected_mass_hit": mass_hit,
        "selected_mass_total": mass_total,
        "mean_selected_mass": mass_hit / mass_total,
    }


def _tertile_thresholds(
    observations: list[dict[str, int | float | str]],
    field: str,
) -> dict[tuple[str, int], tuple[float, float]]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in observations:
        values[(str(row["policy"]), int(row["layer"]))].append(float(row[field]))
    result = {}
    for key, items in values.items():
        tensor = torch.tensor(items, dtype=torch.float64)
        result[key] = (
            float(torch.quantile(tensor, 1 / 3)),
            float(torch.quantile(tensor, 2 / 3)),
        )
    return result


def _bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "middle"
    return "high"


def _route_observations(rows: list[dict[str, Any]]) -> list[dict[str, int | float | str]]:
    observations: list[dict[str, int | float | str]] = []
    for row in rows:
        _json_path, tensor_path = _sample_paths(
            cast(Stage, row["stage"]), int(row["row_index"]), str(row["policy"])
        )
        tensors = load_file(str(tensor_path))
        token_ids = tensors["natural_router_topk_ids"]
        token_weights = tensors["natural_router_topk_weights"].float()
        token_logits = tensors["natural_router_logits"].float()
        subsets = tensors["subsets"]
        boundaries = [int(value) for value in tensors["boundaries"].tolist()]
        norms = tensors["planning_residual_norms"].float()
        cosines = tensors["planning_residual_router_input_cosine"].float()
        for boundary_index, boundary in enumerate(boundaries):
            end = (
                boundaries[boundary_index + 1]
                if boundary_index + 1 < len(boundaries)
                else token_ids.shape[0]
            )
            for token_index in range(boundary, end):
                anchor = token_index - boundary
                for layer in range(token_ids.shape[1]):
                    allowed = set(int(value) for value in subsets[boundary_index, layer])
                    ids = token_ids[token_index, layer]
                    weights = token_weights[token_index, layer]
                    mask = torch.tensor([int(value) in allowed for value in ids])
                    top = token_logits[token_index, layer].topk(TOP_K + 1).values
                    observations.append(
                        {
                            "policy": str(row["policy"]),
                            "sample_id": str(row["sample_id"]),
                            "boundary": boundary,
                            "anchor": anchor + 1,
                            "layer": layer,
                            "layer_block": layer // 8,
                            "context_bucket": (
                                "0-31" if boundary < 32 else "32-63" if boundary < 64 else "64-127"
                            ),
                            "router_margin": float(top[TOP_K - 1] - top[TOP_K]),
                            "residual_norm": float(norms[boundary_index, anchor, layer]),
                            "residual_router_input_cosine": float(
                                cosines[boundary_index, anchor, layer]
                            ),
                            "route_hits": int(mask.sum()),
                            "route_slots": int(ids.numel()),
                            "selected_mass_hit": float(weights[mask].double().sum()),
                            "selected_mass_total": float(weights.double().sum()),
                        }
                    )
    return observations


def _stratified_outputs(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    observations = _route_observations(rows)
    thresholds = {
        "router_margin": _tertile_thresholds(observations, "router_margin"),
        "residual_norm": _tertile_thresholds(observations, "residual_norm"),
        "residual_router_input_cosine": _tertile_thresholds(
            observations, "residual_router_input_cosine"
        ),
    }
    grouped: dict[str, dict[str, dict[str, list[dict[str, int | float | str]]]]] = {
        axis: defaultdict(lambda: defaultdict(list))
        for axis in (
            "anchor",
            "layer",
            "layer_block",
            "context_bucket",
            "router_margin_tertile",
            "residual_norm_tertile",
            "residual_router_input_cosine_tertile",
        )
    }
    for row in observations:
        policy = str(row["policy"])
        layer = int(row["layer"])
        values = {
            "anchor": str(row["anchor"]),
            "layer": str(layer),
            "layer_block": str(row["layer_block"]),
            "context_bucket": str(row["context_bucket"]),
            "router_margin_tertile": _bucket(
                float(row["router_margin"]), thresholds["router_margin"][(policy, layer)]
            ),
            "residual_norm_tertile": _bucket(
                float(row["residual_norm"]), thresholds["residual_norm"][(policy, layer)]
            ),
            "residual_router_input_cosine_tertile": _bucket(
                float(row["residual_router_input_cosine"]),
                thresholds["residual_router_input_cosine"][(policy, layer)],
            ),
        }
        for axis, value in values.items():
            grouped[axis][policy][value].append(row)
    strata: dict[str, object] = {}
    for axis, by_policy in grouped.items():
        strata[axis] = {
            policy: {
                value: _observation_aggregate(items) for value, items in sorted(values.items())
            }
            for policy, values in sorted(by_policy.items())
        }
    strata["per_policy_layer_tertile_thresholds"] = {
        field: {
            f"{policy}/layer-{layer}": list(values)
            for (policy, layer), values in sorted(by_layer.items())
        }
        for field, by_layer in thresholds.items()
    }
    worst_by_policy: dict[str, list[dict[str, int | float | str]]] = defaultdict(list)
    for row in observations:
        worst_by_policy[str(row["policy"])].append(row)
    worst = []
    for policy_rows in worst_by_policy.values():
        worst.extend(
            sorted(
                policy_rows,
                key=lambda row: (
                    float(row["selected_mass_hit"]) / float(row["selected_mass_total"]),
                    float(row["route_hits"]) / float(row["route_slots"]),
                    str(row["sample_id"]),
                    int(row["boundary"]),
                    int(row["anchor"]),
                    int(row["layer"]),
                ),
            )[:100]
        )
    worst_rows = [
        {
            **cast(dict[str, object], row),
            "route_hit_rate": int(row["route_hits"]) / int(row["route_slots"]),
            "selected_mass_coverage": float(row["selected_mass_hit"])
            / float(row["selected_mass_total"]),
        }
        for row in worst
    ]
    return strata, worst_rows


def _write_gzip_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate_stage(stage: Stage) -> dict[str, object]:
    _config, manifest = _read_protocol()
    partition, _ = _stage_partition(stage)
    specs = stage_specs(stage)
    rows: list[dict[str, Any]] = []
    for reference in manifest["partitions"][partition]["rows"]:
        for spec in specs:
            row = _load_valid_sample(
                stage,
                int(reference["row_index"]),
                str(reference["sample_id"]),
                spec,
            )
            if row is None:
                raise RuntimeError(f"missing sample before {stage} aggregation")
            rows.append(row)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
    aggregates = {
        policy: _aggregate_policy(policy_rows) for policy, policy_rows in sorted(by_policy.items())
    }
    root = OUTPUT / stage
    write_json_atomic(
        root / "aggregates.json",
        {
            "analysis_id": ANALYSIS_ID,
            "analysis_config_sha256": CONFIG_SHA256,
            "stage": stage,
            "policies": aggregates,
        },
    )
    _write_gzip_jsonl(
        root / "raw_metrics.jsonl.gz",
        [cast(dict[str, object], metric) for row in rows for metric in row["metrics"]],
    )
    strata, worst = _stratified_outputs(rows)
    write_json_atomic(root / "stratified_metrics.json", strata)
    write_json_atomic(root / "worst_cases.json", worst)
    write_json_atomic(
        root / "cost_report.json",
        {
            policy: {
                key: value
                for key, value in aggregate.items()
                if "probe" in key or "cuda" in key or key in {"attention_queries", "router_calls"}
            }
            for policy, aggregate in aggregates.items()
        },
    )
    audit = {
        "all_pass": _audit_pass(rows),
        "sample_policy_rows": len(rows),
        "production_cache_rng_shadow_audits": True,
        "calibration_artifact_loaded_by_pseudo_probe": False,
    }
    write_json_atomic(root / "cache_rng_information_audit.json", audit)
    decision: dict[str, object] = {
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "stage": stage,
    }
    if stage == "residual_smoke":
        selected = _ranked_selection(aggregates, set(aggregates))
        selected_residual = selected.removeprefix("residual__")
        selection = {
            **decision,
            "selected_policy": selected,
            "selected_residual": selected_residual,
            "ranking_uses_accuracy": False,
        }
        write_json_atomic(root / "selection.json", selection)
        decision.update({"result": "RESIDUAL_SELECTED_FOR_CONTENT_SMOKE", **selection})
    elif stage == "content_smoke":
        eligible = {key for key in aggregates if "exact_future" not in key}
        selected = _ranked_selection(aggregates, eligible)
        selected_content = selected.removeprefix("content__")
        selection = {
            **decision,
            "selected_policy": selected,
            "selected_content": selected_content,
            "diagnostic_oracles_excluded": True,
            "ranking_uses_accuracy": False,
        }
        write_json_atomic(root / "selection.json", selection)
        decision.update({"result": "CONTENT_SELECTED_FOR_DEVELOPMENT", **selection})
    elif stage == "development":
        eligible = {key for key in aggregates if key.startswith("candidate__")}
        selected = _ranked_selection(aggregates, eligible, include_transfer=True)
        selection = {
            **decision,
            "selected_candidate": selected,
            "ranking_uses_accuracy": False,
        }
        write_json_atomic(root / "selection.json", selection)
        candidate = aggregates[selected]
        previous = aggregates["reference__previous_route_commitment"]
        static = aggregates["reference__static_frequency"]
        candidate_samples = {str(row["sample_id"]): _per_sample(row) for row in by_policy[selected]}
        previous_samples = {
            str(row["sample_id"]): _per_sample(row)
            for row in by_policy["reference__previous_route_commitment"]
        }
        bootstrap = _paired_bootstrap(candidate_samples, previous_samples)
        write_json_atomic(root / "paired_bootstrap.json", bootstrap)
        leave_one_out = {}
        for omitted in sorted(candidate_samples):
            retained = [key for key in candidate_samples if key != omitted]
            leave_one_out[omitted] = {
                "candidate_minus_previous_route_hit": sum(
                    candidate_samples[key][0] - previous_samples[key][0] for key in retained
                )
                / len(retained),
                "candidate_minus_previous_selected_mass": sum(
                    candidate_samples[key][1] - previous_samples[key][1] for key in retained
                )
                / len(retained),
            }
        write_json_atomic(root / "leave_one_out.json", leave_one_out)
        route_delta = float(candidate["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_delta = float(candidate["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        gate = (
            route_delta >= 0.05
            and mass_delta >= 0.05
            and float(candidate["mean_route_hit"]) >= float(static["mean_route_hit"])
            and float(candidate["mean_selected_mass"]) >= float(static["mean_selected_mass"])
            and float(candidate["estimated_transfer_reduction"]) >= 0.30
            and audit["all_pass"]
        )
        decision.update(
            {
                "selected_candidate": selected,
                "route_hit_improvement_over_previous": route_delta,
                "selected_mass_improvement_over_previous": mass_delta,
                "progress_gate_pass": gate,
                "decision": "CANDIDATE_FOR_NEW_HELD_OUT_ROUTE" if gate else "STOP/PIVOT",
            }
        )
    else:
        selected = next(key for key in aggregates if key.startswith("candidate__"))
        candidate = aggregates[selected]
        previous = aggregates["reference__previous_route_commitment"]
        oracle = aggregates["reference__hard_oracle_commitment"]
        route_gap = float(oracle["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_gap = float(oracle["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        route_delta = float(candidate["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_delta = float(candidate["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        route_recovery = route_delta / route_gap if route_gap > 0 else 0.0
        mass_recovery = mass_delta / mass_gap if mass_gap > 0 else 0.0
        strong = (
            route_delta >= 0.05
            and mass_delta >= 0.05
            and route_recovery >= 0.25
            and mass_recovery >= 0.25
            and float(candidate["mean_route_hit"]) >= 0.6967
            and float(candidate["mean_selected_mass"]) >= 0.7162
            and float(candidate["estimated_transfer_reduction"]) >= 0.30
            and audit["all_pass"]
        )
        candidate_samples = {str(row["sample_id"]): _per_sample(row) for row in by_policy[selected]}
        previous_samples = {
            str(row["sample_id"]): _per_sample(row)
            for row in by_policy["reference__previous_route_commitment"]
        }
        bootstrap = _paired_bootstrap(candidate_samples, previous_samples)
        write_json_atomic(root / "paired_bootstrap.json", bootstrap)
        decision.update(
            {
                "selected_candidate": selected,
                "route_hit_improvement_over_previous": route_delta,
                "selected_mass_improvement_over_previous": mass_delta,
                "route_hit_oracle_gap_recovery": route_recovery,
                "selected_mass_oracle_gap_recovery": mass_recovery,
                "strong_candidate_gate_pass": strong,
                "decision": "CANDIDATE_FOR_FROZEN_CLOSED_LOOP_PILOT" if strong else "STOP/PIVOT",
            }
        )
    write_json_atomic(root / "decision.json", decision)
    return decision


def validate_stage(stage: Stage) -> dict[str, object]:
    _config, manifest = _read_protocol()
    partition, _ = _stage_partition(stage)
    specs = stage_specs(stage)
    rows = 0
    bytes_total = 0
    for reference in manifest["partitions"][partition]["rows"]:
        for spec in specs:
            loaded = _load_valid_sample(
                stage,
                int(reference["row_index"]),
                str(reference["sample_id"]),
                spec,
            )
            if loaded is None:
                raise RuntimeError(f"missing {stage} row during validation")
            rows += 1
            bytes_total += int(loaded["tensor_bytes"])
    expected = len(manifest["partitions"][partition]["rows"]) * len(specs)
    result = {
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "stage": stage,
        "sample_policy_rows": rows,
        "expected_sample_policy_rows": expected,
        "row_count_pass": rows == expected,
        "checksum_resume_pass": True,
        "tensor_bytes": bytes_total,
    }
    write_json_atomic(OUTPUT / stage / "resume_audit.json", result)
    return result


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate", "validate"))
    parser.add_argument(
        "--stage",
        required=True,
        choices=("residual_smoke", "content_smoke", "development", "held_out"),
    )
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    stage = cast(Stage, args.stage)
    if args.command == "run":
        result: object = run_stage(stage, physical_gpu=args.gpu)
    elif args.command == "aggregate":
        result = aggregate_stage(stage)
    else:
        result = validate_stage(stage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
