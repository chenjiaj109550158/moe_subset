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
from safetensors.torch import save_file
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
    _restore_rng,
    _rng_equal,
    _rng_snapshot,
)
from pseudoroute.benchmark.runner import _encode_saved_rendered_prompt, _load_model
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

Stage = Literal["residual_smoke", "content_smoke", "development", "held_out"]
ResidualVariant = Literal[
    "zero",
    "previous_window_position_aligned",
    "previous_window_last_repeated",
    "previous_window_mean_repeated",
]
ContentVariant = Literal[
    "sampled_repeat_independent",
    "sampled_repeat_causal",
    "recent_sequence_independent",
    "recent_sequence_causal",
    "exact_future_independent",
    "exact_future_causal",
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


def _probe_for_spec(
    model: nn.Module,
    ops: Qwen3MoePrefetchOps,
    spec: PolicySpec,
) -> QwenPseudoEmbeddingProbe | None:
    if spec.role != "pseudo":
        return None
    if spec.content is None or spec.residual is None:
        raise ValueError("pseudo policy spec is incomplete")
    content: PseudoContent = (
        "provided_sequence"
        if spec.content.startswith(("recent", "exact"))
        else "sampled_next_token"
    )
    attention: PseudoAttention = "causal" if spec.content.endswith("causal") else "independent"
    contribution: ExpertContribution = "zero" if spec.residual == "zero" else "provided_residual"
    return QwenPseudoEmbeddingProbe(
        model,
        ops,
        None,
        QwenPseudoVariant(spec.key, content, attention, contribution),
        anchors=tuple(range(1, HORIZON + 1)),
        budget=BUDGET,
    )


def _anchor_ids(
    spec: PolicySpec,
    boundary: int,
    prompt_tokens: tuple[int, ...],
    source_tokens: tuple[int, ...],
) -> tuple[int, ...] | None:
    if spec.content is None or spec.content.startswith("sampled_repeat"):
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
) -> tuple[dict[str, object], dict[str, Tensor]]:
    started = time.time()
    model_config = _source_model(accuracy, physical_gpu)
    inputs = _encode_saved_rendered_prompt(tokenizer, model_config, str(source["rendered_prompt"]))
    source_tokens = tuple(int(value) for value in source["generated_token_ids"])
    route_tokens = min(route_token_cap, len(source_tokens))
    if route_tokens < HORIZON or inputs["input_ids"].shape[1] < HORIZON:
        raise ValueError("residual-window replay requires at least eight prompt and output tokens")
    device = inputs["input_ids"].device
    prompt_rng = _rng_snapshot(device)
    with (
        NativeRouteCaptureContext(ops) as prompt_routes,
        MoeOutputCaptureContext(ops, tail_tokens=HORIZON) as prompt_residuals,
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
    residual_bank = moe_output_bank(prompt_residual_records, expected_layers=ops.num_layers)
    router_input_bank = moe_router_input_bank(
        prompt_residual_records, expected_layers=ops.num_layers
    )
    history = route_scores(prompt_route_records, tail_tokens=HORIZON, experts=ops.num_experts)
    prompt_tokens = tuple(int(value) for value in inputs["input_ids"][0].tolist())
    probe = _probe_for_spec(model, ops, spec)
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(ops.num_layers)}
    metrics: list[dict[str, object]] = []
    probe_costs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    boundaries: list[int] = []
    subsets: list[Tensor] = []
    pseudo_logits: list[Tensor] = []
    pseudo_probabilities: list[Tensor] = []
    residual_norms: list[Tensor] = []
    residual_cosines: list[Tensor] = []
    residual_digests: list[str] = []
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
        norms, cosine = _residual_statistics(transformed, router_input_bank)
        residual_norms.append(norms)
        residual_cosines.append(cosine)
        residual_digests.append(_tensor_sha256(transformed))
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
            if probe is None or spec.selection is None or spec.residual is None:
                raise AssertionError("pseudo policy construction failed")
            anchor_ids = _anchor_ids(spec, boundary, prompt_tokens, source_tokens)
            result = probe.predict(
                cache,
                sampled_next_token_id=source_tokens[boundary],
                current_token_id=(
                    prompt_tokens[-1] if boundary == 0 else source_tokens[boundary - 1]
                ),
                anchor_token_ids=anchor_ids,
                provided_residuals=(transformed if spec.residual != "zero" else None),
                provided_residual_source=(
                    "current_policy_previous_window_" + spec.residual
                    if spec.residual != "zero"
                    else None
                ),
            )
            active = candidate_subsets(result, history, spec.selection)
            probe_costs.append({"boundary": boundary, **asdict(result.cost)})
            audits.append({"boundary": boundary, **result.audit})
            pseudo_logits.append(
                torch.stack([result.raw_router_logits[layer] for layer in range(ops.num_layers)])
            )
            pseudo_probabilities.append(
                torch.stack(
                    [result.pre_topk_probabilities[layer] for layer in range(ops.num_layers)]
                )
            )
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
    }
    if pseudo_logits:
        tensors["pseudo_router_logits"] = torch.stack(pseudo_logits)
        tensors["pseudo_pre_topk_probabilities"] = torch.stack(pseudo_probabilities)
    row: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
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
        "route_tokens": route_tokens,
        "boundaries": boundaries,
        "metrics": metrics,
        "probe_costs": probe_costs,
        "cache_rng_audits": audits,
        "prompt_capture_audit": {
            "last_eight_prompt_tokens": True,
            "native_full_expert_prefill": True,
            "production_rng_unchanged": prompt_rng_unchanged,
            "residual_bank_shape": list(residual_bank.shape),
        },
        "residual_bank_sha256_by_boundary": residual_digests,
        "argmax_matches": argmax_matches,
        "argmax_compared": argmax_compared,
        "argmax_agreement": argmax_matches / argmax_compared if argmax_compared else 1.0,
        "first_argmax_divergence": first_argmax_divergence,
        "first_route_divergence": first_route_divergence,
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
        "router_calls": sum(int(cost["router_calls"]) for cost in costs),
    }


def _ranked_selection(
    aggregates: dict[str, dict[str, int | float]],
    eligible: set[str],
) -> str:
    return sorted(
        eligible,
        key=lambda key: (
            -float(aggregates[key]["mean_selected_mass"]),
            -float(aggregates[key]["mean_route_hit"]),
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
        "route_hit_delta_percentile_95": [route[lo], route[hi]],
        "selected_mass_delta_percentile_95": [mass[lo], mass[hi]],
    }


def _per_sample(row: dict[str, Any]) -> tuple[float, float]:
    hits = sum(int(metric["route_hits"]) for metric in row["metrics"])
    slots = sum(int(metric["route_slots"]) for metric in row["metrics"])
    mass_hit = sum(float(metric["selected_mass_hit"]) for metric in row["metrics"])
    mass_total = sum(float(metric["selected_mass_total"]) for metric in row["metrics"])
    return hits / slots, mass_hit / mass_total


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
        selected = _ranked_selection(aggregates, eligible)
        selection = {
            **decision,
            "selected_candidate": selected,
            "ranking_uses_accuracy": False,
        }
        write_json_atomic(root / "selection.json", selection)
        candidate = aggregates[selected]
        previous = aggregates["reference__previous_route_commitment"]
        static = aggregates["reference__static_frequency"]
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
