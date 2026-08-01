"""Calibration-free native-Qwen recent-content mechanism smoke."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch import Tensor

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    Qwen3MoePrefetchOps,
    physical_expert_bytes,
)
from pseudoroute.benchmark.pseudo_embedding_analysis_config import load_analysis_config
from pseudoroute.benchmark.pseudo_embedding_calibration_free_analysis import (
    Aggregate,
    calibration_free_subsets,
)
from pseudoroute.benchmark.pseudo_embedding_route import (
    _route_metrics,
    _save_tensors_atomic,
    _source_model,
    _source_rows,
    _top_b,
    load_focused_suite_and_manifest,
    run_route_sample,
)
from pseudoroute.benchmark.qwen_pseudo import (
    QwenPseudoEmbeddingProbe,
    QwenPseudoVariant,
)
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

DEFAULT_CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_content_smoke_v1.yaml")
CONFIG_SHA256 = "66cc4643310e8d1b1ed0239c57a02c86664510db16489b03fde90bd7f1c5f7f0"
ANALYSIS_ID = "pseudo_embedding_calibration_free_content_smoke_v1"

SAMPLED = "sampled_repeat_independent_zero"
RECENT_INDEPENDENT = "recent_window_independent_zero"
RECENT_CAUSAL = "recent_window_causal_zero"
RECENT_BLEND = "recent_window_equal_history_blend"
TRUE_INDEPENDENT = "true_future_independent_zero"
TRUE_CAUSAL = "true_future_causal_zero"
DEPLOYABLE = (SAMPLED, RECENT_INDEPENDENT, RECENT_CAUSAL, RECENT_BLEND)
DIAGNOSTIC = (TRUE_INDEPENDENT, TRUE_CAUSAL)
PARENT_CANDIDATES = (
    "zero_pseudo_all_anchors",
    "recent_route_prior_w8",
    "one_known_plus_seven_history",
    "equal_evidence_future_blend",
    "native_topk_core_plus_history_reserve",
)
DIRECT_VARIANTS = (
    QwenPseudoVariant(SAMPLED, "sampled_next_token", "independent", "zero"),
    QwenPseudoVariant(RECENT_INDEPENDENT, "provided_sequence", "independent", "zero"),
    QwenPseudoVariant(RECENT_CAUSAL, "provided_sequence", "causal", "zero"),
    QwenPseudoVariant(TRUE_INDEPENDENT, "provided_sequence", "independent", "zero"),
    QwenPseudoVariant(TRUE_CAUSAL, "provided_sequence", "causal", "zero"),
)


def _load_protocol(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("content-smoke protocol bytes changed")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("content-smoke protocol root must be a mapping")
    row = cast(dict[str, Any], raw)
    if (
        row.get("analysis_id") != ANALYSIS_ID
        or row.get("status") != "protocol_frozen_before_model_execution"
        or row.get("route_token_cap") != 16
        or row.get("artifact_root")
        != "artifacts/pseudo_embedding_calibration_free_content_smoke_v1"
    ):
        raise ValueError("content-smoke frozen scope changed")
    if tuple(item["sample_id"] for item in row["rows"]) != ("test-786", "test-394"):
        raise ValueError("content-smoke sample IDs changed")
    if (
        tuple(row["deployable_variants"]) != DEPLOYABLE
        or tuple(row["diagnostic_oracles"]) != DIAGNOSTIC
    ):
        raise ValueError("content-smoke candidate set changed")
    calibration = cast(dict[str, Any], row["calibration_free"])
    if any(
        calibration[key] != "forbidden"
        for key in (
            "learned_parameters",
            "fitted_coefficients",
            "offline_expert_priors",
            "offline_route_transition_tables",
            "default_vector_values",
        )
    ):
        raise ValueError("content-smoke calibration-free boundary changed")
    return row


def recent_anchor_token_ids(
    boundary: int,
    prompt_token_ids: tuple[int, ...],
    source_token_ids: tuple[int, ...],
    *,
    horizon: int = 8,
) -> tuple[int, ...]:
    """Use sampled-next content plus only already-realized recent content."""
    if boundary < 0 or boundary >= len(source_token_ids):
        raise ValueError("recent-content boundary is outside the saved trajectory")
    history = (*prompt_token_ids, *source_token_ids[:boundary])
    if len(history) < horizon - 1:
        raise ValueError("recent-content history is shorter than the pseudo horizon")
    return (source_token_ids[boundary], *history[-(horizon - 1) :])


def true_future_anchor_token_ids(
    boundary: int,
    source_token_ids: tuple[int, ...],
    *,
    horizon: int = 8,
) -> tuple[int, ...]:
    """Return the explicitly non-deployable exact-content diagnostic sequence."""
    result = source_token_ids[boundary : boundary + horizon]
    if len(result) != horizon:
        raise ValueError("true-future diagnostic requires a full pseudo horizon")
    return result


def anchor_token_provider(
    variant: QwenPseudoVariant,
    boundary: int,
    prompt_token_ids: tuple[int, ...],
    source_token_ids: tuple[int, ...],
) -> tuple[int, ...] | None:
    if variant.key == SAMPLED:
        return None
    if variant.key in (RECENT_INDEPENDENT, RECENT_CAUSAL):
        return recent_anchor_token_ids(boundary, prompt_token_ids, source_token_ids)
    if variant.key in (TRUE_INDEPENDENT, TRUE_CAUSAL):
        return true_future_anchor_token_ids(boundary, source_token_ids)
    raise ValueError(f"unscoped explicit-anchor variant: {variant.key}")


def _history_distribution(
    natural_ids: Tensor,
    natural_weights: Tensor,
    boundary: int,
    layer: int,
    experts: int,
    horizon: int,
) -> Tensor | None:
    if boundary == 0:
        return None
    scores = torch.zeros(experts, dtype=torch.float64)
    for token in range(max(0, boundary - horizon), boundary):
        scores.scatter_add_(
            0,
            natural_ids[token, layer].long(),
            natural_weights[token, layer].double(),
        )
    total = scores.sum()
    if not total:
        raise ValueError("realized route history has zero selected mass")
    return scores / total


def blend_subsets(
    probabilities: Tensor,
    natural_ids: Tensor,
    natural_weights: Tensor,
    boundaries: tuple[int, ...],
    *,
    horizon: int,
    budget: int,
    experts: int,
) -> Tensor:
    """Apply the frozen equal-evidence pseudo/history formula without fitted values."""
    if probabilities.ndim != 4 or tuple(probabilities.shape[2:]) != (horizon, experts):
        raise ValueError("recent-content probability tensor has the wrong shape")
    if probabilities.shape[0] != len(boundaries):
        raise ValueError("recent-content boundaries and probability windows differ")
    result: list[list[tuple[int, ...]]] = []
    for boundary_index, boundary in enumerate(boundaries):
        by_layer = []
        for layer in range(probabilities.shape[1]):
            pseudo = probabilities[boundary_index, layer].double()
            history = _history_distribution(
                natural_ids,
                natural_weights,
                boundary,
                layer,
                experts,
                horizon,
            )
            utility = pseudo.sum(dim=0)
            if history is not None:
                utility = pseudo[0] + 0.5 * pseudo[1:].sum(dim=0) + 0.5 * (horizon - 1) * history
            by_layer.append(_top_b(utility, budget))
        result.append(by_layer)
    return torch.tensor(result, dtype=torch.int64)


def _append_blend_metrics(
    row: dict[str, object],
    tensors: dict[str, Tensor],
    ops: Qwen3MoePrefetchOps,
) -> float:
    started = time.perf_counter()
    boundaries = tuple(int(value) for value in tensors["boundaries"].tolist())
    subsets = blend_subsets(
        tensors[f"{RECENT_INDEPENDENT}__pre_topk_probabilities"],
        tensors["router_topk_ids"],
        tensors["router_topk_weights"],
        boundaries,
        horizon=8,
        budget=32,
        experts=128,
    )
    tensors[f"{RECENT_BLEND}__subsets"] = subsets
    metrics = cast(list[dict[str, object]], row["metrics"])
    resident: dict[int, tuple[int, ...]] = {layer: () for layer in range(ops.num_layers)}
    expert_bytes = physical_expert_bytes(ops, 0)
    route_tokens = int(cast(Any, row["route_tokens"]))
    for boundary_index, boundary in enumerate(boundaries):
        end = min(boundary + 8, route_tokens)
        for layer in range(ops.num_layers):
            subset = tuple(int(value) for value in subsets[boundary_index, layer].tolist())
            metrics.append(
                _route_metrics(
                    sample_id=str(row["sample_id"]),
                    boundary=boundary,
                    layer=layer,
                    method=RECENT_BLEND,
                    ids=tensors["router_topk_ids"][boundary:end, layer],
                    weights=tensors["router_topk_weights"][boundary:end, layer],
                    logits=tensors["router_logits"][boundary:end, layer],
                    subset=subset,
                    previous_subset=resident[layer],
                    expert_bytes=expert_bytes,
                )
            )
            resident[layer] = subset
    row["derived_methods"] = [RECENT_BLEND]
    return time.perf_counter() - started


def _sample_paths(output: Path, row_index: int) -> tuple[Path, Path]:
    root = output / "route/samples" / f"{row_index:05d}"
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid_sample(
    output: Path,
    row_index: int,
    sample_id: str,
) -> dict[str, Any] | None:
    json_path, tensor_path = _sample_paths(output, row_index)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial content-smoke sample artifact: {json_path}")
    row = cast(dict[str, Any], json.loads(json_path.read_text(encoding="utf-8")))
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_id") != sample_id
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"corrupt or incompatible content-smoke sample: {json_path}")
    return row


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, int | float]]:
    totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "rows": 0,
            "route_hits": 0,
            "route_slots": 0,
            "selected_mass_hit": 0.0,
            "selected_mass_total": 0.0,
            "fallback_loads": 0,
            "prefetch_loads": 0,
            "subset_churn": 0,
        }
    )
    for row in rows:
        target = totals[str(row["method"])]
        for key in ("rows", "route_hits", "route_slots", "fallback_loads", "prefetch_loads"):
            target[key] = (
                int(target[key]) + int(cast(Any, row[key]))
                if key != "rows"
                else int(target[key]) + 1
            )
        target["subset_churn"] = int(target["subset_churn"]) + int(cast(Any, row["subset_churn"]))
        target["selected_mass_hit"] = float(target["selected_mass_hit"]) + float(
            cast(Any, row["selected_mass_hit"])
        )
        target["selected_mass_total"] = float(target["selected_mass_total"]) + float(
            cast(Any, row["selected_mass_total"])
        )
    result: dict[str, dict[str, int | float]] = {}
    for method, target in totals.items():
        slots = int(target["route_slots"])
        selected_total = float(target["selected_mass_total"])
        result[method] = {
            **target,
            "mean_route_hit": int(target["route_hits"]) / slots,
            "mean_selected_mass": float(target["selected_mass_hit"]) / selected_total,
            "fallback_frequency": int(target["fallback_loads"]) / slots,
            "estimated_transfer_reduction": 1
            - (int(target["fallback_loads"]) + int(target["prefetch_loads"])) / slots,
        }
    return result


def _per_anchor(
    samples: list[tuple[dict[str, Any], dict[str, Tensor]]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    states: dict[tuple[str, int], dict[str, float | int]] = defaultdict(
        lambda: {"hits": 0, "slots": 0, "mass_hit": 0.0, "mass_total": 0.0}
    )
    for row, tensors in samples:
        boundaries = tuple(int(value) for value in tensors["boundaries"].tolist())
        route_tokens = int(row["route_tokens"])
        for method in (*DEPLOYABLE, *DIAGNOSTIC):
            subsets = tensors[f"{method}__subsets"]
            for boundary_index, boundary in enumerate(boundaries):
                for anchor in range(min(8, route_tokens - boundary)):
                    token = boundary + anchor
                    for layer in range(tensors["router_topk_ids"].shape[1]):
                        allowed = set(subsets[boundary_index, layer].tolist())
                        ids = tensors["router_topk_ids"][token, layer]
                        weights = tensors["router_topk_weights"][token, layer].double()
                        mask = torch.tensor([int(value) in allowed for value in ids])
                        state = states[(method, anchor + 1)]
                        state["hits"] = int(state["hits"]) + int(mask.sum())
                        state["slots"] = int(state["slots"]) + ids.numel()
                        state["mass_hit"] = float(state["mass_hit"]) + float(weights[mask].sum())
                        state["mass_total"] = float(state["mass_total"]) + float(weights.sum())
    result: dict[str, dict[str, dict[str, int | float]]] = {}
    for method in (*DEPLOYABLE, *DIAGNOSTIC):
        result[method] = {}
        for anchor in range(1, 9):
            state = states[(method, anchor)]
            result[method][str(anchor)] = {
                **state,
                "mean_route_hit": int(state["hits"]) / int(state["slots"]),
                "mean_selected_mass": float(state["mass_hit"]) / float(state["mass_total"]),
            }
    return result


def _parent_candidate_replication(
    samples: list[tuple[dict[str, Any], dict[str, Tensor]]],
) -> dict[str, object]:
    """Replicate only formulas frozen in the parent protocol; do not rerank this smoke."""
    totals = {key: Aggregate() for key in PARENT_CANDIDATES}
    by_boundary = {(key, boundary): Aggregate() for key in PARENT_CANDIDATES for boundary in (0, 8)}
    by_sample = {
        (key, str(row["sample_id"])): Aggregate() for row, _ in samples for key in PARENT_CANDIDATES
    }
    for row, tensors in samples:
        sample_id = str(row["sample_id"])
        ids = tensors["router_topk_ids"]
        weights = tensors["router_topk_weights"]
        probabilities = tensors[f"{SAMPLED}__pre_topk_probabilities"]
        resident: dict[str, dict[int, tuple[int, ...]]] = {
            key: {layer: () for layer in range(48)} for key in PARENT_CANDIDATES
        }
        for boundary_index, boundary_value in enumerate(tensors["boundaries"].tolist()):
            boundary = int(boundary_value)
            subsets = calibration_free_subsets(
                probabilities[boundary_index],
                ids,
                weights,
                boundary,
                horizon=8,
                budget=32,
                experts=128,
                top_k=8,
            )
            end = min(boundary + 8, ids.shape[0])
            for key in PARENT_CANDIDATES:
                for layer in range(48):
                    arguments = (
                        ids[boundary:end, layer],
                        weights[boundary:end, layer],
                        subsets[key][layer],
                        resident[key][layer],
                    )
                    totals[key].update(*arguments)
                    by_boundary[(key, boundary)].update(*arguments)
                    by_sample[(key, sample_id)].update(*arguments)
                    resident[key][layer] = subsets[key][layer]
    return {
        "role": "predeclared_parent_formula_replication_not_content_smoke_ranking",
        "source_parent_config_fingerprint": (
            "c09d70901bacafabdbacda398390574f7c24f67f634909d39b2084be3e04f898"
        ),
        "aggregates": {key: totals[key].as_dict() for key in PARENT_CANDIDATES},
        "by_boundary": {
            key: {str(boundary): by_boundary[(key, boundary)].as_dict() for boundary in (0, 8)}
            for key in PARENT_CANDIDATES
        },
        "by_sample": {
            key: {
                str(row["sample_id"]): by_sample[(key, str(row["sample_id"]))].as_dict()
                for row, _ in samples
            }
            for key in PARENT_CANDIDATES
        },
        "excluded_from_deployable_candidate_ranking": True,
    }


def _stratified_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    groupers: tuple[tuple[str, Callable[[dict[str, object]], str]], ...] = (
        ("by_boundary", lambda row: str(row["boundary"])),
        ("by_sample", lambda row: str(row["sample_id"])),
        ("by_eight_layer_block", lambda row: f"{int(cast(Any, row['layer'])) // 8}"),
    )
    for label, value in groupers:
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[value(row)].append(row)
        result[label] = {
            group: _aggregate_metrics(selected) for group, selected in sorted(groups.items())
        }
    return result


def _centered_cosine(left: Tensor, right: Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float() - left.float().mean(),
            right.float() - right.float().mean(),
            dim=0,
        )
    )


def _router_sensitivity(
    samples: list[tuple[dict[str, Any], dict[str, Tensor]]],
) -> dict[str, object]:
    direct = (*DEPLOYABLE[:-1], *DIAGNOSTIC)
    alignment: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    intervention: dict[str, list[tuple[float, float]]] = defaultdict(list)
    jaccard: dict[str, list[float]] = defaultdict(list)
    anchor_one_max: dict[str, float] = defaultdict(float)
    for _, tensors in samples:
        natural = tensors["router_logits"]
        sampled = tensors[f"{SAMPLED}__raw_router_logits"]
        sampled_subsets = tensors[f"{SAMPLED}__subsets"]
        for variant in direct:
            predicted = tensors[f"{variant}__raw_router_logits"]
            subsets = tensors[f"{variant}__subsets"]
            anchor_one_max[variant] = max(
                anchor_one_max[variant],
                float((predicted[:, :, 0] - sampled[:, :, 0]).abs().max()),
            )
            for boundary_index, boundary_value in enumerate(tensors["boundaries"].tolist()):
                boundary = int(boundary_value)
                for layer in range(48):
                    left = set(sampled_subsets[boundary_index, layer].tolist())
                    right = set(subsets[boundary_index, layer].tolist())
                    jaccard[variant].append(len(left & right) / len(left | right))
                    for anchor in range(8):
                        actual = natural[boundary + anchor, layer]
                        probe = predicted[boundary_index, layer, anchor]
                        alignment[(variant, anchor + 1)].append(
                            (
                                _centered_cosine(probe, actual),
                                len(
                                    set(probe.topk(8).indices.tolist())
                                    & set(actual.topk(8).indices.tolist())
                                )
                                / 8,
                            )
                        )
                        if variant != SAMPLED:
                            baseline = sampled[boundary_index, layer, anchor]
                            scale = float(actual.float().std()) + 1e-12
                            centered_delta = (probe - probe.mean()) - (baseline - baseline.mean())
                            intervention[variant].append(
                                (
                                    float(centered_delta.float().square().mean().sqrt()) / scale,
                                    len(
                                        set(probe.topk(8).indices.tolist())
                                        & set(baseline.topk(8).indices.tolist())
                                    )
                                    / 8,
                                )
                            )
    return {
        "natural_alignment_by_anchor": {
            variant: {
                str(anchor): {
                    "centered_logit_cosine": _mean(row[0] for row in alignment[(variant, anchor)]),
                    "natural_top8_overlap": _mean(row[1] for row in alignment[(variant, anchor)]),
                }
                for anchor in range(1, 9)
            }
            for variant in direct
        },
        "intervention_vs_sampled_repeat": {
            variant: {
                "normalized_centered_logit_rms": _mean(row[0] for row in intervention[variant]),
                "pseudo_top8_overlap": _mean(row[1] for row in intervention[variant]),
                "subset_jaccard": _mean(jaccard[variant]),
                "anchor1_max_absolute_logit_delta": anchor_one_max[variant],
            }
            for variant in direct
            if variant != SAMPLED
        },
    }


def _cost_report(rows: list[dict[str, Any]]) -> dict[str, object]:
    records = [record for row in rows for record in row["probe_costs"]]
    variants: dict[str, dict[str, object]] = {}
    for key in (*DEPLOYABLE[:-1], *DIAGNOSTIC):
        selected = [row for row in records if row["variant"] == key]
        variants[key] = {
            "calls": len(selected),
            "mean_latency_seconds_measured": _mean(
                float(row["latency_seconds_measured"]) for row in selected
            ),
            "max_temporary_cuda_bytes_measured": max(
                int(row["temporary_cuda_bytes_measured"]) for row in selected
            ),
            "attention_queries": sum(int(row["attention_queries"]) for row in selected),
            "attention_calls": sum(int(row["attention_calls"]) for row in selected),
            "router_calls": sum(int(row["router_calls"]) for row in selected),
            "cpu_gpu_synchronizations": sum(
                int(row["cpu_gpu_synchronizations"]) for row in selected
            ),
        }
    variants[RECENT_BLEND] = {
        "calls": len(rows),
        "model_probe_calls": 0,
        "source": RECENT_INDEPENDENT,
        "total_cpu_postprocess_seconds_measured": sum(
            float(row["blend_postprocess_seconds_measured"]) for row in rows
        ),
    }
    return {"variants": variants}


def _cache_rng_audit(rows: list[dict[str, Any]]) -> dict[str, object]:
    records = [record for row in rows for record in row["cache_rng_audits"]]
    for record in records:
        key = str(record["variant"])
        record["diagnostic_future_content_oracle"] = key in DIAGNOSTIC
        record["deployable_forbidden_future_tokens_present"] = key in DIAGNOSTIC
    deployable = [row for row in records if row["variant"] not in DIAGNOSTIC]
    return {
        "calls": len(records),
        "deployable_calls": len(deployable),
        "diagnostic_oracle_calls": len(records) - len(deployable),
        "all_production_cache_sequence_lengths_unchanged": all(
            row["production_cache_sequence_length_before"]
            == row["production_cache_sequence_length_after"]
            for row in records
        ),
        "all_production_cache_signatures_unchanged": all(
            bool(row["production_cache_signature_unchanged"]) for row in records
        ),
        "all_generation_rng_states_unchanged": all(
            bool(row["production_rng_unchanged"]) for row in records
        ),
        "all_shadow_caches_discarded": all(bool(row["shadow_cache_discarded"]) for row in records),
        "deployable_future_token_leakage": any(
            bool(row["deployable_forbidden_future_tokens_present"]) for row in deployable
        ),
        "records": records,
    }


def _decision(aggregates: dict[str, dict[str, int | float]]) -> dict[str, object]:
    ranking = sorted(
        DEPLOYABLE,
        key=lambda key: (
            -float(aggregates[key]["mean_selected_mass"]),
            -float(aggregates[key]["mean_route_hit"]),
            key,
        ),
    )
    new_ranking = [key for key in ranking if key != SAMPLED]
    best_new = new_ranking[0]
    control = aggregates[SAMPLED]
    candidate = aggregates[best_new]
    improved = float(candidate["mean_route_hit"]) > float(control["mean_route_hit"]) and float(
        candidate["mean_selected_mass"]
    ) > float(control["mean_selected_mass"])
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": (
            "MECHANISM_PROMISING_FOR_NEW_FROZEN_DEVELOPMENT" if improved else "STOP/PIVOT"
        ),
        "best_deployable": ranking[0],
        "best_new_deployable": best_new,
        "best_new_route_hit_gain_over_repeat": float(candidate["mean_route_hit"])
        - float(control["mean_route_hit"]),
        "best_new_selected_mass_gain_over_repeat": float(candidate["mean_selected_mass"])
        - float(control["mean_selected_mass"]),
        "diagnostic_oracles_excluded_from_ranking": True,
        "accuracy_used": False,
        "terminal_v1_decision_unchanged": True,
        "result_role": "two_row_mechanism_hypothesis_only",
    }


def _content_sensitivity(
    aggregates: dict[str, dict[str, int | float]],
) -> dict[str, object]:
    def comparison(left: str, right: str) -> dict[str, float]:
        return {
            "route_hit_delta": float(aggregates[left]["mean_route_hit"])
            - float(aggregates[right]["mean_route_hit"]),
            "selected_mass_delta": float(aggregates[left]["mean_selected_mass"])
            - float(aggregates[right]["mean_selected_mass"]),
        }

    return {
        "recent_independent_vs_sampled_repeat": comparison(RECENT_INDEPENDENT, SAMPLED),
        "recent_causal_vs_recent_independent": comparison(RECENT_CAUSAL, RECENT_INDEPENDENT),
        "recent_blend_vs_recent_independent": comparison(RECENT_BLEND, RECENT_INDEPENDENT),
        "true_future_independent_vs_sampled_repeat": comparison(TRUE_INDEPENDENT, SAMPLED),
        "true_future_causal_vs_true_future_independent": comparison(TRUE_CAUSAL, TRUE_INDEPENDENT),
        "interpretation_boundary": (
            "true-future comparisons diagnose content headroom only and are non-deployable"
        ),
    }


def _git_revision() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_head": head, "worktree_clean": not dirty, "pid": os.getpid(), "ppid": os.getppid()}


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_report(
    path: Path,
    aggregates: dict[str, dict[str, int | float]],
    sensitivity: dict[str, object],
    decision: dict[str, object],
    parent_replication: dict[str, object],
) -> None:
    lines = [
        f"# {ANALYSIS_ID}",
        "",
        f"Mechanism result: **{decision['decision']}**. The focused v1 terminal decision remains "
        "**STOP/PIVOT**.",
        "",
        "This is a two-row, 16-token-per-row, native-Qwen teacher-forced route smoke. It is "
        "not held-out evidence, task accuracy, hard closed-loop generation, or a speedup claim.",
        "",
        "## Deployable calibration-free methods",
        "",
    ]
    for key in DEPLOYABLE:
        row = aggregates[key]
        lines.append(
            f"- `{key}`: hit {float(row['mean_route_hit']):.6f}, mass "
            f"{float(row['mean_selected_mass']):.6f}, simulated transfer reduction "
            f"{float(row['estimated_transfer_reduction']):.6f}."
        )
    lines.extend(("", "## Diagnostic content oracles", ""))
    for key in DIAGNOSTIC:
        row = aggregates[key]
        lines.append(
            f"- `{key}`: hit {float(row['mean_route_hit']):.6f}, mass "
            f"{float(row['mean_selected_mass']):.6f}. This uses future true token content and "
            "is excluded from deployable ranking."
        )
    recent = cast(dict[str, float], sensitivity["recent_independent_vs_sampled_repeat"])
    oracle = cast(dict[str, float], sensitivity["true_future_independent_vs_sampled_repeat"])
    parent_rows = cast(dict[str, dict[str, Any]], parent_replication["aggregates"])
    parent_best = max(
        PARENT_CANDIDATES,
        key=lambda key: (
            float(parent_rows[key]["mean_selected_mass"]),
            float(parent_rows[key]["mean_route_hit"]),
        ),
    )
    lines.extend(
        (
            "",
            "## Main comparisons",
            "",
            "- Recent independent content versus sampled repeat: hit "
            f"{recent['route_hit_delta']:+.6f}, "
            f"mass {recent['selected_mass_delta']:+.6f}.",
            f"- True-future independent content headroom versus sampled repeat: hit "
            f"{oracle['route_hit_delta']:+.6f}, mass {oracle['selected_mass_delta']:+.6f}.",
            f"- Parent-formula replication is best for `{parent_best}` at hit "
            f"{float(parent_rows[parent_best]['mean_route_hit']):.6f}, mass "
            f"{float(parent_rows[parent_best]['mean_selected_mass']):.6f}; this diagnostic "
            "does not rerank the frozen content-smoke candidates.",
            "",
            "## Calibration and claim boundary",
            "",
            "Deployable methods use only the sampled next token, current read-only production "
            "cache, last seven already-realized prompt/generated token IDs, and the immediately "
            "preceding realized route window. No learned/fitted value, offline expert prior, "
            "route-transition "
            "table, saved default-vector value, answer, correctness, or task accuracy is used. The "
            "all-zero expert tensor exists only to satisfy the native probe interface. Transfer is "
            "simulated; probe latency and CUDA temporary memory are measured.",
            "",
        )
    )
    _write_bytes_atomic(path, "\n".join(lines).encode())


def _artifact_manifest(output: Path) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded:
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(output / "artifact_manifest.json", result)
    return result


def _validate_manifest(output: Path) -> dict[str, Any]:
    manifest = cast(
        dict[str, Any],
        json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8")),
    )
    if manifest.get("analysis_config_sha256") != CONFIG_SHA256:
        raise ValueError("content-smoke artifact manifest config changed")
    for row in manifest["artifacts"]:
        path = output / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"content-smoke artifact checksum mismatch: {path}")
    return manifest


def validate(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    protocol = _load_protocol(config_path)
    output = Path(protocol["artifact_root"])
    status = cast(
        dict[str, Any],
        json.loads((output / "pipeline_status.json").read_text(encoding="utf-8")),
    )
    if (
        status.get("state") != "complete"
        or status.get("stage") != "report_v1"
        or status.get("analysis_config_sha256") != CONFIG_SHA256
    ):
        raise ValueError("content-smoke pipeline is incomplete")
    samples = []
    for reference in protocol["rows"]:
        row = _valid_sample(
            output,
            int(reference["row_index"]),
            str(reference["sample_id"]),
        )
        if row is None:
            raise ValueError("content-smoke sample is missing")
        samples.append(row)
    manifest = _validate_manifest(output)
    return {
        "state": "valid",
        "decision": status["decision"],
        "samples": len(samples),
        "artifacts": manifest["artifact_count"],
    }


def run(config_path: Path, physical_gpu: int) -> dict[str, object]:
    started = time.perf_counter()
    protocol = _load_protocol(config_path)
    output = Path(protocol["artifact_root"])
    output.mkdir(parents=True, exist_ok=True)
    suite, _ = load_focused_suite_and_manifest(Path(protocol["source_suite"]["config"]))
    if suite.fingerprint() != protocol["source_suite"]["config_fingerprint"]:
        raise ValueError("focused source suite fingerprint changed")
    parent = load_analysis_config(protocol["parent_analysis"]["config"])
    if parent.fingerprint() != protocol["parent_analysis"]["config_fingerprint"]:
        raise ValueError("parent analysis fingerprint changed")
    sources = _source_rows(suite)
    references = tuple(protocol["rows"])
    completed: list[dict[str, Any]] = []
    missing = []
    for reference in references:
        existing = _valid_sample(
            output,
            int(reference["row_index"]),
            str(reference["sample_id"]),
        )
        if existing is None:
            missing.append(reference)
        else:
            completed.append(existing)
    if missing:
        accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
        model_config = _source_model(accuracy, physical_gpu)
        seed_everything(suite.decode.seed)
        torch.cuda.set_device(torch.device(model_config.device))
        model, tokenizer = _load_model(model_config, accuracy)
        ops = Qwen3MoePrefetchOps(model)
        if (ops.num_layers, ops.num_experts, ops.top_k) != (48, 128, 8):
            raise ValueError("runtime Qwen routed-layer facts changed")
        zero_defaults = DefaultVectorArtifact(
            torch.zeros((48, 128), dtype=torch.int64),
            torch.zeros((48, 128, ops.hidden_size), dtype=torch.bfloat16),
            "calibration-free-synthetic-zero-interface-v1",
            "synthetic_zero_interface_not_an_expert_prior",
        )
        probes = tuple(
            QwenPseudoEmbeddingProbe(
                model,
                ops,
                zero_defaults,
                variant,
                anchors=tuple(range(1, 9)),
                budget=32,
            )
            for variant in DIRECT_VARIANTS
        )
        ignored_static = {layer: tuple(range(32)) for layer in range(48)}
        for reference in missing:
            row_index = int(reference["row_index"])
            source = sources[row_index]
            if source["sample_id"] != reference["sample_id"]:
                raise ValueError("content-smoke source row identity changed")
            json_path, tensor_path = _sample_paths(output, row_index)
            try:
                row, tensors = run_route_sample(
                    suite,
                    model,
                    tokenizer,
                    ops,
                    source,
                    probes,
                    ignored_static,
                    partition="mechanism_smoke",
                    max_route_tokens=16,
                    physical_gpu=physical_gpu,
                    anchor_token_provider=anchor_token_provider,
                    include_reference_methods=False,
                )
                blend_seconds = _append_blend_metrics(row, tensors, ops)
            except Exception as error:
                failure = json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")
                write_json_atomic(
                    failure,
                    {
                        "schema_version": 1,
                        "state": "failed",
                        "analysis_id": ANALYSIS_ID,
                        "analysis_config_sha256": CONFIG_SHA256,
                        "row_index": row_index,
                        "sample_id": reference["sample_id"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "pid": os.getpid(),
                        "ppid": os.getppid(),
                    },
                )
                raise
            row.update(
                {
                    "analysis_id": ANALYSIS_ID,
                    "analysis_config_sha256": CONFIG_SHA256,
                    "information_regime": (
                        "deployable_calibration_free_recent_history_plus_separate_true_future_diagnostics"
                    ),
                    "default_vector_values_loaded": False,
                    "synthetic_zero_interface": True,
                    "blend_postprocess_seconds_measured": blend_seconds,
                }
            )
            _save_tensors_atomic(tensor_path, tensors)
            row.update(
                {
                    "tensor_path": str(tensor_path.relative_to(output)),
                    "tensor_bytes": tensor_path.stat().st_size,
                    "tensor_sha256": sha256_file(tensor_path),
                }
            )
            write_json_atomic(json_path, row)
            completed.append(cast(dict[str, Any], row))
            print(
                json.dumps(
                    {
                        "stage": "content_smoke",
                        "sample_id": row["sample_id"],
                        "tokens": row["route_tokens"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del probes, zero_defaults, ops, model
        torch.cuda.empty_cache()
    completed.sort(key=lambda row: int(row["row_index"]))
    samples_with_tensors = []
    all_metrics: list[dict[str, object]] = []
    for row in completed:
        tensor_path = output / str(row["tensor_path"])
        tensors = load_file(str(tensor_path))
        samples_with_tensors.append((row, tensors))
        all_metrics.extend(cast(list[dict[str, object]], row["metrics"]))
    aggregates = _aggregate_metrics(all_metrics)
    if set(aggregates) != set((*DEPLOYABLE, *DIAGNOSTIC)):
        raise ValueError("content-smoke aggregate method set changed")
    per_anchor = _per_anchor(samples_with_tensors)
    parent_replication = _parent_candidate_replication(samples_with_tensors)
    stratified = _stratified_metrics(all_metrics)
    router_sensitivity = _router_sensitivity(samples_with_tensors)
    costs = _cost_report(completed)
    cache_audit = _cache_rng_audit(completed)
    if not all(
        (
            cache_audit["all_production_cache_sequence_lengths_unchanged"],
            cache_audit["all_production_cache_signatures_unchanged"],
            cache_audit["all_generation_rng_states_unchanged"],
            cache_audit["all_shadow_caches_discarded"],
            not cache_audit["deployable_future_token_leakage"],
        )
    ):
        raise RuntimeError("content-smoke cache/RNG/information audit failed")
    decision = _decision(aggregates)
    sensitivity = _content_sensitivity(aggregates)
    write_json_atomic(output / "resolved_config.json", protocol)
    write_json_atomic(output / "resolved_environment.json", _software_hardware())
    write_json_atomic(output / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(output / "aggregates.json", aggregates)
    write_json_atomic(output / "per_anchor.json", per_anchor)
    write_json_atomic(output / "parent_candidate_replication.json", parent_replication)
    write_json_atomic(output / "stratified_metrics.json", stratified)
    write_json_atomic(output / "router_sensitivity.json", router_sensitivity)
    write_json_atomic(output / "probe_costs.json", costs)
    write_json_atomic(output / "cache_rng_audit.json", cache_audit)
    write_json_atomic(output / "content_sensitivity.json", sensitivity)
    write_json_atomic(output / "decision.json", decision)
    raw = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in all_metrics
    ).encode()
    _write_bytes_atomic(output / "raw_metrics.jsonl.gz", gzip.compress(raw, mtime=0))
    _write_report(
        output / "report.md",
        aggregates,
        sensitivity,
        decision,
        parent_replication,
    )
    failed = sorted(str(path.relative_to(output)) for path in output.rglob("*.FAILED.json"))
    resume = {
        "schema_version": 1,
        "state": "complete",
        "samples_expected": 2,
        "samples_validated": len(completed),
        "atomic_json_tensor_pairs": True,
        "checksum_resume": True,
        "failed_markers_preserved": failed,
    }
    write_json_atomic(output / "resume_audit.json", resume)
    provenance = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "source_samples": ["test-786", "test-394"],
        "model": suite.model.model_id,
        "model_revision": suite.model.revision,
        "precision": suite.model.precision,
        "model_loaded_for_sample_generation": True,
        "gpu_used_for_sample_generation": True,
        "model_loaded_this_aggregation_invocation": bool(missing),
        "gpu_used_this_aggregation_invocation": bool(missing),
        "network_downloads": False,
        "route_evaluation": "teacher_forced_saved_v17_trajectory_open_loop",
        "task_accuracy_measured": False,
        "actual_closed_loop_generation": False,
        "future_true_tokens_used_by_deployable_methods": False,
        "future_true_tokens_used_by_diagnostic_oracles": True,
        "diagnostic_oracles_excluded_from_deployable_ranking": True,
        "default_vector_values_loaded_or_used": False,
        "offline_calibration_statistics_used": False,
        "learned_or_fitted_parameters": False,
        "transfer_kind": "simulated",
        "probe_cost_kind": "measured_native_qwen_gpu",
        "analysis_elapsed_seconds_measured": time.perf_counter() - started,
    }
    write_json_atomic(output / "provenance.json", provenance)
    manifest = _artifact_manifest(output)
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": decision["decision"],
        "best_deployable": decision["best_deployable"],
        "samples": len(completed),
        "artifact_count": manifest["artifact_count"],
    }
    write_json_atomic(output / "pipeline_status.json", status)
    validate(config_path)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--physical-gpu", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args.config, args.physical_gpu) if args.command == "run" else validate(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
