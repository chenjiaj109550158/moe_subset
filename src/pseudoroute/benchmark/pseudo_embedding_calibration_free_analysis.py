"""Tensor-only router sensitivity and calibration-free candidate analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor

from pseudoroute.benchmark.pseudo_embedding_analysis_config import (
    AnalysisConfig,
    load_analysis_config,
)
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic

DEFAULT_CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_analysis_v1.yaml")

SOURCE_VARIANTS = {
    "primary": "sampled_next_independent_default_topk",
    "current": "current_token_independent_default_topk",
    "causal": "sampled_next_causal_default_topk",
    "zero": "sampled_next_independent_zero",
    "expected": "expected_top8_independent_default_topk",
}
SOURCE_METHODS = (
    "hard_oracle_commitment",
    "previous_route_commitment",
    "static_frequency",
    *SOURCE_VARIANTS.values(),
)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _top_b(scores: Tensor, budget: int) -> tuple[int, ...]:
    ranking = sorted(
        range(scores.numel()),
        key=lambda expert: (-float(scores[expert]), expert),
    )
    return tuple(sorted(ranking[:budget]))


def _top_b_with_core(scores: Tensor, core: Tensor, budget: int, top_k: int) -> tuple[int, ...]:
    core_ids = sorted(
        range(core.numel()),
        key=lambda expert: (-float(core[expert]), expert),
    )[:top_k]
    core_set = set(core_ids)
    remainder = [
        expert
        for expert in sorted(
            range(scores.numel()),
            key=lambda expert: (-float(scores[expert]), expert),
        )
        if expert not in core_set
    ]
    return tuple(sorted((*core_ids, *remainder[: budget - top_k])))


def _centered_cosine(left: Tensor, right: Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float() - left.float().mean(),
            right.float() - right.float().mean(),
            dim=0,
        )
    )


def _topk_overlap(left: Tensor, right: Tensor, top_k: int) -> float:
    left_ids = set(left.float().topk(top_k).indices.tolist())
    right_ids = set(right.float().topk(top_k).indices.tolist())
    return len(left_ids & right_ids) / top_k


def _id_overlap(left: Tensor, right: Tensor) -> float:
    left_ids = set(left.tolist())
    right_ids = set(right.tolist())
    return len(left_ids & right_ids) / len(left_ids)


@dataclass
class Aggregate:
    route_hits: int = 0
    route_slots: int = 0
    selected_mass_hit: float = 0.0
    selected_mass_total: float = 0.0
    fallback_loads: int = 0
    prefetch_loads: int = 0
    subset_churn: int = 0
    rows: int = 0

    def update(
        self,
        ids: Tensor,
        weights: Tensor,
        subset: tuple[int, ...],
        previous_subset: tuple[int, ...],
    ) -> None:
        subset_set = set(subset)
        mask = torch.tensor([int(expert) in subset_set for expert in ids.reshape(-1)])
        flattened_weights = weights.float().reshape(-1)
        hits = int(mask.sum())
        self.route_hits += hits
        self.route_slots += ids.numel()
        self.selected_mass_hit += float(flattened_weights[mask].double().sum())
        self.selected_mass_total += float(flattened_weights.double().sum())
        self.fallback_loads += ids.numel() - hits
        self.prefetch_loads += len(subset_set - set(previous_subset))
        self.subset_churn += len(subset_set.symmetric_difference(previous_subset))
        self.rows += 1

    def merge(self, other: Aggregate) -> None:
        for field in (
            "route_hits",
            "route_slots",
            "selected_mass_hit",
            "selected_mass_total",
            "fallback_loads",
            "prefetch_loads",
            "subset_churn",
            "rows",
        ):
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def as_dict(self) -> dict[str, int | float]:
        return {
            "rows": self.rows,
            "route_hits": self.route_hits,
            "route_slots": self.route_slots,
            "mean_route_hit": self.route_hits / self.route_slots,
            "selected_mass_hit": self.selected_mass_hit,
            "selected_mass_total": self.selected_mass_total,
            "mean_selected_mass": self.selected_mass_hit / self.selected_mass_total,
            "fallback_loads": self.fallback_loads,
            "fallback_frequency": self.fallback_loads / self.route_slots,
            "prefetch_loads": self.prefetch_loads,
            "subset_churn": self.subset_churn,
            "estimated_transfer_reduction": 1
            - (self.fallback_loads + self.prefetch_loads) / self.route_slots,
        }


def _history_distribution(
    ids: Tensor,
    weights: Tensor,
    boundary: int,
    layer: int,
    experts: int,
    horizon: int,
) -> Tensor | None:
    if boundary == 0:
        return None
    score = torch.zeros(experts, dtype=torch.float64)
    for token in range(max(0, boundary - horizon), boundary):
        score.scatter_add_(
            0,
            ids[token, layer].long(),
            weights[token, layer].double(),
        )
    return score / score.sum()


def calibration_free_subsets(
    probabilities: Tensor,
    natural_ids: Tensor,
    natural_weights: Tensor,
    boundary: int,
    *,
    horizon: int = 8,
    budget: int = 32,
    experts: int = 128,
    top_k: int = 8,
) -> dict[str, dict[int, tuple[int, ...]]]:
    """Construct every frozen candidate without offline calibration values."""
    if probabilities.ndim != 3 or tuple(probabilities.shape[1:]) != (horizon, experts):
        raise ValueError("zero-pseudo probability tensor has the wrong shape")
    result: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)
    for layer in range(probabilities.shape[0]):
        pseudo = probabilities[layer].double()
        fallback = _top_b(pseudo.sum(dim=0), budget)
        result["zero_pseudo_all_anchors"][layer] = fallback
        history = _history_distribution(
            natural_ids,
            natural_weights,
            boundary,
            layer,
            experts,
            horizon,
        )
        if history is None:
            for key in (
                "recent_route_prior_w8",
                "one_known_plus_seven_history",
                "equal_evidence_future_blend",
                "native_topk_core_plus_history_reserve",
            ):
                result[key][layer] = fallback
            continue
        anchor_one = pseudo[0]
        result["recent_route_prior_w8"][layer] = _top_b(history, budget)
        result["one_known_plus_seven_history"][layer] = _top_b(
            anchor_one + (horizon - 1) * history,
            budget,
        )
        result["equal_evidence_future_blend"][layer] = _top_b(
            anchor_one + 0.5 * pseudo[1:].sum(dim=0) + 0.5 * (horizon - 1) * history,
            budget,
        )
        result["native_topk_core_plus_history_reserve"][layer] = _top_b_with_core(
            history,
            anchor_one,
            budget,
            top_k,
        )
    return dict(result)


def _load_source_subsets(
    config: AnalysisConfig,
) -> dict[tuple[str, int, int, str], tuple[int, ...]]:
    path = Path(config.source.artifact_root) / "route/development/raw_metrics.csv.gz"
    result = {}
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            result[
                (
                    str(row["sample_id"]),
                    int(row["boundary"]),
                    int(row["layer"]),
                    str(row["method"]),
                )
            ] = tuple(int(value) for value in json.loads(str(row["subset"])))
    return result


def _validate_source(config: AnalysisConfig) -> dict[str, object]:
    root = Path(config.source.artifact_root)
    manifest_path = root / "artifact_manifest.json"
    if sha256_file(manifest_path) != config.source.artifact_manifest_sha256:
        raise ValueError("source artifact manifest checksum changed")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("config_fingerprint") != config.source.config_fingerprint:
        raise ValueError("source artifact config fingerprint changed")
    for row in cast(list[dict[str, Any]], manifest["artifacts"]):
        path = root / str(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"source artifact changed: {path}")
    for sample in config.source.samples:
        path = Path(sample.tensor_path)
        if sha256_file(path) != sample.tensor_sha256:
            raise ValueError(f"source sample tensor changed: {path}")
    return {
        "state": "valid",
        "source_artifacts_validated": int(manifest["artifact_count"]),
        "sample_tensors_validated": len(config.source.samples),
    }


def _raw_row(
    sample_id: str,
    boundary: int,
    layer: int,
    candidate: str,
    ids: Tensor,
    weights: Tensor,
    subset: tuple[int, ...],
    previous_subset: tuple[int, ...],
) -> dict[str, object]:
    state = Aggregate()
    state.update(ids, weights, subset, previous_subset)
    return {
        "sample_id": sample_id,
        "boundary": boundary,
        "layer": layer,
        "candidate": candidate,
        "realized_tokens": ids.shape[0],
        "subset": list(subset),
        **state.as_dict(),
    }


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _write_markdown_atomic(path: Path, value: str) -> None:
    _write_bytes_atomic(path, value.encode())


def _git_revision() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_head": head, "worktree_clean": not dirty, "pid": os.getpid()}


def _candidate_decision(
    config: AnalysisConfig,
    candidates: dict[str, dict[str, int | float]],
    previous: dict[str, Any],
) -> dict[str, object]:
    references = config.candidate_progress_reference
    rows = []
    for key in (candidate.key for candidate in config.candidate_constructions):
        metrics = candidates[key]
        hit_gain = float(metrics["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_gain = float(metrics["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        checks = {
            "route_hit_improvement_pass": hit_gain >= references.minimum_route_hit_improvement,
            "selected_mass_improvement_pass": mass_gain
            >= references.minimum_selected_mass_improvement,
            "transfer_reduction_pass": float(metrics["estimated_transfer_reduction"])
            >= references.minimum_simulated_transfer_reduction,
            "calibration_free_information_boundary_pass": True,
        }
        rows.append(
            {
                "candidate": key,
                "metrics": metrics,
                "route_hit_improvement_over_previous": hit_gain,
                "selected_mass_improvement_over_previous": mass_gain,
                "checks": checks,
                "eligible_for_new_protocol": all(checks.values()),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(cast(dict[str, Any], row["metrics"])["mean_selected_mass"]),
            -float(cast(dict[str, Any], row["metrics"])["mean_route_hit"]),
            str(row["candidate"]),
        ),
    )
    eligible = [row for row in ranked if row["eligible_for_new_protocol"]]
    return {
        "schema_version": 1,
        "analysis_id": config.analysis_id,
        "config_fingerprint": config.fingerprint(),
        "result_role": references.result_role,
        "accuracy_used": False,
        "candidates": rows,
        "best_observed_candidate": ranked[0]["candidate"],
        "qualifying_candidate": eligible[0]["candidate"] if eligible else None,
        "decision": "CANDIDATE_FOR_NEW_PROTOCOL" if eligible else "STOP/PIVOT",
        "v1_decision_unchanged": True,
    }


def _report(
    config: AnalysisConfig,
    sensitivity: dict[str, object],
    candidates: dict[str, dict[str, int | float]],
    decision: dict[str, object],
    previous: dict[str, Any],
) -> str:
    lines = [
        f"# {config.analysis_id}",
        "",
        f"Exploratory decision: **{decision['decision']}**. The terminal v1 decision remains "
        "**STOP/PIVOT**.",
        "",
        "This is a four-row, tensor-only, calibration-free hypothesis analysis. It is not "
        "held-out evidence, task accuracy, actual closed-loop generation, or a speedup claim.",
        "",
        "## Calibration-free candidates",
        "",
        f"Frozen previous-route reference: hit {float(previous['mean_route_hit']):.6f}, mass "
        f"{float(previous['mean_selected_mass']):.6f}, simulated transfer reduction "
        f"{float(previous['estimated_transfer_reduction']):.6f}.",
        "",
    ]
    decision_rows = {
        str(row["candidate"]): row for row in cast(list[dict[str, Any]], decision["candidates"])
    }
    for key, metrics in candidates.items():
        row = decision_rows[key]
        lines.append(
            f"- `{key}`: hit {float(metrics['mean_route_hit']):.6f} "
            f"({float(row['route_hit_improvement_over_previous']):+.6f}), mass "
            f"{float(metrics['mean_selected_mass']):.6f} "
            f"({float(row['selected_mass_improvement_over_previous']):+.6f}), simulated "
            f"transfer reduction {float(metrics['estimated_transfer_reduction']):.6f}."
        )
    horizon = cast(dict[str, Any], sensitivity["horizon"])
    temporal = cast(dict[str, Any], sensitivity["temporal_persistence"])
    components = cast(dict[str, Any], sensitivity["component_interventions"])
    lines.extend(
        (
            "",
            "## Decisive sensitivity observations",
            "",
            f"- Primary route hit is {horizon['primary']['1']['mean_route_hit']:.6f} at anchor "
            f"1 and {horizon['primary']['8']['mean_route_hit']:.6f} at anchor 8.",
            f"- Natural top-8 overlap from anchor 1 to lag 8 is "
            f"{temporal['natural']['8']['topk_overlap']:.6f}; repeated-token pseudo anchor "
            f"overlap is {temporal['pseudo_position']['8']['topk_overlap']:.6f}.",
            f"- Expected-top-8 changes primary pseudo very little: top-8 overlap "
            f"{components['expected']['pseudo_topk_overlap']:.6f}, subset Jaccard "
            f"{components['expected']['subset_jaccard']:.6f}.",
            f"- Removing default contributions changes pseudo top-8 overlap to "
            f"{components['zero']['pseudo_topk_overlap']:.6f} and subset Jaccard to "
            f"{components['zero']['subset_jaccard']:.6f} relative to primary.",
            "",
            "## Claim boundary",
            "",
            "Candidate formulas use only the sampled next-token zero probe and current-policy "
            "online realized route history. No learned or fitted value, offline prior, default "
            "vector, future token, answer, correctness, or accuracy enters candidate selection. "
            "Source replay is open-loop; transfer is simulated; model and GPU runtime were not "
            "measured in this tensor-only analysis.",
            "",
        )
    )
    return "\n".join(lines)


def _artifact_manifest(config: AnalysisConfig, output: Path) -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    rows = []
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        relative = str(path.relative_to(output))
        if relative in excluded:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1,
        "state": "complete",
        "analysis_id": config.analysis_id,
        "config_fingerprint": config.fingerprint(),
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return manifest


def validate(config: AnalysisConfig) -> dict[str, object]:
    _validate_source(config)
    output = Path(config.artifact_root)
    manifest = cast(
        dict[str, Any],
        json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8")),
    )
    if manifest.get("config_fingerprint") != config.fingerprint():
        raise ValueError("analysis artifact config fingerprint changed")
    for row in cast(list[dict[str, Any]], manifest["artifacts"]):
        path = output / str(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"analysis artifact changed: {path}")
    status = cast(
        dict[str, Any],
        json.loads((output / "pipeline_status.json").read_text(encoding="utf-8")),
    )
    if status.get("state") != "complete":
        raise ValueError("analysis pipeline is not complete")
    return {
        "state": "valid",
        "decision": status["decision"],
        "artifacts": manifest["artifact_count"],
    }


def run(config: AnalysisConfig) -> dict[str, object]:
    started = time.perf_counter()
    source_audit = _validate_source(config)
    output = Path(config.artifact_root)
    output.mkdir(parents=True, exist_ok=True)
    source_subsets = _load_source_subsets(config)
    candidate_keys = tuple(candidate.key for candidate in config.candidate_constructions)
    candidate_global = {key: Aggregate() for key in candidate_keys}
    candidate_sample: dict[tuple[str, str], Aggregate] = defaultdict(Aggregate)
    candidate_layer: dict[tuple[str, int], Aggregate] = defaultdict(Aggregate)
    candidate_horizon: dict[tuple[str, int], Aggregate] = defaultdict(Aggregate)
    source_horizon: dict[tuple[str, int], Aggregate] = defaultdict(Aggregate)
    source_layer: dict[tuple[str, int], Aggregate] = defaultdict(Aggregate)
    component_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    component_jaccard: dict[str, list[float]] = defaultdict(list)
    natural_temporal: dict[int, list[tuple[float, float]]] = defaultdict(list)
    pseudo_position: dict[int, list[tuple[float, float]]] = defaultdict(list)
    margin_events: list[tuple[float, str, float, float]] = []
    raw_candidate_rows: list[dict[str, object]] = []

    for sample in config.source.samples:
        tensors = load_file(sample.tensor_path)
        natural_ids = tensors["router_topk_ids"]
        natural_weights = tensors["router_topk_weights"].float()
        natural_logits = tensors["router_logits"].float()
        boundaries = [int(value) for value in tensors["boundaries"]]
        token_count, layers, top_k = natural_ids.shape
        resident: dict[str, dict[int, tuple[int, ...]]] = {
            key: {layer: tuple() for layer in range(layers)} for key in candidate_keys
        }
        for boundary_index, boundary in enumerate(boundaries):
            end = min(boundary + config.operating_point.horizon, token_count)
            zero_probabilities = tensors[f"{SOURCE_VARIANTS['zero']}__pre_topk_probabilities"][
                boundary_index
            ]
            candidate_subsets = calibration_free_subsets(
                zero_probabilities,
                natural_ids,
                natural_weights,
                boundary,
            )
            for layer in range(layers):
                ids = natural_ids[boundary:end, layer]
                weights = natural_weights[boundary:end, layer]
                for key in candidate_keys:
                    subset = candidate_subsets[key][layer]
                    previous_subset = resident[key][layer]
                    for aggregate in (
                        candidate_global[key],
                        candidate_sample[(key, sample.sample_id)],
                        candidate_layer[(key, layer)],
                    ):
                        aggregate.update(ids, weights, subset, previous_subset)
                    for anchor in range(end - boundary):
                        candidate_horizon[(key, anchor + 1)].update(
                            ids[anchor : anchor + 1],
                            weights[anchor : anchor + 1],
                            subset,
                            previous_subset,
                        )
                    raw_candidate_rows.append(
                        _raw_row(
                            sample.sample_id,
                            boundary,
                            layer,
                            key,
                            ids,
                            weights,
                            subset,
                            previous_subset,
                        )
                    )
                    resident[key][layer] = subset
                for method in SOURCE_METHODS:
                    subset = source_subsets[(sample.sample_id, boundary, layer, method)]
                    source_layer[(method, layer)].update(ids, weights, subset, tuple())
                    for anchor in range(end - boundary):
                        source_horizon[(method, anchor + 1)].update(
                            ids[anchor : anchor + 1],
                            weights[anchor : anchor + 1],
                            subset,
                            tuple(),
                        )

            primary_logits = tensors[f"{SOURCE_VARIANTS['primary']}__raw_router_logits"]
            primary_subsets = tensors[f"{SOURCE_VARIANTS['primary']}__subsets"]
            for layer in range(layers):
                for short in ("current", "causal", "zero", "expected"):
                    other_subset = tensors[f"{SOURCE_VARIANTS[short]}__subsets"][
                        boundary_index, layer
                    ]
                    left_subset = set(primary_subsets[boundary_index, layer].tolist())
                    right_subset = set(other_subset.tolist())
                    component_jaccard[short].append(
                        len(left_subset & right_subset) / len(left_subset | right_subset)
                    )
                for anchor in range(end - boundary):
                    token = boundary + anchor
                    natural = natural_logits[token, layer]
                    scale = float(natural.std()) + 1e-12
                    primary = primary_logits[boundary_index, layer, anchor]
                    margin_values = natural.topk(top_k + 1).values
                    margin_value = float(margin_values[top_k - 1] - margin_values[top_k])
                    for short, variant in SOURCE_VARIANTS.items():
                        predicted = tensors[f"{variant}__raw_router_logits"][
                            boundary_index, layer, anchor
                        ]
                        margin_events.append(
                            (
                                margin_value,
                                short,
                                _centered_cosine(predicted, natural),
                                _id_overlap(
                                    predicted.topk(top_k).indices,
                                    natural_ids[token, layer],
                                ),
                            )
                        )
                    for short in ("current", "causal", "zero", "expected"):
                        other = tensors[f"{SOURCE_VARIANTS[short]}__raw_router_logits"][
                            boundary_index, layer, anchor
                        ]
                        delta = (primary - primary.mean()) - (other - other.mean())
                        component_values[short].append(
                            (
                                float(delta.square().mean().sqrt()) / scale,
                                _topk_overlap(primary, other, top_k),
                            )
                        )
                    if anchor > 0:
                        first = primary_logits[boundary_index, layer, 0]
                        delta = (first - first.mean()) - (primary - primary.mean())
                        pseudo_position[anchor + 1].append(
                            (
                                float(delta.square().mean().sqrt()) / scale,
                                _topk_overlap(first, primary, top_k),
                            )
                        )
            for lag in range(1, config.operating_point.horizon):
                if boundary + lag >= token_count:
                    continue
                for layer in range(layers):
                    natural_temporal[lag + 1].append(
                        (
                            _centered_cosine(
                                natural_logits[boundary, layer],
                                natural_logits[boundary + lag, layer],
                            ),
                            _id_overlap(
                                natural_ids[boundary, layer],
                                natural_ids[boundary + lag, layer],
                            ),
                        )
                    )

    candidate_metrics: dict[str, dict[str, int | float]] = {
        key: candidate_global[key].as_dict() for key in candidate_keys
    }
    source_aggregates = cast(
        dict[str, Any],
        json.loads(
            (Path(config.source.artifact_root) / "route/development/aggregates.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    previous = cast(dict[str, Any], source_aggregates["previous_route_commitment"])
    source_zero = cast(dict[str, Any], source_aggregates[SOURCE_VARIANTS["zero"]])
    reconstructed_zero = candidate_metrics["zero_pseudo_all_anchors"]
    if (
        reconstructed_zero["route_hits"] != source_zero["route_hits"]
        or reconstructed_zero["route_slots"] != source_zero["route_slots"]
        or abs(
            float(reconstructed_zero["mean_selected_mass"])
            - float(source_zero["mean_selected_mass"])
        )
        > 1e-12
    ):
        raise ValueError("zero-pseudo candidate reconstruction changed")
    decision = _candidate_decision(config, candidate_metrics, previous)
    horizon: dict[str, dict[str, dict[str, int | float]]] = {}
    for label, method in {
        "oracle": "hard_oracle_commitment",
        "previous": "previous_route_commitment",
        "primary": SOURCE_VARIANTS["primary"],
        "zero": SOURCE_VARIANTS["zero"],
        "static": "static_frequency",
    }.items():
        horizon[label] = {
            str(anchor): source_horizon[(method, anchor)].as_dict()
            for anchor in range(1, config.operating_point.horizon + 1)
        }
    for key in candidate_keys:
        horizon[key] = {
            str(anchor): candidate_horizon[(key, anchor)].as_dict()
            for anchor in range(1, config.operating_point.horizon + 1)
        }
    temporal = {
        "natural": {
            str(lag): {
                "centered_logit_cosine": _mean(value[0] for value in rows),
                "topk_overlap": _mean(value[1] for value in rows),
            }
            for lag, rows in sorted(natural_temporal.items())
        },
        "pseudo_position": {
            str(lag): {
                "normalized_centered_logit_rms": _mean(value[0] for value in rows),
                "topk_overlap": _mean(value[1] for value in rows),
            }
            for lag, rows in sorted(pseudo_position.items())
        },
    }
    components = {
        short: {
            "normalized_centered_logit_rms": _mean(value[0] for value in component_values[short]),
            "pseudo_topk_overlap": _mean(value[1] for value in component_values[short]),
            "subset_jaccard": _mean(component_jaccard[short]),
        }
        for short in ("current", "causal", "zero", "expected")
    }
    sorted_margins = sorted(row[0] for row in margin_events if row[1] == "primary")
    margin_cutoffs = [
        sorted_margins[int(len(sorted_margins) * quantile / 4)] for quantile in range(1, 4)
    ]
    margin_analysis: dict[str, Any] = {"cutoffs": margin_cutoffs, "variants": {}}
    for short in SOURCE_VARIANTS:
        rows = [row for row in margin_events if row[1] == short]
        buckets = []
        for index in range(4):
            lower = float("-inf") if index == 0 else margin_cutoffs[index - 1]
            upper = float("inf") if index == 3 else margin_cutoffs[index]
            selected = [row for row in rows if lower <= row[0] < upper]
            buckets.append(
                {
                    "quartile": index + 1,
                    "rows": len(selected),
                    "centered_logit_cosine": _mean(row[2] for row in selected),
                    "natural_topk_overlap": _mean(row[3] for row in selected),
                }
            )
        cast(dict[str, Any], margin_analysis["variants"])[short] = buckets
    layer_analysis = {
        "source": {
            method: {str(index): source_layer[(method, index)].as_dict() for index in range(48)}
            for method in SOURCE_METHODS
        },
        "candidates": {
            key: {str(index): candidate_layer[(key, index)].as_dict() for index in range(48)}
            for key in candidate_keys
        },
    }
    per_sample: dict[str, dict[str, dict[str, int | float]]] = {
        key: {
            sample.sample_id: candidate_sample[(key, sample.sample_id)].as_dict()
            for sample in config.source.samples
        }
        for key in candidate_keys
    }
    leave_one_out: dict[str, dict[str, dict[str, int | float]]] = {}
    for key in candidate_keys:
        leave_one_out[key] = {}
        for excluded in config.source.samples:
            aggregate = Aggregate()
            for included in config.source.samples:
                if included.sample_id != excluded.sample_id:
                    aggregate.merge(candidate_sample[(key, included.sample_id)])
            leave_one_out[key][excluded.sample_id] = aggregate.as_dict()
    sensitivity: dict[str, object] = {
        "horizon": horizon,
        "temporal_persistence": temporal,
        "component_interventions": components,
        "router_margin": margin_analysis,
        "layer": layer_analysis,
    }
    write_json_atomic(output / "resolved_config.json", config.model_dump(mode="json"))
    write_json_atomic(output / "resolved_execution_revision.json", _git_revision())
    write_json_atomic(output / "source_audit.json", source_audit)
    write_json_atomic(output / "sensitivity.json", sensitivity)
    write_json_atomic(output / "candidate_aggregates.json", candidate_metrics)
    write_json_atomic(output / "candidate_per_sample.json", per_sample)
    write_json_atomic(output / "candidate_leave_one_out.json", leave_one_out)
    write_json_atomic(output / "decision.json", decision)
    raw_payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in raw_candidate_rows
    ).encode()
    _write_bytes_atomic(output / "candidate_rows.jsonl.gz", gzip.compress(raw_payload, mtime=0))
    _write_markdown_atomic(
        output / "report.md",
        _report(config, sensitivity, candidate_metrics, decision, previous),
    )
    provenance = {
        "schema_version": 1,
        "analysis_id": config.analysis_id,
        "config_fingerprint": config.fingerprint(),
        "source_samples": [sample.sample_id for sample in config.source.samples],
        "source_kind": "existing_authoritative_open_loop_development_tensors",
        "model_loaded": False,
        "gpu_used": False,
        "network_downloads": False,
        "accuracy_or_answers_used": False,
        "future_true_tokens_used_by_candidate_methods": False,
        "default_vector_values_used_by_candidate_methods": False,
        "offline_calibration_statistics_used_by_candidate_methods": False,
        "learned_or_fitted_parameters": False,
        "online_history_reference": "teacher_forced_natural_history_for_open_loop_analysis",
        "transfer_kind": "simulated",
        "runtime_kind": "cpu_analysis_only_not_model_runtime",
        "analysis_elapsed_seconds_measured": time.perf_counter() - started,
    }
    write_json_atomic(output / "provenance.json", provenance)
    manifest = _artifact_manifest(config, output)
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": config.analysis_id,
        "config_fingerprint": config.fingerprint(),
        "decision": decision["decision"],
        "best_observed_candidate": decision["best_observed_candidate"],
        "qualifying_candidate": decision["qualifying_candidate"],
        "artifact_count": manifest["artifact_count"],
    }
    write_json_atomic(output / "pipeline_status.json", status)
    validate(config)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("run", "validate"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_analysis_config(args.config)
    result = run(config) if args.command == "run" else validate(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
