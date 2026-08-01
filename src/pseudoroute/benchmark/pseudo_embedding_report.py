"""Route aggregates, bootstrap intervals, gates, and focused-pilot decisions."""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import numpy as np

from pseudoroute.benchmark.pseudo_embedding_config import (
    PseudoEmbeddingSuiteConfig,
    SampleManifest,
)
from pseudoroute.benchmark.pseudo_embedding_route import (
    _load_valid_sample,
    _sample_paths,
)
from pseudoroute.benchmark.subset_trace import write_json_atomic


@dataclass
class Aggregate:
    route_hits: int = 0
    route_slots: int = 0
    selected_mass_hit: float = 0.0
    selected_mass_total: float = 0.0
    full_mass_hit: float = 0.0
    full_mass_total: float = 0.0
    fallback_loads: int = 0
    prefetch_loads: int = 0
    subset_churn: int = 0
    transfer_bytes: int = 0
    reference_bytes: int = 0
    rows: int = 0

    def update(self, row: dict[str, Any]) -> None:
        self.route_hits += int(row["route_hits"])
        self.route_slots += int(row["route_slots"])
        self.selected_mass_hit += float(row["selected_mass_hit"])
        self.selected_mass_total += float(row["selected_mass_total"])
        self.full_mass_hit += float(row["full_mass_hit"])
        self.full_mass_total += float(row["full_mass_total"])
        self.fallback_loads += int(row["fallback_loads"])
        self.prefetch_loads += int(row["prefetch_loads"])
        self.subset_churn += int(row["subset_churn"])
        self.transfer_bytes += int(row["lossless_transfer_bytes"])
        self.reference_bytes += int(row["natural_reference_bytes"])
        self.rows += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "route_hits": self.route_hits,
            "route_slots": self.route_slots,
            "mean_route_hit": self.route_hits / self.route_slots,
            "mean_selected_mass": self.selected_mass_hit / self.selected_mass_total,
            "mean_full_router_mass": self.full_mass_hit / self.full_mass_total,
            "fallback_frequency": 1 - self.route_hits / self.route_slots,
            "fallback_loads": self.fallback_loads,
            "prefetch_loads": self.prefetch_loads,
            "subset_churn": self.subset_churn,
            "lossless_transfer_bytes": self.transfer_bytes,
            "natural_reference_bytes": self.reference_bytes,
            "estimated_transfer_reduction": (1 - self.transfer_bytes / self.reference_bytes),
        }


def _write_raw_csv_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty raw route table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = [key for key in rows[0] if key not in {"subset"}] + ["subset"]
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["subset"] = json.dumps(value["subset"], separators=(",", ":"))
            writer.writerow(value)
    os.replace(temporary, path)


def _sample_method_values(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, float]]]:
    states: dict[tuple[str, str], Aggregate] = defaultdict(Aggregate)
    for row in rows:
        states[(str(row["sample_id"]), str(row["method"]))].update(row)
    values: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (sample_id, method), state in states.items():
        aggregate = state.as_dict()
        values[method][sample_id] = (
            float(aggregate["mean_route_hit"]),
            float(aggregate["mean_selected_mass"]),
        )
    return dict(values)


def _paired_bootstrap(
    values: dict[str, dict[str, tuple[float, float]]],
    left: str,
    right: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    ids = sorted(set(values[left]) & set(values[right]))
    if not ids or set(ids) != set(values[left]) or set(ids) != set(values[right]):
        raise ValueError(f"paired bootstrap sample mismatch: {left}/{right}")
    differences = np.asarray(
        [
            (
                values[left][sample_id][0] - values[right][sample_id][0],
                values[left][sample_id][1] - values[right][sample_id][1],
            )
            for sample_id in ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(samples, len(ids)))
    draws = differences[indices].mean(axis=1)
    return {
        "left": left,
        "right": right,
        "samples": len(ids),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "mean_route_hit_difference": float(differences[:, 0].mean()),
        "route_hit_ci95": [
            float(np.quantile(draws[:, 0], 0.025)),
            float(np.quantile(draws[:, 0], 0.975)),
        ],
        "mean_selected_mass_difference": float(differences[:, 1].mean()),
        "selected_mass_ci95": [
            float(np.quantile(draws[:, 1], 0.025)),
            float(np.quantile(draws[:, 1], 0.975)),
        ],
    }


def _margin_thresholds(rows: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    values: dict[int, list[float]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (
            str(row["sample_id"]),
            int(row["boundary"]),
            int(row["layer"]),
        )
        if key in seen:
            continue
        seen.add(key)
        values[key[2]].append(float(row["router_margin"]))
    return {
        layer: (
            float(np.quantile(layer_values, 1 / 3)),
            float(np.quantile(layer_values, 2 / 3)),
        )
        for layer, layer_values in values.items()
    }


def _margin_bucket(value: float, thresholds: tuple[float, float]) -> str:
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "mid"
    return "high"


def _aggregate_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    states: dict[str, Aggregate] = defaultdict(Aggregate)
    for row in rows:
        states[str(row["method"])].update(row)
    return {method: state.as_dict() for method, state in sorted(states.items())}


def _strata_rows(
    rows: list[dict[str, Any]],
    thresholds: dict[int, tuple[float, float]],
) -> list[dict[str, object]]:
    states: dict[tuple[str, int, str, str], Aggregate] = defaultdict(Aggregate)
    for row in rows:
        layer = int(row["layer"])
        key = (
            str(row["method"]),
            layer,
            str(row["context_bucket"]),
            _margin_bucket(float(row["router_margin"]), thresholds[layer]),
        )
        states[key].update(row)
    return [
        {
            "method": key[0],
            "layer": key[1],
            "context_bucket": key[2],
            "router_margin_bucket": key[3],
            **state.as_dict(),
        }
        for key, state in sorted(states.items())
    ]


def _cost_report(samples: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for cost in sample["probe_costs"]:
            rows[str(cost["variant"])].append(cost)
    result: dict[str, dict[str, object]] = {}
    for variant, costs in sorted(rows.items()):
        latency = [float(row["latency_seconds_measured"]) for row in costs]
        result[variant] = {
            "boundaries": len(costs),
            "mean_probe_latency_seconds_measured": fmean(latency),
            "p95_probe_latency_seconds_measured": float(np.quantile(latency, 0.95)),
            "max_probe_latency_seconds_measured": max(latency),
            "max_temporary_cuda_bytes_measured": max(
                int(row["temporary_cuda_bytes_measured"]) for row in costs
            ),
            "persistent_default_vector_bytes": max(
                int(row["persistent_default_vector_bytes"]) for row in costs
            ),
            "attention_queries": sum(int(row["attention_queries"]) for row in costs),
            "attention_calls": sum(int(row["attention_calls"]) for row in costs),
            "router_calls": sum(int(row["router_calls"]) for row in costs),
            "cpu_gpu_synchronizations": sum(int(row["cpu_gpu_synchronizations"]) for row in costs),
        }
    return result


def _audit_report(samples: list[dict[str, Any]]) -> dict[str, object]:
    audits = [
        audit
        for sample in samples
        for audit in cast(list[dict[str, Any]], sample["cache_rng_audits"])
    ]
    required = (
        "production_cache_signature_unchanged",
        "production_rng_unchanged",
        "shadow_cache_discarded",
        "native_attention",
        "native_rope",
        "native_router",
        "layer_scoped_expert_ids",
    )
    return {
        "audit_rows": len(audits),
        "all_required_invariants_pass": all(
            all(bool(row[field]) for field in required) for row in audits
        )
        and all(not bool(row["forbidden_inputs_present"]) for row in audits),
        "required_fields": list(required),
        "forbidden_inputs_absent": all(not bool(row["forbidden_inputs_present"]) for row in audits),
        "unobserved_default_pairs": sorted(
            {int(row["unobserved_default_pairs"]) for row in audits}
        ),
    }


def _development_gate(
    suite: PseudoEmbeddingSuiteConfig,
    aggregates: dict[str, dict[str, Any]],
    costs: dict[str, dict[str, object]],
    audit: dict[str, object],
) -> dict[str, object]:
    previous = aggregates["previous_route_commitment"]
    static = aggregates["static_frequency"]
    oracle = aggregates["hard_oracle_commitment"]
    route_gap = float(oracle["mean_route_hit"]) - float(previous["mean_route_hit"])
    mass_gap = float(oracle["mean_selected_mass"]) - float(previous["mean_selected_mass"])
    candidates = []
    for variant in suite.variants:
        row = aggregates[variant.key]
        route_improvement = float(row["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_improvement = float(row["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        checks = {
            "route_hit_improvement_pass": (
                route_improvement >= suite.progress_gate.minimum_route_hit_improvement_over_previous
            ),
            "selected_mass_improvement_pass": (
                mass_improvement
                >= suite.progress_gate.minimum_selected_mass_improvement_over_previous
            ),
            "not_worse_than_static_route_hit_pass": float(row["mean_route_hit"])
            >= float(static["mean_route_hit"]),
            "not_worse_than_static_selected_mass_pass": float(row["mean_selected_mass"])
            >= float(static["mean_selected_mass"]),
            "resident_fraction_pass": suite.operating_point.resident_fraction
            == suite.progress_gate.required_resident_fraction,
            "transfer_reduction_pass": float(row["estimated_transfer_reduction"])
            >= suite.progress_gate.minimum_estimated_transfer_reduction,
            "cache_rng_information_pass": bool(audit["all_required_invariants_pass"]),
            "measured_probe_cost_pass": variant.key in costs
            and int(cast(Any, costs[variant.key]["boundaries"])) > 0,
        }
        candidates.append(
            {
                "variant": variant.key,
                "metrics": row,
                "route_hit_improvement_over_previous": route_improvement,
                "selected_mass_improvement_over_previous": mass_improvement,
                "development_oracle_minus_previous_route_hit_gap": route_gap,
                "development_oracle_minus_previous_selected_mass_gap": mass_gap,
                "development_route_hit_gap_recovery": (
                    route_improvement / route_gap if route_gap > 0 else 0.0
                ),
                "development_selected_mass_gap_recovery": (
                    mass_improvement / mass_gap if mass_gap > 0 else 0.0
                ),
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    eligible.sort(
        key=lambda candidate: (
            -float(cast(dict[str, Any], candidate["metrics"])["mean_selected_mass"]),
            -float(cast(dict[str, Any], candidate["metrics"])["mean_route_hit"]),
            float(
                cast(
                    Any,
                    costs[str(candidate["variant"])]["mean_probe_latency_seconds_measured"],
                )
            ),
            str(candidate["variant"]),
        )
    )
    selected = str(eligible[0]["variant"]) if eligible else None
    return {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "selection_uses_accuracy": False,
        "candidates": candidates,
        "selected_variant": selected,
        "progress_gate_pass": selected is not None,
        "decision": "NARROW" if selected is not None else "STOP/PIVOT",
    }


def _held_out_gate(
    suite: PseudoEmbeddingSuiteConfig,
    aggregates: dict[str, dict[str, Any]],
    selected_variant: str,
) -> dict[str, object]:
    oracle = aggregates["hard_oracle_commitment"]
    previous = aggregates["previous_route_commitment"]
    pseudo = aggregates[selected_variant]
    hit_gap = float(oracle["mean_route_hit"]) - float(previous["mean_route_hit"])
    mass_gap = float(oracle["mean_selected_mass"]) - float(previous["mean_selected_mass"])
    hit_recovery = (
        (float(pseudo["mean_route_hit"]) - float(previous["mean_route_hit"])) / hit_gap
        if hit_gap > 0
        else 0.0
    )
    mass_recovery = (
        (float(pseudo["mean_selected_mass"]) - float(previous["mean_selected_mass"])) / mass_gap
        if mass_gap > 0
        else 0.0
    )
    checks = {
        "route_hit_gap_recovery_pass": hit_recovery
        >= suite.held_out_strong_candidate_gate.minimum_oracle_minus_previous_gap_recovery,
        "selected_mass_gap_recovery_pass": mass_recovery
        >= suite.held_out_strong_candidate_gate.minimum_oracle_minus_previous_gap_recovery,
        "reference_route_hit_pass": float(pseudo["mean_route_hit"])
        >= suite.held_out_strong_candidate_gate.reference_minimum_mean_route_hit,
        "reference_selected_mass_pass": float(pseudo["mean_selected_mass"])
        >= suite.held_out_strong_candidate_gate.reference_minimum_mean_selected_mass,
    }
    return {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "selected_variant": selected_variant,
        "oracle_minus_previous_route_hit_gap": hit_gap,
        "oracle_minus_previous_selected_mass_gap": mass_gap,
        "route_hit_gap_recovery": hit_recovery,
        "selected_mass_gap_recovery": mass_recovery,
        "checks": checks,
        "held_out_route_gate_pass": all(checks.values()),
        "decision": "NARROW" if all(checks.values()) else "STOP/PIVOT",
        "actual_closed_loop_authorized": all(checks.values()),
    }


def aggregate_route_partition(
    suite: PseudoEmbeddingSuiteConfig,
    manifest: SampleManifest,
    output: Path,
    partition: str,
) -> dict[str, object]:
    references = manifest.partitions[partition].rows
    samples = []
    for reference in references:
        json_path, tensor_path = _sample_paths(
            output,
            partition,
            reference.row_index,
        )
        row = _load_valid_sample(suite, json_path, tensor_path)
        if row is None:
            raise ValueError(f"missing route sample for aggregate: {json_path}")
        if row["sample_id"] != reference.sample_id or row["partition"] != partition:
            raise ValueError(f"route aggregate provenance mismatch: {json_path}")
        samples.append(row)
    rows = [cast(dict[str, Any], metric) for sample in samples for metric in sample["metrics"]]
    root = output / "route" / partition
    _write_raw_csv_gzip(root / "raw_metrics.csv.gz", rows)
    aggregates = _aggregate_rows(rows)
    thresholds = _margin_thresholds(rows)
    strata = _strata_rows(rows, thresholds)
    costs = _cost_report(samples)
    audit = _audit_report(samples)
    values = _sample_method_values(rows)
    paired = []
    for method in aggregates:
        if method in {"previous_route_commitment", "static_frequency"}:
            continue
        paired.append(
            _paired_bootstrap(
                values,
                method,
                "previous_route_commitment",
                samples=suite.route_evaluation.bootstrap_samples,
                seed=suite.route_evaluation.bootstrap_seed,
            )
        )
    worst = {
        metric: sorted(
            (
                {
                    key: value
                    for key, value in row.items()
                    if key
                    in {
                        "sample_id",
                        "boundary",
                        "layer",
                        "method",
                        "route_hit_rate",
                        "selected_mass_coverage",
                        "router_margin",
                        "context_bucket",
                        "subset",
                    }
                }
                for row in rows
            ),
            key=lambda row: (float(row[metric]), str(row["sample_id"])),
        )[: suite.route_evaluation.worst_case_count]
        for metric in ("route_hit_rate", "selected_mass_coverage")
    }
    write_json_atomic(root / "aggregates.json", aggregates)
    write_json_atomic(
        root / "strata.json",
        {
            "margin_thresholds": {str(layer): list(value) for layer, value in thresholds.items()},
            "rows": strata,
        },
    )
    write_json_atomic(root / "worst_cases.json", worst)
    write_json_atomic(root / "paired_bootstrap.json", paired)
    write_json_atomic(root / "probe_cost.json", costs)
    write_json_atomic(root / "cache_rng_audit.json", audit)
    expected_key = "expected_top8_independent_default_topk"
    if expected_key in aggregates:
        previous = aggregates["previous_route_commitment"]
        oracle = aggregates["hard_oracle_commitment"]
        route_improvement = float(aggregates[expected_key]["mean_route_hit"]) - float(
            previous["mean_route_hit"]
        )
        mass_improvement = float(aggregates[expected_key]["mean_selected_mass"]) - float(
            previous["mean_selected_mass"]
        )
        route_gap = float(oracle["mean_route_hit"]) - float(previous["mean_route_hit"])
        mass_gap = float(oracle["mean_selected_mass"]) - float(previous["mean_selected_mass"])
        write_json_atomic(
            output / "expected_top_m_ablation.json",
            {
                "schema_version": 1,
                "state": "complete",
                "suite_id": suite.suite_id,
                "config_fingerprint": suite.fingerprint(),
                "variant": expected_key,
                "top_m": suite.optional_expected_embedding.top_m,
                "metrics": aggregates[expected_key],
                "cost": costs[expected_key],
                "selection_role": "auxiliary_not_in_frozen_mandatory_variant_ranking",
                "route_hit_improvement_over_previous": route_improvement,
                "selected_mass_improvement_over_previous": mass_improvement,
                "development_route_hit_gap_recovery": (
                    route_improvement / route_gap if route_gap > 0 else 0.0
                ),
                "development_selected_mass_gap_recovery": (
                    mass_improvement / mass_gap if mass_gap > 0 else 0.0
                ),
                "incompatible_proxy": False,
            },
        )
    gate: dict[str, object] | None = None
    if partition == "development":
        gate = _development_gate(suite, aggregates, costs, audit)
        write_json_atomic(output / "development_selection.json", gate)
    elif partition == "held_out_route":
        selection = cast(
            dict[str, Any],
            json.loads((output / "development_selection.json").read_text(encoding="utf-8")),
        )
        selected = selection.get("selected_variant")
        if not isinstance(selected, str):
            raise ValueError("held-out aggregate requires a selected development variant")
        gate = _held_out_gate(suite, aggregates, selected)
        write_json_atomic(output / "held_out_route_gate.json", gate)
    elif partition == "mechanism_smoke":
        primary = suite.variants[0].key
        changed = any(
            row["method"] == primary
            and row["subset"]
            != next(
                candidate["subset"]
                for candidate in rows
                if candidate["sample_id"] == row["sample_id"]
                and candidate["boundary"] == row["boundary"]
                and candidate["layer"] == row["layer"]
                and candidate["method"] == "static_frequency"
            )
            for row in rows
        )
        gate = {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "config_fingerprint": suite.fingerprint(),
            "rows": len(rows),
            "eight_token_windows": all(int(row["realized_tokens"]) == 8 for row in rows),
            "cache_rng_information_pass": audit["all_required_invariants_pass"],
            "primary_subset_changed_from_static": changed,
            "evaluator_nonempty": bool(aggregates),
            "mechanism_smoke_pass": bool(audit["all_required_invariants_pass"])
            and changed
            and bool(aggregates),
        }
        write_json_atomic(output / "mechanism_smoke_audit.json", gate)
    envelope = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "partition": partition,
        "samples": len(samples),
        "raw_metric_rows": len(rows),
        "methods": sorted(aggregates),
        "aggregate_rows": len(aggregates),
        "strata_rows": len(strata),
        "worst_case_rows": sum(len(value) for value in worst.values()),
        "audit": audit,
        "gate": gate,
    }
    write_json_atomic(root / "envelope.json", envelope)
    return envelope
