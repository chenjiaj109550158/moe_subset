"""Oracle sweep orchestration and aggregation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

from pseudoroute.oracle.metrics import (
    expert_union_sizes,
    full_mass_coverage,
    segment_hit_rate,
    selected_mass_coverage,
    union_coverage,
)
from pseudoroute.oracle.selectors import (
    OracleSelector,
    oracle_utilities,
    select_top_b,
)
from pseudoroute.oracle.windows import OracleSample, iter_windows
from pseudoroute.types import ModelSpec


@dataclass(frozen=True)
class OracleSweepRow:
    information_regime: str
    sample_id: str
    start: int
    horizon: int
    selector: str
    budget_ratio: float
    budget_by_layer: str
    subset_by_layer: str
    hit_rate: float
    selected_mass_coverage: float
    full_mass_coverage: float
    union_coverage: float
    mean_union_size: float


@dataclass(frozen=True)
class OracleLayerRow:
    information_regime: str
    sample_id: str
    start: int
    horizon: int
    layer_idx: int
    union_size: int


ResultRow: TypeAlias = OracleSweepRow | OracleLayerRow


def run_oracle_sweep(
    samples: tuple[OracleSample, ...],
    spec: ModelSpec,
    *,
    horizons: tuple[int, ...],
    budget_ratios: tuple[float, ...],
    selectors: tuple[OracleSelector, ...],
    gamma: float,
    load_cost_lambda: float,
) -> tuple[list[OracleSweepRow], list[OracleLayerRow]]:
    rows: list[OracleSweepRow] = []
    layer_rows: list[OracleLayerRow] = []
    num_experts_values = set(spec.num_experts_by_layer.values())
    if len(num_experts_values) != 1:
        raise NotImplementedError("M2 sweep currently requires equal expert counts across layers")
    num_experts = next(iter(num_experts_values))
    load_costs = {key: float(size) for key, size in spec.expert_bytes.items()}
    for window in iter_windows(samples, horizons):
        unions = expert_union_sizes(window)
        for layer, size in unions.items():
            layer_rows.append(
                OracleLayerRow(
                    "oracle", window.sample_id, window.start, window.horizon, layer, size
                )
            )
        for ratio in budget_ratios:
            budgets = {
                layer: min(
                    spec.num_experts_by_layer[layer],
                    max(1, math.ceil(ratio * spec.top_k_by_layer[layer])),
                )
                for layer in spec.moe_layer_indices
            }
            for selector in selectors:
                utilities = oracle_utilities(
                    window,
                    selector,
                    num_experts=num_experts,
                    gamma=gamma,
                    load_costs=load_costs,
                    load_cost_lambda=load_cost_lambda,
                )
                subsets = select_top_b(utilities, budgets)
                rows.append(
                    OracleSweepRow(
                        information_regime="oracle",
                        sample_id=window.sample_id,
                        start=window.start,
                        horizon=window.horizon,
                        selector=selector.value,
                        budget_ratio=ratio,
                        budget_by_layer=json.dumps(budgets, sort_keys=True),
                        subset_by_layer=json.dumps(subsets, sort_keys=True),
                        hit_rate=segment_hit_rate(window, subsets),
                        selected_mass_coverage=selected_mass_coverage(window, subsets),
                        full_mass_coverage=full_mass_coverage(window, subsets),
                        union_coverage=union_coverage(window, subsets),
                        mean_union_size=sum(unions.values()) / len(unions),
                    )
                )
    return rows, layer_rows


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
