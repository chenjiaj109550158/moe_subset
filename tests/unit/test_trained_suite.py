from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from pseudoroute.trained.config import TrainedSuiteConfig, load_trained_suite_config
from pseudoroute.trained.oracle import TrainedOracleRow, aggregate_oracle_rows
from pseudoroute.trained.runner import apply_decision_gate


def _suite() -> TrainedSuiteConfig:
    return load_trained_suite_config("configs/trained/oracle_gate_v1.yaml")


def _oracle_row(*, context: str, margin: str, start: int) -> TrainedOracleRow:
    return TrainedOracleRow(
        "olmoe",
        "wikitext",
        f"sample-{start}",
        "evaluation",
        0,
        start,
        4,
        16,
        2.0,
        "selected_routing_mass",
        context,
        0.1,
        margin,
        0.9,
        0.95,
        None,
        12,
        0.25,
        100,
        25.0,
        "[[1, 2]]",
        "[1, 2]",
    )


def test_oracle_aggregate_can_stratify_context_and_margin() -> None:
    suite = _suite()
    rows = [
        _oracle_row(context="early_0_15", margin="low", start=0),
        _oracle_row(context="late_32_plus", margin="high", start=32),
    ]
    combined = aggregate_oracle_rows(suite, rows)
    stratified = aggregate_oracle_rows(suite, rows, stratify=True)
    assert {row.context_position_bucket for row in combined} == {"all"}
    assert {row.router_margin_bucket for row in combined} == {"all"}
    assert {row.context_position_bucket for row in stratified} == {
        "early_0_15",
        "late_32_plus",
    }
    assert {row.router_margin_bucket for row in stratified} == {"low", "high"}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_decision_gate_requires_worst_tail_and_closed_loop_quality(tmp_path: Path) -> None:
    suite = _suite()
    suite = suite.model_copy(update={"models": (suite.models[0],)})
    root = tmp_path / "models" / "olmoe"
    root.mkdir(parents=True)
    oracle_rows: list[dict[str, object]] = []
    for domain in ("wikitext", "gsm8k"):
        for method, selected, byte_count in (
            ("selected_routing_mass", 0.98, 50.0),
            ("on_demand", 1.0, 100.0),
            ("previous_route", 0.80, 70.0),
            ("static_frequency", 0.82, 60.0),
        ):
            oracle_rows.append(
                {
                    "domain": domain,
                    "horizon": 4,
                    "budget": 16,
                    "method": method,
                    "route_hit_rate": 0.95,
                    "selected_mass_coverage": selected,
                    "estimated_h2d_bytes_per_token": byte_count,
                }
            )
    _write_csv(root / "oracle_windows.csv", oracle_rows)
    closed_rows: list[dict[str, object]] = []
    for domain in ("wikitext", "gsm8k"):
        closed_rows.extend(
            [
                {
                    "domain": domain,
                    "policy": "hard_oracle_commitment",
                    "exact_token_agreement": 0.5,
                    "relative_perplexity_increase": 0.2,
                    "fallback_frequency": 0.0,
                    "estimated_h2d_bytes_per_token": 50.0,
                },
                {
                    "domain": domain,
                    "policy": "lossless_oracle_residency",
                    "exact_token_agreement": 1.0,
                    "relative_perplexity_increase": 0.0,
                    "fallback_frequency": 0.2,
                    "estimated_h2d_bytes_per_token": 80.0,
                },
            ]
        )
    _write_csv(root / "closed_loop.csv", closed_rows)
    (root / "validation.json").write_text(
        json.dumps({"num_experts_by_layer": {"0": 64}}), encoding="utf-8"
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    decision = apply_decision_gate(suite, tmp_path)
    model = cast(list[dict[str, Any]], decision["model_decisions"])[0]
    assert model["decision"] == "STOP/PIVOT"
    assert "open-loop points pass" in model["reason"]
    assert model["gate_evidence"]["hard_commitment_quality_pass"] is False
    assert model["gate_evidence"]["lossless_fallback_pass"] is False
