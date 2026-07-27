"""Strict heterogeneous-run aggregation and dependency-free paper outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pseudoroute.reporting.manifest import RunManifest, load_and_validate_manifest


@dataclass(frozen=True)
class AggregationSummary:
    included_run_ids: tuple[str, ...]
    incomplete_runs: tuple[str, ...]
    negative_results: int


def _metric_rows(root: Path, manifest: RunManifest) -> list[dict[str, object]]:
    metrics: dict[str, Any] = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    link = f"../runs/{root.name}/metrics.json"
    rows: list[dict[str, object]] = []
    scalar_keys = {
        "oracle": (
            "mean_full_mass_coverage",
            "mean_selected_mass_coverage",
            "mean_hit_rate",
            "num_windows",
        ),
        "factorial": (
            "scsp_cosine_absolute_tolerance",
            "num_examples",
            "num_metric_rows",
        ),
        "probe": ("equal_cost_ceiling_us", "target_fingerprint_verified"),
        "simulator": ("scenarios", "route_tokens"),
        "static": ("scenarios", "termination_events", "replan_events"),
        "runtime": (
            "median_h2d_bytes",
            "median_exposed_stall_ms",
            "median_tpot_ms",
            "peak_allocated_bytes",
        ),
    }
    for key in scalar_keys.get(manifest.result_kind, ()):
        if key in metrics and not isinstance(metrics[key], (dict, list)):
            rows.append(
                {
                    "result_kind": manifest.result_kind,
                    "metric": key,
                    "value": metrics[key],
                    "run_id": manifest.run_id,
                    "source": link,
                }
            )
    if not rows:
        rows.append(
            {
                "result_kind": manifest.result_kind,
                "metric": "completed",
                "value": True,
                "run_id": manifest.run_id,
                "source": link,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["| Result | Metric | Value (source run) |", "|---|---|---:|"]
    for row in rows:
        cell = f"[{row['value']} · {row['run_id']}]({row['source']})"
        lines.append(f"| {row['result_kind']} | {row['metric']} | {cell} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_tables(root: Path, rows: list[dict[str, object]]) -> None:
    definitions = {
        "table_a_model_hardware.md": {"runtime"},
        "table_b_oracle_feasibility.md": {"oracle"},
        "table_c_factorial_analysis.md": {"factorial"},
        "table_d_probe_comparison.md": {"probe"},
        "table_e_end_to_end_runtime.md": {"simulator", "runtime"},
        "table_f_static_residency.md": {"static"},
    }
    for filename, kinds in definitions.items():
        selected = [row for row in rows if row["result_kind"] in kinds]
        if selected:
            _write_markdown(root / filename, selected)


def _write_figure_index(path: Path, included: list[tuple[Path, RunManifest]]) -> None:
    lines = ["# Paper figure index", ""]
    for root, manifest in included:
        for artifact in manifest.artifacts:
            if artifact.path.endswith(".svg"):
                lines.append(
                    f"- `{manifest.run_id}` ({manifest.result_kind}): "
                    f"[{artifact.path}](../runs/{root.name}/{artifact.path})"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    numeric = [row for row in rows if isinstance(row["value"], (int, float))]
    width, height, margin = 800, 360, 50
    maximum = (
        max((abs(float(cast(int | float, row["value"]))) for row in numeric), default=1.0) or 1.0
    )
    bars = []
    for index, row in enumerate(numeric):
        x = margin + index * max(1, (width - 2 * margin) / max(1, len(numeric)))
        bar_width = max(4.0, (width - 2 * margin) / max(1, len(numeric)) - 4)
        bar_height = abs(float(cast(int | float, row["value"]))) / maximum * (height - 2 * margin)
        y = height - margin - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="#2864dc"><title>'
            f"{row['result_kind']}:{row['metric']} run={row['run_id']}</title></rect>"
        )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="25" text-anchor="middle">Primary results (saved tables only)</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}"
y2="{height - margin}" stroke="black"/>
{"".join(bars)}</svg>\n''',
        encoding="utf-8",
    )


def _worst_case(root: Path, manifest: RunManifest) -> dict[str, object] | None:
    definitions = {
        "oracle": ("oracle_windows.csv", "selected_mass_coverage", min),
        "factorial": ("factorial_metrics.csv", "value", min),
        "probe": ("probe_results.csv", "subset_regret", max),
        "simulator": ("simulation_summary.csv", "route_coverage", min),
        "static": ("closed_loop_summary.csv", "mean_out_of_subset_mass", max),
        "runtime": ("repetitions.csv", "tpot_ms", max),
    }
    definition = definitions.get(manifest.result_kind)
    if definition is None:
        return None
    filename, metric, selector = definition
    path = root / filename
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if manifest.result_kind == "factorial":
        rows = [row for row in rows if row.get("metric") == "router_logit_cosine"]
    if not rows:
        return None
    row = selector(rows, key=lambda item: float(item[metric]))
    return {
        "run_id": manifest.run_id,
        "result_kind": manifest.result_kind,
        "criterion": f"worst_{metric}",
        "value": row[metric],
        "source": f"../runs/{root.name}/{filename}",
        "row": json.dumps(row, sort_keys=True),
    }


def aggregate_runs(input_root: Path, output_root: Path) -> AggregationSummary:
    manifests = sorted(input_root.rglob("run_manifest.json"))
    if not manifests:
        raise ValueError(f"no run manifests found under {input_root}")
    seen: set[str] = set()
    included: list[tuple[Path, RunManifest]] = []
    incomplete: list[str] = []
    for path in manifests:
        root = path.parent
        manifest = load_and_validate_manifest(root)
        if manifest.run_id in seen:
            raise ValueError(f"duplicate run_id detected: {manifest.run_id}")
        seen.add(manifest.run_id)
        if not manifest.complete:
            incomplete.append(str(root))
            continue
        included.append((root, manifest))
    if not included:
        raise ValueError("no complete compatible runs to aggregate")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [row for root, manifest in included for row in _metric_rows(root, manifest)]
    _write_csv(output_root / "paper_table_primary.csv", rows)
    _write_markdown(output_root / "paper_table_primary.md", rows)
    _write_paper_tables(output_root, rows)
    _write_plot(output_root / "paper_figure_primary.svg", rows)
    _write_figure_index(output_root / "paper_figure_index.md", included)
    failures = [
        {"run_id": manifest.run_id, "result_kind": manifest.result_kind, "finding": finding}
        for _, manifest in included
        for finding in manifest.negative_results
    ]
    worst_cases = [
        case for root, manifest in included if (case := _worst_case(root, manifest)) is not None
    ]
    if worst_cases:
        _write_csv(output_root / "failure_cases.csv", worst_cases)
    report = {
        "schema_version": 1,
        "incomplete_runs": incomplete,
        "negative_results": failures,
        "note": "Negative and null results are retained; incomplete runs are excluded.",
    }
    (output_root / "failure_cases.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Failure and negative-result report", ""]
    lines.extend(
        f"- `{item['run_id']}` ({item['result_kind']}): {item['finding']}" for item in failures
    )
    lines.extend(f"- Incomplete and excluded: `{item}`" for item in incomplete)
    if not failures and not incomplete:
        lines.append("- No automatically detected negative or incomplete results.")
    lines.extend(("", "## Automatically selected worst cases", ""))
    lines.extend(
        f"- `{item['run_id']}` ({item['result_kind']}): "
        f"[{item['criterion']}={item['value']}]({item['source']})"
        for item in worst_cases
    )
    (output_root / "failure_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = AggregationSummary(
        tuple(manifest.run_id for _, manifest in included), tuple(incomplete), len(failures)
    )
    (output_root / "aggregation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "included_run_ids": summary.included_run_ids,
                "incomplete_runs": summary.incomplete_runs,
                "negative_results": summary.negative_results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
