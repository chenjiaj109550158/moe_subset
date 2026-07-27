"""Dependency-free SVG plots for M2 oracle summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from pseudoroute.oracle.sweep import OracleLayerRow, OracleSweepRow

_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")


def _svg_line_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 30, 50, 70
    xs = [point[0] for points in series.values() for point in points]
    ys = [point[1] for points in series.values() for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(1.0, max(ys))
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}" stroke="black"/>'
        ),
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle">{escape(x_label)}</text>',
        (
            f'<text x="18" y="{height / 2}" '
            f'transform="rotate(-90 18 {height / 2})" text-anchor="middle">'
            f"{escape(y_label)}</text>"
        ),
    ]
    for index, (label, points) in enumerate(sorted(series.items())):
        color = _COLORS[index % len(_COLORS)]
        ordered = sorted(points)
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in ordered)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(
            f'<text x="{width - right - 150}" y="{top + 18 * index}" '
            f'fill="{color}">{escape(label)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_oracle_plots(
    output_dir: Path,
    rows: list[OracleSweepRow],
    layer_rows: list[OracleLayerRow],
) -> None:
    hit_groups: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for row in rows:
        hit_groups[(row.selector, row.budget_ratio, row.horizon)].append(row.hit_rate)
    hit_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (selector, ratio, horizon), values in hit_groups.items():
        hit_series[f"{selector}:rho={ratio:g}"].append((horizon, sum(values) / len(values)))
    _svg_line_plot(
        output_dir / "sch_hit_rate.svg",
        title="SCH-style oracle hit rate",
        x_label="Horizon",
        y_label="Mean top-k hit rate",
        series=dict(hit_series),
    )
    union_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for layer_row in layer_rows:
        union_groups[(layer_row.layer_idx, layer_row.horizon)].append(layer_row.union_size)
    union_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (layer, horizon), union_values in union_groups.items():
        union_series[f"layer={layer}"].append(
            (float(horizon), sum(union_values) / len(union_values))
        )
    _svg_line_plot(
        output_dir / "expert_union_size.svg",
        title="Expert union size by horizon",
        x_label="Horizon",
        y_label="Mean union size",
        series=dict(union_series),
    )
