"""Dependency-free M5 router geometry and criticality plots."""

from __future__ import annotations

from pathlib import Path

from pseudoroute.analysis.expert_criticality import InterventionRow
from pseudoroute.analysis.router_geometry import SVDMetricRow, TokenGeometryRow


def write_singular_value_plot(path: Path, rows: list[SVDMetricRow]) -> None:
    selected = [row for row in rows if row.method == "exact"]
    width, height = 700, 420
    left, right, top, bottom = 70, 30, 40, 60
    maximum = max(row.singular_value for row in selected) or 1.0
    max_rank = max(row.rank for row in selected)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="350" y="25" text-anchor="middle">Router singular spectra</text>',
    ]
    colors = ("#2563eb", "#dc2626", "#16a34a")
    for layer in sorted({row.layer_idx for row in selected}):
        points = []
        for row in selected:
            if row.layer_idx == layer:
                x = left + row.rank / max_rank * (width - left - right)
                y = height - bottom - row.singular_value / maximum * (height - top - bottom)
                points.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[layer % len(colors)]}" stroke-width="2"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_margin_histogram(path: Path, rows: list[TokenGeometryRow]) -> None:
    values = [row.topk_margin for row in rows]
    bins = 12
    maximum = max(values) or 1.0
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int(value / maximum * bins))] += 1
    width, height = 700, 420
    bar_width = 560 / bins
    max_count = max(counts) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="350" y="25" text-anchor="middle">Top-k margin distribution</text>',
    ]
    for index, count in enumerate(counts):
        bar_height = 320 * count / max_count
        parts.append(
            f'<rect x="{70 + index * bar_width:.2f}" y="{370 - bar_height:.2f}" '
            f'width="{bar_width - 2:.2f}" height="{bar_height:.2f}" fill="#2563eb"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_criticality_plot(path: Path, rows: list[InterventionRow]) -> None:
    width, height = 700, 420
    max_error = max((row.immediate_output_relative_l2 for row in rows), default=1.0) or 1.0
    max_kl = max((row.next_token_kl for row in rows), default=1.0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="350" y="25" text-anchor="middle">Expert intervention criticality</text>',
    ]
    for row in rows:
        x = 70 + row.immediate_output_relative_l2 / max_error * 590
        y = 370 - row.next_token_kl / max_kl * 320
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#dc2626"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
