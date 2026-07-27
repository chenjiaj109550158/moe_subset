"""Dependency-free M4 factorial heatmap and horizon curve plots."""

from __future__ import annotations

from pathlib import Path

from pseudoroute.analysis.dapq_factorial import BootstrapRow


def write_position_dominance_heatmap(path: Path, rows: list[BootstrapRow]) -> None:
    selected = [
        row
        for row in rows
        if row.condition == "POSITION_DOMINANCE"
        and not row.context_swapped
        and row.metric == "router_logit_cosine"
    ]
    layers = sorted({row.layer_idx for row in selected})
    columns = sorted({(row.position_offset, row.horizon) for row in selected})
    cell_w, cell_h = 92, 42
    width, height = 150 + cell_w * len(columns), 100 + cell_h * len(layers)
    lookup = {(row.layer_idx, row.position_offset, row.horizon): row.mean for row in selected}
    scale = max((abs(value) for value in lookup.values()), default=1.0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="25" font-size="17">Router-logit position dominance</text>',
    ]
    for column, (offset, horizon) in enumerate(columns):
        x = 130 + column * cell_w
        parts.append(f'<text x="{x + 5}" y="55" font-size="11">d={offset},h={horizon}</text>')
    for row_index, layer in enumerate(layers):
        y = 70 + row_index * cell_h
        parts.append(f'<text x="20" y="{y + 25}" font-size="12">layer {layer}</text>')
        for column, (offset, horizon) in enumerate(columns):
            value = lookup[(layer, offset, horizon)]
            intensity = min(255, int(80 + 175 * abs(value) / scale))
            color = (
                f"rgb({255 - intensity // 2},{255 - intensity // 2},255)"
                if value >= 0
                else f"rgb(255,{255 - intensity // 2},{255 - intensity // 2})"
            )
            x = 130 + column * cell_w
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" '
                f'fill="{color}" stroke="#666"/>'
            )
            parts.append(f'<text x="{x + 8}" y="{y + 25}" font-size="11">{value:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_metric_horizon_plot(path: Path, rows: list[BootstrapRow]) -> None:
    selected = [
        row
        for row in rows
        if row.metric == "router_logit_cosine"
        and not row.context_swapped
        and row.position_offset == min(item.position_offset for item in rows)
        and row.condition in {"SC_SP", "DC_SP", "SC_DP", "DC_DP"}
    ]
    width, height = 760, 460
    left, right, top, bottom = 80, 30, 50, 70
    max_horizon = max(row.horizon for row in selected)
    colors = {"SC_SP": "#111827", "DC_SP": "#2563eb", "SC_DP": "#dc2626", "DC_DP": "#9333ea"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="380" y="28" text-anchor="middle" font-size="18">'
            "Router-logit cosine by horizon</text>"
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}" stroke="black"/>'
        ),
    ]
    for condition, color in colors.items():
        grouped: dict[int, list[float]] = {}
        for row in selected:
            if row.condition == condition:
                grouped.setdefault(row.horizon, []).append(row.mean)
        points = []
        for horizon, values in sorted(grouped.items()):
            x = left + horizon / max_horizon * (width - left - right)
            y = height - bottom - (sum(values) / len(values) + 1) / 2 * (height - top - bottom)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
