"""Dependency-free M9 fixed/adaptive comparison plot."""

from __future__ import annotations

from pathlib import Path
from typing import cast


def write_static_adaptive_plot(path: Path, rows: list[dict[str, object]]) -> None:
    width, height, margin = 760, 420, 60

    def number(row: dict[str, object], key: str) -> float:
        return float(cast(int | float, row[key]))

    points = []
    for row in rows:
        x = margin + number(row, "transfer_bytes") / max(
            1.0, max(number(item, "transfer_bytes") for item in rows)
        ) * (width - 2 * margin)
        y = height - margin - number(row, "exact_token_rate") * (height - 2 * margin)
        color = "#d84a3a" if row["mode"] == "adaptive" else "#2864dc"
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle">planned transfer bytes</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})"
text-anchor="middle">exact token rate</text>
{"".join(points)}</svg>\n'''
    path.write_text(svg, encoding="utf-8")


def write_simulated_static_adaptive_plot(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot the event-simulator latency/traffic comparison."""
    width, height, margin = 760, 420, 60

    def number(row: dict[str, object], key: str) -> float:
        return float(cast(int | float, row[key]))

    max_bytes = max(1.0, max(number(row, "transfer_bytes") for row in rows))
    max_tpot = max(1.0, max(number(row, "mean_tpot_us") for row in rows))
    points = []
    for row in rows:
        x = margin + number(row, "transfer_bytes") / max_bytes * (width - 2 * margin)
        y = height - margin - number(row, "mean_tpot_us") / max_tpot * (height - 2 * margin)
        color = "#d84a3a" if row["mode"] == "adaptive" else "#2864dc"
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle">simulated transfer bytes</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})"
text-anchor="middle">simulated mean TPOT (us)</text>
<text x="{width / 2}" y="28" text-anchor="middle">SIMULATED</text>
{"".join(points)}</svg>\n'''
    path.write_text(svg, encoding="utf-8")
