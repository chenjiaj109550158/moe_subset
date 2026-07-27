"""SVG Pareto output for simulated M8 sensitivity sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import cast


def write_simulator_pareto(path: Path, rows: list[dict[str, object]]) -> None:
    width, height, margin = 760, 420, 60

    def number(row: dict[str, object], key: str) -> float:
        return float(cast(int | float, row[key]))

    maximum_bytes = max(1.0, max(number(row, "bytes_per_token") for row in rows))
    maximum_tpot = max(1.0, max(number(row, "mean_tpot_us") for row in rows))
    points = []
    for row in rows:
        x = margin + number(row, "bytes_per_token") / maximum_bytes * (width - 2 * margin)
        y = height - margin - number(row, "mean_tpot_us") / maximum_tpot * (height - 2 * margin)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2864dc"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle">SIMULATED M8 Pareto sweep</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle">H2D bytes/token</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})"
text-anchor="middle">mean simulated TPOT (µs)</text>
{"".join(points)}</svg>\n'''
    path.write_text(svg, encoding="utf-8")
