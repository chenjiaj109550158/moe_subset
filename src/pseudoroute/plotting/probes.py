"""Dependency-free SVG plots for M6 probe quality and online cost."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class ProbeResult(Protocol):
    @property
    def probe_name(self) -> str: ...

    @property
    def routing_mass_coverage(self) -> float: ...

    @property
    def latency_p95_us(self) -> float: ...


def write_quality_latency_plot(path: Path, rows: Sequence[ProbeResult]) -> None:
    width, height, margin = 720, 420, 58
    max_latency = max(row.latency_p95_us for row in rows) * 1.05
    points = []
    labels = []
    for row in rows:
        x = margin + row.latency_p95_us / max_latency * (width - 2 * margin)
        y = height - margin - row.routing_mass_coverage * (height - 2 * margin)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2864dc"/>')
        labels.append(
            f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-size="11">{row.probe_name}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle">online p95 latency (µs)</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})"
text-anchor="middle">routing-mass coverage</text>
{"".join(points)}{"".join(labels)}</svg>\n'''
    path.write_text(svg, encoding="utf-8")


class RegretResult(Protocol):
    @property
    def probe_name(self) -> str: ...

    @property
    def equal_budget_experts(self) -> int: ...

    @property
    def subset_regret(self) -> float: ...


def write_subset_regret_plot(path: Path, rows: Sequence[RegretResult]) -> None:
    width, height, margin = 720, 420, 58
    budgets = sorted({row.equal_budget_experts for row in rows})
    maximum = max(row.subset_regret for row in rows) or 1.0
    names = sorted({row.probe_name for row in rows})
    palette = ("#2864dc", "#d84a3a", "#2a9d55", "#7d4bc6", "#d28b18", "#168a91")
    paths = []
    for name_index, name in enumerate(names):
        selected = sorted(
            (row for row in rows if row.probe_name == name),
            key=lambda row: row.equal_budget_experts,
        )
        points = []
        for row in selected:
            x = margin + (row.equal_budget_experts - budgets[0]) / max(
                1, budgets[-1] - budgets[0]
            ) * (width - 2 * margin)
            y = height - margin - row.subset_regret / maximum * (height - 2 * margin)
            points.append(f"{x:.2f},{y:.2f}")
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{palette[name_index % len(palette)]}" stroke-width="2"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle">expert budget B</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})"
text-anchor="middle">subset regret</text>
{"".join(paths)}</svg>\n'''
    path.write_text(svg, encoding="utf-8")
