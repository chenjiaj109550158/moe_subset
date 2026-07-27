"""Dependency-free M3 quality-versus-transfer SVG."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from pseudoroute.execution.closed_loop import ClosedLoopSummary


def write_quality_transfer_plot(path: Path, summaries: list[ClosedLoopSummary]) -> None:
    width, height = 760, 460
    left, right, top, bottom = 90, 30, 50, 70
    xs = [row.transfer_bytes_per_token for row in summaries]
    x_max = max(xs) if max(xs) > 0 else 1.0

    def sx(value: float) -> float:
        return left + value / x_max * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - value * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">'
        "Oracle quality versus estimated transfer</text>",
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle">'
        "Estimated transfer bytes/token</text>",
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" '
        'text-anchor="middle">Exact token rate</text>',
    ]
    for row in summaries:
        label = escape(f"{row.policy}:h={row.horizon}:rho={row.budget_ratio:g}")
        x, y = sx(row.transfer_bytes_per_token), sy(row.exact_token_rate)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb"/>')
        parts.append(f"<title>{label}</title>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
