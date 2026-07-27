import csv
import json
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_closed_loop_cli_writes_tables_metrics_and_plot(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "experiment": {"name": "test", "seed": 1234, "information_regime": "oracle"},
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "prompt_tokens": [1, 2, 3],
        "max_new_tokens": 3,
        "grid": {
            "horizons": [2],
            "budget_ratios": [1.0],
            "gamma": 0.9,
            "policies": ["natural", "lossless_fallback", "masked_substitution"],
        },
        "gate_a": {"minimum_exact_token_rate": 1.0, "minimum_transfer_reduction": 0.0},
    }
    config_path = tmp_path / "closed_loop.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    assert (
        main(["closed-loop-eval", "--config", str(config_path), "--output-dir", str(output)]) == 0
    )
    with (output / "closed_loop_summary.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 3
    assert {row["policy"] for row in summaries} == {
        "natural",
        "lossless_fallback",
        "masked_substitution",
    }
    assert (output / "closed_loop_routes.csv").is_file()
    assert (output / "closed_loop_windows.csv").is_file()
    assert (output / "quality_vs_transfer.svg").read_text().startswith("<svg")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["information_regime"] == "oracle"
    assert metrics["evaluation_mode"] == "closed_loop"
    assert metrics["model_scope"] == "tiny_only"
    assert (output / "DONE").read_text() == "complete\n"
