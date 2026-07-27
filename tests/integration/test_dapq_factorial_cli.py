import csv
import json
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_dapq_factorial_cli_writes_offline_tables_and_plots(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "experiment": {"name": "test", "seed": 3, "information_regime": "offline_teacher_forced"},
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "documents": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        "boundary": 2,
        "horizons": [1, 2],
        "content_seed": 7,
        "position_offsets": [3],
        "context_swap": True,
        "capture": {
            "pre_rope_query": True,
            "post_rope_query": True,
            "post_attention_state": True,
            "router_input": True,
            "router_logits": True,
            "topk": True,
        },
        "bootstrap": {"samples": 10, "confidence": 0.9},
    }
    config_path = tmp_path / "factorial.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    assert main(["dapq-factorial", "--config", str(config_path), "--output-dir", str(output)]) == 0
    with (output / "factorial_examples.csv").open() as stream:
        examples = list(csv.DictReader(stream))
    assert len(examples) == 16
    assert {row["condition"] for row in examples} == {"SC_SP", "DC_SP", "SC_DP", "DC_DP"}
    assert all(row["information_regime"] == "offline_teacher_forced" for row in examples)
    assert (output / "factorial_metrics.csv").is_file()
    assert (output / "factorial_bootstrap.csv").is_file()
    assert (output / "position_dominance_heatmap.svg").read_text().startswith("<svg")
    assert (output / "router_logit_horizon.svg").read_text().startswith("<svg")
    metrics = json.loads((output / "metrics.json").read_text())
    assert all(metrics["scsp_teacher_forced_validation"].values())
    assert metrics["information_regime"] == "offline_teacher_forced"
    assert (output / "DONE").read_text() == "complete\n"
