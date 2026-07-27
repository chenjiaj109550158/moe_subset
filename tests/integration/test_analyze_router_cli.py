import csv
import json
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_analyze_router_cli_writes_geometry_and_intervention_artifacts(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "experiment": {"name": "test", "seed": 2, "information_regime": "offline_teacher_forced"},
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "documents": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "randomized_svd": {"rank": 2, "oversample": 1, "power_iterations": 1},
        "covariance_rank": 2,
        "boundary_epsilon": 0.1,
        "top_b": 3,
        "interventions": {
            "max_targets": 2,
            "kinds": ["zero_selected_contribution", "substitute_next_available"],
        },
    }
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    assert main(["analyze-router", "--config", str(config_path), "--output-dir", str(output)]) == 0
    required = {
        "svd.csv",
        "router_coordinates.csv",
        "expert_geometry.csv",
        "pairwise_boundaries.csv",
        "token_geometry.csv",
        "expert_interventions.csv",
        "covariance_sketch.safetensors",
        "singular_values.svg",
        "topk_margins.svg",
        "expert_criticality.svg",
        "metrics.json",
        "resolved_config.json",
        "DONE",
    }
    assert required <= {path.name for path in output.iterdir()}
    with (output / "expert_interventions.csv").open() as stream:
        interventions = list(csv.DictReader(stream))
    assert len(interventions) == 4
    assert all(row["information_regime"] == "offline_teacher_forced" for row in interventions)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["model_scope"] == "tiny_only"
    assert metrics["source_tokens"] == 8
    assert (output / "DONE").read_text() == "complete\n"
