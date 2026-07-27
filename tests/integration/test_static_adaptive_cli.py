import csv
import json
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_simulate_offload_m9_logs_fixed_adaptive_events(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "experiment": {"name": "m9", "seed": 7, "information_regime": "oracle"},
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "calibration_documents": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "max_interventions": 3,
        "prompt_tokens": [1, 2],
        "max_new_tokens": 4,
        "horizons": [3],
        "total_experts_per_layer": 2,
        "static_fractions": [0.5],
        "ranking_methods": ["frequency"],
        "out_of_subset_threshold": 0.0,
        "hardware": {
            "name": "test",
            "bandwidth_bytes_per_us": 1024.0,
            "fixed_latency_us": 2.0,
            "compute_us_per_token": 3.0,
        },
    }
    path = tmp_path / "m9.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    assert main(["simulate-offload", "--config", str(path), "--output-dir", str(output)]) == 0
    with (output / "closed_loop_summary.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert {row["mode"] for row in summaries} == {"fixed", "adaptive"}
    with (output / "termination_events.csv").open() as stream:
        terminations = list(csv.DictReader(stream))
    assert terminations
    assert {row["reason"] for row in terminations} == {"aggregate_out_of_subset_mass"}
    with (output / "replan_events.csv").open() as stream:
        replans = list(csv.DictReader(stream))
    adaptive = [row for row in replans if row["mode"] == "adaptive"]
    assert [int(row["boundary"]) for row in adaptive] == [2, 3, 4, 5]
    for row in replans:
        static = {tuple(value) for value in json.loads(row["static_experts"])}
        evicted = {tuple(value) for value in json.loads(row["eviction_delta"])}
        assert static.isdisjoint(evicted)
    with (output / "simulation_summary.csv").open() as stream:
        simulated = list(csv.DictReader(stream))
    assert len(simulated) == 2
    assert {row["mode"] for row in simulated} == {"fixed", "adaptive"}
    assert {row["baseline"] for row in simulated} == {"custom_plan"}
    assert {row["simulated"] for row in simulated} == {"True"}
    for row in simulated:
        assert float(row["perfect_overlap_bound_us"]) <= float(row["total_time_us"])
        assert float(row["total_time_us"]) <= float(row["no_overlap_bound_us"])
    required = {
        "static_rankings.csv",
        "simulation_summary.csv",
        "simulation_timeline.csv",
        "fixed_vs_adaptive.svg",
        "fixed_vs_adaptive_simulated.svg",
        "metrics.json",
        "DONE",
    }
    assert required <= {item.name for item in output.iterdir()}
