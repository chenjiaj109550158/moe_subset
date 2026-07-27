import csv
import json
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_m8_simulate_offload_exports_all_baselines_and_timeline(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "experiment": {
            "name": "m8",
            "seed": 3,
            "information_regime": "offline_teacher_forced",
        },
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "prompt_tokens": [1, 2],
        "generated_tokens": 3,
        "baselines": [
            "on_demand",
            "lru",
            "lfu",
            "lossless_predictor",
            "one_step_commitment",
            "multi_step_commitment",
        ],
        "per_layer_budget": 2,
        "probe_us": 2.0,
        "max_concurrent_transfers": 2,
        "dense_compute_us_per_token": 10.0,
        "expert_compute_us": 2.0,
        "sweep": {
            "capacity_experts": [4],
            "bandwidth_bytes_per_s": [1.0e9],
            "fixed_latency_us": [5.0],
            "horizons": [2],
            "overlap_transfers": [True],
        },
    }
    path = tmp_path / "m8.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    assert main(["simulate-offload", "--config", str(path), "--output-dir", str(output)]) == 0
    with (output / "simulation_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["baseline"] for row in rows} == set(config["baselines"])
    assert {row["simulated"] for row in rows} == {"True"}
    with (output / "cache_events.csv").open() as stream:
        events = list(csv.DictReader(stream))
    assert {row["event"] for row in events} >= {
        "load_start",
        "load_end",
        "expert_compute_start",
    }
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["simulated"] is True
    assert all(metrics["invariants"].values())
    assert (output / "simulated_pareto.svg").exists()
    assert (output / "DONE").read_text() == "complete\n"
