import json
from pathlib import Path

import pytest
import torch
import yaml

from pseudoroute.cli import main


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": {
            "name": "m10-test",
            "seed": 1234,
            "information_regime": "online_pre_sample",
        },
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "prompt_tokens": [1, 2],
        "gpu_slots_per_layer": 2,
        "pinned_memory": True,
        "warmup_tokens": 0,
        "measured_tokens": 2,
        "repetitions": 1,
        "subset_experts_by_layer": {0: [0, 1], 1: [0, 1]},
        "subset_miss_policy": "lossless_fallback",
    }


def test_benchmark_offload_dry_run_is_hardware_independent(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config()))
    assert main(["benchmark-offload", "--config", str(path), "--dry-run"]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for M10")
def test_benchmark_offload_exports_real_cuda_measurements(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config()))
    output = tmp_path / "run"
    assert main(["benchmark-offload", "--config", str(path), "--output-dir", str(output)]) == 0
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["runtime"] == "real_cuda_offload"
    assert metrics["synchronous_equivalent"] is True
    assert metrics["asynchronous_equivalent"] is True
    assert metrics["median_h2d_bytes"] > 0
    assert metrics["resident_expert_bytes"] < metrics["pinned_cpu_bytes"]
    assert metrics["timing_sources"]["transfer_and_stall"] == "CUDA events"
    assert {
        "repetitions.csv",
        "transfers.csv",
        "generated.csv",
        "metrics.json",
        "resolved_config.json",
        "model_manifest.json",
        "hardware.json",
        "environment.json",
        "DONE",
    } <= {path.name for path in output.iterdir()}
