import json

from pseudoroute.config import load_config
from pseudoroute.utils.runs import create_run_directory, run_id


def test_run_directory_is_reproducible(tmp_path) -> None:
    config = load_config("configs/model/tiny_moe.yaml")
    first = create_run_directory(tmp_path, config, actual_device="cpu")
    second = create_run_directory(tmp_path, config, actual_device="cpu")
    assert first == second == tmp_path / run_id(config)
    environment = json.loads((first / "environment.json").read_text())
    assert environment["information_regime"] == "online_pre_sample"
    assert environment["actual_device"] == "cpu"
