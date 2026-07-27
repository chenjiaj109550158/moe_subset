import json

from pseudoroute.cli import main


def test_inspect_model_dry_run(capsys) -> None:
    assert main(["inspect-model", "--config", "configs/model/tiny_moe.yaml", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["information_regime"] == "online_pre_sample"
    assert result["synthetic_expert_bytes_total"] == 32768
