from pathlib import Path

from pseudoroute.cli import main


def test_primary_reproduction_dry_run_lists_required_result_families(
    tmp_path: Path, capsys
) -> None:
    assert (
        main(
            [
                "reproduce",
                "--suite",
                "primary",
                "--config-root",
                "configs/paper",
                "--output-dir",
                str(tmp_path / "results"),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    for name in ("oracle", "factorial", "probe", "simulator", "runtime"):
        assert name in output
