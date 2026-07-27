import csv
import json
from pathlib import Path

from pseudoroute.cli import main


def test_tiny_oracle_sweep_writes_tables_and_plots(tmp_path: Path) -> None:
    output = tmp_path / "oracle"
    assert (
        main(
            [
                "oracle-sweep",
                "--config",
                "configs/experiment/oracle_tiny.yaml",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["information_regime"] == "oracle"
    assert metrics["evaluation_mode"] == "open_loop"
    assert metrics["num_windows"] == 66
    assert metrics["num_rows"] == 792
    with (output / "oracle_windows.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 792
    assert {row["information_regime"] for row in rows} == {"oracle"}
    assert {row["selector"] for row in rows} == {
        "binary_count",
        "selected_routing_mass",
        "full_router_mass",
        "cost_aware_selected_mass",
    }
    assert (output / "oracle_layers.csv").is_file()
    assert (output / "sch_hit_rate.svg").read_text().startswith("<svg")
    assert (output / "expert_union_size.svg").read_text().startswith("<svg")
    assert (output / "DONE").read_text() == "complete\n"
