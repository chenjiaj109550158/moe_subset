import json
import shutil
from pathlib import Path

import pytest

from pseudoroute.reporting.aggregate import aggregate_runs
from pseudoroute.reporting.manifest import finalize_run_manifest


def _run(
    root: Path,
    *,
    command: str = "oracle-sweep",
    kind: str = "oracle",
    complete: bool = True,
    negative: bool = False,
) -> Path:
    root.mkdir(parents=True)
    (root / "resolved_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": {"information_regime": "offline_teacher_forced"},
                "unique": root.name,
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "metrics.json").write_text(
        json.dumps({"gate_a_passed": not negative, "windows": 3}) + "\n"
    )
    if complete:
        (root / "DONE").write_text("complete\n")
    finalize_run_manifest(root, command=command, result_kind=kind)
    return root


def test_aggregate_excludes_and_reports_incomplete_and_retains_negative(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    complete = _run(runs / "complete", negative=True)
    incomplete = _run(runs / "incomplete", complete=False)
    output = tmp_path / "paper"
    summary = aggregate_runs(runs, output)
    assert len(summary.included_run_ids) == 1
    assert summary.incomplete_runs == (str(incomplete),)
    assert summary.negative_results == 1
    report = json.loads((output / "failure_cases.json").read_text())
    assert report["negative_results"][0]["finding"] == "gate_a_passed=false"
    table = (output / "paper_table_primary.md").read_text()
    assert complete.name in table
    assert all("[" in line and "](" in line for line in table.splitlines()[2:])


def test_aggregate_rejects_incompatible_schema_and_duplicate_run_ids(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = _run(runs / "first")
    manifest_path = first / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_runs(runs, tmp_path / "bad-schema")
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    shutil.copytree(first, runs / "duplicate")
    with pytest.raises(ValueError, match="duplicate run_id"):
        aggregate_runs(runs, tmp_path / "duplicate-output")


def test_paper_plot_regenerates_exactly_from_saved_tables(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _run(runs / "one")
    first = tmp_path / "first"
    second = tmp_path / "second"
    aggregate_runs(runs, first)
    aggregate_runs(runs, second)
    assert (first / "paper_figure_primary.svg").read_bytes() == (
        second / "paper_figure_primary.svg"
    ).read_bytes()
