"""Offline source/config/evidence/row/decision verification; never loads a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]

from pseudoroute.restart.report import aggregate
from pseudoroute.restart.state import digest, write_json


def verify(out: Path) -> dict[str, object]:
    d = json.loads((out / "DECISION.json").read_text())
    schema = json.loads(Path("configs/restart/decision.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(d)
    for evidence in d["evidence"]:
        path = (out / evidence["path"]).resolve()
        if not path.is_relative_to(out.resolve()) or digest(path) != evidence["sha256"]:
            raise ValueError(f"evidence path/hash mismatch: {path}")
    repair_file = out / "reporting_repair.json"
    repair = json.loads(repair_file.read_text()) if repair_file.exists() else {}
    allowed_reporting_files = {
        "src/pseudoroute/restart/report.py",
        "src/pseudoroute/restart/verify.py",
    }
    for path, expected in json.loads((out / "execution_source_hashes.json").read_text()).items():
        current = digest(Path(path))
        if current == expected:
            continue
        change = repair.get("source_changes", {}).get(path, {})
        archived = (out / change.get("archived_execution_source", "")).resolve()
        if (
            path not in allowed_reporting_files
            or change.get("execution_sha256") != expected
            or change.get("current_sha256") != current
            or not archived.is_relative_to(out.resolve())
            or not archived.is_file()
            or digest(archived) != expected
        ):
            raise ValueError(f"execution source changed: {path}")
    for path, expected in repair.get("additional_analysis_source_hashes", {}).items():
        if digest(Path(path)) != expected:
            raise ValueError(f"reporting regression source changed: {path}")
    result = aggregate(out)
    if repair:
        reference = (out / repair["normalized_original_aggregate"]).resolve()
        if (
            not reference.is_relative_to(out.resolve())
            or digest(reference) != repair["normalized_original_aggregate_sha256"]
            or result != json.loads(reference.read_text())
        ):
            raise ValueError("reporting repair changed frozen statistics or decisions")
    if (
        result["primary_metrics"] != d["primary_metrics"]
        or result["decision_result"]["overall"] != d["overall"]
    ):
        raise ValueError("aggregate/decision mismatch")
    for gate, status in result["decision_result"]["gates"].items():
        if d["gates"][gate]["status"] != status:
            raise ValueError("gate mismatch")
    for name in ("runtime", "window_subset", "pseudo_predictor"):
        if d[name]["status"] != result["decision_result"][name]:
            raise ValueError(f"disposition mismatch: {name}")
    for row_path in (out / "runs").rglob("*.json"):
        row = json.loads(row_path.read_text())
        if not row["measured"] or row["reference_tokens_used_for_generation"]:
            raise ValueError("invalid model measurement provenance")
        if row["mode"] == "controlled_fixed_length" and row["stage"] != "runtime_grid":
            if row["production_forwards"] != 128 or row["generated_tokens"] != 129:
                raise ValueError("invalid production throughput denominator")
        if row["dtype"] != "bfloat16" or row["slots_per_layer"] != 32:
            raise ValueError("unequal precision or expert memory budget")
    if f"**{d['overall']}**" not in (out / "FINAL_REPORT.md").read_text():
        raise ValueError("report contradicts machine decision")
    return {
        "state": "PASS",
        "evidence_files": len(d["evidence"]),
        "rows": result["actual_row_count"],
        "schema": "PASS",
        "row_checksums": "PASS",
        "source_hashes": "PASS",
        "decision_recomputed": "PASS",
        "reporting_only_repair_verified": bool(repair),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.run_dir)
    write_json(a.run_dir / "verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
