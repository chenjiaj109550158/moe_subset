"""Completed reports must serialize, and repair records cannot excuse runtime edits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pseudoroute.restart import report, verify
from pseudoroute.restart.state import digest


def test_complete_paired_aggregate_is_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles/runtime_summary.json").write_text("{}")
    (tmp_path / "STATUS.json").write_text(
        json.dumps({"identity": "unit-fixture", "terminal_state": "COMPLETE"})
    )
    (tmp_path / "real_model_correctness.json").write_text('{"state": "PASS"}')
    (tmp_path / "protocol_frozen_2.yaml").write_text(
        yaml.safe_dump({"candidate": "H", "policies": ["E", "H", "S"], "quality_n": 2, "claims": 3})
    )
    fixtures = {}
    for policy in ("E", "H", "S"):
        for index in range(42):
            quality = index < 2
            row = {
                "stage": "confirmation_quality" if quality else "confirmation_performance",
                "policy": policy,
                "sample_id": str(index if quality else (index - 2) // 5),
                "repetition": 0 if quality else (index - 2) % 5,
                "correct": policy == "E" or (policy == "H" and index == 0),
                "truncated": False,
                "generated_tokens": 129,
                "request_wall_seconds": 1.0,
                "decode_wall_seconds": 1.0,
                "production_forwards": 128,
                "pseudo_positions": 0,
                "production_forwards_per_second": 0.9 if policy == "H" else 1.0,
                "offload": {"phases": {"production": {"h2d_bytes": 128}}},
                "token_ready_latencies": [{"seconds": 0.01}],
                "peak_cuda_allocated_bytes": 0,
            }
            name = f"{policy}_{index}.json"
            fixtures[name] = row
            (tmp_path / "runs" / name).write_text("unit fixture, not a measured row")
    monkeypatch.setattr(report, "load_row", lambda path, identity: fixtures[path.name])
    monkeypatch.setattr(report, "summarize", lambda rows: {})
    result = report.aggregate(tmp_path)
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded) == result
    assert result["primary_metrics"]["completed_paired_quality_n"] == 2
    assert result["primary_metrics"]["paired_losses"] == 1
    for key in (
        "dynamic_value_confirmed",
        "static_dominance_confirmed",
        "sufficient_negative_evidence",
    ):
        assert type(result["decision_inputs"][key]) is bool


def test_reporting_repair_cannot_authorize_model_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = "src/pseudoroute/restart/model.py"
    old = tmp_path / "archived.py"
    old.write_text("different original measurement source")
    (tmp_path / "DECISION.json").write_text('{"evidence": []}')
    (tmp_path / "execution_source_hashes.json").write_text(json.dumps({path: digest(old)}))
    (tmp_path / "reporting_repair.json").write_text(
        json.dumps(
            {
                "source_changes": {
                    path: {
                        "execution_sha256": digest(old),
                        "current_sha256": digest(Path(path)),
                        "archived_execution_source": "archived.py",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(
        verify.jsonschema,
        "Draft202012Validator",
        lambda *args, **kwargs: SimpleNamespace(validate=lambda value: None),
    )
    with pytest.raises(ValueError, match="execution source changed"):
        verify.verify(tmp_path)
