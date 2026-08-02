from __future__ import annotations

import json
from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_embedding_residual_window_report import (
    build_artifact_manifest,
    validate_artifact_manifest,
)


def test_residual_window_artifact_manifest_detects_checksum_change(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "nested/result.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({"value": 1}), encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path)
    assert manifest["artifact_count"] == 1
    assert validate_artifact_manifest(tmp_path)["state"] == "complete"

    artifact.write_text(json.dumps({"value": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_artifact_manifest(tmp_path)
