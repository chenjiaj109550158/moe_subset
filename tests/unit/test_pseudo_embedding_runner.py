from __future__ import annotations

import json
from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_runner import (
    build_artifact_manifest,
    validate_artifact_manifest,
)

CONFIG = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml")


def test_focused_artifact_manifest_detects_checksum_change(tmp_path: Path) -> None:
    suite = load_pseudo_embedding_config(CONFIG)
    value = tmp_path / "value.json"
    value.write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    manifest = build_artifact_manifest(suite, tmp_path)
    assert manifest["artifact_count"] == 1
    validate_artifact_manifest(suite, tmp_path)
    value.write_text(json.dumps({"state": "damaged"}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_artifact_manifest(suite, tmp_path)
