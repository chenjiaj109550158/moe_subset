from __future__ import annotations

import json
from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_embedding_closed_loop import (
    _load_checksummed_row,
    _write_checksummed_row,
    paired_accuracy_bootstrap,
)
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config

CONFIG = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml")


def test_closed_loop_sample_checksum_resume_rejects_tampering(tmp_path: Path) -> None:
    suite = load_pseudo_embedding_config(CONFIG)
    path = tmp_path / "sample.json"
    row: dict[str, object] = {
        "state": "complete",
        "config_fingerprint": suite.fingerprint(),
        "correct": True,
    }
    _write_checksummed_row(path, row)
    assert _load_checksummed_row(path, suite) is not None
    damaged = json.loads(path.read_text(encoding="utf-8"))
    damaged["correct"] = False
    path.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _load_checksummed_row(path, suite)


def test_paired_accuracy_bootstrap_is_paired_and_deterministic() -> None:
    correctness = {
        "vanilla_v17": {"a": True, "b": True, "c": False},
        "pseudo": {"a": True, "b": False, "c": True},
    }
    first = paired_accuracy_bootstrap(correctness, samples=100, seed=7)
    second = paired_accuracy_bootstrap(correctness, samples=100, seed=7)
    assert first == second
    assert first[0]["mean_accuracy_difference"] == pytest.approx(0.0)
