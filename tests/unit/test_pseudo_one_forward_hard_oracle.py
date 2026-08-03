from __future__ import annotations

import json
from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import _write_checksummed
from pseudoroute.benchmark.pseudo_one_forward_hard_oracle import (
    BUDGET,
    HORIZON,
    IDS,
    _exact_matches,
    _load_checksummed,
)


def test_hard_oracle_scope_and_exact_match_accounting() -> None:
    assert (HORIZON, BUDGET, len(IDS)) == (8, 32, 8)
    assert _exact_matches([1, 2, 3], [1, 9]) == (1, 3)
    assert _exact_matches([], []) == (0, 0)


def test_hard_oracle_row_checksum_resume(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    row = {
        "state": "complete",
        "pilot_id": "pseudo_one_forward_hard_oracle_v1",
        "config_sha256": ("60e03148a24ad16859ca21631a3644d4db6930b0bdaddc482060646b77e902b6"),
        "sample_manifest_sha256": (
            "fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046"
        ),
        "row_index": 44,
        "sample_id": "test-44",
        "policy": "hard_oracle_commitment",
        "max_new_tokens": 512,
    }
    _write_checksummed(path, row)
    loaded = _load_checksummed(
        path,
        row_index=44,
        sample_id="test-44",
        max_new_tokens=512,
    )
    assert loaded is not None
    assert loaded["row_payload_sha256"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sample_id"] = "test-tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _load_checksummed(
            path,
            row_index=44,
            sample_id="test-44",
            max_new_tokens=512,
        )
