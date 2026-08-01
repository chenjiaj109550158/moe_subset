import hashlib
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_content_smoke_v1.yaml")


def test_content_smoke_protocol_is_frozen_and_calibration_free() -> None:
    payload = CONFIG.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "66cc4643310e8d1b1ed0239c57a02c86664510db16489b03fde90bd7f1c5f7f0"
    )
    row = yaml.safe_load(payload)
    assert row["rows"] == [
        {"row_index": 786, "sample_id": "test-786"},
        {"row_index": 394, "sample_id": "test-394"},
    ]
    assert row["route_token_cap"] == 16
    assert row["calibration_free"]["default_vector_values"] == "forbidden"
    assert row["diagnostic_oracles_excluded_from_deployable_ranking"] is True
