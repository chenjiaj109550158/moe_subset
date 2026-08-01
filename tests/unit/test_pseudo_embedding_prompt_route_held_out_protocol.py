import hashlib
import json
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_prompt_route_held_out_v1.yaml")
MANIFEST = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1_samples.json")


def test_prompt_route_held_out_protocol_is_frozen_and_disjoint() -> None:
    payload = CONFIG.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "1f6f5ab8372aa6f63796cfda49601745bf2736490ee21c8807fd9657b9a996ef"
    )
    row = yaml.safe_load(payload)
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["partitions"]["held_out_route"]["rows"]
    observed = row["rows"]
    assert observed == expected
    development = {0, 439, 879, 1318}
    mechanism = {786, 394}
    assert not ({item["row_index"] for item in observed} & (development | mechanism))
    assert row["selected_candidate"]["key"] == ("equal_sampled_pseudo_prompt_or_recent_history")
    assert row["bootstrap"] == {
        "unit": "sample",
        "samples": 10000,
        "seed": 20260801,
        "interval": "percentile_95",
    }
