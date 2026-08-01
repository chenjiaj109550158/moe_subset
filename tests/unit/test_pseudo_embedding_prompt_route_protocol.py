import hashlib
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_prompt_route_smoke_v1.yaml")


def test_prompt_route_protocol_is_frozen_and_calibration_free() -> None:
    payload = CONFIG.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "6c348925820b914b1b8a021f68e121373b085e2973c19dc50030a1477bc3cf03"
    )
    row = yaml.safe_load(payload)
    assert [(item["row_index"], item["sample_id"]) for item in row["rows"]] == [
        (786, "test-786"),
        (394, "test-394"),
    ]
    assert row["prompt_route_source"]["window"] == "last_8_prompt_tokens"
    assert row["calibration_free"]["offline_expert_priors"] == "forbidden"
    assert row["progress_reference"] == {
        "baseline": "sampled_repeat_independent_zero",
        "minimum_route_hit_improvement": 0.05,
        "minimum_selected_mass_improvement": 0.05,
        "minimum_simulated_transfer_reduction": 0.30,
    }
