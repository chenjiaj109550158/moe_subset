import hashlib
from pathlib import Path

import yaml

CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_prompt_route_development_v1.yaml")


def test_prompt_route_development_protocol_is_frozen() -> None:
    payload = CONFIG.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "f1384397deba1e2cdb702dd4878af4092eaa64cd7625617900a22802d4813a25"
    )
    row = yaml.safe_load(payload)
    assert [(item["row_index"], item["sample_id"]) for item in row["rows"]] == [
        (0, "test-0"),
        (439, "test-439"),
        (879, "test-879"),
        (1318, "test-1318"),
    ]
    assert row["route_token_cap"] == 128
    assert row["calibration_free"]["fitted_coefficients"] == "forbidden"
    assert row["progress_gate"]["baseline"] == "previous_route_commitment"
    assert row["progress_gate"]["minimum_route_hit_improvement"] == 0.05
    assert row["progress_gate"]["minimum_selected_mass_improvement"] == 0.05
