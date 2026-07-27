from pathlib import Path

import pytest
from pydantic import ValidationError

from pseudoroute.config import AppConfig, load_config, resolve_device

CONFIG = Path("configs/model/tiny_moe.yaml")


def test_config_round_trip() -> None:
    config = load_config(CONFIG)
    assert AppConfig.model_validate_json(config.resolved_json()) == config
    assert config.fingerprint() == load_config(CONFIG).fingerprint()


def test_unknown_keys_are_rejected() -> None:
    raw = load_config(CONFIG).model_dump()
    raw["model"]["future_tokens"] = [99]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_auto_device_resolves_to_an_available_backend() -> None:
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu") == "cpu"
