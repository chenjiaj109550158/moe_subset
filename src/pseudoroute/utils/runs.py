"""Reproducible run-directory creation."""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path

from pseudoroute.config import AppConfig


def run_id(config: AppConfig) -> str:
    return config.fingerprint()[:16]


def create_run_directory(root: Path, config: AppConfig, *, actual_device: str) -> Path:
    destination = root / run_id(config)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination / "resolved_config.json"
    expected = config.resolved_json()
    if resolved.exists() and resolved.read_text(encoding="utf-8") != expected:
        raise FileExistsError(f"incompatible run directory: {destination}")
    resolved.write_text(expected, encoding="utf-8")
    environment = {
        "information_regime": config.experiment.information_regime.value,
        "actual_device": actual_device,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (destination / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
