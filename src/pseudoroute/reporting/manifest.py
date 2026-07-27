"""Complete, checksummed run envelopes shared by all M11 result kinds."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RUN_MANIFEST_SCHEMA_VERSION = 1


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    command: str
    result_kind: str
    information_regime: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_utc: str
    complete: bool
    negative_results: tuple[str, ...]
    artifacts: tuple[ArtifactRecord, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _negative_results(metrics: dict[str, Any]) -> tuple[str, ...]:
    results: list[str] = []
    for key, value in metrics.items():
        if key.endswith("_passed") and value is False:
            results.append(f"{key}=false")
    if metrics.get("asynchronous_equivalent") is False:
        results.append("asynchronous equivalence failed")
    simulation = metrics.get("simulation_results")
    if isinstance(simulation, list) and any(
        isinstance(row, dict) and float(row.get("route_coverage", 1.0)) < 1.0 for row in simulation
    ):
        results.append("at least one simulated commitment has route_coverage<1")
    gate_b = metrics.get("gate_b")
    if isinstance(gate_b, dict) and gate_b.get("passed_for_configured_model") is False:
        results.append("factorial Gate B did not pass for the configured model")
    return tuple(sorted(set(results)))


def finalize_run_manifest(root: Path, *, command: str, result_kind: str) -> RunManifest:
    config_path = root / "resolved_config.json"
    metrics_path = root / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"run lacks resolved_config.json or metrics.json: {root}")
    config_bytes = config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    run_id = hashlib.sha256(f"{command}:{result_kind}:{config_sha}".encode()).hexdigest()[:16]
    config = json.loads(config_bytes)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    regime = str(
        metrics.get(
            "information_regime",
            config.get("experiment", {}).get("information_regime", "unspecified"),
        )
    )
    files = tuple(
        ArtifactRecord(
            path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=_sha256(path)
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        command=command,
        result_kind=result_kind,
        information_regime=regime,
        config_sha256=config_sha,
        created_utc=datetime.now(UTC).isoformat(),
        complete=(root / "DONE").read_text(encoding="utf-8") == "complete\n"
        if (root / "DONE").is_file()
        else False,
        negative_results=_negative_results(metrics),
        artifacts=files,
    )
    (root / "run_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_and_validate_manifest(root: Path) -> RunManifest:
    path = root / "run_manifest.json"
    if not path.is_file():
        raise ValueError(f"missing run_manifest.json: {root}")
    manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"incompatible run manifest schema {manifest.schema_version}: {root}")
    for artifact in manifest.artifacts:
        artifact_path = root / artifact.path
        if not artifact_path.is_file():
            raise ValueError(f"missing manifested artifact: {artifact_path}")
        if (
            artifact_path.stat().st_size != artifact.bytes
            or _sha256(artifact_path) != artifact.sha256
        ):
            raise ValueError(f"artifact checksum mismatch: {artifact_path}")
    return manifest
