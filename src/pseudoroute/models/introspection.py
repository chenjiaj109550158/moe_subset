"""Stable JSON export for typed model specifications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pseudoroute.types import ExpertKey, ModelSpec


def model_manifest(spec: ModelSpec, *, revision: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_id": spec.model_id,
        "revision": revision,
        "architecture": spec.architecture,
        "num_layers": spec.num_layers,
        "moe_layer_indices": list(spec.moe_layer_indices),
        "num_experts_by_layer": spec.num_experts_by_layer,
        "top_k_by_layer": spec.top_k_by_layer,
        "hidden_size": spec.hidden_size,
        "uses_rope": spec.uses_rope,
        "pre_norm": spec.pre_norm,
        "shared_experts_by_layer": spec.shared_experts_by_layer,
        "routing_semantics_by_layer": spec.routing_semantics_by_layer,
        "expert_bytes": {
            f"{key.layer_idx}:{key.expert_idx}": size for key, size in spec.expert_bytes.items()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def export_model_manifest(path: Path, spec: ModelSpec, *, revision: str) -> None:
    payload = model_manifest(spec, revision=revision)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_model_manifest(path: Path) -> ModelSpec:
    """Load the stable manifest representation into a typed model specification."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expert_bytes = {
        ExpertKey(*(int(part) for part in key.split(":"))): int(value)
        for key, value in payload["expert_bytes"].items()
    }
    return ModelSpec(
        model_id=str(payload["model_id"]),
        architecture=str(payload["architecture"]),
        num_layers=int(payload["num_layers"]),
        moe_layer_indices=tuple(int(value) for value in payload["moe_layer_indices"]),
        num_experts_by_layer={
            int(key): int(value) for key, value in payload["num_experts_by_layer"].items()
        },
        top_k_by_layer={int(key): int(value) for key, value in payload["top_k_by_layer"].items()},
        hidden_size=int(payload["hidden_size"]),
        uses_rope=bool(payload["uses_rope"]),
        pre_norm=bool(payload["pre_norm"]),
        expert_bytes=expert_bytes,
        shared_experts_by_layer={
            int(key): int(value)
            for key, value in payload.get("shared_experts_by_layer", {}).items()
        },
        routing_semantics_by_layer={
            int(key): str(value)
            for key, value in payload.get("routing_semantics_by_layer", {}).items()
        },
    )
