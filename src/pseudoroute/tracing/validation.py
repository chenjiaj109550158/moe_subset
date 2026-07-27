"""Trace integrity validation."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from pseudoroute.tracing.manifest import TraceManifest
from pseudoroute.tracing.store import sha256_file


class TraceValidationError(ValueError):
    pass


def validate_trace(root: Path, *, require_complete: bool = True) -> TraceManifest:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise TraceValidationError("missing manifest.json")
    try:
        manifest = TraceManifest.model_validate_json(manifest_path.read_text())
    except Exception as error:
        raise TraceValidationError(f"invalid manifest: {error}") from error
    if require_complete and not manifest.complete:
        raise TraceValidationError("trace is not marked complete")
    if manifest.num_samples != len(manifest.shards):
        raise TraceValidationError("sample count does not match shard count")
    expected_offset = 0
    seen: set[str] = set()
    token_total = 0
    for record in manifest.shards:
        if record.sample_id in seen:
            raise TraceValidationError(f"duplicate sample id: {record.sample_id}")
        seen.add(record.sample_id)
        if record.start_offset != expected_offset or record.end_offset <= record.start_offset:
            raise TraceValidationError(f"non-contiguous offsets for {record.sample_id}")
        path = root / record.path
        if not path.is_file():
            raise TraceValidationError(f"missing shard: {record.path}")
        if sha256_file(path) != record.sha256:
            raise TraceValidationError(f"checksum mismatch: {record.path}")
        tensors = load_file(str(path))
        required = {
            "token_ids",
            "position_ids",
            "is_prompt",
            "router_topk_ids",
            "router_topk_weights",
        }
        if not required.issubset(tensors):
            raise TraceValidationError(f"missing required arrays: {record.path}")
        if manifest.trace_level == "router_logits" and "router_logits" not in tensors:
            raise TraceValidationError(f"missing router logits: {record.path}")
        token_count = int(tensors["token_ids"].numel())
        if token_count != record.num_tokens:
            raise TraceValidationError(f"token count mismatch: {record.path}")
        positions = tensors["position_ids"]
        if not bool((positions == torch.arange(token_count)).all()):
            raise TraceValidationError(f"non-monotonic positions: {record.path}")
        ids = tensors["router_topk_ids"]
        if ids.ndim != 3 or ids.shape[1] != manifest.num_moe_layers:
            raise TraceValidationError(f"invalid top-k shape: {record.path}")
        for layer in range(manifest.num_moe_layers):
            layer_ids = ids[:, layer]
            if bool((layer_ids < 0).any()) or bool(
                (layer_ids >= manifest.num_experts_by_layer[layer]).any()
            ):
                raise TraceValidationError(f"invalid expert id in layer {layer}: {record.path}")
            if layer_ids.shape[-1] != manifest.top_k_by_layer[layer]:
                raise TraceValidationError(f"invalid top-k count in layer {layer}: {record.path}")
        expected_offset = record.end_offset
        token_total += token_count
    if token_total != manifest.num_tokens:
        raise TraceValidationError("manifest token total mismatch")
    return manifest
