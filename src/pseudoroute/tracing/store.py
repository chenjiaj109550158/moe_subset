"""Atomic sharded-safetensors route store with sample-level resume."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from pseudoroute.models.base import TraceLevel
from pseudoroute.tracing.manifest import ShardRecord, TraceManifest
from pseudoroute.types import ModelSpec, RouterTrace

_SAFE_SAMPLE = re.compile(r"[^A-Za-z0-9_.-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class TraceStore:
    def __init__(self, root: Path, manifest: TraceManifest, *, resume: bool = False) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.shard_root = root / "shards"
        root.mkdir(parents=True, exist_ok=True)
        self.shard_root.mkdir(exist_ok=True)
        if self.manifest_path.exists():
            existing = TraceManifest.model_validate_json(self.manifest_path.read_text())
            if not resume:
                raise FileExistsError(f"trace store already exists: {root}")
            if existing.model_dump(
                exclude={"shards", "num_samples", "num_tokens", "complete", "created_at"}
            ) != manifest.model_dump(
                exclude={"shards", "num_samples", "num_tokens", "complete", "created_at"}
            ):
                raise ValueError("resume manifest is incompatible with requested trace")
            self.manifest = existing
        else:
            self.manifest = manifest
            self._commit_manifest()

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        spec: ModelSpec,
        model_revision: str,
        dataset_id: str,
        dataset_revision: str,
        dataset_split: str,
        trace_level: TraceLevel,
        resume: bool = False,
    ) -> TraceStore:
        identity = json.dumps(
            {
                "model": spec.model_id,
                "model_revision": model_revision,
                "dataset": dataset_id,
                "dataset_revision": dataset_revision,
                "dataset_split": dataset_split,
                "trace_level": trace_level.value,
            },
            sort_keys=True,
        )
        manifest = TraceManifest(
            trace_id=fingerprint_text(identity)[:16],
            model_id=spec.model_id,
            model_revision=model_revision,
            model_fingerprint=fingerprint_text(repr(spec)),
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            dataset_split=dataset_split,
            dataset_fingerprint=fingerprint_text(
                f"{dataset_id}@{dataset_revision}:{dataset_split}"
            ),
            trace_level=trace_level.value,
            storage_dtype="float32",
            num_moe_layers=len(spec.moe_layer_indices),
            num_experts_by_layer=spec.num_experts_by_layer,
            top_k_by_layer=spec.top_k_by_layer,
        )
        return cls(root, manifest, resume=resume)

    @property
    def completed_sample_ids(self) -> frozenset[str]:
        return frozenset(shard.sample_id for shard in self.manifest.shards)

    def append_sample(
        self,
        sample_id: str,
        token_ids: Tensor,
        traces: tuple[RouterTrace, ...],
        *,
        is_prompt: bool = True,
    ) -> bool:
        if sample_id in self.completed_sample_ids:
            return False
        if self.manifest.complete:
            raise RuntimeError("cannot append to a completed trace")
        tokens = token_ids.detach().cpu().reshape(-1).to(torch.int64)
        token_count = int(tokens.numel())
        layers = self.manifest.num_moe_layers
        expected = token_count * layers
        if len(traces) != expected:
            raise ValueError(f"expected {expected} trace records, got {len(traces)}")
        by_key = {(trace.token_position, trace.layer_idx): trace for trace in traces}
        ordered = [
            by_key[(position, layer)] for position in range(token_count) for layer in range(layers)
        ]
        logits = torch.stack([trace.raw_logits.reshape(-1) for trace in ordered]).reshape(
            token_count, layers, -1
        )
        topk_ids = (
            torch.stack([trace.topk_ids.reshape(-1) for trace in ordered])
            .reshape(token_count, layers, -1)
            .to(torch.int64)
        )
        topk_weights = (
            torch.stack([trace.topk_weights.reshape(-1) for trace in ordered])
            .reshape(token_count, layers, -1)
            .float()
        )
        tensors: dict[str, Tensor] = {
            "token_ids": tokens,
            "position_ids": torch.arange(token_count, dtype=torch.int64),
            "is_prompt": torch.full((token_count,), is_prompt, dtype=torch.bool),
            "router_topk_ids": topk_ids,
            "router_topk_weights": topk_weights,
        }
        if self.manifest.trace_level == TraceLevel.ROUTER_LOGITS.value:
            tensors["router_logits"] = logits.float()
        safe_id = _SAFE_SAMPLE.sub("_", sample_id).strip("._") or "sample"
        index = len(self.manifest.shards)
        relative = Path("shards") / f"{index:06d}-{safe_id}.safetensors"
        destination = self.root / relative
        temporary = destination.with_suffix(".safetensors.tmp")
        save_file(tensors, str(temporary), metadata={"sample_id": sample_id})
        os.replace(temporary, destination)
        start = self.manifest.num_tokens
        record = ShardRecord(
            sample_id=sample_id,
            path=str(relative),
            sha256=sha256_file(destination),
            num_tokens=token_count,
            start_offset=start,
            end_offset=start + token_count,
        )
        self.manifest.shards.append(record)
        self.manifest.num_samples += 1
        self.manifest.num_tokens += token_count
        self._commit_manifest()
        return True

    def load_sample(self, sample_id: str) -> dict[str, Tensor]:
        records = [record for record in self.manifest.shards if record.sample_id == sample_id]
        if len(records) != 1:
            raise KeyError(sample_id)
        return load_file(str(self.root / records[0].path))

    def mark_complete(self) -> None:
        self.manifest.complete = True
        self._commit_manifest()

    def _commit_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(self.manifest.model_dump_json(indent=2) + "\n")
        os.replace(temporary, self.manifest_path)
