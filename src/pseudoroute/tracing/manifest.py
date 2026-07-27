"""Versioned trace manifest models."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ShardRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    path: str
    sha256: str
    num_tokens: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class TraceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    trace_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    information_regime: str = "offline_teacher_forced"
    model_id: str
    model_revision: str
    model_fingerprint: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    dataset_fingerprint: str
    trace_level: str
    storage_dtype: str
    num_samples: int = Field(default=0, ge=0)
    num_tokens: int = Field(default=0, ge=0)
    num_moe_layers: int = Field(gt=0)
    num_experts_by_layer: dict[int, int]
    top_k_by_layer: dict[int, int]
    shards: list[ShardRecord] = Field(default_factory=list)
    complete: bool = False
