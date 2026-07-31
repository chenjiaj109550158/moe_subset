"""Versioned execution-scope amendments for the frozen subset-oracle suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pseudoroute.benchmark.subset_config import TASKS, ModelKey, TaskKey

FullActualPolicy = Literal["hard_oracle_commitment", "previous_route_commitment"]
PhysicalGpu = Literal[0, 1]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseSuiteReference(StrictModel):
    suite_id: Literal["benchmark_subset_oracle_v1"]
    config_path: Literal["configs/benchmark/benchmark_subset_oracle_v1.yaml"]
    config_sha256: str = Field(min_length=64, max_length=64)
    config_fingerprint: str = Field(min_length=64, max_length=64)
    selected_operating_points_artifact: Literal[
        "artifacts/benchmark_subset_oracle_v1_r2_authoritative/selected_operating_points.json"
    ]
    selected_operating_points_sha256: str = Field(min_length=64, max_length=64)


class FullStageScope(StrictModel):
    models: tuple[Literal["gpt_oss_20b"], ...]
    tasks: tuple[TaskKey, ...]
    sample_scope: Literal["all_v17_rows"]
    actual_policies: tuple[FullActualPolicy, ...]
    expected_actual_rows: int = Field(gt=0)
    physical_gpus_by_model: dict[ModelKey, tuple[PhysicalGpu, ...]]
    max_concurrent_workers_per_physical_gpu: Literal[1]
    preserve_existing_unscheduled_rows: Literal[True]
    preserve_failed_markers: Literal[True]
    aggregate_requires_only_scheduled_policies: Literal[True]


class FocusedNextStage(StrictModel):
    model: Literal["qwen3_30b_a3b"]
    task: Literal["gsm8k"]
    horizon: Literal[8]
    budget: Literal[32]
    resident_fraction: float = Field(ge=0, le=1)
    policies: tuple[
        Literal[
            "hard_oracle_commitment",
            "previous_route_commitment",
            "pseudo_embedding_commitment",
        ],
        ...,
    ]
    pseudo_embedding_protocol_required_before_execution: Literal[True]
    learned_predictor_training: Literal[False]


class SubsetExecutionScope(StrictModel):
    schema_version: Literal[1]
    scope_revision: Literal[1, 2]
    scope_id: Literal[
        "benchmark_subset_oracle_v1_hard_only_full_v1",
        "benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2",
    ]
    authorized_at_utc: str
    authority: Literal["explicit_user_scope_reduction"]
    amendment_kind: Literal["execution_schedule_only"]
    base_suite: BaseSuiteReference
    full_stage: FullStageScope
    focused_next_stage: FocusedNextStage

    @model_validator(mode="after")
    def frozen_scope_invariants(self) -> SubsetExecutionScope:
        if self.full_stage.models != ("gpt_oss_20b",):
            raise ValueError("hard-only full scope must retain the preselected GPT model")
        if self.full_stage.actual_policies != ("hard_oracle_commitment",):
            raise ValueError("execution scope is hard-oracle-only for the full stage")
        expected_scope = {
            1: (
                "benchmark_subset_oracle_v1_hard_only_full_v1",
                TASKS,
                2608,
            ),
            2: (
                "benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2",
                ("gsm8k",),
                1319,
            ),
        }[self.scope_revision]
        expected_id, expected_tasks, expected_rows = expected_scope
        if self.scope_id != expected_id:
            raise ValueError(f"scope revision {self.scope_revision} has the wrong scope ID")
        if self.full_stage.tasks != expected_tasks:
            raise ValueError(
                f"scope revision {self.scope_revision} must retain tasks {expected_tasks}"
            )
        if self.full_stage.expected_actual_rows != expected_rows:
            raise ValueError(
                f"scope revision {self.scope_revision} must contain exactly "
                f"{expected_rows:,} GPT rows"
            )
        if self.full_stage.physical_gpus_by_model != {"gpt_oss_20b": (0, 1)}:
            raise ValueError("GPT hard shards must be distributed over physical GPUs 0 and 1")
        if self.focused_next_stage.resident_fraction != 0.25:
            raise ValueError("focused Qwen budget must retain 25% routed-expert residency")
        if self.focused_next_stage.policies != (
            "hard_oracle_commitment",
            "previous_route_commitment",
            "pseudo_embedding_commitment",
        ):
            raise ValueError("focused policy order changed")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_subset_execution_scope(path: str | Path) -> SubsetExecutionScope:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("subset execution-scope config root must be a mapping")
    return SubsetExecutionScope.model_validate(raw)
