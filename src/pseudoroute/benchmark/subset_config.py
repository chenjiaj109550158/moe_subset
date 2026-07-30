"""Strict frozen configuration for benchmark-aligned subset-oracle evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskKey = Literal["humaneval", "mbpp_plus", "gsm8k", "aime24", "aime25", "strategyqa"]
ModelKey = Literal["qwen3_30b_a3b", "gpt_oss_20b"]

TASKS = ("humaneval", "mbpp_plus", "gsm8k", "aime24", "aime25", "strategyqa")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceAccuracyConfig(StrictModel):
    suite_id: Literal["speculating_experts_accuracy_v17"]
    config: str
    artifact_root: str
    config_fingerprint: str = Field(min_length=64, max_length=64)
    required_pipeline_state: Literal["complete"]
    required_pipeline_stage: Literal["oracle_v17"]
    reuse: tuple[str, ...]


class SubsetModelConfig(StrictModel):
    key: ModelKey
    architecture: Literal["qwen3_moe", "gpt_oss"]
    physical_gpu: Literal[0, 1]
    routed_layers: int = Field(gt=0)
    routed_experts_per_layer: int = Field(gt=0)
    shared_experts_per_layer: int = Field(ge=0)
    native_top_k: int = Field(gt=0)
    budgets: tuple[int, ...]
    full_router_mass: Literal["native_full_softmax", "unavailable_selected_logits_softmax_only"]


class TraceSample(StrictModel):
    row_index: int = Field(ge=0)
    sample_id: str = Field(min_length=1)


class SubsetTraceConfig(StrictModel):
    source_policy: Literal["vanilla"]
    information_regime: Literal["offline_teacher_forced_saved_v17_trajectory"]
    horizons: tuple[int, ...]
    boundary_rule: Literal["non_overlapping_start_zero_per_horizon"]
    max_decode_tokens_per_sample: int = Field(gt=0)
    replay_chunk_tokens: int = Field(gt=0)
    store_router_logits: Literal[True]
    store_native_dtype: Literal[True]
    sample_rows: dict[TaskKey, tuple[TraceSample, ...]]
    autoregressive_parity_smoke_rows: dict[TaskKey, str]
    autoregressive_parity_max_tokens: int = Field(gt=0)


class SelectionConfig(StrictModel):
    methods: tuple[
        Literal[
            "future_selected_routing_mass",
            "future_binary_count",
            "previous_route",
            "static_frequency",
        ],
        ...,
    ]
    qwen_additional_methods: tuple[Literal["future_full_router_mass"], ...]
    exact_cardinality: Literal[True]
    tie_break: Literal["ascending_layer_scoped_expert_id"]
    previous_route_history: Literal["previous_realized_window_same_horizon"]
    previous_route_first_window: Literal["static_frequency"]
    static_frequency_source: Literal["v17_disjoint_wikitext_default_vector_counts"]


class TransferModelConfig(StrictModel):
    timing_kind: Literal["simulated_not_measured_runtime"]
    bandwidth_gib_per_second: float = Field(gt=0)
    fixed_latency_microseconds_per_load: float = Field(ge=0)
    resident_capacity: Literal["exact_budget_per_layer"]
    window_prefetch: Literal["load_new_subset_members_at_boundary"]
    fallback: Literal["transient_load_each_missing_natural_expert_per_layer_token"]
    reference: Literal["natural_on_demand_transient_native_selected_experts"]
    shared_experts: Literal["excluded_from_routed_budget_and_transfer"]


class OpenLoopConfig(StrictModel):
    raw_format: Literal["csv_gzip"]
    aggregate_statistics: tuple[Literal["mean", "median", "p05", "p95", "worst"], ...]
    stratify_by: tuple[str, ...]
    context_buckets: tuple[Literal["0-31", "32-63", "64-127"], ...]
    router_margin_buckets: Literal["per_model_task_layer_tertiles"]
    worst_case_count_per_model_task_metric: int = Field(gt=0)


class OperatingPointConfig(StrictModel):
    hard_oracle_method: Literal["future_selected_routing_mass"]
    require_every_task_before_selection: Literal[True]
    maximum_points_per_model: Literal[1]
    exclude_all_expert_budget: Literal[True]
    maximum_resident_fraction: float = Field(gt=0, lt=1)
    minimum_mean_route_hit: float = Field(ge=0, le=1)
    minimum_mean_selected_routing_mass: float = Field(ge=0, le=1)
    minimum_p05_route_hit: float = Field(ge=0, le=1)
    minimum_worst_route_hit: float = Field(ge=0, le=1)
    minimum_transfer_reduction: float = Field(ge=0, le=1)
    minimum_selected_mass_improvement_over_best_baseline: float = Field(ge=0, le=1)
    maximum_lossless_fallback_frequency: float = Field(ge=0, le=1)
    deterministic_order: tuple[str, ...]


class ClosedLoopConfig(StrictModel):
    smoke_rows: dict[TaskKey, str]
    smoke_horizon: Literal[16]
    smoke_budget_rule: Literal["native_top_k"]
    smoke_point_is_not_an_operating_candidate: Literal[True]
    smoke_max_new_tokens: int = Field(gt=0)
    full_sample_scope: Literal["all_v17_rows"]
    policies: tuple[
        Literal[
            "natural_v17_reuse",
            "lossless_oracle_residency",
            "hard_oracle_commitment",
            "previous_route_commitment",
        ],
        ...,
    ]
    oracle_future: Literal["natural_rollout_from_current_policy_boundary_state"]
    cache_semantics: Literal["rewind_natural_lookahead_then_replay_policy_window"]
    hard_mask_semantics: Literal[
        "mask_outside_subset_before_native_topk_and_native_renormalization"
    ]
    decode_semantics: Literal["exact_v17_per_model_task_settings"]
    sample_shards: int = Field(gt=0)
    max_concurrent_workers_per_physical_gpu: Literal[1]
    atomic_unit: Literal["one_sample_policy_row"]
    aime_gpt_max_new_tokens: Literal[32768]
    metrics: tuple[str, ...]


class VanillaGateRow(StrictModel):
    successes: int = Field(ge=0)
    samples: int = Field(gt=0)
    allowed_drop_questions: int = Field(ge=1)


class PairedAccuracyGate(StrictModel):
    definition: str
    confidence_interval: Literal["paired_difference_normal_95"]
    require_each_task: Literal[True]
    vanilla: dict[ModelKey, dict[TaskKey, VanillaGateRow]]


class DecisionConfig(StrictModel):
    labels: tuple[Literal["GO", "NARROW", "STOP/PIVOT"], ...]
    per_model_task_point: Literal[True]
    no_macro_average_override: Literal[True]
    predictor_authorization: Literal["only_all_task_GO_or_explicit_task_scoped_NARROW"]
    perfect_oracle_failure_action: Literal["stop_predictor_training"]


class SubsetOracleSuiteConfig(StrictModel):
    schema_version: Literal[1]
    protocol_revision: Literal[1]
    suite_id: Literal["benchmark_subset_oracle_v1"]
    source_accuracy: SourceAccuracyConfig
    models: tuple[SubsetModelConfig, ...]
    trace: SubsetTraceConfig
    selection: SelectionConfig
    transfer_model: TransferModelConfig
    open_loop: OpenLoopConfig
    operating_point_selection: OperatingPointConfig
    closed_loop: ClosedLoopConfig
    paired_accuracy_gate: PairedAccuracyGate
    decision: DecisionConfig

    @model_validator(mode="after")
    def frozen_invariants(self) -> SubsetOracleSuiteConfig:
        if self.trace.horizons != (1, 2, 4, 8, 16):
            raise ValueError("subset oracle horizons are frozen")
        if tuple(self.trace.sample_rows) != TASKS:
            raise ValueError("trace rows must retain frozen task order")
        if tuple(self.trace.autoregressive_parity_smoke_rows) != TASKS:
            raise ValueError("trace smoke rows must cover every task in order")
        if tuple(self.closed_loop.smoke_rows) != TASKS:
            raise ValueError("closed-loop smoke rows must cover every task in order")
        if tuple(model.key for model in self.models) != ("qwen3_30b_a3b", "gpt_oss_20b"):
            raise ValueError("model order is frozen")
        expected = {
            "qwen3_30b_a3b": ("qwen3_moe", 1, 48, 128, 8, (8, 16, 32, 64, 128)),
            "gpt_oss_20b": ("gpt_oss", 0, 24, 32, 4, (4, 8, 16, 24, 32)),
        }
        for model in self.models:
            observed = (
                model.architecture,
                model.physical_gpu,
                model.routed_layers,
                model.routed_experts_per_layer,
                model.native_top_k,
                model.budgets,
            )
            if observed != expected[model.key] or model.shared_experts_per_layer != 0:
                raise ValueError(f"frozen model facts changed for {model.key}")
        for task, rows in self.trace.sample_rows.items():
            if len(rows) != 4 or len({row.sample_id for row in rows}) != 4:
                raise ValueError(f"{task}: exactly four unique trace samples are required")
            if self.trace.autoregressive_parity_smoke_rows[task] != rows[0].sample_id:
                raise ValueError(f"{task}: parity smoke must use the first frozen trace row")
            if self.closed_loop.smoke_rows[task] != rows[0].sample_id:
                raise ValueError(f"{task}: closed-loop smoke must use the first frozen trace row")
        if set(self.paired_accuracy_gate.vanilla) != {model.key for model in self.models}:
            raise ValueError("paired accuracy gate must cover both models")
        for model_key, tasks in self.paired_accuracy_gate.vanilla.items():
            if tuple(tasks) != TASKS:
                raise ValueError(f"paired accuracy rows incomplete for {model_key}")
            if any(row.successes > row.samples for row in tasks.values()):
                raise ValueError(f"invalid vanilla successes for {model_key}")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_subset_oracle_config(path: str | Path) -> SubsetOracleSuiteConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("subset-oracle config root must be a mapping")
    return SubsetOracleSuiteConfig.model_validate(raw)
