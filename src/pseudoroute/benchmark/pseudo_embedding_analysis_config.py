"""Frozen configuration for calibration-free pseudo-embedding analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSample(StrictModel):
    row_index: int = Field(ge=0, lt=1319)
    sample_id: str
    tensor_path: str
    tensor_sha256: str = Field(min_length=64, max_length=64)


class SourceConfig(StrictModel):
    suite_id: Literal["pseudo_embedding_qwen_gsm8k_v1"]
    config: str
    config_fingerprint: Literal["a81f36b5ec4a4222ca7a459f9f9c1536d88bef5ba143c151c9e70498157d8cbc"]
    artifact_root: str
    artifact_manifest_sha256: str = Field(min_length=64, max_length=64)
    partition: Literal["development"]
    samples: tuple[SourceSample, ...]


class InformationBoundary(StrictModel):
    deployable_allowed_inputs: tuple[
        Literal[
            "sampled_next_token",
            "current_policy_cache",
            "current_policy_realized_route_history",
        ],
        ...,
    ]
    deployable_forbidden_inputs: tuple[
        Literal[
            "future_true_tokens",
            "vanilla_future_trajectory",
            "benchmark_answer",
            "correctness",
            "offline_calibration_statistics",
        ],
        ...,
    ]
    diagnostic_oracles_must_be_separate: Literal[True]
    diagnostic_oracles_excluded_from_candidate_ranking: Literal[True]


class CalibrationFree(StrictModel):
    learned_parameters: Literal["forbidden"]
    fitted_mixture_weights: Literal["forbidden"]
    offline_expert_priors: Literal["forbidden"]
    offline_route_transition_tables: Literal["forbidden"]
    default_vector_values_in_candidate_methods: Literal["forbidden"]
    online_history_scope: Literal["current_policy_only"]
    first_boundary_fallback: Literal["zero_pseudo_all_anchors"]


class OperatingPoint(StrictModel):
    horizon: Literal[8]
    budget: Literal[32]
    routed_experts: Literal[128]
    native_top_k: Literal[8]


AnalysisKey = Literal[
    "horizon_route_coverage",
    "natural_temporal_persistence",
    "layer_sensitivity",
    "router_margin_sensitivity",
    "component_intervention_sensitivity",
    "calibration_free_candidate_comparison",
    "leave_one_sample_out_stability",
]

CandidateRule = Literal[
    "sum_zero_pseudo_probabilities_all_anchors",
    "previous_window_selected_mass_frequency",
    "p_anchor1_plus_h_minus_1_recent_prior",
    "p_anchor1_plus_half_future_pseudo_plus_half_history_per_unknown_anchor",
    "anchor1_topk_core_then_recent_history_fill",
]


class CandidateConstruction(StrictModel):
    key: str
    rule: CandidateRule


class CandidateProgressReference(StrictModel):
    baseline: Literal["previous_route_commitment"]
    minimum_route_hit_improvement: float = Field(ge=0, le=1)
    minimum_selected_mass_improvement: float = Field(ge=0, le=1)
    minimum_simulated_transfer_reduction: float = Field(ge=0, le=1)
    selection_uses_accuracy: Literal[False]
    result_role: Literal["hypothesis_generation_only_requires_new_frozen_held_out_scope"]


class AnalysisConfig(StrictModel):
    schema_version: Literal[1]
    analysis_id: Literal["pseudo_embedding_calibration_free_analysis_v1"]
    status: Literal["protocol_frozen_before_analysis"]
    artifact_root: str
    source: SourceConfig
    information_boundary: InformationBoundary
    calibration_free: CalibrationFree
    operating_point: OperatingPoint
    analyses: tuple[AnalysisKey, ...]
    candidate_constructions: tuple[CandidateConstruction, ...]
    metrics: tuple[str, ...]
    candidate_progress_reference: CandidateProgressReference

    @model_validator(mode="after")
    def frozen_scope(self) -> AnalysisConfig:
        observed_samples = tuple(
            (sample.row_index, sample.sample_id) for sample in self.source.samples
        )
        if observed_samples != (
            (0, "test-0"),
            (439, "test-439"),
            (879, "test-879"),
            (1318, "test-1318"),
        ):
            raise ValueError("analysis source rows changed")
        if tuple(candidate.key for candidate in self.candidate_constructions) != (
            "zero_pseudo_all_anchors",
            "recent_route_prior_w8",
            "one_known_plus_seven_history",
            "equal_evidence_future_blend",
            "native_topk_core_plus_history_reserve",
        ):
            raise ValueError("calibration-free candidate set changed")
        expected_allowed = (
            "sampled_next_token",
            "current_policy_cache",
            "current_policy_realized_route_history",
        )
        if self.information_boundary.deployable_allowed_inputs != expected_allowed:
            raise ValueError("deployable information boundary changed")
        thresholds = self.candidate_progress_reference
        if (
            thresholds.minimum_route_hit_improvement,
            thresholds.minimum_selected_mass_improvement,
            thresholds.minimum_simulated_transfer_reduction,
        ) != (0.05, 0.05, 0.30):
            raise ValueError("candidate progress references changed")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("analysis config root must be a mapping")
    return AnalysisConfig.model_validate(raw)
