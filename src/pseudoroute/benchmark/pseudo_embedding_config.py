"""Frozen configuration and sample-manifest validation for the focused Qwen pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceAccuracyConfig(StrictModel):
    suite_id: Literal["speculating_experts_accuracy_v17"]
    config: str
    artifact_root: str
    config_fingerprint: str = Field(min_length=64, max_length=64)
    required_pipeline_state: Literal["complete"]
    required_pipeline_stage: Literal["oracle_v17"]


class ModelConfig(StrictModel):
    key: Literal["qwen3_30b_a3b"]
    architecture: Literal["qwen3_moe"]
    model_id: Literal["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    revision: Literal["0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"]
    precision: Literal["bfloat16"]
    routed_layers: Literal[48]
    routed_experts_per_layer: Literal[128]
    native_top_k: Literal[8]
    shared_experts_per_layer: Literal[0]


class DatasetConfig(StrictModel):
    key: Literal["gsm8k"]
    dataset_id: Literal["openai/gsm8k"]
    config: Literal["main"]
    split: Literal["test"]
    revision: Literal["740312add88f781978c0658806c59bc2815b9866"]
    frozen_rows: Literal[1319]
    original_v17_max_new_tokens: Literal[512]
    prompt_and_parser: Literal["exact_v17_reuse"]


class DecodeConfig(StrictModel):
    batch_size: Literal[1]
    do_sample: Literal[False]
    seed: Literal[20260727]
    use_cache: Literal[True]


class OperatingPointConfig(StrictModel):
    horizon: Literal[8]
    budget_per_layer: Literal[32]
    resident_fraction: float = Field(gt=0, lt=1)
    routed_experts_per_layer: Literal[128]
    native_top_k: Literal[8]


class DefaultVectorConfig(StrictModel):
    root: str
    tensor_sha256: Literal["2a315f6f3dc65ff62b656d0eb92b6781267e5dd13cbcaf4c09a5b5806060fd30"]
    manifest_sha256: Literal["3c5e645666e12eb96e6d693ca8251d81a8a04200cb5a2abab043226e32077f87"]
    artifact_fingerprint: Literal[
        "baae202908e3a2d3f11f980eb99c52e6815b51a7386f47654dd1e25fcbdab1fd"
    ]
    definition: Literal["mean_unweighted_selected_expert_output"]
    unobserved_layer_expert_pairs: Literal[444]
    unobserved_value: Literal["zero_vector"]


class SampleManifestConfig(StrictModel):
    path: str
    sha256: str = Field(min_length=64, max_length=64)


class InformationBoundaryConfig(StrictModel):
    primary_regime: Literal["online_post_sample"]
    deployable_input: Literal["sampled_next_token_and_current_policy_production_cache_only"]
    forbidden_inputs: tuple[
        Literal[
            "future_true_tokens",
            "vanilla_future_trajectory",
            "benchmark_answer",
            "correctness",
        ],
        ...,
    ]
    production_cache: Literal["read_only_or_copy_on_write"]
    production_hidden_state_mutation: Literal["forbidden"]
    production_rng_mutation: Literal["forbidden"]
    shadow_cache_lifetime: Literal["discard_after_each_boundary"]


class VariantConfig(StrictModel):
    key: str
    content: Literal["sampled_next_token", "current_token"]
    attention: Literal["independent", "causal"]
    expert_contribution: Literal["default_vector_selected_topk_mixture", "zero"]
    role: Literal[
        "primary",
        "content_baseline",
        "causal_ablation",
        "expert_contribution_ablation",
    ]


class ExpectedEmbeddingConfig(StrictModel):
    status: Literal["deferred_unless_measured_cost_is_reasonable"]
    top_m: Literal[8]
    required_blocker_artifact_if_not_run: Literal["expected_top_m_blocker.json"]


class ProbeConfig(StrictModel):
    anchors: tuple[Literal[1, 2, 3, 4, 5, 6, 7, 8], ...]
    position_rule: Literal["exact_future_cache_positions_with_native_qwen_rope"]
    attention_semantics: Literal["native_qwen_attention"]
    router_semantics: Literal["exact_native_qwen_gate"]
    utility: Literal["sum_full_pre_topk_probabilities_over_anchors"]
    contribution_mixture: Literal["native_topk_ids_and_normalized_weights"]
    subset_selection: Literal["deterministic_exact_top32_per_layer"]
    tie_break: Literal["ascending_layer_scoped_expert_id"]


class RouteEvaluationConfig(StrictModel):
    development_source: Literal["existing_authoritative_four_trace_rows"]
    held_out_source: Literal["teacher_forced_replay_of_saved_v17_token_trajectories_only"]
    max_decode_tokens_per_sample: Literal[128]
    boundary_rule: Literal["non_overlapping_start_zero_h8"]
    bootstrap_samples: Literal[10000]
    bootstrap_seed: Literal[20260801]
    bootstrap_unit: Literal["sample"]
    stratify_by: tuple[Literal["layer", "context_bucket", "router_margin_bucket"], ...]
    context_buckets: tuple[Literal["0-31", "32-63", "64-127"], ...]
    router_margin_buckets: Literal["held_out_per_layer_tertiles"]
    worst_case_count: Literal[100]


class ProgressGateConfig(StrictModel):
    ranking_uses_task_accuracy: Literal[False]
    minimum_route_hit_improvement_over_previous: float = Field(ge=0, le=1)
    minimum_selected_mass_improvement_over_previous: float = Field(ge=0, le=1)
    require_not_worse_than_static_frequency: Literal[True]
    required_resident_fraction: float = Field(gt=0, lt=1)
    minimum_estimated_transfer_reduction: float = Field(ge=0, le=1)
    require_cache_rng_information_tests: Literal[True]
    require_measured_probe_costs: Literal[True]
    variant_ordering: tuple[str, ...]


class StrongCandidateGateConfig(StrictModel):
    minimum_oracle_minus_previous_gap_recovery: float = Field(ge=0, le=1)
    reference_minimum_mean_route_hit: float = Field(ge=0, le=1)
    reference_minimum_mean_selected_mass: float = Field(ge=0, le=1)


class ClosedLoopConfig(StrictModel):
    required_only_after_progress_gate: Literal[True]
    sample_count: Literal[16]
    first_wave_samples: Literal[8]
    policies_share_exact_rows_prompt_decode_parser_and_caps: Literal[True]
    actual_generation: Literal[True]
    identity_materialization: Literal["forbidden"]
    hard_mask_semantics: Literal["mask_outside_subset_before_native_topk_and_native_normalization"]
    oracle_future: Literal["natural_lookahead_from_current_policy_context"]
    previous_history: Literal["previous_realized_window_pre_mask_natural_routes"]
    previous_first_window: Literal["frozen_static_frequency"]
    pseudo_context: Literal["current_policy_context_only"]
    atomic_unit: Literal["one_sample_policy_row"]
    checksum_resume: Literal[True]
    maximum_workers_per_physical_gpu: Literal[1]
    eta_checkpoint_after_first_wave: Literal[True]
    ask_before_second_wave_if_projected_total_hours_exceeds: float = Field(gt=0)


class AccuracyGateConfig(StrictModel):
    frozen_before_generation: Literal[True]
    samples: Literal[16]
    vanilla_successes: Literal[16]
    vanilla_accuracy: float = Field(ge=0, le=1)
    allowed_drop_formula: Literal["max_1_ceil_2_sqrt_n_p_one_minus_p"]
    allowed_drop_questions: Literal[1]
    minimum_policy_successes: Literal[15]
    paired_ci: Literal["paired_sample_bootstrap_percentile_95"]
    bootstrap_samples: Literal[10000]
    bootstrap_seed: Literal[20260801]


class CostReportingConfig(StrictModel):
    require_measured_latency: Literal[True]
    require_peak_temporary_cuda_memory: Literal[True]
    require_router_calls: Literal[True]
    require_attention_queries: Literal[True]
    require_cpu_gpu_sync_count: Literal[True]
    transfer_and_stall: Literal["simulated_not_runtime"]


class DecisionConfig(StrictModel):
    labels: tuple[Literal["GO", "NARROW", "STOP/PIVOT"], ...]
    maximum_positive_label: Literal["NARROW"]
    progress_gate_failure: Literal["STOP/PIVOT"]
    no_full_dataset_claim: Literal[True]
    no_runtime_speedup_claim: Literal[True]
    learned_predictor_training: Literal["forbidden"]


class SampleRef(StrictModel):
    row_index: int = Field(ge=0, lt=1319)
    sample_id: str
    sha256_rank: str | None = Field(default=None, min_length=64, max_length=64)


class SamplePartition(StrictModel):
    purpose: str
    rows: tuple[SampleRef, ...]


class SampleSelection(StrictModel):
    accuracy_correctness_or_ground_truth_used: Literal[False]
    development_rule: Literal["explicit_reuse_of_four_preexisting_frozen_qwen_gsm8k_trace_rows"]
    eligible_population: Literal[1315]
    partition_assignment: Literal[
        "sort_once_then_allocate_mechanism_2_held_out_route_8_closed_loop_8_plus_8"
    ]
    rank_payload: Literal[
        "pseudo_embedding_qwen_gsm8k_v1|sample-ranking-v1|{row_index}|{sample_id}"
    ]
    ranking: Literal["ascending_sha256_then_ascending_row_index"]
    source_population: Literal[1319]


class SampleManifest(StrictModel):
    schema_version: Literal[1]
    suite_id: Literal["pseudo_embedding_qwen_gsm8k_v1"]
    selection: SampleSelection
    partitions: dict[str, SamplePartition]

    @model_validator(mode="after")
    def frozen_partitions(self) -> SampleManifest:
        expected_counts = {
            "development": 4,
            "mechanism_smoke": 2,
            "held_out_route": 8,
            "closed_loop_wave_1": 8,
            "closed_loop_wave_2": 8,
        }
        if set(self.partitions) != set(expected_counts):
            raise ValueError("sample manifest partition names changed")
        if any(len(self.partitions[key].rows) != count for key, count in expected_counts.items()):
            raise ValueError("sample manifest partition counts changed")
        all_rows = [row for partition in self.partitions.values() for row in partition.rows]
        if len({row.row_index for row in all_rows}) != len(all_rows):
            raise ValueError("sample partitions are not disjoint")
        development = self.partitions["development"].rows
        if tuple((row.row_index, row.sample_id) for row in development) != (
            (0, "test-0"),
            (439, "test-439"),
            (879, "test-879"),
            (1318, "test-1318"),
        ):
            raise ValueError("development rows changed")
        ranked = []
        for index in range(1319):
            if index in {0, 439, 879, 1318}:
                continue
            sample_id = f"test-{index}"
            payload = f"pseudo_embedding_qwen_gsm8k_v1|sample-ranking-v1|{index}|{sample_id}"
            ranked.append((hashlib.sha256(payload.encode()).hexdigest(), index, sample_id))
        ranked.sort()
        observed = [
            row
            for key in (
                "mechanism_smoke",
                "held_out_route",
                "closed_loop_wave_1",
                "closed_loop_wave_2",
            )
            for row in self.partitions[key].rows
        ]
        expected = ranked[: len(observed)]
        if any(
            (row.sha256_rank, row.row_index, row.sample_id) != value
            for row, value in zip(observed, expected, strict=True)
        ):
            raise ValueError("SHA-256 sample ranking or allocation changed")
        return self


class PseudoEmbeddingSuiteConfig(StrictModel):
    schema_version: Literal[1]
    protocol_revision: Literal[1]
    suite_id: Literal["pseudo_embedding_qwen_gsm8k_v1"]
    artifact_root: Literal["artifacts/pseudo_embedding_qwen_gsm8k_v1"]
    protocol_document: Literal["docs/pseudo_embedding_qwen_gsm8k_v1_protocol.md"]
    source_accuracy: SourceAccuracyConfig
    model: ModelConfig
    dataset: DatasetConfig
    decode: DecodeConfig
    operating_point: OperatingPointConfig
    default_vectors: DefaultVectorConfig
    sample_manifest: SampleManifestConfig
    information_boundary: InformationBoundaryConfig
    variants: tuple[VariantConfig, ...]
    optional_expected_embedding: ExpectedEmbeddingConfig
    probe: ProbeConfig
    policies: tuple[
        Literal[
            "hard_oracle_commitment",
            "previous_route_commitment",
            "pseudo_embedding_commitment",
        ],
        ...,
    ]
    route_evaluation: RouteEvaluationConfig
    progress_gate: ProgressGateConfig
    held_out_strong_candidate_gate: StrongCandidateGateConfig
    closed_loop: ClosedLoopConfig
    accuracy_gate: AccuracyGateConfig
    cost_reporting: CostReportingConfig
    decision: DecisionConfig

    @model_validator(mode="after")
    def frozen_invariants(self) -> PseudoEmbeddingSuiteConfig:
        observed_floats = (
            self.operating_point.resident_fraction,
            self.progress_gate.minimum_route_hit_improvement_over_previous,
            self.progress_gate.minimum_selected_mass_improvement_over_previous,
            self.progress_gate.required_resident_fraction,
            self.progress_gate.minimum_estimated_transfer_reduction,
            self.held_out_strong_candidate_gate.minimum_oracle_minus_previous_gap_recovery,
            self.held_out_strong_candidate_gate.reference_minimum_mean_route_hit,
            self.held_out_strong_candidate_gate.reference_minimum_mean_selected_mass,
            self.closed_loop.ask_before_second_wave_if_projected_total_hours_exceeds,
            self.accuracy_gate.vanilla_accuracy,
        )
        if observed_floats != (0.25, 0.05, 0.05, 0.25, 0.30, 0.25, 0.6967, 0.7162, 24.0, 1.0):
            raise ValueError("the frozen numerical gates changed")
        if self.probe.anchors != tuple(range(1, 9)):
            raise ValueError("the v1 pilot requires all eight future anchors")
        if tuple(variant.key for variant in self.variants) != (
            "sampled_next_independent_default_topk",
            "current_token_independent_default_topk",
            "sampled_next_causal_default_topk",
            "sampled_next_independent_zero",
        ):
            raise ValueError("the frozen variant set or ordering changed")
        if self.policies != (
            "hard_oracle_commitment",
            "previous_route_commitment",
            "pseudo_embedding_commitment",
        ):
            raise ValueError("the closed-loop policy set or ordering changed")
        if self.information_boundary.forbidden_inputs != (
            "future_true_tokens",
            "vanilla_future_trajectory",
            "benchmark_answer",
            "correctness",
        ):
            raise ValueError("the information boundary changed")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample_manifest(path: str | Path) -> SampleManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sample manifest root must be an object")
    return SampleManifest.model_validate(raw)


def load_pseudo_embedding_config(
    path: str | Path,
) -> PseudoEmbeddingSuiteConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pseudo-embedding config root must be a mapping")
    suite = PseudoEmbeddingSuiteConfig.model_validate(raw)
    repository_root = config_path.resolve().parents[2]
    manifest_path = repository_root / suite.sample_manifest.path
    if _sha256_file(manifest_path) != suite.sample_manifest.sha256:
        raise ValueError("sample manifest SHA-256 changed")
    load_sample_manifest(manifest_path)
    return suite
