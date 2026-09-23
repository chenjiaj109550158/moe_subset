"""Strict YAML configuration for the M0 tiny model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pseudoroute.types import InformationRegime


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: InformationRegime

    @model_validator(mode="after")
    def natural_m0_regime(self) -> ExperimentConfig:
        if self.information_regime not in {
            InformationRegime.ONLINE_PRE_SAMPLE,
            InformationRegime.ONLINE_POST_SAMPLE,
        }:
            raise ValueError("M0 inspect-model supports online natural inference only")
        return self


class TinyModelConfig(StrictModel):
    device: str = Field(default="auto", min_length=1)
    vocab_size: int = Field(ge=2)
    hidden_size: int = Field(ge=4)
    num_layers: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    num_experts: int = Field(ge=1)
    top_k: int = Field(ge=1)
    expert_hidden_size: int = Field(ge=1)
    max_sequence_length: int = Field(ge=2)
    use_rope: bool = True
    shared_expert: bool = False
    synthetic_expert_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def dimensions_are_valid(self) -> TinyModelConfig:
        if self.device != "auto" and not (
            self.device == "cpu"
            or self.device == "cuda"
            or self.device.startswith("cuda:")
            or self.device == "mps"
        ):
            raise ValueError("device must be auto, cpu, cuda, cuda:<index>, or mps")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if (self.hidden_size // self.num_heads) % 2 and self.use_rope:
            raise ValueError("RoPE requires an even attention head dimension")
        return self


def resolve_device(configured: str) -> str:
    """Resolve the portable ``auto`` setting to the best available accelerator."""
    if configured != "auto":
        if configured.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available: {configured}")
            index = torch.device(configured).index
            if index is not None and index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device is not available: {configured}")
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DecodeConfig(StrictModel):
    max_new_tokens: int = Field(ge=0)
    eos_token_id: int | None = Field(default=None, ge=0)


class AppConfig(StrictModel):
    schema_version: int = Field(ge=1)
    experiment: ExperimentConfig
    model: TinyModelConfig
    decode: DecodeConfig

    def resolved_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    def fingerprint(self) -> str:
        return hashlib.sha256(self.resolved_json().encode()).hexdigest()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return AppConfig.model_validate(raw)


class HFModelConfig(StrictModel):
    adapter: str
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=40, max_length=64)
    device: str = "auto"
    trust_remote_code: bool = False
    cache_dir: str = Field(min_length=1)
    local_files_only: bool = True

    @model_validator(mode="after")
    def safe_remote_code(self) -> HFModelConfig:
        if self.trust_remote_code:
            raise ValueError("M1 reference adapter requires trust_remote_code=false")
        if self.adapter != "hf_mixtral":
            raise ValueError("M1 trace config supports adapter=hf_mixtral")
        return self


class DatasetConfig(StrictModel):
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=40, max_length=64)
    config: str = Field(min_length=1)
    split: str = Field(min_length=1)


class TraceConfig(StrictModel):
    level: Literal["routes_only", "router_logits"]
    storage_dtype: Literal["float32"] = "float32"
    max_samples: int = Field(default=1, ge=1)


class TraceRunConfig(StrictModel):
    schema_version: int = 1
    model: HFModelConfig
    data: DatasetConfig
    trace: TraceConfig
    information_regime: Literal["offline_teacher_forced"]


def load_trace_config(path: str | Path) -> TraceRunConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return TraceRunConfig.model_validate(raw)


class OracleExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["oracle"]


class OracleWindowConfig(StrictModel):
    horizons: tuple[int, ...]
    budget_ratios: tuple[float, ...]
    gamma: float = Field(default=1.0, gt=0, le=1)

    @model_validator(mode="after")
    def valid_grid(self) -> OracleWindowConfig:
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if not self.budget_ratios or any(value <= 0 for value in self.budget_ratios):
            raise ValueError("budget ratios must be positive")
        return self


class OracleMethodConfig(StrictModel):
    selectors: tuple[
        Literal[
            "binary_count",
            "selected_routing_mass",
            "full_router_mass",
            "cost_aware_selected_mass",
        ],
        ...,
    ]
    load_cost_lambda: float = Field(default=0.0, ge=0)


class OracleSweepConfig(StrictModel):
    schema_version: int = 1
    experiment: OracleExperimentConfig
    tiny_model_config: str | None = Field(default=None, min_length=1)
    trace_dir: str | None = Field(default=None, min_length=1)
    samples: tuple[tuple[int, ...], ...] = ()
    window: OracleWindowConfig
    oracle: OracleMethodConfig

    @model_validator(mode="after")
    def valid_samples(self) -> OracleSweepConfig:
        if (self.tiny_model_config is None) == (self.trace_dir is None):
            raise ValueError("set exactly one of tiny_model_config or trace_dir")
        if self.tiny_model_config is not None:
            if not self.samples or any(not sample for sample in self.samples):
                raise ValueError("tiny oracle sweep requires non-empty samples")
            if max(self.window.horizons) > max(len(sample) for sample in self.samples):
                raise ValueError("at least one sample must support the maximum horizon")
        elif self.samples:
            raise ValueError("trace-backed oracle sweep does not accept inline samples")
        return self


def load_oracle_config(path: str | Path) -> OracleSweepConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return OracleSweepConfig.model_validate(raw)


class ClosedLoopExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["oracle"]


class ClosedLoopGridConfig(StrictModel):
    horizons: tuple[int, ...]
    budget_ratios: tuple[float, ...]
    gamma: float = Field(default=1.0, gt=0, le=1)
    policies: tuple[
        Literal[
            "natural",
            "lossless_fallback",
            "masked_substitution",
            "masked_truncation_preserve",
            "masked_truncation_renormalize",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def valid_grid(self) -> ClosedLoopGridConfig:
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if not self.budget_ratios or any(value < 1 for value in self.budget_ratios):
            raise ValueError("M3 ratios must be >=1 so substitution can preserve top-k")
        if not self.policies:
            raise ValueError("at least one policy is required")
        return self


class GateAConfig(StrictModel):
    minimum_exact_token_rate: float = Field(ge=0, le=1)
    minimum_transfer_reduction: float = Field(ge=0, lt=1)


class ClosedLoopConfig(StrictModel):
    schema_version: int = 1
    experiment: ClosedLoopExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int = Field(ge=1)
    grid: ClosedLoopGridConfig
    gate_a: GateAConfig

    @model_validator(mode="after")
    def valid_prompt(self) -> ClosedLoopConfig:
        if not self.prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        return self


def load_closed_loop_config(path: str | Path) -> ClosedLoopConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ClosedLoopConfig.model_validate(raw)


class FactorialExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["offline_teacher_forced"]


class FactorialCaptureConfig(StrictModel):
    pre_rope_query: bool
    post_rope_query: bool
    post_attention_state: bool
    router_input: bool
    router_logits: bool
    topk: bool

    @model_validator(mode="after")
    def required_router_targets(self) -> FactorialCaptureConfig:
        if not self.router_logits or not self.topk:
            raise ValueError("factorial analysis requires router_logits and topk")
        return self


class FactorialBootstrapConfig(StrictModel):
    samples: int = Field(ge=1)
    confidence: float = Field(gt=0, lt=1)


class FactorialConfig(StrictModel):
    schema_version: int = 1
    experiment: FactorialExperimentConfig
    tiny_model_config: str | None = Field(default=None, min_length=1)
    hf_trace_config: str | None = Field(default=None, min_length=1)
    documents: tuple[tuple[int, ...], ...]
    boundary: int = Field(ge=1)
    horizons: tuple[int, ...]
    content_seed: int = Field(ge=0)
    position_offsets: tuple[int, ...]
    context_swap: bool
    capture: FactorialCaptureConfig
    bootstrap: FactorialBootstrapConfig

    @model_validator(mode="after")
    def valid_factorial(self) -> FactorialConfig:
        if (self.tiny_model_config is None) == (self.hf_trace_config is None):
            raise ValueError("set exactly one of tiny_model_config or hf_trace_config")
        if len(self.documents) < 2 and self.context_swap:
            raise ValueError("context swap requires at least two documents")
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if not self.position_offsets or any(value == 0 for value in self.position_offsets):
            raise ValueError("position offsets must be nonzero")
        needed = self.boundary + max(self.horizons)
        if any(len(document) < needed for document in self.documents):
            raise ValueError("factorial span would cross a document boundary")
        return self


def load_factorial_config(path: str | Path) -> FactorialConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return FactorialConfig.model_validate(raw)


class RouterAnalysisExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["offline_teacher_forced"]


class RandomizedSVDConfig(StrictModel):
    rank: int = Field(ge=1)
    oversample: int = Field(ge=0)
    power_iterations: int = Field(ge=0)


class InterventionConfig(StrictModel):
    max_targets: int = Field(ge=1)
    kinds: tuple[
        Literal[
            "zero_selected_contribution",
            "remove_and_renormalize",
            "substitute_next_available",
        ],
        ...,
    ]


class RouterAnalysisConfig(StrictModel):
    schema_version: int = 1
    experiment: RouterAnalysisExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    documents: tuple[tuple[int, ...], ...]
    randomized_svd: RandomizedSVDConfig
    covariance_rank: int = Field(ge=1)
    boundary_epsilon: float = Field(gt=0)
    top_b: int = Field(ge=1)
    interventions: InterventionConfig

    @model_validator(mode="after")
    def valid_router_analysis(self) -> RouterAnalysisConfig:
        if not self.documents or any(not document for document in self.documents):
            raise ValueError("router analysis requires non-empty documents")
        if not self.interventions.kinds:
            raise ValueError("at least one intervention kind is required")
        return self


def load_router_analysis_config(path: str | Path) -> RouterAnalysisConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return RouterAnalysisConfig.model_validate(raw)


class PredictorExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["offline_teacher_forced"]


class PredictorSplitConfig(StrictModel):
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def leaves_test_group(self) -> PredictorSplitConfig:
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("split fractions must leave a test split")
        return self


class PredictorModelConfig(StrictModel):
    ridge_lambda: float = Field(gt=0)
    mlp_hidden: int = Field(ge=1)
    mlp_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    rf_trees: int = Field(ge=1)
    rf_feature_candidates: int = Field(ge=1)
    markov_smoothing: float = Field(gt=0)


class PredictorTrainConfig(StrictModel):
    schema_version: int = 1
    experiment: PredictorExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    documents: tuple[tuple[int, ...], ...]
    horizons: tuple[int, ...]
    history_length: int = Field(ge=1)
    split: PredictorSplitConfig
    predictors: PredictorModelConfig

    @model_validator(mode="after")
    def valid_predictor_data(self) -> PredictorTrainConfig:
        if len(self.documents) < 3 or any(not document for document in self.documents):
            raise ValueError("predictor training requires at least three non-empty documents")
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must be positive")
        required = self.history_length + max(self.horizons)
        if any(len(document) < required for document in self.documents):
            raise ValueError("each document must contain at least one complete predictor window")
        return self


class ProbeEvaluationExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["online_pre_sample"]


class ProbeEvaluationConfig(StrictModel):
    schema_version: int = 1
    experiment: ProbeEvaluationExperimentConfig
    training_config: str = Field(min_length=1)
    predictor_dir: str = Field(min_length=1)
    top_b: int = Field(ge=1)
    latency_repetitions: int = Field(ge=1)
    rolling_lookback: int = Field(ge=1)
    rolling_decay: float = Field(gt=0, le=1)


def load_predictor_train_config(path: str | Path) -> PredictorTrainConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return PredictorTrainConfig.model_validate(raw)


def load_probe_evaluation_config(path: str | Path) -> ProbeEvaluationConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ProbeEvaluationConfig.model_validate(raw)


class DefaultVectorCalibrationConfig(StrictModel):
    schema_version: int = 1
    experiment: PredictorExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    documents: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def nonempty_documents(self) -> DefaultVectorCalibrationConfig:
        if not self.documents or any(not document for document in self.documents):
            raise ValueError("default calibration requires non-empty documents")
        return self


class ShadowProbeExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["online_pre_sample", "online_post_sample"]


class ShadowProbeEvaluationConfig(StrictModel):
    schema_version: int = 1
    experiment: ShadowProbeExperimentConfig
    training_config: str = Field(min_length=1)
    predictor_dataset_dir: str = Field(min_length=1)
    default_vectors_dir: str = Field(min_length=1)
    anchors: tuple[int, ...]
    top_b: int = Field(ge=1)
    latency_repetitions: int = Field(ge=1)
    wrong_position_offset: int
    fixed_token_id: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_anchors(self) -> ShadowProbeEvaluationConfig:
        if not self.anchors or any(anchor < 1 for anchor in self.anchors):
            raise ValueError("anchors must be positive")
        if self.wrong_position_offset == 0:
            raise ValueError("wrong-position offset must be nonzero")
        return self


def load_default_vector_config(path: str | Path) -> DefaultVectorCalibrationConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return DefaultVectorCalibrationConfig.model_validate(raw)


def load_shadow_probe_config(path: str | Path) -> ShadowProbeEvaluationConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ShadowProbeEvaluationConfig.model_validate(raw)


class M9ExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["oracle"]


class M9HardwareConfig(StrictModel):
    name: str = Field(min_length=1)
    bandwidth_bytes_per_us: float = Field(gt=0)
    fixed_latency_us: float = Field(ge=0)
    compute_us_per_token: float = Field(ge=0)
    expert_compute_us: float = Field(default=0.0, ge=0)
    probe_us: float = Field(default=0.0, ge=0)
    max_concurrent_transfers: int = Field(default=1, ge=1)
    overlap_transfers: bool = False


class StaticAdaptiveConfig(StrictModel):
    schema_version: int = 1
    experiment: M9ExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    calibration_documents: tuple[tuple[int, ...], ...]
    max_interventions: int = Field(ge=1)
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int = Field(ge=1)
    horizons: tuple[int, ...]
    total_experts_per_layer: int = Field(ge=1)
    static_fractions: tuple[float, ...]
    ranking_methods: tuple[
        Literal[
            "frequency",
            "covariance_variance",
            "quality_sensitivity",
            "downstream_influence",
            "miss_risk",
            "combined",
        ],
        ...,
    ]
    out_of_subset_threshold: float = Field(ge=0)
    hardware: M9HardwareConfig

    @model_validator(mode="after")
    def valid_m9_grid(self) -> StaticAdaptiveConfig:
        if not self.calibration_documents or any(not item for item in self.calibration_documents):
            raise ValueError("M9 calibration documents cannot be empty")
        if not self.prompt_tokens:
            raise ValueError("M9 prompt cannot be empty")
        if not self.horizons or any(value < 1 for value in self.horizons):
            raise ValueError("M9 horizons must be positive")
        if not self.static_fractions or any(not 0 <= value <= 1 for value in self.static_fractions):
            raise ValueError("static fractions must be in [0, 1]")
        if not self.ranking_methods:
            raise ValueError("at least one ranking method is required")
        return self


def load_static_adaptive_config(path: str | Path) -> StaticAdaptiveConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return StaticAdaptiveConfig.model_validate(raw)


class SimulatorExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["offline_teacher_forced"]


class SimulatorSweepConfig(StrictModel):
    capacity_experts: tuple[int, ...]
    bandwidth_bytes_per_s: tuple[float, ...]
    fixed_latency_us: tuple[float, ...]
    horizons: tuple[int, ...]
    overlap_transfers: tuple[bool, ...]

    @model_validator(mode="after")
    def positive_sweep(self) -> SimulatorSweepConfig:
        if (
            not self.capacity_experts
            or any(value < 1 for value in self.capacity_experts)
            or not self.bandwidth_bytes_per_s
            or any(value <= 0 for value in self.bandwidth_bytes_per_s)
            or not self.fixed_latency_us
            or any(value < 0 for value in self.fixed_latency_us)
            or not self.horizons
            or any(value < 1 for value in self.horizons)
            or not self.overlap_transfers
        ):
            raise ValueError("invalid simulator sensitivity sweep")
        return self


class OffloadSimulatorConfig(StrictModel):
    schema_version: int = 1
    experiment: SimulatorExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    prompt_tokens: tuple[int, ...]
    generated_tokens: int = Field(ge=1)
    baselines: tuple[
        Literal[
            "on_demand",
            "lru",
            "lfu",
            "lossless_predictor",
            "one_step_commitment",
            "multi_step_commitment",
        ],
        ...,
    ]
    per_layer_budget: int = Field(ge=1)
    probe_us: float = Field(ge=0)
    max_concurrent_transfers: int = Field(ge=1)
    dense_compute_us_per_token: float = Field(ge=0)
    expert_compute_us: float = Field(ge=0)
    sweep: SimulatorSweepConfig

    @model_validator(mode="after")
    def valid_simulator(self) -> OffloadSimulatorConfig:
        if not self.prompt_tokens or not self.baselines:
            raise ValueError("simulator prompt and baselines cannot be empty")
        return self


def load_offload_simulator_config(path: str | Path) -> OffloadSimulatorConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return OffloadSimulatorConfig.model_validate(raw)


class OffloadRuntimeExperimentConfig(StrictModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    information_regime: Literal["online_pre_sample"]


class OffloadRuntimeConfig(StrictModel):
    schema_version: int = 1
    experiment: OffloadRuntimeExperimentConfig
    tiny_model_config: str = Field(min_length=1)
    prompt_tokens: tuple[int, ...]
    gpu_slots_per_layer: int = Field(ge=1)
    pinned_memory: bool = True
    warmup_tokens: int = Field(ge=0)
    measured_tokens: int = Field(ge=1)
    repetitions: int = Field(ge=1)
    subset_experts_by_layer: dict[int, tuple[int, ...]] | None = None
    subset_miss_policy: Literal["lossless_fallback", "hard_commit"] = "lossless_fallback"
    atol: float = Field(default=1e-6, ge=0)
    rtol: float = Field(default=1e-5, ge=0)

    @model_validator(mode="after")
    def valid_runtime(self) -> OffloadRuntimeConfig:
        if not self.prompt_tokens:
            raise ValueError("runtime prompt cannot be empty")
        if self.subset_experts_by_layer is not None and any(
            not experts for experts in self.subset_experts_by_layer.values()
        ):
            raise ValueError("configured subset layers cannot be empty")
        return self


def load_offload_runtime_config(path: str | Path) -> OffloadRuntimeConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return OffloadRuntimeConfig.model_validate(raw)
