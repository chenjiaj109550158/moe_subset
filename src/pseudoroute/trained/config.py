"""Strict versioned configuration for the trained-model oracle gate."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainedModelConfig(StrictModel):
    key: str
    adapter: Literal["hf_olmoe", "hf_qwen2_moe", "hf_mixtral", "hf_deepseek_v2", "hf_gpt_oss"]
    model_id: str
    revision: str = Field(min_length=40, max_length=40)
    tier: Literal["floating_point", "mxfp4"]
    placement: Literal["cuda0", "balanced_2gpu"]
    full_router_mass_valid: bool = True


class TrainedDatasetConfig(StrictModel):
    key: Literal["wikitext", "gsm8k"]
    dataset_id: str
    config: str
    split: str
    revision: str = Field(min_length=40, max_length=40)
    row_ids: tuple[int, ...]
    prompt_template: str


class TrainedTraceConfig(StrictModel):
    max_length: int = Field(ge=17)
    seed: int = Field(ge=0)


class TrainedOracleConfig(StrictModel):
    horizons: tuple[int, ...]
    budget_multiples: tuple[float, ...]
    absolute_budgets: tuple[int, ...]
    bootstrap_samples: int = Field(ge=100)
    confidence: float = Field(gt=0.5, lt=1)


class TrainedClosedLoopConfig(StrictModel):
    sample_ids: tuple[str, ...]
    max_new_tokens: int = Field(ge=4)
    horizon: int = Field(ge=1)
    budget_multiple: float = Field(ge=1)
    simulated_bandwidth_gib_s: float = Field(gt=0)
    simulated_fixed_latency_us: float = Field(ge=0)


class TrainedGateThresholds(StrictModel):
    minimum_transfer_reduction: float = Field(ge=0, le=1)
    minimum_oracle_hit_rate: float = Field(ge=0, le=1)
    minimum_oracle_selected_mass: float = Field(ge=0, le=1)
    minimum_p05_hit_rate: float = Field(ge=0, le=1)
    maximum_relative_perplexity_increase: float = Field(ge=0)
    maximum_lossless_fallback_rate: float = Field(ge=0, le=1)
    minimum_baseline_improvement: float = Field(ge=0)


class TrainedSuiteConfig(StrictModel):
    schema_version: Literal[1]
    suite_id: str
    cache_dir: str
    models: tuple[TrainedModelConfig, ...]
    datasets: tuple[TrainedDatasetConfig, ...]
    trace: TrainedTraceConfig
    oracle: TrainedOracleConfig
    closed_loop: TrainedClosedLoopConfig
    thresholds: TrainedGateThresholds

    @model_validator(mode="after")
    def unique_and_complete(self) -> TrainedSuiteConfig:
        if len({model.key for model in self.models}) != len(self.models):
            raise ValueError("model keys must be unique")
        if len({dataset.key for dataset in self.datasets}) != len(self.datasets):
            raise ValueError("dataset keys must be unique")
        if set(self.oracle.horizons) != {1, 2, 4, 8, 16}:
            raise ValueError("trained gate requires horizons 1,2,4,8,16")
        if {dataset.key for dataset in self.datasets} != {"wikitext", "gsm8k"}:
            raise ValueError("trained gate requires pinned WikiText and GSM8K")
        return self


def load_trained_suite_config(path: str | Path) -> TrainedSuiteConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TrainedSuiteConfig.model_validate(raw)
