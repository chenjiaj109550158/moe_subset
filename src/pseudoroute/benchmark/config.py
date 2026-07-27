"""Strict configuration for the paper-comparison accuracy suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AccuracyModelConfig(StrictModel):
    key: Literal["qwen3_30b_a3b", "gpt_oss_20b"]
    architecture: Literal["qwen3_moe", "gpt_oss"]
    model_id: str
    revision: str = Field(min_length=40, max_length=40)
    precision_tier: Literal["bfloat16", "mxfp4"]
    device: Literal["cuda:0", "cuda:1"]
    reasoning_effort: Literal["none", "low", "medium", "high"]
    max_new_tokens_overrides: dict[str, int] = Field(default_factory=dict)
    honor_task_stop_strings: bool


class AccuracyDatasetConfig(StrictModel):
    key: Literal["humaneval", "mbpp_plus", "gsm8k", "aime24", "aime25", "strategyqa"]
    dataset_id: str
    config: str | None = None
    split: str
    revision: str = Field(min_length=40, max_length=40)
    expected_samples: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)


class CalibrationConfig(StrictModel):
    dataset_id: str
    config: str
    split: str
    revision: str = Field(min_length=40, max_length=40)
    row_ids: tuple[int, ...]
    max_tokens_per_row: int = Field(gt=0)
    definition: Literal["mean_unweighted_selected_expert_output"]
    unobserved_expert_value: Literal["zero"]


class DecodeConfig(StrictModel):
    deterministic: Literal[True]
    do_sample: Literal[False]
    seed: int = Field(ge=0)
    batch_size: Literal[1]
    use_cache: Literal[True]
    chat_template_current_date: str


class AlignmentConfig(StrictModel):
    standard_errors: float = Field(gt=0)
    minimum_discrete_questions: int = Field(ge=1)
    require_all_tasks: bool


class PaperResult(StrictModel):
    vanilla_accuracy: float = Field(ge=0, le=1)
    vanilla_standard_error: float = Field(gt=0)
    router_pf_accuracy: float = Field(ge=0, le=1)


class AccuracySuiteConfig(StrictModel):
    schema_version: Literal[1]
    protocol_revision: Literal[3]
    code_execution_sandbox_revision: Literal["v2_preload_doctest_ssl_before_socket_block"]
    suite_id: str
    paper_url: str
    paper_code_url: str
    paper_code_revision: str = Field(min_length=40, max_length=40)
    harness_revision: str = Field(min_length=40, max_length=40)
    evalplus_revision: str = Field(min_length=40, max_length=40)
    cache_dir: str
    dataset_cache_dir: str
    models: tuple[AccuracyModelConfig, ...]
    datasets: tuple[AccuracyDatasetConfig, ...]
    calibration: CalibrationConfig
    decode: DecodeConfig
    policies: tuple[Literal["vanilla", "router_pf", "oracle_pf"], ...]
    oracle_evaluation: Literal["validated_identity_materialization"]
    alignment: AlignmentConfig
    paper_results: dict[str, dict[str, PaperResult]]

    @model_validator(mode="after")
    def validate_cross_references(self) -> AccuracySuiteConfig:
        model_keys = {model.key for model in self.models}
        dataset_keys = {dataset.key for dataset in self.datasets}
        if len(model_keys) != len(self.models):
            raise ValueError("model keys must be unique")
        if len(dataset_keys) != len(self.datasets):
            raise ValueError("dataset keys must be unique")
        if set(self.paper_results) != model_keys:
            raise ValueError("paper_results must cover every model")
        for model_key, tasks in self.paper_results.items():
            if set(tasks) != dataset_keys:
                raise ValueError(f"paper_results[{model_key}] must cover every dataset")
        if self.policies != ("vanilla", "router_pf", "oracle_pf"):
            raise ValueError("policy order is fixed before evaluation")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_accuracy_suite_config(path: str | Path) -> AccuracySuiteConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("accuracy suite config root must be a mapping")
    return AccuracySuiteConfig.model_validate(raw)
