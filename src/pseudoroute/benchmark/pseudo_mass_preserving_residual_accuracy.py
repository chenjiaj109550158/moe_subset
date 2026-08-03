"""Run the frozen 16-row mass-preserving Qwen/GSM8K accuracy pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from safetensors import safe_open

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_closed_loop import paired_accuracy_bootstrap
from pseudoroute.benchmark.pseudo_embedding_route import _source_model
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    _check_reference,
    _write_checksummed,
)
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    run_policy_sample as run_pseudo_policy_sample,
)
from pseudoroute.benchmark.pseudo_one_forward_hard_oracle import (
    _load_checksummed as _load_legacy_oracle,
)
from pseudoroute.benchmark.pseudo_one_forward_hard_oracle import (
    _sample_path as _legacy_oracle_path,
)
from pseudoroute.benchmark.runner import (
    _load_model,
    _model_example,
    _read_jsonl,
    _software_hardware,
)
from pseudoroute.benchmark.subset_closed_loop import (
    run_policy_sample as run_subset_policy_sample,
)
from pseudoroute.benchmark.subset_config import load_subset_oracle_config
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.benchmark.tasks import load_examples
from pseudoroute.utils.determinism import seed_everything

Policy = Literal[
    "hard_oracle_commitment",
    "previous_route_commitment",
    "natural_top8_intersection_zero_missing",
]
Stage = Literal["smoke", "actual"]
Partition = Literal["wave1", "wave2", "all"]

PILOT_ID = "pseudo_mass_preserving_residual_accuracy_pilot_v1"
CONFIG = Path("configs/benchmark/pseudo_mass_preserving_residual_accuracy_pilot_v1.yaml")
SAMPLES = Path("configs/benchmark/pseudo_mass_preserving_residual_accuracy_pilot_v1_samples.json")
CONFIG_SHA256 = "7370d4c3208fcd32ab7b0f903409a7132c14cd784b7360cc7cf7863720f0d46b"
SAMPLES_SHA256 = "ac1aeb3da4778d09628c2e3188fd524392ce253b9fc04b8c795c0312f82dadfe"
OUTPUT = Path("artifacts/pseudo_mass_preserving_residual_accuracy_pilot_v1")
HORIZON, BUDGET, LAYERS, EXPERTS, TOP_K, CAP = 8, 32, 48, 128, 8, 512
POLICIES: tuple[Policy, ...] = (
    "hard_oracle_commitment",
    "previous_route_commitment",
    "natural_top8_intersection_zero_missing",
)
IDS = (
    "test-44",
    "test-632",
    "test-444",
    "test-519",
    "test-1311",
    "test-1264",
    "test-825",
    "test-252",
    "test-668",
    "test-892",
    "test-1136",
    "test-850",
    "test-760",
    "test-842",
    "test-87",
    "test-658",
)
CANDIDATE: Literal["natural_top8_intersection_zero_missing"] = (
    "natural_top8_intersection_zero_missing"
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("mass-residual accuracy protocol fingerprint changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config.get("pilot_id") != PILOT_ID or samples.get("pilot_id") != PILOT_ID:
        raise ValueError("mass-residual accuracy pilot identity changed")
    if (
        tuple(config["policies"]["order"]) != POLICIES
        or tuple(samples["work_assignment"]["policy_order"]) != POLICIES
        or tuple(row["sample_id"] for row in samples["accuracy"]) != IDS
    ):
        raise ValueError("mass-residual accuracy policies or samples changed")
    point = config["operating_point"]
    model = config["model"]
    if (
        point["horizon"],
        point["budget_per_layer"],
        model["routed_layers"],
        model["routed_experts_per_layer"],
        model["native_top_k"],
    ) != (HORIZON, BUDGET, LAYERS, EXPERTS, TOP_K):
        raise ValueError("mass-residual accuracy operating point changed")
    source_items = (
        ("source_gate", "config"),
        ("source_gate", "sample_manifest"),
        ("source_gate", "artifact_manifest"),
        ("source_suite", "config"),
        ("source_suite", "original_sample_manifest"),
        ("source_suite", "frozen_vanilla_rows"),
        ("source_suite", "static_count_tensor"),
        ("source_suite", "static_count_manifest"),
        ("hard_oracle_reuse", "source_config"),
        ("hard_oracle_reuse", "source_artifact_manifest"),
    )
    for section, key in source_items:
        path = Path(config[section][key])
        if sha256_file(path) != config[section][f"{key}_sha256"]:
            raise ValueError(f"mass-residual accuracy source changed: {path}")
    focused = cast(
        dict[str, Any],
        yaml.safe_load(Path(config["source_suite"]["config"]).read_text(encoding="utf-8")),
    )
    original = _json(Path(config["source_suite"]["original_sample_manifest"]))
    original_rows = [
        *original["partitions"]["closed_loop_wave_1"]["rows"],
        *original["partitions"]["closed_loop_wave_2"]["rows"],
    ]
    original_triples = [
        (row["row_index"], row["sample_id"], row["sha256_rank"]) for row in original_rows
    ]
    frozen_triples = [
        (row["row_index"], row["sample_id"], row["sha256_rank"]) for row in samples["accuracy"]
    ]
    if original_triples != frozen_triples:
        raise ValueError("accuracy rows differ from predeclared closed-loop waves")
    hard_config = cast(
        dict[str, Any],
        yaml.safe_load(
            Path(config["hard_oracle_reuse"]["source_config"]).read_text(encoding="utf-8")
        ),
    )
    subset_path = Path(hard_config["source"]["subset_oracle_config"])
    if sha256_file(subset_path) != hard_config["source"]["subset_oracle_config_sha256"]:
        raise ValueError("transitive subset-oracle config changed")
    subset_suite = load_subset_oracle_config(subset_path)
    if subset_suite.fingerprint() != hard_config["source"]["subset_oracle_config_fingerprint"]:
        raise ValueError("transitive subset-oracle fingerprint changed")
    if focused["source_accuracy"]["config"] != hard_config["source"]["accuracy_config"]:
        raise ValueError("focused and oracle accuracy configs differ")
    return config, samples, focused, hard_config


def _source_rows(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["row_index"]): row
        for row in _read_jsonl(Path(config["source_suite"]["frozen_vanilla_rows"]))
        if row.get("state") == "complete"
    }
    if set(rows) != set(range(1319)):
        raise ValueError("frozen Qwen/GSM8K source row set changed")
    return rows


def _static_subsets_count_only(config: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    source = Path(config["source_suite"]["static_count_tensor"])
    if sha256_file(source) != config["source_suite"]["static_count_tensor_sha256"]:
        raise ValueError("static-frequency count source changed")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        count = handle.get_tensor("count")
    if tuple(count.shape) != (LAYERS, EXPERTS):
        raise ValueError("static-frequency count shape changed")
    result = {}
    for layer in range(LAYERS):
        ranking = sorted(
            range(EXPERTS),
            key=lambda expert: (-float(count[layer, expert]), expert),
        )
        result[layer] = tuple(sorted(ranking[:BUDGET]))
    return result


def _sample_path(stage: Stage, row_index: int, policy: Policy) -> Path:
    return OUTPUT / stage / f"{row_index:05d}" / f"{policy}.json"


def _failure_path(stage: Stage, row_index: int, policy: Policy) -> Path:
    path = _sample_path(stage, row_index, policy)
    return path.with_name(f"{policy}.{os.getpid()}.FAILED.json")


def _load_checksummed(
    path: Path,
    *,
    stage: Stage,
    row_index: int,
    sample_id: str,
    policy: Policy,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _json(path)
    expected = payload.pop("row_payload_sha256", None)
    valid = (
        payload.get("state") == "complete"
        and payload.get("artifact_checksum_serialization") == "json_round_trip_v1"
        and payload.get("pilot_id") == PILOT_ID
        and payload.get("config_sha256") == CONFIG_SHA256
        and payload.get("sample_manifest_sha256") == SAMPLES_SHA256
        and payload.get("stage") == stage
        and int(payload.get("row_index", -1)) == row_index
        and payload.get("sample_id") == sample_id
        and payload.get("policy") == policy
        and int(payload.get("max_new_tokens", -1)) == max_new_tokens
        and expected == sha256_json(payload)
    )
    if not valid:
        raise ValueError(f"incompatible or corrupt accuracy row: {path}")
    payload["row_payload_sha256"] = expected
    return payload


def _exact_matches(generated: list[int], reference: list[int]) -> tuple[int, int]:
    total = max(len(generated), len(reference))
    matches = sum(
        index < len(generated) and index < len(reference) and generated[index] == reference[index]
        for index in range(total)
    )
    return matches, total


def _candidate_weight_audit(row: dict[str, object]) -> bool:
    planning = cast(list[dict[str, Any]], row["planning_rows"])
    if not planning:
        return False
    for index, boundary in enumerate(planning):
        audit = cast(dict[str, Any], boundary["probe_audit"])
        if (
            audit.get("subset_residual_execution") != CANDIDATE
            or not bool(audit.get("full_pre_mask_scores_all_experts"))
            or not bool(audit.get("execution_weights_finite_nonnegative"))
            or audit.get("default_fingerprint") is not None
            or float(audit.get("execution_weight_sum_min", -1.0)) < -1e-6
            or float(audit.get("execution_weight_sum_max", 2.0)) > 1.01
        ):
            return False
        supplied = bool(audit.get("execution_subsets_supplied"))
        if index == 0:
            if supplied or audit.get("execution_scope") != "full_native_topk":
                return False
        elif (
            not supplied
            or audit.get("execution_scope") != "previous_realized_window_subset"
            or audit.get("executed_ids_within_supplied_subset") is not True
        ):
            return False
    return True


def _decorate_subset_row(
    row: dict[str, object],
    *,
    source: dict[str, Any],
    policy: Literal["hard_oracle_commitment", "previous_route_commitment"],
    stage: Stage,
    revision: str,
) -> dict[str, object]:
    generated = cast(list[int], row["generated_token_ids"])
    natural = cast(list[int], row["v17_natural_token_ids"])
    exact_matches, exact_total = _exact_matches(generated, natural)
    previous = policy == "previous_route_commitment"
    row.update(
        {
            "pilot_id": PILOT_ID,
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "stage": stage,
            "policy_role": (
                "deployable_calibration_free_reference"
                if previous
                else "nondeployable_routing_information_oracle_ceiling"
            ),
            "current_policy_closed_loop_context_used": True,
            "future_v17_tokens_used_by_policy": False,
            "label_or_correctness_used_during_policy_execution": False,
            "previous_first_window_static_frequency": previous,
            "static_count_tensor_only_used": previous,
            "default_mean_vectors_used": False,
            "all_planning_audits_pass": True,
            "mass_preserving_weight_audit_pass": None,
            "one_extra_pseudo_forward_per_boundary": False,
            "single_extra_forward_deployable_constraint_satisfied": previous,
            "subset_residual_execution": None,
            "planning_rows": [],
            "boundary_count": math.ceil(max(len(generated) - 1, 0) / HORIZON),
            "probe_latency_seconds_measured": 0.0,
            "probe_attention_queries": 0,
            "probe_attention_calls": 0,
            "probe_router_calls": 0,
            "probe_expert_calls": 0,
            "probe_cpu_gpu_synchronizations": 0,
            "probe_peak_temporary_cuda_bytes_measured": 0,
            "oracle_natural_forward_calls": None if not previous else 0,
            "exact_token_matches": exact_matches,
            "exact_token_comparison_tokens": exact_total,
            "v17_token_nll_on_policy_context": row["vanilla_token_nll_on_policy_context"],
            "simulated_planned_transfer_bytes": row["planned_transfer_bytes"],
            "simulated_natural_reference_bytes": row["natural_reference_bytes"],
            "simulated_estimated_transfer_reduction": row["estimated_transfer_reduction"],
            "peak_cuda_allocated_bytes_measured": row["peak_cuda_allocated_bytes"],
            "execution_git_head": revision,
            "source_v17_correct": bool(source["correct"]),
            "source_v17_generated_tokens": int(source["generated_tokens"]),
            "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
            "source_target_sha256": source["target_sha256"],
        }
    )
    return row


def _decorate_candidate_row(
    row: dict[str, object], *, source: dict[str, Any], revision: str
) -> dict[str, object]:
    row.update(
        {
            "pilot_id": PILOT_ID,
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
            "policy": CANDIDATE,
            "policy_role": "deployable_calibration_free_candidate",
            "information_regime": "online_post_sample_current_policy_known_context_only",
            "current_policy_closed_loop_context_used": True,
            "previous_first_window_static_frequency": False,
            "static_count_tensor_only_used": False,
            "default_mean_vectors_used": False,
            "captured_natural_ids_and_weights_preserved": True,
            "missing_natural_mass_zero": True,
            "substitute_experts_used": False,
            "subset_residual_execution": CANDIDATE,
            "execution_git_head": revision,
            "source_v17_correct": bool(source["correct"]),
            "source_v17_generated_tokens": int(source["generated_tokens"]),
            "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
            "source_target_sha256": source["target_sha256"],
        }
    )
    row["mass_preserving_weight_audit_pass"] = _candidate_weight_audit(row)
    if not bool(row["mass_preserving_weight_audit_pass"]):
        raise RuntimeError("candidate mass-preserving weight audit failed")
    return row


def _work(
    samples: dict[str, Any], *, stage: Stage, partition: Partition
) -> list[tuple[dict[str, Any], Policy]]:
    if stage == "smoke":
        return [
            (samples["accuracy"][0], "previous_route_commitment"),
            (samples["accuracy"][0], CANDIDATE),
        ]
    units = []
    for reference in samples["accuracy"]:
        wave = str(reference["partition"])
        if partition == "wave1" and wave != "closed_loop_wave_1":
            continue
        if partition == "wave2" and wave != "closed_loop_wave_2":
            continue
        for policy in POLICIES:
            if wave == "closed_loop_wave_1" and policy == "hard_oracle_commitment":
                continue
            units.append((reference, policy))
    return units


def run(*, stage: Stage, partition: Partition = "all", physical_gpu: int = 0) -> dict[str, object]:
    config, samples, focused, hard_config = _protocol()
    units = _work(samples, stage=stage, partition=partition)
    cap = int(config["execution"]["mechanism_smoke"]["max_new_tokens"]) if stage == "smoke" else CAP
    missing = [
        (reference, policy)
        for reference, policy in units
        if _load_checksummed(
            _sample_path(stage, int(reference["row_index"]), policy),
            stage=stage,
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=cap,
        )
        is None
    ]
    if not missing:
        return {"state": "already_complete", "stage": stage, "rows": len(units)}
    accuracy = load_accuracy_suite_config(focused["source_accuracy"]["config"])
    model_config = _source_model(accuracy, physical_gpu)
    subset_suite = load_subset_oracle_config(hard_config["source"]["subset_oracle_config"])
    subset_models = [value for value in subset_suite.models if value.key == config["model"]["key"]]
    if len(subset_models) != 1:
        raise ValueError("subset suite is missing frozen Qwen model")
    model_subset = subset_models[0].model_copy(update={"physical_gpu": physical_gpu})
    datasets = [dataset for dataset in accuracy.datasets if dataset.key == "gsm8k"]
    if len(datasets) != 1:
        raise ValueError("frozen accuracy suite is missing GSM8K")
    examples = load_examples(datasets[0], cache_dir=accuracy.dataset_cache_dir)
    sources = _source_rows(config)
    static = _static_subsets_count_only(config)
    seed_everything(int(config["decode"]["seed"]))
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed model facts changed")
    revision = _git_head()
    completed = 0
    for reference, policy in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        _check_reference(reference, source)
        example = _model_example(examples[row_index], model_config)
        if example.sample_id != reference["sample_id"]:
            raise ValueError(f"dataset/source mismatch: {reference['sample_id']}")
        max_tokens = min(example.max_new_tokens, cap)
        path = _sample_path(stage, row_index, policy)
        try:
            if policy == CANDIDATE:
                row = run_pseudo_policy_sample(
                    config,
                    accuracy,
                    model_config,
                    model,
                    tokenizer,
                    ops,
                    example,
                    source,
                    policy="sampled_unigram_full_continuation",
                    stage=stage,
                    max_new_tokens=max_tokens,
                    physical_gpu=physical_gpu,
                    execution_git_head=revision,
                    subset_residual_execution=CANDIDATE,
                )
                row = _decorate_candidate_row(row, source=source, revision=revision)
            else:
                subset_policy = policy
                row = run_subset_policy_sample(
                    subset_suite,
                    accuracy,
                    model_subset,
                    model_config,
                    model,
                    tokenizer,
                    ops,
                    example,
                    source,
                    policy=subset_policy,
                    horizon=HORIZON,
                    budget=BUDGET,
                    max_new_tokens=max_tokens,
                    static_subsets=static,
                )
                row = _decorate_subset_row(
                    row,
                    source=source,
                    policy=subset_policy,
                    stage=stage,
                    revision=revision,
                )
            _write_checksummed(path, row)
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "partition": reference["partition"],
                        "sample_id": reference["sample_id"],
                        "policy": policy,
                        "tokens": row["generated_tokens"],
                        "correct": row["correct"],
                        "elapsed_seconds": row["elapsed_seconds_measured"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException as error:
            write_json_atomic(
                _failure_path(stage, row_index, policy),
                {
                    "schema_version": 1,
                    "state": "failed",
                    "pilot_id": PILOT_ID,
                    "config_sha256": CONFIG_SHA256,
                    "sample_manifest_sha256": SAMPLES_SHA256,
                    "stage": stage,
                    "partition": reference["partition"],
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": policy,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "physical_gpu": physical_gpu,
                    "execution_git_head": revision,
                },
            )
            raise
    return {
        "state": "complete",
        "stage": stage,
        "partition": partition,
        "completed_now": completed,
    }


def audit_smoke() -> dict[str, object]:
    config, samples, _, _ = _protocol()
    reference = samples["accuracy"][0]
    rows = {
        policy: _load_checksummed(
            _sample_path("smoke", int(reference["row_index"]), policy),
            stage="smoke",
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            policy=policy,
            max_new_tokens=int(config["execution"]["mechanism_smoke"]["max_new_tokens"]),
        )
        for policy in ("previous_route_commitment", CANDIDATE)
    }
    if any(row is None for row in rows.values()):
        raise ValueError("mass-residual accuracy mechanism smoke is incomplete")
    previous = cast(dict[str, Any], rows["previous_route_commitment"])
    candidate = cast(dict[str, Any], rows[CANDIDATE])
    all_pass = (
        bool(previous["hard_mask_executed"])
        and not bool(previous["identity_materialized"])
        and bool(previous["previous_first_window_static_frequency"])
        and not bool(previous["future_v17_tokens_used_by_policy"])
        and not bool(previous["label_or_correctness_used_during_policy_execution"])
        and bool(candidate["hard_mask_executed"])
        and not bool(candidate["identity_materialized"])
        and bool(candidate["all_planning_audits_pass"])
        and bool(candidate["mass_preserving_weight_audit_pass"])
        and bool(candidate["single_extra_forward_deployable_constraint_satisfied"])
        and int(candidate["probe_attention_calls"]) == int(candidate["boundary_count"]) * LAYERS
        and int(candidate["probe_attention_queries"])
        == int(candidate["boundary_count"]) * LAYERS * HORIZON
        and not bool(candidate["future_v17_tokens_used_by_policy"])
        and not bool(candidate["label_or_correctness_used_during_policy_execution"])
    )
    if not all_pass:
        raise RuntimeError("mass-residual accuracy mechanism smoke audit failed")
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "sample_policy_rows": 2,
        "all_pass": True,
        "actual_hard_mask_execution": True,
        "identity_materialized_rows": 0,
        "previous_first_window_static_frequency": True,
        "candidate_cache_rng_information_and_weight_audits_pass": True,
        "accuracy_or_correctness_selected_policy": False,
    }
    write_json_atomic(OUTPUT / "smoke" / "audit.json", result)
    return result


def _legacy_oracle_rows(config: dict[str, Any], samples: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config["hard_oracle_reuse"]["source_artifact_manifest"])
    if sha256_file(path) != config["hard_oracle_reuse"]["source_artifact_manifest_sha256"]:
        raise ValueError("legacy hard-oracle artifact manifest changed")
    rows = []
    for reference in samples["accuracy"][:8]:
        row = _load_legacy_oracle(
            _legacy_oracle_path(int(reference["row_index"])),
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            max_new_tokens=CAP,
        )
        if row is None:
            raise ValueError("checksum-pinned wave-one hard-oracle row is missing")
        if bool(row.get("identity_materialized")) or not bool(row.get("hard_mask_executed")):
            raise ValueError("legacy hard-oracle execution semantics changed")
        copied = dict(row)
        copied["row_source"] = "checksum_pinned_pseudo_one_forward_hard_oracle_v1"
        rows.append(copied)
    return rows


def _actual_rows(config: dict[str, Any], samples: dict[str, Any]) -> list[dict[str, Any]]:
    legacy = {str(row["sample_id"]): row for row in _legacy_oracle_rows(config, samples)}
    rows = []
    for reference in samples["accuracy"]:
        for policy in POLICIES:
            if (
                reference["partition"] == "closed_loop_wave_1"
                and policy == "hard_oracle_commitment"
            ):
                row = legacy[str(reference["sample_id"])]
            else:
                loaded = _load_checksummed(
                    _sample_path("actual", int(reference["row_index"]), policy),
                    stage="actual",
                    row_index=int(reference["row_index"]),
                    sample_id=str(reference["sample_id"]),
                    policy=policy,
                    max_new_tokens=CAP,
                )
                if loaded is None:
                    raise ValueError("mass-residual accuracy aggregate is incomplete")
                row = loaded
            rows.append(row)
    return rows


def _partial_rows(
    config: dict[str, Any], samples: dict[str, Any], *, partition: str
) -> list[dict[str, Any]]:
    legacy = {str(row["sample_id"]): row for row in _legacy_oracle_rows(config, samples)}
    rows = []
    for reference in samples["accuracy"]:
        if reference["partition"] != partition:
            continue
        for policy in POLICIES:
            if partition == "closed_loop_wave_1" and policy == "hard_oracle_commitment":
                rows.append(legacy[str(reference["sample_id"])])
                continue
            row = _load_checksummed(
                _sample_path("actual", int(reference["row_index"]), policy),
                stage="actual",
                row_index=int(reference["row_index"]),
                sample_id=str(reference["sample_id"]),
                policy=policy,
                max_new_tokens=CAP,
            )
            if row is None:
                raise ValueError(f"incomplete partial accuracy partition: {partition}")
            rows.append(row)
    return rows


def eta_checkpoint() -> dict[str, object]:
    config, samples, _, _ = _protocol()
    wave_one = [
        row
        for row in _partial_rows(config, samples, partition="closed_loop_wave_1")
        if row["policy"] != "hard_oracle_commitment"
    ]
    if len(wave_one) != 16:
        raise ValueError("wave-one ETA checkpoint requires 16 new rows")
    by_policy = {
        policy: [row for row in wave_one if row["policy"] == policy]
        for policy in ("previous_route_commitment", CANDIDATE)
    }
    wave_one_tokens = sum(int(row["source_v17_generated_tokens"]) for row in by_policy[CANDIDATE])
    wave_two_tokens = sum(
        int(row["source_v17_generated_tokens"]) for row in samples["accuracy"][8:]
    )
    ratios = {
        policy: sum(float(row["elapsed_seconds_measured"]) for row in rows) / wave_one_tokens
        for policy, rows in by_policy.items()
    }
    legacy = _legacy_oracle_rows(config, samples)
    oracle_ratio = sum(float(row["elapsed_seconds_measured"]) for row in legacy) / 2101
    completed = sum(float(row["elapsed_seconds_measured"]) for row in wave_one)
    remaining = wave_two_tokens * (
        ratios["previous_route_commitment"] + ratios[CANDIDATE] + oracle_ratio
    )
    projected_hours = (completed + remaining) / 3600
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "wave_one_new_rows": 16,
        "wave_one_runtime_seconds_measured": completed,
        "wave_one_source_tokens_per_policy": wave_one_tokens,
        "wave_two_source_tokens_per_policy": wave_two_tokens,
        "seconds_per_source_token": {**ratios, "hard_oracle_commitment": oracle_ratio},
        "projected_remaining_seconds": remaining,
        "projected_total_hours": projected_hours,
        "approval_required_before_wave_two": projected_hours > 24.0,
        "threshold_hours": 24.0,
    }
    write_json_atomic(OUTPUT / "eta_checkpoint.json", result)
    return result


def _metric(row: dict[str, Any], primary: str, legacy: str) -> float:
    return float(row[primary] if primary in row else row[legacy])


def _policy_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    successes = sum(bool(row["correct"]) for row in rows)
    route_hits = sum(int(row["route_hits"]) for row in rows)
    route_slots = sum(int(row["route_slots"]) for row in rows)
    mass_hit = sum(float(row["selected_mass_hit"]) for row in rows)
    mass_total = sum(float(row["selected_mass_total"]) for row in rows)
    exact_matches = sum(int(row["exact_token_matches"]) for row in rows)
    exact_total = sum(int(row["exact_token_comparison_tokens"]) for row in rows)
    nll_tokens = sum(int(row["nll_tokens"]) for row in rows)
    weighted_nll = (
        sum(
            _metric(
                row,
                "v17_token_nll_on_policy_context",
                "vanilla_token_nll_on_policy_context",
            )
            * int(row["nll_tokens"])
            for row in rows
        )
        / nll_tokens
    )
    generated = sum(int(row["generated_tokens"]) for row in rows)
    elapsed = sum(float(row["elapsed_seconds_measured"]) for row in rows)
    planned = sum(
        int(row.get("simulated_planned_transfer_bytes", row.get("planned_transfer_bytes", 0)))
        for row in rows
    )
    natural = sum(
        int(
            row.get(
                "simulated_natural_reference_bytes",
                row.get("natural_reference_bytes", 0),
            )
        )
        for row in rows
    )
    return {
        "samples": len(rows),
        "successes": successes,
        "accuracy_measured": successes / len(rows),
        "paired_gains_vs_vanilla": 0,
        "paired_losses_vs_vanilla": len(rows) - successes,
        "paired_ties_vs_vanilla": successes,
        "accuracy_gate_pass": successes >= 15,
        "strong_preservation": successes == 16,
        "total_generated_tokens": generated,
        "total_runtime_seconds_measured": elapsed,
        "aggregate_tokens_per_second_measured": generated / elapsed,
        "exact_token_matches": exact_matches,
        "exact_token_comparison_tokens": exact_total,
        "exact_token_agreement_weighted": exact_matches / exact_total,
        "samples_with_any_token_divergence": sum(
            row["first_token_divergence"] is not None for row in rows
        ),
        "route_hit_rate_weighted": route_hits / route_slots,
        "selected_routing_mass_coverage_weighted": mass_hit / mass_total,
        "weighted_v17_token_nll_on_policy_context": weighted_nll,
        "weighted_perplexity": math.exp(weighted_nll) if weighted_nll < 700 else float("inf"),
        "probe_latency_seconds_measured": sum(
            float(row.get("probe_latency_seconds_measured", 0.0)) for row in rows
        ),
        "simulated_planned_transfer_bytes": planned,
        "simulated_natural_reference_bytes": natural,
        "simulated_estimated_transfer_reduction": 1 - planned / natural,
        "peak_cuda_allocated_bytes_max": max(
            int(
                row.get(
                    "peak_cuda_allocated_bytes_measured",
                    row.get("peak_cuda_allocated_bytes", 0),
                )
            )
            for row in rows
        ),
        "actual_hard_closed_loop_rows": len(rows),
        "identity_materialized_rows": sum(bool(row["identity_materialized"]) for row in rows),
    }


def aggregate() -> dict[str, object]:
    config, samples, _, _ = _protocol()
    rows = _actual_rows(config, samples)
    sources = _source_rows(config)
    policy_rows = {policy: [row for row in rows if row["policy"] == policy] for policy in POLICIES}
    correctness: dict[str, dict[str, bool]] = {
        "vanilla_v17": {
            str(reference["sample_id"]): bool(sources[int(reference["row_index"])]["correct"])
            for reference in samples["accuracy"]
        },
        **{
            policy: {str(row["sample_id"]): bool(row["correct"]) for row in values}
            for policy, values in policy_rows.items()
        },
    }
    bootstrap = paired_accuracy_bootstrap(
        correctness,
        samples=int(config["accuracy_gate"]["paired_bootstrap_samples"]),
        seed=int(config["accuracy_gate"]["paired_bootstrap_seed"]),
    )
    summaries = {policy: _policy_summary(values) for policy, values in policy_rows.items()}
    candidate_pass = bool(summaries[CANDIDATE]["accuracy_gate_pass"])
    decision = "NARROW" if candidate_pass else "STOP/PIVOT"
    per_sample = []
    for reference in samples["accuracy"]:
        policy_values = {}
        for policy in POLICIES:
            row = next(
                value
                for value in policy_rows[policy]
                if value["sample_id"] == reference["sample_id"]
            )
            policy_values[policy] = {
                "correct": bool(row["correct"]),
                "generated_tokens": int(row["generated_tokens"]),
                "first_token_divergence": row["first_token_divergence"],
                "first_route_divergence": row["first_route_divergence"],
            }
        per_sample.append(
            {
                "row_index": int(reference["row_index"]),
                "sample_id": reference["sample_id"],
                "vanilla_v17_correct": True,
                "policies": policy_values,
            }
        )
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "samples": 16,
        "total_sample_policy_rows": 48,
        "new_sample_policy_rows": 40,
        "checksum_pinned_reused_rows": 8,
        "vanilla_v17": {
            "samples": 16,
            "successes": sum(correctness["vanilla_v17"].values()),
            "accuracy_measured_preexisting": 1.0,
            "regenerated": False,
        },
        "policies": summaries,
        "paired_accuracy_bootstrap": bootstrap,
        "per_sample": per_sample,
        "accuracy_gate": {
            "minimum_policy_successes": 15,
            "candidate_pass": candidate_pass,
            "strong_candidate_preservation": summaries[CANDIDATE]["strong_preservation"],
        },
        "decision": decision,
        "conclusion_ceiling": "NARROW",
    }
    write_json_atomic(OUTPUT / "actual" / "summary.json", result)
    write_json_atomic(OUTPUT / "actual" / "paired_accuracy_bootstrap.json", {"rows": bootstrap})
    write_json_atomic(OUTPUT / "actual" / "per_sample.json", {"rows": per_sample})
    return result


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Mass-preserving residual Qwen/GSM8K accuracy pilot",
        "",
        "True hard closed-loop generation at H=8 and B=32 on the frozen 16-row scope.",
        "Eight pre-existing wave-one hard-oracle rows are checksum-pinned measured references.",
        "",
        "| Policy | Correct | Accuracy | Token agreement | Route hit | Selected mass | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        values = summary["policies"][policy]
        lines.append(
            f"| {policy} | {values['successes']}/16 | "
            f"{float(values['accuracy_measured']):.4f} | "
            f"{float(values['exact_token_agreement_weighted']):.6f} | "
            f"{float(values['route_hit_rate_weighted']):.6f} | "
            f"{float(values['selected_routing_mass_coverage_weighted']):.6f} | "
            f"{float(values['total_runtime_seconds_measured']):.2f} |"
        )
    candidate = summary["policies"][CANDIDATE]
    return "\n".join(
        [
            *lines,
            "",
            "## Frozen accuracy gate",
            "",
            f"The candidate scored {candidate['successes']}/16; the predeclared gate is 15/16. "
            f"Focused decision: **{summary['decision']}**.",
            "",
            "Accuracy, token identity, route coverage, NLL/perplexity, probe cost, and research-",
            "runner generation time are measured. Expert transfer/stall are simulated. This N=16",
            "pilot does not establish production offloading speedup or full-dataset preservation.",
            "",
        ]
    )


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _manifest() -> dict[str, object]:
    excluded = {"artifact_manifest.json", "pipeline_status.json"}
    artifacts = []
    for path in sorted(value for value in OUTPUT.rglob("*") if value.is_file()):
        relative = str(path.relative_to(OUTPUT))
        if relative in excluded:
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema_version": 1,
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(OUTPUT / "artifact_manifest.json", manifest)
    return manifest


def finalize() -> dict[str, object]:
    config, samples, _, _ = _protocol()
    summary = cast(dict[str, Any], aggregate())
    rows = _actual_rows(config, samples)
    new_rows = [row for row in rows if row.get("row_source") is None]
    legacy = [row for row in rows if row.get("row_source") is not None]
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    external = {
        "schema_version": 1,
        "source_pilot": config["hard_oracle_reuse"]["source_pilot"],
        "source_artifact_manifest": config["hard_oracle_reuse"]["source_artifact_manifest"],
        "source_artifact_manifest_sha256": config["hard_oracle_reuse"][
            "source_artifact_manifest_sha256"
        ],
        "rows": [
            {
                "sample_id": row["sample_id"],
                "row_index": row["row_index"],
                "path": str(_legacy_oracle_path(int(row["row_index"]))),
                "file_sha256": sha256_file(_legacy_oracle_path(int(row["row_index"]))),
                "row_payload_sha256": row["row_payload_sha256"],
                "evaluation_mode": row["evaluation_mode"],
                "identity_materialized": row["identity_materialized"],
            }
            for row in legacy
        ],
    }
    write_json_atomic(OUTPUT / "resolved_external_references.json", external)
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "git_head_at_report": _git_head(),
            "new_row_execution_git_heads": sorted(
                {str(row["execution_git_head"]) for row in new_rows}
            ),
            "external_row_execution_git_heads": sorted(
                {str(row["execution_git_head"]) for row in legacy}
            ),
            "new_row_pids": sorted({int(row["pid"]) for row in new_rows}),
            "new_row_ppids": sorted({int(row["ppid"]) for row in new_rows}),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "actual_closed_loop_gsm8k_accuracy",
                "exact_token_agreement_and_divergence",
                "v17_token_nll_on_policy_context_and_perplexity",
                "route_hit_and_selected_routing_mass",
                "probe_and_total_generation_runtime",
                "peak_cuda_memory",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction", "transfer_stall"],
            "not_measured": [
                "production_offloading_runtime",
                "production_runtime_speedup",
                "full_dataset_accuracy",
            ],
            "identity_materialized_rows": 0,
            "actual_hard_closed_loop_rows": 48,
            "route_replay_rows": 0,
        },
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "smoke_new_rows_validated": 2,
            "actual_new_rows_validated": len(new_rows),
            "external_rows_checksum_validated": len(legacy),
            "total_sample_policy_rows": len(rows),
            "atomic_json_rows": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "pilot_id": PILOT_ID,
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "sample_ids": list(IDS),
            "vanilla_regenerated": False,
            "hard_oracle_wave_one_regenerated": False,
            "hard_oracle_wave_two_generated": True,
            "current_policy_closed_loop_context_used": True,
            "candidate_calibration_free": True,
            "default_mean_vectors_used": False,
            "static_count_used_only_for_previous_first_window": True,
            "future_v17_tokens_used": False,
            "label_or_correctness_used_during_policy_execution": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "pilot_id": PILOT_ID,
            "decision": summary["decision"],
            "scope": "Qwen_GSM8K_H8_B32_sixteen_row_actual_closed_loop_pilot",
            "candidate": CANDIDATE,
            "accuracy_gate_minimum_successes": 15,
            "candidate_successes": summary["policies"][CANDIDATE]["successes"],
            "conclusion_ceiling": "NARROW",
            "runtime_speedup_claim": False,
            "full_dataset_go": False,
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report(summary))
    manifest = _manifest()
    status = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "decision": summary["decision"],
        "total_sample_policy_rows": 48,
        "new_sample_policy_rows": 40,
        "checksum_pinned_reused_rows": 8,
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": True,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    config, samples, _, _ = _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "report_v1":
        raise ValueError("mass-residual accuracy pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual_files = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual_files != recorded:
        raise ValueError("mass-residual accuracy artifact file set changed")
    for artifact in manifest["artifacts"]:
        path = OUTPUT / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"mass-residual accuracy checksum changed: {path}")
    rows = _actual_rows(config, samples)
    if len(rows) != 48:
        raise ValueError("mass-residual total row count changed")
    if any(
        not bool(row["hard_mask_executed"])
        or bool(row["identity_materialized"])
        or not bool(row["outside_subset_router_logits_masked"])
        or not bool(row["native_topk_and_normalization_after_mask"])
        or bool(row.get("future_v17_tokens_used_by_policy", False))
        or bool(row.get("label_or_correctness_used_during_policy_execution", False))
        for row in rows
    ):
        raise ValueError("mass-residual actual row execution audit changed")
    candidate = [row for row in rows if row["policy"] == CANDIDATE]
    previous = [row for row in rows if row["policy"] == "previous_route_commitment"]
    oracle = [row for row in rows if row["policy"] == "hard_oracle_commitment"]
    if (len(candidate), len(previous), len(oracle)) != (16, 16, 16):
        raise ValueError("mass-residual per-policy row count changed")
    if any(
        not bool(row["all_planning_audits_pass"])
        or not bool(row["mass_preserving_weight_audit_pass"])
        or row["subset_residual_execution"] != CANDIDATE
        or not bool(row["single_extra_forward_deployable_constraint_satisfied"])
        for row in candidate
    ):
        raise ValueError("candidate cache/RNG/information/weight audit changed")
    if any(
        not bool(row["previous_first_window_static_frequency"])
        or not bool(row["static_count_tensor_only_used"])
        for row in previous
    ):
        raise ValueError("corrected previous first-window audit changed")
    if len(_legacy_oracle_rows(config, samples)) != 8:
        raise ValueError("external hard-oracle reuse count changed")
    resume = _json(OUTPUT / "resume_audit.json")
    if (
        resume["smoke_new_rows_validated"] != 2
        or resume["actual_new_rows_validated"] != 40
        or resume["external_rows_checksum_validated"] != 8
        or not resume["checksum_resume_pass"]
    ):
        raise ValueError("mass-residual resume audit changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "total_sample_policy_rows": len(rows),
        "new_sample_policy_rows": 40,
        "external_rows": 8,
        "artifacts": manifest["artifact_count"],
    }


def run_all(*, physical_gpu: int = 0) -> dict[str, object]:
    worker = [
        sys.executable,
        "-m",
        "pseudoroute.benchmark.pseudo_mass_preserving_residual_accuracy",
    ]
    subprocess.run(
        [*worker, "run-smoke", "--gpu", str(physical_gpu)],
        check=True,
    )
    audit_smoke()
    subprocess.run(
        [*worker, "run-wave1", "--gpu", str(physical_gpu)],
        check=True,
    )
    eta = eta_checkpoint()
    if bool(eta["approval_required_before_wave_two"]):
        result = {
            "schema_version": 1,
            "state": "awaiting_user_approval",
            "stage": "wave1_eta_checkpoint",
            "pilot_id": PILOT_ID,
            "projected_total_hours": eta["projected_total_hours"],
        }
        write_json_atomic(OUTPUT / "pipeline_status.json", result)
        return result
    subprocess.run(
        [*worker, "run-wave2", "--gpu", str(physical_gpu)],
        check=True,
    )
    return finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "protocol",
            "run-smoke",
            "smoke-audit",
            "run-wave1",
            "eta",
            "run-wave2",
            "run",
            "aggregate",
            "finalize",
            "validate",
            "all",
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.command == "protocol":
        config, samples, _, _ = _protocol()
        result: dict[str, object] = {
            "state": "valid",
            "pilot_id": config["pilot_id"],
            "samples": len(samples["accuracy"]),
            "config_sha256": CONFIG_SHA256,
            "sample_manifest_sha256": SAMPLES_SHA256,
        }
    elif args.command == "run-smoke":
        result = run(stage="smoke", physical_gpu=args.gpu)
    elif args.command == "smoke-audit":
        result = audit_smoke()
    elif args.command == "run-wave1":
        result = run(stage="actual", partition="wave1", physical_gpu=args.gpu)
    elif args.command == "eta":
        result = eta_checkpoint()
    elif args.command == "run-wave2":
        result = run(stage="actual", partition="wave2", physical_gpu=args.gpu)
    elif args.command == "run":
        result = run(stage="actual", partition="all", physical_gpu=args.gpu)
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    elif args.command == "validate":
        result = validate()
    else:
        result = run_all(physical_gpu=args.gpu)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
