"""Run frozen pseudo-executed embedding-composition analysis waves."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from safetensors.torch import load_file

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    PolicySpec,
    _aggregate_policy,
    _paired_bootstrap,
    _per_sample,
    _save_tensors_atomic,
    _write_gzip_jsonl,
    run_policy_sample,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.runner import _load_model
from pseudoroute.benchmark.subset_trace import sha256_file, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_executed_embedding_composition_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "35d9ef891c4bccf8603cc0e4b98d8cbb51022a82f11ed72782293e374938e8f9"
SAMPLES_SHA256 = "d554e10eed706e9bc35bdfbb91c58347c7e7a23fe23526460b06fc4e76bdaac1"
LAYERS = 48
EXPERTS = 128
TOP_K = 8


def _pseudo(
    key: str,
    content: str,
    *,
    diagnostic: bool = False,
) -> PolicySpec:
    return PolicySpec(
        key,
        "pseudo",
        residual="zero",
        content=cast(Any, content),
        selection="first_four_anchor_core_plus_history_fill",
        diagnostic=diagnostic,
    )


SIMPLE_SPECS = (
    _pseudo("sampled_repeat_independent", "sampled_repeat_independent"),
    _pseudo("current_repeat_independent", "current_repeat_independent"),
    _pseudo("recent_sequence_independent", "recent_sequence_independent"),
    _pseudo("sampled_repeat_causal", "sampled_repeat_causal"),
    _pseudo("recent_sequence_causal", "recent_sequence_causal"),
    _pseudo("exact_future_independent", "exact_future_independent", diagnostic=True),
    _pseudo("exact_future_causal", "exact_future_causal", diagnostic=True),
    PolicySpec(
        "provided_previous_residual_control",
        "pseudo",
        residual="previous_window_position_aligned",
        content="sampled_repeat_independent",
        selection="first_four_anchor_core_plus_history_fill",
    ),
    PolicySpec("previous_route_commitment", "previous"),
)

ADVANCED_SPECS = (
    _pseudo("expected_top4_repeat_independent", "expected_top4_repeat_independent"),
    _pseudo("expected_top8_repeat_independent", "expected_top8_repeat_independent"),
    _pseudo("expected_top16_repeat_independent", "expected_top16_repeat_independent"),
    _pseudo(
        "expected_top8_norm_matched_independent",
        "expected_top8_norm_matched_independent",
    ),
    _pseudo("sampled_then_expected_top8_causal", "sampled_then_expected_top8_causal"),
    _pseudo("self_greedy_causal", "self_greedy_causal"),
    _pseudo("self_expected_top8_causal", "self_expected_top8_causal"),
)
ALL_DEVELOPMENT_SPECS = (*SIMPLE_SPECS, *ADVANCED_SPECS)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("composition protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("composition protocol identity changed")
    for source in ("source_suite", "source_mechanism_smoke"):
        manifest = Path(config[source]["artifact_root"]) / "artifact_manifest.json"
        if sha256_file(manifest) != config[source]["artifact_manifest_sha256"]:
            raise ValueError(f"composition {source} artifact changed")
    if (
        sha256_file(Path(config["source_sample_manifest"]["path"]))
        != config["source_sample_manifest"]["sha256"]
    ):
        raise ValueError("composition source sample manifest changed")
    return config, samples


def _paths(partition: str, row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / partition / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid(
    partition: str,
    reference: dict[str, Any],
    spec: PolicySpec,
) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(partition, int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial composition row: {json_path}")
    row = _json(json_path)
    expected_shadow = spec.role == "pseudo" and spec.key != "provided_previous_residual_control"
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("policy_spec_fingerprint") != spec.fingerprint()
        or row.get("shadow_expert_execution") is not expected_shadow
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible composition row: {json_path}")
    return row


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_wave(
    wave: str,
    *,
    physical_gpu: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    config, samples = _protocol()
    specs = SIMPLE_SPECS if wave == "simple_content_attention" else ADVANCED_SPECS
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid composition shard")
    references = [
        reference
        for index, reference in enumerate(samples["partitions"]["development"])
        if index % shard_count == shard_index
    ]
    missing = [
        (reference, spec)
        for reference in references
        for spec in specs
        if _valid("development", reference, spec) is None
    ]
    if not missing:
        return {"state": "already_complete", "rows": len(references) * len(specs)}
    suite = load_pseudo_embedding_config(config["source_suite"]["config"])
    accuracy = load_accuracy_suite_config(suite.source_accuracy.config)
    model_config = _source_model(accuracy, physical_gpu)
    seed = int(config["decode"]["seed"])
    seed_everything(seed)
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed facts changed")
    sources = _source_rows(suite)
    revision = _git_head()
    completed = 0
    for reference, spec in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        json_path, tensor_path = _paths("development", row_index, spec.key)
        try:
            seed_everything(seed)
            row, tensors = run_policy_sample(
                config,
                suite,
                accuracy,
                model,
                tokenizer,
                ops,
                source,
                spec,
                {},
                stage="development",
                route_token_cap=int(config["operating_point"]["development_route_token_cap"]),
                physical_gpu=physical_gpu,
                shadow_expert_execution=(
                    spec.role == "pseudo" and spec.key != "provided_previous_residual_control"
                ),
                analysis_id=ANALYSIS_ID,
                analysis_config_sha256=CONFIG_SHA256,
                sample_manifest_sha256=SAMPLES_SHA256,
            )
            row["execution_git_head"] = revision
            row["analysis_wave"] = wave
            _save_tensors_atomic(tensor_path, tensors)
            row["tensor_sha256"] = sha256_file(tensor_path)
            row["tensor_bytes"] = tensor_path.stat().st_size
            write_json_atomic(json_path, row)
            completed += 1
        except Exception as error:
            failure = json_path.with_name(f"{json_path.stem}.{os.getpid()}.FAILED.json")
            write_json_atomic(
                failure,
                {
                    "state": "failed",
                    "analysis_id": ANALYSIS_ID,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": spec.key,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "execution_git_head": revision,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return {
        "state": "complete",
        "wave": wave,
        "shard_index": shard_index,
        "completed_now": completed,
    }


def _audit_pass(row: dict[str, Any]) -> bool:
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    for audit in row["cache_rng_audits"]:
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                not audit.get("forbidden_inputs_present", True),
            )
        ):
            return False
        if row["shadow_expert_execution"] and not all(
            (
                audit.get("shadow_expert_execution", False),
                audit.get("full_pre_mask_scores_all_experts", False),
                audit.get("shadow_residual_finite", False),
                audit.get("shadow_residual_nonzero", False),
            )
        ):
            return False
    return True


def _router_sensitivity(
    references: list[dict[str, Any]],
    specs: tuple[PolicySpec, ...],
) -> dict[str, object]:
    baseline_key = "sampled_repeat_independent"
    output: dict[str, object] = {}
    for spec in specs:
        if spec.role != "pseudo" or spec.key == "provided_previous_residual_control":
            continue
        squared = 0.0
        baseline_energy = 0.0
        overlap = 0
        slots = 0
        for reference in references:
            row_index = int(reference["row_index"])
            candidate = load_file(str(_paths("development", row_index, spec.key)[1]))[
                "pseudo_router_logits"
            ].double()
            baseline = load_file(str(_paths("development", row_index, baseline_key)[1]))[
                "pseudo_router_logits"
            ].double()
            candidate = candidate - candidate.mean(dim=-1, keepdim=True)
            baseline = baseline - baseline.mean(dim=-1, keepdim=True)
            squared += float((candidate - baseline).square().sum())
            baseline_energy += float(baseline.square().sum())
            candidate_ids = candidate.topk(TOP_K, dim=-1).indices
            baseline_ids = baseline.topk(TOP_K, dim=-1).indices
            overlap += int(
                (candidate_ids.unsqueeze(-1) == baseline_ids.unsqueeze(-2)).any(dim=-1).sum()
            )
            slots += candidate_ids.numel()
        output[spec.key] = {
            "normalized_centered_logit_rms": (
                (squared / baseline_energy) ** 0.5 if baseline_energy else 0.0
            ),
            "pseudo_top8_overlap_with_sampled_repeat": overlap / slots,
            "compared_router_slots": slots,
        }
    return {
        "reference": baseline_key,
        "comparisons": output,
    }


def _composition_strata(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "route_hits": 0.0,
                "route_slots": 0.0,
                "selected_mass_hit": 0.0,
                "selected_mass_total": 0.0,
            }
        )
    )
    observations: list[dict[str, object]] = []
    for row in rows:
        tensors = load_file(
            str(_paths("development", int(row["row_index"]), str(row["policy"]))[1])
        )
        ids = tensors["natural_router_topk_ids"]
        weights = tensors["natural_router_topk_weights"].float()
        subsets = tensors["subsets"]
        boundaries = [int(value) for value in tensors["boundaries"].tolist()]
        for boundary_index, boundary in enumerate(boundaries):
            end = (
                boundaries[boundary_index + 1]
                if boundary_index + 1 < len(boundaries)
                else ids.shape[0]
            )
            for token_index in range(boundary, end):
                anchor = token_index - boundary + 1
                for layer in range(ids.shape[1]):
                    allowed = set(int(value) for value in subsets[boundary_index, layer])
                    token_ids = ids[token_index, layer]
                    token_weights = weights[token_index, layer]
                    mask = torch.tensor([int(value) in allowed for value in token_ids])
                    observation = {
                        "policy": str(row["policy"]),
                        "sample_id": str(row["sample_id"]),
                        "boundary": boundary,
                        "anchor": anchor,
                        "layer": layer,
                        "route_hits": int(mask.sum()),
                        "route_slots": int(mask.numel()),
                        "selected_mass_hit": float(token_weights[mask].double().sum()),
                        "selected_mass_total": float(token_weights.double().sum()),
                    }
                    observations.append(observation)
                    for axis, value in (
                        ("anchor", str(anchor)),
                        ("layer", str(layer)),
                        ("layer_block", str(layer // 8)),
                    ):
                        key = f"{axis}/{row['policy']}/{value}"
                        aggregate = grouped[axis][key]
                        for field in aggregate:
                            aggregate[field] += float(cast(int | float, observation[field]))
    strata: dict[str, object] = {}
    for axis, values in grouped.items():
        strata[axis] = {
            key: {
                **aggregate,
                "mean_route_hit": aggregate["route_hits"] / aggregate["route_slots"],
                "mean_selected_mass": (
                    aggregate["selected_mass_hit"] / aggregate["selected_mass_total"]
                ),
            }
            for key, aggregate in sorted(values.items())
        }
    worst = sorted(
        observations,
        key=lambda row: (
            float(cast(float, row["selected_mass_hit"]))
            / float(cast(float, row["selected_mass_total"])),
            float(cast(int, row["route_hits"])) / float(cast(int, row["route_slots"])),
            str(row["policy"]),
            str(row["sample_id"]),
            cast(int, row["boundary"]),
            cast(int, row["anchor"]),
            cast(int, row["layer"]),
        ),
    )[:200]
    return strata, worst


def aggregate_development() -> dict[str, object]:
    config, samples = _protocol()
    references = samples["partitions"]["development"]
    rows: list[dict[str, Any]] = []
    for reference in references:
        for spec in ALL_DEVELOPMENT_SPECS:
            row = _valid("development", reference, spec)
            if row is None:
                raise RuntimeError(f"missing development row: {reference['sample_id']}/{spec.key}")
            rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    aggregates = {
        policy: _aggregate_policy(policy_rows) for policy, policy_rows in sorted(grouped.items())
    }
    eligible = set(config["development_variants"]["deployable"])
    selected = sorted(
        eligible,
        key=lambda key: (
            -float(aggregates[key]["mean_selected_mass"]),
            -float(aggregates[key]["mean_route_hit"]),
            -float(aggregates[key]["estimated_transfer_reduction"]),
            float(aggregates[key]["mean_probe_latency_seconds"]),
            key,
        ),
    )[0]
    baseline = "sampled_repeat_independent"
    previous = "previous_route_commitment"
    paired_baseline = _paired_bootstrap(
        {str(row["sample_id"]): _per_sample(row) for row in grouped[selected]},
        {str(row["sample_id"]): _per_sample(row) for row in grouped[baseline]},
    )
    paired_previous = _paired_bootstrap(
        {str(row["sample_id"]): _per_sample(row) for row in grouped[selected]},
        {str(row["sample_id"]): _per_sample(row) for row in grouped[previous]},
    )
    root = OUTPUT / "development"
    write_json_atomic(
        root / "aggregates.json",
        {
            "analysis_id": ANALYSIS_ID,
            "analysis_config_sha256": CONFIG_SHA256,
            "policies": aggregates,
        },
    )
    write_json_atomic(
        root / "selection.json",
        {
            "analysis_id": ANALYSIS_ID,
            "selected_deployable": selected,
            "ranking_uses_accuracy": False,
            "diagnostics_excluded": True,
            "selected_minus_sampled_repeat": {
                "mean_route_hit": float(aggregates[selected]["mean_route_hit"])
                - float(aggregates[baseline]["mean_route_hit"]),
                "mean_selected_mass": float(aggregates[selected]["mean_selected_mass"])
                - float(aggregates[baseline]["mean_selected_mass"]),
            },
            "selected_minus_previous_route": {
                "mean_route_hit": float(aggregates[selected]["mean_route_hit"])
                - float(aggregates[previous]["mean_route_hit"]),
                "mean_selected_mass": float(aggregates[selected]["mean_selected_mass"])
                - float(aggregates[previous]["mean_selected_mass"]),
            },
        },
    )
    write_json_atomic(
        root / "paired_bootstrap.json",
        {
            "selected_vs_sampled_repeat": paired_baseline,
            "selected_vs_previous_route": paired_previous,
        },
    )
    write_json_atomic(
        root / "router_sensitivity.json",
        _router_sensitivity(references, ALL_DEVELOPMENT_SPECS),
    )
    strata, worst = _composition_strata(rows)
    write_json_atomic(root / "stratified_metrics.json", strata)
    write_json_atomic(root / "worst_cases.json", worst)
    _write_gzip_jsonl(
        root / "raw_metrics.jsonl.gz",
        [cast(dict[str, object], metric) for row in rows for metric in row["metrics"]],
    )
    audit = {
        "all_pass": all(_audit_pass(row) for row in rows),
        "sample_policy_rows": len(rows),
        "cache_rng_shadow_information": True,
        "future_true_tokens_only_in_diagnostics": True,
        "accuracy_or_correctness_used": False,
    }
    write_json_atomic(root / "audit.json", audit)
    write_json_atomic(
        root / "cost_report.json",
        {
            policy: {
                key: value
                for key, value in aggregate.items()
                if key
                in {
                    "mean_probe_latency_seconds",
                    "max_temporary_cuda_bytes",
                    "attention_queries",
                    "attention_calls",
                    "router_calls",
                    "expert_calls",
                    "total_sample_policy_elapsed_seconds",
                }
            }
            for policy, aggregate in aggregates.items()
        },
    )
    result = {
        "state": "complete",
        "sample_policy_rows": len(rows),
        "selected_deployable": selected,
        "all_audits_pass": audit["all_pass"],
    }
    write_json_atomic(root / "pipeline_status.json", result)
    return result


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run-simple", "run-advanced", "aggregate-development"),
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.command == "aggregate-development":
        print(json.dumps(aggregate_development(), indent=2, sort_keys=True))
        return
    result = run_wave(
        (
            "simple_content_attention"
            if args.command == "run-simple"
            else "expected_and_autoregressive"
        ),
        physical_gpu=args.gpu,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
