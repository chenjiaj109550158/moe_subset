"""Run the frozen matched-eight Qwen/GSM8K hard routing-oracle pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_closed_loop import paired_accuracy_bootstrap
from pseudoroute.benchmark.pseudo_embedding_route import _source_model
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    _check_reference,
    _source_rows,
    _write_checksummed,
)
from pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot import (
    _protocol as _parent_protocol,
)
from pseudoroute.benchmark.runner import _load_model, _model_example, _software_hardware
from pseudoroute.benchmark.subset_closed_loop import run_policy_sample
from pseudoroute.benchmark.subset_config import load_subset_oracle_config
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic
from pseudoroute.benchmark.tasks import load_examples
from pseudoroute.utils.determinism import seed_everything

PILOT_ID = "pseudo_one_forward_hard_oracle_v1"
POLICY: Literal["hard_oracle_commitment"] = "hard_oracle_commitment"
CONFIG = Path("configs/benchmark/pseudo_one_forward_hard_oracle_v1.yaml")
SAMPLES = Path("configs/benchmark/pseudo_one_forward_accuracy_pilot_v1_samples.json")
CONFIG_SHA256 = "60e03148a24ad16859ca21631a3644d4db6930b0bdaddc482060646b77e902b6"
SAMPLES_SHA256 = "fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046"
OUTPUT = Path("artifacts/pseudo_one_forward_hard_oracle_v1")
HORIZON = 8
BUDGET = 32
LAYERS = 48
EXPERTS = 128
TOP_K = 8
IDS = (
    "test-44",
    "test-632",
    "test-444",
    "test-519",
    "test-1311",
    "test-1264",
    "test-825",
    "test-252",
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("matched hard-oracle config fingerprint changed")
    if sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("matched hard-oracle sample fingerprint changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    parent_config, samples = _parent_protocol()
    if config.get("pilot_id") != PILOT_ID:
        raise ValueError("matched hard-oracle pilot identity changed")
    if tuple(config["dataset"]["sample_ids"]) != IDS:
        raise ValueError("matched hard-oracle config sample order changed")
    if tuple(row["sample_id"] for row in samples["accuracy"]) != IDS:
        raise ValueError("matched hard-oracle source sample order changed")
    for key in (
        "parent_config",
        "resolved_sample_manifest",
        "parent_artifact_manifest",
        "subset_oracle_config",
        "accuracy_config",
        "frozen_vanilla_rows",
    ):
        path = Path(config["source"][key])
        if sha256_file(path) != config["source"][f"{key}_sha256"]:
            raise ValueError(f"matched hard-oracle source changed: {path}")
    suite = load_subset_oracle_config(config["source"]["subset_oracle_config"])
    if suite.fingerprint() != config["source"]["subset_oracle_config_fingerprint"]:
        raise ValueError("matched hard-oracle base config fingerprint changed")
    point = config["operating_point"]
    if (
        point["horizon"],
        point["budget_per_layer"],
        point["routed_experts_per_layer"],
        point["native_top_k"],
    ) != (HORIZON, BUDGET, EXPERTS, TOP_K):
        raise ValueError("matched hard-oracle operating point changed")
    if config["policy"]["name"] != POLICY:
        raise ValueError("matched hard-oracle policy changed")
    return config, samples, parent_config


def _sample_path(row_index: int) -> Path:
    return OUTPUT / "actual" / f"{row_index:05d}" / f"{POLICY}.json"


def _failure_path(row_index: int) -> Path:
    return _sample_path(row_index).with_name(f"{POLICY}.{os.getpid()}.FAILED.json")


def _load_checksummed(
    path: Path, *, row_index: int, sample_id: str, max_new_tokens: int
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _json(path)
    expected = payload.pop("row_payload_sha256", None)
    if (
        payload.get("state") != "complete"
        or payload.get("artifact_checksum_serialization") != "json_round_trip_v1"
        or payload.get("pilot_id") != PILOT_ID
        or payload.get("config_sha256") != CONFIG_SHA256
        or payload.get("sample_manifest_sha256") != SAMPLES_SHA256
        or int(payload.get("row_index", -1)) != row_index
        or payload.get("sample_id") != sample_id
        or payload.get("policy") != POLICY
        or int(payload.get("max_new_tokens", -1)) != max_new_tokens
        or expected != sha256_json(payload)
    ):
        raise ValueError(f"incompatible or corrupt hard-oracle artifact: {path}")
    payload["row_payload_sha256"] = expected
    return payload


def _exact_matches(generated: list[int], reference: list[int]) -> tuple[int, int]:
    total = max(len(generated), len(reference))
    matches = sum(
        index < len(generated) and index < len(reference) and generated[index] == reference[index]
        for index in range(total)
    )
    return matches, total


def run(*, physical_gpu: int = 0) -> dict[str, object]:
    config, samples, parent_config = _protocol()
    cap = int(config["dataset"]["original_v17_max_new_tokens"])
    missing = [
        reference
        for reference in samples["accuracy"]
        if _load_checksummed(
            _sample_path(int(reference["row_index"])),
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            max_new_tokens=cap,
        )
        is None
    ]
    if not missing:
        return {"state": "already_complete", "rows": len(samples["accuracy"])}
    accuracy = load_accuracy_suite_config(config["source"]["accuracy_config"])
    model_config = _source_model(accuracy, physical_gpu)
    suite = load_subset_oracle_config(config["source"]["subset_oracle_config"])
    subset_matches = [model for model in suite.models if model.key == config["model"]["key"]]
    if len(subset_matches) != 1:
        raise ValueError("base subset suite is missing frozen Qwen model")
    model_subset = subset_matches[0].model_copy(update={"physical_gpu": physical_gpu})
    datasets = [dataset for dataset in accuracy.datasets if dataset.key == config["dataset"]["key"]]
    if len(datasets) != 1:
        raise ValueError("frozen accuracy suite is missing GSM8K")
    examples = load_examples(datasets[0], cache_dir=accuracy.dataset_cache_dir)
    sources = _source_rows(parent_config)
    seed_everything(int(config["decode"]["seed"]))
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = Qwen3MoePrefetchOps(model)
    if (ops.num_layers, ops.num_experts, ops.top_k) != (LAYERS, EXPERTS, TOP_K):
        raise ValueError("runtime Qwen routed model facts changed")
    revision = _git_head()
    completed = 0
    for reference in missing:
        row_index = int(reference["row_index"])
        source = sources[row_index]
        _check_reference(reference, source)
        example = _model_example(examples[row_index], model_config)
        if example.sample_id != reference["sample_id"]:
            raise ValueError(f"dataset/source mismatch: {reference['sample_id']}")
        max_tokens = min(example.max_new_tokens, cap)
        try:
            row = run_policy_sample(
                suite,
                accuracy,
                model_subset,
                model_config,
                model,
                tokenizer,
                ops,
                example,
                source,
                policy=POLICY,
                horizon=HORIZON,
                budget=BUDGET,
                max_new_tokens=max_tokens,
            )
            generated = cast(list[int], row["generated_token_ids"])
            natural = cast(list[int], row["v17_natural_token_ids"])
            exact_matches, exact_total = _exact_matches(generated, natural)
            row.update(
                {
                    "pilot_id": PILOT_ID,
                    "config_sha256": CONFIG_SHA256,
                    "sample_manifest_sha256": SAMPLES_SHA256,
                    "policy_role": "nondeployable_routing_information_oracle_ceiling",
                    "hard_mask_executed": True,
                    "identity_materialized": False,
                    "outside_subset_router_logits_masked": True,
                    "native_topk_and_normalization_after_mask": True,
                    "current_policy_closed_loop_context_used": True,
                    "future_v17_tokens_used_by_policy": False,
                    "label_or_correctness_used_during_policy_execution": False,
                    "oracle_lookahead_cache_and_rng_restored": True,
                    "exact_token_matches": exact_matches,
                    "exact_token_comparison_tokens": exact_total,
                    "execution_git_head": revision,
                    "source_v17_correct": bool(source["correct"]),
                    "source_v17_generated_tokens": int(source["generated_tokens"]),
                    "source_rendered_prompt_sha256": source["rendered_prompt_sha256"],
                    "source_target_sha256": source["target_sha256"],
                }
            )
            _write_checksummed(_sample_path(row_index), row)
            completed += 1
            print(
                json.dumps(
                    {
                        "sample_id": reference["sample_id"],
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
                _failure_path(row_index),
                {
                    "schema_version": 1,
                    "state": "failed",
                    "pilot_id": PILOT_ID,
                    "config_sha256": CONFIG_SHA256,
                    "sample_manifest_sha256": SAMPLES_SHA256,
                    "row_index": row_index,
                    "sample_id": reference["sample_id"],
                    "policy": POLICY,
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
    return {"state": "complete", "completed_now": completed}


def _actual_rows(config: dict[str, Any], samples: dict[str, Any]) -> list[dict[str, Any]]:
    cap = int(config["dataset"]["original_v17_max_new_tokens"])
    rows = []
    for reference in samples["accuracy"]:
        row = _load_checksummed(
            _sample_path(int(reference["row_index"])),
            row_index=int(reference["row_index"]),
            sample_id=str(reference["sample_id"]),
            max_new_tokens=cap,
        )
        if row is None:
            raise ValueError("matched hard-oracle aggregate is incomplete")
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, object]:
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
            float(row["vanilla_token_nll_on_policy_context"]) * int(row["nll_tokens"])
            for row in rows
        )
        / nll_tokens
    )
    generated = sum(int(row["generated_tokens"]) for row in rows)
    elapsed = sum(float(row["elapsed_seconds_measured"]) for row in rows)
    planned = sum(int(row["planned_transfer_bytes"]) for row in rows)
    natural = sum(int(row["natural_reference_bytes"]) for row in rows)
    return {
        "samples": len(rows),
        "successes": successes,
        "accuracy_measured": successes / len(rows),
        "paired_gains_vs_vanilla": 0,
        "paired_losses_vs_vanilla": len(rows) - successes,
        "paired_ties_vs_vanilla": successes,
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
        "simulated_planned_transfer_bytes": planned,
        "simulated_natural_reference_bytes": natural,
        "simulated_estimated_transfer_reduction": 1 - planned / natural,
        "peak_cuda_allocated_bytes_max": max(int(row["peak_cuda_allocated_bytes"]) for row in rows),
        "actual_hard_closed_loop_rows": len(rows),
        "identity_materialized_rows": sum(bool(row["identity_materialized"]) for row in rows),
    }


def aggregate() -> dict[str, object]:
    config, samples, parent_config = _protocol()
    rows = _actual_rows(config, samples)
    sources = _source_rows(parent_config)
    correctness = {
        "vanilla_v17": {
            str(reference["sample_id"]): bool(sources[int(reference["row_index"])]["correct"])
            for reference in samples["accuracy"]
        },
        POLICY: {str(row["sample_id"]): bool(row["correct"]) for row in rows},
    }
    bootstrap = paired_accuracy_bootstrap(
        correctness,
        samples=int(config["accuracy_reporting"]["paired_bootstrap_samples"]),
        seed=int(config["accuracy_reporting"]["paired_bootstrap_seed"]),
    )
    policy_summary = _summary(rows)
    result = {
        "schema_version": 1,
        "state": "complete",
        "pilot_id": PILOT_ID,
        "config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "samples": len(rows),
        "sample_policy_rows": len(rows),
        "vanilla_v17": {
            "samples": len(rows),
            "successes": sum(correctness["vanilla_v17"].values()),
            "accuracy_measured_preexisting": sum(correctness["vanilla_v17"].values()) / len(rows),
            "regenerated": False,
        },
        POLICY: policy_summary,
        "paired_accuracy_bootstrap": bootstrap,
        "per_sample": [
            {
                "row_index": int(row["row_index"]),
                "sample_id": row["sample_id"],
                "vanilla_v17_correct": bool(row["source_v17_correct"]),
                "hard_oracle_correct": bool(row["correct"]),
                "generated_tokens": int(row["generated_tokens"]),
                "exact_token_agreement": float(row["exact_token_agreement"]),
                "first_token_divergence": row["first_token_divergence"],
                "first_route_divergence": row["first_route_divergence"],
                "elapsed_seconds_measured": float(row["elapsed_seconds_measured"]),
            }
            for row in rows
        ],
        "decision": "PILOT_NARROW_DIAGNOSTIC_ORACLE_CEILING",
        "does_not_modify_parent_pilot_decision": True,
        "conclusion_ceiling": "PILOT_NARROW",
    }
    write_json_atomic(OUTPUT / "summary.json", result)
    write_json_atomic(OUTPUT / "paired_accuracy_bootstrap.json", {"rows": bootstrap})
    return result


def _report(summary: dict[str, Any]) -> str:
    values = summary[POLICY]
    ci = summary["paired_accuracy_bootstrap"][0]["paired_bootstrap_ci95"]
    return "\n".join(
        [
            "# Matched-eight Qwen/GSM8K hard routing-oracle pilot",
            "",
            "True hard closed-loop generation at H=8 and B=32 on the exact eight parent rows.",
            "",
            f"- Hard oracle accuracy (measured): {values['successes']}/8 "
            f"({float(values['accuracy_measured']):.2%})",
            "- Frozen same-row vanilla accuracy (pre-existing measured rows): 8/8 (100.00%)",
            f"- Paired gains/losses/ties versus vanilla: "
            f"{values['paired_gains_vs_vanilla']}/{values['paired_losses_vs_vanilla']}/"
            f"{values['paired_ties_vs_vanilla']}",
            f"- Paired accuracy difference bootstrap 95% CI: "
            f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]",
            f"- Exact-token agreement (weighted): "
            f"{float(values['exact_token_agreement_weighted']):.6f}",
            f"- Route hit (weighted): {float(values['route_hit_rate_weighted']):.6f}",
            f"- Selected routing mass (weighted): "
            f"{float(values['selected_routing_mass_coverage_weighted']):.6f}",
            f"- Total measured runtime: {float(values['total_runtime_seconds_measured']):.2f} s",
            f"- Simulated transfer reduction: "
            f"{float(values['simulated_estimated_transfer_reduction']):.6f}",
            "",
            "The oracle looks ahead from its own closed-loop context and truly masks outside each ",
            "layer's top-32 subset. Accuracy/runtime are measured; transfer is simulated. ",
            "N=8 only supports a PILOT_NARROW diagnostic and does not alter the parent decision.",
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
    config, samples, _ = _protocol()
    summary = cast(dict[str, Any], aggregate())
    rows = _actual_rows(config, samples)
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "git_head_at_report": _git_head(),
            "row_execution_git_heads": sorted({str(row["execution_git_head"]) for row in rows}),
            "row_pids": sorted({int(row["pid"]) for row in rows}),
            "row_ppids": sorted({int(row["ppid"]) for row in rows}),
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
                "total_generation_runtime_and_peak_cuda_memory",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": ["production_offloading_runtime", "production_runtime_speedup"],
        },
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "actual_sample_policy_rows_validated": len(rows),
            "atomic_json_rows": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "pilot_id": PILOT_ID,
            "parent_pilot_id": config["source"]["parent_pilot_id"],
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "sample_ids": list(IDS),
            "vanilla_regenerated": False,
            "hard_oracle_generation_executed": True,
            "identity_materialized": False,
            "current_policy_closed_loop_context_used": True,
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
            "scope": "Qwen_GSM8K_H8_B32_matched_eight_hard_oracle_diagnostic",
            "conclusion_ceiling": "PILOT_NARROW",
            "deployable": False,
            "does_not_modify_parent_pilot_decision": True,
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
        "actual_sample_policy_rows": len(rows),
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": True,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", status)
    validate()
    return status


def validate() -> dict[str, object]:
    config, samples, _ = _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "report_v1":
        raise ValueError("matched hard-oracle pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual_files = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual_files != recorded:
        raise ValueError("matched hard-oracle artifact file set changed")
    for artifact in manifest["artifacts"]:
        path = OUTPUT / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"matched hard-oracle checksum changed: {path}")
    rows = _actual_rows(config, samples)
    if len(rows) != 8 or tuple(str(row["sample_id"]) for row in rows) != IDS:
        raise ValueError("matched hard-oracle row count or order changed")
    if any(
        row["policy"] != POLICY
        or not bool(row["hard_mask_executed"])
        or bool(row["identity_materialized"])
        or not bool(row["outside_subset_router_logits_masked"])
        or not bool(row["native_topk_and_normalization_after_mask"])
        or not bool(row["current_policy_closed_loop_context_used"])
        or not bool(row["oracle_lookahead_cache_and_rng_restored"])
        or bool(row["future_v17_tokens_used_by_policy"])
        or bool(row["label_or_correctness_used_during_policy_execution"])
        for row in rows
    ):
        raise ValueError("matched hard-oracle row audit changed")
    resume = _json(OUTPUT / "resume_audit.json")
    if resume["actual_sample_policy_rows_validated"] != 8 or not resume["checksum_resume_pass"]:
        raise ValueError("matched hard-oracle resume audit changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "actual_sample_policy_rows": len(rows),
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate", "finalize", "validate"))
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.command == "run":
        result = run(physical_gpu=args.gpu)
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
