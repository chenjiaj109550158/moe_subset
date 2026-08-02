"""Run the frozen one-forward state-correction analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
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
    run_policy_sample,
)
from pseudoroute.benchmark.pseudo_embedding_route import _source_model, _source_rows
from pseudoroute.benchmark.qwen_pseudo import StateCorrection
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import (
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_one_forward_state_correction_v1"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "7f3c519996c0a190930ef9de4edc138200e3a659287bc171d966e09219803c44"
SAMPLES_SHA256 = "350cc9ddcc003daa6973d247434bc600ecfc108ad7b9773f5e0cb997936ec80c"
LAYERS = 48
EXPERTS = 128
TOP_K = 8


@dataclass(frozen=True)
class CorrectionSpec:
    key: str
    hidden_state_correction: StateCorrection = "none"
    residual_correction: StateCorrection = "none"

    def policy(self) -> PolicySpec:
        return PolicySpec(
            self.key,
            "pseudo",
            residual="zero",
            content="recent_sequence_causal",
            selection="first_four_anchor_core_plus_history_fill",
        )

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


SPECS = (
    CorrectionSpec("recent_sequence_causal_uncorrected"),
    CorrectionSpec(
        "recent_sequence_causal_hidden_velocity_linear_norm",
        hidden_state_correction="recent_linear_norm",
    ),
    CorrectionSpec(
        "recent_sequence_causal_residual_velocity_linear_norm",
        residual_correction="recent_linear_norm",
    ),
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("one-forward correction protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("one-forward correction protocol identity changed")
    source_path = Path(str(config["source"]["config"]))
    if sha256_file(source_path) != config["source"]["config_sha256"]:
        raise ValueError("source composition config checksum changed")
    source = cast(dict[str, Any], yaml.safe_load(source_path.read_text(encoding="utf-8")))
    if (
        sha256_file(Path(str(config["source"]["sample_manifest"])))
        != config["source"]["sample_manifest_sha256"]
    ):
        raise ValueError("source composition sample manifest checksum changed")
    source_manifest = Path(str(source["artifact_root"])) / "artifact_manifest.json"
    if sha256_file(source_manifest) != config["source"]["artifact_manifest_sha256"]:
        raise ValueError("source composition artifact manifest checksum changed")
    suite_manifest = Path(str(source["source_suite"]["artifact_root"])) / "artifact_manifest.json"
    if sha256_file(suite_manifest) != source["source_suite"]["artifact_manifest_sha256"]:
        raise ValueError("source suite artifact manifest checksum changed")
    return config, samples, source


def _paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "development" / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid(reference: dict[str, Any], spec: CorrectionSpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial one-forward correction row: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("correction_spec_fingerprint") != spec.fingerprint()
        or row.get("hidden_state_correction") != spec.hidden_state_correction
        or row.get("residual_correction") != spec.residual_correction
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible one-forward correction row: {json_path}")
    return row


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(*, physical_gpu: int, shard_index: int, shard_count: int) -> dict[str, object]:
    config, samples, source_config = _protocol()
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid one-forward correction shard")
    references = [
        reference
        for index, reference in enumerate(samples["partitions"]["development"])
        if index % shard_count == shard_index
    ]
    missing = [
        (reference, spec)
        for reference in references
        for spec in SPECS
        if _valid(reference, spec) is None
    ]
    if not missing:
        return {"state": "already_complete", "rows": len(references) * len(SPECS)}
    suite = load_pseudo_embedding_config(source_config["source_suite"]["config"])
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
    for reference, correction in missing:
        row_index = int(reference["row_index"])
        json_path, tensor_path = _paths(row_index, correction.key)
        try:
            seed_everything(seed)
            row, tensors = run_policy_sample(
                config,
                suite,
                accuracy,
                model,
                tokenizer,
                ops,
                sources[row_index],
                correction.policy(),
                {},
                stage="development",
                route_token_cap=int(config["operating_point"]["route_token_cap"]),
                physical_gpu=physical_gpu,
                shadow_expert_execution=True,
                analysis_id=ANALYSIS_ID,
                analysis_config_sha256=CONFIG_SHA256,
                sample_manifest_sha256=SAMPLES_SHA256,
                hidden_state_correction=correction.hidden_state_correction,
                residual_correction=correction.residual_correction,
            )
            row["correction_spec"] = asdict(correction)
            row["correction_spec_fingerprint"] = correction.fingerprint()
            row["execution_git_head"] = revision
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
                    "policy": correction.key,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "execution_git_head": revision,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    return {"state": "complete", "shard_index": shard_index, "completed_now": completed}


def _audit_pass(row: dict[str, Any], spec: CorrectionSpec, config: dict[str, Any]) -> bool:
    if not row["prompt_capture_audit"]["production_rng_unchanged"]:
        return False
    expected = config["audits"]
    for cost, audit in zip(row["probe_costs"], row["cache_rng_audits"], strict=True):
        if not all(
            (
                audit.get("production_cache_signature_unchanged", False),
                audit.get("production_rng_unchanged", False),
                audit.get("shadow_cache_discarded", False),
                audit.get("one_causal_forward_per_boundary", False),
                audit.get("correction_history_finite", False),
                audit.get("shadow_expert_execution", False),
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                not audit.get("forbidden_inputs_present", True),
                cost["attention_calls"] == expected["expected_attention_calls_per_boundary"],
                cost["attention_queries"] == expected["expected_attention_queries_per_boundary"],
                cost["router_calls"] == expected["expected_router_calls_per_boundary"],
                cost["expert_calls"] == expected["expected_expert_calls_per_boundary"],
            )
        ):
            return False
        corrected = spec.hidden_state_correction != "none" or spec.residual_correction != "none"
        if corrected != bool(audit.get("correction_history_source")):
            return False
    return True


def _router_sensitivity(references: list[dict[str, Any]]) -> dict[str, object]:
    baseline_key = SPECS[0].key
    output: dict[str, object] = {}
    for spec in SPECS[1:]:
        squared = 0.0
        baseline_energy = 0.0
        overlap = 0
        slots = 0
        for reference in references:
            row_index = int(reference["row_index"])
            candidate = load_file(str(_paths(row_index, spec.key)[1]))[
                "pseudo_router_logits"
            ].double()
            baseline = load_file(str(_paths(row_index, baseline_key)[1]))[
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
            "pseudo_top8_overlap_with_uncorrected": overlap / slots,
            "compared_router_slots": slots,
        }
    return {"reference": baseline_key, "comparisons": output}


def _anchor_strata(rows: list[dict[str, Any]]) -> dict[str, object]:
    totals: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: {anchor: [0.0, 0.0, 0.0, 0.0] for anchor in range(1, 9)}
    )
    for row in rows:
        tensors = load_file(str(_paths(int(row["row_index"]), str(row["policy"]))[1]))
        ids = tensors["natural_router_topk_ids"]
        weights = tensors["natural_router_topk_weights"].float()
        subsets = tensors["subsets"]
        boundaries = [int(value) for value in tensors["boundaries"].tolist()]
        for boundary_index, boundary in enumerate(boundaries):
            end = (
                boundaries[boundary_index + 1] if boundary_index + 1 < len(boundaries) else len(ids)
            )
            for token_index in range(boundary, end):
                anchor = token_index - boundary + 1
                for layer in range(ids.shape[1]):
                    allowed = torch.isin(ids[token_index, layer], subsets[boundary_index, layer])
                    value = totals[str(row["policy"])][anchor]
                    value[0] += float(allowed.sum())
                    value[1] += float(allowed.numel())
                    value[2] += float(weights[token_index, layer][allowed].double().sum())
                    value[3] += float(weights[token_index, layer].double().sum())
    return {
        policy: {
            str(anchor): {
                "mean_route_hit": value[0] / value[1],
                "mean_selected_mass": value[2] / value[3],
                "route_slots": int(value[1]),
            }
            for anchor, value in anchors.items()
        }
        for policy, anchors in totals.items()
    }


def aggregate() -> dict[str, object]:
    config, samples, _source = _protocol()
    references = samples["partitions"]["development"]
    rows: list[dict[str, Any]] = []
    for reference in references:
        for spec in SPECS:
            row = _valid(reference, spec)
            if row is None:
                raise RuntimeError(f"missing correction row: {reference['sample_id']}/{spec.key}")
            rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    policies = {
        policy: _aggregate_policy(policy_rows) for policy, policy_rows in sorted(grouped.items())
    }
    baseline_key = SPECS[0].key
    baseline = policies[baseline_key]
    comparisons: dict[str, object] = {}
    route_signals: list[str] = []
    for spec in SPECS[1:]:
        candidate = policies[spec.key]
        route_delta = float(candidate["mean_route_hit"]) - float(baseline["mean_route_hit"])
        mass_delta = float(candidate["mean_selected_mass"]) - float(baseline["mean_selected_mass"])
        signal = (
            route_delta
            >= float(
                config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"]
            )
            and mass_delta
            >= float(
                config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"]
            )
            and float(candidate["mean_probe_latency_seconds"])
            <= float(config["analysis"]["maximum_mean_probe_latency_seconds"])
        )
        if signal:
            route_signals.append(spec.key)
        comparisons[spec.key] = {
            "mean_route_hit_delta": route_delta,
            "mean_selected_mass_delta": mass_delta,
            "estimated_transfer_reduction_delta": float(candidate["estimated_transfer_reduction"])
            - float(baseline["estimated_transfer_reduction"]),
            "mean_probe_latency_seconds_delta": float(candidate["mean_probe_latency_seconds"])
            - float(baseline["mean_probe_latency_seconds"]),
            "paired_bootstrap": _paired_bootstrap(
                {str(row["sample_id"]): _per_sample(row) for row in grouped[spec.key]},
                {str(row["sample_id"]): _per_sample(row) for row in grouped[baseline_key]},
                draws=int(config["analysis"]["bootstrap_samples"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
            "route_signal": signal,
        }
    audit = {
        "all_pass": all(
            _audit_pass(row, next(spec for spec in SPECS if spec.key == row["policy"]), config)
            for row in rows
        ),
        "sample_policy_rows": len(rows),
        "one_causal_forward_per_boundary": True,
        "accuracy_or_correctness_used": False,
    }
    root = OUTPUT / "development"
    write_json_atomic(root / "aggregates.json", {"analysis_id": ANALYSIS_ID, "policies": policies})
    write_json_atomic(root / "comparisons.json", comparisons)
    write_json_atomic(root / "router_sensitivity.json", _router_sensitivity(references))
    write_json_atomic(root / "anchor_strata.json", _anchor_strata(rows))
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
            for policy, aggregate in policies.items()
        },
    )
    decision = {
        "analysis_id": ANALYSIS_ID,
        "decision": "DEVELOPMENT_ROUTE_SIGNAL" if route_signals else "STOP/PIVOT",
        "route_signal_variants": route_signals,
        "held_out_route_authorized": False,
        "task_accuracy_authorized": False,
    }
    write_json_atomic(root / "decision.json", decision)
    status = {
        "state": "complete",
        "sample_policy_rows": len(rows),
        "all_audits_pass": audit["all_pass"],
        "decision": decision["decision"],
    }
    write_json_atomic(root / "pipeline_status.json", status)
    return status


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
        "state": "complete",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "sample_manifest_sha256": SAMPLES_SHA256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json_atomic(OUTPUT / "artifact_manifest.json", manifest)
    return manifest


def _report() -> str:
    policies = _json(OUTPUT / "development" / "aggregates.json")["policies"]
    comparisons = _json(OUTPUT / "development" / "comparisons.json")
    decision = _json(OUTPUT / "development" / "decision.json")
    lines = [
        "# One-forward state correction v1",
        "",
        "All variants use one native causal eight-token pseudo forward and execute fresh "
        "MoE residuals under the previous realized B=32 subset after boundary zero.",
        "",
        "| Variant | Route hit | Selected mass | Transfer reduction | Probe s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, values in policies.items():
        lines.append(
            f"| {key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(values['estimated_transfer_reduction']):.6f} | "
            f"{float(values['mean_probe_latency_seconds']):.4f} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for key, values in comparisons.items():
        lines.append(
            f"- {key}: route-hit delta {float(values['mean_route_hit_delta']):+.6f}, "
            f"selected-mass delta {float(values['mean_selected_mass_delta']):+.6f}, "
            f"route signal `{values['route_signal']}`."
        )
    lines.extend(
        [
            "",
            f"Development-only decision: **{decision['decision']}**.",
            "",
            "Route values are teacher-forced on each policy's own hard-subset state. "
            "Probe/replay cost is measured and transfer is simulated. No held-out route, "
            "task accuracy, free generation, exact-token identity, or runtime speedup "
            "was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples, _source = _protocol()
    status = _json(OUTPUT / "development" / "pipeline_status.json")
    if status.get("state") != "complete" or not status.get("all_audits_pass"):
        raise ValueError("one-forward correction development aggregate is incomplete")
    row_paths = sorted(
        path
        for path in (OUTPUT / "development" / "samples").glob("*/*.json")
        if ".FAILED." not in path.name
    )
    rows = [_json(path) for path in row_paths]
    failures = sorted(str(path.relative_to(OUTPUT)) for path in OUTPUT.rglob("*.FAILED.json"))
    decision = _json(OUTPUT / "development" / "decision.json")
    write_json_atomic(OUTPUT / "resolved_config.json", config)
    write_json_atomic(OUTPUT / "resolved_sample_manifest.json", samples)
    write_json_atomic(OUTPUT / "resolved_environment.json", _software_hardware())
    write_json_atomic(
        OUTPUT / "resolved_execution_revision.json",
        {
            "git_head_at_report": _git_head(),
            "row_execution_git_heads": sorted({str(row["execution_git_head"]) for row in rows}),
            "pid": os.getpid(),
            "ppid": os.getppid(),
        },
    )
    write_json_atomic(
        OUTPUT / "measured_vs_simulated.json",
        {
            "measured": [
                "teacher_forced_policy_state_route_hit",
                "teacher_forced_policy_state_selected_mass",
                "native_one_forward_probe_latency_and_memory",
                "attention_router_expert_calls",
            ],
            "simulated": ["expert_transfer_bytes", "transfer_reduction"],
            "not_measured": [
                "task_accuracy",
                "free_generation",
                "exact_token_identity",
                "nll_or_perplexity",
                "closed_loop_runtime",
                "runtime_speedup",
            ],
        },
    )
    write_json_atomic(
        OUTPUT / "resume_audit.json",
        {
            "state": "complete",
            "sample_policy_rows_validated": len(rows),
            "atomic_json_tensor_pairs": True,
            "checksum_resume_pass": True,
            "failed_markers_preserved": failures,
        },
    )
    write_json_atomic(
        OUTPUT / "provenance.json",
        {
            "analysis_id": ANALYSIS_ID,
            "model": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "development_sample_ids": [
                row["sample_id"] for row in samples["partitions"]["development"]
            ],
            "learned_or_fitted_parameters": False,
            "default_vector_values_loaded": False,
            "task_accuracy_measured": False,
            "network_downloads": False,
        },
    )
    write_json_atomic(
        OUTPUT / "decision.json",
        {
            "analysis_id": ANALYSIS_ID,
            "decision": decision["decision"],
            "scope": "Qwen_GSM8K_H8_B32_one_forward_state_correction_development",
            "held_out_route_authorized": False,
            "task_accuracy_authorized": False,
            "runtime_speedup_claim": False,
        },
    )
    _write_text_atomic(OUTPUT / "report.md", _report())
    manifest = _manifest()
    final = {
        "schema_version": 1,
        "state": "complete",
        "stage": "report_v1",
        "analysis_id": ANALYSIS_ID,
        "analysis_config_sha256": CONFIG_SHA256,
        "decision": decision["decision"],
        "sample_policy_rows": len(rows),
        "artifact_count": manifest["artifact_count"],
        "task_accuracy_executed": False,
    }
    write_json_atomic(OUTPUT / "pipeline_status.json", final)
    validate()
    return final


def validate() -> dict[str, object]:
    _protocol()
    status = _json(OUTPUT / "pipeline_status.json")
    if status.get("state") != "complete" or status.get("stage") != "report_v1":
        raise ValueError("one-forward correction pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("one-forward correction artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"one-forward correction checksum changed: {path}")
    if _json(OUTPUT / "resume_audit.json")["sample_policy_rows_validated"] != 12:
        raise ValueError("one-forward correction row count changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "sample_policy_rows": status["sample_policy_rows"],
        "artifacts": manifest["artifact_count"],
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate", "finalize", "validate"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.command == "run":
        result = run(
            physical_gpu=args.gpu,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.command == "aggregate":
        result = aggregate()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
