"""Run the frozen one-forward protected-anchor correction analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

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
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    CONFIG_SHA256 as SOURCE_CONFIG_SHA256,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    SAMPLES_SHA256 as SOURCE_SAMPLES_SHA256,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    SPECS as SOURCE_SPECS,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    _protocol as _source_protocol,
)
from pseudoroute.benchmark.pseudo_one_forward_state_correction import (
    _valid as _source_valid,
)
from pseudoroute.benchmark.qwen_pseudo import StateCorrection
from pseudoroute.benchmark.runner import _load_model, _software_hardware
from pseudoroute.benchmark.subset_trace import sha256_file, sha256_json, write_json_atomic
from pseudoroute.utils.determinism import seed_everything

ANALYSIS_ID = "pseudo_one_forward_protected_anchor_v2"
CONFIG = Path(f"configs/analysis/{ANALYSIS_ID}.yaml")
SAMPLES = Path(f"configs/analysis/{ANALYSIS_ID}_samples.json")
OUTPUT = Path(f"artifacts/{ANALYSIS_ID}")
CONFIG_SHA256 = "5bac51be6be2ab90aed563eb9208ce23afa772c9b109e9fa1bf66ee319bb81ee"
SAMPLES_SHA256 = "d37ff792acb6cb0d2fe0158040f7e3db8d3555783492c46ad4349362ada302ea"
LAYERS = 48
EXPERTS = 128
TOP_K = 8
HORIZON = 8

CorrectionTarget = Literal["router_input", "fresh_moe_residual"]


@dataclass(frozen=True)
class ProtectedSpec:
    key: str
    correction_target: CorrectionTarget
    coefficients: tuple[float, ...]
    max_relative_delta_norm: float | None
    selection: str

    @property
    def hidden_state_correction(self) -> StateCorrection:
        return "recent_linear_norm" if self.correction_target == "router_input" else "none"

    @property
    def residual_correction(self) -> StateCorrection:
        return "recent_linear_norm" if self.correction_target == "fresh_moe_residual" else "none"

    def policy(self) -> PolicySpec:
        return PolicySpec(
            self.key,
            "pseudo",
            residual="zero",
            content="recent_sequence_causal",
            selection=self.selection,
        )

    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


DAMPED = tuple(index / HORIZON for index in range(HORIZON))
UNDAMPED = tuple(float(index) for index in range(HORIZON))
SPECS = (
    ProtectedSpec(
        "hidden_velocity_anchor1_zero_horizon_damped",
        "router_input",
        DAMPED,
        None,
        "first_four_anchor_core_plus_history_fill",
    ),
    ProtectedSpec(
        "residual_velocity_anchor1_zero_undamped",
        "fresh_moe_residual",
        UNDAMPED,
        None,
        "first_four_anchor_core_plus_history_fill",
    ),
    ProtectedSpec(
        "residual_velocity_anchor1_zero_horizon_damped",
        "fresh_moe_residual",
        DAMPED,
        None,
        "first_four_anchor_core_plus_history_fill",
    ),
    ProtectedSpec(
        "residual_velocity_anchor1_zero_horizon_damped_cap25",
        "fresh_moe_residual",
        DAMPED,
        0.25,
        "first_four_anchor_core_plus_history_fill",
    ),
    ProtectedSpec(
        "residual_velocity_anchor1_zero_horizon_damped_cap25_anchor1_core",
        "fresh_moe_residual",
        DAMPED,
        0.25,
        "anchor_one_top8_plus_later_corrected_utility_fill",
    ),
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(CONFIG) != CONFIG_SHA256 or sha256_file(SAMPLES) != SAMPLES_SHA256:
        raise ValueError("protected-anchor protocol checksum changed")
    config = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    samples = _json(SAMPLES)
    if config["analysis_id"] != ANALYSIS_ID or samples["analysis_id"] != ANALYSIS_ID:
        raise ValueError("protected-anchor protocol identity changed")
    if (
        sha256_file(Path(config["source"]["config"])) != SOURCE_CONFIG_SHA256
        or config["source"]["config_sha256"] != SOURCE_CONFIG_SHA256
    ):
        raise ValueError("protected-anchor source config changed")
    if (
        sha256_file(Path(config["source"]["sample_manifest"])) != SOURCE_SAMPLES_SHA256
        or config["source"]["sample_manifest_sha256"] != SOURCE_SAMPLES_SHA256
    ):
        raise ValueError("protected-anchor source samples changed")
    source_manifest = Path(config["source"]["artifact_root"]) / "artifact_manifest.json"
    if sha256_file(source_manifest) != config["source"]["artifact_manifest_sha256"]:
        raise ValueError("protected-anchor source artifact changed")
    source_config, source_samples, composition = _source_protocol()
    if source_samples["partitions"]["development"] != samples["partitions"]["development"]:
        raise ValueError("protected-anchor development IDs changed")
    if source_config["analysis_id"] != config["source"]["analysis_id"]:
        raise ValueError("protected-anchor source analysis identity changed")
    configured = {row["key"]: row for row in config["variants"]}
    for spec in SPECS:
        row = configured.get(spec.key)
        coefficient_key = str(row["coefficients"]) if row is not None else ""
        expected = (
            config["analytic_constraints"]["horizon_damped_coefficients"]
            if coefficient_key == "horizon_damped_coefficients"
            else config["analytic_constraints"]["protected_undamped_coefficients"]
        )
        if row is None or [float(value) for value in expected] != list(spec.coefficients):
            raise ValueError(f"protected-anchor spec changed: {spec.key}")
        if row["max_relative_delta_norm"] != spec.max_relative_delta_norm:
            raise ValueError(f"protected-anchor cap changed: {spec.key}")
        if row["subset_selection"] != spec.selection:
            raise ValueError(f"protected-anchor selection changed: {spec.key}")
    return config, samples, composition


def _paths(row_index: int, policy: str) -> tuple[Path, Path]:
    root = OUTPUT / "development" / "samples" / f"{row_index:05d}" / policy
    return root.with_suffix(".json"), root.with_suffix(".safetensors")


def _valid(reference: dict[str, Any], spec: ProtectedSpec) -> dict[str, Any] | None:
    json_path, tensor_path = _paths(int(reference["row_index"]), spec.key)
    if not json_path.exists() and not tensor_path.exists():
        return None
    if not json_path.is_file() or not tensor_path.is_file():
        raise ValueError(f"partial protected-anchor row: {json_path}")
    row = _json(json_path)
    if (
        row.get("state") != "complete"
        or row.get("analysis_id") != ANALYSIS_ID
        or row.get("analysis_config_sha256") != CONFIG_SHA256
        or row.get("sample_manifest_sha256") != SAMPLES_SHA256
        or row.get("sample_id") != reference["sample_id"]
        or row.get("protected_spec_fingerprint") != spec.fingerprint()
        or row.get("correction_anchor_coefficients") != list(spec.coefficients)
        or row.get("correction_max_relative_delta_norm") != spec.max_relative_delta_norm
        or row.get("tensor_sha256") != sha256_file(tensor_path)
        or row.get("tensor_bytes") != tensor_path.stat().st_size
    ):
        raise ValueError(f"incompatible protected-anchor row: {json_path}")
    return row


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(*, physical_gpu: int, shard_index: int, shard_count: int) -> dict[str, object]:
    config, samples, composition = _protocol()
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid protected-anchor shard")
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
    suite = load_pseudo_embedding_config(composition["source_suite"]["config"])
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
        json_path, tensor_path = _paths(row_index, spec.key)
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
                spec.policy(),
                {},
                stage="development",
                route_token_cap=int(config["operating_point"]["route_token_cap"]),
                physical_gpu=physical_gpu,
                shadow_expert_execution=True,
                analysis_id=ANALYSIS_ID,
                analysis_config_sha256=CONFIG_SHA256,
                sample_manifest_sha256=SAMPLES_SHA256,
                hidden_state_correction=spec.hidden_state_correction,
                residual_correction=spec.residual_correction,
                correction_anchor_coefficients=spec.coefficients,
                correction_max_relative_delta_norm=spec.max_relative_delta_norm,
            )
            row["protected_spec"] = asdict(spec)
            row["protected_spec_fingerprint"] = spec.fingerprint()
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
    return {"state": "complete", "shard_index": shard_index, "completed_now": completed}


def _source_rows_checked(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for reference in references:
        row = _source_valid(reference, SOURCE_SPECS[0])
        if row is None:
            raise RuntimeError(f"missing source baseline: {reference['sample_id']}")
        rows.append(row)
    return rows


def _tensor_path(row: dict[str, Any]) -> Path:
    if row["analysis_id"] == ANALYSIS_ID:
        return _paths(int(row["row_index"]), str(row["policy"]))[1]
    root = Path("artifacts/pseudo_one_forward_state_correction_v1/development/samples")
    return root / f"{int(row['row_index']):05d}" / f"{row['policy']}.safetensors"


def _anchor_strata(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    totals: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: {anchor: [0.0, 0.0, 0.0, 0.0] for anchor in range(1, HORIZON + 1)}
    )
    for row in rows:
        tensors = load_file(str(_tensor_path(row)))
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


def _router_sensitivity(
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, object]:
    baseline = {str(row["sample_id"]): row for row in source_rows}
    output: dict[str, object] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["policy"])].append(row)
    for policy, rows in grouped.items():
        squared = 0.0
        baseline_energy = 0.0
        overlap = 0
        slots = 0
        anchor_one_exact = 0
        anchor_one_slots = 0
        for row in rows:
            candidate = load_file(str(_tensor_path(row)))["pseudo_router_logits"].double()
            reference = load_file(str(_tensor_path(baseline[str(row["sample_id"])])))[
                "pseudo_router_logits"
            ].double()
            candidate_centered = candidate - candidate.mean(dim=-1, keepdim=True)
            reference_centered = reference - reference.mean(dim=-1, keepdim=True)
            squared += float((candidate_centered - reference_centered).square().sum())
            baseline_energy += float(reference_centered.square().sum())
            candidate_ids = candidate.topk(TOP_K, dim=-1).indices
            reference_ids = reference.topk(TOP_K, dim=-1).indices
            overlap += int(
                (candidate_ids.unsqueeze(-1) == reference_ids.unsqueeze(-2)).any(dim=-1).sum()
            )
            slots += candidate_ids.numel()
            anchor_one_exact += int(torch.equal(candidate[:, :, 0], reference[:, :, 0]))
            anchor_one_slots += 1
        output[policy] = {
            "normalized_centered_logit_rms": (
                (squared / baseline_energy) ** 0.5 if baseline_energy else 0.0
            ),
            "pseudo_top8_overlap_with_uncorrected": overlap / slots,
            "anchor_one_logits_exact_sample_rows": anchor_one_exact,
            "anchor_one_logits_compared_sample_rows": anchor_one_slots,
        }
    return {"reference": SOURCE_SPECS[0].key, "comparisons": output}


def _audit_pass(row: dict[str, Any], spec: ProtectedSpec, config: dict[str, Any]) -> bool:
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
                audit.get("anchor_one_correction_protected", False),
                audit.get("shadow_expert_execution", False),
                audit.get("executed_ids_within_supplied_subset") in (True, None),
                not audit.get("forbidden_inputs_present", True),
                audit.get("anchor_correction_coefficients") == list(spec.coefficients),
                audit.get("correction_max_relative_delta_norm") == spec.max_relative_delta_norm,
                cost["attention_calls"] == expected["expected_attention_calls_per_boundary"],
                cost["attention_queries"] == expected["expected_attention_queries_per_boundary"],
                cost["router_calls"] == expected["expected_router_calls_per_boundary"],
                cost["expert_calls"] == expected["expected_expert_calls_per_boundary"],
            )
        ):
            return False
    return True


def aggregate() -> dict[str, object]:
    config, samples, _composition = _protocol()
    references = samples["partitions"]["development"]
    source_rows = _source_rows_checked(references)
    candidate_rows: list[dict[str, Any]] = []
    for reference in references:
        for spec in SPECS:
            row = _valid(reference, spec)
            if row is None:
                raise RuntimeError(f"missing protected row: {reference['sample_id']}/{spec.key}")
            candidate_rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped[SOURCE_SPECS[0].key] = source_rows
    for row in candidate_rows:
        grouped[str(row["policy"])].append(row)
    policies = {policy: _aggregate_policy(rows) for policy, rows in sorted(grouped.items())}
    anchors = _anchor_strata([*source_rows, *candidate_rows])
    baseline_key = SOURCE_SPECS[0].key
    baseline = policies[baseline_key]
    baseline_anchor = anchors[baseline_key]["1"]
    comparisons: dict[str, object] = {}
    signals: list[str] = []
    all_audits = all(
        _audit_pass(row, next(spec for spec in SPECS if spec.key == row["policy"]), config)
        for row in candidate_rows
    )
    for spec in SPECS:
        candidate = policies[spec.key]
        candidate_anchor = anchors[spec.key]["1"]
        route_delta = float(candidate["mean_route_hit"]) - float(baseline["mean_route_hit"])
        mass_delta = float(candidate["mean_selected_mass"]) - float(baseline["mean_selected_mass"])
        anchor_route_delta = float(candidate_anchor["mean_route_hit"]) - float(
            baseline_anchor["mean_route_hit"]
        )
        anchor_mass_delta = float(candidate_anchor["mean_selected_mass"]) - float(
            baseline_anchor["mean_selected_mass"]
        )
        signal = (
            route_delta
            >= float(
                config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"]
            )
            and mass_delta
            >= float(
                config["analysis"]["route_signal_minimum_absolute_improvement_over_uncorrected"]
            )
            and anchor_route_delta
            >= -float(config["analysis"]["maximum_anchor_one_absolute_regression"])
            and anchor_mass_delta
            >= -float(config["analysis"]["maximum_anchor_one_absolute_regression"])
            and float(candidate["mean_probe_latency_seconds"])
            <= float(config["analysis"]["maximum_mean_probe_latency_seconds"])
            and all_audits
        )
        if signal:
            signals.append(spec.key)
        comparisons[spec.key] = {
            "mean_route_hit_delta": route_delta,
            "mean_selected_mass_delta": mass_delta,
            "anchor_one_route_hit_delta": anchor_route_delta,
            "anchor_one_selected_mass_delta": anchor_mass_delta,
            "estimated_transfer_reduction_delta": float(candidate["estimated_transfer_reduction"])
            - float(baseline["estimated_transfer_reduction"]),
            "mean_probe_latency_seconds_delta": float(candidate["mean_probe_latency_seconds"])
            - float(baseline["mean_probe_latency_seconds"]),
            "paired_bootstrap": _paired_bootstrap(
                {str(row["sample_id"]): _per_sample(row) for row in grouped[spec.key]},
                {str(row["sample_id"]): _per_sample(row) for row in source_rows},
                draws=int(config["analysis"]["bootstrap_samples"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
            "route_signal": signal,
        }
    ranked = sorted(
        (spec.key for spec in SPECS),
        key=lambda key: (
            -float(policies[key]["mean_selected_mass"]),
            -float(policies[key]["mean_route_hit"]),
            float(policies[key]["mean_probe_latency_seconds"]),
            key,
        ),
    )
    root = OUTPUT / "development"
    write_json_atomic(root / "aggregates.json", {"analysis_id": ANALYSIS_ID, "policies": policies})
    write_json_atomic(root / "comparisons.json", comparisons)
    write_json_atomic(root / "anchor_strata.json", anchors)
    write_json_atomic(
        root / "router_sensitivity.json", _router_sensitivity(source_rows, candidate_rows)
    )
    write_json_atomic(
        root / "per_sample.json",
        {
            policy: {str(row["sample_id"]): _per_sample(row) for row in rows}
            for policy, rows in grouped.items()
        },
    )
    audit = {
        "all_pass": all_audits,
        "candidate_sample_policy_rows": len(candidate_rows),
        "source_reference_rows": len(source_rows),
        "one_causal_forward_per_boundary": True,
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
            for policy, aggregate in policies.items()
        },
    )
    decision = {
        "analysis_id": ANALYSIS_ID,
        "decision": "DEVELOPMENT_ROUTE_SIGNAL" if signals else "STOP/PIVOT",
        "ranked_candidate_variants": ranked,
        "route_signal_variants": signals,
        "selected_route_signal_variant": next((key for key in ranked if key in signals), None),
        "held_out_route_authorized": False,
        "task_accuracy_authorized": False,
    }
    write_json_atomic(root / "decision.json", decision)
    status = {
        "state": "complete",
        "candidate_sample_policy_rows": len(candidate_rows),
        "source_reference_rows": len(source_rows),
        "all_audits_pass": all_audits,
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
    anchors = _json(OUTPUT / "development" / "anchor_strata.json")
    sensitivity = _json(OUTPUT / "development" / "router_sensitivity.json")["comparisons"]
    decision = _json(OUTPUT / "development" / "decision.json")
    baseline_key = SOURCE_SPECS[0].key
    lines = [
        "# One-forward protected-anchor correction v2",
        "",
        "All candidates protect the known-token first anchor and use one native causal "
        "eight-token pseudo traversal per boundary.",
        "",
        "| Variant | Route hit | Selected mass | Anchor-1 hit | Anchor-1 mass | Probe s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in (baseline_key, *(spec.key for spec in SPECS)):
        values = policies[key]
        anchor = anchors[key]["1"]
        lines.append(
            f"| {key} | {float(values['mean_route_hit']):.6f} | "
            f"{float(values['mean_selected_mass']):.6f} | "
            f"{float(anchor['mean_route_hit']):.6f} | "
            f"{float(anchor['mean_selected_mass']):.6f} | "
            f"{float(values['mean_probe_latency_seconds']):.4f} |"
        )
    lines.extend(["", "## Comparisons to checksum-pinned uncorrected v1", ""])
    for spec in SPECS:
        values = comparisons[spec.key]
        router = sensitivity[spec.key]
        bootstrap = values["paired_bootstrap"]
        lines.append(
            f"- {spec.key}: hit/mass delta "
            f"{float(values['mean_route_hit_delta']):+.6f}/"
            f"{float(values['mean_selected_mass_delta']):+.6f}; anchor-1 delta "
            f"{float(values['anchor_one_route_hit_delta']):+.6f}/"
            f"{float(values['anchor_one_selected_mass_delta']):+.6f}; paired 95% "
            f"{bootstrap['route_hit_delta_percentile_95']} and "
            f"{bootstrap['selected_mass_delta_percentile_95']}; centered-logit RMS "
            f"{float(router['normalized_centered_logit_rms']):.6f}; signal "
            f"`{values['route_signal']}`."
        )
    lines.extend(
        [
            "",
            f"Development-only decision: **{decision['decision']}**. Selected route "
            f"signal: `{decision['selected_route_signal_variant']}`.",
            "",
            "Route values are teacher-forced on each policy's own hard-subset state. "
            "Probe/replay cost is measured and transfer is simulated. No held-out route, "
            "task accuracy, free generation, exact-token identity, or runtime speedup was "
            "measured.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize() -> dict[str, object]:
    config, samples, _composition = _protocol()
    status = _json(OUTPUT / "development" / "pipeline_status.json")
    if status.get("state") != "complete" or not status.get("all_audits_pass"):
        raise ValueError("protected-anchor development aggregate is incomplete")
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
            "source_reference_manifest_sha256": config["source"]["artifact_manifest_sha256"],
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
            "candidate_sample_policy_rows_validated": len(rows),
            "source_reference_rows_validated": 4,
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
            "source_reference_analysis": config["source"]["analysis_id"],
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
            "selected_route_signal_variant": decision["selected_route_signal_variant"],
            "scope": "Qwen_GSM8K_H8_B32_one_forward_protected_anchor_development",
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
        "candidate_sample_policy_rows": len(rows),
        "source_reference_rows": 4,
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
        raise ValueError("protected-anchor pipeline is incomplete")
    manifest = _json(OUTPUT / "artifact_manifest.json")
    actual = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file()
        and str(path.relative_to(OUTPUT)) not in {"artifact_manifest.json", "pipeline_status.json"}
    }
    recorded = {str(row["path"]) for row in manifest["artifacts"]}
    if actual != recorded:
        raise ValueError("protected-anchor artifact file set changed")
    for row in manifest["artifacts"]:
        path = OUTPUT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"protected-anchor checksum changed: {path}")
    resume = _json(OUTPUT / "resume_audit.json")
    if resume["candidate_sample_policy_rows_validated"] != 20:
        raise ValueError("protected-anchor candidate row count changed")
    if resume["source_reference_rows_validated"] != 4:
        raise ValueError("protected-anchor source row count changed")
    return {
        "state": "valid",
        "decision": status["decision"],
        "candidate_sample_policy_rows": status["candidate_sample_policy_rows"],
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
