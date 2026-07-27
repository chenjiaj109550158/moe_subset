"""Resumable trained-suite runner, envelopes, decision gate, and saved-table plot."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import traceback
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import torch

from pseudoroute.trained.config import TrainedModelConfig, TrainedSuiteConfig
from pseudoroute.trained.oracle import run_model_oracle
from pseudoroute.trained.task import collect_model_traces


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_envelope(root: Path, *, state: str, error: dict[str, object] | None = None) -> None:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"task_envelope.json", "DONE"}:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    envelope = {
        "schema_version": 1,
        "state": state,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
        "error": error,
    }
    (root / "task_envelope.json").write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolved_model_config_matches(
    root: Path, suite: TrainedSuiteConfig, model: TrainedModelConfig
) -> bool:
    path = root / "resolved_config.json"
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    expected = {
        "suite": suite.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
    }
    return bool(actual == expected)


def _run_model(suite: TrainedSuiteConfig, model: TrainedModelConfig, output_root: Path) -> None:
    root = output_root / "models" / model.key
    if (root / "DONE").exists() and _resolved_model_config_matches(root, suite, model):
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "DONE").unlink(missing_ok=True)
    try:
        collected = (
            _resolved_model_config_matches(root, suite, model)
            and (root / "validation.json").exists()
            and (root / "closed_loop.csv").exists()
        )
        if not collected:
            with (
                (root / "stdout.log").open("a", encoding="utf-8") as stdout,
                (root / "stderr.log").open("a", encoding="utf-8") as stderr,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                collect_model_traces(suite, model, output_root / "models")
        run_model_oracle(suite, model, root)
        (root / "FAILED.json").unlink(missing_ok=True)
        (root / "DONE").write_text("complete\n", encoding="utf-8")
        _write_envelope(root, state="complete")
    except BaseException as error:
        failure: dict[str, object] = {
            "model": model.key,
            "model_id": model.model_id,
            "revision": model.revision,
            "tier": model.tier,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        (root / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_envelope(root, state="failed", error=failure)
        if model.tier == "floating_point":
            raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def model_num_experts(root: Path) -> dict[int, int]:
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    return {int(layer): int(count) for layer, count in validation["num_experts_by_layer"].items()}


def _operating_points(model: TrainedModelConfig, root: Path) -> list[dict[str, object]]:
    rows = _read_csv(root / "oracle_windows.csv")
    grouped: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], int(row["horizon"]), int(row["budget"]), row["method"])].append(row)
    points = []
    methods = {key[:3] for key in grouped}
    for domain, horizon, budget in sorted(methods):
        oracle = grouped.get((domain, horizon, budget, "selected_routing_mass"), [])
        demand = grouped.get((domain, horizon, budget, "on_demand"), [])
        if not oracle or not demand:
            continue
        baseline_values = []
        for baseline in ("previous_route", "static_frequency"):
            values = grouped.get((domain, horizon, budget, baseline), [])
            if values:
                baseline_values.append(
                    mean(float(value["selected_mass_coverage"]) for value in values)
                )
        hit_values = [float(value["route_hit_rate"]) for value in oracle]
        selected_values = [float(value["selected_mass_coverage"]) for value in oracle]
        oracle_bytes = mean(float(value["estimated_h2d_bytes_per_token"]) for value in oracle)
        demand_bytes = mean(float(value["estimated_h2d_bytes_per_token"]) for value in demand)
        points.append(
            {
                "model": model.key,
                "domain": domain,
                "horizon": horizon,
                "budget": budget,
                "mean_hit_rate": mean(hit_values),
                "p05_hit_rate": sorted(hit_values)[max(0, math.ceil(0.05 * len(hit_values)) - 1)],
                "worst_hit_rate": min(hit_values),
                "mean_selected_mass": mean(selected_values),
                "transfer_reduction": 1 - oracle_bytes / demand_bytes if demand_bytes else 0.0,
                "baseline_improvement": mean(selected_values) - max(baseline_values or [0.0]),
            }
        )
    return points


def _closed_quality(root: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(root / "closed_loop.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    return {
        policy: {
            "exact_token_agreement": mean(float(row["exact_token_agreement"]) for row in values),
            "relative_perplexity_increase": mean(
                float(row["relative_perplexity_increase"]) for row in values
            ),
            "fallback_frequency": mean(float(row["fallback_frequency"]) for row in values),
            "bytes_per_token": mean(float(row["estimated_h2d_bytes_per_token"]) for row in values),
        }
        for policy, values in grouped.items()
    }


def _closed_quality_by_domain(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    rows = _read_csv(root / "closed_loop.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["policy"])].append(row)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (domain, policy), values in grouped.items():
        result[domain][policy] = {
            "exact_token_agreement": mean(float(row["exact_token_agreement"]) for row in values),
            "relative_perplexity_increase": mean(
                float(row["relative_perplexity_increase"]) for row in values
            ),
            "fallback_frequency": mean(float(row["fallback_frequency"]) for row in values),
            "bytes_per_token": mean(float(row["estimated_h2d_bytes_per_token"]) for row in values),
        }
    return dict(result)


def apply_decision_gate(suite: TrainedSuiteConfig, output_root: Path) -> dict[str, object]:
    thresholds = suite.thresholds
    model_decisions = []
    all_points = []
    for model in suite.models:
        root = output_root / "models" / model.key
        if model.tier == "mxfp4":
            if not (root / "DONE").exists():
                failure = json.loads((root / "FAILED.json").read_text())
                model_decisions.append(
                    {
                        "model": model.key,
                        "tier": model.tier,
                        "decision": "STOP/PIVOT",
                        "reason": "native MXFP4 fused route capture/runtime incompatibility",
                        "failure": failure["message"],
                    }
                )
                continue
        if not (root / "DONE").exists():
            raise RuntimeError(f"primary model did not complete: {model.key}")
        points = _operating_points(model, root)
        all_points.extend(points)
        quality = _closed_quality(root)
        quality_by_domain = _closed_quality_by_domain(root)
        all_expert_count = min(model_num_experts(root).values())
        qualifying = [
            point
            for point in points
            if float(str(point["mean_hit_rate"])) >= thresholds.minimum_oracle_hit_rate
            and float(str(point["mean_selected_mass"])) >= thresholds.minimum_oracle_selected_mass
            and float(str(point["p05_hit_rate"])) >= thresholds.minimum_p05_hit_rate
            and float(str(point["worst_hit_rate"])) >= thresholds.minimum_p05_hit_rate
            and float(str(point["transfer_reduction"])) >= thresholds.minimum_transfer_reduction
            and float(str(point["baseline_improvement"])) >= thresholds.minimum_baseline_improvement
            and int(str(point["budget"])) < all_expert_count
        ]
        domains = {str(point["domain"]) for point in qualifying}
        hard_ok = all(
            policies["hard_oracle_commitment"]["relative_perplexity_increase"]
            <= thresholds.maximum_relative_perplexity_increase
            for policies in quality_by_domain.values()
        )
        fallback_ok = all(
            policies["lossless_oracle_residency"]["fallback_frequency"]
            <= thresholds.maximum_lossless_fallback_rate
            for policies in quality_by_domain.values()
        )
        if domains == {"wikitext", "gsm8k"} and hard_ok:
            decision = "GO"
            reason = "both domains meet open-loop, tail, baseline, transfer, and hard-quality gates"
        elif qualifying and (hard_ok or fallback_ok):
            decision = "NARROW"
            reason = (
                "feasibility is restricted by domain/layer/tail behavior or "
                "requires lossless fallback"
            )
        else:
            decision = "STOP/PIVOT"
            if qualifying:
                reason = (
                    "open-loop points pass, but hard-commitment quality and lossless "
                    "fallback fail the predeclared gates"
                )
            else:
                reason = (
                    "no below-all-expert operating point jointly satisfies coverage, "
                    "P05/worst tail, baseline, and transfer gates"
                )
        model_decisions.append(
            {
                "model": model.key,
                "tier": model.tier,
                "decision": decision,
                "reason": reason,
                "qualifying_operating_points": qualifying,
                "closed_loop_quality": quality,
                "closed_loop_quality_by_domain": quality_by_domain,
                "gate_evidence": {
                    "hard_commitment_quality_pass": hard_ok,
                    "lossless_fallback_pass": fallback_ok,
                    "worst_hit_uses_predeclared_p05_floor": thresholds.minimum_p05_hit_rate,
                    "requires_budget_below_all_routed_experts": True,
                },
            }
        )
    primary = [entry for entry in model_decisions if entry["tier"] == "floating_point"]
    if all(entry["decision"] == "GO" for entry in primary):
        overall = "GO"
    elif any(entry["decision"] in {"GO", "NARROW"} for entry in primary):
        overall = "NARROW"
    else:
        overall = "STOP/PIVOT"
    result = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "thresholds_predeclared_in": f"versioned suite {suite.suite_id}",
        "model_decisions": model_decisions,
        "overall_decision": overall,
        "scope": {
            "checkpoints": [model.model_id + "@" + model.revision for model in suite.models],
            "datasets": [dataset.dataset_id + "@" + dataset.revision for dataset in suite.datasets],
            "hardware": "2x NVIDIA A100-SXM4-80GB",
            "precision": "checkpoint dtype; gpt-oss separate native MXFP4 tier",
            "decoding": (
                f"batch-1 deterministic greedy, {suite.closed_loop.max_new_tokens} new tokens"
            ),
            "timing": "transfer/stall simulated, not measured runtime",
        },
    }
    (output_root / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "operating_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_points[0]))
        writer.writeheader()
        writer.writerows(all_points)
    _write_plot_from_table(output_root / "operating_points.csv", output_root / "oracle_gate.svg")
    return result


def _write_plot_from_table(table: Path, destination: Path) -> None:
    rows = _read_csv(table)
    width, height = 900, 480
    colors = {"olmoe": "#2563eb", "qwen": "#059669", "mixtral": "#dc2626", "deepseek": "#7c3aed"}
    points = []
    for row in rows:
        if row["horizon"] != "4":
            continue
        x = 70 + float(row["transfer_reduction"]) * 760
        y = 420 - float(row["mean_selected_mass"]) * 350
        color = colors.get(row["model"], "#111827")
        title = f"{row['model']} {row['domain']} B={row['budget']}"
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
            f"<title>{title}</title></circle>"
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" '
        'font-size="18">Saved-table oracle coverage vs transfer reduction '
        "(horizon 4)</text>"
        '<line x1="70" y1="420" x2="830" y2="420" stroke="#111"/>'
        '<line x1="70" y1="70" x2="70" y2="420" stroke="#111"/>'
        '<text x="450" y="460" text-anchor="middle" font-family="sans-serif">'
        "estimated transfer reduction</text>"
        '<text x="18" y="245" transform="rotate(-90 18 245)" '
        'text-anchor="middle" font-family="sans-serif">selected mass coverage</text>'
        + "".join(points)
        + "</svg>"
    )
    destination.write_text(svg, encoding="utf-8")


def _environment_record() -> dict[str, object]:
    import psutil  # type: ignore[import-untyped]

    packages = (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "datasets",
        "huggingface-hub",
        "accelerate",
        "kernels",
        "kernels-data",
        "numpy",
        "pydantic",
        "PyYAML",
    )
    gpu_records = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_records.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    driver_path = Path("/proc/driver/nvidia/version")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "torch_cuda_runtime": torch.version.cuda,
        "nvidia_driver": (
            driver_path.read_text(encoding="utf-8").splitlines()[0]
            if driver_path.exists()
            else None
        ),
        "gpus": gpu_records,
        "cpu_ram_bytes": int(psutil.virtual_memory().total),
        "environment": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_DATASETS_CACHE")
        },
    }


def run_trained_suite(
    suite: TrainedSuiteConfig, output_root: Path, *, dry_run: bool = False
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(suite.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        return {
            "suite_id": suite.suite_id,
            "models": [model.key for model in suite.models],
            "datasets": [dataset.key for dataset in suite.datasets],
            "resumable": True,
        }
    (output_root / "environment.json").write_text(
        json.dumps(_environment_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for model in suite.models:
        _run_model(suite, model, output_root)
    result = apply_decision_gate(suite, output_root)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    _write_envelope(output_root, state="complete")
    return result
