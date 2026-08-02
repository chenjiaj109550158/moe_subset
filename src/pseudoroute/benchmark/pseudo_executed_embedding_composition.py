"""Run frozen pseudo-executed embedding-composition analysis waves."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from pseudoroute.benchmark.config import load_accuracy_suite_config
from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps
from pseudoroute.benchmark.pseudo_embedding_config import load_pseudo_embedding_config
from pseudoroute.benchmark.pseudo_embedding_residual_window import (
    PolicySpec,
    _save_tensors_atomic,
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


def run_simple(
    *,
    physical_gpu: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    config, samples = _protocol()
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
        for spec in SIMPLE_SPECS
        if _valid("development", reference, spec) is None
    ]
    if not missing:
        return {"state": "already_complete", "rows": len(references) * len(SIMPLE_SPECS)}
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
            row["analysis_wave"] = "simple_content_attention"
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
        "wave": "simple_content_attention",
        "shard_index": shard_index,
        "completed_now": completed,
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-simple",))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    result = run_simple(
        physical_gpu=args.gpu,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
