"""Checksum-resumable natural replay traces sourced from audited v17 rows."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from pseudoroute.benchmark.config import AccuracyModelConfig, AccuracySuiteConfig
from pseudoroute.benchmark.prefetch import (
    SubsetExecutionContext,
    SubsetRouteRecord,
    build_prefetch_ops,
    load_default_vectors,
    physical_expert_bytes,
)
from pseudoroute.benchmark.runner import (
    _generate,
    _load_model,
    _model_example,
    _read_jsonl,
    _render,
    _software_hardware,
)
from pseudoroute.benchmark.subset_config import (
    SubsetModelConfig,
    SubsetOracleSuiteConfig,
    TraceSample,
)
from pseudoroute.benchmark.tasks import BenchmarkExample, load_examples
from pseudoroute.utils.determinism import seed_everything


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_model(accuracy: AccuracySuiteConfig, model: SubsetModelConfig) -> AccuracyModelConfig:
    matches = [candidate for candidate in accuracy.models if candidate.key == model.key]
    if len(matches) != 1:
        raise ValueError(f"missing source accuracy model {model.key}")
    return matches[0]


def _source_rows(root: Path, model: str, task: str) -> list[dict[str, Any]]:
    path = root / "models" / model / "results" / "vanilla" / task / "samples.jsonl"
    rows = [row for row in _read_jsonl(path) if row.get("state") == "complete"]
    return sorted(rows, key=lambda row: int(row["row_index"]))


def audit_source_accuracy(
    suite: SubsetOracleSuiteConfig,
    accuracy: AccuracySuiteConfig,
    output_root: Path,
) -> dict[str, object]:
    """Independently hash and row-audit immutable v17 inputs without modifying them."""
    source = Path(suite.source_accuracy.artifact_root)
    pipeline_path = source / "pipeline_status.json"
    summary_path = source / "summary.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if pipeline.get("state") != suite.source_accuracy.required_pipeline_state:
        raise ValueError("source v17 pipeline is not complete")
    if pipeline.get("stage") != suite.source_accuracy.required_pipeline_stage:
        raise ValueError("source v17 pipeline stage changed")
    if summary.get("suite_id") != suite.source_accuracy.suite_id:
        raise ValueError("source v17 summary suite changed")
    if summary.get("config_fingerprint") != suite.source_accuracy.config_fingerprint:
        raise ValueError("source v17 summary fingerprint changed")
    if accuracy.fingerprint() != suite.source_accuracy.config_fingerprint:
        raise ValueError("source v17 config fingerprint changed")
    if summary.get("model_vanilla_alignment") != {
        "gpt_oss_20b": True,
        "qwen3_30b_a3b": True,
    }:
        raise ValueError("source v17 vanilla alignment gate changed")

    artifacts: list[dict[str, object]] = []
    total_rows = 0
    for path in (Path(suite.source_accuracy.config), pipeline_path, summary_path):
        artifacts.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    for model in accuracy.models:
        validation = source / "models" / model.key / "validation.json"
        defaults_root = source / "models" / model.key / "default_vectors"
        defaults = load_default_vectors(defaults_root)
        for path in (
            validation,
            defaults_root / "manifest.json",
            defaults_root / "default_vectors.safetensors",
        ):
            artifacts.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        for dataset in accuracy.datasets:
            task_root = source / "models" / model.key / "results" / "vanilla" / dataset.key
            rows = _source_rows(source, model.key, dataset.key)
            indices = [int(row["row_index"]) for row in rows]
            sample_ids = [str(row["sample_id"]) for row in rows]
            if (
                len(rows) != dataset.expected_samples
                or indices != list(range(dataset.expected_samples))
                or len(set(sample_ids)) != dataset.expected_samples
            ):
                raise ValueError(f"source v17 rows invalid for {model.key}/{dataset.key}")
            total_rows += len(rows)
            for path in (task_root / "samples.jsonl", task_root / "DONE"):
                artifacts.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        artifacts.append(
            {
                "model": model.key,
                "default_vector_fingerprint": defaults.fingerprint,
                "kind": "validated_tensor_content_fingerprint",
            }
        )
    if total_rows != 5216:
        raise ValueError(f"source v17 expected 5,216 vanilla rows, got {total_rows}")
    audit: dict[str, object] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "subset_config_fingerprint": suite.fingerprint(),
        "source_suite_id": suite.source_accuracy.suite_id,
        "source_config_fingerprint": accuracy.fingerprint(),
        "source_pipeline_state": pipeline["state"],
        "source_pipeline_stage": pipeline["stage"],
        "source_vanilla_rows": total_rows,
        "source_oracle_identity_kind": "identity_materialized_layer_ahead_not_subset_accuracy",
        "artifacts": artifacts,
    }
    write_json_atomic(output_root / "source_v17_audit.json", audit)
    return audit


def _find_example(
    examples: tuple[BenchmarkExample, ...], reference: TraceSample
) -> BenchmarkExample:
    if reference.row_index >= len(examples):
        raise ValueError(f"trace row {reference.row_index} is outside the dataset")
    example = examples[reference.row_index]
    if example.sample_id != reference.sample_id:
        raise ValueError(
            f"frozen sample mismatch at row {reference.row_index}: "
            f"{example.sample_id} != {reference.sample_id}"
        )
    return example


def _capture_lm_head(module: nn.Module, target: list[Tensor]) -> Any:
    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        if not isinstance(output, Tensor):
            raise RuntimeError("language-model head returned a non-tensor")
        target.append(output[:, -1].detach().float().cpu())

    return module.register_forward_hook(hook)


def _records_by_layer(
    records: tuple[SubsetRouteRecord, ...], expected_layers: int
) -> dict[int, SubsetRouteRecord]:
    by_layer = {record.layer: record for record in records}
    if tuple(sorted(by_layer)) != tuple(range(expected_layers)) or len(records) != expected_layers:
        raise RuntimeError("natural trace did not capture every MoE layer exactly once")
    return by_layer


def _collect_teacher_trace(
    model: nn.Module,
    ops: Any,
    inputs: dict[str, Tensor],
    token_ids: list[int],
    *,
    chunk_tokens: int,
) -> tuple[dict[str, Tensor], list[Tensor]]:
    with torch.inference_mode():
        prefill = cast(Any, model)(**inputs, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    teacher_lm_logits = [cast(Tensor, prefill.logits[:, -1]).detach().float().cpu()]
    logits_by_layer: dict[int, list[Tensor]] = {layer: [] for layer in range(ops.num_layers)}
    ids_by_layer: dict[int, list[Tensor]] = {layer: [] for layer in range(ops.num_layers)}
    weights_by_layer: dict[int, list[Tensor]] = {layer: [] for layer in range(ops.num_layers)}
    device = inputs["input_ids"].device
    with SubsetExecutionContext(ops, "natural") as context, torch.inference_mode():
        for start in range(0, len(token_ids), chunk_tokens):
            chunk = torch.tensor(
                token_ids[start : start + chunk_tokens], dtype=torch.long, device=device
            )[None]
            output = cast(
                Any,
                model(
                    input_ids=chunk,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                ),
            )
            cache = output.past_key_values
            teacher_lm_logits.extend(tensor.detach().float().cpu() for tensor in output.logits[0])
            by_layer = _records_by_layer(context.drain(), ops.num_layers)
            for layer, record in by_layer.items():
                logits_by_layer[layer].append(record.natural.logits)
                ids_by_layer[layer].append(record.natural.ids)
                weights_by_layer[layer].append(record.natural.weights)
    tensors = {
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
        "router_logits": torch.stack(
            [torch.cat(logits_by_layer[layer], dim=0) for layer in range(ops.num_layers)],
            dim=1,
        ),
        "router_topk_ids": torch.stack(
            [torch.cat(ids_by_layer[layer], dim=0) for layer in range(ops.num_layers)],
            dim=1,
        ),
        "router_topk_weights": torch.stack(
            [torch.cat(weights_by_layer[layer], dim=0) for layer in range(ops.num_layers)],
            dim=1,
        ),
    }
    return tensors, teacher_lm_logits


def _autoregressive_parity(
    model: nn.Module,
    tokenizer: Any,
    ops: Any,
    inputs: dict[str, Tensor],
    example: BenchmarkExample,
    model_config: AccuracyModelConfig,
    accuracy: AccuracySuiteConfig,
    source_tokens: list[int],
    teacher: dict[str, Tensor],
    teacher_lm_logits: list[Tensor],
    max_tokens: int,
) -> dict[str, object]:
    count = min(max_tokens, len(source_tokens))
    if count < 2:
        raise ValueError("parity smoke requires at least two saved decode tokens")
    smoke_example = replace(example, max_new_tokens=count)
    captured_lm: list[Tensor] = []
    lm_head = cast(nn.Module, cast(Any, model).lm_head)
    handle = _capture_lm_head(lm_head, captured_lm)
    seed_everything(accuracy.decode.seed)
    try:
        with SubsetExecutionContext(ops, "natural") as context:
            generated, _ = _generate(
                model,
                tokenizer,
                inputs,
                smoke_example,
                model_config,
                do_sample=accuracy.do_sample_for(model_config, example.task),
            )
            records = context.drain()
    finally:
        handle.remove()
    if generated != source_tokens[: len(generated)] or len(generated) != count:
        raise RuntimeError(
            f"autoregressive v17 token parity failed: {generated} != {source_tokens[:count]}"
        )
    expected_record_count = ops.num_layers * count
    if len(records) != expected_record_count:
        raise RuntimeError(
            f"autoregressive route capture count {len(records)} != {expected_record_count}"
        )
    # The first call covers the prompt. Subsequent calls cover generated tokens 0..count-2.
    decode_records = records[ops.num_layers :]
    max_router_delta = 0.0
    route_ids_equal = True
    route_weights_delta = 0.0
    for position in range(count - 1):
        step = decode_records[position * ops.num_layers : (position + 1) * ops.num_layers]
        by_layer = _records_by_layer(tuple(step), ops.num_layers)
        for layer, record in by_layer.items():
            max_router_delta = max(
                max_router_delta,
                float(
                    (record.natural.logits[0] - teacher["router_logits"][position, layer])
                    .abs()
                    .max()
                ),
            )
            route_ids_equal = route_ids_equal and torch.equal(
                record.natural.ids[0], teacher["router_topk_ids"][position, layer]
            )
            route_weights_delta = max(
                route_weights_delta,
                float(
                    (record.natural.weights[0] - teacher["router_topk_weights"][position, layer])
                    .abs()
                    .max()
                ),
            )
    if len(captured_lm) != count:
        raise RuntimeError(f"autoregressive LM logit capture count {len(captured_lm)} != {count}")
    max_lm_delta = max(
        float((captured_lm[index] - teacher_lm_logits[index]).abs().max()) for index in range(count)
    )
    tolerance = 5e-3 if model_config.precision_tier == "mxfp4" else 1e-3
    if not route_ids_equal or max_router_delta > tolerance or route_weights_delta > tolerance:
        raise RuntimeError(
            "teacher-forced route parity failed: "
            f"ids={route_ids_equal}, logits={max_router_delta}, weights={route_weights_delta}"
        )
    if max_lm_delta > tolerance:
        raise RuntimeError(f"teacher-forced LM logit parity failed: {max_lm_delta}")
    return {
        "generated_tokens_equal": True,
        "route_ids_equal": route_ids_equal,
        "max_router_logit_delta": max_router_delta,
        "max_router_weight_delta": route_weights_delta,
        "max_lm_logit_delta": max_lm_delta,
        "tokens": count,
        "tolerance": tolerance,
    }


def _save_trace_shard(path: Path, tensors: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def _existing_shards(manifest_path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_fingerprint") != fingerprint:
        raise ValueError(f"trace manifest config mismatch: {manifest_path}")
    records = cast(list[dict[str, Any]], manifest.get("shards", []))
    for record in records:
        path = manifest_path.parent / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"trace shard missing/size mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"trace shard checksum mismatch: {path}")
    return records


def collect_model_traces(
    suite: SubsetOracleSuiteConfig,
    accuracy: AccuracySuiteConfig,
    model_subset: SubsetModelConfig,
    output_root: Path,
) -> dict[str, object]:
    """Load one checkpoint and atomically collect every frozen representative trace."""
    model_config = _source_model(accuracy, model_subset)
    model_root = output_root / "natural_traces" / model_subset.key
    manifest_path = model_root / "manifest.json"
    existing = _existing_shards(manifest_path, suite.fingerprint())
    completed = {(str(row["task"]), str(row["sample_id"])) for row in existing}
    expected = sum(len(rows) for rows in suite.trace.sample_rows.values())
    if manifest_path.is_file():
        finished_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            finished_manifest.get("state") == "complete"
            and len(existing) == expected
            and set(finished_manifest.get("parity", {})) == set(suite.trace.sample_rows)
        ):
            return validate_trace_manifest(suite, model_subset, output_root)
    parity: dict[str, object] = {}
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parity = cast(dict[str, object], previous_manifest.get("parity", {}))
        if not set(parity).issubset(set(suite.trace.sample_rows)):
            raise ValueError(f"trace parity tasks are invalid: {manifest_path}")
    seed_everything(accuracy.decode.seed)
    torch.cuda.set_device(torch.device(model_config.device))
    model, tokenizer = _load_model(model_config, accuracy)
    ops = build_prefetch_ops(model, model_config.architecture)
    if (
        ops.num_layers != model_subset.routed_layers
        or ops.num_experts != model_subset.routed_experts_per_layer
        or ops.top_k != model_subset.native_top_k
    ):
        raise ValueError(f"runtime model facts changed for {model_subset.key}")
    expert_bytes = {
        str(layer): physical_expert_bytes(ops, layer) for layer in range(ops.num_layers)
    }
    started = time.time()
    shards = list(existing)
    source_root = Path(suite.source_accuracy.artifact_root)
    for dataset in accuracy.datasets:
        examples = load_examples(dataset, cache_dir=accuracy.dataset_cache_dir)
        source_rows = _source_rows(source_root, model_subset.key, dataset.key)
        by_id = {str(row["sample_id"]): row for row in source_rows}
        for reference in suite.trace.sample_rows[dataset.key]:
            key = (dataset.key, reference.sample_id)
            if key in completed:
                continue
            example = _model_example(_find_example(examples, reference), model_config)
            inputs, rendered = _render(tokenizer, model_config, accuracy, example)
            source = by_id.get(reference.sample_id)
            if source is None or int(source["row_index"]) != reference.row_index:
                raise ValueError(f"missing frozen v17 row {model_subset.key}/{key}")
            if rendered != source["rendered_prompt"]:
                raise ValueError(f"rendered prompt changed for {model_subset.key}/{key}")
            rendered_sha = hashlib.sha256(rendered.encode()).hexdigest()
            if rendered_sha != source["rendered_prompt_sha256"]:
                raise ValueError(f"rendered prompt checksum changed for {model_subset.key}/{key}")
            source_tokens = [int(value) for value in source["generated_token_ids"]]
            token_ids = source_tokens[: suite.trace.max_decode_tokens_per_sample]
            if not token_ids:
                raise ValueError(f"source v17 row has no decode token: {model_subset.key}/{key}")
            tensors, teacher_lm = _collect_teacher_trace(
                model,
                ops,
                inputs,
                token_ids,
                chunk_tokens=suite.trace.replay_chunk_tokens,
            )
            if reference.sample_id == suite.trace.autoregressive_parity_smoke_rows[dataset.key]:
                parity[dataset.key] = _autoregressive_parity(
                    model,
                    tokenizer,
                    ops,
                    inputs,
                    example,
                    model_config,
                    accuracy,
                    source_tokens,
                    tensors,
                    teacher_lm,
                    suite.trace.autoregressive_parity_max_tokens,
                )
            relative = Path("shards") / dataset.key / f"{reference.row_index:05d}.safetensors"
            path = model_root / relative
            _save_trace_shard(path, tensors)
            record: dict[str, object] = {
                "task": dataset.key,
                "row_index": reference.row_index,
                "sample_id": reference.sample_id,
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tokens": len(token_ids),
                "source_generated_tokens": len(source_tokens),
                "source_token_ids_sha256": sha256_json(source_tokens),
                "traced_token_ids_sha256": sha256_json(token_ids),
                "rendered_prompt_sha256": rendered_sha,
                "source_row_sha256": sha256_json(source),
                "information_regime": suite.trace.information_regime,
            }
            shards.append(record)
            manifest = {
                "schema_version": 1,
                "state": "running",
                "suite_id": suite.suite_id,
                "config_fingerprint": suite.fingerprint(),
                "model": model_subset.model_dump(mode="json"),
                "source_model": model_config.model_dump(mode="json"),
                "expert_bytes_by_layer": expert_bytes,
                "parity": parity,
                "shards": shards,
            }
            write_json_atomic(manifest_path, manifest)
            completed.add(key)
            print(
                json.dumps(
                    {
                        "stage": "trace",
                        "model": model_subset.key,
                        "task": dataset.key,
                        "sample_id": reference.sample_id,
                        "tokens": len(token_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(shards) != expected or set(parity) != set(suite.trace.sample_rows):
        raise RuntimeError(
            f"trace incomplete for {model_subset.key}: shards={len(shards)}/{expected}, "
            f"parity={sorted(parity)}"
        )
    manifest = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "model": model_subset.model_dump(mode="json"),
        "source_model": model_config.model_dump(mode="json"),
        "expert_bytes_by_layer": expert_bytes,
        "parity": parity,
        "shards": shards,
        "elapsed_seconds_current_process": time.time() - started,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "physical_gpu": model_subset.physical_gpu,
        "environment": _software_hardware(),
    }
    write_json_atomic(manifest_path, manifest)
    (model_root / "DONE").write_text("complete\n", encoding="utf-8")
    return manifest


def validate_trace_manifest(
    suite: SubsetOracleSuiteConfig, model: SubsetModelConfig, output_root: Path
) -> dict[str, Any]:
    model_root = output_root / "natural_traces" / model.key
    path = model_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise ValueError(f"trace is not complete for {model.key}")
    if manifest.get("config_fingerprint") != suite.fingerprint():
        raise ValueError(f"trace config mismatch for {model.key}")
    expected = sum(len(rows) for rows in suite.trace.sample_rows.values())
    shards = cast(list[dict[str, Any]], manifest.get("shards", []))
    if len(shards) != expected:
        raise ValueError(f"trace shard count mismatch for {model.key}")
    for record in shards:
        shard = model_root / str(record["path"])
        if shard.stat().st_size != int(record["bytes"]) or sha256_file(shard) != record["sha256"]:
            raise ValueError(f"trace checksum mismatch: {shard}")
        tensors = load_file(str(shard))
        tokens = int(record["tokens"])
        expected_shapes = {
            "token_ids": (tokens,),
            "router_logits": (tokens, model.routed_layers, model.routed_experts_per_layer),
            "router_topk_ids": (tokens, model.routed_layers, model.native_top_k),
            "router_topk_weights": (tokens, model.routed_layers, model.native_top_k),
        }
        if {key: tuple(value.shape) for key, value in tensors.items()} != expected_shapes:
            raise ValueError(f"trace tensor shape mismatch: {shard}")
        if sha256_json(tensors["token_ids"].tolist()) != record["traced_token_ids_sha256"]:
            raise ValueError(f"trace token checksum mismatch: {shard}")
    return cast(dict[str, Any], manifest)
