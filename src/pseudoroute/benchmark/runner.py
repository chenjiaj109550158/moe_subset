"""Resumable trained-model accuracy runner and paper comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.benchmark.config import (
    AccuracyModelConfig,
    AccuracySuiteConfig,
    load_accuracy_suite_config,
)
from pseudoroute.benchmark.prefetch import (
    DefaultVectorArtifact,
    PrefetchStats,
    build_prefetch_ops,
    calibrate_default_vectors,
    load_default_vectors,
    policy_context,
    save_default_vectors,
)
from pseudoroute.benchmark.scoring import score_response, wilson_interval
from pseudoroute.benchmark.tasks import BenchmarkExample, load_examples
from pseudoroute.utils.determinism import seed_everything


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(cast(dict[str, Any], json.loads(line)))
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _merge_task_shards(task_root: Path, shard_count: int, expected: int) -> bool:
    shards_root = task_root / "shards"
    rows: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        stem = f"{shard_index:05d}-of-{shard_count:05d}"
        if not (shards_root / f"{stem}.DONE").is_file():
            return False
        shard_rows = [
            row
            for row in _read_jsonl(shards_root / f"{stem}.jsonl")
            if row.get("state") == "complete"
        ]
        shard_expected = len(range(shard_index, expected, shard_count))
        if len(shard_rows) != shard_expected:
            return False
        rows.extend(shard_rows)
    row_indices = [int(row["row_index"]) for row in rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if (
        len(rows) != expected
        or len(set(row_indices)) != expected
        or set(row_indices) != set(range(expected))
        or len(set(sample_ids)) != expected
    ):
        raise RuntimeError(f"invalid task shards in {task_root}")
    rows.sort(key=lambda row: int(row["row_index"]))
    _write_jsonl_atomic(task_root / "samples.jsonl", rows)
    (task_root / "DONE").write_text("complete\n", encoding="utf-8")
    return True


def _software_hardware() -> dict[str, object]:
    import datasets  # type: ignore[import-untyped]
    import safetensors
    import transformers

    try:
        gpu = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "safetensors": safetensors.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cast(Any, torch.backends.cudnn).version(),
        "gpus": gpu,
    }


def _load_model(config: AccuracyModelConfig, suite: AccuracySuiteConfig) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=suite.cache_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=suite.cache_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype="auto",
        device_map={"": config.device},
    )
    module = cast(nn.Module, model)
    module.eval()
    return module, tokenizer


def _render(
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    suite: AccuracySuiteConfig,
    example: BenchmarkExample,
) -> tuple[dict[str, Tensor], str]:
    messages: list[dict[str, str]] = [{"role": "user", "content": example.user_prompt}]
    kwargs: dict[str, object] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "current_date": suite.decode.chat_template_current_date,
    }
    if model_config.reasoning_effort != "none":
        kwargs["reasoning_effort"] = model_config.reasoning_effort
    if example.assistant_prefix is None:
        kwargs["add_generation_prompt"] = True
    else:
        messages.append({"role": "assistant", "content": example.assistant_prefix})
        kwargs["continue_final_message"] = True
    encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if not hasattr(encoded, "items"):
        raise RuntimeError("chat template did not return a mapping")
    inputs = {key: value.to(model_config.device) for key, value in encoded.items()}
    rendered = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return inputs, rendered


def _encode_saved_rendered_prompt(
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    rendered: str,
) -> dict[str, Tensor]:
    """Re-encode immutable rendered prompt bytes saved by the source suite."""
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if not hasattr(encoded, "items"):
        raise RuntimeError("saved rendered prompt encoding did not return a mapping")
    inputs = {key: value.to(model_config.device) for key, value in encoded.items()}
    roundtrip = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    if roundtrip != rendered:
        raise RuntimeError("saved rendered prompt failed exact tokenizer round-trip")
    return inputs


def _model_example(
    example: BenchmarkExample, model_config: AccuracyModelConfig
) -> BenchmarkExample:
    if example.task in {"humaneval", "mbpp_plus"} and not model_config.code_assistant_prefill:
        return replace(example, assistant_prefix=None)
    return example


def _calibration_batches(
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    suite: AccuracySuiteConfig,
) -> tuple[dict[str, Tensor], ...]:
    from datasets import load_dataset

    config = suite.calibration
    dataset = load_dataset(
        config.dataset_id,
        config.config,
        split=config.split,
        revision=config.revision,
        cache_dir=suite.dataset_cache_dir,
    )
    batches = []
    for row_id in config.row_ids:
        text = str(dataset[row_id]["text"]).strip()
        if not text:
            raise ValueError(f"calibration row {row_id} is empty")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_tokens_per_row,
        )
        batches.append({key: value.to(model_config.device) for key, value in encoded.items()})
    return tuple(batches)


def _ensure_defaults(
    root: Path,
    model: nn.Module,
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    suite: AccuracySuiteConfig,
) -> DefaultVectorArtifact:
    target = root / "default_vectors"
    if (target / "manifest.json").exists():
        return load_default_vectors(target)
    ops = build_prefetch_ops(model, model_config.architecture)
    batches = _calibration_batches(tokenizer, model_config, suite)
    started = time.time()
    defaults = calibrate_default_vectors(ops, batches)
    save_default_vectors(
        target,
        defaults,
        {
            "model_id": model_config.model_id,
            "model_revision": model_config.revision,
            "architecture": model_config.architecture,
            "precision_tier": model_config.precision_tier,
            "dataset_id": suite.calibration.dataset_id,
            "dataset_revision": suite.calibration.revision,
            "dataset_split": suite.calibration.split,
            "row_ids": list(suite.calibration.row_ids),
            "max_tokens_per_row": suite.calibration.max_tokens_per_row,
            "unobserved_expert_value": suite.calibration.unobserved_expert_value,
            "elapsed_seconds": time.time() - started,
        },
    )
    return defaults


def _forward_logits(model: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    with torch.inference_mode():
        output = cast(Any, model)(**inputs, use_cache=False, return_dict=True)
    return cast(Tensor, output.logits)


def validate_policy_path(
    root: Path,
    model: nn.Module,
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    suite: AccuracySuiteConfig,
    defaults: DefaultVectorArtifact,
) -> dict[str, object]:
    path = root / "validation.json"
    if path.exists():
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    example = load_examples(suite.datasets[-1], cache_dir=suite.dataset_cache_dir)[0]
    inputs, rendered = _render(tokenizer, model_config, suite, example)
    ops = build_prefetch_ops(model, model_config.architecture)
    torch.cuda.reset_peak_memory_stats(torch.device(model_config.device))
    native = _forward_logits(model, inputs)
    with policy_context(ops, "vanilla") as natural_context:
        explicit = _forward_logits(model, inputs)
    with policy_context(ops, "oracle_pf") as oracle_context:
        oracle = _forward_logits(model, inputs)
    with policy_context(ops, "router_pf", defaults) as router_context:
        router = _forward_logits(model, inputs)
    native_explicit_delta = float((native - explicit).abs().max())
    native_oracle_delta = float((native - oracle).abs().max())
    tolerance = 5e-3 if model_config.precision_tier == "mxfp4" else 1e-3
    validation = {
        "model_id": model_config.model_id,
        "revision": model_config.revision,
        "architecture": model_config.architecture,
        "precision_tier": model_config.precision_tier,
        "layers": ops.num_layers,
        "experts": ops.num_experts,
        "top_k": ops.top_k,
        "hidden_size": ops.hidden_size,
        "rendered_prompt_sha256": _sha256_text(rendered),
        "native_explicit_max_abs_logit_delta": native_explicit_delta,
        "native_oracle_max_abs_logit_delta": native_oracle_delta,
        "native_explicit_argmax_equal": bool(
            torch.equal(native.argmax(dim=-1), explicit.argmax(dim=-1))
        ),
        "native_oracle_argmax_equal": bool(
            torch.equal(native.argmax(dim=-1), oracle.argmax(dim=-1))
        ),
        "router_pf_finite": bool(torch.isfinite(router).all()),
        "explicit_natural_stats": natural_context.stats.as_dict(),
        "oracle_stats": oracle_context.stats.as_dict(),
        "router_pf_stats": router_context.stats.as_dict(),
        "default_vector_fingerprint": defaults.fingerprint,
        "peak_cuda_allocated_bytes": int(
            torch.cuda.max_memory_allocated(torch.device(model_config.device))
        ),
        "parity_tolerance": tolerance,
        "passed": (
            native_explicit_delta <= tolerance
            and native_oracle_delta <= tolerance
            and bool(torch.equal(native.argmax(dim=-1), explicit.argmax(dim=-1)))
            and bool(torch.equal(native.argmax(dim=-1), oracle.argmax(dim=-1)))
            and bool(torch.isfinite(router).all())
        ),
    }
    path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError(f"{model_config.key}: policy-path validation failed")
    return validation


def _stat_delta(before: PrefetchStats, after: PrefetchStats) -> dict[str, int | float]:
    delta = PrefetchStats(
        compared_tokens=after.compared_tokens - before.compared_tokens,
        exact_topk_tokens=after.exact_topk_tokens - before.exact_topk_tokens,
        selected_slots=after.selected_slots - before.selected_slots,
        matching_selected_slots=after.matching_selected_slots - before.matching_selected_slots,
        calls=after.calls - before.calls,
    )
    return delta.as_dict()


def _copy_stats(stats: PrefetchStats) -> PrefetchStats:
    return PrefetchStats(**asdict(stats))


def _generate(
    model: nn.Module,
    tokenizer: Any,
    inputs: dict[str, Tensor],
    example: BenchmarkExample,
    model_config: AccuracyModelConfig,
    *,
    do_sample: bool,
) -> tuple[list[int], str]:
    input_length = int(inputs["input_ids"].shape[-1])
    kwargs: dict[str, object] = {
        **inputs,
        "max_new_tokens": example.max_new_tokens,
        # Sampling inherits temperature/top-p/top-k from the pinned
        # checkpoint's generation_config.json.
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if model_config.honor_task_stop_strings and example.stop_strings:
        kwargs["stop_strings"] = list(example.stop_strings)
        kwargs["tokenizer"] = tokenizer
    with torch.inference_mode():
        output = cast(Any, model).generate(**kwargs)
    generated = cast(Tensor, output)[0, input_length:].detach().cpu()
    ids = [int(value) for value in generated.tolist()]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return ids, text


def _generation_compatibility_payload(
    config: dict[str, Any], model_key: str, task_key: str
) -> dict[str, Any]:
    models = [item for item in config["models"] if item["key"] == model_key]
    datasets = [item for item in config["datasets"] if item["key"] == task_key]
    if len(models) != 1 or len(datasets) != 1:
        raise ValueError(f"missing generation config for {model_key}/{task_key}")
    model = dict(models[0])
    # Logical CUDA placement is execution provenance, not decoding semantics.
    model.pop("device", None)
    sampling_overrides = dict(model.pop("do_sample_overrides", {}))
    if task_key in {"humaneval", "mbpp_plus"}:
        model.setdefault("code_assistant_prefill", True)
    else:
        model.pop("code_assistant_prefill", None)
    overrides = dict(model["max_new_tokens_overrides"])
    model["max_new_tokens_overrides"] = (
        {task_key: overrides[task_key]} if task_key in overrides else {}
    )
    decode = dict(config["decode"])
    decode["do_sample"] = sampling_overrides.get(task_key, decode["do_sample"])
    return {
        "schema_version": config["schema_version"],
        "harness_revision": config["harness_revision"],
        "evalplus_revision": config["evalplus_revision"],
        "model": model,
        "dataset": datasets[0],
        "decode": decode,
    }


def _can_reuse_imported_score(source_protocol: int, target_protocol: int, task_key: str) -> bool:
    if source_protocol == target_protocol:
        return True
    if (source_protocol, target_protocol) in {(7, 8), (8, 9)}:
        return True
    if (source_protocol, target_protocol) == (9, 10):
        return task_key not in {"humaneval", "mbpp_plus"}
    if target_protocol == 11 and source_protocol in {9, 10}:
        return task_key not in {"humaneval", "mbpp_plus"}
    if target_protocol == 12 and source_protocol in {9, 10, 11}:
        if task_key == "gsm8k":
            return False
        return source_protocol == 11 or task_key not in {"humaneval", "mbpp_plus"}
    if target_protocol == 13 and source_protocol in {9, 10, 11, 12}:
        if task_key == "mbpp_plus":
            return False
        if task_key == "gsm8k":
            return source_protocol == 12
        return source_protocol >= 11 or task_key != "humaneval"
    if (source_protocol, target_protocol) == (13, 14):
        return True
    return source_protocol in {5, 6} and target_protocol == 7 and task_key != "gsm8k"


def import_compatible_vanilla_results(
    suite: AccuracySuiteConfig,
    output_root: Path,
    source_root: Path,
    models: tuple[AccuracyModelConfig, ...],
    tasks: tuple[str, ...],
) -> int:
    source_config_path = source_root / "resolved_config.json"
    if not source_config_path.is_file():
        raise ValueError(f"missing source resolved config: {source_config_path}")
    source_config = cast(dict[str, Any], json.loads(source_config_path.read_text(encoding="utf-8")))
    current_config = suite.model_dump(mode="json")
    imported = 0
    for model in models:
        model_root = output_root / "models" / model.key
        for task_key in tasks:
            if _generation_compatibility_payload(
                source_config, model.key, task_key
            ) != _generation_compatibility_payload(current_config, model.key, task_key):
                raise ValueError(
                    f"source and target generation configurations are incompatible: "
                    f"{model.key}/{task_key}"
                )
            dataset = next(item for item in suite.datasets if item.key == task_key)
            examples = load_examples(dataset, cache_dir=suite.dataset_cache_dir)
            source_path = (
                source_root
                / "models"
                / model.key
                / "results"
                / "vanilla"
                / task_key
                / "samples.jsonl"
            )
            source_rows = [
                row for row in _read_jsonl(source_path) if row.get("state") == "complete"
            ]
            target_root = model_root / "results" / "vanilla" / task_key
            target_path = target_root / "samples.jsonl"
            completed = {
                str(row["sample_id"])
                for row in _read_jsonl(target_path)
                if row.get("state") == "complete"
            }
            for source in source_rows:
                sample_id = str(source["sample_id"])
                if sample_id in completed:
                    continue
                index = int(source["row_index"])
                example = _model_example(examples[index], model)
                if example.sample_id != sample_id:
                    raise ValueError(f"source row/sample mismatch: {model.key}/{task_key}/{index}")
                if (
                    source.get("model_revision") != model.revision
                    or source.get("dataset_revision") != dataset.revision
                ):
                    raise ValueError(f"source revision mismatch: {model.key}/{task_key}/{index}")
                reuse_score = _can_reuse_imported_score(
                    int(source_config["protocol_revision"]), suite.protocol_revision, task_key
                )
                if reuse_score:
                    correct = bool(source["correct"])
                    parsed_answer = str(source["parsed_answer"])
                    score_detail = str(source["score_detail"])
                else:
                    score = score_response(example, str(source["generated_text"]))
                    correct = score.correct
                    parsed_answer = score.parsed_answer
                    score_detail = score.detail
                row = dict(source)
                row.update(
                    {
                        "suite_id": suite.suite_id,
                        "config_fingerprint": suite.fingerprint(),
                        "correct": correct,
                        "parsed_answer": parsed_answer,
                        "score_detail": score_detail,
                        "do_sample": suite.do_sample_for(model, task_key),
                        "derived": True,
                        "derived_from_suite": source.get("suite_id"),
                        "derived_from_config_fingerprint": source.get("config_fingerprint"),
                        "derivation": (
                            "generation_and_scorer_compatible_metadata_rewrite"
                            if reuse_score
                            else "generation_compatible_deterministic_rescore"
                        ),
                    }
                )
                _append_jsonl(target_path, cast(dict[str, object], row))
                completed.add(sample_id)
                imported += 1
            if len(completed) == dataset.expected_samples:
                (target_root / "DONE").write_text("complete\n", encoding="utf-8")
    return imported


def _run_task(
    root: Path,
    model: nn.Module,
    tokenizer: Any,
    model_config: AccuracyModelConfig,
    suite: AccuracySuiteConfig,
    defaults: DefaultVectorArtifact,
    policy: str,
    task_key: str,
    max_samples: int | None,
    shard_count: int,
    shard_index: int,
) -> None:
    dataset_config = next(item for item in suite.datasets if item.key == task_key)
    examples = load_examples(dataset_config, cache_dir=suite.dataset_cache_dir)
    override = model_config.max_new_tokens_overrides.get(task_key)
    if override is not None:
        if override <= 0:
            raise ValueError(f"{model_config.key}/{task_key}: invalid token override")
        examples = tuple(replace(example, max_new_tokens=override) for example in examples)
    if max_samples is not None:
        examples = examples[:max_samples]
    indexed_examples = tuple(enumerate(examples))
    selected_examples = tuple(
        item for item in indexed_examples if item[0] % shard_count == shard_index
    )
    task_root = root / "results" / policy / task_key
    shard_stem = f"{shard_index:05d}-of-{shard_count:05d}"
    result_path = (
        task_root / "samples.jsonl"
        if shard_count == 1
        else task_root / "shards" / f"{shard_stem}.jsonl"
    )
    shard_done = task_root / "shards" / f"{shard_stem}.DONE"
    if shard_count > 1:
        shard_done.unlink(missing_ok=True)
    completed = {
        str(row["sample_id"]) for row in _read_jsonl(result_path) if row.get("state") == "complete"
    }
    task_root.mkdir(parents=True, exist_ok=True)
    ops = build_prefetch_ops(model, model_config.architecture)
    context: Any
    if policy == "router_pf":
        context = policy_context(ops, "router_pf", defaults)
    elif policy == "oracle_pf":
        context = policy_context(ops, "oracle_pf")
    else:
        context = nullcontext(None)
    with context as active:
        for index, example in selected_examples:
            if example.sample_id in completed:
                continue
            seed_everything(suite.decode.seed)
            example = _model_example(example, model_config)
            inputs, rendered = _render(tokenizer, model_config, suite, example)
            before = _copy_stats(active.stats) if active is not None else PrefetchStats()
            torch.cuda.reset_peak_memory_stats(torch.device(model_config.device))
            started = time.time()
            token_ids, text = _generate(
                model,
                tokenizer,
                inputs,
                example,
                model_config,
                do_sample=suite.do_sample_for(model_config, task_key),
            )
            elapsed = time.time() - started
            after = _copy_stats(active.stats) if active is not None else PrefetchStats()
            score = score_response(example, text)
            row: dict[str, object] = {
                "schema_version": 1,
                "state": "complete",
                "suite_id": suite.suite_id,
                "config_fingerprint": suite.fingerprint(),
                "model": model_config.key,
                "model_id": model_config.model_id,
                "model_revision": model_config.revision,
                "precision_tier": model_config.precision_tier,
                "policy": policy,
                "task": task_key,
                "dataset_id": dataset_config.dataset_id,
                "dataset_revision": dataset_config.revision,
                "dataset_split": dataset_config.split,
                "row_index": index,
                "sample_id": example.sample_id,
                "user_prompt": example.user_prompt,
                "assistant_prefix": example.assistant_prefix,
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": _sha256_text(rendered),
                "target": example.target if task_key not in {"humaneval", "mbpp_plus"} else None,
                "target_sha256": _sha256_text(example.target),
                "generated_token_ids": token_ids,
                "generated_text": text,
                "generated_text_sha256": _sha256_text(text),
                "generated_tokens": len(token_ids),
                "do_sample": suite.do_sample_for(model_config, task_key),
                "parsed_answer": score.parsed_answer,
                "correct": score.correct,
                "score_detail": score.detail,
                "elapsed_seconds_measured": elapsed,
                "tokens_per_second_measured": len(token_ids) / elapsed if elapsed else None,
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(torch.device(model_config.device))
                ),
                "route_metrics": _stat_delta(before, after),
            }
            _append_jsonl(result_path, row)
            print(
                json.dumps(
                    {
                        "model": model_config.key,
                        "policy": policy,
                        "task": task_key,
                        "index": index,
                        "total": len(examples),
                        "shard_count": shard_count,
                        "shard_index": shard_index,
                        "correct": score.correct,
                        "tokens": len(token_ids),
                        "elapsed": round(elapsed, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(selected_examples)
    actual = sum(row.get("state") == "complete" for row in _read_jsonl(result_path))
    if actual != expected:
        raise RuntimeError(f"{model_config.key}/{policy}/{task_key}: {actual}/{expected} complete")
    if shard_count == 1:
        (task_root / "DONE").write_text("complete\n", encoding="utf-8")
    else:
        shard_done.write_text("complete\n", encoding="utf-8")
        _merge_task_shards(task_root, shard_count, len(examples))


def _materialize_oracle_task(
    root: Path,
    model_config: AccuracyModelConfig,
    suite: AccuracySuiteConfig,
    task_key: str,
    max_samples: int | None,
) -> None:
    dataset = next(item for item in suite.datasets if item.key == task_key)
    vanilla_path = root / "results" / "vanilla" / task_key / "samples.jsonl"
    vanilla = [row for row in _read_jsonl(vanilla_path) if row.get("state") == "complete"]
    expected = (
        min(dataset.expected_samples, max_samples) if max_samples else dataset.expected_samples
    )
    if len(vanilla) != expected:
        raise RuntimeError(
            f"{model_config.key}/oracle_pf/{task_key}: vanilla is only "
            f"{len(vanilla)}/{expected} complete"
        )
    target = root / "results" / "oracle_pf" / task_key
    target.mkdir(parents=True, exist_ok=True)
    output = target / "samples.jsonl"
    completed = {
        str(row["sample_id"]) for row in _read_jsonl(output) if row.get("state") == "complete"
    }
    for source in vanilla:
        sample_id = str(source["sample_id"])
        if sample_id in completed:
            continue
        row = dict(source)
        row.update(
            {
                "policy": "oracle_pf",
                "derived": True,
                "derived_from_policy": "vanilla",
                "derivation": "validated_exact_native_route_identity",
                "oracle_information": "perfect_layer_ahead_natural_ids_and_weights",
                "elapsed_seconds_measured": None,
                "tokens_per_second_measured": None,
                "peak_cuda_allocated_bytes": None,
                "route_metrics": {
                    "exact_topk_agreement": 1.0,
                    "selected_route_hit_rate": 1.0,
                },
            }
        )
        _append_jsonl(output, cast(dict[str, object], row))
    if len(_read_jsonl(output)) != expected:
        raise RuntimeError(f"{model_config.key}/oracle_pf/{task_key}: materialization incomplete")
    (target / "DONE").write_text("complete\n", encoding="utf-8")


def run_model(
    suite: AccuracySuiteConfig,
    model_config: AccuracyModelConfig,
    output_root: Path,
    *,
    policies: tuple[str, ...],
    tasks: tuple[str, ...],
    max_samples: int | None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> None:
    root = output_root / "models" / model_config.key
    root.mkdir(parents=True, exist_ok=True)
    worker_suffix = "" if shard_count == 1 else f".shard-{shard_index}-of-{shard_count}"
    running_path = root / f"RUNNING{worker_suffix}"
    failed_path = root / f"FAILED{worker_suffix}.json"
    manifest_path = root / f"manifest{worker_suffix}.json"
    running_path.write_text("running\n", encoding="utf-8")
    started = time.time()
    try:
        seed_everything(suite.decode.seed)
        torch.cuda.set_device(torch.device(model_config.device))
        model, tokenizer = _load_model(model_config, suite)
        defaults = _ensure_defaults(root, model, tokenizer, model_config, suite)
        validation = validate_policy_path(root, model, tokenizer, model_config, suite, defaults)
        for policy in policies:
            for task in tasks:
                if policy == "oracle_pf":
                    _materialize_oracle_task(root, model_config, suite, task, max_samples)
                else:
                    _run_task(
                        root,
                        model,
                        tokenizer,
                        model_config,
                        suite,
                        defaults,
                        policy,
                        task,
                        max_samples,
                        shard_count,
                        shard_index,
                    )
        manifest = {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "config_fingerprint": suite.fingerprint(),
            "model": model_config.model_dump(mode="json"),
            "policies": list(policies),
            "tasks": list(tasks),
            "max_samples": max_samples,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "validation": validation,
            "default_vector_fingerprint": defaults.fingerprint,
            "elapsed_seconds": time.time() - started,
            "software_hardware": _software_hardware(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        failed_path.unlink(missing_ok=True)
        if shard_count == 1:
            (root / "DONE").write_text("complete\n", encoding="utf-8")
    except BaseException as error:
        failure = {
            "state": "failed",
            "model": model_config.key,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        failed_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    finally:
        running_path.unlink(missing_ok=True)


def _paired_policy_comparisons(
    suite: AccuracySuiteConfig,
    paired: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for model in suite.models:
        for dataset in suite.datasets:
            triples = [
                policies
                for (model_key, task, _), policies in paired.items()
                if model_key == model.key
                and task == dataset.key
                and {"vanilla", "router_pf", "oracle_pf"} <= policies.keys()
            ]
            if not triples:
                continue
            differences = [
                int(bool(triple["oracle_pf"]["correct"]))
                - int(bool(triple["router_pf"]["correct"]))
                for triple in triples
            ]
            total = len(differences)
            difference = sum(differences) / total
            if total > 1:
                sample_variance = sum((value - difference) ** 2 for value in differences) / (
                    total - 1
                )
                standard_error = math.sqrt(sample_variance / total)
            else:
                standard_error = 0.0
            oracle_accuracy = (
                sum(bool(triple["oracle_pf"]["correct"]) for triple in triples) / total
            )
            router_accuracy = (
                sum(bool(triple["router_pf"]["correct"]) for triple in triples) / total
            )
            paper_router = suite.paper_results[model.key][dataset.key].router_pf_accuracy
            comparisons.append(
                {
                    "model": model.key,
                    "precision_tier": model.precision_tier,
                    "task": dataset.key,
                    "paired_samples": total,
                    "oracle_accuracy": oracle_accuracy,
                    "router_pf_accuracy": router_accuracy,
                    "oracle_minus_local_router_pf": difference,
                    "paired_difference_standard_error": standard_error,
                    "paired_difference_ci95_low": max(-1.0, difference - 1.96 * standard_error),
                    "paired_difference_ci95_high": min(1.0, difference + 1.96 * standard_error),
                    "oracle_only_correct": sum(value == 1 for value in differences),
                    "router_pf_only_correct": sum(value == -1 for value in differences),
                    "both_same_correctness": sum(value == 0 for value in differences),
                    "paper_router_pf_accuracy": paper_router,
                    "router_pf_minus_paper": router_accuracy - paper_router,
                    "oracle_minus_paper_router_pf": oracle_accuracy - paper_router,
                }
            )
    return comparisons


def aggregate(suite: AccuracySuiteConfig, output_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    paired: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for model in suite.models:
        for policy in suite.policies:
            for dataset in suite.datasets:
                path = (
                    output_root
                    / "models"
                    / model.key
                    / "results"
                    / policy
                    / dataset.key
                    / "samples.jsonl"
                )
                samples = [
                    sample for sample in _read_jsonl(path) if sample.get("state") == "complete"
                ]
                if not samples:
                    continue
                successes = sum(bool(sample["correct"]) for sample in samples)
                total = len(samples)
                accuracy = successes / total
                standard_error = math.sqrt(accuracy * (1 - accuracy) / total)
                lower, upper = wilson_interval(successes, total)
                paper = suite.paper_results[model.key][dataset.key]
                paper_reference = (
                    paper.vanilla_accuracy if policy == "vanilla" else paper.router_pf_accuracy
                )
                compared_tokens = sum(
                    int(
                        cast(
                            Any,
                            cast(dict[str, object], sample.get("route_metrics", {})).get(
                                "compared_tokens", 0
                            ),
                        )
                    )
                    for sample in samples
                )
                exact_topk_tokens = sum(
                    int(
                        cast(
                            Any,
                            cast(dict[str, object], sample.get("route_metrics", {})).get(
                                "exact_topk_tokens", 0
                            ),
                        )
                    )
                    for sample in samples
                )
                selected_slots = sum(
                    int(
                        cast(
                            Any,
                            cast(dict[str, object], sample.get("route_metrics", {})).get(
                                "selected_slots", 0
                            ),
                        )
                    )
                    for sample in samples
                )
                matching_selected_slots = sum(
                    int(
                        cast(
                            Any,
                            cast(dict[str, object], sample.get("route_metrics", {})).get(
                                "matching_selected_slots", 0
                            ),
                        )
                    )
                    for sample in samples
                )
                row: dict[str, object] = {
                    "model": model.key,
                    "precision_tier": model.precision_tier,
                    "policy": policy,
                    "task": dataset.key,
                    "successes": successes,
                    "samples": total,
                    "accuracy": accuracy,
                    "standard_error": standard_error,
                    "wilson95_low": lower,
                    "wilson95_high": upper,
                    "paper_reference_accuracy": paper_reference,
                    "local_minus_paper": accuracy - paper_reference,
                }
                if compared_tokens:
                    row.update(
                        {
                            "route_compared_tokens": compared_tokens,
                            "route_exact_topk_agreement": (exact_topk_tokens / compared_tokens),
                            "route_selected_slots": selected_slots,
                            "route_selected_hit_rate": (matching_selected_slots / selected_slots),
                        }
                    )
                if policy == "vanilla":
                    tolerance = max(
                        suite.alignment.standard_errors * paper.vanilla_standard_error,
                        suite.alignment.minimum_discrete_questions / dataset.expected_samples,
                    )
                    row["alignment_tolerance"] = tolerance
                    row["vanilla_aligned"] = (
                        total == dataset.expected_samples
                        and abs(accuracy - paper.vanilla_accuracy) <= tolerance
                    )
                rows.append(row)
                for sample in samples:
                    key = (model.key, dataset.key, str(sample["sample_id"]))
                    paired.setdefault(key, {})[policy] = sample
    alignment: dict[str, bool] = {}
    for model in suite.models:
        model_rows = [
            row for row in rows if row["model"] == model.key and row["policy"] == "vanilla"
        ]
        alignment[model.key] = len(model_rows) == len(suite.datasets) and all(
            bool(row.get("vanilla_aligned")) for row in model_rows
        )
    exact_text: dict[str, dict[str, float]] = {}
    for model in suite.models:
        exact_text[model.key] = {}
        for dataset in suite.datasets:
            pairs = [
                policies
                for (model_key, task, _), policies in paired.items()
                if model_key == model.key
                and task == dataset.key
                and "vanilla" in policies
                and "oracle_pf" in policies
            ]
            if pairs:
                exact_text[model.key][dataset.key] = sum(
                    pair["vanilla"]["generated_token_ids"]
                    == pair["oracle_pf"]["generated_token_ids"]
                    for pair in pairs
                ) / len(pairs)
    summary = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "paper_disclosure": (
            "reconstruction: public paper/code omit benchmark driver, prompts, seeds, "
            "default-vector corpus/artifacts, and GPT-OSS PF implementation"
        ),
        "model_vanilla_alignment": alignment,
        "oracle_vanilla_exact_token_agreement": exact_text,
        "paired_policy_comparisons": _paired_policy_comparisons(suite, paired),
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows:
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _suite_results_complete(suite: AccuracySuiteConfig, output_root: Path) -> bool:
    for model in suite.models:
        for policy in suite.policies:
            for dataset in suite.datasets:
                path = (
                    output_root
                    / "models"
                    / model.key
                    / "results"
                    / policy
                    / dataset.key
                    / "samples.jsonl"
                )
                samples = [row for row in _read_jsonl(path) if row.get("state") == "complete"]
                if len(samples) != dataset.expected_samples:
                    return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_accuracy_suite(suite: AccuracySuiteConfig, output_root: Path) -> dict[str, object]:
    summary = aggregate(suite, output_root)
    if not _suite_results_complete(suite, output_root):
        raise RuntimeError("accuracy suite is incomplete; refusing to write a complete envelope")
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_envelope.json":
            continue
        relative = path.relative_to(output_root)
        if relative.parts[0] == "protocol_history" or path.name == "RUNNING":
            continue
        artifacts.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    envelope: dict[str, object] = {
        "schema_version": 1,
        "state": "complete",
        "suite_id": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
    }
    (output_root / "artifact_envelope.json").write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def validate_accuracy_envelope(output_root: Path) -> dict[str, object]:
    path = output_root / "artifact_envelope.json"
    envelope = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    artifacts = cast(list[dict[str, object]], envelope["artifacts"])
    for artifact in artifacts:
        artifact_path = output_root / str(artifact["path"])
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(cast(Any, artifact["bytes"]))
            or _sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise ValueError(f"artifact checksum mismatch: {artifact_path}")
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudoroute-accuracy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-key")
    parser.add_argument("--policies")
    parser.add_argument("--tasks")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--materialize-oracle-only", action="store_true")
    parser.add_argument("--import-compatible-vanilla-from")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--validate-envelope-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = load_accuracy_suite_config(args.config)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-count must be positive and shard-index must be in range")
    output_root = Path(args.output_dir)
    if args.validate_envelope_only:
        print(json.dumps(validate_accuracy_envelope(output_root), indent=2, sort_keys=True))
        return 0
    if args.finalize_only:
        print(json.dumps(finalize_accuracy_suite(suite, output_root), indent=2, sort_keys=True))
        return 0
    if args.aggregate_only:
        print(json.dumps(aggregate(suite, output_root), indent=2, sort_keys=True))
        return 0
    models = (
        tuple(model for model in suite.models if model.key == args.model_key)
        if args.model_key
        else suite.models
    )
    if not models:
        raise SystemExit(f"unknown model key: {args.model_key}")
    policies = tuple(args.policies.split(",")) if args.policies else suite.policies
    tasks = (
        tuple(args.tasks.split(","))
        if args.tasks
        else tuple(dataset.key for dataset in suite.datasets)
    )
    unknown_policies = set(policies) - set(suite.policies)
    unknown_tasks = set(tasks) - {dataset.key for dataset in suite.datasets}
    if unknown_policies or unknown_tasks:
        raise SystemExit(
            f"unknown policies/tasks: {sorted(unknown_policies)} {sorted(unknown_tasks)}"
        )
    if args.materialize_oracle_only:
        if args.shard_count != 1:
            raise SystemExit("oracle materialization does not accept row sharding")
        if policies != ("oracle_pf",):
            raise SystemExit("--materialize-oracle-only requires --policies oracle_pf")
        for model in models:
            root = output_root / "models" / model.key
            for task in tasks:
                _materialize_oracle_task(root, model, suite, task, args.max_samples)
        print(json.dumps(aggregate(suite, output_root), indent=2, sort_keys=True))
        return 0
    estimate = {
        "suite": suite.suite_id,
        "config_fingerprint": suite.fingerprint(),
        "models": [model.key for model in models],
        "policies": list(policies),
        "tasks": list(tasks),
        "max_samples": args.max_samples,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "output_dir": str(output_root),
    }
    if args.dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(
        json.dumps(suite.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "environment.json").write_text(
        json.dumps(_software_hardware(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.import_only and not args.import_compatible_vanilla_from:
        raise SystemExit("--import-only requires --import-compatible-vanilla-from")
    if args.import_compatible_vanilla_from:
        imported = import_compatible_vanilla_results(
            suite,
            output_root,
            Path(args.import_compatible_vanilla_from),
            models,
            tasks,
        )
        print(json.dumps({"imported_compatible_vanilla_rows": imported}, sort_keys=True))
    if args.import_only:
        print(json.dumps(aggregate(suite, output_root), indent=2, sort_keys=True))
        return 0
    for model in models:
        run_model(
            suite,
            model,
            output_root,
            policies=policies,
            tasks=tasks,
            max_samples=args.max_samples,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    print(json.dumps(aggregate(suite, output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
