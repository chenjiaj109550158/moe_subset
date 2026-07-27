"""Per-model trained checkpoint validation and real-text trace collection."""

from __future__ import annotations

import hashlib
import json
import resource
import threading
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional

from pseudoroute.execution.routing_policy import NaturalRoutingPolicy
from pseudoroute.models.base import MoEModelAdapter, TraceLevel, TraceRequest
from pseudoroute.models.introspection import export_model_manifest, model_manifest
from pseudoroute.tracing import TraceStore, validate_trace
from pseudoroute.tracing.store import sha256_file
from pseudoroute.trained.config import (
    TrainedDatasetConfig,
    TrainedModelConfig,
    TrainedSuiteConfig,
)
from pseudoroute.trained.loader import input_device, load_adapter, load_tokenizer


class _RssPeakSampler:
    """Sample process RSS for one model task instead of process-lifetime high water."""

    def __init__(self) -> None:
        import psutil  # type: ignore[import-untyped]

        self._process = psutil.Process()
        self._peak = int(self._process.memory_info().rss)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            self._peak = max(self._peak, int(self._process.memory_info().rss))

    def stop(self) -> int:
        self._peak = max(self._peak, int(self._process.memory_info().rss))
        self._stop.set()
        self._thread.join()
        return self._peak


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        payload = cast(str, backend.to_str())
    else:
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _snapshot_path(cache_dir: str, model: TrainedModelConfig) -> Path:
    cache_name = "models--" + model.model_id.replace("/", "--")
    return Path(cache_dir) / cache_name / "snapshots" / model.revision


def _checkpoint_records(cache_dir: str, model: TrainedModelConfig) -> list[dict[str, object]]:
    records = []
    for path in sorted(_snapshot_path(cache_dir, model).glob("model*.safetensors")):
        resolved = path.resolve()
        blob_name = resolved.name
        records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": blob_name if len(blob_name) == 64 else sha256_file(path),
                "checksum_source": "hf_content_addressed_blob"
                if len(blob_name) == 64
                else "local_hash",
            }
        )
    if not records:
        raise FileNotFoundError(f"no Transformers safetensors shards for {model.model_id}")
    return records


def _render(dataset: TrainedDatasetConfig, row: dict[str, Any]) -> str:
    return dataset.prompt_template.format(**row).strip()


def _load_rows(
    dataset: TrainedDatasetConfig, cache_dir: str
) -> tuple[tuple[int, dict[str, Any], str], ...]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    data = load_dataset(
        dataset.dataset_id,
        dataset.config,
        split=dataset.split,
        revision=dataset.revision,
        cache_dir=str(Path(cache_dir).parent / "datasets"),
    )
    rows = []
    for row_id in dataset.row_ids:
        raw = dict(data[row_id])
        rendered = _render(dataset, raw)
        if not rendered:
            raise ValueError(f"empty rendered sample {dataset.key}-{row_id}")
        rows.append((row_id, raw, rendered))
    return tuple(rows)


def _greedy_two_steps(adapter: MoEModelAdapter, prompt: torch.Tensor, *, policy: bool) -> list[int]:
    cache: object | None = None
    current = prompt
    generated: list[int] = []
    for _ in range(2):
        if policy:
            result = adapter.forward_with_policy(
                current, NaturalRoutingPolicy(), kv_cache=cache, use_cache=True
            )
        else:
            result = adapter.run_base_forward(current, kv_cache=cache, use_cache=True)
        cache = result.kv_cache
        token = result.logits[:, -1].argmax(dim=-1)
        generated.append(int(token.item()))
        current = token[:, None].to(input_device(adapter))
    return generated


def _validate_loaded_adapter(
    adapter: MoEModelAdapter,
    tokenizer: Any,
    model: TrainedModelConfig,
    smoke_text: str,
) -> dict[str, object]:
    device = input_device(adapter)
    tokens = tokenizer(smoke_text, return_tensors="pt", truncation=True, max_length=8)[
        "input_ids"
    ].to(device)
    base = adapter.run_base_forward(tokens)
    traced = adapter.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    natural = adapter.forward_with_policy(tokens, NaturalRoutingPolicy())
    if not torch.equal(base.logits, traced.logits):
        raise AssertionError("traced forward changed checkpoint logits")
    if not torch.equal(base.logits, natural.logits):
        raise AssertionError("NaturalRoutingPolicy changed checkpoint logits")
    expected_records = int(tokens.numel()) * len(adapter.spec.moe_layer_indices)
    if len(traced.traces) != expected_records:
        raise AssertionError(f"expected {expected_records} trace records, got {len(traced.traces)}")
    for trace in traced.traces:
        route = adapter.route_from_state(
            trace.layer_idx,
            torch.zeros(
                (1, adapter.spec.hidden_size),
                device=next(cast(Any, adapter).model.parameters()).device,
                dtype=next(cast(Any, adapter).model.parameters()).dtype,
            ),
        )
        if route.raw_logits.shape[-1] != adapter.spec.num_experts_by_layer[trace.layer_idx]:
            raise AssertionError("route_from_state expert axis mismatch")
        break
    base_tokens = _greedy_two_steps(adapter, tokens, policy=False)
    policy_tokens = _greedy_two_steps(adapter, tokens, policy=True)
    if base_tokens != policy_tokens:
        raise AssertionError("natural-policy generation diverged")
    handles = adapter.iter_moe_layers()
    if tuple(handle.layer_idx for handle in handles) != adapter.spec.moe_layer_indices:
        raise AssertionError("MoE handle indices do not match ModelSpec")
    for handle in handles:
        if handle.shared_expert_count != adapter.spec.shared_experts_by_layer.get(
            handle.layer_idx, 0
        ):
            raise AssertionError("shared expert metadata mismatch")
    model_object = cast(Any, adapter).model
    device_map = {
        key: str(value) for key, value in getattr(model_object, "hf_device_map", {}).items()
    }
    return {
        "model_id": model.model_id,
        "revision": model.revision,
        "architecture": adapter.spec.architecture,
        "moe_layer_indices": list(adapter.spec.moe_layer_indices),
        "num_experts_by_layer": adapter.spec.num_experts_by_layer,
        "top_k_by_layer": adapter.spec.top_k_by_layer,
        "shared_experts_by_layer": adapter.spec.shared_experts_by_layer,
        "routing_semantics_by_layer": adapter.spec.routing_semantics_by_layer,
        "base_trace_logits_exact": True,
        "base_natural_policy_logits_exact": True,
        "natural_policy_generation_exact": True,
        "generated_tokens": base_tokens,
        "finite_logits": bool(torch.isfinite(traced.logits).all()),
        "device_map": device_map,
        "quantization_tier": model.tier,
    }


def _qualified_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _inspect_loaded_adapter(
    adapter: MoEModelAdapter, model: TrainedModelConfig
) -> dict[str, object]:
    """Persist structure and a direct native-router call before trace validation."""
    handles = adapter.iter_moe_layers()
    if not handles:
        raise RuntimeError("trained adapter exposes no MoE layers")
    first = handles[0]
    router_parameter = next(first.router.parameters())
    state = torch.zeros(
        (1, adapter.spec.hidden_size),
        device=router_parameter.device,
        dtype=router_parameter.dtype,
    )
    with torch.inference_mode():
        route = adapter.route_from_state(first.layer_idx, state)
    model_object = cast(Any, adapter).model
    layers = cast(Any, model_object).model.layers
    expert_containers = []
    for layer_idx in adapter.spec.moe_layer_indices:
        mlp = layers[layer_idx].mlp
        expert_containers.append(_qualified_name(getattr(mlp, "experts", mlp)))
    quantization = getattr(cast(Any, model_object).config, "quantization_config", None)
    if quantization is not None and hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    if quantization is not None and not isinstance(
        quantization, (dict, list, str, int, float, bool)
    ):
        quantization = str(quantization)
    expert_bytes = tuple(adapter.spec.expert_bytes.values())
    return {
        "model_id": model.model_id,
        "revision": model.revision,
        "architecture": adapter.spec.architecture,
        "tier": model.tier,
        "quantization_config": quantization,
        "structure_validated": True,
        "moe_layer_indices": list(adapter.spec.moe_layer_indices),
        "num_experts_by_layer": adapter.spec.num_experts_by_layer,
        "top_k_by_layer": adapter.spec.top_k_by_layer,
        "shared_experts_by_layer": adapter.spec.shared_experts_by_layer,
        "routing_semantics_by_layer": adapter.spec.routing_semantics_by_layer,
        "router_module_classes": sorted({_qualified_name(handle.router) for handle in handles}),
        "expert_container_classes": sorted(set(expert_containers)),
        "physical_expert_bytes_min": min(expert_bytes),
        "physical_expert_bytes_max": max(expert_bytes),
        "direct_native_router_call": {
            "layer_idx": first.layer_idx,
            "raw_logits_shape": list(route.raw_logits.shape),
            "pre_topk_scores_shape": list(route.pre_topk_scores.shape),
            "selected_ids": route.topk_ids.reshape(-1).tolist(),
            "selected_weights": route.topk_weights.float().reshape(-1).tolist(),
            "finite": bool(
                torch.isfinite(route.raw_logits).all()
                and torch.isfinite(route.pre_topk_scores).all()
                and torch.isfinite(route.topk_weights).all()
            ),
        },
        "model_fingerprint": model_manifest(adapter.spec, revision=model.revision)["fingerprint"],
    }


def collect_model_traces(
    suite: TrainedSuiteConfig,
    model: TrainedModelConfig,
    output_root: Path,
) -> dict[str, object]:
    started = time.time()
    rss_sampler = _RssPeakSampler()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.reset_peak_memory_stats()
    rows_by_dataset = {
        dataset.key: _load_rows(dataset, suite.cache_dir) for dataset in suite.datasets
    }
    tokenizer = load_tokenizer(model, suite.cache_dir)
    tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
    adapter = load_adapter(model, suite.cache_dir)
    root = output_root / model.key
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint_files.json").write_text(
        json.dumps(_checkpoint_records(suite.cache_dir, model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "inspection.json").write_text(
        json.dumps(_inspect_loaded_adapter(adapter, model), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke_text = rows_by_dataset["wikitext"][0][2]
    validation = _validate_loaded_adapter(adapter, tokenizer, model, smoke_text)
    source_records: list[dict[str, object]] = []
    quality_path = root / "trace_quality.json"
    existing_quality = (
        json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else []
    )
    quality_by_sample: dict[str, dict[str, object]] = {
        str(record["sample_id"]): record for record in existing_quality
    }
    for dataset in suite.datasets:
        trace_root = root / "traces" / dataset.key
        store = TraceStore.create(
            trace_root,
            spec=adapter.spec,
            model_revision=model.revision,
            dataset_id=dataset.dataset_id,
            dataset_revision=dataset.revision,
            dataset_config=dataset.config,
            dataset_split=dataset.split,
            trace_level=TraceLevel.ROUTER_LOGITS,
            resume=(trace_root / "manifest.json").exists(),
            tokenizer_fingerprint=tokenizer_fingerprint,
            seed=suite.trace.seed,
            full_router_mass_valid=model.full_router_mass_valid,
        )
        export_model_manifest(
            trace_root / "model_manifest.json", adapter.spec, revision=model.revision
        )
        for row_id, raw, rendered in rows_by_dataset[dataset.key]:
            sample_id = f"{dataset.key}-{row_id}"
            text_hash = hashlib.sha256(rendered.encode()).hexdigest()
            source_records.append(
                {
                    "sample_id": sample_id,
                    "domain": dataset.key,
                    "dataset_id": dataset.dataset_id,
                    "dataset_revision": dataset.revision,
                    "dataset_split": dataset.split,
                    "source_row_id": row_id,
                    "prompt_rendering": dataset.prompt_template,
                    "rendered_text": rendered,
                    "rendered_text_sha256": text_hash,
                    "answer": raw.get("answer") if dataset.key == "gsm8k" else None,
                }
            )
            completed = sample_id in store.completed_sample_ids
            if completed and sample_id in quality_by_sample:
                continue
            token_ids = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=suite.trace.max_length,
            )["input_ids"].to(input_device(adapter))
            result = adapter.run_base_forward(
                token_ids,
                trace_request=None if completed else TraceRequest(TraceLevel.ROUTER_LOGITS),
            )
            if not completed:
                store.append_sample(
                    sample_id,
                    token_ids,
                    result.traces,
                    is_prompt=True,
                    source_row_id=str(row_id),
                    domain=dataset.key,
                    prompt_rendering=dataset.prompt_template,
                    source_text_sha256=text_hash,
                )
            logits = result.logits[:, :-1].float()
            labels = token_ids[:, 1:].to(logits.device)
            nll = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            quality_by_sample[sample_id] = {
                "sample_id": sample_id,
                "domain": dataset.key,
                "num_tokens": int(token_ids.numel()),
                "teacher_forced_nll": float(nll),
                "teacher_forced_perplexity": float(nll.exp()),
            }
        if not store.manifest.complete:
            store.mark_complete()
        validate_trace(trace_root)
    quality_records = [quality_by_sample[str(record["sample_id"])] for record in source_records]
    (root / "source_samples.json").write_text(
        json.dumps(source_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "trace_quality.json").write_text(
        json.dumps(quality_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from pseudoroute.trained.closed_loop import run_closed_loop

    closed_loop_rows = run_closed_loop(suite, model, adapter, tokenizer, root)
    peak_cuda = [
        torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
    ]
    peak_cpu_rss_bytes = rss_sampler.stop()
    validation.update(
        tokenizer_fingerprint=tokenizer_fingerprint,
        model_fingerprint=model_manifest(adapter.spec, revision=model.revision)["fingerprint"],
        peak_cuda_allocated_bytes=peak_cuda,
        peak_cpu_rss_bytes=peak_cpu_rss_bytes,
        process_lifetime_peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_seconds=time.time() - started,
        closed_loop_rows=len(closed_loop_rows),
    )
    (root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "source_samples.json").write_text(
        json.dumps(source_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "trace_quality.json").write_text(
        json.dumps(quality_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(
            {
                "suite": suite.model_dump(mode="json"),
                "model": model.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return validation
