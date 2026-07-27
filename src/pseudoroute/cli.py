"""Command-line surface for PseudoRoute-MoE."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch

from pseudoroute.config import (
    PredictorTrainConfig,
    load_closed_loop_config,
    load_config,
    load_default_vector_config,
    load_factorial_config,
    load_offload_runtime_config,
    load_offload_simulator_config,
    load_oracle_config,
    load_predictor_train_config,
    load_probe_evaluation_config,
    load_router_analysis_config,
    load_shadow_probe_config,
    load_static_adaptive_config,
    load_trace_config,
    resolve_device,
)
from pseudoroute.models.base import TraceLevel
from pseudoroute.models.tiny_moe import TinyMoE
from pseudoroute.utils.determinism import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudoroute")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "inspect-model",
        "collect-traces",
        "oracle-sweep",
        "closed-loop-eval",
        "dapq-factorial",
        "analyze-router",
        "build-default-vectors",
        "train-predictor",
        "evaluate-probe",
        "simulate-offload",
        "benchmark-offload",
        "trained-suite",
        "accuracy-suite",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--output-dir")
        subparser.add_argument("--dry-run", action="store_true")
    aggregate = subparsers.add_parser("aggregate-results")
    aggregate.add_argument("--input-root", required=True)
    aggregate.add_argument("--output-dir", required=True)
    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("--suite", default="primary")
    reproduce.add_argument("--config-root", default="configs/paper")
    reproduce.add_argument("--output-dir", required=True)
    reproduce.add_argument("--dry-run", action="store_true")
    return parser


def inspect_model(config_path: str, *, dry_run: bool) -> int:
    config = load_config(config_path)
    device = resolve_device(config.model.device)
    if dry_run:
        print(
            json.dumps(
                {
                    "command": "inspect-model",
                    "information_regime": config.experiment.information_regime.value,
                    "actual_device": str(device),
                    "device": device,
                    "config_fingerprint": config.fingerprint(),
                    "synthetic_expert_bytes_total": (
                        config.model.num_layers
                        * config.model.num_experts
                        * config.model.synthetic_expert_bytes
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    seed_everything(config.experiment.seed)
    device = resolve_device(config.model.device)
    model = TinyMoE(config.model, seed=config.experiment.seed, device=device).to(device)
    sample = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    output = model(sample)
    generated = model.generate(
        sample,
        max_new_tokens=config.decode.max_new_tokens,
        eos_token_id=config.decode.eos_token_id,
    )
    result = {
        "information_regime": config.experiment.information_regime.value,
        "actual_device": str(device),
        "device": device,
        "config_fingerprint": config.fingerprint(),
        "model": model.manifest(),
        "inspection": {
            "input_shape": list(sample.shape),
            "logits_shape": list(output.logits.shape),
            "route_records": len(output.traces),
            "generated_token_ids": generated.tolist(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def collect_traces(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    config = load_trace_config(config_path)
    if output_dir is None:
        raise SystemExit("collect-traces requires --output-dir")
    estimate = {
        "command": "collect-traces",
        "information_regime": config.information_regime,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "dataset_id": config.data.dataset_id,
        "dataset_revision": config.data.revision,
        "expected_trace_bytes_upper_bound": config.trace.max_samples * 1024 * 1024,
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("install pseudoroute-moe[hf] for collect-traces") from error
    from pathlib import Path

    from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
    from pseudoroute.tracing import TraceStore, collect_sample, validate_trace

    device = resolve_device(config.model.device)
    adapter = HFMixtralAdapter.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        device=device,
        cache_dir=config.model.cache_dir,
        local_files_only=config.model.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        cache_dir=config.model.cache_dir,
        local_files_only=config.model.local_files_only,
        trust_remote_code=False,
    )
    cache_root = str(Path(config.model.cache_dir).parent / "datasets")
    dataset = load_dataset(
        config.data.dataset_id,
        config.data.config,
        split=config.data.split,
        revision=config.data.revision,
        cache_dir=cache_root,
    )
    level = TraceLevel(config.trace.level)
    root = Path(output_dir)
    from pseudoroute.models.introspection import export_model_manifest

    store = TraceStore.create(
        root,
        spec=adapter.spec,
        model_revision=config.model.revision,
        dataset_id=config.data.dataset_id,
        dataset_revision=config.data.revision,
        dataset_split=config.data.split,
        trace_level=level,
        resume=root.joinpath("manifest.json").exists(),
    )
    export_model_manifest(
        root / "model_manifest.json", adapter.spec, revision=config.model.revision
    )
    if store.manifest.complete:
        manifest = validate_trace(root)
        print(manifest.model_dump_json(indent=2))
        return 0
    collected = 0
    for row_index, row in enumerate(dataset):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        token_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=16)[
            "input_ids"
        ].to(device)
        if collect_sample(
            adapter,
            store,
            sample_id=f"{config.data.split}-{row_index}",
            token_ids=token_ids,
            trace_level=level,
        ):
            collected += 1
        if collected >= config.trace.max_samples:
            break
    if collected == 0 and not store.completed_sample_ids:
        raise RuntimeError("no non-empty dataset samples were collected")
    store.mark_complete()
    manifest = validate_trace(root)
    print(manifest.model_dump_json(indent=2))
    return 0


def oracle_sweep(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    from pathlib import Path

    from pseudoroute.models.introspection import load_model_manifest
    from pseudoroute.oracle.selectors import OracleSelector
    from pseudoroute.oracle.sweep import run_oracle_sweep, write_csv
    from pseudoroute.oracle.windows import OracleSample, load_trace_samples
    from pseudoroute.plotting.oracle import write_oracle_plots

    config = load_oracle_config(config_path)
    if output_dir is None:
        raise SystemExit("oracle-sweep requires --output-dir")
    trace_samples: tuple[OracleSample, ...] | None = None
    if config.trace_dir is not None:
        trace_root = Path(config.trace_dir)
        trace_samples = load_trace_samples(trace_root)
        sample_lengths = tuple(int(sample.token_ids.numel()) for sample in trace_samples)
        source = f"trace:{trace_root}"
    else:
        sample_lengths = tuple(len(sample) for sample in config.samples)
        source = f"tiny:{config.tiny_model_config}"
    estimated_windows = sum(
        max(0, length - horizon + 1)
        for length in sample_lengths
        for horizon in config.window.horizons
    )
    estimate = {
        "command": "oracle-sweep",
        "information_regime": "oracle",
        "open_loop": True,
        "source": source,
        "estimated_windows": estimated_windows,
        "estimated_rows": estimated_windows
        * len(config.window.budget_ratios)
        * len(config.oracle.selectors),
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed oracle run already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if trace_samples is not None:
        samples = trace_samples
        spec = load_model_manifest(Path(config.trace_dir or "") / "model_manifest.json")
    else:
        if config.tiny_model_config is None:
            raise AssertionError("validated tiny sweep is missing its model config")
        tiny_config = load_config(config.tiny_model_config)
        seed_everything(config.experiment.seed)
        device = resolve_device(tiny_config.model.device)
        from pseudoroute.models.adapters.tiny import TinyMoEAdapter
        from pseudoroute.models.base import TraceRequest

        adapter = TinyMoEAdapter(
            TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device)
        )
        built_samples = []
        for sample_index, values in enumerate(config.samples):
            token_ids = torch.tensor([values], dtype=torch.long, device=device)
            result = adapter.run_base_forward(
                token_ids, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS)
            )
            by_key = {(trace.token_position, trace.layer_idx): trace for trace in result.traces}
            ordered = [
                by_key[(position, layer)]
                for position in range(len(values))
                for layer in adapter.spec.moe_layer_indices
            ]
            shape = (len(values), len(adapter.spec.moe_layer_indices), -1)
            built_samples.append(
                OracleSample(
                    sample_id=f"tiny-{sample_index}",
                    token_ids=token_ids.cpu(),
                    topk_ids=torch.stack([trace.topk_ids.reshape(-1) for trace in ordered])
                    .reshape(shape)
                    .cpu(),
                    topk_weights=torch.stack([trace.topk_weights.reshape(-1) for trace in ordered])
                    .reshape(shape)
                    .cpu(),
                    router_logits=torch.stack([trace.raw_logits.reshape(-1) for trace in ordered])
                    .reshape(shape)
                    .cpu(),
                )
            )
        samples = tuple(built_samples)
        spec = adapter.spec
    rows, layer_rows = run_oracle_sweep(
        samples,
        spec,
        horizons=config.window.horizons,
        budget_ratios=config.window.budget_ratios,
        selectors=tuple(OracleSelector(value) for value in config.oracle.selectors),
        gamma=config.window.gamma,
        load_cost_lambda=config.oracle.load_cost_lambda,
    )
    write_csv(root / "oracle_windows.csv", list(rows))
    write_csv(root / "oracle_layers.csv", list(layer_rows))
    write_oracle_plots(root, rows, layer_rows)
    metrics = {
        "schema_version": 1,
        "information_regime": "oracle",
        "evaluation_mode": "open_loop",
        "source": source,
        "num_windows": estimated_windows,
        "num_rows": len(rows),
        "mean_hit_rate": sum(row.hit_rate for row in rows) / len(rows),
        "mean_selected_mass_coverage": (
            sum(row.selected_mass_coverage for row in rows) / len(rows)
        ),
        "mean_full_mass_coverage": sum(row.full_mass_coverage for row in rows) / len(rows),
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def closed_loop_eval(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    from dataclasses import asdict
    from pathlib import Path
    from typing import Any

    from pseudoroute.execution.closed_loop import evaluate_closed_loop
    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.closed_loop import write_quality_transfer_plot

    config = load_closed_loop_config(config_path)
    if output_dir is None:
        raise SystemExit("closed-loop-eval requires --output-dir")
    run_count = (
        len(config.grid.horizons) * len(config.grid.budget_ratios) * len(config.grid.policies)
    )
    estimate = {
        "command": "closed-loop-eval",
        "information_regime": "oracle",
        "evaluation_mode": "closed_loop",
        "model_scope": "tiny_only",
        "estimated_runs": run_count,
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed closed-loop run already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny_config = load_config(config.tiny_model_config)
    seed_everything(config.experiment.seed)
    device = resolve_device(tiny_config.model.device)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    prompt = torch.tensor([config.prompt_tokens], dtype=torch.long, device=device)
    summaries = []
    route_rows = []
    window_rows = []
    for horizon in config.grid.horizons:
        for ratio in config.grid.budget_ratios:
            for policy in config.grid.policies:
                summary, routes, windows = evaluate_closed_loop(
                    adapter,
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    horizon=horizon,
                    budget_ratio=ratio,
                    gamma=config.grid.gamma,
                    policy_name=policy,
                )
                summaries.append(summary)
                route_rows.extend(routes)
                window_rows.extend(windows)

    def write_table(path: Path, rows: list[Any]) -> None:
        dictionaries = [asdict(row) for row in rows]
        if not dictionaries:
            raise ValueError(f"cannot write empty table: {path.name}")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)

    write_table(root / "closed_loop_summary.csv", list(summaries))
    write_table(root / "closed_loop_routes.csv", list(route_rows))
    write_table(root / "closed_loop_windows.csv", list(window_rows))
    write_quality_transfer_plot(root / "quality_vs_transfer.svg", summaries)
    lossless = {
        (row.horizon, row.budget_ratio): row
        for row in summaries
        if row.policy == "lossless_fallback"
    }
    qualifying = []
    for row in summaries:
        if not row.policy.startswith("masked_"):
            continue
        reference = lossless[(row.horizon, row.budget_ratio)]
        reduction = (
            1.0 - row.total_transfer_bytes / reference.total_transfer_bytes
            if reference.total_transfer_bytes
            else 0.0
        )
        if (
            row.exact_token_rate >= config.gate_a.minimum_exact_token_rate
            and reduction >= config.gate_a.minimum_transfer_reduction
        ):
            qualifying.append(
                {
                    "policy": row.policy,
                    "horizon": row.horizon,
                    "budget_ratio": row.budget_ratio,
                    "exact_token_rate": row.exact_token_rate,
                    "transfer_reduction_vs_lossless": reduction,
                }
            )
    metrics = {
        "schema_version": 1,
        "information_regime": "oracle",
        "evaluation_mode": "closed_loop",
        "model_scope": "tiny_only",
        "num_runs": len(summaries),
        "gate_a": {
            "passed": bool(qualifying),
            "criterion": {
                "minimum_exact_token_rate": config.gate_a.minimum_exact_token_rate,
                "minimum_transfer_reduction_vs_same_grid_lossless": (
                    config.gate_a.minimum_transfer_reduction
                ),
            },
            "qualifying_regions": qualifying,
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def dapq_factorial(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    from dataclasses import asdict
    from pathlib import Path
    from typing import Any, cast

    from pseudoroute.analysis.dapq_factorial import (
        add_position_dominance,
        capture_factorial,
        compare_capture,
        paired_bootstrap,
    )
    from pseudoroute.analysis.pseudo_sequences import (
        FactorialCondition,
        OfflineDocument,
        OfflinePseudoSequence,
        build_factorial_sequences,
    )
    from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.factorial import (
        write_metric_horizon_plot,
        write_position_dominance_heatmap,
    )

    config = load_factorial_config(config_path)
    if output_dir is None:
        raise SystemExit("dapq-factorial requires --output-dir")
    contexts = 2 if config.context_swap else 1
    estimated_examples = len(config.documents) * len(config.position_offsets) * 4 * contexts
    estimate = {
        "command": "dapq-factorial",
        "information_regime": "offline_teacher_forced",
        "model_scope": "tiny" if config.tiny_model_config is not None else "pinned_hf",
        "estimated_examples": estimated_examples,
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed factorial run already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(config.experiment.seed)
    if config.tiny_model_config is not None:
        tiny_config = load_config(config.tiny_model_config)
        device = resolve_device(tiny_config.model.device)
        adapter: TinyMoEAdapter | HFMixtralAdapter = TinyMoEAdapter(
            TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device)
        )
        vocab_size = tiny_config.model.vocab_size
        model_scope = "tiny_only"
        capture_support = {
            "pre_rope_query": True,
            "post_rope_query": True,
            "post_attention_state": True,
            "router_input": True,
            "router_logits": True,
            "topk": True,
        }
    else:
        if config.hf_trace_config is None:
            raise AssertionError("validated factorial config is missing a model source")
        trace_config = load_trace_config(config.hf_trace_config)
        device = resolve_device(trace_config.model.device)
        adapter = HFMixtralAdapter.from_pretrained(
            trace_config.model.model_id,
            revision=trace_config.model.revision,
            device=device,
            cache_dir=trace_config.model.cache_dir,
            local_files_only=True,
        )
        vocab_size = int(cast(Any, adapter.model).config.vocab_size)
        model_scope = "pinned_hf_router_level"
        capture_support = {
            "pre_rope_query": True,
            "post_rope_query": False,
            "post_attention_state": True,
            "router_input": True,
            "router_logits": True,
            "topk": True,
        }
    documents = tuple(
        OfflineDocument(f"tiny-{index}", torch.tensor(values, device=device))
        for index, values in enumerate(config.documents)
    )
    all_sequences: list[OfflinePseudoSequence] = []
    for offset in config.position_offsets:
        all_sequences.extend(
            build_factorial_sequences(
                documents,
                boundary=config.boundary,
                horizon=max(config.horizons),
                vocab_size=vocab_size,
                content_seed=config.content_seed,
                position_offset=offset,
                include_context_swap=config.context_swap,
            )
        )
    captures = [capture_factorial(adapter, sequence) for sequence in all_sequences]
    targets = {
        (capture.sequence.sample_id, capture.sequence.position_offset): capture
        for capture in captures
        if capture.sequence.condition is FactorialCondition.SC_SP
        and not capture.sequence.context_swapped
    }
    metric_rows = []
    for capture in captures:
        target = targets[(capture.sequence.sample_id, capture.sequence.position_offset)]
        metric_rows.extend(
            compare_capture(
                capture,
                target,
                horizons=config.horizons,
                top_k=next(iter(adapter.spec.top_k_by_layer.values())),
            )
        )
    metric_rows = add_position_dominance(metric_rows)
    bootstrap_rows = paired_bootstrap(
        metric_rows,
        bootstrap_samples=config.bootstrap.samples,
        confidence=config.bootstrap.confidence,
        seed=config.experiment.seed,
    )

    def write_table(path: Path, rows: list[Any]) -> None:
        dictionaries = [asdict(row) if not isinstance(row, dict) else row for row in rows]
        if not dictionaries:
            raise ValueError(f"cannot write empty table: {path.name}")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)

    example_rows = [
        {
            "information_regime": "offline_teacher_forced",
            "sample_id": sequence.sample_id,
            "condition": sequence.condition.value,
            "context_swapped": sequence.context_swapped,
            "boundary": sequence.boundary,
            "horizon": sequence.horizon,
            "content_seed": sequence.content_seed,
            "position_offset": sequence.position_offset,
            "input_ids": json.dumps(sequence.input_ids.reshape(-1).tolist()),
            "position_ids": json.dumps(sequence.position_ids.reshape(-1).tolist()),
            "true_future_ids": json.dumps(sequence.true_future_ids.reshape(-1).tolist()),
        }
        for sequence in all_sequences
    ]
    write_table(root / "factorial_examples.csv", example_rows)
    write_table(root / "factorial_metrics.csv", list(metric_rows))
    write_table(root / "factorial_bootstrap.csv", list(bootstrap_rows))
    write_position_dominance_heatmap(root / "position_dominance_heatmap.svg", bootstrap_rows)
    write_metric_horizon_plot(root / "router_logit_horizon.svg", bootstrap_rows)
    scsp_rows = [row for row in metric_rows if row.condition == "SC_SP" and not row.context_swapped]
    exact_metrics = {
        "logit_mse": all(row.value == 0.0 for row in scsp_rows if row.metric == "logit_mse"),
        "topk_recall": all(row.value == 1.0 for row in scsp_rows if row.metric == "topk_recall"),
        "router_logit_cosine": all(
            abs(row.value - 1.0) < 1e-2 for row in scsp_rows if row.metric == "router_logit_cosine"
        ),
    }
    gate_candidates = [
        row
        for row in bootstrap_rows
        if row.condition == "POSITION_DOMINANCE"
        and not row.context_swapped
        and row.metric in {"router_logit_cosine", "topb_window_utility_recall"}
    ]
    positive = [row for row in gate_candidates if row.mean > 0]
    significant = [row for row in gate_candidates if row.ci_lower > 0]
    gate_passed = bool(gate_candidates) and (
        len(positive) / len(gate_candidates) >= 0.5 and bool(significant)
    )
    metrics = {
        "schema_version": 1,
        "information_regime": "offline_teacher_forced",
        "model_scope": model_scope,
        "num_examples": len(all_sequences),
        "num_metric_rows": len(metric_rows),
        "num_bootstrap_rows": len(bootstrap_rows),
        "scsp_teacher_forced_validation": exact_metrics,
        "gate_b": {
            "passed_for_configured_model": gate_passed,
            "criterion": (
                "at least half of router-logit/subset position-dominance cells have "
                "positive means and at least one paired confidence interval excludes zero"
            ),
            "positive_cells": len(positive),
            "significant_positive_cells": len(significant),
            "total_cells": len(gate_candidates),
        },
        "capture_support": capture_support,
        "scsp_cosine_absolute_tolerance": 1e-2,
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def analyze_router(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    from dataclasses import asdict
    from pathlib import Path
    from typing import Any

    from safetensors.torch import save_file

    from pseudoroute.analysis.expert_criticality import run_sampled_interventions
    from pseudoroute.analysis.router_geometry import (
        BoundaryRow,
        CoordinateRow,
        ExpertGeometryRow,
        TokenGeometryRow,
        covariance_aware_logit_variance,
        empirical_covariance_sketch,
        exact_router_svd,
        pairwise_boundaries,
        randomized_router_svd,
        row_space_coordinates,
        svd_metric_rows,
        token_geometry,
    )
    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.router import (
        write_criticality_plot,
        write_margin_histogram,
        write_singular_value_plot,
    )

    config = load_router_analysis_config(config_path)
    if output_dir is None:
        raise SystemExit("analyze-router requires --output-dir")
    estimate = {
        "command": "analyze-router",
        "information_regime": "offline_teacher_forced",
        "model_scope": "tiny_only",
        "documents": len(config.documents),
        "tokens": sum(len(document) for document in config.documents),
        "maximum_intervention_rows": (
            config.interventions.max_targets * len(config.interventions.kinds)
        ),
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed router analysis already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny_config = load_config(config.tiny_model_config)
    seed_everything(config.experiment.seed)
    device = resolve_device(tiny_config.model.device)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    if config.top_b > tiny_config.model.num_experts:
        raise ValueError("top_b cannot exceed the number of experts")
    samples = tuple(
        (f"tiny-{index}", torch.tensor([document], dtype=torch.long, device=device))
        for index, document in enumerate(config.documents)
    )
    activations_by_layer: dict[int, list[tuple[str, Any]]] = {
        layer: [] for layer in adapter.spec.moe_layer_indices
    }
    with torch.inference_mode():
        for sample_id, tokens in samples:
            output = adapter.model(tokens, capture_trace=True, capture_activations=True)
            for activation in output.activations:
                activations_by_layer[activation.layer_idx].append((sample_id, activation))

    svd_rows = []
    coordinate_rows: list[CoordinateRow] = []
    expert_rows: list[ExpertGeometryRow] = []
    boundary_rows: list[BoundaryRow] = []
    token_rows: list[TokenGeometryRow] = []
    sketch_tensors = {}
    for layer_idx in adapter.spec.moe_layer_indices:
        layer = adapter._layer(layer_idx)
        weight = layer.router.weight.detach().cpu()
        bias = layer.router.bias.detach().cpu() if layer.router.bias is not None else None
        exact = exact_router_svd(weight)
        randomized_rank = min(config.randomized_svd.rank, min(weight.shape))
        randomized = randomized_router_svd(
            weight,
            rank=randomized_rank,
            oversample=config.randomized_svd.oversample,
            power_iterations=config.randomized_svd.power_iterations,
            seed=config.experiment.seed + layer_idx,
        )
        svd_rows.extend(svd_metric_rows(layer_idx, weight, (exact, randomized)))
        records = activations_by_layer[layer_idx]
        states = torch.cat(
            [
                activation.router_input.cpu()
                for _, activation in records
                if activation.router_input is not None
            ]
        )
        logits = torch.cat([activation.router_logits.cpu() for _, activation in records])
        topk_ids = torch.cat([activation.topk_ids.cpu() for _, activation in records])
        probabilities = logits.double().softmax(dim=-1)
        sketch = empirical_covariance_sketch(states, rank=config.covariance_rank)
        sketch_tensors[f"layer_{layer_idx}_mean"] = sketch.mean.contiguous()
        sketch_tensors[f"layer_{layer_idx}_covariance"] = sketch.covariance.contiguous()
        sketch_tensors[f"layer_{layer_idx}_components"] = sketch.components.contiguous()
        sketch_tensors[f"layer_{layer_idx}_eigenvalues"] = sketch.eigenvalues.contiguous()
        variances = covariance_aware_logit_variance(weight, sketch.covariance)
        for expert_idx in range(weight.shape[0]):
            expert_rows.append(
                ExpertGeometryRow(
                    layer_idx,
                    expert_idx,
                    float(torch.linalg.vector_norm(weight[expert_idx].double())),
                    float(variances[expert_idx]),
                    float((topk_ids == expert_idx).any(dim=-1).double().mean()),
                    float(probabilities[:, expert_idx].mean()),
                )
            )
        boundaries = pairwise_boundaries(
            states, topk_ids, weight, bias, epsilon=config.boundary_epsilon
        )
        boundary_rows.extend(BoundaryRow(layer_idx, *values) for values in boundaries)
        entropy, top1_gap, margins, cumulative_mass = token_geometry(
            probabilities,
            logits.double(),
            top_k=adapter.spec.top_k_by_layer[layer_idx],
            top_b=config.top_b,
        )
        coordinates = row_space_coordinates(states, exact.vh)
        for row_index, (sample_id, activation) in enumerate(records):
            token_rows.append(
                TokenGeometryRow(
                    sample_id,
                    activation.token_position,
                    layer_idx,
                    float(entropy[row_index]),
                    float(top1_gap[row_index]),
                    float(margins[row_index]),
                    float(cumulative_mass[row_index]),
                )
            )
            coordinate_rows.extend(
                CoordinateRow(
                    sample_id, activation.token_position, layer_idx, coordinate_idx, float(value)
                )
                for coordinate_idx, value in enumerate(coordinates[row_index])
            )
    save_file(sketch_tensors, str(root / "covariance_sketch.safetensors"))
    intervention_rows = run_sampled_interventions(
        adapter,
        samples,
        max_targets=config.interventions.max_targets,
        seed=config.experiment.seed,
        interventions=config.interventions.kinds,
    )

    def write_table(path: Path, rows: list[Any]) -> None:
        dictionaries = [asdict(row) for row in rows]
        if not dictionaries:
            raise ValueError(f"cannot write empty table: {path.name}")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)

    write_table(root / "svd.csv", list(svd_rows))
    write_table(root / "router_coordinates.csv", list(coordinate_rows))
    write_table(root / "expert_geometry.csv", list(expert_rows))
    write_table(root / "pairwise_boundaries.csv", list(boundary_rows))
    write_table(root / "token_geometry.csv", list(token_rows))
    write_table(root / "expert_interventions.csv", list(intervention_rows))
    write_singular_value_plot(root / "singular_values.svg", svd_rows)
    write_margin_histogram(root / "topk_margins.svg", token_rows)
    write_criticality_plot(root / "expert_criticality.svg", intervention_rows)
    metrics = {
        "schema_version": 1,
        "information_regime": "offline_teacher_forced",
        "model_scope": "tiny_only",
        "layers": len(adapter.spec.moe_layer_indices),
        "token_layer_rows": len(token_rows),
        "source_tokens": sum(tokens.numel() for _, tokens in samples),
        "intervention_rows": len(intervention_rows),
        "mean_route_entropy": sum(row.route_entropy for row in token_rows) / len(token_rows),
        "mean_topk_margin": sum(row.topk_margin for row in token_rows) / len(token_rows),
        "mean_intervention_next_token_kl": (
            sum(row.next_token_kl for row in intervention_rows) / len(intervention_rows)
        ),
        "mean_downstream_topk_flip_rate": (
            sum(row.downstream_topk_flip_rate for row in intervention_rows) / len(intervention_rows)
        ),
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _predictor_documents(
    config: PredictorTrainConfig, device: str
) -> tuple[tuple[str, torch.Tensor], ...]:
    raw_documents = config.documents
    return tuple(
        (
            f"document-{index}",
            torch.tensor([values], dtype=torch.long, device=device),
        )
        for index, values in enumerate(raw_documents)
    )


def train_predictor(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    import time
    from dataclasses import asdict
    from pathlib import Path

    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.probes.learned import save_learned_probe
    from pseudoroute.training.dataset import build_predictor_examples, save_predictor_dataset
    from pseudoroute.training.evaluate import fit_markov_transitions, save_markov_probe
    from pseudoroute.training.train import train_all

    config = load_predictor_train_config(config_path)
    if output_dir is None:
        raise SystemExit("train-predictor requires --output-dir")
    estimated_rows = sum(
        sum(
            max(0, len(document) - config.history_length - horizon + 1)
            for horizon in config.horizons
        )
        for document in config.documents
    )
    estimate = {
        "command": "train-predictor",
        "information_regime": "offline_teacher_forced",
        "documents": len(config.documents),
        "estimated_rows": estimated_rows,
        "predictors": [
            "direct_linear",
            "direct_ridge",
            "direct_mlp",
            "adept_style_rf",
            "markov_transition",
        ],
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed predictor run already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny_config = load_config(config.tiny_model_config)
    device = resolve_device(tiny_config.model.device)
    seed_everything(config.experiment.seed)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    documents = _predictor_documents(config, device)
    calibration_started = time.perf_counter()
    dataset, _ = build_predictor_examples(
        adapter,
        documents,
        horizons=config.horizons,
        history_length=config.history_length,
        train_fraction=config.split.train_fraction,
        validation_fraction=config.split.validation_fraction,
        seed=config.experiment.seed,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    save_predictor_dataset(root / "dataset", dataset)
    learned, costs = train_all(
        dataset,
        ridge_lambda=config.predictors.ridge_lambda,
        mlp_hidden=config.predictors.mlp_hidden,
        mlp_epochs=config.predictors.mlp_epochs,
        learning_rate=config.predictors.learning_rate,
        rf_trees=config.predictors.rf_trees,
        rf_feature_candidates=config.predictors.rf_feature_candidates,
        seed=config.experiment.seed,
    )
    for probe in learned:
        save_learned_probe(root / "models" / probe.name, probe)
    train_ids = frozenset(row.sample_id for row in dataset.rows if row.split == "train")
    markov_started = time.perf_counter()
    transitions = fit_markov_transitions(
        adapter,
        documents,
        train_sample_ids=train_ids,
        smoothing=config.predictors.markov_smoothing,
    )
    markov_seconds = time.perf_counter() - markov_started
    from pseudoroute.probes.heuristics import MarkovTransitionProbe

    markov = MarkovTransitionProbe(transitions)
    save_markov_probe(root / "models" / markov.name, markov)
    cost_rows = [asdict(cost) for cost in costs]
    cost_rows.append(
        {
            "probe_name": "markov_transition",
            "calibration_rows": len(dataset.indices("train")),
            "training_seconds": markov_seconds,
            "epochs": 1,
            "approximate_flops": 0,
            "parameter_count": transitions.numel(),
        }
    )
    with (root / "training_cost.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)
    summary = {
        **estimate,
        "calibration_seconds": calibration_seconds,
        "calibration_tokens": sum(len(document) for document in config.documents),
        "rows": len(dataset.rows),
        "split_documents": {
            split: sorted({row.sample_id for row in dataset.rows if row.split == split})
            for split in ("train", "validation", "test")
        },
        "training_cost": cost_rows,
        "safe_serialization": "JSON manifests plus safetensors; no pickle",
    }
    (root / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def evaluate_probe_command(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "default_vectors_dir" in raw:
        return evaluate_shadow_probes(config_path, output_dir=output_dir, dry_run=dry_run)
    import csv
    from dataclasses import asdict

    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.probes import write_quality_latency_plot
    from pseudoroute.probes.heuristics import (
        CurrentRouteProbe,
        RollingFrequencyProbe,
        RollingMassProbe,
    )
    from pseudoroute.probes.learned import load_learned_probe
    from pseudoroute.training.dataset import build_predictor_examples, load_predictor_dataset
    from pseudoroute.training.evaluate import evaluate_probe, finalize_equal_cost, load_markov_probe

    config = load_probe_evaluation_config(config_path)
    if output_dir is None:
        raise SystemExit("evaluate-probe requires --output-dir")
    estimate = {
        "command": "evaluate-probe",
        "information_regime": "online_pre_sample",
        "probe_count": 9,
        "equal_budget_experts": config.top_b,
        "latency_repetitions": config.latency_repetitions,
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed probe evaluation already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    train_config = load_predictor_train_config(config.training_config)
    if train_config.experiment.seed != config.experiment.seed:
        raise ValueError("evaluation seed must match training seed for an identical grouped split")
    tiny_config = load_config(train_config.tiny_model_config)
    device = resolve_device(tiny_config.model.device)
    seed_everything(config.experiment.seed)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    documents = _predictor_documents(train_config, device)
    rebuilt, states = build_predictor_examples(
        adapter,
        documents,
        horizons=train_config.horizons,
        history_length=train_config.history_length,
        train_fraction=train_config.split.train_fraction,
        validation_fraction=train_config.split.validation_fraction,
        seed=train_config.experiment.seed,
    )
    predictor_root = Path(config.predictor_dir)
    stored = load_predictor_dataset(predictor_root / "dataset")
    if (
        stored.rows != rebuilt.rows
        or not torch.equal(stored.targets, rebuilt.targets)
        or not torch.equal(stored.features, rebuilt.features)
    ):
        raise ValueError(
            "evaluation data does not exactly match the persisted common oracle target"
        )
    learned_names = ("direct_linear", "direct_ridge", "direct_mlp", "adept_style_rf")
    probes = [
        CurrentRouteProbe(),
        RollingFrequencyProbe(config.rolling_lookback, config.rolling_decay),
        RollingMassProbe(config.rolling_lookback, config.rolling_decay),
        load_markov_probe(predictor_root / "models" / "markov_transition"),
        *(load_learned_probe(predictor_root / "models" / name) for name in learned_names),
    ]
    raw_results = []
    for probe in probes:
        result, _ = evaluate_probe(
            probe,
            stored,
            states,
            split="test",
            top_b=config.top_b,
            latency_repetitions=config.latency_repetitions,
        )
        raw_results.append(result)
    rows = finalize_equal_cost(raw_results)
    write_quality_latency_plot(root / "quality_vs_latency.svg", rows)
    dictionaries = [asdict(row) for row in rows]
    with (root / "probe_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    summary = {
        **estimate,
        "common_target": "aggregate selected-routing mass over the exact future window",
        "target_fingerprint_verified": True,
        "test_documents": sorted({row.sample_id for row in stored.rows if row.split == "test"}),
        "equal_cost_ceiling_us": rows[0].equal_cost_ceiling_us,
        "results": dictionaries,
    }
    (root / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_default_vectors(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import time
    from pathlib import Path

    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.probes.default_vectors import (
        calibrate_default_vectors,
        same_token_next_layer_router_error,
        save_default_vectors,
    )

    config = load_default_vector_config(config_path)
    if output_dir is None:
        raise SystemExit("build-default-vectors requires --output-dir")
    estimate = {
        "command": "build-default-vectors",
        "information_regime": "offline_teacher_forced",
        "documents": len(config.documents),
        "calibration_tokens": sum(len(document) for document in config.documents),
        "definition": "expert_output",
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed default-vector run already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny_config = load_config(config.tiny_model_config)
    device = resolve_device(tiny_config.model.device)
    seed_everything(config.experiment.seed)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    documents = _predictor_documents(config, device)  # type: ignore[arg-type]
    started = time.perf_counter()
    store = calibrate_default_vectors(adapter, documents)
    calibration_seconds = time.perf_counter() - started
    save_default_vectors(root, store)
    errors = [
        error
        for _, tokens in documents
        for error in same_token_next_layer_router_error(adapter, tokens, store)
    ]
    metrics = {
        **estimate,
        "calibration_seconds": calibration_seconds,
        "counts": store.count.tolist(),
        "calibration_fingerprint": store.calibration_fingerprint,
        "same_token_next_layer_router_rmse_mean": sum(errors) / len(errors),
        "same_token_next_layer_router_rmse": errors,
        "safe_serialization": "JSON manifest plus safetensors",
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def evaluate_shadow_probes(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    from dataclasses import asdict, replace
    from pathlib import Path

    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.probes import write_quality_latency_plot, write_subset_regret_plot
    from pseudoroute.probes.default_vectors import (
        DefaultVectorShadowRolloutProbe,
        load_default_vectors,
    )
    from pseudoroute.probes.heuristics import CurrentRouteProbe
    from pseudoroute.probes.pseudo import (
        FuturePositionRephasedProbe,
        PseudoTokenProbe,
        UncertaintyEnsembleProbe,
    )
    from pseudoroute.training.dataset import build_predictor_examples, load_predictor_dataset
    from pseudoroute.training.evaluate import evaluate_probe, finalize_equal_cost
    from pseudoroute.types import InformationRegime

    config = load_shadow_probe_config(config_path)
    if output_dir is None:
        raise SystemExit("evaluate-probe requires --output-dir")
    estimate = {
        "command": "evaluate-probe",
        "milestone": "M7",
        "information_regime": config.experiment.information_regime,
        "probe_count": 10,
        "anchors": list(config.anchors),
        "budget_sweep": list(range(1, config.top_b + 1)),
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed shadow-probe evaluation already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    train_config = load_predictor_train_config(config.training_config)
    tiny_config = load_config(train_config.tiny_model_config)
    device = resolve_device(tiny_config.model.device)
    seed_everything(config.experiment.seed)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    documents = _predictor_documents(train_config, device)
    rebuilt, base_states = build_predictor_examples(
        adapter,
        documents,
        horizons=train_config.horizons,
        history_length=train_config.history_length,
        train_fraction=train_config.split.train_fraction,
        validation_fraction=train_config.split.validation_fraction,
        seed=train_config.experiment.seed,
    )
    dataset = load_predictor_dataset(Path(config.predictor_dataset_dir))
    if dataset.rows != rebuilt.rows or not torch.equal(dataset.targets, rebuilt.targets):
        raise ValueError("M7 evaluation does not match the persisted M6 common oracle target")
    states = base_states
    if config.experiment.information_regime == "online_post_sample":
        by_id = {sample_id: tokens for sample_id, tokens in documents}
        states = tuple(
            replace(
                state,
                information_regime=InformationRegime.ONLINE_POST_SAMPLE,
                next_token_id=int(by_id[row.sample_id][0, row.boundary_position + 1]),
            )
            for state, row in zip(base_states, dataset.rows, strict=True)
        )
    defaults = load_default_vectors(Path(config.default_vectors_dir))
    content: Literal["current_token", "sampled_next_token"] = (
        "sampled_next_token"
        if config.experiment.information_regime == "online_post_sample"
        else "current_token"
    )
    correct_rephased = FuturePositionRephasedProbe(adapter, "correct")
    random_rephased = FuturePositionRephasedProbe(
        adapter, "correct", random_content=True, seed=config.experiment.seed
    )
    probes = [
        CurrentRouteProbe(),
        correct_rephased,
        FuturePositionRephasedProbe(adapter, "no_rephase"),
        FuturePositionRephasedProbe(adapter, "wrong_position", config.wrong_position_offset),
        random_rephased,
        PseudoTokenProbe(adapter, "independent", content, config.fixed_token_id),
        PseudoTokenProbe(adapter, "causal", content, config.fixed_token_id),
        DefaultVectorShadowRolloutProbe(adapter, defaults, content, "independent"),
        DefaultVectorShadowRolloutProbe(adapter, defaults, content, "causal"),
        UncertaintyEnsembleProbe((correct_rephased, random_rephased)),
    ]
    raw_results = []
    for budget in range(1, config.top_b + 1):
        for probe in probes:
            result, _ = evaluate_probe(
                probe,
                dataset,
                states,
                split="test",
                top_b=budget,
                latency_repetitions=config.latency_repetitions,
                anchor_candidates=config.anchors,
            )
            raw_results.append(result)
    rows = finalize_equal_cost(raw_results)
    dictionaries = [asdict(row) for row in rows]
    with (root / "probe_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    maximum_budget_rows = [row for row in rows if row.equal_budget_experts == config.top_b]
    write_quality_latency_plot(root / "quality_vs_latency.svg", maximum_budget_rows)
    write_subset_regret_plot(root / "subset_regret_curves.svg", rows)
    metrics = {
        **estimate,
        "common_target": "aggregate selected-routing mass over the exact future window",
        "target_fingerprint_verified": True,
        "equal_cost_ceiling_us": rows[0].equal_cost_ceiling_us,
        "results": dictionaries,
        "default_vector_definition": defaults.definition,
        "calibration_fingerprint": defaults.calibration_fingerprint,
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def simulate_trace_offload(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    from dataclasses import asdict

    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.simulator import write_simulator_pareto
    from pseudoroute.simulation.engine import RouteToken, SimHardware, simulate
    from pseudoroute.types import ExpertKey

    config = load_offload_simulator_config(config_path)
    if output_dir is None:
        raise SystemExit("simulate-offload requires --output-dir")
    scenario_count = (
        len(config.baselines)
        * len(config.sweep.capacity_experts)
        * len(config.sweep.bandwidth_bytes_per_s)
        * len(config.sweep.fixed_latency_us)
        * len(config.sweep.horizons)
        * len(config.sweep.overlap_transfers)
    )
    estimate = {
        "command": "simulate-offload",
        "milestone": "M8",
        "simulated": True,
        "information_regime": "offline_teacher_forced",
        "scenarios": scenario_count,
        "route_tokens": config.generated_tokens,
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed M8 simulation exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny_config = load_config(config.tiny_model_config)
    device = resolve_device(tiny_config.model.device)
    seed_everything(config.experiment.seed)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    trajectory = torch.tensor([config.prompt_tokens], dtype=torch.long, device=device)
    route_tokens = []
    trace_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for token_index in range(config.generated_tokens):
            output = adapter.model(trajectory, capture_trace=True)
            position = trajectory.shape[1] - 1
            traces = sorted(
                (trace for trace in output.traces if trace.token_position == position),
                key=lambda trace: trace.layer_idx,
            )
            by_layer = tuple(
                tuple(
                    ExpertKey(trace.layer_idx, int(expert))
                    for expert in trace.topk_ids.reshape(-1).tolist()
                )
                for trace in traces
            )
            route_tokens.append(RouteToken(token_index, by_layer))
            for trace in traces:
                trace_rows.append(
                    {
                        "token_index": token_index,
                        "token_position": position,
                        "layer_idx": trace.layer_idx,
                        "topk_ids": json.dumps(trace.topk_ids.reshape(-1).tolist()),
                    }
                )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            trajectory = torch.cat((trajectory, token), dim=1)
    summaries: list[dict[str, object]] = []
    timeline_rows: list[dict[str, object]] = []
    hardware_rows: list[dict[str, object]] = []
    scenario_id = 0
    expert_size = tiny_config.model.synthetic_expert_bytes
    for baseline in config.baselines:
        for capacity in config.sweep.capacity_experts:
            for bandwidth in config.sweep.bandwidth_bytes_per_s:
                for latency in config.sweep.fixed_latency_us:
                    for horizon in config.sweep.horizons:
                        for overlap in config.sweep.overlap_transfers:
                            scenario_id += 1
                            hardware = SimHardware(
                                capacity * expert_size,
                                bandwidth,
                                latency,
                                config.max_concurrent_transfers,
                                config.dense_compute_us_per_token,
                                config.expert_compute_us,
                                overlap,
                            )
                            summary, events = simulate(
                                tuple(route_tokens),
                                adapter.spec.expert_bytes,
                                hardware,
                                baseline=baseline,
                                horizon=horizon,
                                per_layer_budget=config.per_layer_budget,
                                probe_us=config.probe_us,
                            )
                            dimensions = {
                                "scenario_id": scenario_id,
                                "capacity_experts": capacity,
                                "bandwidth_bytes_per_s": bandwidth,
                                "fixed_latency_us": latency,
                                "horizon": horizon,
                                "overlap_transfers": overlap,
                            }
                            summaries.append({**dimensions, **asdict(summary)})
                            hardware_rows.append({**dimensions, **asdict(hardware)})
                            for event_index, event in enumerate(events):
                                timeline_rows.append(
                                    {
                                        **dimensions,
                                        "event_index": event_index,
                                        "timestamp_us": event.timestamp_us,
                                        "event": event.event,
                                        "expert": (
                                            ""
                                            if event.expert is None
                                            else (
                                                f"{event.expert.layer_idx}:"
                                                f"{event.expert.expert_idx}"
                                            )
                                        ),
                                        "layer_idx": event.layer_idx,
                                        "bytes": event.bytes,
                                        "stream": event.stream,
                                        "reason": event.reason,
                                        "token_index": event.token_index,
                                    }
                                )

    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows(root / "route_trace.csv", trace_rows)
    write_rows(root / "simulation_summary.csv", summaries)
    write_rows(root / "cache_events.csv", timeline_rows)
    write_rows(root / "hardware_sweep.csv", hardware_rows)
    write_simulator_pareto(root / "simulated_pareto.svg", summaries)
    metrics = {
        **estimate,
        "baselines": list(config.baselines),
        "event_rows": len(timeline_rows),
        "simulation_results": summaries,
        "invariants": {
            "capacity_enforced": True,
            "load_completion_enforced": True,
            "aggregate_bandwidth_enforced": True,
            "perfect_and_no_overlap_bounds": True,
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "hardware.json").write_text(
        json.dumps(hardware_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "simulation_results"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def simulate_static_adaptive(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    from dataclasses import asdict

    from pseudoroute.analysis.static_residency import calibrate_static_signals, rankings
    from pseudoroute.execution.adaptive import run_static_adaptive_closed_loop
    from pseudoroute.execution.routing_policy import MaskedSubstitutionPolicy
    from pseudoroute.models.adapters.tiny import TinyMoEAdapter
    from pseudoroute.plotting.static_adaptive import (
        write_simulated_static_adaptive_plot,
        write_static_adaptive_plot,
    )
    from pseudoroute.simulation.engine import RouteToken, SimHardware, simulate
    from pseudoroute.types import ExpertKey

    config = load_static_adaptive_config(config_path)
    if output_dir is None:
        raise SystemExit("simulate-offload requires --output-dir")
    scenario_count = (
        len(config.ranking_methods) * len(config.static_fractions) * len(config.horizons) * 2
    )
    estimate = {
        "command": "simulate-offload",
        "milestone": "M9",
        "information_regime": "oracle",
        "simulation_scope": "m8_event_engine_custom_plan",
        "scenarios": scenario_count,
        "calibration_tokens": sum(len(item) for item in config.calibration_documents),
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed M9 run already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny_config = load_config(config.tiny_model_config)
    device = resolve_device(tiny_config.model.device)
    seed_everything(config.experiment.seed)
    adapter = TinyMoEAdapter(TinyMoE(tiny_config.model, seed=config.experiment.seed, device=device))
    documents = tuple(
        (f"calibration-{index}", torch.tensor([tokens], dtype=torch.long, device=device))
        for index, tokens in enumerate(config.calibration_documents)
    )
    signals = calibrate_static_signals(
        adapter, documents, max_interventions=config.max_interventions, seed=config.experiment.seed
    )
    all_rankings = rankings(signals)
    signal_rows: list[dict[str, object]] = []
    for layer in adapter.spec.moe_layer_indices:
        for expert in range(adapter.spec.num_experts_by_layer[layer]):
            signal_rows.append(
                {
                    "layer_idx": layer,
                    "expert_idx": expert,
                    "frequency": float(signals.frequency[layer, expert]),
                    "covariance_variance": float(signals.covariance_variance[layer, expert]),
                    "quality_sensitivity": float(signals.quality_sensitivity[layer, expert]),
                    "downstream_influence": float(signals.downstream_influence[layer, expert]),
                    "miss_risk": float(signals.miss_risk[layer, expert]),
                    "combined": float(signals.combined[layer, expert]),
                }
            )
    prompt = torch.tensor([config.prompt_tokens], dtype=torch.long, device=device)
    summary_rows: list[dict[str, object]] = []
    simulation_rows: list[dict[str, object]] = []
    replan_rows: list[dict[str, object]] = []
    termination_rows: list[dict[str, object]] = []
    timeline_rows: list[dict[str, object]] = []
    scenario_id = 0
    for method in config.ranking_methods:
        for fraction in config.static_fractions:
            for horizon in config.horizons:
                for adaptive in (False, True):
                    scenario_id += 1
                    summary, plans, terminations = run_static_adaptive_closed_loop(
                        adapter,
                        prompt,
                        max_new_tokens=config.max_new_tokens,
                        horizon=horizon,
                        total_budget=config.total_experts_per_layer,
                        static_fraction=fraction,
                        static_method=method,
                        ranking=all_rankings[method],
                        adaptive=adaptive,
                        threshold=config.out_of_subset_threshold,
                    )
                    summary_rows.append({"scenario_id": scenario_id, **asdict(summary)})
                    for plan_index, plan in enumerate(plans):
                        replan_rows.append(
                            {
                                "scenario_id": scenario_id,
                                "replan_index": plan_index,
                                "mode": summary.mode,
                                "boundary": plan.boundary,
                                "reason": plan.reason,
                                "planned_horizon": plan.planned_horizon,
                                "static_experts": json.dumps(
                                    [[key.layer_idx, key.expert_idx] for key in plan.static_experts]
                                ),
                                "dynamic_experts": json.dumps(
                                    [
                                        [key.layer_idx, key.expert_idx]
                                        for key in plan.dynamic_experts
                                    ]
                                ),
                                "load_delta": json.dumps(
                                    [[key.layer_idx, key.expert_idx] for key in plan.load_delta]
                                ),
                                "eviction_delta": json.dumps(
                                    [[key.layer_idx, key.expert_idx] for key in plan.eviction_delta]
                                ),
                            }
                        )
                    termination_rows.extend(
                        {"scenario_id": scenario_id, **asdict(event)} for event in terminations
                    )
                    generated = json.loads(summary.generated_tokens)
                    authoritative = prompt.clone()
                    policy_schedule: dict[int, frozenset[ExpertKey]] = {}
                    plan_by_boundary = {plan.boundary: plan for plan in plans}
                    prefetch_schedule = {
                        plan.boundary - len(config.prompt_tokens): plan.allowed for plan in plans
                    }
                    route_tokens = []
                    policy = MaskedSubstitutionPolicy()
                    active_plan = plans[0]
                    for token_index, token_id in enumerate(generated):
                        boundary = int(authoritative.shape[1])
                        if boundary in plan_by_boundary:
                            active_plan = plan_by_boundary[boundary]
                        position = boundary - 1
                        policy_schedule[position] = active_plan.allowed
                        executed = adapter.forward_with_policy(
                            authoritative,
                            policy,
                            allowed_experts_by_position=policy_schedule,
                        )
                        records = sorted(
                            (
                                record
                                for record in executed.executed_routes
                                if record.token_position == position
                            ),
                            key=lambda record: record.layer_idx,
                        )
                        route_tokens.append(
                            RouteToken(
                                token_index,
                                tuple(
                                    tuple(
                                        ExpertKey(record.layer_idx, int(expert))
                                        for expert in record.executed_topk_ids.reshape(-1).tolist()
                                    )
                                    for record in records
                                ),
                            )
                        )
                        authoritative = torch.cat(
                            (
                                authoritative,
                                torch.tensor([[token_id]], device=device, dtype=torch.long),
                            ),
                            dim=1,
                        )
                    hardware = SimHardware(
                        config.total_experts_per_layer
                        * len(adapter.spec.moe_layer_indices)
                        * tiny_config.model.synthetic_expert_bytes,
                        config.hardware.bandwidth_bytes_per_us * 1_000_000,
                        config.hardware.fixed_latency_us,
                        config.hardware.max_concurrent_transfers,
                        config.hardware.compute_us_per_token,
                        config.hardware.expert_compute_us,
                        config.hardware.overlap_transfers,
                    )
                    simulated, timeline = simulate(
                        tuple(route_tokens),
                        adapter.spec.expert_bytes,
                        hardware,
                        baseline="custom_plan",
                        horizon=horizon,
                        per_layer_budget=config.total_experts_per_layer,
                        probe_us=config.hardware.probe_us,
                        prefetch_schedule=prefetch_schedule,
                    )
                    simulation_rows.append(
                        {"scenario_id": scenario_id, "mode": summary.mode, **asdict(simulated)}
                    )
                    timeline_rows.extend(
                        {
                            "scenario_id": scenario_id,
                            "mode": summary.mode,
                            **asdict(event),
                        }
                        for event in timeline
                    )

    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("scenario_id\n", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows(root / "static_rankings.csv", signal_rows)
    write_rows(root / "closed_loop_summary.csv", summary_rows)
    write_rows(root / "replan_events.csv", replan_rows)
    write_rows(root / "termination_events.csv", termination_rows)
    write_rows(root / "simulation_summary.csv", simulation_rows)
    write_rows(root / "simulation_timeline.csv", timeline_rows)
    write_static_adaptive_plot(root / "fixed_vs_adaptive.svg", summary_rows)
    write_simulated_static_adaptive_plot(root / "fixed_vs_adaptive_simulated.svg", simulation_rows)
    metrics = {
        **estimate,
        "hardware": config.hardware.model_dump(mode="json"),
        "termination_events": len(termination_rows),
        "replan_events": len(replan_rows),
        "ranking_methods": list(config.ranking_methods),
        "static_fractions": list(config.static_fractions),
        "horizons": list(config.horizons),
        "results": summary_rows,
        "simulation_results": simulation_rows,
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    compact = {
        key: value for key, value in metrics.items() if key not in {"results", "simulation_results"}
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


def simulate_offload(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "baselines" in raw:
        return simulate_trace_offload(config_path, output_dir=output_dir, dry_run=dry_run)
    return simulate_static_adaptive(config_path, output_dir=output_dir, dry_run=dry_run)


def benchmark_offload(config_path: str, *, output_dir: str | None, dry_run: bool) -> int:
    import csv
    import platform
    import statistics
    from dataclasses import asdict

    from pseudoroute.runtime.offload_engine import TinyMoEOffloadEngine
    from pseudoroute.types import ExpertKey

    config = load_offload_runtime_config(config_path)
    estimate = {
        "command": "benchmark-offload",
        "milestone": "M10",
        "runtime": "real_cuda_offload",
        "model_scope": "deterministic_tiny_moe",
        "gpu_slots_per_layer": config.gpu_slots_per_layer,
        "warmup_tokens": config.warmup_tokens,
        "measured_tokens": config.measured_tokens,
        "repetitions": config.repetitions,
        "output_dir": output_dir,
    }
    if dry_run:
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return 0
    if output_dir is None:
        raise SystemExit("benchmark-offload requires --output-dir")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark-offload requires a CUDA device")
    root = Path(output_dir)
    if (root / "DONE").exists():
        raise FileExistsError(f"completed M10 benchmark already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tiny = load_config(config.tiny_model_config)
    if config.gpu_slots_per_layer >= tiny.model.num_experts:
        raise ValueError("M10 acceptance requires fewer slots than full experts")
    device = torch.device("cuda")
    prompt = torch.tensor([config.prompt_tokens], dtype=torch.long, device=device)
    seed_everything(config.experiment.seed)
    base = TinyMoE(tiny.model, seed=config.experiment.seed, device=device)
    with torch.inference_mode():
        base_logits = base(prompt, capture_trace=False).logits
    del base
    torch.cuda.empty_cache()

    subset = None
    if config.subset_experts_by_layer is not None:
        subset = frozenset(
            ExpertKey(layer, expert)
            for layer, experts in config.subset_experts_by_layer.items()
            for expert in experts
        )

    def make_engine(asynchronous: bool) -> TinyMoEOffloadEngine:
        model = TinyMoE(tiny.model, seed=config.experiment.seed, device=device)
        return TinyMoEOffloadEngine(
            model,
            slots_per_layer=config.gpu_slots_per_layer,
            pinned_memory=config.pinned_memory,
            asynchronous=asynchronous,
        )

    sync_engine = make_engine(False)
    sync_engine.reset_metrics()
    sync_logits = sync_engine.forward(prompt)
    sync_metrics = sync_engine.finish_metrics()
    synchronous_equivalent = torch.allclose(
        base_logits, sync_logits, atol=config.atol, rtol=config.rtol
    )
    if not synchronous_equivalent:
        raise RuntimeError("lossless synchronous offload differs from base model")
    del sync_engine
    torch.cuda.empty_cache()

    engine = make_engine(True)
    engine.reset_metrics()
    async_logits = engine.forward(prompt)
    asynchronous_equivalent = torch.allclose(
        sync_logits, async_logits, atol=config.atol, rtol=config.rtol
    )
    if not asynchronous_equivalent:
        raise RuntimeError("asynchronous offload differs from synchronous reference")
    engine.finish_metrics()

    def decode(tokens: torch.Tensor, count: int, plan: frozenset[ExpertKey] | None) -> torch.Tensor:
        result = tokens.clone()
        for _ in range(count):
            logits = engine.forward(result, subset_plan=plan, miss_policy=config.subset_miss_policy)
            result = torch.cat((result, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        return result

    if config.warmup_tokens:
        decode(prompt, config.warmup_tokens, subset)
        torch.cuda.synchronize(device)
    repetition_rows: list[dict[str, object]] = []
    repetition_metrics = []
    transfer_rows: list[dict[str, object]] = []
    generated_rows: list[dict[str, object]] = []
    for repetition in range(config.repetitions):
        engine.reset_metrics()
        generated = decode(prompt, config.measured_tokens, subset)
        metrics = engine.finish_metrics()
        repetition_metrics.append(metrics)
        row = {
            "repetition": repetition,
            "tpot_ms": metrics.host_elapsed_ms / config.measured_tokens,
            **{key: value for key, value in asdict(metrics).items() if key != "transfer_records"},
        }
        repetition_rows.append(row)
        generated_rows.append(
            {"repetition": repetition, "generated_tokens": json.dumps(generated.tolist())}
        )
        transfer_rows.extend(
            {
                "repetition": repetition,
                "layer_idx": record.key.layer_idx,
                "expert_idx": record.key.expert_idx,
                **{key: value for key, value in asdict(record).items() if key not in {"key"}},
            }
            for record in metrics.transfer_records
        )

    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("repetition\n", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows(root / "repetitions.csv", repetition_rows)
    write_rows(root / "transfers.csv", transfer_rows)
    write_rows(root / "generated.csv", generated_rows)
    h2d = [metrics.h2d_bytes for metrics in repetition_metrics]
    tpot = [metrics.host_elapsed_ms / config.measured_tokens for metrics in repetition_metrics]
    stall = [metrics.exposed_stall_ms for metrics in repetition_metrics]
    metrics_json = {
        **estimate,
        "synchronous_equivalent": synchronous_equivalent,
        "asynchronous_equivalent": asynchronous_equivalent,
        "max_abs_error_base_sync": float((base_logits - sync_logits).abs().max()),
        "max_abs_error_sync_async": float((sync_logits - async_logits).abs().max()),
        "median_h2d_bytes": statistics.median(h2d),
        "median_exposed_stall_ms": statistics.median(stall),
        "median_tpot_ms": statistics.median(tpot),
        "peak_allocated_bytes": max(metrics.peak_allocated_bytes for metrics in repetition_metrics),
        "peak_reserved_bytes": max(metrics.peak_reserved_bytes for metrics in repetition_metrics),
        "resident_expert_bytes": engine.resident_expert_bytes,
        "pinned_memory_requested": config.pinned_memory,
        "pinned_cpu_bytes": engine.pinned_cpu_bytes,
        "subset_plan": None
        if subset is None
        else [[key.layer_idx, key.expert_idx] for key in sorted(subset)],
        "subset_miss_policy": config.subset_miss_policy,
        "sync_reference_h2d_bytes": sync_metrics.h2d_bytes,
        "timing_sources": {
            "transfer_and_stall": "CUDA events",
            "TPOT": "host perf_counter with terminal CUDA synchronize",
            "memory": "torch CUDA allocator peak counters",
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "model_manifest.json").write_text(
        json.dumps(engine.model.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }
    (root / "hardware.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    (root / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics_json, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "inspect-model":
        return inspect_model(arguments.config, dry_run=arguments.dry_run)
    if arguments.command == "collect-traces":
        return collect_traces(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "oracle-sweep":
        return oracle_sweep(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "closed-loop-eval":
        return closed_loop_eval(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "dapq-factorial":
        return dapq_factorial(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "analyze-router":
        return analyze_router(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "build-default-vectors":
        return build_default_vectors(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "train-predictor":
        return train_predictor(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "evaluate-probe":
        return evaluate_probe_command(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "simulate-offload":
        return simulate_offload(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "benchmark-offload":
        return benchmark_offload(
            arguments.config, output_dir=arguments.output_dir, dry_run=arguments.dry_run
        )
    if arguments.command == "accuracy-suite":
        from pseudoroute.benchmark.runner import main as accuracy_main

        if arguments.output_dir is None:
            raise SystemExit("accuracy-suite requires --output-dir")
        forwarded = [
            "--config",
            arguments.config,
            "--output-dir",
            arguments.output_dir,
        ]
        if arguments.dry_run:
            forwarded.append("--dry-run")
        return accuracy_main(forwarded)
    if arguments.command == "trained-suite":
        from pseudoroute.trained.config import load_trained_suite_config
        from pseudoroute.trained.runner import run_trained_suite

        if arguments.output_dir is None:
            raise SystemExit("trained-suite requires --output-dir")
        result = run_trained_suite(
            load_trained_suite_config(arguments.config),
            Path(arguments.output_dir),
            dry_run=arguments.dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "aggregate-results":
        from pseudoroute.reporting.aggregate import aggregate_runs

        summary = aggregate_runs(Path(arguments.input_root), Path(arguments.output_dir))
        print(
            json.dumps(
                {
                    "included_run_ids": summary.included_run_ids,
                    "incomplete_runs": summary.incomplete_runs,
                    "negative_results": summary.negative_results,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "reproduce":
        from pseudoroute.reporting.reproduce import reproduce_primary

        reproduce_primary(
            suite=arguments.suite,
            config_root=Path(arguments.config_root),
            output_root=Path(arguments.output_dir),
            dry_run=arguments.dry_run,
        )
        return 0
    raise SystemExit(
        f"{arguments.command} belongs to a later milestone "
        "and is intentionally not implemented in M0"
    )


if __name__ == "__main__":
    raise SystemExit(main())
