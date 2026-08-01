# PseudoRoute-MoE

Research framework for position-conditioned pseudo routing states and
memory-budgeted mixture-of-experts inference.

The repository has completed **M11 at deterministic primary-suite scope** and the
subsequent **multi-model trained-MoE oracle feasibility gate**. It contains a
deterministic, accelerator-aware tiny causal MoE model, typed configuration
loading, route tracing, reproducible run-directory utilities, and the complete
normative specification in `docs/spec/`. The trained gate tested four
floating-point checkpoints plus a separate native-MXFP4 tier and concluded
**STOP/PIVOT** under the pinned v2 protocol. Simulated transfer/stall estimates,
actual model-quality results, and measured adapter memory are labeled separately.

## Completed benchmark subset and focused pseudo pilot

The benchmark-aligned pivot reuses the checksum-valid v17 prompts, tokens, and
evaluators without rerunning vanilla. Its frozen base config is
`configs/benchmark/benchmark_subset_oracle_v1.yaml`. After preserving 808 actual
hard rows from the superseded six-task scope, the explicit 2026-07-31 amendment
`configs/benchmark/benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2.yaml` limits full
accuracy execution to all 1,319 GPT-OSS GSM8K rows at `(H=1,B=4)`. It completed
1,242/1,319 for both vanilla and true hard closed loop, with exact-token
agreement 1.0 and no paired gains or losses. Because `H=1,B=4` equals native
top-k, this is a task-scoped `NARROW` natural-route ceiling, not multi-token
subset evidence or runtime speedup. Existing
HumanEval/MBPP+/previous-route rows and interruption markers remain provenance
and are excluded from the scoped aggregate.

The separately frozen Qwen3-30B-A3B/GSM8K pseudo-embedding pilot at
`(H=8,B=32)` is also complete with **STOP/PIVOT**. Native mechanism smoke passed,
but all four mandatory training-free variants underperformed previous-route on
the four predeclared development traces. The primary sampled-next-token/default-
vector variant reached route hit 0.533015 and selected mass 0.534416 versus
0.609385 and 0.629428 for previous-route. The zero-contribution ablation was the
best mandatory variant at 0.566499/0.575402, still below previous-route. The
frozen development gate therefore prohibited held-out route evaluation and
actual closed-loop accuracy; no task-accuracy or identity-materialized pseudo
rows were produced. See the [focused report](artifacts/pseudo_embedding_qwen_gsm8k_v1/report.md)
and [protocol](docs/pseudo_embedding_qwen_gsm8k_v1_protocol.md).

A separately versioned, strictly calibration-free follow-up then analyzed the
failure without relabeling v1. Native interventions found that repeated sampled
content changes much less than natural routes, recent token IDs do not repair
the hidden direction, and causal rollout with zero MoE residual is harmful.
Same-request prompt/generated route history plus independent sampled-token
pseudo scores passed four-row development, but failed the committed eight-row
held-out gate: route hit/selected mass were 0.658353/0.680584 versus
0.617671/0.637396 for previous route, only +0.040682/+0.043188. Oracle-gap
recovery was 11.7%/12.6%, below 25%. The result is **STOP/PIVOT**; no accuracy or
actual closed-loop rows were run. See the
[calibration-free synthesis](docs/pseudo_embedding_calibration_free_synthesis_v1.md)
and [held-out report](artifacts/pseudo_embedding_calibration_free_prompt_route_held_out_v1/report.md).

## Trained-model oracle gate

The checksum-valid final suite is `artifacts/trained_gate/suite_v2_final/` and is
recreated by the versioned `configs/trained/oracle_gate_v2.yaml`. It uses four
independent WikiText-2 validation rows and four GSM8K test rows per tokenizer,
horizons 1/2/4/8/16, absolute and native-top-k-multiple budgets, eight oracle or
cache methods, and 1,000-sample bootstrap confidence intervals. Closed-loop
quality uses complete prompts and 64-token deterministic greedy continuations.

| Checkpoint | Final decision | Decisive result |
|---|---|---|
| OLMoE-1B-7B-0125 | STOP/PIVOT | One WikiText open-loop point passes, but hard quality and 47.97% mean lossless fallback fail |
| Qwen1.5-MoE-A2.7B | STOP/PIVOT | Both-domain B=32/H=16 open-loop points pass, but hard quality and 80.22% fallback fail |
| Mixtral-8x7B-v0.1 | STOP/PIVOT | No point jointly passes the P05 and worst-window open-loop gate; hard quality also fails |
| DeepSeek-V2-Lite-Chat | STOP/PIVOT | Both-domain B=32/H=8 points pass, but hard quality and 70.70% fallback fail |
| gpt-oss-20b (MXFP4) | STOP/PIVOT | Direct native router inspection passes; fused MXFP4 forward bypasses the common trace hook |

The overall result is scoped to these revisions, two datasets, sampled rows,
64-token batch-1 greedy decoding, checkpoint precision, budgets, and two A100s.
It does not authorize predictor training or production runtime optimization. See
[the trained-model plan](docs/trained_model_plan.md) and
[decision log](docs/decisions.md).

## Development

```bash
python -m pip install -e '.[dev,hf]'
pseudoroute inspect-model --config configs/model/tiny_moe.yaml
ruff check .
mypy
pytest
PSEUDOROUTE_RUN_EXTERNAL=1 pytest tests/integration/test_hf_mixtral.py
pseudoroute collect-traces --config configs/model/hf_tiny_mixtral.yaml --output-dir artifacts/traces/hf_tiny_mixtral
pseudoroute oracle-sweep --config configs/experiment/oracle_tiny.yaml --output-dir artifacts/oracle/tiny_m2
pseudoroute closed-loop-eval --config configs/experiment/closed_loop_tiny.yaml --output-dir artifacts/closed_loop/tiny_m3_final
pseudoroute dapq-factorial --config configs/experiment/dapq_factorial_tiny.yaml --output-dir artifacts/factorial/tiny_m4_final
pseudoroute analyze-router --config configs/experiment/analyze_router_tiny.yaml --output-dir artifacts/router/tiny_m5_final
pseudoroute build-default-vectors --config configs/experiment/default_vectors_tiny.yaml --output-dir artifacts/m7_default_vectors
pseudoroute evaluate-probe --config configs/experiment/evaluate_shadow_probe_tiny.yaml --output-dir artifacts/m7_shadow_probe_pre_sample_final_v2
pseudoroute simulate-offload --config configs/experiment/static_adaptive_tiny.yaml --output-dir artifacts/m9_static_adaptive_tiny_m8_final_v2
pseudoroute simulate-offload --config configs/experiment/simulate_offload_tiny.yaml --output-dir artifacts/m8_simulate_offload_tiny_definitive
pseudoroute benchmark-offload --config configs/experiment/benchmark_offload_tiny.yaml --output-dir artifacts/m10_benchmark_offload_tiny_definitive_v2
pseudoroute reproduce --suite primary --config-root configs/paper --output-dir artifacts/reproduction/m11_primary_definitive_v2
pseudoroute aggregate-results --input-root artifacts/reproduction/m11_primary_definitive_v2/runs --output-dir artifacts/reproduction/m11_primary_definitive_v2/paper
pseudoroute trained-suite --config configs/trained/oracle_gate_v2.yaml --output-dir artifacts/trained_gate/suite_v2_final
```

See [docs/reproducibility.md](docs/reproducibility.md) for manifest guarantees, individual benchmark runners, failure retention, and hardware requirements.

The inspection result is labeled with its information regime. M0 performs only
natural inference and does not expose future tokens to an online API. The
example config uses `device: auto`, which selects CUDA when available and falls
back to MPS or CPU; set an explicit device when reproducibility requires it.

Codex agents may download milestone-relevant models and datasets when disk space
is sufficient. They must estimate required space first, retain room for generated
traces/artifacts, pin revisions where possible, and record source fingerprints.
Large downloads are cached outside version control and are never required by the
default unit-test suite.
