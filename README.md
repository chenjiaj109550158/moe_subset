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

## Active benchmark subset-oracle pivot

The benchmark-aligned pivot reuses the checksum-valid v17 prompts, tokens, and
evaluators without rerunning vanilla. Its frozen base config is
`configs/benchmark/benchmark_subset_oracle_v1.yaml`; the explicit 2026-07-30
scope amendment is
`configs/benchmark/benchmark_subset_oracle_v1_hard_only_full_v1.yaml`. The full
stage completes only the selected GPT-OSS `(H=1,B=4)` hard-oracle policy over six
datasets. Preserved partial previous-route rows are provenance, not a completed
full baseline. The next planned, separately frozen experiment focuses on
Qwen3-30B-A3B/GSM8K `(H=8,B=32)` and pseudo embedding; it does not authorize
learned predictor training.

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
