# PseudoRoute-MoE

Research framework for position-conditioned pseudo routing states and
memory-budgeted mixture-of-experts inference.

The repository has completed **M11 at deterministic primary-suite scope**. It contains a deterministic, accelerator-aware
tiny causal MoE model, typed configuration loading, route tracing, reproducible
run-directory utilities, and the complete normative specification in
`docs/spec/`. It includes M0–M7, the M8 simulator, M9 adaptive residency, the PyTorch-only M10 runtime, and M11 checksummed reproduction, aggregation, paper tables/plots, and failure reporting. Simulated and measured outputs are labeled separately.

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
