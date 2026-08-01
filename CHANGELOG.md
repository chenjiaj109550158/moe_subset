# Changelog

## Unreleased

- Complete the frozen Qwen3-30B-A3B/GSM8K `H=8,B=32` pseudo-embedding focused
  pilot with native shadow attention/RoPE/router execution, cache/RNG audits,
  route/cost/strata/worst-case artifacts, atomic resume, and a development-gated
  STOP/PIVOT that forbids held-out and actual accuracy execution.

- Complete the multi-model trained-MoE oracle gate with common OLMoE/Qwen/Mixtral/DeepSeek/gpt-oss adapters, trace schema v2, pinned real-text traces, oracle/cache sweeps, context/margin bootstrap summaries, 64-token closed loop, resumable envelopes, and a preserved overall STOP/PIVOT result.

- Implement M11 benchmark runners, checksummed run envelopes, strict aggregation, Tables A–F, plot index/regeneration, concrete failure-case and negative-result reports, resumable primary reproduction, and `aggregate-results`/`reproduce`.

- Implement M10 CPU-resident expert handles, optional pinned memory, fixed GPU slots, synchronous/asynchronous CUDA transfer paths, event dependencies, subset-plan execution, measured runtime artifacts, and `benchmark-offload`.

- Implement M5 exact/randomized router SVD, row-space coordinates, covariance/variance, boundaries, entropy/margins, sampled expert interventions, analysis tables/plots, and `analyze-router`.

- Implement M4 offline SC/SP–DC/DP factorial construction, context/position controls, adapter activation capture, per-layer/horizon metrics, paired bootstrap CIs, plots, tiny and pinned-real runs, and a negative Gate B decision.

- Implement M3 explicit routing policies, fixed-window oracle planning, authoritative closed-loop generation, route/divergence/quality/transfer tables, Gate A reporting, and a quality-versus-transfer plot for the tiny model.

- Implement M2 sample-bounded oracle windows, additive count/mass/cost-aware top-B selection, expert-union and SCH metrics, CSV aggregation, SVG plots, and tiny/trace-backed `oracle-sweep`.

- Implement M1 adapter contracts, registry, tiny and pinned Hugging Face Mixtral adapters.
- Add native-route/logit parity regression tests and `NaturalRoutingPolicy`.
- Add atomic sharded-safetensors trace collection, resume, manifests, and validation.

- Bootstrap M0 with typed configuration, deterministic tiny MoE inference,
  natural route tracing, reproducible run directories, CLI skeleton, and tests.
