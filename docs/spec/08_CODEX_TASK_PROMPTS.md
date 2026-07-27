# Incremental Codex Task Prompts

These prompts are intended to be issued one at a time to Codex after providing the full specification directory. They keep implementation aligned with the research plan.

## Prompt 0 — Read and plan

```text
Read every Markdown file in docs/spec, beginning with 00_CODEX_START_HERE.md. Do not implement model-specific optimizations yet.

Create:
1. STATUS.md with milestone M0 and a file-level checklist.
2. docs/decisions.md containing the fixed initial assumptions and any implementation decisions you must make.
3. A proposed repository tree matching the specification.
4. A dependency plan that avoids unnecessary packages.
5. A model/dataset download and cache plan that estimates disk use before downloading.

Then bootstrap the package, formatting/type-check/test configuration, CLI skeleton, and deterministic utilities. Run the test suite and update STATUS.md with exact results.
```

## Prompt 1 — Tiny MoE vertical slice

```text
Implement milestone M0.

Build a deterministic, device-configurable tiny decoder-only MoE model with optional RoPE, pre-norm attention, a linear top-k router, configurable experts, and deterministic generation. Use an explicit device setting; the default example should select CUDA when available and fall back to CPU for portability and CI. Implement route capture and a ModelSpec.

Add unit tests for natural routing, layer-scoped ExpertKey, deterministic generation, and exact trace values. Add `pseudoroute inspect-model` and a tiny model config.

External model or dataset downloads are permitted when useful and disk space is sufficient, but M0 acceptance must not depend on a large download. Estimate disk use first and record anything downloaded. Run all checks and update STATUS.md.
```

## Prompt 2 — Adapter and trace store

```text
Implement milestone M1 using the model adapter and trace-store contracts in 03_SYSTEM_ARCHITECTURE.md and 07_CONFIG_AND_DATA_SCHEMAS.md.

First make the tiny model use the same adapter interface. Then implement one real Hugging Face MoE adapter, downloading a verified model and a suitable evaluation dataset if they are not already cached and disk space is sufficient. Pin and record their identifiers and revisions. Validate structure before inference. Add route-only and router-logit trace collection, chunked storage, manifest, resume, and validation.

Add regression tests that hooks and NaturalRoutingPolicy do not change model outputs. External-model tests must be optional/skippable in CI. Update model support documentation and STATUS.md.
```

## Prompt 3 — Oracle analysis

```text
Implement milestone M2.

Add window iteration, binary-count oracle, selected-routing-mass oracle, full-router-mass oracle, expert union metrics, SCH-style curves, and cost-aware top-B selection. Implement `pseudoroute oracle-sweep`.

Use brute-force tests on tiny cases to prove additive oracle selection is correct. Prevent windows from crossing sample boundaries. Produce tabular outputs and basic plots. Run a tiny-model end-to-end sweep and update STATUS.md.
```

## Prompt 4 — Closed-loop constrained generation

```text
Implement milestone M3.

Create RoutingPolicy, NaturalRoutingPolicy, MaskedSubstitutionPolicy, MaskedTruncationPolicy, and LosslessFallbackPolicy. Add fixed-window oracle planning and closed-loop generation.

Natural and lossless fallback modes must reproduce the base model. Constrained runs must continue from their own generated trajectory. Record natural and executed routes, divergence points, out-of-subset mass, transfer-byte estimates, and quality metrics.

Add `pseudoroute closed-loop-eval`, tests, and an oracle quality-versus-transfer plot. State whether Gate A passes on the tiny model; do not generalize to real models yet.
```

## Prompt 5 — DapQ-style factorial analysis

```text
Implement milestone M4 exactly as specified.

Build an offline-only pseudo-sequence constructor with SC/SP, DC/SP, SC/DP, and DC/DP conditions. Capture pre-RoPE query, post-RoPE query, post-attention state, router input, router logits, and top-k where the adapter supports them.

Add context-swap and position-offset controls. Enforce a type boundary so these offline future tokens cannot be passed to online probes.

Implement per-layer/per-horizon metrics, paired bootstrap confidence intervals, and `pseudoroute dapq-factorial`. Validate SC/SP against teacher-forced ground truth on the tiny model.
```

## Prompt 6 — Router geometry and criticality

```text
Implement milestone M5.

Add exact/randomized SVD of router weights, router-visible coordinates, empirical covariance sketches, covariance-aware logit variance, pairwise boundary analysis, route entropy, and top-k margins.

Prove in tests that exact row-space projection reproduces router logits and that the margin stability condition is correct.

Add sampled expert interventions measuring immediate output error, next-token KL, and downstream route changes. Implement `pseudoroute analyze-router`.
```

## Prompt 7 — Baseline probes

```text
Implement milestone M6.

Add deployable CurrentRoute, RollingFrequency, RollingMass, MarkovTransition, DirectLinear/Ridge, DirectMLP, and ADEPTStyleRF probes. All online probes must accept only DeployableDecodeState.

Build grouped predictor datasets split by prompt/document, serialize models safely, record training/calibration cost, and benchmark online latency. Implement `pseudoroute train-predictor` and `pseudoroute evaluate-probe`.

Compare every probe against the same oracle window-utility target and include equal-cost reporting.
```

## Prompt 8 — Position-conditioned probes

```text
Implement milestone M7.

Start with FuturePositionRephasedProbe and its no-rephase/wrong-position ablations. Then implement independent and causal pseudo-token probes with a read-only production KV-cache interface.

Implement streaming default-vector calibration and validate same-token next-layer prediction. Extend it into DefaultVectorShadowRolloutProbe over future-position anchors. Support pre_sample and post_sample regimes without leakage. Isolate probe RNG from generation RNG.

Measure probe latency and temporary memory. Add all required tests and update STATUS.md with limitations by model adapter.
```

## Prompt 9 — Selector and simulator

```text
Implement milestone M8.

Add window utility aggregation, uncertainty-aware scores, adjusted top-B selection, variable-size knapsack, subset plans, and exact load/eviction deltas.

Build an event-driven expert-cache simulator with on-demand, LRU, LFU, lossless predictor, one-step commitment, and multi-step commitment baselines. Enforce capacity, bandwidth, latency, and load-completion invariants.

Implement `pseudoroute simulate-offload`, timeline export, and sensitivity sweeps. Clearly label outputs as simulated.
```

## Prompt 10 — Static residency and adaptive horizon

```text
Implement milestone M9.

Add static expert rankings by frequency, covariance-aware variance, quality sensitivity, downstream routing influence, miss risk, and combined score. Split the HBM budget into static and dynamic portions.

Add out-of-subset-mass monitoring and adaptive early termination. Log every termination and re-plan event. Compare fixed and adaptive horizons in closed-loop generation and simulation.
```

## Prompt 11 — Real offload engine

```text
Implement milestone M10 only after all prior correctness tests pass.

Build a single-GPU, batch-1 expert offload prototype with CPU-resident expert tensors, optional pinned memory, fixed preallocated GPU slots, a synchronous reference path, an asynchronous transfer stream, and CUDA event dependencies.

First prove lossless synchronous output equivalence, then asynchronous equivalence. Integrate subset plans and measure real H2D bytes, exposed stall, TPOT, and peak memory. Do not add Triton or custom CUDA until profiling shows the need.

Implement `pseudoroute benchmark-offload`.
```

## Prompt 12 — Full evaluation

```text
Implement milestone M11.

Add benchmark runners, complete run manifests, aggregation, paper-ready tables/plots, failure-case reports, and a reproducibility command. Reject incompatible/incomplete runs and retain negative results.

Create scripts that regenerate the primary oracle, factorial, probe, simulator, and real-runtime results from configs. Update README, model support matrix, STATUS.md, and CHANGELOG.md.
```

## Prompt for code review after any milestone

```text
Review the current repository against all specification documents. Focus on:
1. future-information leakage;
2. mismatch from native router semantics;
3. production KV mutation by pseudo probes;
4. incorrect expert byte/capacity accounting;
5. open-loop metrics being mislabeled as closed-loop;
6. simulator results being presented as real speedups;
7. train/test prompt leakage;
8. missing tests or undocumented assumptions.

Fix confirmed issues, add regression tests, and update STATUS.md. Do not broaden scope during this review.
```

## Prompt for experiment audit

```text
Audit every completed run under artifacts/runs.

For each run, verify:
- resolved config and information regime;
- model/dataset fingerprints;
- git commit and environment;
- terminal marker;
- no future leakage;
- correct routing policy;
- HBM capacity compliance;
- quality and transfer metrics;
- whether timing is simulated or measured.

Produce an audit table and quarantine invalid runs without deleting them.
```
