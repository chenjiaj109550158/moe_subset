# Implementation Roadmap

## Overview

The roadmap is organized as vertical milestones. Each milestone must leave the repository runnable and tested.

Do not start the optimized offload engine before the oracle and closed-loop milestones are complete.

## M0 — Repository bootstrap and tiny MoE

### Deliverables

- package skeleton;
- typed configuration loading;
- logging and run-directory utilities;
- deterministic tiny MoE model;
- natural generation;
- router trace capture;
- basic test suite;
- CLI skeleton;
- continuous integration configuration.

### Required tests

- deterministic forward/generation;
- exact router top-\(k\);
- expert IDs are layer-scoped;
- device-agnostic tests pass on CPU and the configured accelerator;
- a CUDA smoke test runs when CUDA is available;
- config round-trip;
- run directory is reproducible.

### Acceptance

```bash
pseudoroute inspect-model --config configs/model/tiny_moe.yaml
pytest
```

must pass in a clean environment.

## M1 — Generic model introspection and first real adapter

### Deliverables

- `MoEModelAdapter` interface;
- adapter registry;
- first real model adapter;
- model manifest export;
- route-only and router-logit hooks;
- trace storage and validation;
- exact-output regression test on a short prompt.

### Required tests

- hooks do not change next-token logits;
- captured route equals model's native route;
- trace resume does not duplicate positions;
- trace schema validation catches corruption;
- unsupported architecture fails with actionable message.

### Acceptance

Collect and validate a small real-model trace.

## M2 — Oracle window analysis

### Deliverables

- window iterator;
- count and mass oracle selectors;
- SCH-style metrics;
- expert union analysis;
- open-loop sweep CLI;
- plots and result aggregation.

### Required tests

- synthetic oracle top-\(B\) matches brute force for additive scores;
- segment boundaries are correct;
- horizon does not cross sample boundaries;
- budgets are obeyed;
- metrics match hand calculations.

### Acceptance

Produce a complete oracle sweep on the tiny model and a small real trace.

## M3 — Closed-loop constrained generation

### Deliverables

- routing-policy abstraction;
- mask injection;
- substitution;
- truncation variants;
- lossless fallback reference;
- fixed-window planner using oracle subsets;
- generated-route tracing;
- divergence and quality metrics.

### Required tests

- natural policy exactly matches base model;
- lossless fallback exactly matches base model;
- mask never activates unavailable experts;
- substitution preserves top-\(k\) cardinality when budget permits;
- truncation semantics match config;
- constrained generation re-plans from its own trajectory.

### Acceptance

Generate oracle quality–transfer curves. Decide whether the project passes Gate A.

## M4 — DapQ-style factorial analysis

### Deliverables

- teacher-forced pseudo-sequence constructor;
- content and position intervention library;
- pre-/post-RoPE query capture where supported;
- router-input and router-logit comparison;
- per-layer/per-horizon metrics;
- bootstrap confidence intervals;
- heatmap plots.

### Required tests

- SC/SP reproduces ground truth within tolerance;
- DC/SP changes content but preserves intended positions;
- SC/DP preserves token IDs but changes positions;
- pseudo inputs cannot enter online APIs;
- no sample crosses document boundary.

### Acceptance

Produce the full 2×2 analysis on the tiny model and a smaller real-model sample. Decide whether the project passes Gate B.

## M5 — Router geometry and expert criticality

### Deliverables

- router SVD/effective-rank analysis;
- row-space coordinates;
- covariance sketch;
- top-\(k\) margin analysis;
- expert pair boundary analysis;
- sampled expert interventions;
- quality and downstream-routing influence metrics.

### Required tests

- exact row-space projection reproduces logits;
- low-rank reconstruction error is correct;
- margin stability theorem verified on synthetic cases;
- intervention restores original state after completion.

### Acceptance

Produce per-layer router geometry and criticality artifacts.

## M6 — Deployable baseline probes

### Deliverables

- current-route reuse;
- rolling frequency/mass;
- Markov transition;
- direct ridge/linear predictor;
- direct MLP predictor;
- ADEPT-style RF;
- predictor dataset creation;
- leakage-safe split;
- online latency benchmark.

### Required tests

- online probes accept only `DeployableDecodeState`;
- train/test prompt IDs do not overlap;
- predictor serialization is safe;
- probabilities and shapes are valid;
- inference is deterministic.

### Acceptance

A common evaluation command compares every baseline against the same oracle target.

## M7 — Position-conditioned pseudo routing probes

### Deliverables

- future-position rephased probe;
- pseudo-token independent probe;
- pseudo-token causal probe;
- production KV read-only wrapper;
- default-vector collector;
- default-vector shadow rollout;
- sparse horizon anchors;
- uncertainty ensemble;
- probe cost accounting.

### Required tests

- shadow cache does not mutate production cache;
- future positions are correct;
- pre-sample mode cannot use next token;
- post-sample mode may use exactly one sampled next token;
- pseudo RNG does not alter generation RNG;
- default-vector counts/means match brute force on tiny model.

### Acceptance

Compare probes under equal measured cost and produce subset-regret curves.

## M8 — Cost-aware subset selection and simulator

### Deliverables

- adjusted top-\(B\) selector;
- variable-size knapsack;
- global budget extension;
- cache state machine;
- event-driven transfer simulator;
- hardware profiles;
- LRU/LFU/on-demand baselines;
- overlap model;
- timeline export.

### Required tests

- no capacity violation;
- byte accounting exact;
- no use before load completion;
- bandwidth sharing correct;
- perfect/no-overlap bounds;
- synthetic timeline matches expected TPOT.

### Acceptance

Produce simulator Pareto plots, clearly labeled simulated.

## M9 — Static plus dynamic residency and adaptive termination

### Deliverables

- static ranking methods;
- expert criticality integration;
- static/dynamic budget split;
- out-of-subset mass monitoring;
- early termination;
- adaptive horizon;
- sensitivity sweeps.

### Required tests

- static experts are never evicted;
- dynamic budget respects remaining capacity;
- early termination logs reason;
- re-planning occurs at the correct boundary;
- no infinite re-plan loop.

### Acceptance

Show whether static/dynamic and adaptive horizon improve the oracle or deployable Pareto frontier.

## M10 — Real single-GPU offload engine

### Deliverables

- CPU expert handles;
- pinned-memory setup;
- fixed GPU slots;
- synchronous reference swapping;
- asynchronous transfer stream;
- CUDA event synchronization;
- resident-cache integration;
- lossless and hard-commit execution;
- per-token timeline.

### Required tests

- synchronous offload matches base model in lossless mode;
- asynchronous result matches synchronous reference;
- slot replacement cannot race with compute;
- CUDA event dependencies are correct;
- measured resident bytes obey budget;
- graceful skip of CUDA-specific tests when CUDA is unavailable in CI.

### Acceptance

Run one real model whose full experts exceed the configured GPU expert budget.

## M11 — Full evaluation and artifact generation

### Deliverables

- benchmark runners;
- end-to-end experiment scripts;
- result aggregator;
- paper tables;
- paper plots;
- failure-case reports;
- reproducibility manifest;
- model support matrix;
- final documentation.

### Required tests

- aggregator rejects incompatible schemas;
- duplicated runs are detected;
- incomplete runs are excluded and reported;
- plots can be regenerated from saved tabular data;
- every table cell links to run IDs.

### Acceptance

A clean command sequence regenerates the primary results.

## Cross-cutting task: documentation

At every milestone update:

- root README;
- `STATUS.md`;
- model support matrix;
- CLI help;
- configuration examples;
- known limitations;
- changelog.

## Cross-cutting task: profiling

Profile in stages:

1. Python overhead;
2. model forward;
3. trace capture;
4. pseudo attention;
5. default-vector mixing;
6. subset selection;
7. transfer;
8. synchronization;
9. expert compute.

Do not optimize a stage before measuring its share of TPOT.

## Cross-cutting task: artifact size control

Full hidden traces can become enormous. Implement:

- trace-level switches;
- token/layer sampling;
- float16 storage;
- compression where lossless enough;
- per-layer capture;
- shard quotas;
- dry-run size estimates;
- deletion-safe manifests.

## Suggested first vertical slice

The first scientifically meaningful slice is:

1. tiny model;
2. natural route trace;
3. oracle mass subset;
4. substitution-constrained generation;
5. H2D byte simulation;
6. quality–transfer plot.

The second slice is:

1. first real model;
2. SC/SP, DC/SP, SC/DP, DC/DP;
3. current-route baseline;
4. rephased future-position probe;
5. per-layer router-logit heatmap.

The third slice is:

1. default vectors;
2. shadow rollout;
3. cost-aware subset;
4. adaptive termination;
5. simulator.

## Optimization backlog, not initial requirements

Only consider after M10 correctness:

- fused expert slot kernels;
- Triton expert execution;
- quantized CPU storage;
- decompression while transferring;
- multi-stream expert loads;
- CUDA graphs;
- pinned-memory pool;
- direct-storage/SSD tier;
- multi-GPU;
- continuous batching;
- speculative decoding integration.
