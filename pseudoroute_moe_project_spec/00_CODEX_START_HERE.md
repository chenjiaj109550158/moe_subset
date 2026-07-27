# Codex Start Here

## Role

You are the primary implementation agent for a new research repository named `pseudoroute-moe`.

Start from an empty repository. Read every specification document in this directory before implementing model-specific optimizations. The documents are normative: when implementation choices conflict, follow the priority order below.

1. Correctness and no future leakage.
2. Reproducibility and testability.
3. Exact baseline model semantics.
4. Scientific usefulness.
5. Runtime performance.
6. Breadth of model support.

Do not silently weaken a requirement. Record unavoidable deviations in `STATUS.md` and `docs/DECISIONS.md`.

## Mission

Build an end-to-end research framework for studying and implementing position-conditioned pseudo routing states and horizon-level expert commitment for memory-budgeted MoE inference.

The framework must support all of the following:

1. Instrument a Hugging Face-style MoE causal language model and collect decode-time router traces.
2. Compute oracle segment/window expert subsets from true future routing.
3. Run closed-loop constrained generation using fixed expert subsets.
4. Reproduce a DapQ-style content-versus-position factorial analysis for future MoE routing.
5. Analyze router geometry, top-\(k\) margins, temporal locality, and expert criticality.
6. Implement training-free and learned future-routing baselines.
7. Implement future-position pseudo routing probes.
8. Aggregate predicted future routes into a cost-aware per-layer subset.
9. Simulate HBM expert caching and CPU-to-GPU transfers.
10. Implement a real single-GPU expert offload prototype after the simulator and oracle gates pass.
11. Produce paper-ready metrics, tables, and plots with fully recorded configs.

## First action

Create the following repository skeleton without implementing model-specific logic yet:

```text
pseudoroute-moe/
├── README.md
├── pyproject.toml
├── LICENSE
├── .gitignore
├── configs/
├── docs/
├── scripts/
├── src/pseudoroute/
├── tests/
├── artifacts/.gitkeep
├── STATUS.md
└── CHANGELOG.md
```

Then copy these specification documents into `docs/spec/` or retain them at repository root.

Create `STATUS.md` with:

- current milestone;
- completed tasks;
- failing tests;
- known limitations;
- next exact tasks;
- decisions that need human review.

## Engineering rules

### 1. Keep offline and online APIs separate

Use explicit types or namespaces:

```python
OfflineFutureTrace
DeployableDecodeState
```

An online probe must not accept an `OfflineFutureTrace`. Do not rely only on comments. Make leakage difficult at the type and API level.

Every result file must contain:

```yaml
information_regime: oracle | offline_teacher_forced | online_pre_sample | online_post_sample
```

### 2. Preserve original model behavior

Before adding masks or offloading, verify that the adapter reproduces the unmodified model:

- same next-token logits within configured tolerance;
- same router top-\(k\) indices;
- same router weights;
- deterministic output under fixed seed and greedy decoding.

A model adapter is not accepted until these tests pass.

### 3. Build a tiny MoE test model

Do not make unit tests download a large model. Implement a deterministic tiny causal MoE model with:

- configurable layers, experts, hidden size, and top-\(k\);
- a router linear layer;
- expert MLPs;
- explicit router logits and expert outputs;
- deterministic generation;
- optional synthetic transfer latency.

Use it to test every algorithmic contract.

### 4. Simulator before real offloading

The first performance result must come from a trace-driven simulator. A simulator result must be labeled as simulated and must never be presented as an end-to-end speedup.

Only implement the real offload engine after:

- oracle closed-loop feasibility is demonstrated;
- cache behavior is validated against hand-computed examples;
- transfer accounting tests pass;
- probe overhead is measured.

### 5. No hidden fallback

Execution semantics must be an explicit enum:

```python
class MissPolicy(Enum):
    LOSSLESS_FALLBACK = "lossless_fallback"
    SUBSTITUTE = "substitute"
    TRUNCATE = "truncate"
    EARLY_TERMINATE_WINDOW = "early_terminate_window"
    CPU_EXECUTE = "cpu_execute"
```

Never change policy automatically without logging it.

### 6. Avoid model-name conditionals in core logic

Use a model-adapter registry. Architecture-specific code belongs under `src/pseudoroute/models/adapters/`.

### 7. Make expensive traces optional

Router logits and top-\(k\) traces are mandatory. Full router inputs, post-attention residuals, attention queries, and expert outputs are opt-in because they are large.

### 8. Configuration drives every experiment

No paper experiment may depend on hard-coded constants. All runs must save a resolved config, environment manifest, git commit, random seeds, and hardware profile.

### 9. Test numerical and semantic invariants

Required tests include:

- subset size never exceeds the budget;
- selected experts belong to the correct layer;
- no cross-layer expert-ID aliasing;
- online probes never access future arrays;
- H2D bytes equal the sum of loaded expert bytes;
- resident-set capacity is never exceeded;
- a lossless fallback run matches the base model;
- an oracle subset computed from a window is optimal for the specified additive score;
- top-\(k\) stability bound is correct on synthetic logits;
- pseudo shadow cache cannot mutate the production KV cache.

### 10. Optimize only after profiling

Avoid Triton/CUDA extensions in the initial milestones. Establish correctness with PyTorch. Add optimized kernels only when a profiler shows a material bottleneck.

### 11. Download models and datasets when needed

Codex agents may download milestone-relevant models and datasets when local disk space permits. Before downloading, estimate the download and expanded-cache sizes, check available space, and leave enough headroom for traces and experiment artifacts. Reuse a configurable cache, pin model and dataset revisions where supported, and record source identifiers, revisions, cache location, and fingerprints in run manifests. Do not commit downloaded weights, raw datasets, credentials, or access tokens to Git. Large external-model tests must remain optional/skippable in CI.

## Initial assumptions

These are defaults, not universal truths:

- standard decoder-only, pre-norm, RoPE-based MoE;
- top-\(k\) routing;
- one expert subset per MoE layer;
- a global window boundary for all MoE layers;
- batch size 1;
- expert weights live in CPU memory when not resident;
- dense attention and non-expert weights remain on GPU if memory permits;
- the core online method uses no speculative draft model;
- the primary method is training-free;
- predictor training is allowed only for baselines or later extensions.

All assumptions must be configurable or isolated behind an interface.

## Milestone order

Implement in this order:

1. **M0 — Repository and tiny model**
2. **M1 — Model introspection and exact trace collection**
3. **M2 — Oracle window analysis**
4. **M3 — Closed-loop constrained routing**
5. **M4 — DapQ-style routing factorial analysis**
6. **M5 — Router geometry and expert criticality**
7. **M6 — Deployable probe baselines**
8. **M7 — Position-conditioned pseudo routing probes**
9. **M8 — Cost-aware subset selection and cache simulator**
10. **M9 — Static + dynamic residency**
11. **M10 — Real offload prototype**
12. **M11 — Full benchmark and paper artifacts**

Do not skip M2 or M3. A predictor is not useful if the oracle hard-subset method is intrinsically too damaging.

## Required command surface

The final repository must expose a single CLI entry point named `pseudoroute` with at least:

```bash
pseudoroute inspect-model
pseudoroute collect-traces
pseudoroute oracle-sweep
pseudoroute closed-loop-eval
pseudoroute dapq-factorial
pseudoroute analyze-router
pseudoroute build-default-vectors
pseudoroute evaluate-probe
pseudoroute train-predictor
pseudoroute simulate-offload
pseudoroute benchmark-offload
pseudoroute aggregate-results
```

Each command must support `--config`, `--output-dir`, and `--dry-run`.

## Definition of done

The repository is complete when:

- all required commands exist and are documented;
- unit and integration tests pass;
- at least one real MoE model adapter works;
- oracle and closed-loop sweeps run end to end;
- the DapQ-style routing analysis produces per-layer/per-horizon results;
- at least four deployable baselines and two pseudo-routing probes are implemented;
- the simulator reports exact memory and transfer accounting;
- the real offload prototype runs on one GPU with CPU-resident experts;
- all metrics are aggregated into reproducible tables;
- no online result uses future information;
- limitations and negative findings are reported rather than hidden.

## Codex behavior during implementation

At the beginning of each milestone:

1. Read the relevant sections of all specifications.
2. Add a file-level checklist to `STATUS.md`.
3. Implement the smallest correct vertical slice.
4. Add tests before optimization.
5. Run formatting, type checking, and tests.
6. Update `STATUS.md` with exact results and remaining limitations.

When a paper detail is ambiguous or unavailable, implement the documented *style of baseline*, label it clearly, and do not claim exact reproduction.
