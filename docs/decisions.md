# Decision Log

## Fixed initial assumptions

These assumptions are normative defaults from the specification. They are not universal claims and must remain configurable or isolated behind interfaces:

- decoder-only, pre-norm, RoPE-based MoE architecture;
- standard top-k routing with exact model-specific semantics delegated to adapters;
- batch size 1 and ordinary autoregressive decoding;
- M0 acceptance uses no external model or dataset download; later downloads require pinned revisions, disk estimates, headroom, configurable caches, and manifest records;
- explicit device selection; `auto` prefers CUDA, then MPS, then CPU, while the resolved device is recorded;
- one expert subset per MoE layer and a shared global window boundary;
- expert weights in CPU memory when not resident;
- dense attention and non-expert weights on GPU when memory permits;
- no speculative draft model in the core online method;
- training-free primary method; learned predictors are later baselines or extensions;
- simulator before real offloading, after oracle and constrained-generation gates;
- offline future traces and online deployable state remain separate types;
- pre-sample state never includes the next token; post-sample state may include exactly the already sampled next token and no later token;
- miss policy is always explicit and never silently changed;
- milestone-relevant model and dataset downloads are allowed after checking disk capacity, with pinned revisions and recorded fingerprints.

## D-20260727-001 — Keep M0 strictly natural-routing only

**Status:** accepted  
**Context:** The specification requires ordered milestones and prohibits skipping
to constrained execution or real offloading.  
**Decision:** M0 exposes natural inference and trace capture only. Miss policies
are typed now but have no execution implementation until M3.  
**Alternatives considered:** Implement the suggested scientific vertical slice
through oracle selection and substitution.  
**Consequences:** The repository satisfies M0 without prematurely introducing
future information or lossy execution.  
**Experiments affected:** M0 model inspection and deterministic smoke tests.  
**Migration required:** None; M2 and M3 add their own isolated modules.

## D-20260727-002 — Use strict Pydantic configuration models

**Status:** accepted  
**Context:** M0 requires typed configuration, round trips, and rejection of
unknown keys.  
**Decision:** Validate YAML with strict Pydantic models and serialize resolved
configuration deterministically.  
**Alternatives considered:** Dataclasses with handwritten validation.  
**Consequences:** Configuration errors fail before model construction and the
resolved representation has a stable hash.  
**Experiments affected:** All command configuration.  
**Migration required:** Extend the schema with explicit fields in later milestones.


## D-20260727-003 — Make ModelSpec canonical metadata

**Status:** accepted  
**Context:** M0 requires a typed `ModelSpec`, while CLI inspection also needs JSON-safe metadata.  
**Decision:** The tiny model exposes a typed `spec` property using layer-scoped `ExpertKey` byte entries; its JSON manifest is derived from that object.  
**Alternatives considered:** Keep an untyped manifest dictionary as the only metadata representation.  
**Consequences:** Internal contracts remain typed and layer-safe while CLI output remains portable JSON.  
**Experiments affected:** Model inspection and all later adapter manifests.  
**Migration required:** Future adapters implement the same `ModelSpec` fields.

## D-20260727-004 — Use native Mixtral router outputs

**Status:** accepted  
**Context:** Generic hooks risk drifting from architecture-specific routing semantics.  
**Decision:** The Hugging Face Mixtral adapter requests native `router_logits` and uses the native gate for route-from-state; parity tests compare IDs, weights, and logits.  
**Alternatives considered:** A generic softmax/top-k forward hook.  
**Consequences:** M1 supports one architecture exactly without claiming generic router semantics.  
**Experiments affected:** Mixtral inspection and trace collection.  
**Migration required:** Each future architecture gets its own validated adapter.

## D-20260727-005 — Sharded safetensors trace store

**Status:** accepted  
**Context:** Traces require safe persistence, checksums, random access, resume, and atomic commits without unnecessary dependencies.  
**Decision:** Store one atomic safetensors shard per sample with a versioned JSON manifest and SHA-256 checksums.  
**Alternatives considered:** Zarr or unsafe torch/pickle serialization.  
**Consequences:** Route-only reads and sample resume are simple; very large runs may later group samples per shard.  
**Experiments affected:** All trace-based milestones.  
**Migration required:** Preserve schema versioning when adding optional arrays.

## D-20260727-006 — Keep M2 oracle analysis explicitly open-loop

**Status:** accepted  
**Context:** M2 may inspect teacher-forced future routes, but closed-loop constrained execution belongs to M3.  
**Decision:** Oracle windows consume `OracleSample` tensors and emit only `information_regime=oracle`, `evaluation_mode=open_loop` results. No online state or routing policy accepts them.  
**Alternatives considered:** Reuse future traces inside generation.  
**Consequences:** M2 measures local routing consistency without claiming generation quality.  
**Experiments affected:** All oracle sweeps.  
**Migration required:** M3 must generate and trace its own authoritative constrained trajectory.

## D-20260727-007 — Exact-cardinality deterministic top-B for additive M2 scores

**Status:** accepted  
**Context:** Count, routing-mass, and fixed load-cost-adjusted utilities are additive across experts.  
**Decision:** Select exactly B experts independently per layer by descending utility, breaking ties by expert index. Budgets are `ceil(ratio * top_k)`, clamped to `[1, num_experts]`.  
**Alternatives considered:** Threshold selection and variable-size knapsack.  
**Consequences:** Results are deterministic and exhaustive tiny-case tests can prove optimality. Equal-sized experts make cardinality top-B appropriate for M2.  
**Experiments affected:** Binary-count, selected/full mass, and cost-aware sweeps.  
**Migration required:** Add explicit knapsack handling before claiming support for heterogeneous expert sizes.

## D-20260727-008 — Replay historical policy decisions by token position

**Status:** accepted  
**Context:** The tiny model recomputes full prefixes without a KV cache. Applying only the newest subset to that prefix would retroactively change earlier token states.  
**Decision:** Persist the allowed subset for every executed token position and replay that schedule on every constrained forward. Prompt positions without a recorded decision use natural routing.  
**Alternatives considered:** Apply the current subset to the whole prefix or add an M3 KV cache.  
**Consequences:** Full-prefix recomputation is inefficient but semantically follows the constrained trajectory and preserves historical decisions.  
**Experiments affected:** M3 tiny closed-loop evaluation.  
**Migration required:** A future cached decoder must prove parity with position-indexed replay.

## D-20260727-009 — Predeclare a narrow tiny-model Gate A criterion

**Status:** accepted  
**Context:** The specification requires a Gate A decision but does not prescribe a universal numerical threshold.  
**Decision:** Tiny Gate A passes if any lossy oracle-planned grid point has exact token rate 1.0 and at least 25% lower estimated transfer bytes than lossless fallback at the same horizon and budget ratio.  
**Alternatives considered:** KL-only or partial-token thresholds.  
**Consequences:** The decision is strict about this tiny greedy continuation and directly tied to transfer accounting, but cannot be generalized to real models.  
**Experiments affected:** `tiny_m3_gate_a`.  
**Migration required:** Real-model Gate A requires task-specific quality tolerances and measured or simulated transfer economics.

## D-20260727-010 — Make factorial future inputs a separate offline type

**Status:** accepted  
**Context:** SC conditions contain true future tokens and must never enter deployable probes.  
**Decision:** Construct them only as `OfflinePseudoSequence`; online entry points require and runtime-check `DeployableDecodeState`. No online state gains an optional future field.  
**Alternatives considered:** A shared sequence request with an information-regime flag.  
**Consequences:** Leakage fails structurally and at runtime rather than relying on comments.  
**Experiments affected:** All M4 factorial runs and later online probes.  
**Migration required:** Future offline interventions extend the offline namespace only.

## D-20260727-011 — Report adapter capture support without fabricating post-RoPE values

**Status:** accepted  
**Context:** The tiny model exposes both query stages, while the current native Mixtral implementation does not expose its post-RoPE query through a stable hook.  
**Decision:** Tiny reports every requested activation. Mixtral reports pre-RoPE projection, post-attention state, router input/logits, and top-k; post-RoPE is explicitly false in config/manifest and its metric is absent.  
**Alternatives considered:** Treat pre-RoPE as post-RoPE or patch Transformers internals.  
**Consequences:** Cross-model tables have a documented missing metric but no semantic conflation or fragile global patch.  
**Experiments affected:** M4 pinned Mixtral sample.  
**Migration required:** Add an architecture-versioned post-RoPE capture only with a parity test.

## D-20260727-012 — Gate B is negative and forces a context/history pivot

**Status:** accepted  
**Context:** Position dominance is a hypothesis, not an axiom.  
**Decision:** Gate B requires at least half of router-logit/subset dominance cells to have positive paired means and at least one paired CI excluding zero. Tiny yields 0/24 positive; the small pinned Mixtral sample yields 1/8 positive and none significant.  
**Alternatives considered:** Select a favorable layer or metric post hoc.  
**Consequences:** Do not claim position dominance or prioritize a position-only probe. Preserve the factorial result and emphasize context/history-conditioned designs in later milestones.  
**Experiments affected:** M4 Gate B and later probe prioritization.  
**Migration required:** Re-evaluate on meaningful pretrained models and larger document samples before any broader claim.

## D-20260727-013 — Use an exact reference plus seeded randomized SVD

**Status:** accepted  
**Context:** M5 needs proof-grade row-space behavior and a scalable approximation path.  
**Decision:** Compute float64 thin SVD as the exact reference. Implement seeded randomized range finding with configurable oversampling and power iteration for approximation.  
**Alternatives considered:** Depend on a third-party randomized linear algebra package.  
**Consequences:** Exact projection and reconstruction tests remain simple; randomized results are reproducible without another dependency.  
**Experiments affected:** Router geometry analysis.  
**Migration required:** Large sharded routers may require distributed matrix products.

## D-20260727-014 — Make sampled interventions non-mutating routing policies

**Status:** accepted  
**Context:** Criticality interventions must restore original state and should not risk corrupting shared model weights.  
**Decision:** Express zero, removal, and substitution through a position-scoped policy schedule. Capture baseline/intervened post-MoE states and downstream natural routes in separate forwards; never edit parameters.  
**Alternatives considered:** Temporarily overwrite expert parameters or install mutable global hooks.  
**Consequences:** Restoration is guaranteed by construction and verified by full state-dict equality. Full-prefix recomputation is slower but deterministic.  
**Experiments affected:** M5 tiny expert criticality.  
**Migration required:** Architecture-specific intervention kernels must reproduce this reference.

## D-20260727-015 — Treat route flips and router influence as separate outcomes

**Status:** accepted  
**Context:** A perturbation can change downstream router distributions without crossing a top-k boundary.  
**Decision:** Always report downstream router KL alongside exact top-k flip count/rate.  
**Alternatives considered:** Use route flips as the sole influence metric.  
**Consequences:** The tiny run's zero mean flip rate is not misrepresented as zero downstream influence.  
**Experiments affected:** M5 criticality tables and summaries.  
**Migration required:** None.

## D-20260727-016 — Use one selected-routing-mass target for every M6 probe

**Status:** accepted  
**Context:** Baseline comparisons are invalid if each predictor receives a favorable target or split.  
**Decision:** Every M6 predictor estimates the per-layer expert sum of natural selected gate mass over the same future window. Dataset rows are grouped by document before train/validation/test assignment, and evaluation exactly compares rebuilt features and targets with the persisted dataset.  
**Alternatives considered:** Binary route occurrence, full softmax mass, or probe-specific targets.  
**Consequences:** All eight baselines share identical test rows, target tensors, expert budget, and latency accounting.  
**Experiments affected:** `train-predictor` and `evaluate-probe`.  
**Migration required:** Alternative targets must be separate named experiments, never silent replacements.

## D-20260727-017 — Keep learned M6 artifacts pickle-free

**Status:** accepted  
**Context:** Learned predictors must be portable without arbitrary-code deserialization.  
**Decision:** Persist tensors only as safetensors and typed metadata as JSON. Direct linear/ridge, MLP, ADEPT-style stump ensemble, Markov transitions, and predictor datasets all use this boundary.  
**Alternatives considered:** `torch.save`, joblib, scikit-learn pickle, and an added Random Forest dependency.  
**Consequences:** Loading rejects unknown model kinds and non-safetensors manifests; the ADEPT baseline is explicitly style-compatible rather than an exact reproduction.  
**Experiments affected:** All M6 training and evaluation.  
**Migration required:** New predictor types require an explicit schema branch and safe tensor representation.

## D-20260727-018 — Treat production K/V as immutable input to M7

**Status:** accepted  
**Context:** Shadow probes may read production attention state but must never advance or mutate generation state.  
**Decision:** The tiny adapter builds a frozen `ReadOnlyProductionKVCache` with a content fingerprint. Independent probes allocate no pseudo K/V; causal probes append only to probe-local `ShadowKVBuffer` objects and verify the production fingerprint on return.  
**Alternatives considered:** Deep-copy and append to the production-shaped cache, or patch attention globally.  
**Consequences:** Cache mutation fails explicitly and the reference path is architecture-specific and slower than a native cached decoder.  
**Experiments affected:** Every M7 shadow probe.  
**Migration required:** Each real adapter needs a validated read-only K/V and residual-composition implementation.

## D-20260727-019 — Keep pre/post-sample M7 regimes structurally distinct

**Status:** accepted  
**Context:** A sampled next token is legal only after sampling and must not become a channel for later future tokens.  
**Decision:** Pre-sample probes use current/fixed content only. Post-sample evaluation constructs `DeployableDecodeState` with exactly the already sampled next token; repeated-next-token pseudo anchors may use that one ID but receive no later IDs. Probe RNG runs inside a restoring RNG fork.  
**Alternatives considered:** A shared optional future-token array or teacher-forced pseudo sequence.  
**Consequences:** Leakage checks remain runtime-enforced and result rows retain the actual information regime.  
**Experiments affected:** Pseudo-token and default-vector shadow rollouts.  
**Migration required:** None.

## D-20260727-020 — Define M9 ranking signals without test-set tuning

**Status:** accepted  
**Context:** Static rankings require heterogeneous signals, but all weights and proxy definitions must be declared before evaluation.  
**Decision:** Frequency is selected-token frequency; covariance variance is `diag(W covariance W^T)`; quality sensitivity is next-token KL under sampled zero-contribution intervention; downstream influence is downstream router KL; miss risk is the calibrated rate of newly appearing experts relative to previous-route reuse. Combined score min-max normalizes all five per layer, sums them with fixed unit weights, and divides by expert bytes.  
**Alternatives considered:** Tune weights on the evaluation trajectory or silently substitute router row norm for intervention measurements.  
**Consequences:** Single-factor and combined rankings are reproducible, though sparse intervention coverage can leave zero-valued experts.  
**Experiments affected:** M9 static residency only.  
**Migration required:** Real-model calibration needs broader intervention coverage and a train/validation-only weight selection protocol.

## D-20260727-021 — Use a narrow M9 reference timeline because M8 is absent

**Status:** accepted  
**Context:** The requested M9 comparison depends on simulation, but the full M8 simulator was not present.  
**Decision:** `simulate-offload` emits `m9_reference_serial_transfer_no_overlap`: exact planned bytes, fixed per-load latency, configured bandwidth, and serial compute. Initial static preload is charged. It does not claim LRU/LFU baselines, bandwidth sharing, overlap, or the complete M8 state machine.  
**Alternatives considered:** Mislabel the analytical timeline as full M8, skip simulation, or implement a real runtime.  
**Consequences:** Fixed/adaptive M9 plans can be compared consistently while the missing milestone remains visible.  
**Experiments affected:** M9 reference simulation only.  
**Migration required:** Complete M8 before M10 and replace/reference-check this timeline against the event-driven simulator.

**Update:** superseded by D-20260727-024; M9 now runs through the capacity-checked M8 event simulator.

## D-20260727-022 — Model concurrent transfers as aggregate-bandwidth waves

**Status:** accepted  
**Context:** M8 must enforce concurrency and aggregate bandwidth without pretending to reproduce a particular CUDA/DMA scheduler.  
**Decision:** Loads issued together form waves capped by `max_concurrent_transfers`. Each wave pays one concurrent fixed-latency interval and transfers its total bytes at the configured aggregate bandwidth. Unequal transfers conservatively complete at the common wave end.  
**Alternatives considered:** Give every stream full bandwidth, or implement undocumented hardware-specific progressive reallocation.  
**Consequences:** Aggregate bandwidth is exact and never exceeded; small unequal transfers may complete later than a fluid-sharing implementation.  
**Experiments affected:** All M8 simulated timelines.  
**Migration required:** Hardware-calibrated progressive sharing may replace the conservative wave model with matching synthetic tests.

## D-20260727-023 — Separate lossless cache baselines from hard commitments

**Status:** accepted  
**Context:** Commitment can reduce bytes by omitting natural experts, while cache and predictor baselines must load every miss.  
**Decision:** On-demand, LRU, LFU, and lossless predictor demand-load all missing natural experts and report route coverage 1. One-step and multi-step commitment execute only planned resident routes and report explicit natural-route coverage.  
**Alternatives considered:** Silently demand-load commitment misses or compare TPOT without coverage.  
**Consequences:** Faster commitment points cannot be mistaken for lossless performance.  
**Experiments affected:** M8 baseline and Pareto tables.  
**Migration required:** Closed-loop quality must accompany commitment simulation on real models.

## D-20260727-024 — Replay authoritative M9 routes through the M8 event engine

**Status:** accepted  
**Context:** M9 needs one consistent fixed/adaptive comparison after M8 became available.  
**Decision:** Reconstruct each scenario's executed route trace from its own recorded generated trajectory and constrained plan, then feed it to the M8 engine with boundary-indexed custom prefetch plans. Static experts are protected during plan replacement. Online state still receives no future token IDs; the oracle information regime remains explicit.  
**Alternatives considered:** Retain the serial analytical reference or simulate natural routes instead of executed routes.  
**Consequences:** M9 exports capacity-checked load, eviction, probe, and compute events; exact bytes and TPOT are comparable across fixed and adaptive modes. Results remain simulated and oracle-planned.  
**Experiments affected:** M9 `simulate-offload`.  
**Migration required:** Real adapters must provide equivalent policy replay before using this path; real transfer measurements remain out of scope.

## D-20260727-025 — Establish correctness with fixed PyTorch slots before custom kernels

**Status:** accepted  
**Context:** M10 requires a real transfer path, but optimization must follow correctness and profiling.  
**Decision:** The first runtime targets the deterministic tiny adapter on one CUDA GPU at batch 1. Expert tensors live only in optionally pinned CPU handles; equal-size GPU slots are preallocated per layer. The synchronous path blocks on each transfer. The asynchronous path uses one transfer stream, non-blocking copies, transfer-completion events, compute-stream waits, and slot compute-completion events before replacement.  
**Alternatives considered:** Moving whole modules per token, allocating destination tensors in the decode loop, Triton, or custom CUDA.  
**Consequences:** Base→synchronous and synchronous→asynchronous equality are independently gated. Metrics are genuine CUDA measurements, but the tiny full-prefix implementation does not establish production-model speedups or maximal overlap.  
**Experiments affected:** M10 `benchmark-offload`.  
**Migration required:** Add adapter-specific slot execution and cached incremental decoding before benchmarking a production MoE; profile before introducing custom kernels.

## D-20260727-026 — Aggregate heterogeneous results through a strict envelope

**Status:** accepted  
**Context:** Oracle, factorial, probe, simulator, static-residency, and runtime commands intentionally emit different domain tables. Flattening them would silently erase semantics.  
**Decision:** M11 wraps each reproduced run in schema-1 `run_manifest.json`: stable run ID, command/result kind, information regime, resolved-config digest, terminal state, negative findings, and SHA-256/size for every artifact. Aggregation validates envelopes and source files before reading kind-specific scalar/table fields.  
**Alternatives considered:** A permissive directory scan, one universal metrics schema, or dropping failed/negative runs.  
**Consequences:** Schema mismatches, checksum changes, and duplicate IDs fail loudly. Incomplete runs are retained and reported but excluded. Every paper-table cell and indexed figure links to a run ID/source artifact. Interrupted suites resume only checksum-valid completed tasks.  
**Experiments affected:** All M11 primary reproduction and aggregation.  
**Migration required:** New result kinds must declare their paper metrics and worst-case criterion without changing existing envelopes.
