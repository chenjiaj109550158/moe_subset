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

## D-20260727-027 — Extend one common adapter contract for trained MoE semantics

**Status:** accepted
**Context:** OLMoE, Qwen, Mixtral, DeepSeek and gpt-oss differ in scoring,
normalization, shared experts, layer indices, placement and weight representation.
**Decision:** Keep every architecture behind `MoEModelAdapter`; add native logits
and pre-top-k scores, layer-scoped routed IDs, shared-expert counts, routing
semantics, physical expert bytes, and reversible native-router policy hooks to the
common contract. Cache immutable `ModelSpec` metadata. DeepSeek remote code is
allowed only at its pinned revision and only with restoring in-memory compatibility
contexts. Mixtral remains full precision on two A100s. gpt-oss remains a separate
MXFP4 tier.
**Alternatives considered:** Standalone model-specific analysis scripts, editing
the Hub cache, silently quantizing Mixtral, or treating shared experts as routed
residency.
**Consequences:** Four floating-point checkpoints reproduce native routes and
policy parity through one runner. gpt-oss direct routing/physical storage is
inspectable, while its fused forward incompatibility fails explicitly.
**Experiments affected:** `trained_oracle_gate_v1` and v2.
**Migration required:** Any new model must add literal/native-reference tests and
pass the same contract before entering the common sweep.

## D-20260727-028 — Replace the truncated v1 closed-loop sample before final claims

**Status:** accepted
**Context:** The development v1 protocol used GSM8K row 0 at a 48-token prompt
limit; every tokenizer truncated the question before `Answer:`, and 12 generated
tokens could not yield a meaningful answer metric.
**Decision:** Preserve v1 as protocol-development evidence. For the final v2 run,
keep checkpoints, dataset revisions and rows, trace length, oracle horizons,
budgets, methods, bootstrap count, transfer model, and numerical gate thresholds
unchanged; use row 500, whose complete prompt is 40/42/47/43 tokens across the four
tokenizers, and predeclare 64-token deterministic decoding before final v2
aggregates. Run into a fresh output directory.
**Alternatives considered:** Publish the truncated result, silently replace the
prompt, or drop GSM answer scoring.
**Consequences:** Final answer extraction is meaningful and fully disclosed. All
four natural trajectories still miss the ground-truth answer; the negative gate
is not rescued by the correction.
**Experiments affected:** Final trained closed-loop quality only.
**Migration required:** Future task-quality runs must validate rendered prompt
completeness per tokenizer before execution.

## D-20260727-029 — The trained-model oracle gate is STOP/PIVOT

**Status:** accepted
**Context:** The final v2 gate requires below-all-expert residency, mean hit ≥0.90,
selected mass ≥0.95, P05 and worst hit ≥0.80, simulated transfer reduction ≥0.30,
and ≥0.05 improvement over previous/static baselines. Closed loop requires hard
relative-perplexity increase ≤0.05 or lossless fallback ≤0.10 in each domain.
**Decision:** Classify OLMoE, Qwen, Mixtral, DeepSeek, gpt-oss and the overall
stage as **STOP/PIVOT**. OLMoE has one qualifying WikiText open-loop point; Qwen
and DeepSeek have both-domain points; Mixtral has none after worst-tail filtering.
No floating-point model passes hard quality or fallback. gpt-oss cannot expose a
common trace through its fused native MXFP4 path. Do not train a predictor and do
not implement production runtime optimization.
**Alternatives considered:** Macro-average away model failures, treat lossless
fallback as quality improvement, ignore worst windows, issue a narrow GO from
open-loop coverage alone, or delay the primary decision for gpt-oss.
**Consequences:** The repository preserves a multi-model negative result. Any
follow-up needs a new predeclared pivot and cannot claim generality beyond the
exact checkpoints, two datasets, sampled rows, 64-token batch-1 greedy decoding,
budgets, precision and two-A100 hardware. Planning runtime was not measured, so
it cannot weaken the STOP decision.
**Experiments affected:** `trained_oracle_gate_v2` and every proposed post-oracle
stage.
**Migration required:** Explicit human authorization plus a versioned new protocol
for a quality-aware fallback or other revised hypothesis.

## D-20260730-030 — Pivot to a benchmark-aligned multi-token subset oracle

**Status:** accepted
**Context:** D-20260727-029 rejected learned routing after a small trained-model
gate, while the completed v17 accuracy suite preserves aligned vanilla prompts,
tokens, targets, and evaluators for Qwen3-30B-A3B and GPT-OSS-20B. Explicit human
authorization requested a new future-aware, fixed-window, budgeted-subset
question without rerunning vanilla or training a predictor.
**Decision:** Freeze `benchmark_subset_oracle_v1` before producing subset results.
Replay deterministic quartile samples from saved v17 trajectories for the full
H/B/method grid. Admit at most one future-selected-mass point per model only when
all six tasks pass the predeclared coverage, tail, residency, fallback, baseline,
and transfer gates. Reuse v17 natural accuracy; identity-materialize lossless
accuracy only after cross-task real-forward parity; require actual closed-loop
generation for hard oracle and previous-route commitment. At every hard-oracle
boundary, look ahead naturally from the current policy state, rewind cache and
RNG, then replay the constrained window.
Use a fixed `H=16`, native-top-k mechanism smoke solely to prove lossless
identity and hard route modification; it is not an operating candidate and is
kept separate from the post-selection smoke.
**Alternatives considered:** Relabel v17 `oracle_pf` as multi-token evidence,
select points using smoke accuracy, regenerate the full vanilla suite, use a
non-native GPT full-softmax proxy, or train a predictor immediately.
**Consequences:** The new stage measures the requested routing-information upper
bound with benchmark accuracy and explicit simulation/runtime labels. A failed
perfect oracle remains a terminal predictor-training stop. The 128-token
representative trace cap limits open-loop context claims and is disclosed;
full hard accuracy retains every v17 task cap, including GPT AIME at 32,768.
**Experiments affected:** `benchmark_subset_oracle_v1` only; prior v17 and
`trained_oracle_gate_v2` artifacts and decisions remain unchanged.
**Migration required:** None before the frozen v1 trace and smoke gates pass.

## D-20260730-031 — Revise replay mechanics after a pre-result parity stop

**Status:** accepted
**Context:** Protocol revision 1 completed its v17 audit but saved no trace
shards. Qwen chunk-16 teacher replay failed its mandatory comparison with the
fixed autoregressive sample (`route_ids_equal=false`), so revision-1 traces were
inadmissible. GPT-OSS also stopped before loading because offline kernel
`version=1` resolution queried version metadata despite the exact CUDA build
already existing in the local cache.
**Decision:** Preserve the complete revision-1 failure directory. Freeze
revision 2 before retrying: force the saved decode trajectory with a logits
processor through the checkpoint's native one-token `generate` cache path,
capture routes with non-mutating native forward hooks, and validate the replay
against an independent native unpatched autoregressive generation. Use the
supported `LOCAL_KERNELS` mapping to the existing cached kernel
commit `9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0`. Hash every file in that cached
variant and state explicitly that no download occurred. Preserve the first
`_r2` CLI-validation attempt, which was interrupted during Torch import with
zero trace shards and zero result rows; its environment record predates the
clean revision-2 commit. Use a fresh `_r2_final` root for admissible results.
Do not alter the experimental grid, selectors, samples, gates, transfer model,
decoding, or scoring.
**Alternatives considered:** Accept non-parity chunked routes, silently relax
route-ID parity, download kernel metadata, delete the failed artifacts, or
continue under the old fingerprint.
**Consequences:** Replay is slower but matches the required autoregressive
execution shape. The mechanical dependency repair cannot use accuracy and
cannot change operating-point selection.
**Experiments affected:** `benchmark_subset_oracle_v1` natural replay and GPT
model loading only.
**Migration required:** Revision-2 trace parity and offline kernel provenance
must pass before open-loop aggregation.

## D-20260730-032 — Replay immutable v17 prompt bytes and complete short-output parity

**Status:** accepted
**Context:** The first clean revision-2 trace attempt stopped before any complete
model trace or open-loop grid. GPT's chat template ignored the configured date
and rendered the wall clock, changing only `2026-07-29` to `2026-07-30`; both
strings had 207 tokens. Qwen saved 20 partial shards and then a short
AIME25 output exposed that teacher replay retained `N` rather than `N+1` LM logits
because the final saved output token is predicted but never processed.
**Decision:** Treat the audited v17 `rendered_prompt` string as the immutable
input, encode it with no added special tokens, and require exact tokenizer
decode round-trip plus the saved SHA-256. Retain all `N+1` LM-head prediction
positions while storing routes for only the `N` processed decode inputs. Use a
fresh `_r2_authoritative` artifact root and preserve both stopped attempts.
Do not change the config fingerprint, samples, grid, gates, decoding, scoring, or
selection rules.
**Alternatives considered:** Accept a one-day prompt drift, skip prompt parity,
ignore parity for short rows, change the host clock, or reconstruct model inputs
from a mutable template.
**Consequences:** Both teacher replay and actual closed-loop generation start from
the exact prompt bytes that produced the v17 reference tokens. Existing strict
route/token/logit gates remain in force before any trace is admitted.
**Experiments affected:** `benchmark_subset_oracle_v1` natural replay and
closed-loop prompt construction only.
**Migration required:** Both models must complete all 24 authoritative trace
shards and all six parity records before open-loop aggregation.

## D-20260730-033 — Fork sliding-window caches for non-destructive oracle lookahead

**Status:** accepted
**Context:** Both authoritative natural traces and full open-loop grids completed
with exact row counts. The predeclared gate selected only GPT `(H=1,B=4)`; Qwen
had no qualifying point. During the independent mechanism smoke, GPT's native
`DynamicSlidingWindowLayer` rejected crop after its cumulative length exceeded
the window because evicted states cannot be reconstructed. It saved no successful
smoke row; Qwen completed all 12 smoke rows with exact non-sliding crop rewind.
**Decision:** For any cache containing a sliding layer, shallow-copy the cache and
mutable layer objects for natural lookahead while sharing existing boundary KV
tensors. Installed native dynamic-cache updates concatenate and assign new
tensors on the fork, which is discarded after subset selection. Verify that the
original cache sequence length, layer identities, KV identities/data pointers and
version counters where available, and sliding cumulative lengths are unchanged.
Restore RNG in a nested `finally`. Keep exact crop rewind for non-sliding caches.
Record the clean resume commit under immutable `execution_revisions/` provenance.
Do not change the frozen grid, selected point, gates, samples, or evaluators.
**Alternatives considered:** Ignore the crop error, reset sliding cumulative length,
disable the sliding window, deep-copy the full KV payload at every token, or
recompute every boundary from the prompt.
**Consequences:** Lookahead can proceed from the exact GPT boundary state without
requiring already-evicted KV entries. Copy-on-write adds actual planning work and
is included in measured closed-loop runtime; simulated transfer remains separate.
**Experiments affected:** `benchmark_subset_oracle_v1` GPT mechanism, selected,
and full closed-loop lookahead only; completed traces/grids and Qwen smoke remain.
**Migration required:** A real GPT mechanism smoke must prove original-cache
preservation, exact lossless tokens, and hard executed-route change before resume.

## D-20260730-034 — Reduce full baselines and focus pseudo-embedding work

**Status:** accepted
**Context:** The first full shard saved eight GPT hard-oracle rows and seven
previous-route rows before an explicit user pause. The same user then reduced the
scope because a paired full suite would spend substantial GPU time on a weak
baseline instead of the primary pseudo-embedding hypothesis. The selected GPT
suite contains 2,608 v17 rows and about 1.995 million reference output tokens.
**Decision:** Preserve the revision-2 scientific config, fingerprint, selected
point, samples, decoding, oracle semantics, evaluators, and all existing artifacts.
Add the separately fingerprinted execution scope
`benchmark_subset_oracle_v1_hard_only_full_v1`: finish only actual
`hard_oracle_commitment` at the already selected GPT `(H=1,B=4)` point over all
six datasets. Do not schedule the remaining full previous-route rows. Retain its
seven completed rows and the user-interrupted `.FAILED.json` marker as provenance,
but exclude unscheduled rows from full aggregation requirements.

Distribute the four deterministic full shards over two identical physical A100s
in two waves, with at most one worker per GPU. Record actual physical placement in
every new row and retain one-sample-policy atomic resume. This placement change
does not alter model weights, seeds, sample-to-shard assignment, prompt bytes,
sampling, cache semantics, or the base suite fingerprint.

After the hard ceiling is complete, focus the next research stage on
Qwen3-30B-A3B plus GSM8K at `(H=8,B=32)`, comparing hard oracle,
previous-route, and pseudo embedding. Freeze a separate pseudo-embedding
development/held-out protocol before executing that stage. Do not train a learned
predictor.
**Alternatives considered:** Continue all 5,216 paired full rows, discard partial
previous-route artifacts, silently edit the frozen base config, run all H/B points
closed loop, or begin an unspecified pseudo probe immediately.
**Consequences:** The full suite answers the hard-oracle accuracy-ceiling question
but no longer claims an all-task full previous-route comparison. Deployable-method
claims will be restricted to the later Qwen/GSM8K focused experiment. The original
STOP/PIVOT rules remain unchanged outside this explicit human amendment.
**Experiments affected:** `benchmark_subset_oracle_v1` full scheduling and its
aggregate required-policy set; later `pseudo_embedding_qwen_gsm8k_v1` planning.
**Migration required:** Validate and commit the execution scope and two-GPU
placement regressions before resuming any long worker.

## D-20260731-035 — Restrict full hard accuracy to GPT-OSS/GSM8K

**Status:** accepted
**Context:** The hard-only six-task execution had saved 808 actual GPT rows after
about sixteen hours, including 536/1,319 GSM8K rows and all assigned shard-0
GSM8K rows. Its measured remaining all-task time was about two days. The user
explicitly chose to finish one challenging dataset first because the six-task
run was too expensive. Existing Qwen `(H=8,B=32)` open-loop evidence also makes
GSM8K a useful focused task: future-oracle mean route hit is 0.9587 and lossless
fallback is 0.0413, weaker than the other substantive tasks at that point.
**Decision:** Preserve the frozen base suite, selected GPT `(H=1,B=4)` point,
prompt bytes, samples, decoding, evaluator, source vanilla rows, and every saved
artifact. Replace only the execution schedule with the separately fingerprinted
`benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2` scope. Require all 1,319 GSM8K
hard-oracle rows and exclude unscheduled tasks and previous-route rows from the
scoped aggregate. Preserve their successful rows and `.FAILED.json` markers as
provenance. Pass the task filter through the existing closed-loop runner and
reporter. Use deterministic per-GPU shard queues so a GPU can start its next
shard without waiting for the other GPU; never run more than one worker per GPU.
**Alternatives considered:** Finish all six GPT tasks, discard completed non-GSM
rows, run only a small GSM8K sample, use AIME25 despite its 30-question discrete
uncertainty, or write a standalone evaluator.
**Consequences:** The final hard result supports only GPT-OSS/GSM8K. A passing
task gate is at most **NARROW**, never an all-six-task GO; unscheduled datasets
have no measured hard-commitment conclusion. The later Qwen/GSM8K pseudo-
embedding comparison remains separately gated and does not authorize learned
predictor training.
**Experiments affected:** `benchmark_subset_oracle_v1` full closed-loop schedule,
scoped aggregation, decision, and provenance only.
**Migration required:** Lock the revision-2 execution-scope fingerprint, test
task filtering and per-GPU queues, commit a clean execution revision, then resume
the 783 missing GSM8K rows from their atomic sample artifacts.

## D-20260801-036 — Stop the Qwen/GSM8K pseudo-embedding pilot at development

**Status:** accepted
**Context:** The separately frozen `pseudo_embedding_qwen_gsm8k_v1` pilot tested
training-free native-Qwen shadow routing at `(H=8,B=32)`. Config fingerprint
`a81f36b5ec4a4222ca7a459f9f9c1536d88bef5ba143c151c9e70498157d8cbc`, exact
sample partitions, four mandatory variants, route/cost progress gates, held-out
gap recovery, and a 16-row accuracy gate were committed before pseudo results.
Native two-row mechanism smoke passed its cache, RNG, attention, RoPE, router,
shadow-lifetime, information-boundary, and non-static-subset checks.

On the four frozen development traces, hard oracle achieved route hit/selected
mass 0.958714/0.976395, previous-route 0.609385/0.629428, and static frequency
0.339705/0.346426. The primary sampled-next-token independent default-vector
variant achieved 0.533015/0.534416. The strongest mandatory ablation was zero
expert contribution at 0.566499/0.575402. All four mandatory variants regressed
against previous-route, with paired 95% confidence intervals for both deltas
strictly below zero. The optional expected-top-8 embedding also regressed at
0.533448/0.534745. No mandatory variant met the two +0.05 improvement gates;
some also missed 30% simulated transfer reduction.

**Decision:** Record a focused **STOP/PIVOT** and select no pseudo variant. Apply
the frozen early-stop rule: do not run the disjoint held-out route set and do not
run hard-oracle, previous-route, or pseudo actual closed-loop accuracy. Preserve
all six successful atomic route rows, raw/aggregate/stratified/worst-case/cost
artifacts, checksums, and four failure markers. Do not train a learned predictor,
expand GSM8K, change IDs/token caps, recalibrate defaults, or claim runtime
speedup.

**Alternatives considered:** Choose the least-negative zero ablation, rank using
smoke or GSM8K correctness, continue to held-out despite the progress gate,
materialize vanilla identities as accuracy evidence, silently tolerate replay
drift, or enlarge the sample scope.

**Consequences:** This negative result is limited to pinned Qwen/GSM8K
`H=8,B=32`, four development traces, the declared training-free variants, and
the reused default vectors. It shows that this pseudo construction does not
shrink the previous-to-oracle route gap at the focused point. It is open-loop
route evidence with measured probe cost and simulated transfer, not measured
task accuracy or actual constrained generation. The default artifact has 444
unobserved layer/expert pairs saved as zero and is not a complete prior. Fresh
cross-process BF16 replay missed strict authoritative router tolerance on all
four development rows; authoritative v17 tensors remained the scoring targets,
and same-process mechanism smoke supplies native cache/RNG semantics evidence.

**Experiments affected:** `pseudo_embedding_qwen_gsm8k_v1` only. Earlier GPT
hard-only `NARROW`, trained-model `STOP/PIVOT`, and M11 conclusions are unchanged.

**Migration required:** Any follow-up requires a new predeclared hypothesis and
scope. Preserve this terminal artifact root and do not resume held-out or actual
accuracy under v1.

## D-20260801-037 — Keep calibration-free prompt routing at STOP/PIVOT

**Status:** accepted
**Context:** After D-20260801-036 stopped the original pseudo variants, the user
authorized a calibration-free mechanism analysis without changing the focused
model, task, `H=8,B=32` point, or terminal v1 decision. Tensor interventions and
native two-row smokes found that repeated sampled content changes too slowly,
recent token IDs do not help, exact future token contents have diagnostic
headroom, and causal propagation with zero MoE residual is harmful. Capturing
same-request prompt routes fixed the first-window static fallback. The four-row
development protocol selected one equal sampled-pseudo/history candidate at
0.664737 route hit and 0.688248 selected mass.

The separately frozen eight-row held-out protocol used config SHA-256
`1f6f5ab8372aa6f63796cfda49601745bf2736490ee21c8807fd9657b9a996ef`.
It completed 393,216 route slots. The candidate achieved 0.658353 route hit,
0.680584 selected mass, and 0.439423 simulated transfer reduction versus
previous route 0.617671/0.637396. Gains were +0.040682/+0.043188 with paired 95%
intervals [0.036001, 0.046422] and [0.038634, 0.049149]. Oracle-gap recovery was
0.116775/0.125752. Both +0.05 improvements, both 25% gap recoveries, and both
absolute references failed; transfer and every cache/RNG/information audit
passed.

**Decision:** Retain **STOP/PIVOT** and run no task accuracy or actual hard
closed-loop generation. Preserve the completed eight-row artifact root and its
manifest SHA-256
`3f1b509d7d012e8266ee6a28e1863efb44f00a946b29646a484c859a5ecd08db`.
Do not treat prompt-route gain as pseudo-embedding gain: roughly half of the
observed improvement comes from replacing boundary-zero static frequency with
same-request prompt routes. Keep post-hoc horizon schedules diagnostic only.

**Alternatives considered:** Promote the development pass directly to accuracy,
accept positive-but-subthreshold held-out intervals, tune an anchor cutoff on
held-out rows, call prompt-history improvement a pseudo rollout success, use
future-token or default-vector values, or expand the row set.

**Consequences:** No learned/fitted value, offline expert prior, default-vector
value, route-transition table, future true token, answer, correctness, or
accuracy entered a deployable candidate. The result is open-loop route evidence
with measured probe/replay cost and simulated transfer, not task accuracy,
closed-loop quality, or speedup. A post-hoc three-anchor/history formula reached
0.664205/0.687282 but is not validation and still misses the gate.

**Experiments affected:** The calibration-free tensor, content-smoke,
prompt-route smoke/development, and held-out analysis roots only. D-20260801-036,
GPT hard-only `NARROW`, trained-model `STOP/PIVOT`, and M11 remain unchanged.

**Migration required:** A future attempt should first test current-request
online MoE-residual/state reuse and autoregressive shadow contents under a new
committed protocol. New rows or expanded sample scope require explicit human
authorization. No actual accuracy is permitted from this failed gate.

## D-20260802-038 — Stop the previous-window MoE-residual pseudo experiment at development

**Context:** The user clarified that one step means an `H=8` output window. Full-
expert prefill can capture the exact MoE mixture output for the last eight prompt
tokens; after commitment, each hard policy can capture the mixture outputs it
actually executes and use them to plan the following window. A new protocol
froze residual/content ablations, seven analytic pseudo/history combinations,
four development IDs, new disjoint held-out IDs, and gates before results.

**Decision:** Keep the result at **STOP/PIVOT**. Position-aligned residuals and
sampled-repeat independent anchors were selected without accuracy. The best
development formula, first-four-anchor core plus history fill, reached
0.639706 route hit and 0.649761 selected mass versus previous-route
0.623634/0.634308. Its +0.016072/+0.015453 gains missed both required +0.05
checks, despite passing static, 25% residency, 0.399547 simulated-transfer, and
all cache/RNG/information/parity checks. Do not run the new held-out route set or
the actual closed-loop accuracy partition.

**Alternatives considered:** Promote positive paired intervals, select the
slightly higher-hit inverse-decay method instead of the frozen mass-first rule,
tune layer-specific or anchor-specific coefficients after observing results,
use exact future tokens, read default-vector values, or expand samples.

**Consequences:** Previous-window executed residuals materially outperform zero
residuals in mechanism smoke, but the development gain is only about 4.9%/4.6%
of the oracle-minus-previous gap. Gains concentrate in early layers and early
anchors; layers 46–47 regress. These 64 teacher-forced hard-policy-state rows
are route evidence with measured probe/replay cost and simulated transfer, not
task accuracy, exact-token identity, free generation, or speedup.

**Experiments affected:** Only
`pseudo_embedding_qwen_gsm8k_residual_window_v1`. The original focused pilot,
calibration-free prompt-route held-out result, GPT hard-only `NARROW`, trained-
model `STOP/PIVOT`, and M11 are unchanged.

**Migration required:** Preserve the 171-artifact root and manifest SHA-256
`720e3e0ae68efeddd770226f7d953969c8313997455ac8520a39788b4dd71771`.
A future calibration-free attempt must predeclare how it addresses late-anchor
and late-layer decay; it may not reuse these development results as held-out
validation. New rows or any accuracy execution require new authorization.

## D-20260802-039 — Narrow the executed-pseudo composition to route evidence

**Status:** accepted

**Context:** The previous residual-bank experiment copied the preceding window's
MoE output into a new pseudo pass. The user clarified the intended mechanism:
run the pseudo embedding through the actual MoE at every planning step, using
the preceding realized per-layer subset for expert execution. Full-expert
prefill permits native top-8 execution at boundary zero; all later boundaries
execute only the prior B=32 subset. A separately committed protocol froze four
development and eight disjoint held-out route IDs, content/attention variants,
particle follow-ups, route-only ranking, and gates before model results. It used
no learned or calibrated values.

Development selected `self_greedy_causal` at 0.788767 route hit and 0.805782
selected mass. On held-out rows it reached 0.765119/0.784870 versus
0.611168/0.621438 for previous route and 0.669112/0.682921 for sampled-token
repeat. Its aggregate gains over previous route were +0.153951/+0.163432; paired
sample-bootstrap 95% intervals were [0.139943, 0.168826] and
[0.148633, 0.179291]. Simulated transfer reduction was 0.555027. Every
cache/RNG/shadow/information audit passed. Top-2 and top-4 probability-weighted
particle rollouts improved route hit over greedy by only +0.000590/+0.000621 and
selected mass by +0.002408/+0.001951, while measured probe latency increased
from 2.1108 seconds to 4.0874/7.9009 seconds on development.

**Decision:** Record **NARROW** only for calibration-free Qwen/GSM8K
`H=8,B=32` route analysis. Fresh autoregressive pseudo state plus a native MoE
residual executed under the previous subset materially closes more route error
than repeated sampled-token content or previous route. Keep greedy self-rollout
as the best composition; do not promote particle branching. Do not execute task
accuracy under this analysis protocol: it deliberately did not compute the
predeclared oracle-gap recovery condition or freeze a paired allowed-drop rule
for generation.

**Alternatives considered:** Reuse the prior window's residual bank, use zero
MoE contribution, repeat sampled/current/expected-token embeddings independently,
use causal recent true tokens, rank exact-future diagnostics, promote particles
for their tiny metric gain, or infer a full-pilot GO from route metrics alone.

**Consequences:** The final root validates 104 atomic rows and 242 checksummed
artifacts; manifest SHA-256 is
`e00e88fa8c20c90682f49b52415792681d688c02b74c887cb32be348d5efcac7`.
Four failure markers remain provenance, including the particle causal-dispatch
bug fixed in commit `a58bd6f`. Measurements are teacher-forced on each policy's
own hard-subset state, not vanilla route replay, but they are not free generation
or task accuracy. Transfer is simulated; probe/replay time and memory are
measured. No exact-token, NLL/perplexity, runtime-speedup, or full-dataset claim
is supported.

**Experiments affected:** Only
`pseudo_executed_embedding_composition_v1`. Earlier pseudo `STOP/PIVOT`, GPT
hard-only `NARROW`, trained-model `STOP/PIVOT`, and M11 decisions remain
unchanged.

**Migration required:** Any actual closed-loop pilot needs a new committed
execution amendment that freezes the same policy/IDs/token caps, adds an
oracle-gap computation and paired accuracy allowed-drop before generation, and
keeps sample scope within prior authorization. Expanded rows, learned predictors,
or default-vector recalibration still require explicit human authorization.

## D-20260802-040 — Stop one-forward linear state corrections at development

**Status:** accepted

**Context:** The user requested two calibration-free ways to enrich the cheap
one-forward recent-sequence mechanism without paying for eight autoregressive
shadow forwards. A separately committed protocol froze four existing
development rows, fixed coefficients 1 through 8, and three otherwise identical
policies. Every boundary performs one native causal eight-token pseudo traversal.
It executes fresh Qwen MoE residuals under full native top-8 access at boundary
zero and the current policy's previous realized B=32 subset thereafter. Method
one adds the last-two-token router-input velocity before each native gate;
method two adds the last-two-token realized MoE-output velocity to the freshly
executed pseudo MoE contribution. Per-anchor L2 norm is restored after either
addition.

The uncorrected policy reached 0.700267 route hit, 0.714061 selected mass, and
0.465892 simulated transfer reduction. Router-input velocity reached
0.648031/0.661326/0.391418, with deltas -0.052236/-0.052736 and paired 95%
intervals [-0.054891, -0.047923]/[-0.060275, -0.045458]. Residual velocity
reached 0.676310/0.691583/0.434408, with deltas -0.023956/-0.022478 and intervals
[-0.028493, -0.019735]/[-0.029072, -0.015884]. Both miss the frozen +0.02 route
signal in the wrong direction.

**Decision:** Record **STOP/PIVOT** for both linear corrections and do not run a
held-out route set or task accuracy. Keep the uncorrected one-forward
recent-sequence mechanism as the stronger cheap reference; retain the earlier
autoregressive self-greedy composition as the best route-only method, with its
higher measured probe cost.

**Alternatives considered:** Tune the coefficient scale after seeing these
rows, change the frozen 1–8 schedule to 0–7, gate corrections by layer/router
margin, promote the less-negative residual variant, add held-out rows, or inspect
task accuracy despite the failed route signal.

**Consequences:** The router is highly sensitive to naïve last-token velocity.
Hidden correction changed centered router logits by 1.291 normalized RMS and
retained only 0.195 pseudo top-8 overlap with the uncorrected route; residual
correction was milder at 0.957 RMS and 0.388 overlap but still harmful. Most
importantly, the known-token first anchor fell from 0.924479/0.949430 route
hit/mass to 0.734782/0.762451 and 0.825602/0.861247. Thus a fixed extrapolation
disturbs the most trustworthy state before it can help later anchors.

All cache/RNG/shadow/information and one-forward call-count audits pass. The
root validates 12 atomic pairs, 41 checksummed artifacts, checksum resume, and
zero failure markers; manifest SHA-256 is
`b01f9ce91a9c636b5ad7db9fd8a343920f5a6b70910fc4047536ee0fbdc1ac21`.
These are teacher-forced current-policy hard-subset route measurements with
measured probe/replay cost and simulated transfer, not task accuracy, free
generation, exact-token identity, or runtime speedup.

**Experiments affected:** Only `pseudo_one_forward_state_correction_v1`.
Earlier route-analysis `NARROW`, focused pseudo `STOP/PIVOT`, GPT hard-only
`NARROW`, trained-model `STOP/PIVOT`, and M11 conclusions are unchanged.

**Migration required:** Preserve this terminal development result. Any new
coefficient schedule, adaptive gate, held-out evaluation, or accuracy stage must
be separately frozen before results. New rows or expanded scope still require
explicit human authorization.

## D-20260802-041 — Stop protected-anchor correction after a small development gain

**Status:** accepted

**Context:** D-20260802-040 found that fixed 1–8 velocity corrections corrupted
the strongest known-token anchor. Before new output, the user authorized and the
project committed a new four-row protocol that sets anchor one's coefficient to
zero. Five calibration-free candidates isolate hidden versus residual state,
undamped versus `(anchor-1)/8` horizon damping, a correction-vector norm cap
equal to the fixed 25% resident fraction, and a selector that reserves anchor-one
top-8. Every candidate still uses one causal eight-token pseudo traversal and
fresh native MoE residuals executed under the previous realized B=32 subset.

The checksum-pinned uncorrected reference scored 0.700267 route hit and 0.714061
selected mass. Protected residual horizon damping scored 0.703389/0.716302, with
+0.003123/+0.002240 gains positive on all four samples and paired 95% intervals
[0.001261, 0.004985]/[0.001702, 0.002779]. Undamped residual correction remained
negative at -0.001607/-0.002789. The 25% cap reduced centered-logit RMS from
0.467 to 0.197 but shrank gain to +0.000763/+0.000417. Reserving only anchor-one
top-8 and filling from later corrected anchors regressed -0.042857/-0.058856.

**Decision:** Keep **STOP/PIVOT**. Anchor-one protection plus damping converts
the previous large regression into a reproducible small positive effect, but
the gains recover only about 16%/11% of the frozen +0.02 route signal. Do not
promote any candidate to held-out route evaluation or task accuracy.

**Alternatives considered:** Accept any wholly positive interval, tune damping
or cap strength on these four rows, retain the anchor-one-only core despite its
regression, add layer/margin gates post hoc, or inspect task accuracy after the
predeclared route gate failed.

**Consequences:** Protecting the known sampled-token anchor is necessary for a
stable velocity experiment but is not the main missing information. Residual
damping helps anchors 2, 3, 6, and 7 while anchors 4, 5, and 8 remain mixed or
negative. The strong anchor-one-only subset ablation shows that the previous
first-four/history breadth is carrying important coverage. Same-cache tests
prove coefficient zero preserves anchor-one router logits; separately executed
BF16 artifacts are not used for bitwise identity claims.

All 20 new atomic pairs, four source rows, 58 artifacts, checksum resume, and
cache/RNG/shadow/information/call-count audits validate with zero failure
markers. Manifest SHA-256 is
`455db962eee8379480a353353e1168e0e26e90dc8de2739fbf2ea4ec2369f84e`.
The measurements are teacher-forced current-policy route/probe evidence with
simulated transfer, not held-out validation, task accuracy, free generation,
exact-token identity, or runtime speedup.

**Experiments affected:** Only `pseudo_one_forward_protected_anchor_v2`.
Earlier route-analysis `NARROW`, pseudo `STOP/PIVOT`, GPT hard-only `NARROW`,
trained-model `STOP/PIVOT`, and M11 conclusions are unchanged.

**Migration required:** Preserve this development result and do not tune on the
four rows. A materially different information source—not another post-hoc
coefficient—needs a separately frozen protocol. New rows, held-out evaluation,
or accuracy still require explicit authorization.

## D-20260802-042 — Stop token-aligned state retrieval at development

**Status:** accepted

**Context:** After protected temporal corrections remained too small, the user
authorized a materially different calibration-free information source. Before
implementation and model output, a new protocol froze the same four Qwen/GSM8K
development rows and five one-forward candidates. Full-expert prefill records
every prompt token's native router input and exact executed MoE output. Each
later hard window appends only the current policy's already-realized states.
Pseudo anchors retrieve the most recent same-token state; optional fallback uses
native input-embedding cosine. Retrieved state is norm-matched into either the
fresh router input or the fresh pseudo MoE residual. No future true token after
the sampled token, vanilla state, answer, accuracy, fitted value, offline prior,
route table, or default vector is accessible.

The checksum-pinned uncorrected reference scored 0.700267 route hit and 0.714061
selected mass. Exact residual addition scored 0.699565/0.712088, deltas
-0.000702/-0.001973; its paired 95% intervals were
[-0.002767, 0.000814]/[-0.003723, -0.000403]. Embedding-nearest residual
addition ranked first by selected mass at 0.699025/0.712238, deltas
-0.001241/-0.001824, with intervals
[-0.002828, 0.000692]/[-0.002754, -0.000351]. Residual replacement and both
router-input variants regressed more. Exact-token retrieval covered 0.953125 of
anchors; fallback covered the remaining 0.046875. Thus retrieval availability
was not the limiting factor.

**Decision:** Record **STOP/PIVOT**. Do not run held-out route evaluation or task
accuracy. Token identity is too coarse to supply the missing future hidden-state
direction, and direct addition can perturb even the trustworthy anchor-one
state. Preserve the uncorrected one-forward recent-sequence mechanism as the
cheap reference and the earlier autoregressive self-greedy route result as the
stronger but more expensive composition evidence.

**Alternatives considered:** Tune similarity weights or layer gates on the four
rows, use a broader offline token-state dictionary, fit a projection between
retrieved and fresh state, promote the tiny route-hit regression because cost is
low, or inspect accuracy despite the failed frozen route gate.

**Consequences:** All candidates retained 45.05%–46.24% simulated transfer
reduction and 0.3354–0.3527 second measured total planning latency, so cost was
not the failure. Even exact residual addition regressed anchor-one hit/mass by
-0.005697/-0.006014. All cache/RNG/shadow/information/call-count audits pass.
Twenty atomic candidate pairs, four source references, 59 artifacts, checksum
resume, and zero failure markers validate. Manifest SHA-256 is
`512c0670002ceee7b201c7f13b948e63766225887a5b778bceae4924e19ed893`.
These are teacher-forced current-policy route measurements with measured probe/
retrieval cost and simulated transfer—not held-out validation, task accuracy,
free generation, exact-token identity, runtime, or speedup.

**Experiments affected:** Only
`pseudo_one_forward_token_aligned_retrieval_v1`. Earlier protected/linear
`STOP/PIVOT`, executed-pseudo route `NARROW`, focused pseudo `STOP/PIVOT`, GPT
hard-only `NARROW`, trained-model `STOP/PIVOT`, and M11 conclusions remain
unchanged.

**Migration required:** Preserve these four rows as development evidence and do
not tune retrieval weights on them. A future calibration-free attempt needs a
new information source beyond raw token identity—such as a separately frozen
analytic attention/context transform—before any new rows. Held-out, accuracy,
expanded samples, learned parameters, or offline calibration still require
explicit authorization.

## D-20260802-043 — Stop midpoint shifted self-conditioning at development

**Status:** accepted

**Context:** The user authorized one more calibration-free, single-forward
attempt after token-aligned retrieval failed. Before implementation or model
output, the project committed a four-row Qwen/GSM8K protocol. Each boundary
performs one native causal H=8 traversal with fresh MoE residuals under full
top-8 access initially and the current policy's preceding realized B=32 subset
thereafter. After zero-based layer 23, native final norm and LM head predict a
shifted content token from anchors 1–7. The native input-embedding difference
between predicted and initial content shifts anchors 2–8 once; anchor one is
bitwise protected. Greedy, expected-top-8, and greedy-with-25%-cap variants were
fixed in advance.

The checksum-pinned uncorrected reference scored 0.700267 route hit and 0.714061
selected mass. Greedy shifting scored 0.700521/0.714245, deltas
+0.000254/+0.000183. Expected-top-8 ranked first at 0.700531/0.714320, deltas
+0.000264/+0.000259, with paired 95% intervals
[-0.000183, 0.000712]/[-0.000265, 0.000714]. The shift changed all seven
predicted top-1 tokens, but expected-top-8 embedding deltas averaged only 0.0470
of midpoint hidden norm. Greedy deltas averaged 0.0542 and peaked at 0.0935, so
the frozen 25% cap never activated and capped/uncapped greedy route tensors were
identical.

**Decision:** Record **STOP/PIVOT** and do not run held-out route evaluation or
task accuracy. A single midpoint token-embedding shift has a measurable but
negligible router effect and misses both frozen +0.02 route-signal requirements
by roughly two orders of magnitude. Preserve the uncorrected one-forward method
as the cheap reference and the earlier fully autoregressive self-greedy rollout
as the stronger, more expensive route-only evidence.

**Alternatives considered:** Tune refresh layer or shift scale on these rows,
apply multiple refreshes, replace norm matching after seeing the results,
promote a positive point estimate whose interval includes zero, or run accuracy
despite the failed frozen route gate.

**Consequences:** All cache/RNG/shadow/information audits pass, including
bitwise preservation of anchor one and exactly one causal traversal, 48 native
attention/router/expert calls, and one seven-query LM-head refresh per boundary.
The retained non-gating BF16 post-cast norm diagnostic uses a tighter
`rtol=atol=1e-3` threshold and passes only 29/96 boundary-policy cases. The
implementation applies norm restoration and its FP32 native test passes; this
diagnostic was not silently promoted into or removed from the frozen gate.
Measured mean probe latency was 0.3027–0.3183 seconds and simulated transfer
reduction was 0.4666–0.4667. Twelve atomic candidate pairs, four source
references, 44 manifest artifacts, checksum resume, and zero failure markers
validate. Manifest SHA-256 is
`f5098b420819830c12ca8cd7613baf6a784314b70477b3ea02a720d4a966aa5c`.
These are teacher-forced current-policy route measurements, not held-out
validation, task accuracy, free generation, exact-token identity, runtime, or
speedup.

**Experiments affected:** Only
`pseudo_one_forward_midlayer_self_conditioning_v1`. Earlier one-forward
`STOP/PIVOT`, executed-pseudo route `NARROW`, focused pseudo `STOP/PIVOT`, GPT
hard-only `NARROW`, trained-model `STOP/PIVOT`, and M11 conclusions remain
unchanged.

**Migration required:** Preserve these development rows and do not tune the
midpoint or shift scale on them. A materially different calibration-free
information transform must be frozen before new output. Held-out, accuracy,
expanded rows, learned parameters, or offline calibration still require
explicit authorization.

## D-20260802-044 — Retain current-context continuation as development route signal

**Status:** accepted

**Context:** After midpoint shifted self-conditioning missed the +0.02 gate, the
user authorized the first proposed calibration-free alternative: replace the
recent-sequence pseudo content with a continuation copied from tokens already
known in the current request. Before implementation or model output, the project
committed a four-row Qwen/GSM8K protocol. At each boundary the visible context
ends at the sampled-next token. Earlier exact-token matches cannot overlap that
query, copied successors must already be known, and equal matches choose the
most recent occurrence. Three fixed variants compared longest-suffix full copy,
sampled-token unigram full copy, and longest-suffix partial copy with recent-
sequence fill. The selector and one native causal H=8 traversal were unchanged;
fresh native MoE residuals came from full top-8 access initially and the current
policy's previous realized B=32 subset later.

The checksum-pinned uncorrected reference scored 0.700267 route hit and
0.714061 selected mass. `sampled_unigram_full_continuation` ranked first at
0.730357/0.747085, gains of +0.030090/+0.033024. Paired 95% intervals were
[0.025065, 0.034587]/[0.026498, 0.038776], and every development row improved.
It found a full seven-token known continuation at 20/32 boundaries, averaged
0.079 ms of lookup and 0.3055 seconds of native probe time per boundary, and
retained 0.5022 simulated transfer reduction. Longest-suffix full and partial
variants both scored 0.726573/0.742827; no partial-only match occurred, so those
two route tensors were identical.

**Decision:** Record **DEVELOPMENT_ROUTE_SIGNAL** and retain
`sampled_unigram_full_continuation` as the leading cheap one-forward hypothesis.
Do not reinterpret this as held-out validation, task accuracy, a full pilot GO,
or evidence of speedup. The frozen protocol authorized no held-out route or
accuracy execution regardless of the development outcome.

**Alternatives considered:** Prefer the more specific longest suffix despite a
lower measured route signal, fit a minimum suffix length or probability
threshold on these four rows, use an offline n-gram table, expand the row set
after seeing the result, or run task accuracy immediately.

**Consequences:** All cache/RNG/shadow/information audits pass, including one
causal traversal, 48 native attention/router/expert calls, 384 attention
queries, zero LM-head calls per boundary, known-context-only copied indices, and
deterministic tie-breaking. Twelve atomic candidate pairs, four checksum-pinned
source references, 45 manifest artifacts, checksum resume, row counts,
provenance, and zero failure markers validate. Manifest SHA-256 is
`1ce50ae324605f7a9ac60e87242c37b9856c9e2db5a5e98879990864c4f854d9`.
The route evidence is teacher-forced on each current hard policy's own state;
lookup and probe costs are measured, transfer is simulated, and free generation,
exact-token identity, NLL/perplexity, closed-loop runtime, and speedup were not
measured.

**Experiments affected:** Only `pseudo_one_forward_context_continuation_v1`.
Earlier one-forward `STOP/PIVOT`, executed-pseudo route `NARROW`, focused
pseudo `STOP/PIVOT`, GPT hard-only `NARROW`, trained-model `STOP/PIVOT`,
and M11 conclusions remain unchanged.

**Migration required:** Preserve these four rows as development evidence and do
not tune matching rules or thresholds on them. Any held-out evaluation needs a
new frozen disjoint manifest and gate; task accuracy, expanded samples, learned
parameters, offline calibration, or downloads still require explicit
authorization.

## D-20260803-045 — Narrow one-forward accuracy evidence with one allowed loss

**Status:** accepted

**Context:** After the current-context continuation development route signal,
the user authorized a small actual GSM8K accuracy test. Before any new accuracy
generation, the project committed an eight-row manifest, three policies,
512-token cap, one-question allowed-drop rule, strong 8/8 preservation signal,
paired bootstrap, execution order, and measured-versus-simulated boundary. All
rows use Qwen3-30B-A3B at `H=8,B=32` with true outside-subset logit masking and
the policy's own closed-loop context. Recent-sequence and sampled-unigram are
deployable one-pseudo-traversal policies. Future-exact content obtains seven
full-expert greedy successors from a copy-on-write current-policy cache before
the same pseudo traversal, so it is explicitly nondeployable and is not a
routing oracle.

Recent-sequence and sampled-unigram each preserved 7/8 frozen vanilla-correct
answers. Their paired comparison was one gain, one loss, and six ties, with
mean accuracy difference zero. Sampled-unigram improved aggregate route hit and
selected mass from 0.730267/0.745076 to 0.757617/0.772934, but the route gain
did not become an accuracy gain. It fixed `test-1311` relative to recent-
sequence and lost `test-1264`. Future-exact content reached still higher
0.783271/0.803592 route coverage but only 6/8 accuracy. Its paired accuracy
difference from frozen vanilla was -0.25 with a small-sample bootstrap interval
of [-0.625, 0.0].

**Decision:** Record **PILOT_NARROW_WITH_ONE_ALLOWED_LOSS**, capped at
`PILOT_NARROW`. Both deployable policies pass the frozen 7/8 allowed-drop rule,
but neither provides strong 8/8 preservation, neither dominates the other, and
`N=8` cannot support a full-dataset GO. Do not select sampled-unigram solely
from its higher aggregate route coverage. Do not describe future-exact content
as deployable, a perfect routing oracle, or a ceiling on answer accuracy.

**Alternatives considered:** Promote sampled-unigram because it has higher
route hit/mass, combine policies after inspecting their two discordant samples,
treat future token content as a perfect oracle, extend the sample set after
observing accuracy, rerun vanilla, change token caps, or infer offloading
speedup from the research runner.

**Consequences:** All 24 sample/policy rows are actual hard closed-loop
generation; zero rows are identity-materialized. Production cache/RNG, shadow
discard, information boundary, native top-k/normalization, hard masking,
atomic-row checksums, resume, row counts, and the 50-artifact manifest validate
with zero failure markers. The two deployable policy losses illustrate why
route coverage is not an accuracy surrogate: each losing trajectory eventually
reported route hit above 0.94 after diverging, while a lower-coverage paired
policy answered correctly. Accuracy, token identity, NLL/perplexity, route
coverage, probe cost, memory, and research-runner runtime are measured;
transfer reduction is simulated and production offloading speedup is not
measured. Artifact-manifest SHA-256 is
`12e9d371485b7375561f8f194fbdb6e3f1ea57fda02cddabfd12a17a408eef67`.

**Experiments affected:** Only `pseudo_one_forward_accuracy_pilot_v1` gains
actual accuracy evidence. The context-continuation development route signal,
earlier one-forward `STOP/PIVOT` results, executed-pseudo route `NARROW`,
focused pseudo `STOP/PIVOT`, GPT hard-only `NARROW`, trained-model
`STOP/PIVOT`, and M11 conclusions remain unchanged.

**Migration required:** Preserve the eight IDs, raw rows, one gain/one loss
pairing, and nondeployable diagnostic as pilot evidence. Any combined selector,
additional rows, different cap, learned/calibrated value, model/dataset change,
or production offloading benchmark requires a separately frozen protocol and
appropriate authorization.

## D-20260803-046 — Retain matched-eight hard routing oracle as a diagnostic ceiling

**Status:** accepted

**Context:** After the completed one-forward accuracy pilot, the user requested
the true hard routing-oracle accuracy on the same eight frozen Qwen/GSM8K rows.
Before any oracle generation, a separately versioned post-hoc amendment pinned
the existing row manifest, `H=8,B=32`, greedy v17 prompts/parser/512-token cap,
policy semantics, paired reporting rule, and nondeployable claim boundary. At
each boundary the oracle performs natural full-expert lookahead from the hard
policy's own current context, sums future selected routing weights per layer,
selects deterministic top-32 subsets, then actually masks outside-subset router
logits before native top-8 selection and normalization.

Hard oracle preserved all eight answers, matching frozen same-row vanilla at
8/8 with paired gains/losses/ties 0/0/8 and a conditional eight-row paired
bootstrap difference interval of `[0,0]`. It did not preserve the natural token
trajectory: weighted exact-token agreement was 0.524826, six of eight samples
diverged, and hard generation produced 2,120 tokens versus 2,101 frozen vanilla
tokens. Weighted route hit and selected routing mass were 0.946483 and
0.966458. Measured research-runner time was 7,706.46 seconds; simulated
transfer reduction was 0.766503.

**Decision:** Record **PILOT_NARROW_DIAGNOSTIC_ORACLE_CEILING** for only the
matched-eight Qwen/GSM8K `H=8,B=32` scope. The perfect-routing-information
ceiling preserved answer accuracy on this small sample, but it is not
deployable, is not a one-extra-forward pseudo-embedding method, and does not
establish full-dataset preservation or offloading speedup. It does not revise
the parent `PILOT_NARROW_WITH_ONE_ALLOWED_LOSS` pseudo-policy decision.

**Alternatives considered:** Treat future-exact pseudo content as the routing
oracle, materialize vanilla identity rows, rerun vanilla, replace the frozen
IDs or cap, use a vanilla future trajectory instead of the current hard policy
context, infer runtime speedup from simulated transfer, or promote eight
matched successes to full-dataset GO.

**Consequences:** All eight rows are actual hard closed-loop generation and
zero are identity-materialized. Cache/RNG lookahead restoration, current-policy
context, hard-mask/native-router semantics, raw-row checksums, resume, row
count, provenance, and the 19-artifact manifest validate with zero failure
markers. Artifact-manifest SHA-256 is
`336d97c416aba6a80ad205228667bbbde570378a872c5cf7bd707544ce98cc28`.
Accuracy, token identity, route coverage, NLL/perplexity, peak allocation, and
research-runner runtime are measured; transfer and its reduction are simulated.

**Experiments affected:** Only `pseudo_one_forward_hard_oracle_v1` gains this
matched-eight diagnostic. The parent one-forward accuracy pilot, context-
continuation route signal, earlier one-forward `STOP/PIVOT`, executed-pseudo
route `NARROW`, focused pseudo `STOP/PIVOT`, GPT hard-only `NARROW`, trained-
model `STOP/PIVOT`, and M11 conclusions remain unchanged.

**Migration required:** Preserve the frozen IDs, raw rows, config/sample
fingerprints, paired 8/8 result, token divergences, and nondeployable scope.
Additional rows, a different cap or operating point, learned/calibrated values,
model/dataset changes, or a production offloading benchmark require a new
predeclared protocol and explicit authorization.

## D-20260803-047 — Freeze a paired two-row real Qwen expert-offload speed pilot

**Status:** accepted

**Context:** The user authorized implementation of actual expert-weight
offloading and requested a small GSM8K speed comparison between traditional
per-token exact routing and `natural_top8_intersection_zero_missing`. Prior
Qwen transfer and stall values were simulations; the existing real runtime was
TinyMoE-only and cannot serve as Qwen evidence.

**Decision:** Before any new model output, freeze
`qwen_real_offload_speed_pilot_v1` on the first two already-predeclared
wave-one rows, `test-44` and `test-632`. Give both policies exactly 32 CUDA
expert slots per layer. The traditional baseline executes unmasked natural
top-8 with deterministic LRU loads. The candidate uses H=8/B=32 hard
commitment and the frozen mass-preserving zero-missing pseudo residual. Require
pinned bfloat16 CPU sources, actual CPU-to-CUDA copies, CUDA-event transfer and
stall timing, atomic resume, and exact token agreement with the corresponding
full-resident reference rows. Alternate AB/BA policy order across the two
samples. Exclude model extraction and slot allocation from inference
throughput but report them separately.

**Alternatives considered:** Compare against an 8-slot traditional baseline,
reuse simulated bytes as speed, include model load in decode throughput, choose
short rows after observing runtime, add new GSM8K IDs, or claim overlap without
implementing it.

**Consequences:** This is a two-row engineering pilot on one A100 and its
observed PCIe link. It may measure implementation-specific speed and actual
transfer reduction, but it cannot establish full-dataset accuracy, general
production speedup, NVMe behavior, multi-GPU/NVLink behavior, or
transfer/compute-overlap gains.

**Experiments affected:** Only the new real-offload pilot. All prior Qwen
accuracy, route, hard-oracle, and simulated-transfer conclusions remain
unchanged.

**Migration required:** Any additional row, token-cap change, resident budget,
cache policy, overlap schedule, quantization, model revision, dataset, NVMe
path, or multi-GPU configuration requires a separately frozen amendment.
## D-20260803-048 — Stop the real-offload pseudo candidate after identity failure and no speedup

**Status:** accepted

**Context:** D-20260803-047 froze a two-row single-A100 comparison of
traditional exact native-top-8 offloading and
`natural_top8_intersection_zero_missing_h8_b32`. The implementation moved all
57,982,058,496 routed-expert bytes to pinned CPU memory, removed full CUDA
expert parameters, allocated 14,495,514,624 bytes of B=32 CUDA slots, performed
actual H2D copies, and measured transfer/stall with CUDA events. An initial
candidate run exposed sensitivity to the expert GEMM batching order. The engine
was corrected to reproduce Qwen's top-k-position-major native batching,
including zero-weight rows for resident experts, and proportional bitwise CUDA
tests passed. Candidate trajectories still diverged, indicating that
full-resident versus offloaded BF16 state differences were amplified by router
and subset decisions rather than a missing-weight load.

**Decision:** Record
**STOP_PIVOT_CANDIDATE_IDENTITY_GATE_FAILURE** for only the two-row
Qwen/GSM8K H=8,B=32 engineering scope. Traditional offload preserved frozen
vanilla tokens and answers 2/2. The candidate answered 2/2 and had zero
production expert misses, but preserved its frozen mask-only token trajectory
0/2, with first divergences at tokens 4 and 2. Aggregate post-prefill decode
throughput was 0.509055 forwards/s for traditional and 0.503535 for the
candidate: 0.989157x, or -1.084%. Actual H2D bytes fell 28.770% and CUDA-event
transfer time fell 31.386%, but no runtime speedup was measured.

**Alternatives considered:** Treat two correct answers as an accuracy GO,
discard divergent rows, silently loosen exact identity, rerun the recovered
263-token row, compare raw generated tokens/s despite different trajectory
lengths, infer speedup from H2D reduction, change B/H or sample IDs after
seeing results, or add overlap/fusion without a new protocol.

**Consequences:** The completed measurement and the failed identity gate are
both retained. Four actual rows, two smoke rows, physical memory audits,
checksummed resume, and the 18-entry manifest validate. Two failed markers and
two diagnostic rows remain provenance; one completed row is explicitly
recovered from its checksum-valid diagnostic. All reported output/runtime/H2D
values are measured; legacy candidate transfer estimates remain simulated.
This Python runner is deliberately unoverlapped and instrumented, so the
negative speed result is implementation-specific and does not prove an
optimized offloader cannot improve.

**Experiments affected:** Only
`qwen_real_offload_speed_pilot_v1`. Prior mask-only accuracy, hard-oracle,
route-level, simulated-transfer, trained-model, and TinyMoE runtime conclusions
remain unchanged.

**Migration required:** Preserve the two IDs, AB/BA order, token caps, raw rows,
diagnostics, failed markers, fingerprints, and STOP/PIVOT decision. Any
additional samples, tolerance-based identity criterion, fused expert kernel,
overlap schedule, prefill policy, B/H change, quantization, NVMe/NVLink path,
model/dataset revision, or production claim requires a separately frozen
amendment.

## D-20260804-049 — Narrow penultimate-token joint planning after one wave-one loss

**Status:** accepted; awaiting human confirmation before implementation

**Context:** The user proposed using the token consumed by an H=8 window's last
production forward to plan the next subset, so a future runtime could combine
that real bridge and eight pseudo anchors instead of waiting for the newly
sampled final token and issuing a separate forward. Before implementing actual
joint execution or offloading, the frozen
`pseudo_penultimate_joint_qwen_gsm8k_wave1_v1` protocol required a no-offload
logical-equivalence test on the same eight wave-one GSM8K rows. Its disposable
shadow bridge preserves the intended state dependency while cache, RNG, and
bridge parity audits distinguish it from a runtime benchmark.

**Decision:** Record
**PILOT_NARROW_ONE_ALLOWED_LOSS_AWAITING_USER_CONFIRMATION**. The candidate
scored 7/8 versus 8/8 for both frozen vanilla and the checksum-pinned
online-post-sample baseline, with paired gains/losses/ties 0/1/7. It meets the
predeclared minimum 7/8 gate but not the strong 8/8 signal. Its weighted route
hit/selected mass were 0.685858/0.706211, below the online baseline's
0.713199/0.734273. All 288 non-bootstrap bridge checks and all cache/RNG/shadow
audits passed. Stop at the required checkpoint and ask the user whether to
proceed; do not infer engineering authorization from the minimum gate pass.

**Alternatives considered:** Start the actual offloader before checking
accuracy, use the just-sampled last token even though it is unavailable at the
proposed planning time, treat the duplicate bridge latency as joint runtime,
rerun the checksum-pinned baseline, change IDs or caps after observing the one
loss, or expand beyond wave one.

**Consequences:** Eight new rows are actual resident-weight hard closed-loop
generation and none is identity materialized. Accuracy, token/route metrics,
logical-simulator cost, and parity are measured; transfer is simulated. The
candidate's 0.540137 simulator tokens/s includes duplicate bridge work and says
nothing about the proposed joint implementation's speed. Actual offloading,
prefetch overlap, fused/joint latency, and production speedup remain
unimplemented and unmeasured. N=8 supports only the scoped pilot decision.

**Experiments affected:** Only
`pseudo_penultimate_joint_qwen_gsm8k_wave1_v1`. Prior online-post-sample
accuracy, hard-oracle, route-analysis, and real-offload results remain unchanged.

**Migration required:** Preserve the exact eight IDs, original token caps,
config/sample fingerprints, raw rows, 7/8 result, failed `test-252` row, parity
audits, checksums, and measured/simulated partition. Actual joint/fused/offload
implementation requires the pending user confirmation and a separately frozen
engineering amendment; any additional rows, different timing, B/H, model,
dataset, or speed claim requires new scope.
