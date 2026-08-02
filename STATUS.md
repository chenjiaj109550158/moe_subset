# Project Status

## Current milestone

Qwen3-30B-A3B/GSM8K pseudo-embedding focused pilot v1 and its separately
versioned calibration-free mechanism and one-forward state-correction analyses
— including protected-anchor v2, token-aligned state retrieval v1, and mid-layer
shifted self-conditioning v1 — complete with scoped **STOP/PIVOT** decisions.
Earlier M11, trained-model, and subset-oracle artifacts remain complete and
unchanged.

## Qwen/GSM8K pseudo-embedding focused pilot

`pseudo_embedding_qwen_gsm8k_v1` froze config fingerprint
`a81f36b5ec4a4222ca7a459f9f9c1536d88bef5ba143c151c9e70498157d8cbc`, exact
sample IDs, route/cost gates, and the 16-row accuracy rule before any pseudo
result. It used pinned Qwen revision `0d7cf239...` in BF16, frozen GSM8K v17
trajectories, `H=8,B=32`, native top-8 routing, and 25% resident experts.

- Native two-row mechanism smoke passed 8-token windows, subset change,
  evaluator, production cache identity/data/version, RNG, shadow discard,
  attention/RoPE/router, and information-boundary audits.
- Four development traces produced 24,576 sample/boundary/layer/method metric
  rows. Oracle route hit/mass was 0.958714/0.976395; previous-route was
  0.609385/0.629428; static frequency was 0.339705/0.346426.
- The primary pseudo variant scored 0.533015/0.534416. The best mandatory
  ablation, zero expert contribution, scored 0.566499/0.575402. Every mandatory
  variant missed both required +0.05 improvements over previous-route; the
  expected-top-8 auxiliary also regressed at 0.533448/0.534745.
- Development therefore selected no variant and emitted **STOP/PIVOT**. The
  frozen stop rule marked held-out route evaluation not run and forbade all
  actual closed-loop accuracy. Measured task-accuracy rows and identity-
  materialized rows are both zero.
- Probe time/memory/router/attention costs are measured. Transfer is simulated;
  stall is not estimated and no speedup is claimed. The reused default-vector
  artifact contains 444 unobserved layer/expert pairs saved as zero and is not a
  complete expert prior.
- Six atomic route samples and their tensor checksums validate; four historical
  failure markers are retained. Fresh cross-process BF16 replay met strict
  authoritative router tolerance on 0/4 development rows, so authoritative v17
  route tensors—not replay drift—were the natural-route scoring target.

Artifacts: `artifacts/pseudo_embedding_qwen_gsm8k_v1/`. No learned predictor was
trained and no model or dataset was downloaded.

## Calibration-free pseudo-embedding mechanism analysis

The follow-up preserved the terminal v1 decision and used no learned/fitted
value, offline expert prior, route-transition table, default-vector value,
future true token in a candidate, answer, correctness, or accuracy. The full
plan and synthesis are in
`docs/pseudo_embedding_calibration_free_synthesis_v1.md`.

- Tensor and native-Qwen interventions show that the router is strongly
  token/hidden-direction sensitive. Repeated pseudo content changes too little
  across positions; recent token IDs do not help; a zero-residual causal shadow
  makes routing worse. Exact future contents have diagnostic headroom, but are
  forbidden for deployment.
- Same-request prompt routes solved the first-window static fallback. Four-row
  development selected the equal sampled-pseudo/history utility at
  0.664737/0.688248, with +0.055351/+0.058820 over previous route.
- A separately committed eight-row held-out route run completed 8/8 samples and
  393,216 slots. The candidate scored 0.658353 route hit, 0.680584 selected
  mass, and 0.439423 simulated transfer reduction versus previous-route
  0.617671/0.637396.
- Gains were positive on all eight samples, but only +0.040682/+0.043188. The
  paired 95% intervals stayed below +0.05, oracle-gap recovery was only
  0.116775/0.125752, and both absolute references failed. The strong-candidate
  gate therefore emitted **STOP/PIVOT** and accuracy remained forbidden.
- All 128 probe calls, eight prompt captures, cache/RNG/shadow/information
  audits, atomic row pairs, checksums, resume audit, and 31-artifact manifest
  validate. Probe/runtime is measured; transfer is simulated; actual hard
  generation, task accuracy, exact-token identity, and speedup were not
  measured.
- Post-hoc horizon damping found a three-anchor/history hypothesis at
  0.664205/0.687282, still below the frozen gate. Because it was observed after
  held-out results, it is not validation and cannot be promoted without a new
  disjoint protocol.

Held-out artifacts:
`artifacts/pseudo_embedding_calibration_free_prompt_route_held_out_v1/`.
Artifact-manifest SHA-256:
`3f1b509d7d012e8266ee6a28e1863efb44f00a946b29646a484c859a5ecd08db`.

## Previous-window MoE-residual pseudo analysis

`pseudo_embedding_qwen_gsm8k_residual_window_v1` froze config SHA-256
`361ba85b995b1c2d8573816eadf73861ea0f9f89bed5158221ae6e26f87caf7b`,
sample-manifest SHA-256
`cd6957e34f72f69290ebd9cb147be54ccee3a3dcfe6ef5358f6d34dc850d1255`,
all formulas, IDs, gates, and caps before model execution.

- Full-expert prefill supplied the final eight prompt-token MoE mixture outputs;
  later windows reused each hard policy's own eight actually executed mixture
  outputs. No default-vector values, learned/fitted parameters, future true
  tokens, answers, correctness, or accuracy entered selectable candidates.
- Two-row smoke selected position-aligned residuals at 0.668783 route hit and
  0.674895 selected mass, versus zero contribution at 0.568848/0.570145.
  Sampled-next-token repeated content with independent anchors then won the
  deployable content rule. Exact-future content was diagnostic-only.
- Seven analytic pseudo/history constructions ran on the four frozen
  development rows. The best, a first-four-anchor top-8 union filled by previous
  route history, scored 0.639706/0.649761 versus previous-route
  0.623634/0.634308 and oracle 0.949966/0.970436.
- The gains were only +0.016072/+0.015453, with paired four-sample 95% intervals
  [0.011566, 0.020695] and [0.011292, 0.020388]. Oracle-gap recovery was only
  4.9%/4.6%. Both predeclared +0.05 progress checks failed; the decision is
  **STOP/PIVOT**.
- All 64 atomic sample-policy rows, 171 manifest artifacts, cache/RNG/shadow/
  native-semantics audits, hard-mask execution, previous-route control parity,
  row counts, checksums, and resume audit validate. There are no failed markers.
- The new held-out route set and the 16-row actual closed-loop accuracy pilot
  are explicitly not run. Task accuracy, exact-token identity, free-generation
  NLL/perplexity/runtime, and runtime speedup were not measured. Transfer is
  simulated; probe/replay latency and temporary memory are measured.

Artifacts: `artifacts/pseudo_embedding_qwen_gsm8k_residual_window_v1/`.
Artifact-manifest SHA-256:
`720e3e0ae68efeddd770226f7d953969c8313997455ac8520a39788b4dd71771`.

## One-forward state-correction analysis

`pseudo_one_forward_state_correction_v1` froze config SHA-256
`7f3c519996c0a190930ef9de4edc138200e3a659287bc171d966e09219803c44`,
the four development IDs, formulas, costs, and stop rule before new model
results. Every variant performs one native causal eight-token pseudo traversal
per boundary. Boundary zero uses full native top-8 access; later boundaries
execute the current policy's previous realized B=32 subset to produce a fresh
pseudo MoE residual.

- The uncorrected recent-sequence baseline reproduced 0.700267 route hit and
  0.714061 selected mass at 0.3082 seconds per boundary probe.
- Layer-wise router-input velocity scored 0.648031/0.661326, deltas
  -0.052236/-0.052736. Fresh residual plus historical residual velocity scored
  0.676310/0.691583, deltas -0.023956/-0.022478.
- Both four-sample paired 95% intervals were wholly negative. Hidden velocity
  changed the centered router logits by 1.291 normalized RMS and retained only
  0.195 pseudo top-8 overlap with the uncorrected probe; residual velocity was
  milder at 0.957 RMS and 0.388 overlap.
- The known-token first anchor explains much of the failure: the uncorrected
  0.924479/0.949430 fell to 0.734782/0.762451 and 0.825602/0.861247. Neither
  correction recovered the loss over later anchors.
- All cache/RNG/shadow/information and one-forward call-count audits pass. All
  12 atomic JSON+safetensors pairs checksum-resume, with zero failed markers.
  Held-out route and task accuracy were not run under the frozen stop rule.

The development-only decision is **STOP/PIVOT**. Route metrics are measured on
teacher-forced current-policy hard-subset state; probe/replay cost is measured,
transfer reduction is simulated, and no speedup is claimed. Artifacts:
`artifacts/pseudo_one_forward_state_correction_v1/`. Artifact-manifest SHA-256:
`b01f9ce91a9c636b5ad7db9fd8a343920f5a6b70910fc4047536ee0fbdc1ac21`.

## Protected-anchor one-forward correction v2

`pseudo_one_forward_protected_anchor_v2` froze config SHA-256
`5bac51be6be2ab90aed563eb9208ce23afa772c9b109e9fa1bf66ee319bb81ee`,
sample SHA-256
`d37ff792acb6cb0d2fe0158040f7e3db8d3555783492c46ad4349362ada302ea`,
five analytic candidates, four reused development IDs, and gates before new
model output. All candidates retain one native causal eight-token pseudo
traversal and fresh MoE execution under the previous realized B=32 subset.

- Setting anchor one's coefficient to zero plus residual horizon damping scored
  0.703389 route hit and 0.716302 selected mass versus the checksum-pinned
  uncorrected 0.700267/0.714061. Gains were +0.003123/+0.002240 and positive on
  all four rows; paired intervals were [0.001261, 0.004985] and
  [0.001702, 0.002779].
- That is only about 16%/11% of the required +0.02 signal. Hidden damping was
  essentially neutral at +0.000529/-0.000160; residual protection without
  damping remained negative at -0.001607/-0.002789.
- A 25% correction-vector norm cap reduced centered-logit RMS from 0.467 to
  0.197 but also reduced gain to +0.000763/+0.000417. It was too restrictive at
  this fixed structural value.
- Reserving only anchor-one top-8 and filling the other 24 experts from corrected
  anchors two through eight discarded useful first-four/history breadth and
  regressed by -0.042857/-0.058856 on all four rows.
- Same-cache native tests prove coefficient zero leaves anchor-one router logits
  exact. Separately executed BF16 source/candidate artifacts are not required to
  be bitwise identical, and their hard policies realize different later contexts.
- All 20 new atomic pairs, four source references, 58 manifest artifacts,
  checksum resume, and cache/RNG/shadow/information/call-count audits validate;
  there are zero failure markers.

The frozen +0.02 route signal failed, so the development-only decision is
**STOP/PIVOT** and neither held-out route nor accuracy was run. Route/probe costs
are measured on teacher-forced current-policy hard-subset state; transfer is
simulated and no speedup is claimed. Artifact-manifest SHA-256:
`455db962eee8379480a353353e1168e0e26e90dc8de2739fbf2ea4ec2369f84e`.

## Token-aligned one-forward state retrieval v1

`pseudo_one_forward_token_aligned_retrieval_v1` froze config SHA-256
`19a368fe6a5e819cc0e5b3a6c3528ccf9baf94a66ba4733a278e9c0d1e5cadff`,
sample SHA-256
`dc18107f178100f41df1ffccdf23f07ddd3f611044f4b7f7ac5da1edbad36c69`,
the same four development IDs, five variants, costs, and gates before
implementation or model output.

- Full-expert prefill captures each prompt token's native router input and exact
  executed MoE output. Later banks append only the current hard policy's already-
  realized tokens. Every retrieved index is strictly pre-boundary; no future
  token, answer, accuracy, learned value, offline prior, or default vector enters
  a candidate.
- Recent-sequence anchors 2–8 always found an exact token-aligned state. Across
  all anchors the exact fraction was 0.953125; embedding fallback covered the
  remaining 0.046875, which consisted of previously unseen anchor-one tokens.
- The checksum-pinned uncorrected reference scored 0.700267 route hit and
  0.714061 selected mass. Exact residual addition scored 0.699565/0.712088;
  embedding-nearest fallback plus residual addition scored 0.699025/0.712238.
  Router-input variants and residual replacement regressed further.
- Even the mild exact residual addition changed anchor one by
  -0.005697/-0.006014. No variant approached the frozen +0.02 route signal;
  selected-mass paired intervals for the two additive residual variants were
  wholly negative.
- Mean retrieval latency was 0.0003–0.0016 seconds and mean total planning
  latency 0.3354–0.3527 seconds; simulated transfer reduction remained
  0.4505–0.4624. Cost and transfer were acceptable, but route quality failed.
- All 20 atomic candidate pairs, four source references, 59 manifest artifacts,
  cache/RNG/shadow/information/call-count audits, checksums, and resume validation
  pass with zero failure markers.

The development-only decision is **STOP/PIVOT**. Held-out route and task
accuracy were not run. Metrics are teacher-forced on each policy's own hard-
subset state, probe/retrieval cost is measured, transfer is simulated, and no
free-generation, exact-token, closed-loop runtime, or speedup claim is made.
Artifact-manifest SHA-256:
`512c0670002ceee7b201c7f13b948e63766225887a5b778bceae4924e19ed893`.

## Mid-layer shifted self-conditioning v1

`pseudo_one_forward_midlayer_self_conditioning_v1` froze config SHA-256
`75f4c028871632139a3f68f290726cb4834931a3a89cd2cc16883e050ebbb67b`,
sample SHA-256
`8314356459cbf1bf36bfafffecc0a0a2bb7fd750c22b85ef50cb921ff4b8f535`,
the same four development IDs, three variants, call counts, and gates before
implementation or model output.

- Every boundary performs one causal H=8 traversal. After zero-based layer 23,
  native final norm and LM head predict tokens from anchors 1–7; their input-
  embedding deltas shift anchors 2–8 once before layers 24–47. Anchor one is
  protected and remains bitwise unchanged at the refresh.
- Greedy shifting scored 0.700521 route hit and 0.714245 selected mass versus
  the checksum-pinned uncorrected 0.700267/0.714061, gains of only
  +0.000254/+0.000183. Expected-top-8 embeddings ranked first at
  0.700531/0.714320, gains of +0.000264/+0.000259; both paired 95% intervals
  include zero for selected mass.
- The prediction changed all seven shifted top-1 tokens, but the expected-top-8
  embedding delta averaged only 0.0470 of midpoint hidden norm. Greedy averaged
  0.0542 and never exceeded 0.0935, so the fixed 25% cap was inactive and the
  capped/uncapped greedy route tensors were identical.
- Post-refresh layers gained only about +0.00053 route hit and +0.00052 selected
  mass over the uncorrected late-layer stratum. This is real but far below the
  frozen +0.02 overall signal, while simulated transfer reduction remained
  about 0.4667 and measured probe latency was 0.3027–0.3183 seconds/boundary.
- All 12 atomic JSON+safetensors candidate pairs, four source references, 44
  manifest artifacts, checksum resume, and cache/RNG/shadow/information/call-
  count audits validate with zero failure markers.
- A separate non-gating BF16 post-cast norm diagnostic using
  `torch.allclose(rtol=atol=1e-3)` passed only 29/96 boundary-policy checks. The
  implementation applies L2 norm restoration and its FP32 native test passes;
  the stricter-than-BF16 0.1% diagnostic is retained rather than hidden.

The development-only decision is **STOP/PIVOT**. Route metrics are teacher-
forced on each current hard policy's own state; probe and LM-head costs are
measured, transfer is simulated, and held-out route, task accuracy, free
generation, exact-token identity, NLL/perplexity, closed-loop runtime, and
speedup were not measured. Artifact-manifest SHA-256:
`f5098b420819830c12ca8cd7613baf6a784314b70477b3ea02a720d4a966aa5c`.

## Completed GPT-OSS/GSM8K hard scope

`benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2` completed all 1,319 actual hard
rows. Vanilla and true hard closed loop both scored 1,242/1,319 (94.162244%),
with exact-token agreement 1.0 and no paired gains/losses. The scoped decision is
`NARROW`; `H=1,B=4` equals native top-k and does not establish multi-token
constrained-subset success or runtime speedup. Superseded rows and failure
markers remain provenance.

## M11 file-level checklist

- [x] `src/pseudoroute/reporting/manifest.py` — versioned envelopes, checksums, completion and negative-result records
- [x] `src/pseudoroute/reporting/aggregate.py` — strict aggregation, Tables A–F, plot/index, and failure reports
- [x] `src/pseudoroute/reporting/reproduce.py` — ordered, failure-retaining, checksum-resumable primary runner
- [x] `src/pseudoroute/cli.py` — `aggregate-results` and `reproduce`
- [x] `configs/paper/primary.yaml` — seven-task primary suite
- [x] `scripts/regenerate_*.sh`, `scripts/reproduce_primary.sh` — individual and end-to-end runners
- [x] `tests/unit/test_reporting.py`, `tests/integration/test_reproduce_cli.py` — all required M11 regressions
- [x] `docs/reproducibility.md`, README, model matrix, decisions, status, and changelog
- [x] `README.md`, `docs/model_support.md`, and `docs/decisions.md`

## Completed tasks

- Expert handles own the sole CPU parameter copies and record pinning, byte size, slot, state, and transfer-completion event.
- Equal-size GPU slots are allocated once per layer; the decode loop performs no expert destination allocation.
- Synchronous transfers provide the correctness reference before asynchronous execution is admitted.
- The asynchronous path uses one CUDA transfer stream, non-blocking copies, transfer-completion events, compute-stream waits, and compute-completion events before slot replacement.
- Lossless fallback preserves natural routing; hard commitment masks routing to a validated subset plan.
- Real H2D bytes, CUDA-event transfer/stall time, synchronized host TPOT, allocated/reserved peaks, resident expert bytes, and pinned CPU bytes are exported.

## M11 acceptance run

Recorded 2026-07-27 (UTC). No model or dataset was downloaded.

- Artifacts: `artifacts/reproduction/m11_primary_definitive_v2/`.
- Seven checksum-valid completed runs: oracle, factorial, predictor training, probe, simulator, static residency, and real CUDA runtime; zero incomplete or failed runs.
- Generated linked Tables A–F, combined CSV/Markdown, aggregate SVG, figure index, concrete worst-case CSV/Markdown, JSON negative-result report, aggregation manifest, and reproduction manifest.
- Retained two negative results: tiny factorial Gate B did not pass; at least one simulated commitment had route coverage below 1.
- Concrete worst cases cover oracle selected-mass coverage, factorial router-logit effect, probe subset regret, simulator coverage, static out-of-subset mass, and runtime TPOT.
- Resume was exercised after interruption: six checksum-valid completed tasks were retained and the suite continued without overwrite.
- No model or dataset was downloaded.

## Trained-model oracle gate v2

Completed 2026-07-27 UTC without changing the M11 baseline. Final artifacts are
`artifacts/trained_gate/suite_v2_final/`; the versioned runner config is
`configs/trained/oracle_gate_v2.yaml`. The preliminary v1/12-token run exposed a
truncated GSM8K closed-loop prompt. Final v2 changed only the closed-loop sample
to complete row 500 and extended deterministic decoding to 64 tokens; thresholds
and open-loop grids were fixed before the final v2 aggregates.

- Four floating-point checkpoints completed checksum-valid adapter inspection,
  eight real-text trace shards, common oracle sweep, context/margin-stratified
  bootstrap summaries, worst cases, and ten closed-loop rows each.
- OLMoE: 16×64 routed, top-8, no shared expert; 49.95 GiB RSS and 25.83 GiB
  peak CUDA allocated. Final decision: **STOP/PIVOT**.
- Qwen1.5-MoE: 24×60 routed, top-4 plus one shared expert; 48.80 GiB RSS and
  26.75 GiB CUDA. Final decision: **STOP/PIVOT**.
- Mixtral: 32×8 routed, top-2, full-precision layers 0–15/16–31 on two A100s;
  88.99 GiB RSS and 45.25/45.25 GiB CUDA. Final decision: **STOP/PIVOT**.
- DeepSeek-V2-Lite: MoE layers 1–26, 64 routed plus two shared, top-6; 18.72
  GiB RSS and 30.41 GiB CUDA. The direct import and DynamicCache failures are
  retained; only reversible in-memory shims are used. Final decision:
  **STOP/PIVOT**.
- gpt-oss: 24×32 top-4 biased-logit routing and native MXFP4 expert storage were
  inspected, but fused execution exposed no per-layer router hook records. It is
  retained as a separate **STOP/PIVOT** failure; `metal/model.bin` remains cached.
- Final tables contain 857,728 raw oracle windows, 232,180 global aggregate rows,
  and 1,705,301 context-position × router-margin aggregate rows, all with saved
  source tables and checksum envelopes.
- Natural and lossless closed-loop outputs are exactly equal for every model.
  Mean lossless fallback is 47.97%, 80.22%, 28.82%, and 70.70%; every value
  exceeds the 10% gate. Hard-commitment mean token agreement is 20.31%, 19.53%,
  26.56%, and 6.25%, with unacceptable perplexity changes.
- Transfer bytes and exposed stall are simulated at 25 GiB/s plus 10 µs/load.
  Model logits, NLL/perplexity, token agreement, GSM8K extraction, and memory are
  actual checkpoint executions; no end-to-end runtime speedup is claimed.

Overall decision: **STOP/PIVOT**. Do not train a route predictor or implement a
production offload runtime from these results. A new stage requires a separately
predeclared pivot, such as quality-aware soft fallback or matched-budget
closed-loop evaluation, and must not relabel this negative result.

## Exact check results

Focused-pilot final checks recorded 2026-08-02 UTC on Python 3.14.6, PyTorch
2.13.0+cu130, Transformers 5.14.1, and 2 × A100-SXM4-80GB.

- `ruff format --check .`: PASS; 210 files already formatted.
- `ruff check .`: PASS.
- `mypy src/pseudoroute`: PASS; no issues in 109 source files.
- Focused residual-window/Qwen/shadow tests: PASS; 28 passed.
- Particle-dispatch focused tests: PASS; 12 passed.
- Token-aligned retrieval protocol/native tests: PASS; 7 passed.
- `python -m pytest -ra`: PASS; 215 passed, 3 expected skips in 17.32s.
- Five calibration-free artifact validators: PASS; manifest counts
  11/20/18/22/31, final held-out 8/8 atomic rows, zero actual accuracy rows, and
  terminal `complete/report_v1` `STOP/PIVOT`.
- Residual-window validator: PASS; 64/64 atomic sample-policy rows, 171 manifest
  artifacts, zero failed markers, and terminal `complete/report_v1`
  `STOP/PIVOT`.
- Executed-pseudo composition validator: PASS; 104/104 atomic sample-policy
  rows, 242 manifest artifacts, four preserved failure markers, and terminal
  `complete/report_v1` route-analysis `NARROW`.
- One-forward state-correction validator: PASS; 12/12 atomic sample-policy
  rows, 41 manifest artifacts, zero failed markers, checksum resume, and
  terminal `complete/report_v1` `STOP/PIVOT`.
- Protected-anchor v2 validator: PASS; 20/20 new candidate pairs plus four
  checksum-pinned source references, 58 manifest artifacts, zero failed markers,
  checksum resume, and terminal `complete/report_v1` `STOP/PIVOT`.
- Token-aligned retrieval v1 validator: PASS; 20/20 new candidate pairs plus
  four checksum-pinned source references, 59 manifest artifacts, zero failed
  markers, checksum resume, and terminal `complete/report_v1` `STOP/PIVOT`.

Retained 2026-07-27 acceptance records below were not rerun in this focused
session:

- `PSEUDOROUTE_RUN_EXTERNAL=1 pytest -ra tests/integration/test_hf_mixtral.py`: PASS; 2 passed in 5.76s from the pinned cache.
- `python -m build`: PASS; sdist and wheel built.
- Offline post-change M11 `reproduce`: PASS at
  `artifacts/reproduction/trained_gate_post/`; seven completed runs, zero failures
  or incomplete runs, two retained negative results, all paper tables/plots and
  terminal `DONE`.
- Final trained v2 envelopes: PASS; root 114 artifacts, four completed model
  envelopes with 26 artifacts each, one five-artifact expected gpt-oss failure
  envelope, and zero independent SHA-256/size mismatches.

## Failing tests

None.

## Limitations

- M11 primary evidence remains tiny-model evidence. The trained gate is a separate, external-cache suite and does not broaden M11 runtime claims.
- The aggregate-bandwidth wave model conservatively gives unequal concurrent transfers a common completion time.
- M11 route traces remain tiny-model trajectories. The trained v2 gate adds actual 64-token closed-loop quality, but only for two prompts per model.
- The lossless predictor uses previous-route reuse and demand fallback, not a trained M6/M7 predictor artifact.
- Capacity is global expert bytes; dense weights, KV cache, allocator reserve, and workspaces are assumed already reserved outside this budget.
- CPU execution and global-budget learned allocation are not included in the acceptance grid, though global byte knapsack is implemented.
- Decode recomputes the full prefix because the tiny adapter has no production KV cache.
- One transfer stream and per-layer equal-size slots are supported; variable-size packing, batching, CPU execution fallback, and multi-GPU are out of scope.
- Lossless subset misses load on demand. Hard commitment is implemented but the definitive quality run uses lossless fallback.
- CUDA allocator peaks exclude driver/context allocations; CPU accounting covers expert handles, not process RSS.
- No Triton or custom CUDA was added because correctness precedes profiling.

## Next exact tasks

The original focused pseudo-embedding v1 stage, calibration-free prompt-route
follow-up, previous-window residual-bank experiment, and one-forward state
experiments, including protected-anchor v2 and token-aligned retrieval v1,
and mid-layer shifted self-conditioning v1 remain terminal at **STOP/PIVOT**.
Token identity and a single midpoint embedding shift are not useful enough at
this point. The executed-pseudo composition analysis is terminal at a route-only
**NARROW**: preserve its positive held-out route evidence, but do not run
actual accuracy from that analysis protocol because held-out oracle-gap recovery
and a paired allowed-drop rule were not frozen there. A generation stage
requires a new committed execution amendment using fixed existing IDs and gates
before any accuracy is observed. No predictor training, expanded dataset scope,
default-vector recalibration, or production runtime work is authorized; new
sample rows require explicit human authorization.

## Decisions needing human review

None.
