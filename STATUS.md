# Project Status

## Current milestone

Qwen3-30B-A3B/GSM8K pseudo-embedding focused pilot v1 and its separately
versioned calibration-free mechanism analysis — complete with scoped
**STOP/PIVOT** decisions. Earlier M11, trained-model, and subset-oracle artifacts
remain complete and unchanged.

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

Focused-pilot final checks recorded 2026-08-01 UTC on Python 3.14.6, PyTorch
2.13.0+cu130, Transformers 5.14.1, and 2 × A100-SXM4-80GB.

- `ruff format --check .`: PASS; 183 files already formatted.
- `ruff check .`: PASS.
- `mypy src/pseudoroute`: PASS; no issues in 101 source files.
- Focused held-out/prompt/development/content/Qwen/subset tests: PASS; 26 passed.
- `python -m pytest -ra`: PASS; 171 passed, 3 expected skips in 15.82s.
- Five calibration-free artifact validators: PASS; manifest counts
  11/20/18/22/31, final held-out 8/8 atomic rows, zero actual accuracy rows, and
  terminal `complete/report_v1` `STOP/PIVOT`.

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

The focused pseudo-embedding v1 stage and calibration-free held-out follow-up
are terminal at **STOP/PIVOT**. Preserve their negative artifacts and do not run
actual accuracy. No predictor training, expanded dataset scope, default-vector
recalibration, or production runtime work is authorized. The next defensible
hypothesis is online current-request MoE-residual/state reuse plus autoregressive
shadow content and predeclared horizon damping. Any execution needs a new,
scoped protocol; new sample rows require explicit human authorization.

## Decisions needing human review

None.
