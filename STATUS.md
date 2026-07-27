# Project Status

## Current milestone

Post-M11 trained-model oracle gate v2 — complete with an overall **STOP/PIVOT** decision. M11 remains complete and unchanged at deterministic tiny-model scope.

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

Final post-change checks recorded 2026-07-27 UTC on Python 3.14.6, PyTorch
2.13.0+cu130, Transformers 5.14.1, and 2 × A100-SXM4-80GB.

- `ruff format --check .`: PASS; 127 files already formatted.
- `ruff check .`: PASS.
- `mypy`: PASS; no issues in 76 source files.
- Focused trained adapter/runner/trace/oracle tests: PASS; 16 passed in 6.76s.
- `pytest -ra`: PASS; 99 passed, 3 expected skips in 12.01s.
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

The trained oracle stage is terminal at **STOP/PIVOT**. No predictor training or production runtime work is authorized. Preserve the negative artifacts and require a new, scoped, predeclared research question before any follow-up.

## Decisions needing human review

None.
