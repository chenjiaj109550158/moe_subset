# Project Status

## Current milestone

M11 — Reproducible primary results and paper artifact generation, complete at deterministic tiny-model scope.

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

## Exact check results

Recorded 2026-07-27 (UTC) on Python 3.14.6 and PyTorch 2.13.0+cu130.

- `ruff format --check .`: PASS; 109 files already formatted.
- `ruff check .`: PASS.
- `mypy`: PASS; no issues in 64 source files.
- `pytest -ra`: PASS; 91 passed, 3 skipped in 10.60s. Skips are two opt-in external-model tests and the unavailable-CPU inverse case because CUDA is available.
- `python -m build`: PASS; sdist and wheel built.
- Full M11 `reproduce`: PASS; seven completed run envelopes, zero failures/incomplete runs, Tables A–F, figure index, plot, failure reports, manifests, and `DONE`.

## Failing tests

None.

## Limitations

- M11 primary evidence remains tiny-model evidence; the pinned Hugging Face adapter is not in the default suite.
- The aggregate-bandwidth wave model conservatively gives unequal concurrent transfers a common completion time.
- Route traces are natural tiny-model greedy trajectories; commitment quality is represented by route coverage, not closed-loop task quality.
- The lossless predictor uses previous-route reuse and demand fallback, not a trained M6/M7 predictor artifact.
- Capacity is global expert bytes; dense weights, KV cache, allocator reserve, and workspaces are assumed already reserved outside this budget.
- CPU execution and global-budget learned allocation are not included in the acceptance grid, though global byte knapsack is implemented.
- Mixtral trace ingestion is not wired into this CLI run.
- Decode recomputes the full prefix because the tiny adapter has no production KV cache.
- One transfer stream and per-layer equal-size slots are supported; variable-size packing, batching, CPU execution fallback, and multi-GPU are out of scope.
- Lossless subset misses load on demand. Hard commitment is implemented but the definitive quality run uses lossless fallback.
- CUDA allocator peaks exclude driver/context allocations; CPU accounting covers expert handles, not process RSS.
- No Triton or custom CUDA was added because correctness precedes profiling.

## Next exact tasks

No further milestone is authorized; do not claim production-model generality.

## Decisions needing human review

None.
