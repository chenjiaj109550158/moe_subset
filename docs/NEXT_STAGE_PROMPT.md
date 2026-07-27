# Next-stage prompt: trained-MoE oracle feasibility gate

Copy the prompt below into a new Codex session rooted at this repository.

---

You are continuing the PseudoRoute-MoE project in this repository. Read
`STATUS.md`, `README.md`, `docs/model_support.md`, `docs/download_cache_plan.md`,
`docs/reproducibility.md`, `docs/spec/01_RESEARCH_SPEC.md`,
`docs/spec/04_ALGORITHMS.md`, `docs/spec/05_EXPERIMENT_PLAN.md`, and
`docs/decisions.md` before changing code.

The deterministic TinyMoE M11 suite is complete. The next objective is **not**
to optimize TinyMoE further. Establish whether window-level expert residency is
actually feasible on a trained, language-capable MoE model. Treat this as an
oracle feasibility gate: do not invest in a learned predictor or production
runtime optimization unless the trained-model oracle results justify it.

## Required workflow

### 1. Preserve and verify the baseline

- Inspect Git status and do not overwrite unrelated user changes.
- Run the existing fast quality checks and record the exact environment.
- Verify the existing TinyMoE primary suite still passes without downloading
  models or datasets.
- Keep simulated results, measured runtime results, and model-quality results
  explicitly separated.

### 2. Perform a no-download preflight

Before downloading anything:

- Inspect available CPU RAM, GPU model/VRAM, CUDA/PyTorch versions, free disk
  space, and existing model caches.
- Inspect the current Hugging Face Mixtral adapter and external integration
  tests.
- Identify the smallest suitable **trained, language-capable, sparse MoE**
  checkpoint that exercises real routing and fits the available environment.
  Prefer an already supported architecture and a pinned immutable revision.
- Estimate model, tokenizer, dataset, trace, and temporary-artifact storage.
  Preserve adequate free space for generated traces and results.
- Write the proposed model, revision, license, expected resource use, and
  fallback choice to `docs/trained_model_plan.md`.
- If no meaningful trained MoE fits, stop after producing the plan and report
  the exact blocker. Do not substitute a random tiny model and call it
  trained-model evidence.

Do not download until the estimate and cache plan have been recorded. Reuse an
existing cache when checksums/revisions can be verified. Never commit model
weights, datasets, caches, or generated run artifacts.

### 3. Complete trained-model trace support

- Extend the existing adapter rather than bypassing the common adapter
  contract.
- Reproduce the model's native router semantics exactly, including score
  function, bias, top-k normalization, shared experts, and layer-scoped expert
  IDs.
- Add structure validation and literal or reference-backed route regression
  tests.
- Support natural teacher-forced trace collection for router logits,
  pre-top-k scores, selected expert IDs, selected weights, token positions, and
  layer IDs.
- Use real text tokenization and document the dataset/task sampling procedure.
- Pin and record model revision, tokenizer fingerprint, dataset revision,
  software environment, seeds, and information regime.
- A small smoke sample may be used first, but final claims must not be based on
  a synthetic vocabulary or randomly initialized routing.

### 4. Run the oracle feasibility sweep

Run natural routing traces and window-level oracle selection over a grid that
is large enough to expose trade-offs:

- multiple prompts/documents and at least two text domains or tasks;
- multiple context positions;
- window lengths including short and longer windows;
- expert budgets expressed both as absolute counts and multiples of native
  top-k;
- binary-count and routing-mass oracles;
- per-layer and global summaries;
- bootstrap confidence intervals over independent samples.

At minimum report:

- route hit rate;
- selected routing-mass coverage and full-probability-mass coverage;
- expert-union size;
- subset churn and estimated H2D bytes per generated token;
- mean, median, P05/P95, and worst-window results;
- results by layer, horizon, budget, task/domain, and router margin.

Do not report only averages. Persist concrete worst cases with source sample,
boundary, layer, natural route, chosen subset, and metric values.

### 5. Validate closed-loop quality

For a smaller but meaningful evaluation subset, compare:

1. natural routing;
2. lossless subset residency with demand fallback;
3. hard commitment to the oracle subset;
4. a simple non-oracle baseline such as previous-route reuse, LRU/LFU, or
   static frequency.

Use deterministic decoding first. Measure:

- exact token agreement and divergence position;
- negative log-likelihood or perplexity where applicable;
- task metric when the selected dataset supplies one;
- out-of-subset routing mass and fallback frequency;
- transfer estimate and exposed-stall estimate, clearly labeled simulated
  unless measured by the runtime.

The oracle is chosen from the natural trajectory and is therefore optimistic;
state this explicitly. Do not describe lossless fallback as a quality
improvement, and do not infer production latency from simulated timing.

### 6. Apply an explicit decision gate

Define thresholds in the experiment configuration before examining final
results. At minimum the gate must require:

- useful coverage at a memory budget materially below all-expert residency;
- improvement over simple residency/cache baselines;
- acceptable tail and worst-case behavior, not only mean behavior;
- acceptable closed-loop quality for hard commitment, or a sufficiently low
  fallback rate for the lossless strategy;
- a credible transfer/stall reduction after accounting for planning cost.

Classify the outcome:

- **GO:** oracle trade-off is strong enough to justify trained utility
  prediction and runtime integration;
- **NARROW:** feasibility exists only for specific layers, horizons, tasks, or
  lossless fallback; restrict the next design accordingly;
- **STOP/PIVOT:** the oracle cannot beat simple baselines under a meaningful
  budget, so do not train a predictor. Document whether shorter windows,
  static residency, cache prediction, or another target is the better pivot.

Negative results are first-class outputs and must remain in manifests and
reports.

### 7. Reproducibility and handoff

- Add versioned configs and a resumable command/script for the trained-model
  oracle suite.
- Retain resolved configs, stdout/stderr, checksums, completion/failure
  envelopes, source fingerprints, raw tabular results, and plots generated only
  from saved tables.
- Add unit and integration tests proportional to new adapter and trace paths.
- Run formatting, lint, type checking, focused tests, and the feasible full
  test suite.
- Update `README.md`, `STATUS.md`, `docs/model_support.md`,
  `docs/reproducibility.md`, `docs/decisions.md`, and `CHANGELOG.md` with exact
  commands, results, limitations, and the GO/NARROW/STOP decision.
- Keep the working tree clean with logically scoped commits. Do not rewrite the
  existing M11 baseline commit.

## Completion criteria

The stage is complete only when one of these is true:

1. a checksum-valid trained-model oracle suite and closed-loop quality sample
   have produced a documented GO/NARROW/STOP decision; or
2. the no-download preflight proves that the available environment cannot run a
   meaningful trained MoE, and a concrete resource/model plan documents what is
   required next.

Do not claim generality beyond the tested checkpoint, tasks, hardware, budgets,
and decoding regime.

---

The intended ordering is:

`preflight → adapter/trace validation → oracle sweep → closed-loop quality → decision gate`

Only a GO or narrowly scoped NARROW result authorizes learned-predictor and
runtime-optimization work.
