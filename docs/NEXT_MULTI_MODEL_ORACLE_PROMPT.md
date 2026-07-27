# Next-stage prompt: multi-model trained-MoE oracle gate

Copy the prompt below into a new Codex session rooted at this repository.

---

Continue the PseudoRoute-MoE trained-model oracle stage in this repository.

First read `STATUS.md`, `README.md`, `docs/trained_model_plan.md`,
`docs/model_support.md`, `docs/reproducibility.md`,
`docs/spec/01_RESEARCH_SPEC.md`, `docs/spec/02_RELATED_WORK_AND_NOVELTY.md`,
`docs/spec/04_ALGORITHMS.md`, `docs/spec/05_EXPERIMENT_PLAN.md`, and
`docs/decisions.md`.

Inspect Git status and preserve unrelated user changes. Do not rewrite the M11
baseline or commits `f3c91b1`, `7f92685`, and `61bb766`.

## Current state

The deterministic TinyMoE M11 suite is complete. These trained checkpoints are
already present in the ignored project Hugging Face cache:

1. `allenai/OLMoE-1B-7B-0125`
   - revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`
   - native adapter and literal route tests exist
   - checksum-valid natural-forward smoke passed
2. `Qwen/Qwen1.5-MoE-A2.7B`
   - revision `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`
   - 24 layers, 60 routed experts, top-4, and a shared expert
   - eight weight shards are downloaded
3. `openai/gpt-oss-20b`
   - revision `6cee5e81ee83917806bbde320786a8fb61efebee`
   - three Transformers shards are checksum-valid
   - native MXFP4; keep results separate from floating-point models
   - `metal/model.bin` is cached; do not delete it without authorization
4. `mistralai/Mixtral-8x7B-v0.1`
   - revision `fc7ac94680e38d7348cfa806e51218e6273104b0`
   - 19 safetensors shards are checksum-valid
   - alternate consolidated weights were not downloaded
   - the existing adapter needs trained-checkpoint and multi-device validation
5. `deepseek-ai/DeepSeek-V2-Lite-Chat`
   - revision `85864749cd611b4353ce1decdb286193298f64c7`
   - four shards are checksum-valid
   - reviewed remote code may use `trust_remote_code=True` only at this revision
   - Transformers 5.14.1 needs a minimal in-memory compatibility shim for the
     removed `is_torch_fx_available`
   - a 10-token smoke passed: 26 MoE layers, top-6, finite logits, and
     32.65 GB peak CUDA allocation
   - preserve 64 routed experts, two shared experts, scaling, normalization,
     and native router semantics

Hardware is 2 × A100 80GB with approximately 235 GiB CPU RAM. About 218 GB
disk remained free after downloads.

## Objective

Complete the trained-model oracle feasibility gate. Do not train a predictor or
implement production runtime optimization until oracle and closed-loop results
justify a GO or narrowly scoped NARROW decision.

Use OLMoE, Qwen1.5-MoE, Mixtral, and DeepSeek-V2-Lite first. Treat gpt-oss as a
separate quantized-model tier; it must not delay the primary floating-point
decision if its MXFP4 trace path is incompatible with the common metrics.

## Required order

### 1. Reverify the baseline

Run formatting, lint, mypy, focused tests, and the feasible full test suite.
Reverify the no-download TinyMoE primary suite. Record exact software and
hardware versions. Keep simulated, measured-runtime, and model-quality outputs
separate.

### 2. Finish adapter validation

Extend the common adapter contract. Do not bypass it with standalone
model-specific analysis scripts.

For every trained architecture:

- validate model structure and MoE-layer indices;
- reproduce native router logits and pre-top-k scores;
- reproduce exact selected IDs and weights;
- preserve native scoring, normalization, routing scaling, and shared experts;
- use layer-scoped routed-expert IDs;
- distinguish routed residency from always-active shared experts;
- verify traced/untraced logits and natural-policy generation parity;
- record peak CPU/GPU memory;
- add literal or native-reference regression tests.

For DeepSeek, isolate and document the minimal compatibility shim, do not edit
cached upstream code, and retain the initial direct-import failure as a
validation artifact.

For Mixtral, use two-A100 sharding or documented CPU/GPU placement without
silently quantizing.

For gpt-oss, inspect native routing and expert-weight representation before
adapting and label all MXFP4-specific limitations.

### 3. Build the trained trace suite

Use real tokenized text from at least:

- WikiText-2 validation at the pinned revision;
- GSM8K test at the pinned revision.

Persist model/tokenizer fingerprints, dataset revision and row IDs, prompt
rendering, seeds, information regime, router logits, native pre-top-k scores,
selected IDs/weights, token positions, layer IDs, routed/shared metadata,
checksums, resolved configs, stdout/stderr, and completion/failure envelopes.

Start with smoke samples, then use multiple independent documents/prompts,
context positions, and both domains. Final claims cannot use synthetic tokens
or randomly initialized routing.

### 4. Run the common oracle sweep

Predeclare thresholds before examining final aggregates.

At minimum use:

- horizons `1, 2, 4, 8, 16`;
- absolute budgets and multiples of native top-k;
- binary-count, selected-routing-mass, and full-router-mass oracles where valid;
- previous-route reuse, static frequency, and an LRU/LFU-style baseline.

Report route hit rate, selected/full mass coverage, expert-union size, subset
churn, estimated H2D bytes/token, mean, median, P05, P95, worst windows, and
bootstrap confidence intervals. Stratify by model, layer, horizon, budget,
domain/task, context-position bucket, and router margin.

Persist concrete worst cases with source sample, boundary, layer, natural
route, selected subset, and metric values. Do not macro-average away
model-specific failures.

### 5. Validate closed-loop quality

On a smaller meaningful subset compare:

1. natural routing;
2. lossless residency with demand fallback;
3. hard commitment to the natural-trajectory oracle subset;
4. previous-route/static/LRU-style non-oracle baseline.

Use deterministic decoding first. Measure exact token agreement, first
divergence, NLL/perplexity, GSM8K answer metric where applicable,
out-of-subset mass, fallback frequency, transfer-byte estimate, and exposed
stall estimate. Label timing simulated unless actually measured.

The oracle uses the natural future trajectory and is optimistic. Never describe
lossless fallback as a quality improvement.

### 6. Apply the decision gate

Thresholds must require:

- useful coverage materially below all-routed-expert residency;
- improvement over simple residency/cache baselines;
- acceptable tail and worst-case behavior;
- acceptable hard-commitment quality or sufficiently low lossless fallback;
- credible transfer/stall reduction after planning cost.

Classify every model and the overall stage as GO, NARROW, or STOP/PIVOT. Do not
issue a cross-model GO when feasibility exists only for specific
architectures, layers, horizons, tasks, or fallback semantics. Preserve
negative results.

### 7. Reproducibility and handoff

Add versioned trained-suite configs and a resumable runner. Generate plots only
from saved tables. Update `README.md`, `STATUS.md`,
`docs/trained_model_plan.md`, `docs/model_support.md`,
`docs/reproducibility.md`, `docs/decisions.md`, and `CHANGELOG.md`.

Run all feasible quality checks, create logically scoped commits, and leave the
working tree clean.

## Completion condition

Finish only when either:

1. checksum-valid trained traces, oracle sweeps, and closed-loop samples
   produce documented per-model and overall GO/NARROW/STOP decisions; or
2. a concrete adapter/runtime incompatibility is retained as a documented
   model failure while feasible primary models still reach a decision.

Do not claim generality beyond the tested checkpoints, datasets, hardware,
precision, budgets, and decoding regime.

