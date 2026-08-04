# Reproducibility

The primary deterministic suite is regenerated with:

```bash
pseudoroute reproduce --suite primary --config-root configs/paper \
  --output-dir artifacts/reproduction/primary
```

Use `--dry-run` to list the ordered commands without creating artifacts. The real-runtime task requires CUDA; all other primary tasks use the portable tiny configuration. No external download is required.

Every reproduced run contains `resolved_config.json`, `metrics.json`, captured stdout/stderr, terminal `DONE` or `FAILED.json`, and `run_manifest.json`. Manifest schema 1 records the command, result kind, information regime, config digest, stable run ID, completion state, retained negative findings, and SHA-256/size for every artifact.

`pseudoroute aggregate-results --input-root RUNS --output-dir PAPER` validates all checksums, rejects incompatible schemas and duplicate run IDs, excludes and reports incomplete runs, and never discards negative findings. Paper tables link each value to its source run. Plots are generated only from saved tabular data.

Individual runners are available as `scripts/regenerate_oracle.sh`, `regenerate_factorial.sh`, `regenerate_probe.sh`, `regenerate_simulator.sh`, and `regenerate_runtime.sh`.

## Trained oracle gate v2

Install the trained dependencies and run only from the pinned local cache:

```bash
python -m pip install -e '.[dev,hf]'
CUDA_VISIBLE_DEVICES=0,1 \
HF_HOME=.cache/pseudoroute/huggingface \
HF_DATASETS_CACHE=.cache/pseudoroute/datasets \
pseudoroute trained-suite \
  --config configs/trained/oracle_gate_v2.yaml \
  --output-dir artifacts/trained_gate/suite_v2_final
```

`--dry-run` validates and lists the models/datasets without loading checkpoints.
The runner is config-aware and resumable: a `DONE` model is reused only when its
resolved suite/model config matches; completed trace shards are checksum-validated
and source/quality metadata are retained. Floating-point model failure stops the
suite. The separate MXFP4 tier retains `FAILED.json` and allows the primary gate
to finish. Every model and the suite root has a completion/failure envelope with
SHA-256 and size for every artifact.

Final environment: Linux 6.8.0-100-generic x86-64; Python 3.14.6; PyTorch
2.13.0+cu130; CUDA runtime 13.0; driver 580.126.09; Transformers 5.14.1;
Tokenizers 0.22.2; Safetensors 0.8.0; Datasets 4.8.5; Accelerate 1.14.0;
Kernels 0.15.2/Kernels-data 0.16.0; NumPy 2.3.5; two NVIDIA
A100-SXM4-80GB GPUs; 235 GiB host RAM and no swap. The `hf` extra declares
Accelerate, Kernels and psutil because validated two-GPU placement, native MXFP4
loading, and per-model RSS sampling require them.

For each floating-point model, `traces/{wikitext,gsm8k}/manifest.json` is schema
2 and includes the pinned model/dataset revision, model/tokenizer fingerprint,
seed, prompt/source provenance, non-contiguous layer IDs, routed/shared metadata,
native pre-top-k scores, and shard checksums. `checkpoint_files.json` records Hub
content-addressed weight checksums. `inspection.json` records the direct native
router call before trace validation.

Saved result tables are:

- `oracle_windows.csv`: source/boundary/layer/method-level metrics and concrete routes/subsets;
- `oracle_aggregates.csv`: global mean/median/P05/P95/worst and 1,000-bootstrap CIs;
- `oracle_stratified_aggregates.csv`: the same metrics by context-position and router-margin buckets;
- `oracle_worst_cases.json`: 200 concrete worst windows;
- `closed_loop.csv` and `closed_loop_worst_cases.json`: actual token/NLL/answer outcomes and simulated transfer/stall fields;
- `operating_points.csv`, `decision.json`, and `oracle_gate.svg`: gate inputs, terminal decisions, and a plot generated only from the saved table.

The final root envelope contains 114 artifacts and was independently rehashed
with zero mismatches. The four model envelopes contain 26 artifacts each; the
gpt-oss failure envelope contains five. The direct DeepSeek 5.14 import failure
and DynamicCache compatibility failure remain under
`artifacts/trained_gate/adapter_validation/deepseek/`. The shim edits only
in-memory module/class attributes inside restoring contexts and never edits the
Hub module cache. `gpt-oss-20b/metal/model.bin` remains a cache symlink to the
13,750,886,400-byte retained blob.

Timing fields named `simulated_*` use 25 GiB/s plus 10 µs/load and are not
measured end-to-end runtime. Adapter elapsed time, CPU RSS, CUDA allocation,
model logits/routes, NLL/perplexity, token agreement, fallback, and GSM answer
extraction come from actual checkpoint execution. Lossless fallback equality is
a correctness result and must never be described as a quality improvement.

## Benchmark subset oracle: GPT-OSS/GSM8K hard scope

The immutable scientific config remains
`configs/benchmark/benchmark_subset_oracle_v1.yaml`. The separately versioned
execution amendment is
`configs/benchmark/benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2.yaml`, with
scope fingerprint
`a5a908ad0124d2041b589a681e7102d3ea5d3f7df90752075799bb9875871d7c`.
It schedules only actual GPT-OSS GSM8K hard commitment at the preselected
`(H=1,B=4)` point and requires all 1,319 frozen v17 rows.

Run or resume offline from the existing model and dataset cache:

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/complete_subset_oracle_v1.py
```

Each successful policy/sample row is written atomically and is reused only when
its complete state and base-config fingerprint match. `.FAILED.json` markers are
retained but do not suppress regeneration of a missing success row. Per-GPU
queues preserve the four frozen logical shards while allowing each GPU to start
its next shard independently; at most one worker occupies a physical GPU.
Unscheduled HumanEval/MBPP+/AIME/StrategyQA and previous-route artifacts remain
under the artifact root as provenance, but the scoped summary, paired CI,
decision, and audit include only GSM8K. A passing scoped gate is reported as
`NARROW`, not as an all-task `GO`.

## Qwen/GSM8K pseudo-embedding focused pilot v1

The immutable config and resolved sample manifest are
`configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml` and
`configs/benchmark/pseudo_embedding_qwen_gsm8k_v1_samples.json`. Their config
fingerprint and manifest SHA-256 are respectively
`a81f36b5ec4a4222ca7a459f9f9c1536d88bef5ba143c151c9e70498157d8cbc` and
`21da318e97315e3f0c19bc5213d24e5781456d980b3048677f3b748bccde9c93`.
The artifact root is `artifacts/pseudo_embedding_qwen_gsm8k_v1`.

Run or resume only from the pinned local caches:

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m pseudoroute.benchmark.pseudo_embedding_runner run --gpu 1

python -m pseudoroute.benchmark.pseudo_embedding_runner validate
```

The runner validates Git/environment/model/default-vector/source-trace
provenance, allows at most one worker on a physical GPU, and commits each route
sample as a checksummed JSON plus safetensors pair. A successful pair resumes
without overwrite; `.FAILED.json` markers remain provenance. Aggregate artifacts
include compressed raw rows, global metrics, sample-paired bootstrap CIs,
layer/context/router-margin strata, concrete worst cases, measured probe costs,
cache/RNG/native-semantics audits, decision, provenance, resume audit, and a
checksummed artifact manifest.

This completed run stopped after the four-row development gate. Therefore a
resume/validate invocation does not load Qwen or start a GPU worker: held-out
route evaluation and the fixed 16-row × three-policy actual pilot remain
unauthorized by the frozen protocol. The six valid route samples are two
mechanism rows and four development rows; actual closed-loop rows are zero.

Development scoring uses the pre-existing checksum-verified v17 Qwen router
tensors for the natural-route target. Fresh teacher-forced BF16 replay provides
the deployable production context for the pseudo probe but did not reproduce the
strict authoritative router tensors across processes on 4/4 rows; those parity
measurements are retained and do not replace the scoring target. Route replay and
expert transfer are open-loop and simulated respectively. Probe latency and
temporary CUDA memory are measured. Task accuracy, exact-token identity, NLL,
perplexity, route divergence under hard generation, and generation runtime are
not measured because actual generation was forbidden. Identity-materialized rows
are zero. The 444 unobserved default-vector pairs remain zero and the artifact is
not a complete expert prior.

## Calibration-free pseudo-embedding follow-up

The follow-up protocols were committed before their corresponding model runs.
They reuse only pinned local caches and fixed v17 rows. The final held-out route
artifact is validated with:

```bash
python -m pseudoroute.benchmark.pseudo_embedding_prompt_route_held_out validate
```

Its immutable config is
`configs/analysis/pseudo_embedding_calibration_free_prompt_route_held_out_v1.yaml`
with SHA-256
`1f6f5ab8372aa6f63796cfda49601745bf2736490ee21c8807fd9657b9a996ef`.
It contains eight checksummed JSON+safetensors pairs, 31 manifest artifacts,
10,000 paired sample-bootstrap draws, cache/RNG/information audits, measured
probe and replay cost, and a `STOP/PIVOT` decision. The authoritative manifest
SHA-256 is
`3f1b509d7d012e8266ee6a28e1863efb44f00a946b29646a484c859a5ecd08db`.

The runner uses same-request native prompt routes only at boundary zero and the
preceding realized route window thereafter. Its candidate uses a synthetic zero
probe interface but no default-vector value or offline route statistic. Static
frequency reads `count` only as a reference; `mean` is never accessed. Route
scoring is teacher-forced open-loop on saved v17 trajectories, transfer is
simulated, and probe/replay cost is measured. Task accuracy, actual hard
closed-loop generation, exact-token identity, NLL/perplexity, and speedup are not
measured. Do not run an accuracy stage after the failed gate.

## Previous-window residual pseudo analysis

The immutable protocol and sample manifest are
`configs/analysis/pseudo_embedding_qwen_gsm8k_residual_window_v1.yaml` and
`configs/analysis/pseudo_embedding_qwen_gsm8k_residual_window_v1_samples.json`,
with SHA-256 values
`361ba85b995b1c2d8573816eadf73861ea0f9f89bed5158221ae6e26f87caf7b` and
`cd6957e34f72f69290ebd9cb147be54ccee3a3dcfe6ef5358f6d34dc850d1255`.
They were committed before model output. The result is finalized and validated
offline with:

```bash
python -m pseudoroute.benchmark.pseudo_embedding_residual_window_report finalize
python -m pseudoroute.benchmark.pseudo_embedding_residual_window_report validate
```

`finalize` is idempotent over completed per-sample rows: it verifies all 64
JSON+safetensors pairs, rebuilds only missing stage aggregates, writes
development strata/worst cases/paired bootstrap, records held-out and closed-loop
as not run after the failed gate, and creates a 171-entry checksum manifest. The
authoritative artifact-manifest SHA-256 is
`720e3e0ae68efeddd770226f7d953969c8313997455ac8520a39788b4dd71771`.

The route runner itself is resumable at one sample/policy pair and uses each
policy's own teacher-forced hard-subset hidden/cache state. It captures native
pre-mask routes and the exact MoE mixture output actually executed during the
preceding window. This is not vanilla open-loop route replay, but saved v17
tokens are still teacher-forced, so it is also not actual closed-loop generation
or task accuracy. Probe/replay latency and CUDA temporary memory are measured;
transfer and stall are simulated. No model/dataset download, default-vector
value, offline calibration statistic, learned parameter, or accuracy-based
selection was used.

## Executed-pseudo embedding composition analysis

The immutable config and sample manifest are
`configs/analysis/pseudo_executed_embedding_composition_v1.yaml` and
`configs/analysis/pseudo_executed_embedding_composition_v1_samples.json`, with
SHA-256 values
`35d9ef891c4bccf8603cc0e4b98d8cbb51022a82f11ed72782293e374938e8f9` and
`d554e10eed706e9bc35bdfbb91c58347c7e7a23fe23526460b06fc4e76bdaac1`.
They were committed before any composition-model result. Run or resume the
predeclared waves offline, with at most one worker per physical GPU:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_executed_embedding_composition run-simple --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_executed_embedding_composition run-advanced --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_executed_embedding_composition run-particles --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_executed_embedding_composition run-held-out --gpu 0 --shard-index 0 --shard-count 2
python -m pseudoroute.benchmark.pseudo_executed_embedding_composition validate
```

Use GPU 1 with shard index 1 for the complementary shard. Successful
JSON+safetensors pairs are checksum-resumed without overwrite. The final root
contains 104 validated rows: 64 development, eight particle follow-up, and 32
held-out route rows. Its 242-entry manifest has SHA-256
`e00e88fa8c20c90682f49b52415792681d688c02b74c887cb32be348d5efcac7`.
Four `.FAILED.json` markers are intentionally retained: two current-token
dispatch failures and two particle-attention configuration failures; later
checksum-valid rows do not erase their provenance.

Every deployable pseudo policy executes native attention/RoPE/router semantics
on disposable shadow state. Boundary zero permits full native top-8 execution;
later boundaries execute only the policy's previous realized B=32 layer subset,
so the MoE residual is freshly produced by the pseudo forward rather than copied
from the previous window. Production cache identities/data/version counters and
RNG are audited unchanged. No default-vector values, learned/fitted parameters,
future true tokens, answers, correctness, or accuracy select a deployable
variant. Exact-future rows are diagnostics only.

Route hit and selected mass are teacher-forced measurements on each policy's own
hard-subset state. Probe/replay latency and temporary memory are measured;
transfer reduction is simulated. The terminal `NARROW` applies only to this
route analysis. Held-out oracle-gap recovery, task accuracy, free generation,
exact-token agreement, NLL/perplexity, closed-loop runtime, and runtime speedup
were not measured, so this artifact cannot by itself authorize or support those
claims.

## One-forward state-correction analysis

The immutable config and sample manifest are
`configs/analysis/pseudo_one_forward_state_correction_v1.yaml` and
`configs/analysis/pseudo_one_forward_state_correction_v1_samples.json`, with
SHA-256 values
`7f3c519996c0a190930ef9de4edc138200e3a659287bc171d966e09219803c44` and
`350cc9ddcc003daa6973d247434bc600ecfc108ad7b9773f5e0cb997936ec80c`.
They freeze four existing development IDs and authorize neither new traces nor
held-out/accuracy execution. Run the two complementary offline shards and then
aggregate, finalize, or validate with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_state_correction run --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_state_correction run --gpu 1 --shard-index 1 --shard-count 2
python -m pseudoroute.benchmark.pseudo_one_forward_state_correction aggregate
python -m pseudoroute.benchmark.pseudo_one_forward_state_correction finalize
python -m pseudoroute.benchmark.pseudo_one_forward_state_correction validate
```

Each row is an atomic JSON+safetensors pair and a valid pair resumes without
loading Qwen. The final root validates 12 pairs and 41 manifest artifacts, with
zero failure markers. The authoritative artifact-manifest SHA-256 is
`b01f9ce91a9c636b5ad7db9fd8a343920f5a6b70910fc4047536ee0fbdc1ac21`.

All three variants use recent-sequence causal content and exactly one native
eight-token pseudo traversal per boundary. They execute fresh native MoE
residuals under full top-8 access at boundary zero and the previous realized
B=32 subset thereafter. The two calibration-free corrections use only the last
two current-policy router inputs or MoE outputs and fixed coefficients 1–8; no
future true token after the sampled boundary token, answer, accuracy, fitted
coefficient, route table, expert prior, or default-vector value is accessed.

The terminal development decision is `STOP/PIVOT`: both corrections regress the
uncorrected baseline with wholly negative paired intervals. Route metrics are
teacher-forced on the current policy's hard-subset state, probe/replay latency
and memory are measured, and transfer is simulated. Held-out route, task
accuracy, free generation, exact-token identity, NLL/perplexity, closed-loop
runtime, and speedup are not measured.

## Protected-anchor one-forward correction v2

The immutable config and sample manifest are
`configs/analysis/pseudo_one_forward_protected_anchor_v2.yaml` and
`configs/analysis/pseudo_one_forward_protected_anchor_v2_samples.json`, with
SHA-256 values
`5bac51be6be2ab90aed563eb9208ce23afa772c9b109e9fa1bf66ee319bb81ee` and
`d37ff792acb6cb0d2fe0158040f7e3db8d3555783492c46ad4349362ada302ea`.
They were committed before implementation and new model output. Run the two
offline shards and then aggregate/finalize/validate with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_protected_anchor run --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_protected_anchor run --gpu 1 --shard-index 1 --shard-count 2
python -m pseudoroute.benchmark.pseudo_one_forward_protected_anchor aggregate
python -m pseudoroute.benchmark.pseudo_one_forward_protected_anchor finalize
python -m pseudoroute.benchmark.pseudo_one_forward_protected_anchor validate
```

The runner checksum-reuses the four uncorrected source rows and never reloads
Qwen when all ten rows in a shard validate. The final root contains 20 new
atomic JSON+safetensors pairs, four source references, 58 checksummed manifest
artifacts, and zero failure markers. The authoritative manifest SHA-256 is
`455db962eee8379480a353353e1168e0e26e90dc8de2739fbf2ea4ec2369f84e`.

All candidates use coefficient zero at anchor one. The fixed schedules are
`(anchor-1)/8` or the undamped `anchor-1`; the optional correction-vector norm
cap is the pinned 25% resident fraction, not a fitted value. One selector keeps
the standard first-four/history coverage, while the isolated core ablation
reserves anchor-one top-8 and fills 24 experts from later corrected utility.
Same-cache tests verify that zero coefficient leaves anchor-one router logits
exact. Cross-process BF16 source/candidate tensors are not treated as bitwise
identity evidence.

The best residual-damped candidate improved route hit/selected mass by only
+0.003123/+0.002240, far below the frozen +0.02 signal, so the result is
`STOP/PIVOT`. Route metrics and probe cost are teacher-forced current-policy
measurements; transfer is simulated. Held-out route, accuracy, free generation,
exact-token identity, NLL/perplexity, runtime, and speedup were not measured.

## Token-aligned one-forward state retrieval v1

The immutable config and sample manifest are
`configs/analysis/pseudo_one_forward_token_aligned_retrieval_v1.yaml` and
`configs/analysis/pseudo_one_forward_token_aligned_retrieval_v1_samples.json`,
with SHA-256 values
`19a368fe6a5e819cc0e5b3a6c3528ccf9baf94a66ba4733a278e9c0d1e5cadff` and
`dc18107f178100f41df1ffccdf23f07ddd3f611044f4b7f7ac5da1edbad36c69`.
They were committed before implementation and model output. Run the two offline
shards and then aggregate/finalize/validate with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_token_aligned_retrieval run --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_token_aligned_retrieval run --gpu 1 --shard-index 1 --shard-count 2
python -m pseudoroute.benchmark.pseudo_one_forward_token_aligned_retrieval aggregate
python -m pseudoroute.benchmark.pseudo_one_forward_token_aligned_retrieval finalize
python -m pseudoroute.benchmark.pseudo_one_forward_token_aligned_retrieval validate
```

Each JSON+safetensors pair is written atomically and checksum-resumed. The final
root validates 20 candidate pairs, four checksum-pinned uncorrected references,
59 manifest artifacts, and zero failure markers. The authoritative manifest
SHA-256 is
`512c0670002ceee7b201c7f13b948e63766225887a5b778bceae4924e19ed893`.

Retrieval banks contain full-prefill native router inputs and exact native MoE
outputs, followed only by states actually executed on the same hard policy's
saved-token replay. Exact-token ties choose the most recent history position;
the optional fallback uses float32 cosine under the pinned model's native input
embedding and also chooses the most recent tie. Similarity is fixed at one for
exact matches, zero for missing exact-only matches, and clamped to `[0,1]` for
fallback. No value is fitted or calibrated.

Every candidate still uses exactly one native causal H=8 pseudo traversal per
boundary, correct RoPE positions, full top-8 access at boundary zero, and the
previous realized B=32 subset later. Production cache/RNG is unchanged and the
shadow cache is discarded. The result is `STOP/PIVOT`: all five variants
regressed the uncorrected route signal. Route metrics are teacher-forced current-
policy measurements, probe/retrieval latency and memory are measured, and
transfer is simulated. Held-out route, task accuracy, free generation, exact-
token identity, NLL/perplexity, closed-loop runtime, and speedup were not
measured.

## Mid-layer shifted self-conditioning v1

The immutable config and sample manifest are
`configs/analysis/pseudo_one_forward_midlayer_self_conditioning_v1.yaml` and
`configs/analysis/pseudo_one_forward_midlayer_self_conditioning_v1_samples.json`,
with SHA-256 values
`75f4c028871632139a3f68f290726cb4834931a3a89cd2cc16883e050ebbb67b` and
`8314356459cbf1bf36bfafffecc0a0a2bb7fd750c22b85ef50cb921ff4b8f535`.
They were committed before implementation and new model output. Reproduce the
two offline shards and aggregate/finalize/validate with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_midlayer_self_conditioning run --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_midlayer_self_conditioning run --gpu 1 --shard-index 1 --shard-count 2
python -m pseudoroute.benchmark.pseudo_one_forward_midlayer_self_conditioning aggregate
python -m pseudoroute.benchmark.pseudo_one_forward_midlayer_self_conditioning finalize
python -m pseudoroute.benchmark.pseudo_one_forward_midlayer_self_conditioning validate
```

Each JSON+safetensors pair is atomic and checksum-resumable. The final root
validates 12 candidate pairs, four checksum-pinned uncorrected references, 44
manifest artifacts, and zero failure markers. The authoritative manifest
SHA-256 is
`f5098b420819830c12ca8cd7613baf6a784314b70477b3ea02a720d4a966aa5c`.

Every candidate uses one causal H=8 forward, native fresh MoE residuals, full
top-8 access at boundary zero, and the current policy's previous realized B=32
subset later. After layer 23, one seven-query native LM-head call produces a
greedy or top-8 expected token embedding for the next anchor. Only anchors 2–8
are shifted, per-anchor L2 norm is restored, and the shadow state continues
through layers 24–47. No future true token, answer, accuracy, learned value,
offline prior, route table, default-vector value, or retrieved state is used.

The raw rows retain a separate non-gating BF16 post-cast norm diagnostic. Its
`torch.allclose(rtol=atol=1e-3)` check passes 29/96 boundary-policy cases. The
implementation applies per-anchor norm restoration and the FP32 native unit
test passes; this 0.1% tolerance is tighter than BF16 quantization and was not a
frozen progress audit.

All cache/RNG/shadow/information and exact call-count audits pass. The result is
`STOP/PIVOT`: the best hit/mass gains were only +0.000264/+0.000259 versus the
frozen +0.02 signal. Metrics are teacher-forced current-policy route evidence;
probe and LM-head costs are measured and transfer is simulated. Held-out route,
task accuracy, free generation, exact-token identity, NLL/perplexity, closed-
loop runtime, and speedup were not measured.

## One-forward current-context continuation v1

The immutable config and sample manifest are
`configs/analysis/pseudo_one_forward_context_continuation_v1.yaml` and
`configs/analysis/pseudo_one_forward_context_continuation_v1_samples.json`,
with SHA-256 values
`519502aa463af860f4f639ff1233bf14d77bfca6ee6cb5aeff28f80817316897` and
`20a38b86b2b04ded8ded9faaf1c9eb0a90cb73668e9c23980188367a3369b4d4`.
They were committed before implementation and model output. Reproduce the two
offline shards and aggregate/finalize/validate with:

~~~bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m pseudoroute.benchmark.pseudo_one_forward_context_continuation run --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m pseudoroute.benchmark.pseudo_one_forward_context_continuation run --gpu 1 --shard-index 1 --shard-count 2
python -m pseudoroute.benchmark.pseudo_one_forward_context_continuation aggregate
python -m pseudoroute.benchmark.pseudo_one_forward_context_continuation finalize
python -m pseudoroute.benchmark.pseudo_one_forward_context_continuation validate
~~~

Each candidate pair is atomic and checksum-resumable. The final root validates
12 candidate pairs, four checksum-pinned uncorrected references, 45 manifest
artifacts, and zero failure markers. The authoritative manifest SHA-256 is
`1ce50ae324605f7a9ac60e87242c37b9856c9e2db5a5e98879990864c4f854d9`.

At boundary t, matching sees only prompt tokens, the current hard policy's
already-realized tokens, and the known sampled-next token. Copied continuation
indices must precede the known-context length. No future true token, vanilla
trajectory, answer, accuracy, offline table, learned/fitted value, default
vector, route prior, or retrieved hidden state enters a candidate. Every
candidate then performs one causal H=8 native traversal with fresh native MoE
residuals: full top-8 access at boundary zero and the policy's preceding
realized B=32 subset thereafter.

All cache/RNG/shadow/information and exact call-count audits pass. The selected
unigram-continuation variant improved hit/mass by +0.030090/+0.033024 and emitted
`DEVELOPMENT_ROUTE_SIGNAL`. This protocol authorizes no held-out route or task
accuracy even after a positive result. Route evidence is teacher-forced on each
current policy's own state; lookup/probe cost is measured, transfer is
simulated, and free generation, exact-token identity, NLL/perplexity, closed-
loop runtime, and speedup were not measured.

## One-forward Qwen/GSM8K accuracy pilot v1

The separately frozen actual-generation config and sample manifest are
`configs/benchmark/pseudo_one_forward_accuracy_pilot_v1.yaml` and
`configs/benchmark/pseudo_one_forward_accuracy_pilot_v1_samples.json`, with
SHA-256 values
`d9515855b897189fde9f36fba151af5467ebc93e46bbf09bb79ad5b39e5f10af` and
`fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046`.
The artifact root is `artifacts/pseudo_one_forward_accuracy_pilot_v1`.

Run or resume a logical shard from the pinned offline cache, then aggregate and
validate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot run \
  --gpu 0 --shard-index 0 --shard-count 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot run \
  --gpu 0 --shard-index 1 --shard-count 2
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot aggregate
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot finalize
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_one_forward_accuracy_pilot validate
```

Logical shard assignment remains the frozen two-shard modulo mapping even when
one physical GPU executes the shards serially. Each completed JSON row is
written atomically and reused only after config, manifest, schema, state, cap,
and payload checksum validation. A missing row is regenerated without deleting
valid rows or provenance. The completed root validates 24 actual rows, three
smoke rows, 50 manifest artifacts, checksum resume, and zero failure markers.
The artifact-manifest SHA-256 is
`12e9d371485b7375561f8f194fbdb6e3f1ea57fda02cddabfd12a17a408eef67`.

The final execution environment was Linux 6.8.0-100-generic, Python 3.14.6,
PyTorch 2.13.0+cu130, CUDA runtime 13.0, driver 580.126.09, Transformers 5.14.1,
and one A100-SXM4-80GB. Frozen vanilla is reused. Accuracy, generated tokens,
token agreement, route/mass coverage, NLL/perplexity, probe latency, total
generation time, and peak CUDA memory are measured. Transfer is simulated.
Future-exact content additionally performs natural autoregressive lookahead and
is excluded from deployable one-forward and speedup claims.

## Matched-eight Qwen/GSM8K hard routing-oracle amendment

The post-hoc amendment is frozen in
`configs/benchmark/pseudo_one_forward_hard_oracle_v1.yaml`, with SHA-256
`60e03148a24ad16859ca21631a3644d4db6930b0bdaddc482060646b77e902b6`.
It reuses the parent resolved sample manifest verbatim at SHA-256
`fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046`.
The artifact root is `artifacts/pseudo_one_forward_hard_oracle_v1`.

Run or resume the sole offline GPU worker, then finalize and validate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_one_forward_hard_oracle run --gpu 0
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_one_forward_hard_oracle finalize
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_one_forward_hard_oracle validate
```

The runner writes one atomic checksum JSON per sample and skips only rows whose
pilot/config/sample identity, policy, cap, state, and payload checksum all
match. Failure markers are preserved. The completed root validates eight
actual hard closed-loop rows, 19 manifest artifacts, checksum resume, and zero
failure markers. Artifact-manifest SHA-256 is
`336d97c416aba6a80ad205228667bbbde570378a872c5cf7bd707544ce98cc28`.

At each boundary the oracle naturally rolls out from the hard policy's current
context, restores or preserves the production cache and RNG, aggregates future
selected routing weights, and selects deterministic per-layer top-32 subsets.
Actual generation masks all other routed experts before native top-k and
normalization. The model and dataset load in offline mode from the pinned cache;
no vanilla rows are regenerated.

Accuracy, token agreement, route coverage, NLL/perplexity, total generation
runtime, and peak CUDA allocation are measured. Expert transfer reduction is
simulated. This diagnostic is nondeployable, not a one-forward pseudo method,
and cannot support a production speedup or full-dataset accuracy claim.
## Real Qwen expert-offload speed pilot v1

The immutable config and sample manifest are
`configs/benchmark/qwen_real_offload_speed_pilot_v1.yaml` and
`configs/benchmark/qwen_real_offload_speed_pilot_v1_samples.json`, at SHA-256
`9276363796e711baf487415a85fde49fc525ae42e7be701972867fb8713ed1ec` and
`aafd570fe56d878074cc6f5666dd26df8da528086776400226c15727b77ee819`.
All model, dataset, vanilla-row, and mask-only reference inputs are local and
checksum-pinned; the runner does not download.

Run or resume in the frozen order:

```bash
CUDA_VISIBLE_DEVICES=0 python -m   pseudoroute.benchmark.qwen_real_offload_speed run --stage smoke --physical-gpu 0
python -m pseudoroute.benchmark.qwen_real_offload_speed audit-smoke
CUDA_VISIBLE_DEVICES=0 python -m   pseudoroute.benchmark.qwen_real_offload_speed run --stage actual --physical-gpu 0
python -m pseudoroute.benchmark.qwen_real_offload_speed finalize
python -m pseudoroute.benchmark.qwen_real_offload_speed validate
```

Rows are atomic and reused only after pilot, config, sample, stage, ID, policy,
cap, serialization, and payload checksum validation. Full expert extraction is
reported in `setup/` and excluded from inference throughput. Failed attempts
are retained. An identity-failure diagnostic contains measured tokens, timing,
H2D metrics, and its own checksum; the runner can recover it into an explicitly
marked completed measurement without repeating GPU generation.

The final root contains four actual closed-loop rows and two smoke rows.
Traditional rows require exact frozen-vanilla identity. Candidate identity is a
separately reported gate: both measured candidates remained correct but
diverged from their frozen mask-only references. Validation requires actual
H2D, pinned CPU sources, 32 CUDA slots/layer, no full expert parameter on CUDA,
zero identity materialization, lossless identity, and zero candidate production
misses. The artifact-manifest file SHA-256 is
`3704e8d3bbb7c4a8f7c8eebe39ea7275214668365c4c9547548a2bfbacfd8ccf`;
its internal payload SHA-256 is
`44bb91376054f514766e8e2bf6ebba69c05e695c0208164218825681a429ff07`.

Task outputs, exact-token identity, wall time, actual H2D bytes, cache events,
and CUDA-event transfer/stall are measured. Legacy route/transfer estimates
inside candidate rows are simulated diagnostics. The runner implements no
transfer/compute overlap and includes research instrumentation overhead; it
does not establish NVMe, NVLink, multi-GPU, fused-kernel, full-dataset, or
production-serving behavior.

## Penultimate-token joint-planning wave-one accuracy checkpoint

The immutable config and sample manifest are
`configs/benchmark/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1.yaml` and
`configs/benchmark/pseudo_penultimate_joint_qwen_gsm8k_wave1_v1_samples.json`,
with SHA-256 values
`57242f91f204c4e765d893dd754e4e523f8fdce62c303827d2e375cec4cc654c` and
`0f4fd75760390fb8ea468af888c8dcd0b22483cdb2af0f96f51a9833c6614d10`.
They reuse the exact eight wave-one IDs and checksum-pinned online baseline.

Run or resume the no-offload logical-equivalence checkpoint from the pinned
offline cache, then finalize and validate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_penultimate_joint_accuracy run-smoke --gpu 0
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_penultimate_joint_accuracy smoke-audit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pseudoroute.benchmark.pseudo_penultimate_joint_accuracy run-wave1 --gpu 0
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_penultimate_joint_accuracy finalize
PYTHONPATH=src python -m \
  pseudoroute.benchmark.pseudo_penultimate_joint_accuracy validate
```

Rows are atomic and checksum-resumable. Final validation covers one smoke row,
eight actual candidate rows, eight checksum-pinned external baseline rows, 25
manifest artifacts, zero failure markers, and a terminal user-confirmation
checkpoint. Artifact-manifest SHA-256 is
`544d50a17c8fcb3300514b9b160376cb79a9eb6e45d0b56986d0ecd6a06c33d3`.

Actual hard closed-loop accuracy, token/route metrics, NLL/perplexity,
logical-simulator cost, and cache/RNG/bridge parity are measured. Transfer is
simulated. All experts remain resident, and the disposable duplicate bridge
exists only to reproduce the intended state dependency. Actual offload runtime,
prefetch overlap, joint/fused runtime, speedup, and full-dataset accuracy are not
measured.
