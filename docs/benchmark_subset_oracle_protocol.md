# Benchmark subset oracle v1 protocol

Recorded and frozen on 2026-07-30 UTC before any subset-oracle route trace,
open-loop aggregate, hard-commitment smoke, or hard-commitment accuracy result
was produced. The machine-readable source is
`configs/benchmark/benchmark_subset_oracle_v1.yaml`, whose resolved content
fingerprint is
`0463596e816da5db528061b50c3b06e81b11bf6d012aa106963ba1c55e712227`.

## Question and information boundary

This stage asks whether a perfect routing-information oracle can make a
layer-local fixed expert subset useful for more than one decode token. It does
not use benchmark labels, reference answers, scorer outcomes, or generated
correctness to select an expert. At each boundary the oracle may inspect only
the next `H` natural router decisions from the current boundary state.

The v17 `oracle_pf` result is not evidence for this question. It is a
native-top-k, one-layer-ahead identity result and was materialized from vanilla
only after real-forward parity. This stage calls that result
`perfect_layer_ahead_natural_ids_and_weights`; it never calls it multi-token
hard-subset accuracy.

## Frozen scope

The checkpoints, revisions, prompts, targets, generated vanilla token IDs,
decoding settings, task caps, datasets, scorers, HumanEval/MBPP+ execution
sandbox, and answer parsers are inherited byte-for-byte from
`speculating_experts_accuracy_v17` at config fingerprint
`896744077611beaf7692ee2c9ac94fcc5326268f86e8b5ff297960a0da152ff9`.
No v17 vanilla task is regenerated.

Qwen3 has 48 MoE layers, 128 layer-scoped routed experts, native top-8 routing,
no shared expert, float32 full-router softmax, and selected top-k
renormalization. GPT-OSS has 24 MoE layers, 32 layer-scoped routed experts,
native top-4 biased-logit routing, selected-logit softmax, no shared expert, and
native MXFP4 expert storage. A softmax over all GPT-OSS logits is not a native
routing mass, so full-router-mass selection is reported only for Qwen3.

Horizons are 1/2/4/8/16. Qwen budgets are 8/16/32/64/128; GPT-OSS budgets are
4/8/16/24/32. The all-expert budget is a reference and is never treated as a
saving. Every subset is selected independently per layer at exact cardinality
`B`, with ascending expert ID as the final tie break.

## Natural replay and open-loop grid

The representative trace rows are the four deterministic dataset quartiles
listed explicitly in the config. The first 128 saved decode tokens per row are
teacher-forced with a KV cache in chunks of 16. Prompt rendering is reconstructed
through the v17 renderer and must match the saved prompt SHA-256. Every shard
saves native selected IDs/weights and native router logits in safetensors, plus
source token and prompt checksums. An eight-token autoregressive replay on the
fixed smoke ID for every task must match the corresponding saved v17 token
prefix and the teacher-forced routes/logits before a trace is admitted.

For each horizon, boundaries are non-overlapping and start at decode position
zero. The required selectors are future selected-routing-mass, future binary
count, previous realized window, static frequency, and native future
full-router mass for Qwen3 only. Previous-route uses static frequency at the
first boundary.

Raw rows retain model, task, sample, boundary, horizon, layer, budget, method,
natural routes, subset, selected/full-mass coverage, route hit, expert union,
subset churn, fallback loads, and transfer fields. Aggregates contain mean,
median, P05, P95, worst, layer/task/context/router-margin strata, and concrete
worst cases.

## Residency and transfer semantics

The fixed subset consumes exactly `B` routed-expert slots per layer. At a
boundary, newly selected members relative to the prior subset are prefetched.
Lossless misses are transient demand loads: each distinct missing natural expert
for a layer/token is loaded, executed with its native ID and weight, and does not
displace a pinned subset member. Slot miss frequency is `1 - route_hit`.
Transfer reduction is relative to transient natural on-demand loading of every
distinct native selected expert for every layer/token. Bytes use checkpoint
physical expert storage. Exposed stall is simulated at 25 GiB/s plus 10 µs per
load; it is never described as measured runtime.

## Operating-point selection

Only future selected-routing-mass points may enter hard-oracle accuracy. A point
must pass every numerical open-loop/fallback/transfer criterion in the config
on every one of the six tasks. All-expert points are excluded and residency must
be at most 75% of routed experts. At most one point per model is chosen with the
fully deterministic ordering in the config. Smoke accuracy cannot alter,
replace, or add a point. If no point passes, that model receives `STOP/PIVOT`
from the open-loop gate.

## Closed-loop execution

Natural accuracy/text/tokens are reused from v17. Lossless accuracy is
identity-materialized only after every-task real-forward smoke proves exact
token identity; its fallback and transfer metrics come from route replay.
The fixed mechanism smoke uses `H=16` and the model's native-top-k budget so it
can prove that hard masking changes an executed route. It is not an operating
candidate and cannot participate in point selection; selected-point smoke runs
separately after the open-loop gate.

Hard oracle and previous-route commitment are actual cached autoregressive
generation. At every hard-oracle boundary, a natural rollout starts from the
current policy state, records the next `H` native routes, and consumes no label
or ground truth. RNG and cache length are saved; after selection, the lookahead
cache is cropped back to the boundary and the policy window is replayed from
that exact state. If cache crop parity is unavailable, the run fails rather
than silently changing semantics.

Hard execution masks every outside-subset router logit before native top-k.
Qwen applies its full softmax/top-k/top-k renormalization; GPT-OSS applies
biased-logit top-k and selected-logit softmax. The frozen model revisions have
no shared experts; any contrary runtime inspection fails before execution.

Generation uses the v17 prompt, per-task cap, sampling mode, checkpoint
temperature/top-p/top-k, stop policy, and seed. In particular GPT-OSS AIME keeps
32,768 new tokens. Each sample/policy row is atomically committed and checksum
addressed. Four logical shards run sequentially per physical GPU, with no more
than one 32k-capable worker on a GPU.

NLL is the v17 vanilla-token cross entropy on the policy's current context.
Per-sample rows also retain measured wall time, exact token agreement, first
token and route divergence, executed routes, task score, planned/fallback bytes,
churn, and simulated stall.

## Accuracy and decisions

The frozen maximum accuracy loss is an integer number of questions, computed
only from v17 vanilla as `ceil(2 * sqrt(N * p * (1-p)))`, with a one-question
minimum. Hard successes must be no lower than vanilla successes minus that
integer. Paired deltas, discordant counts, paired standard errors, and 95%
intervals are still reported. AIME additionally reports integer wins/losses and
total correct, never only percentages.

No macro average can override a task failure. Every model/task/point is labeled
`GO`, `NARROW`, or `STOP/PIVOT`. `NARROW` must name the exact tasks and point to
which it applies. Predictor training remains prohibited unless the final result
is all-task `GO` or an explicitly task-scoped `NARROW`; failure of the perfect
hard oracle stops the predictor path.
