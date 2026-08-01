# Qwen/GSM8K pseudo-embedding focused pilot v1

## Scope and claim boundary

`pseudo_embedding_qwen_gsm8k_v1` is a training-free, versioned pilot for
`Qwen/Qwen3-30B-A3B-Instruct-2507` revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` in bfloat16 on frozen GSM8K v17
test rows. The only operating point is `H=8,B=32`: 32 of 128 routed experts per
layer, native top-8, no shared experts. It cannot support a full-GSM8K `GO`, a
general model claim, or a runtime-speedup claim. Its strongest possible positive
decision is task- and point-scoped `NARROW`.

The three actual policies are `hard_oracle_commitment`,
`previous_route_commitment`, and `pseudo_embedding_commitment`. Every actual row
must be true closed-loop generation. Identity materialization is forbidden.
Transfer and exposed-stall values remain trace simulations; probe and generation
runtime are measured separately.

## Frozen samples and selection

The exact manifest is
`configs/benchmark/pseudo_embedding_qwen_gsm8k_v1_samples.json`. Development
reuses the four pre-existing authoritative Qwen/GSM8K trace rows: `test-0`,
`test-439`, `test-879`, and `test-1318`.

The remaining partitions were fixed without accuracy, correctness, target answer,
or question content. Excluding the four development indices, all 1,315 eligible
`(row_index,sample_id)` pairs were ranked by ascending SHA-256 of:

```text
pseudo_embedding_qwen_gsm8k_v1|sample-ranking-v1|{row_index}|{sample_id}
```

Ties use ascending row index. The first 2 rows are mechanism smoke, the next 8
are the disjoint held-out route set, and the next 16 are the closed-loop pilot in
two fixed 8-row waves. Partitions are disjoint. No ID may be replaced after a
pseudo result is observed.

Only after those IDs were resolved was their frozen vanilla correctness
aggregated for the accuracy gate. The 16-row closed-loop set is 16/16 under v17
vanilla. The frozen allowed drop is
`max(1,ceil(2*sqrt(N*p*(1-p)))) = 1`; each actual policy must score at least
15/16. Paired deltas use a 10,000-resample sample-level percentile bootstrap with
seed 20260801. Accuracy never selects the pseudo variant.

## Information boundary and shadow algorithm

The main regime is `online_post_sample`. At a generation boundary the next token
has already been sampled and is the input to the next production forward. The
probe may read that one token ID and the current policy's production cache. It may
not read later real tokens, a vanilla future trajectory, the GSM8K answer, or any
correctness value.

For anchors 1 through 8, the native Qwen layer path is shadowed at exact future
cache positions. Independent anchors attend to the production prefix only;
causal anchors additionally attend to earlier probe-local pseudo K/V. Each sparse
layer runs native attention and RoPE, the exact native Qwen gate, and records raw
router logits plus full pre-top-k probabilities. The default-vector variant adds
the native pseudo-top-8 normalized mixture of the frozen expert means. The zero
variant adds no routed-expert contribution. Dense/shared paths are absent for
this checkpoint. Shadow residuals continue to the next layer.

Production K/V is read-only or copy-on-write. Before and after each probe, the
runner compares cache sequence length, layer identities, K/V tensor identities,
data pointers, shapes, and version counters. It also compares CPU and CUDA RNG
states. Probe-local K/V is discarded at the boundary. The production hidden
state is never replaced by a shadow state.

Per layer, utility is the sum of full pre-top-k probabilities over eight anchors.
Exactly 32 layer-scoped routed experts are selected by descending utility, with
ascending expert ID as the tie break. Default-vector evidence is limited: 444
layer/expert pairs were unobserved during disjoint WikiText calibration and are
stored as zero vectors. This artifact is not a complete expert prior.

## Frozen variants

The primary variant repeats `sampled_next_token` across independent anchors and
uses the selected/top-k default-vector mixture. The three mandatory ablations are
current-token independent content, sampled-next-token causal pseudo sequence, and
sampled-next-token independent anchors with zero expert contribution.

Expected top-8 next-token embedding is optional only if its measured branching
cost is reasonable. If it is not run, the artifact must contain
`expected_top_m_blocker.json` with the concrete implementation blocker and ETA;
an incompatible proxy cannot replace it.

## Route evaluation and variant selection

Development uses only the four existing traces and a maximum of 128 saved decode
tokens per row. Held-out natural routes may be added only by teacher-forced replay
of the saved v17 token trajectories. A current-load argmax mismatch is recorded
but does not invalidate forced replay; development router tensors still require
exact authoritative-trace parity. Windows are non-overlapping from boundary
zero. Raw data retain every sample/boundary/layer/anchor router score and subset,
plus route hit, selected/full mass, fallback, churn, estimated transfer, latency,
temporary memory, attention queries, router calls, and synchronization count.
Aggregates include layer, context-position, and per-layer router-margin strata,
sample-level bootstrap intervals, and concrete worst cases.

Variants are ranked on development data only. A variant is eligible only when it
simultaneously:

- improves mean route hit over previous route by at least 0.05 absolute;
- improves mean selected routing mass by at least 0.05 absolute;
- is no worse than frozen static frequency on both metrics;
- keeps exactly 25% per-layer residency;
- has at least 30% estimated transfer reduction;
- passes all cache, RNG, native-semantics, and information-boundary tests; and
- has measured probe latency, temporary memory, router calls, and attention queries.

Eligible variants are ordered by mean selected mass, mean route hit, probe
latency, then lexical key. No accuracy is inspected. A held-out strong candidate
must recover at least 25% of the oracle-minus-previous gap on both route hit and
selected mass; the reference thresholds are 0.6967 and 0.7162 respectively.
Failure yields `STOP/PIVOT` and forbids actual accuracy execution.

## Closed-loop execution and stop rule

Only a held-out route-gate pass authorizes the fixed 16-row actual pilot. The
three policies share the exact IDs, immutable v17 rendered prompts, greedy seed,
GSM8K parser, and original 512-token cap. Oracle subsets come from natural
lookahead at the current policy boundary and are rewound before execution.
Previous-route subsets use that policy's preceding realized window of pre-mask
natural route records and static frequency for its first window. Pseudo subsets
use only that policy's current context.

Rows commit atomically by sample and policy with checksums. At most one worker may
occupy a physical GPU. After the first fixed 8-row wave, elapsed sample/token
rates project the cost of all 16 rows × 3 policies. If projected total time is
over 24 hours, the runner saves the completed rows and ETA and stops for explicit
human approval before wave 2; IDs, cap, policies, and sample count remain frozen.

Final reporting separates reused measured vanilla accuracy, actual closed-loop
accuracy, exact-token identity, open-loop route replay, simulated transfer/stall,
measured probe cost, measured generation runtime, and any identity-materialized
provenance. The pilot decision cannot override the earlier full-scope decisions.
