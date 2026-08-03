# Mass-preserving residual Qwen/GSM8K accuracy pilot v1

## Immutable question and scope

This pilot tests whether the held-out route-gate winner preserves GSM8K answers
under true hard-subset Qwen generation. The scope is exactly the 16 IDs already
predeclared as `closed_loop_wave_1` and `closed_loop_wave_2` in
`pseudo_embedding_qwen_gsm8k_v1`; no result, answer, or correctness selected an
ID. The model revision, bfloat16 precision, H=8, B=32, greedy decode, exact v17
prompt/parser/stop rules, and original 512-token cap are immutable. The config
and resolved-manifest SHA-256 values are pinned by the protocol test and must be
recorded here before any new accuracy output.

## Policies and actual execution

All three policies use the same 16 rows and their own closed-loop context. Every
production MoE layer masks logits outside its active layer-local B=32 subset
before native top-8 selection and normalization. No row may be identity
materialized.

`hard_oracle_commitment` performs fresh full-expert natural lookahead from its
own hard-policy boundary context and selects each layer's top-32 experts by
future selected routing mass. The already measured eight wave-1 rows are
checksum-pinned external references and are not rerun; only wave 2 is new.

`previous_route_commitment` uses frozen static-frequency top-32 subsets for the
first generation window. Each later subset is selected from that policy's own
preceding realized window of pre-mask natural route records. Prompt-route first
window rows from legacy evaluators are forbidden.

`natural_top8_intersection_zero_missing` uses the sampled-next-token unigram
continuation, with recent-sequence fallback, in one batched causal H=8 shadow
traversal per boundary. Boundary zero executes the full native shadow top-8.
Later pseudo layers execute only natural top-8 experts resident in the current
production B=32 subset, retain their native weights, and make missing natural
mass contribute zero. They never substitute or renormalize missing experts.
The candidate remains calibration-free and uses no mean default-vector value.

## Information and state boundary

The candidate may read the sampled token, current policy prompt and realized
tokens, native router/model parameters, a copy-on-write production cache, the
current B=32 subset, and its own preceding pre-mask route records. It cannot
read future true tokens, the frozen vanilla future trajectory, answers,
correctness, task accuracy, learned/fitted parameters, offline route or
continuation tables, expert priors, default-vector means, or retrieved hidden
states. The production cache and RNG must be unchanged by planning; the shadow
cache is discarded after every boundary.

The previous policy alone reads the pinned default-vector artifact's count
tensor to construct its frozen static-frequency first window. Mean vectors are
not loaded. The source artifact contains 444 unobserved layer/expert pairs saved
as zero and is not described as a complete expert prior.

## Frozen samples, reuse, execution, and gate

The sample order is test-44, test-632, test-444, test-519, test-1311,
test-1264, test-825, test-252, test-668, test-892, test-1136, test-850,
test-760, test-842, test-87, and test-658. Frozen v17 vanilla is 16/16 and is
not regenerated. Eight wave-1 hard-oracle rows are reused only after their
payload and artifact-manifest checksums validate. The execution therefore adds
40 atomic rows: eight wave-2 hard-oracle, 16 corrected previous-route, and 16
candidate rows.

A fixed 16-token smoke on test-44 checks corrected previous first-window
semantics and candidate cache/RNG/mass-preserving audits; its result cannot
change the policy, IDs, order, or accuracy gate. One physical GPU runs at most
one worker. Rows are atomic, checksum-resumable, and retain failure markers.
No network download is authorized.

Because frozen vanilla is 16/16, the predeclared allowed drop is one question:
each policy needs at least 15/16 successes. Sixteen successes is the strong
preservation signal. Paired accuracy intervals use 10,000 sample-level
bootstrap resamples with seed 20260803. Accuracy cannot tune a variant. Even a
positive result is capped at `NARROW` because N=16.

## Reporting boundary

Measured quantities are actual GSM8K accuracy, generated tokens, exact-token
agreement, first token and route divergence, policy-context NLL/perplexity,
route coverage, probe cost, generation time, and peak CUDA memory. Expert
transfer and stall remain simulations. The result cannot establish production
offloading speedup, full-dataset preservation, hard-oracle deployability, or a
statistical full-suite GO.

## Frozen fingerprints

The immutable config SHA-256 is
`7370d4c3208fcd32ab7b0f903409a7132c14cd784b7360cc7cf7863720f0d46b`.
The immutable sample-manifest SHA-256 is
`ac1aeb3da4778d09628c2e3188fd524392ce253b9fc04b8c795c0312f82dadfe`.
Both are committed before the smoke or any new accuracy row.
