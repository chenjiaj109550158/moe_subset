# Penultimate-joint Qwen/GSM8K wave-1 accuracy pilot v1

## Question and frozen scope

This pilot asks whether moving pseudo planning one production step earlier can
preserve the accuracy of the current online-post-sample method. It uses exactly
the eight already predeclared `closed_loop_wave_1` GSM8K rows, in their original
order, with Qwen3-30B-A3B, bfloat16, greedy decoding, H=8, B=32, saved v17
prompts, parser, stopping rules, and original 512-token caps. No accuracy,
answer, or correctness selected an ID or tuned the variant.

The immutable config SHA-256 is
`57242f91f204c4e765d893dd754e4e523f8fdce62c303827d2e375cec4cc654c`. The
immutable sample-manifest SHA-256 is
`0f4fd75760390fb8ea468af888c8dcd0b22483cdb2af0f96f51a9833c6614d10`.
Both are committed before any new smoke or accuracy output.

## Exact timing hypothesis

For a realized eight-forward window, the token consumed by its eighth and last
production forward is already known. After that forward samples the eighth
window output, the consumed token is the penultimate sampled token. The intended
runtime will combine that real token and eight future pseudo positions into one
causal sequence. The real bridge uses the current hard B=32 production
semantics; pseudo positions use the current B=32 subset and preserve the natural
top-8 intersection weights while making missing mass contribute zero.

Pseudo content is otherwise unchanged from the current candidate: the bridge
token seeds `sampled_unigram_full_continuation`, with `recent_sequence_causal`
fallback, using only the request's known context. The newly sampled last token
is unavailable to planning and cannot affect the next subset. After the bridge
route completes, the selector takes the first four future pseudo-anchor cores
and fills B=32 from all eight realized pre-mask routes of the completed window.
The next subset is committed before the next window's first production forward.

There is no preceding decode window at boundary zero, so the first subset uses
the existing online-post-sample sampled-unigram bootstrap with full native
shadow top-8 access. This exception is fixed rather than selected by results.

## Accuracy-only logical simulator

This checkpoint does not implement or time the fused/offloaded runtime. For an
exact state dependency test, the non-offload simulator forks the production KV
cache before the last real forward, executes that token once on the disposable
cache with hard B=32 semantics, and then runs one causal H=8 pseudo traversal
from the resulting shadow cache. It discards the shadow branch, executes the
real production forward, and verifies that shadow and real bridge natural routes
match before committing the next subset.

The disposable bridge is duplicate work needed only by this accuracy simulator.
Its latency is not an estimate of a future fused joint kernel. Actual expert
offloading, prefetch overlap, joint-kernel runtime, and speedup are not measured.
They remain blocked on explicit user confirmation after this wave-1 report.

## Information, cache, and execution boundary

Planning may use only the policy's prompt, already realized tokens, the known
penultimate token, native model/router parameters, its current B=32 subset,
realized pre-mask routes, and a copy-on-write production cache. It cannot read
the just-produced last token, future tokens, the frozen vanilla trajectory,
answers, correctness, learned/fitted values, offline lookup tables, expert
priors, default-vector values, or retrieved states.

Production is true hard closed loop: logits outside the layer-local B=32 subset
are masked before native top-8 selection and normalization. Production cache and
generation RNG must be unchanged by planning, shadow state must be discarded,
and no row may be identity materialized. Accuracy execution deliberately does
not instantiate the offload engine.

## Samples, gate, and checkpoint

The exact order is test-44, test-632, test-444, test-519, test-1311,
test-1264, test-825, and test-252. Frozen vanilla and the checksum-pinned current
online-post-sample candidate are both 8/8 and are not rerun. A 16-token smoke on
test-44 must exercise at least one later penultimate boundary and cannot alter
the variant, IDs, or gate.

The candidate must score at least 7/8 to stay within the frozen one-question
allowed drop; 8/8 is the strong preservation signal. Paired bootstrap uses
10,000 sample resamples with seed 20260804. The maximum conclusion is
`PILOT_NARROW`. Regardless of the result, the pipeline stops after wave 1 for
user confirmation and does not automatically authorize fused/offload work.

## Reporting boundary

Measured outputs are actual hard closed-loop accuracy, token agreement and
divergence, policy-context route hit and selected mass, NLL/perplexity, logical
simulator cost, and cache/RNG/bridge parity. Transfer is simulated. The report
must not present simulator latency as joint runtime, claim offloading speedup,
claim full-dataset preservation, or call eight rows a full-dataset GO.
