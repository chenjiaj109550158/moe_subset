# One-forward context-continuation protocol v1

## Question and frozen scope

This development-only analysis tests whether a coherent continuation copied
from the current request's known token context gives a better calibration-free
H=8 pseudo sequence than reversing the seven most recent known tokens behind
the sampled-next anchor. It reuses only `test-0`, `test-439`, `test-879`, and
`test-1318`, with Qwen3-30B-A3B BF16, B=32, native top-8, 64 route tokens, and
the checksum-pinned uncorrected recent-sequence reference.

No new row, natural trace, model, dataset, download, held-out evaluation, or
accuracy execution is authorized. Variant ranking uses route and measured cost
only.

## Deployable information boundary

At boundary t, the known context is the rendered prompt followed by current
policy teacher-forced realized tokens strictly before t and the already-known
sampled-next token at t. No token after t, vanilla trajectory, answer,
correctness, accuracy, offline continuation table, fitted value, learned
parameter, default vector, expert prior, route-transition table, or retrieved
hidden state is available.

The current query is a suffix ending at the sampled-next token. Earlier matches
use exact tokenizer IDs, must not overlap the current query, and break equal
length ties at the most recent start. Copied indices must be smaller than the
known-context length. Anchor one is always the sampled-next token.

## Frozen variants

`longest_suffix_full_continuation` selects the longest eligible suffix whose
earlier occurrence has seven known successor tokens and copies those successors
into anchors 2–8. `sampled_unigram_full_continuation` uses only the sampled
token and chooses its most recent earlier occurrence with seven successors.
Both fall back to the unchanged recent-sequence anchors if no match exists.

`longest_suffix_partial_recent_fill` selects the longest eligible suffix with
at least one known successor, copies up to seven successors, and fills remaining
anchor indices with the unchanged recent-sequence token at the same index. If
there is no eligible match it also falls back completely.

No minimum suffix length, probability threshold, tuned maximum length, or
post-result variant is permitted.

## Shared pseudo forward and commitment state

Every candidate runs one native causal eight-query traversal. Boundary zero may
execute Qwen native top-8 experts; later boundaries execute only the policy's
previous realized layer-local B=32 subset. The resulting fresh native MoE
residual advances the same shadow traversal. The selector remains
`first_four_anchor_core_plus_history_fill` so this analysis changes only pseudo
content. Production KV cache and RNG are read-only, and the shadow cache is
discarded after the boundary.

The teacher-forced evaluator then executes the candidate hard subset on the
same saved v17 tokens while retaining the policy's own hard-subset hidden,
router, residual, history, and next-window subset state. It is not vanilla route
replay or free generation.

## Frozen gate and reporting

A route-signal candidate must improve both mean route hit and mean selected mass
by at least +0.02 over the checksum-pinned uncorrected reference, keep anchor-one
absolute regression within 0.005, retain at least 0.30 simulated transfer
reduction, keep measured mean probe latency at or below 0.50 seconds, and pass
all cache/RNG/shadow/call-count/information-boundary audits.

The report must include continuation and fallback coverage, copied-anchor
fraction, suffix-length strata, per-anchor and layer-half route metrics, paired
sample bootstrap intervals, worst cases, content-planning latency, probe cost,
simulated transfer, atomic checksum/resume validation, and provenance.

This protocol authorizes no held-out route or task accuracy regardless of the
development outcome. A positive result is development route signal only; a
failed gate is `STOP/PIVOT`. It does not measure exact-token identity,
free-generation NLL/perplexity, closed-loop runtime, or runtime speedup.
