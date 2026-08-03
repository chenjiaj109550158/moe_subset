# One-forward Qwen/GSM8K accuracy pilot v1

## Question and immutable scope

This small actual-generation pilot tests whether the cheap one-forward route
signals preserve GSM8K answers under true Qwen hard-subset decoding. It reuses
the eight predeclared closed_loop_wave_1 IDs from the original focused pilot.
No ID was selected using accuracy, and no new dataset row, natural trace, model,
revision, precision, or token cap is introduced.

The immutable config SHA-256 is
d9515855b897189fde9f36fba151af5467ebc93e46bbf09bb79ad5b39e5f10af.
The immutable sample-manifest SHA-256 is
fe8012f22e7aec13eb3ae553f725b5b8505387c7693ff0aa08f59a29b693b046.

The three policies are recent_sequence_causal,
sampled_unigram_full_continuation, and the diagnostic
future_exact_content_oracle. Frozen v17 vanilla rows are paired references and
are not regenerated. Accuracy cannot select or tune a pseudo variant.

## Shared true closed-loop generation

Every policy uses the exact saved v17 rendered prompt, greedy decoding, RNG seed,
stop strings, GSM8K parser, and original 512-token cap. Prefill uses native full
experts and captures the last eight prompt routes. Every subsequent production
token explicitly masks router logits outside the active layer-local B=32 subset,
then executes Qwen native top-8 selection and normalization. Generated tokens,
hidden states, KV cache, natural pre-mask routes, MoE residuals, and the next
subset all belong to that policy's own trajectory. No row is identity
materialized.

At an online-post-sample boundary, anchor one is the sampled token that is known
but not yet in the production cache. The H=8 pseudo sequence runs once through
all 48 native Qwen layers with correct positions, causal attention, and a
copy-on-write shadow cache. Boundary zero executes native full top-8 experts in
the shadow. Later boundaries execute only the preceding realized B=32 subset,
thereby producing fresh native pseudo MoE residuals. Production cache/RNG must
remain unchanged and the shadow cache is discarded.

Each layer reserves the union of native top-8 experts from pseudo anchors 1–4,
ranked by their summed pre-top-k probability, then fills remaining B=32 slots
from the previous realized window's pre-mask natural-route utility. At boundary
zero that history is the last eight full-expert prompt routes. Expert IDs and
ties are layer-local and ascending.

## Frozen policy content

recent_sequence_causal uses the sampled token as anchor one and the seven most
recent already-known tokens as anchors 2–8, preserving their chronological
order.

sampled_unigram_full_continuation finds the most recent earlier occurrence of
the sampled token that has seven successors already present in the current
request. It copies those successors into anchors 2–8 and otherwise falls back
to recent_sequence_causal. It cannot inspect a future production token.

future_exact_content_oracle is explicitly nondeployable. From the current
policy cache it performs a full-expert greedy natural rollout to obtain seven
successors, restores production cache and RNG, then supplies the sampled token
plus those successors to the same one-traversal pseudo mechanism. If natural
rollout terminates early, the terminal token is repeated only to maintain eight
pseudo anchors. This policy therefore adds up to seven autoregressive lookahead
calls plus one pseudo traversal per boundary and is excluded from every
single-extra-forward or speedup claim. It is a content-oracle diagnostic, not
the saved v17 future trajectory and not a full routing oracle.

## Information boundary and audits

The two deployable policies may use only the current policy prompt, already
realized tokens, sampled-next token, read-only production cache, preceding
realized subset, previous pre-mask route records, and native model parameters.
They cannot use future true tokens, the vanilla trajectory, answers,
correctness, accuracy, learned/fitted values, offline continuation or route
tables, expert priors, default-vector values, or retrieved hidden states.

The future-exact exception is limited to freshly generated full-expert greedy
successors from the current policy context. It may never read v17 future tokens,
the benchmark answer, or correctness. Each row records hard-mask execution,
identity-materialization false, first token/route divergence, route and mass
coverage, planning calls/cost, production cache/RNG signatures, shadow discard,
and source provenance.

## Samples, gate, and execution

The actual set is exactly test-44, test-632, test-444, test-519,
test-1311, test-1264, test-825, and test-252, in that order. These are
8/8 correct in frozen v17 vanilla and contain 2,101 saved output tokens. A
16-token mechanism smoke uses test-44 but cannot change policies, IDs, gates,
or execution order.

Rows are atomic and checksum-resumable. Work is ordered row-major then by the
three frozen policies and assigned by zero-based work-index modulo two. Each GPU
has at most one worker and one model load. Failed markers are retained. No
network download is permitted.

The small-sample allowed drop is one question: each policy needs at least 7/8
correct to pass the pilot accuracy gate, while 8/8 is the strong preservation
signal. Paired deltas use 10,000 sample-level bootstrap resamples with seed
20260802. Future-exact cannot select the deployable result. No eight-row outcome
can be called full-dataset GO; the strongest possible conclusion is
PILOT_NARROW.

## Reporting boundary

Measured results include actual task accuracy, generated tokens, exact-token
agreement, first divergence, v17-token NLL/perplexity on policy context, route
coverage, probe and total generation time, and peak CUDA memory. Planned
transfer bytes and transfer reduction remain simulation. The report must not
claim production offloading speedup, full-dataset accuracy preservation, or
future-exact deployability.

## Final observed result

The frozen execution completed all 24 actual sample/policy rows. Recent-
sequence and sampled-unigram each scored 7/8; sampled-unigram versus recent was
one paired gain, one paired loss, and six ties. Their aggregate route hit and
selected routing mass were 0.730267/0.745076 and 0.757617/0.772934. Thus both
deployable policies passed the predeclared one-question allowed-drop gate, but
neither met the strong 8/8 preservation signal and sampled-unigram did not
improve paired accuracy.

Future-exact content scored 6/8 at 0.783271 route hit and 0.803592 selected mass.
It made 1,960 natural autoregressive lookahead calls and remains nondeployable.
Its higher route coverage but lower accuracy also confirms that this content
diagnostic is not a perfect routing oracle and that aggregate route coverage is
not an accuracy surrogate after closed-loop trajectory divergence.

All hard-mask, native-top-k, cache/RNG/shadow, information-boundary, atomic-row,
payload-checksum, resume, row-count, and artifact-manifest audits pass. There
are zero failure markers and zero identity-materialized rows. The terminal
decision is `PILOT_NARROW_WITH_ONE_ALLOWED_LOSS`, with conclusion ceiling
`PILOT_NARROW`. The artifact-manifest SHA-256 is
`12e9d371485b7375561f8f194fbdb6e3f1ea57fda02cddabfd12a17a408eef67`.
