# Mass-preserving pseudo residual held-out v1 protocol

This protocol is frozen before any corrected previous-route or current-candidate
held-out model output. It follows the four-row
`pseudo_mass_preserving_residual_v1` development result without using GSM8K
answers, correctness, or task accuracy.

## Why a development reference repair precedes held-out

The inherited residual-window evaluator used the last eight prompt routes for
the first `previous_route_commitment` window. The focused protocol requires the
frozen static-frequency top-32 subset for that first window. Old artifacts
remain immutable provenance, but they cannot be the terminal comparator for
this candidate.

The first stage therefore runs only the correctly defined previous-route policy
on the same four development IDs. Candidate rows are checksum-pinned and
reused. If the candidate does not retain at least +0.05 route hit and +0.05
selected mass over this corrected reference, execution stops before held-out.

## Held-out population and policies

The eight held-out IDs and SHA ranks are copied exactly from the independently
ranked `pseudo_embedding_qwen_gsm8k_residual_window_v1` held-out partition,
committed before the current candidate existed. No new row is selected. The
later composition analysis ran these IDs, but its historical policy outputs are
not reused for the gate because their first-window previous policy has the
legacy semantics.

After the corrected development gate passes, all three policies run on
identical saved v17 token trajectories with a 128-token cap:

- `hard_oracle_commitment`: copy-on-write future natural-route lookahead, then
  a real B=32 hard subset on its own teacher-forced policy state;
- `previous_route_commitment`: frozen static-frequency first window, then its
  own previous realized pre-mask routes;
- `natural_top8_intersection_zero_missing`: sampled-unigram causal H=8 pseudo
  content, full-native boundary-zero shadow residual, and later pseudo
  residuals executed only through the prior realized B=32 subset. Captured
  native IDs and weights are preserved; missing mass is zeroed.

The candidate performs one batched causal H=8 pseudo traversal per boundary.
It uses no learned or fitted parameter, future token after the already sampled
token, answer, correctness, task accuracy, offline route prior, or
default-vector value. Static-frequency count values are used only by the
previous-route reference's first window. The pinned default-vector artifact
still has 444 unobserved layer/expert pairs; its mean vectors are not loaded or
treated as a complete expert prior here.

## Frozen decisions

The held-out strong gate simultaneously requires route hit at least 0.6967,
selected mass at least 0.7162, gains of at least 0.05 over corrected previous
for both metrics, at least 25% recovery of both hard-oracle-minus-previous gaps,
simulated transfer reduction at least 0.30, and all cache/RNG/information/weight
audits.

A failure is `STOP/PIVOT`. A pass is only
`CANDIDATE_FOR_SEPARATELY_FROZEN_CLOSED_LOOP_PILOT`; it does not itself
authorize or report task accuracy. Any accuracy run must reuse the original
predeclared closed-loop IDs and frozen paired accuracy gate in a separately
committed protocol.

Route/policy-state metrics and probe/replay cost are measured. Transfer is
simulated. This analysis measures no task accuracy, free generation,
exact-token identity, NLL/perplexity, closed-loop runtime, or runtime speedup.
