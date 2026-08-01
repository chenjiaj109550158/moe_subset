# Calibration-free prompt-route mechanism smoke v1

## Frozen question

The content smoke showed that recent token embeddings do not improve repeated
sampled-token pseudo routing by themselves, while adding the immediately
preceding realized route window is strongly helpful. Its remaining structural
weakness is boundary zero: no generated route window exists, so the pseudo probe
must carry the entire first commitment.

This amendment tests a deployable calibration-free replacement. During the
same request's native Qwen prefill, it captures the last eight prompt tokens'
natural routed-expert IDs and normalized selected weights. Boundary zero uses
that route distribution; boundary eight uses the current policy's already
realized generated routes. It never borrows routes from another sample, request,
vanilla future, answer, or offline calibration artifact.

The exact scope is frozen before model execution in
`configs/analysis/pseudo_embedding_calibration_free_prompt_route_smoke_v1.yaml`.
It reuses only `test-786` and `test-394`, the existing 16-token content tensors,
and the pinned local Qwen checkpoint. It adds no data, held-out evaluation, task
accuracy, or closed-loop hard generation.

## Frozen candidates

All history distributions are normalized selected routing mass over exactly
eight current-request tokens. All constants follow analytically from `H=8`,
native top-k 8, and `B=32`; none is fitted.

1. `prompt_or_recent_route_prior` selects top-32 by history alone.
2. `sampled_known_plus_prompt_or_recent_history` uses sampled anchor 1 once and
   history seven times.
3. `equal_sampled_pseudo_prompt_or_recent_history` gives equal evidence to each
   unknown sampled-repeat pseudo anchor and history.
4. `sampled_top8_core_plus_prompt_or_recent_history_fill` reserves eight slots
   for sampled anchor 1 and fills 24 from history.
5. `equal_recent_pseudo_prompt_or_recent_history` applies the same equal-evidence
   formula to the recent-content independent pseudo tensor.

The saved sampled-repeat control is the progress reference. Ranking uses route
and simulated transfer metrics only. A candidate must gain at least 0.05 in both
route hit and selected mass and retain at least 0.30 simulated transfer
reduction to justify another frozen development protocol. Two rows can only
support a mechanism hypothesis; the terminal focused-v1 decision remains
`STOP/PIVOT`.

## Required audits

Prompt-route capture runs inside native Qwen prefill and records only the last
eight prompt route rows. It must preserve the returned production cache, RNG,
layer-scoped expert IDs, native routing normalization, and exact prompt/sample
identity. Prompt-route sample tensors and reports are written atomically with
checksums and resume validation. Measured capture/model runtime is reported;
transfer remains simulated and no speedup is claimed.

## Completed result

The repeated-zero control scored 0.524495 route hit and 0.534683 selected mass.
Prompt/recent history alone reached 0.685628/0.716782. Adding the known sampled
anchor reached 0.702393/0.733202, while reserving an anchor-1 top-8 core reached
0.700277/0.733516; their route/mass tradeoff was too close to call robustly on
two rows. All calibration-free prompt candidates passed the smoke progress
references and audits, authorizing only the separately frozen four-row
development comparison.
