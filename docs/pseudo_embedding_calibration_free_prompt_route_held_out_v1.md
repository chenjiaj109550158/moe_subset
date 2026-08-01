# Calibration-free prompt-route held-out route protocol v1

## Frozen held-out question

Four-row development selected exactly one calibration-free candidate:
`equal_sampled_pseudo_prompt_or_recent_history`. This protocol evaluates it on
the eight disjoint held-out route rows already fixed by the original focused
pilot's deterministic SHA-256 sample manifest. It does not change IDs, inspect
accuracy, or select another formula.

For each row, native Qwen prefill supplies the current request's last eight
prompt routes. Teacher-forced replay of at most 128 saved v17 generated tokens
supplies natural scoring routes and current-policy preceding-window history.
The pseudo probe uses the known sampled next token, native attention/RoPE/router,
zero expert contribution, and a read-only/copy-on-write production cache. No
future true token enters the candidate.

The original hard oracle, previous-route, static-frequency, and repeated-zero
pseudo methods are references. Static frequency reads only the frozen count
tensor; the default-vector mean tensor is forbidden. References cannot enter the
candidate utility.

## Frozen strong-candidate gate

The candidate must simultaneously:

- recover at least 25% of both oracle-minus-previous route-hit and selected-mass
  gaps;
- reach mean route hit 0.6967 and selected mass 0.7162;
- improve over previous route by at least 0.05 in both metrics;
- retain at least 0.30 simulated transfer reduction;
- pass cache, RNG, information-boundary, row-count, checksum, and resume audits.

Paired sample bootstrap uses 10,000 draws, seed 20260801, and percentile 95%
intervals. Failure is `STOP/PIVOT`. Passing only permits a separately frozen
closed-loop pilot; it is not task accuracy, actual hard generation, or a speedup
claim. Exact rows and all constants are fixed in
`configs/analysis/pseudo_embedding_calibration_free_prompt_route_held_out_v1.yaml`.

## Completed result

All eight atomic rows completed and the checksum validator accepted 31
artifacts. The selected candidate achieved route hit 0.658353, selected mass
0.680584, and simulated transfer reduction 0.439423. Previous route achieved
0.617671/0.637396, yielding +0.040682/+0.043188; paired 95% sample-bootstrap
intervals were [0.036001, 0.046422] and [0.038634, 0.049149]. Oracle-gap recovery
was only 0.116775/0.125752. Thus both absolute metrics, both +0.05 improvements,
and both 25% gap-recovery checks failed; transfer and cache/RNG/information
checks passed.

The frozen decision is `STOP/PIVOT`. No task accuracy or actual hard closed-loop
generation was run. See `pseudo_embedding_calibration_free_synthesis_v1.md` for
the router-sensitivity analysis and explicitly post-hoc next hypotheses.
