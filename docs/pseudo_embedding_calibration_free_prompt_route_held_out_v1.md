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
