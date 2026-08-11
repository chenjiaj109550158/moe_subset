# pseudo_embedding_calibration_free_prompt_route_development_v1

Development decision: **CANDIDATE_FOR_NEW_FROZEN_HELD_OUT_ROUTE_PROTOCOL**. The focused v1 terminal decision remains **STOP/PIVOT**.

This is four-row open-loop route evidence with same-request native prompt route capture. It is not held-out evidence, task accuracy, hard closed-loop generation, or a speedup claim.

Frozen previous-route reference: hit 0.609385, mass 0.629428, simulated transfer reduction 0.361315.

- `prompt_or_recent_route_prior`: hit 0.632668 (+0.023283), mass 0.655139 (+0.025711), transfer 0.391356; LOO gains hit [+0.021437, +0.026331], mass [+0.023799, +0.029314].
- `sampled_known_plus_prompt_or_recent_history`: hit 0.656374 (+0.046988), mass 0.679779 (+0.050351), transfer 0.422199; LOO gains hit [+0.044443, +0.049569], mass [+0.047570, +0.052955].
- `equal_sampled_pseudo_prompt_or_recent_history`: hit 0.664737 (+0.055351), mass 0.688248 (+0.058820), transfer 0.445071; LOO gains hit [+0.048342, +0.060791], mass [+0.051392, +0.065039].
- `sampled_top8_core_plus_prompt_or_recent_history_fill`: hit 0.657161 (+0.047776), mass 0.684467 (+0.055039), transfer 0.428530; LOO gains hit [+0.045015, +0.050471], mass [+0.051853, +0.057837].

## Claim boundary

Candidates use only the sampled next token, current-request prompt routes, current-policy preceding generated routes, and zero-contribution pseudo probabilities. No learned/fitted value, offline prior, default vector, future token, answer, correctness, or accuracy is used. Transfer is simulated and prefill capture latency is capture-inclusive, not isolated hook overhead. Passing only justifies a separately frozen held-out route protocol.
