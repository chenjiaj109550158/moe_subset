# pseudo_embedding_calibration_free_prompt_route_held_out_v1

Held-out route decision: **STOP/PIVOT**. The focused v1 terminal decision remains **STOP/PIVOT**.

This is eight-row held-out, teacher-forced open-loop route evidence. It is not task accuracy, actual hard closed-loop generation, or a runtime speedup claim.

- `hard_oracle_commitment`: hit 0.966054, mass 0.980837, simulated transfer reduction 0.729212.
- `previous_route_commitment`: hit 0.617671, mass 0.637396, simulated transfer reduction 0.373426.
- `static_frequency`: hit 0.337657, mass 0.345422, simulated transfer reduction 0.306407.
- `sampled_repeat_independent_zero`: hit 0.551692, mass 0.562205, simulated transfer reduction 0.336484.
- `equal_sampled_pseudo_prompt_or_recent_history`: hit 0.658353, mass 0.680584, simulated transfer reduction 0.439423.

## Strong-candidate checks

- Oracle-gap recovery: hit 0.116775, mass 0.125752.
- Paired sample-bootstrap route-gain 95% interval: [0.036000569661458384, 0.046422322591145856].
- Paired sample-bootstrap mass-gain 95% interval: [0.03863407008725561, 0.049148716193464906].

## Claim boundary

The candidate uses no learned/fitted value, offline prior, default-vector value, future token, answer, correctness, or accuracy. Static-frequency counts are reference-only and the default mean tensor is never accessed. Transfer is simulated; probe and total replay runtime are measured. Gate failure stops before task accuracy or closed-loop execution.
