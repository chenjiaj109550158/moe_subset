# Mass-preserving previous-subset residual v1 protocol

This development analysis follows the frozen `moe_residual_dominant` component diagnosis. It
tests whether the fresh MoE residual generated inside one deployable pseudo forward can be made
more faithful without calibration, training, future tokens, full-expert future state, or a second
pseudo traversal.

The pseudo content is fixed before any result to the previously selected single-forward
`sampled_unigram_full_continuation`. The four development rows, H=8, B=32, 64-token route cap,
first-four-anchor subset formula, prompt rendering, v17 saved trajectories, and hard policy-own
teacher-forced evaluator are unchanged. The checksum-pinned prior sampled-unigram rows are the
baseline and are not rerun.

At boundary zero all three candidates execute the native full top-8. At later boundaries they may
execute only experts in the policy's previous realized layer-local B32 subset. They differ only in
how router weights produce the pseudo MoE residual:

1. `natural_top8_intersection_zero_missing` retains the native weights of natural top-8 experts
   already resident. Missing natural expert weight contributes zero.
2. `rerouted_top8_scaled_by_captured_natural_mass` performs the current normalized top-8 reroute
   inside B32, then scales the mixture by the native top-8 weight mass already captured by B32.
3. `captured_natural_plus_substitute_missing_mass` preserves resident natural experts and their
   native weights, then allocates only missing mass to the highest-logit resident substitutes.
   Equal logits use ascending layer-scoped expert ID.

All rules are analytic and use only the current pseudo router scores, native top-8 route, and
previous resident subset. No statistic is fitted from the four rows. Each candidate remains one
batched causal H=8 traversal: 48 attention/router/expert calls and 384 attention queries per full
boundary.

Selection uses development route, mass, anchor-1, simulated transfer, measured probe cost, and
integrity audits only. A candidate must not regress aggregate route hit or selected mass relative
to the pinned sampled-unigram baseline, may regress anchor one by at most 0.005, must retain at
least 30% simulated transfer reduction, stay below 0.50 seconds mean probe latency, and pass every
cache/RNG/information/weight audit. If none is eligible, the decision is `STOP/PIVOT`.

This protocol does not authorize held-out route or task accuracy. An eligible winner still needs a
separate frozen held-out protocol and the original progress gate. No result here is runtime
speedup evidence.
