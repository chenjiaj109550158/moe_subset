# One-forward protected-anchor correction v2

This development-only experiment is frozen before new model output. It tests
whether the negative v1 velocity result came from corrupting the strongest
known-token anchor or from linear state extrapolation itself. It reuses exactly
the four existing Qwen/GSM8K rows and authorizes no new trace, held-out route,
or task-accuracy execution.

Every candidate retains the same calibration-free mechanism: one native causal
eight-token pseudo traversal per boundary, correct future RoPE positions, and a
fresh native MoE residual. Boundary zero may execute full native top-8 experts;
later boundaries execute only the current policy's previous realized per-layer
B=32 subset. Production cache and RNG are read-only and the shadow cache is
discarded.

Anchor one contains the already sampled next token. Its correction coefficient
is therefore fixed to zero. The horizon-damped schedule is structurally fixed
to `(anchor - 1) / 8`; the undamped protection control uses `anchor - 1`. A
separate cap limits the added correction vector to 25% of the fresh state norm,
matching the pinned resident fraction rather than a fitted value. Per-anchor L2
norm is restored after correction.

Five candidates isolate hidden versus residual correction, anchor-one
protection, horizon damping, the norm cap, and subset construction. The final
selection variant reserves the uncorrected anchor-one pseudo top-8 and fills the
remaining 24 experts by the corrected probability utility from anchors two
through eight. Ties use ascending layer-scoped expert ID.

No candidate reads future true tokens after the sampled boundary token, vanilla
future trajectories, answers, correctness, accuracy, learned/fitted values,
route tables, expert priors, or default-vector values. Ranking uses route and
measured probe cost only. A development route signal requires both route hit and
selected mass to improve by at least 0.02 over the checksum-pinned v1
uncorrected baseline, anchor-one hit and mass to regress by no more than 0.005,
probe latency at most 0.50 seconds, and every cache/RNG/information/call-count
audit to pass. Even a route signal does not authorize held-out or accuracy under
this protocol.
