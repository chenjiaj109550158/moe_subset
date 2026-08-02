# One-forward state correction v1

This analysis tests two calibration-free corrections to a single causal
eight-token pseudo forward at the pinned Qwen/GSM8K `H=8,B=32` point. It is
frozen before new model results and reuses only the four existing development
rows. It does not authorize held-out rows or task accuracy.

All variants use the deployable recent-content sequence: the already sampled
next token followed by the seven most recent prompt or current-policy realized
tokens. The eight embeddings are processed together at the correct future RoPE
positions. Boundary zero may execute full native top-8 experts; later boundaries
execute only the current policy's preceding realized layer-local B=32 subset.
There is one native causal attention call per layer and no autoregressive LM-head
token construction.

The hidden-velocity variant stores the last two native MLP router inputs from the
current policy. At each layer it adds anchors 1 through 8 times their newest-minus-
previous difference to the uncorrected pseudo router input, then restores each
anchor's original L2 norm before the exact native gate and expert execution.

The residual-velocity variant stores the last two exact native MoE mixture
outputs. It first executes the pseudo experts under the allowed subset, adds the
same fixed anchor-scaled historical difference to that fresh contribution, and
restores the fresh contribution's L2 norm before adding it to the residual
stream. Historical residuals correct rather than replace fresh pseudo residuals.

Both formulas use only prompt or already realized current-policy state. They use
no future true token after the sampled-next boundary token, vanilla trajectory,
answer, correctness, task accuracy, fitted coefficient, route table, offline
expert prior, or default-vector value. Production cache and RNG remain read-only;
the shadow cache is discarded after each boundary.

Ranking uses route and measured cost only. A method is called a route signal only
if it improves both mean route hit and selected mass by at least 0.02 over the
new-run uncorrected baseline while keeping mean probe latency at or below 0.50
seconds. Transfer remains simulated and no runtime-speedup claim is permitted.
