# Pseudo-executed previous-subset smoke v1

## Question

Can a calibration-free shadow rollout compute a fresh MoE residual from each
pseudo hidden state by executing the previous realized window's layer-local
expert subset, while retaining full pre-mask router scores to plan a different
subset for the next real eight-token window?

This is a two-row, 16-token mechanism smoke only. It does not reopen the
terminal residual-window v1 decision, select a production method, evaluate
held-out rows, or measure task accuracy.

## Exact mechanism

At each pseudo layer, native Qwen attention and future RoPE first produce the
pseudo post-attention state. The exact native router then produces logits over
all 128 routed experts; those pre-mask values are saved for planning. At boundary
zero, all experts are accessible from full-expert prefill, so native top-8 expert
execution produces the pseudo MoE residual. At later boundaries, logits outside
the subset used by the preceding realized window are masked, native top-8 and
normalization are applied inside that subset, and those experts execute on the
current pseudo router input. The newly computed mixture output is added to the
pseudo residual path before the next layer.

The next real-window top-32 subset is selected only after the shadow rollout.
It uses the already frozen first-four-anchor-core-plus-history-fill rule. Because
selection reads the saved full pre-mask scores—not the execution-masked scores—
the next subset can contain experts outside the previous subset.

## Fixed comparison

Both pseudo policies use sampled-next-token repeated content, independent
anchors, native attention/RoPE/router semantics, the same analytic subset rule,
the same two rows, and the same 16-token cap:

- `pseudo_executed_previous_subset` computes a fresh residual by native expert
  execution on the pseudo hidden state;
- `provided_previous_residual_control` injects the previous real window's
  position-aligned executed residual, matching the completed v1 mechanism;
- `previous_route_commitment_reference` supplies a route-only reference.

## Feasibility boundary

Feasible means the native expert path runs, residuals are finite/nonzero and
newly computed, later-boundary executed expert IDs remain inside the prior
subset, full 128-expert pre-mask scores are retained, the next subset can escape,
and cache/RNG/shadow/information audits pass. Route hit, selected mass, probe
latency, temporary CUDA memory, expert calls, router calls, and attention calls
are reported descriptively; this smoke has no improvement gate.

Saved v17 tokens are teacher-forced while every policy propagates its own hard-
subset state. Therefore these are mechanism and route metrics, not actual
closed-loop generation, task accuracy, exact-token identity, or speedup.
