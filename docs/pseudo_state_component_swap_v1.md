# Pseudo-state component swap v1

## Purpose

This is a route/state diagnostic on the four already frozen Qwen/GSM8K
development rows. It asks whether the large gap between a single pseudo
traversal and the true `H=8,B=32` routing oracle comes primarily from attention
state, MoE residual propagation, or accumulated cross-layer hidden-state drift.
It adds no sample IDs and authorizes no task-accuracy or held-out execution.

All outputs are diagnostic oracles. They cannot be described as deployable
pseudo embeddings or directly promoted to an accuracy policy.

## Common execution

Each policy replays the first 64 frozen v17 output tokens on its own hard-subset
state. At every boundary, the pseudo content is the exact next eight saved token
IDs and uses correct future RoPE positions in one native causal traversal.
Boundary zero has full expert access; later pseudo residuals execute the
policy's previous realized per-layer B=32 subset. Candidate subsets retain the
existing first-four-anchor core plus history-fill selector, and actual replay
masks outside-subset logits before native top-k and normalization.

Before the disposable pseudo traversal, one copy-on-write full-expert causal
teacher-forced forward captures the aligned natural attention output, normalized
router input, MoE mixture output, decoder-layer output, and router scores for up
to eight future tokens. Production cache and RNG must remain unchanged and all
shadow caches are discarded.

## Frozen swaps

The unpatched exact-future traversal is compared with attention-output-only,
MoE-residual-only, and joint attention-plus-MoE replacement at every layer.
Five additional variants replace the complete pseudo hidden state once, after
zero-based layer 7, 15, 23, 31, or 39. These checkpoints measure how much
downstream router alignment is recovered when accumulated drift is removed at
different depths.

Every comparison reports route hit, selected mass, centered router-logit cosine,
subset Jaccard, and cosine alignment for attention, router input, MoE residual,
and layer output by sample, boundary, anchor, layer, eight-layer block, and
router-margin quartile. Capture/probe latency, temporary CUDA memory, and exact
native call counts are measured. Transfer remains simulated.

## Attribution and next-method rule

The joint attention-plus-MoE swap is the fixed recoverable-gap reference. A
single component is dominant only if it recovers both at least half of the joint
route-hit and selected-mass gains and leads the other component's mean recovery
by at least 0.10. Otherwise a positive joint gain of at least 0.02 on both
metrics is classified as coupled. A non-positive or sub-threshold joint result
is unresolved. The latest hidden checkpoint recovering at least half of both
joint gains is reported.

The next calibration-free family is predeclared from that classification:

- attention dominant: batched recent/unigram/current-token multiview;
- MoE residual dominant: mass-preserving previous-subset residual;
- coupled: mass-preserving batched multiview;
- unresolved: stop without executing a candidate.

Any chosen family requires a new committed protocol before its first model
output. Component results may select only the family through this rule; task
accuracy, correctness, and post-hoc coefficient tuning are forbidden.
