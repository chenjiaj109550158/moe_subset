# One-forward token-aligned state retrieval v1

This calibration-free development experiment is frozen before implementation
and new model output. It reuses exactly four existing Qwen/GSM8K rows and
authorizes no new traces, held-out route, or task accuracy.

The hypothesis is that semantic state alignment is more useful than temporal
velocity. At every boundary the current policy has a bank of native router
inputs and exact executed MoE outputs for the rendered prompt and already
realized tokens. Each pseudo anchor first retrieves the most recent bank entry
with the same token ID. An optional fallback selects the history token with the
highest float32 cosine similarity under Qwen's native input embedding, breaking
ties by most recent position. Every retrieved index must precede the boundary.

The later seven recent-sequence anchors are themselves already-realized tokens,
so exact-token retrieval should exist for them. The embedding-nearest fallback
primarily tests the newly sampled first anchor when that token has not appeared
in the request. Similarity is one for exact matches, zero for a missing
exact-only match, and clamped to `[0,1]` for embedding fallback.

The fresh pseudo path remains unchanged: one native causal eight-token
traversal, correct future RoPE positions, full native top-8 expert access at
boundary zero, and the current policy's previous realized per-layer B=32 subset
thereafter. Retrieval either mixes into the fresh native MoE output or the
native router input. Additive mixing uses `fresh + similarity * retrieved`;
the replacement control uses `(1-similarity) * fresh + similarity * retrieved`.
Each result is restored to the fresh state's per-anchor L2 norm.

All bank states come from the same current request and policy. No future true
token after the sampled boundary token, vanilla trajectory, answer, correctness,
accuracy, learned/fitted value, route table, expert prior, or default-vector
value is accessible. A route signal requires +0.02 route hit and selected mass
over the checksum-pinned uncorrected reference, no more than 0.005 anchor-one
regression, mean retrieval-plus-probe latency at most 0.50 seconds, and all
cache/RNG/information/call-count audits. This protocol never authorizes
held-out or accuracy execution.
