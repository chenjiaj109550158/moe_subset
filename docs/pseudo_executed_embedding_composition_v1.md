# Pseudo-executed embedding composition v1

This protocol freezes the next calibration-free Qwen/GSM8K analysis before new
model results. Every pseudo layer computes a fresh native MoE contribution on
the pseudo hidden state. Boundary zero uses full native top-8 access; later
boundaries can execute only the policy's preceding realized layer-local B=32
subset. The residual mechanism, H=8/B=32 operating point, native
attention/RoPE/router semantics, and first-four-anchor-core-plus-history-fill
selector remain fixed while pseudo content changes.

Development reuses the four frozen rows test-0, test-439, test-879, and
test-1318 with 64 route tokens. It compares sampled/current/recent content,
independent versus causal anchors, raw and norm-matched expected top-M
embeddings, sampled-hard plus soft future anchors, and deterministic
autoregressive greedy/expected shadow trajectories. Exact-future variants are
information-oracle diagnostics and cannot be selected. The selected deployable
variant is evaluated on the eight previously frozen, unexecuted held-out route
rows with 128 route tokens. Fixed top-2/top-4 deterministic particle variants
are a separately reported follow-up.

No candidate reads labels, answers, correctness, a vanilla future trajectory,
future true tokens, default vectors, learned parameters, fitted coefficients,
or offline expert priors. Production cache identity, storage pointers, version
counters, sequence length, hidden state, and RNG must remain unchanged. All
sample-policy rows use atomic JSON/safetensors pairs and checksum resume.

Ranking uses development route and cost metrics only: selected mass, route hit,
simulated transfer reduction, probe latency, then key. Held-out evaluation does
not authorize task-accuracy generation. Route hit and selected mass are
teacher-forced measurements on each policy's own hard-subset state; transfer is
simulated; probe/replay time and temporary CUDA memory are measured.
