# One-forward mid-layer shifted self-conditioning v1

This calibration-free development protocol is frozen before implementation and
new model output. It reuses exactly four existing Qwen/GSM8K rows and authorizes
no new traces, held-out route evaluation, or task accuracy.

Every boundary begins with the deployable recent-sequence causal anchors: the
known sampled token followed by the last seven prompt or already-realized token
IDs. One disposable native Qwen traversal executes layers 0 through 47 exactly
once. Boundary zero has full natural top-8 expert access; every later boundary
executes the current hard policy's preceding realized layer-local B=32 subset,
so every MoE contribution is a fresh native pseudo residual.

After zero-based layer 23, Qwen's existing final norm and LM head project the
seven source-anchor mid-layer states. Source anchor `a-1` predicts content for
target anchor `a`, for targets two through eight. Greedy content uses native
argmax. Expected-top-8 content uses float32 softmax over the native top eight
logits and the native input embeddings. The target hidden receives predicted
embedding minus its initial anchor embedding and is restored to its pre-refresh
L2 norm. Anchor one is bitwise protected. The capped ablation limits this raw
embedding delta to 25% of the target mid-hidden norm before norm restoration.

This is one model-layer traversal, not eight sequential pseudo decode forwards.
It adds one seven-query vocabulary projection per boundary. Layers 0–23 retain
the unrefreshed baseline route; layers 24–47 see the refreshed anchors. Correct
future RoPE positions and native causal attention/router/MoE semantics remain
unchanged.

No future true token after the sampled boundary token, vanilla trajectory,
answer, correctness, accuracy, fitted value, offline prior, route table, default
vector, or retrieved history state is accessible. Ranking uses route and cost
only. A route signal requires +0.02 route hit and selected mass over the
checksum-pinned uncorrected reference, at most 0.005 anchor-one regression, at
least 30% simulated transfer reduction, at most 0.50 seconds mean probe latency,
and all cache/RNG/information/call-count audits. This protocol never authorizes
held-out or accuracy execution.
