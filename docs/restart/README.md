# Restart-v1 execution contract

This new protocol does not modify historical frozen configs or artifacts.

The shared measurement implementation uses the pinned Qwen BF16 checkpoint, one visible CUDA
GPU, CPU-first safetensors loading, 32 expert slots per layer, and the official vLLM 0.11.0
unquantized fused-experts API. No selected expert matrices are materialized per token.
`python_reference` changes only expert computation in the updated common harness; the runtime
four-grid comparison is therefore an expert-compute ablation, not a replay of the entire old
instrumented harness. Both exact and subset use the optimized backend in all main comparisons.

A prefill sample is output token 1 and is not a production forward. S_0 applies to forwards 1–8,
S_1 to 9–16, etc. At a pseudo boundary the known input to forward 8 is the real bridge. The joint
input is [bridge, legacy-H8-content[:G]]. The bridge uses S_0; pseudo positions use natural top-k
intersection with S_0 and retain original weights. Only the bridge KV position is committed.
The next subset is transferred after that layer's old slot readers finish. No future sampled
output is available to the selector. A final sampled token is not forwarded again. Bootstrap
and unused final-boundary transfers are paid and drained inside decode wall.

G is independently 0/4/8, content horizon remains 8, core anchors remain 4, and d remains 8.
Legacy content with fewer than seven preceding known tokens is explicitly rejected; H/S support
short prompts. Native EOS and stop behavior include the entire prompt plus continuation.

H uses GPU FP32 natural-routing EMA alpha 0.2. HC uses d*normalized_utility minus beta 0.05
for newly loaded experts weighted by their actual uniform BF16 expert bytes. Exact uses natural
top-k and deterministic demand LRU; it does not compute unused predictor statistics. EP is not
implemented, so baseline scope is exact_demand_LRU_only. PW4/PM4 remain conditional branches,
never described as measured if the development gate skips them.

The first freeze precedes all new candidate outputs. The second freeze selects one window
candidate from development only, locks N in {256,128,64}, and uses 3 quality claims (4 with a
pseudo/history comparator) for Bonferroni Clopper–Pearson intervals. A small or incomplete
sample cannot establish GO. The N=256 cap may still leave non-inferiority inconclusive.

`run_all` executes real rows, retains checksums and failed logs, and stops at COMPLETE, BLOCKED,
BUDGET_EXHAUSTED or INTERRUPTED. `verify` loads no model. An identity change cannot reuse old
rows. The agent, not the runner, diagnoses and repairs code failures within the bounded cycles.
