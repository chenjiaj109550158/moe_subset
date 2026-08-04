# Qwen penultimate-token joint offload speed pilot v2

## Why v2 exists

The frozen v1 smoke executed the true joint/offload path and passed cache, RNG,
H2D, asynchronous-prefetch, B=32 residency, and sampled-token checks, but its
predeclared bitwise route-ID gate failed. A q_len=1 sequential forward and the
q_len=9 joint BF16 forward can select different internal GEMM/attention kernels;
their router IDs are therefore not a valid bitwise implementation invariant even
when causal semantics and the sampled bridge token agree. v1 remains preserved
as failed provenance. v2 changes only that validation semantics: the same method,
IDs, caps, work order, model, precision, and task gates remain frozen.

## Question and immutable scope

This engineering pilot asks whether the already-tested penultimate-token
information regime becomes faster when it is implemented as a real joint
forward with real expert offloading. It reuses exactly `test-44` and
`test-632`, the two IDs previously frozen for the real-offload speed pilot,
with their original ordering and 512-token caps. No replacement ID, larger
dataset, new model, new precision, download, or predictor training is allowed.

The model is Qwen3-30B-A3B revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` in BF16 on one A100. Every routed
layer has 128 experts, native top-8 routing, and exactly 32 CUDA expert slots.
All full routed-expert tensors live in pinned CPU memory after setup.

## Actual joint execution

The first H=8 window retains the previously validated online-post-sample
sampled-unigram bootstrap. Later windows replace the eighth one-token
production call plus separate eight-anchor pseudo call with one causal native
model call whose inputs are:

1. the real bridge token consumed by the eighth production step; and
2. eight sampled-unigram pseudo-anchor tokens built only from known context.

Each layer performs one attention call, one router call, and one expert call
over these nine positions. The bridge position is hard-routed inside the current
layer-local B=32 subset. Pseudo positions preserve their natural top-8 weights
for experts in that same subset and assign zero contribution to missing
experts. The first-four pseudo-anchor core plus all eight realized natural
routes of the completed window deterministically select the next B=32.

The joint call uses a copy-on-write cache branch. After all layers finish, only
the bridge KV position is committed to the production cache; all eight pseudo
KV positions are discarded. Therefore one decode step advances the production
cache by exactly one position. The token newly sampled from the bridge logits
is not available to the planner and does not affect the next subset.

After each layer's joint expert computation, that layer's next subset is known.
The runtime schedules its missing experts on the dedicated transfer stream
without making the current compute stream wait. Evicted slots are protected by
their compute-complete events. A later production use waits on any still-pending
slot-ready event, and that wait is recorded as exposed stall. This is actual
CPU-to-CUDA transfer/compute overlap, not trace simulation.

## Correctness smoke

Before timed rows, a 17-token smoke on `test-44` must execute at least two
joint boundaries. The first joint boundary also runs an untimed sequential
bridge-plus-pseudo reference on disposable state. The bridge sampled token must match exactly. Bridge natural/executed
route-ID agreement and next-B32 overlap are measured layer by layer, but are
report-only because q_len=1 and q_len=9 BF16 batching is not bitwise invariant.
The production cache must advance by one position, pseudo KV
must not be committed, generation RNG must remain valid, all production experts
must already be resident, and asynchronous prefetch must be both scheduled and
consumed. This duplicate reference is forbidden in timed rows.

## Paired speed execution and gates

Timed order is fixed AB/BA:

1. test-44 traditional, then joint;
2. test-632 joint, then traditional.

Traditional execution routes all 128 experts every token, executes exact native
top-8, and synchronously loads misses into the same B=32 cache. The primary
metric is aggregate post-prefill decode forwards per second, including policy
planning, actual H2D transfers, and exposed waits. Setup extraction is measured
separately and excluded.

Traditional rows must retain frozen-vanilla token identity 2/2. The candidate
must answer 2/2 and have zero production expert misses. Exact identity with the
no-offload penultimate rows is reported but is not a gate because the prior
offload experiment already established BF16 trajectory sensitivity. With
correctness and invariants intact, a ratio below 1.00 is
`NARROW_NO_SPEEDUP`, 1.00–1.05 is a positive engineering `NARROW`, and at
least 1.05 is a strong engineering `NARROW`. Any smoke invariant or candidate
accuracy failure is `STOP/PIVOT`.

## Reporting boundary

Wall time, task outputs, exact tokens, cache positions, real H2D bytes, transfer
events, exposed stall, phase counts, and peak CUDA memory are measured. Prior
unoverlapped offload rows are external context only. No transfer byte or stall
metric in this pilot is simulated. The result remains a two-row, one-A100,
Python-runtime engineering measurement; it cannot establish full-dataset
accuracy, NVMe/NVLink behavior, multi-GPU behavior, a CUDA-graph/custom-kernel
speedup, or production-serving speedup.
