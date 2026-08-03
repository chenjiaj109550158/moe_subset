# Qwen real expert-offload speed pilot v1

## Question and immutable scope

This two-row engineering pilot asks whether the calibration-free
`natural_top8_intersection_zero_missing` schedule improves measured decode
throughput over a traditional lossless dynamic expert cache when both receive
the same 32 CUDA expert slots per layer. It uses Qwen/GSM8K rows `test-44` and
`test-632`, the first two rows of the already predeclared wave-one manifest.
No accuracy, route, runtime, or method result selected these IDs.

The model revision, bfloat16 precision, exact saved v17 prompts, greedy decode,
stop strings, parser, and original 512-token cap are fixed. Network access,
new datasets, replacement samples, larger caps, and additional rows are not
authorized. The conclusion is capped at `TWO_ROW_ENGINEERING_PILOT`.

## What counts as real offloading

After offline model loading, all 48 full expert tensors are copied into a
bfloat16 CPU store and must be pinned. The original full expert parameters are
removed from CUDA. Each layer then owns exactly 32 fixed-shape CUDA expert
slots. Every reported H2D byte must arise from an actual CPU-to-CUDA
`copy_`; transfer and exposed-wait intervals use CUDA events. Model loading,
CPU extraction, pinning, and slot allocation are measured as setup but excluded
from inference throughput.

Both policies start every sample from an empty logical expert cache and perform
their own exact lossless prompt prefill. Cache replacement is deterministic LRU
with ascending expert ID as the tie break. Copies use a dedicated CUDA stream;
the current stream waits for completion. This v1 does not overlap speculative
expert transfer with useful production compute and must not claim that it does.

## Paired policies

`lossless_dynamic_top8_lru_b32` computes the native all-128 router at every
layer and token, loads any missing natural top-8 experts, and executes the exact
native route with no mask. The 32 slots are a cache, not a routing constraint.
Its generated token IDs must exactly match the checksum-pinned v17 vanilla
reference.

`natural_top8_intersection_zero_missing_h8_b32` performs the same lossless
prefill, then uses the deployable online-post-sample pseudo policy already fixed
by the route and accuracy pilots. Every eight decoded tokens it runs one causal
batched H=8 shadow traversal using sampled-token unigram continuation with
recent-sequence fallback. Boundary zero may dynamically load natural shadow
top-8 experts. Later shadow MoE residuals execute only the natural top-8
intersection with the current B=32 subset, retain native weights, make missing
mass zero, and use no substitutes. It then loads the changed experts for the
next layer-local top-32 commitment. Production logits outside that subset are
masked before native top-8 selection, and a production expert miss is an error.
Its output must exactly match the existing all-weights-resident mask-only row.

The work order is AB on `test-44` and BA on `test-632` to reduce a fixed policy
order bias. There is one repetition because each row is an actual full GSM8K
generation. Rows are atomic and checksum-resumable; failures are retained.

## Timing and interpretation

The primary speed metric is post-prefill decode forwards divided by decode wall
time. The first generated token belongs to prefill and is not in that numerator.
Decode wall time includes planning, actual H2D copies, exposed waits, and production
forward work, and stop checking. The secondary metric includes prompt prefill.
Setup is reported separately. H2D bytes, cache hits/misses, CUDA-event transfer
time, exposed stall, planning time, peak memory, output identity, and parsed
GSM8K correctness are also measured.

This experiment compares one unoverlapped pinned-CPU implementation on one
A100-SXM4-80GB and its observed PCIe link. It cannot establish full-dataset
accuracy, a production-ready serving speedup, NVMe behavior, multi-GPU/NVLink
behavior, or the benefit of transfer/compute overlap.

## Frozen fingerprints

The amended frozen config SHA-256 is
`9276363796e711baf487415a85fde49fc525ae42e7be701972867fb8713ed1ec`.
The amended frozen sample-manifest SHA-256 is
`aafd570fe56d878074cc6f5666dd26df8da528086776400226c15727b77ee819`.
Both are tested and committed before any model smoke or measured offload row.

## Final result

The completed two-row pilot measured 0.509055 post-prefill forwards/s for
traditional exact-top-8 offload and 0.503535 forwards/s for the H=8/B=32
candidate, a 0.989157x ratio (-1.084%). Actual H2D bytes decreased 28.770% and
CUDA-event transfer time decreased 31.386%; candidate production misses were
zero. Traditional generation matched frozen vanilla exactly 2/2. Candidate
answers were correct 2/2, but mask-only exact identity failed 0/2, at first
divergence tokens 4 and 2.

The focused decision is
`STOP_PIVOT_CANDIDATE_IDENTITY_GATE_FAILURE`. The result is true closed-loop
generation with physical expert copies, not identity materialization or trace
simulation. The measured speed result is specific to this unoverlapped,
instrumented Python runner and two rows.
