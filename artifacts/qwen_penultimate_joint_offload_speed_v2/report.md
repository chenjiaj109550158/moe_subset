# qwen_penultimate_joint_offload_speed_v2 report

Decision: **NARROW_NO_SPEEDUP**. This is a two-row, one-A100 engineering measurement.

## Measured speed

- Traditional exact-top-8 B32: 0.520364 decode forwards/s (488 forwards / 937.805799 s).
- Penultimate joint H8/B32: 0.513351 decode forwards/s (456 forwards / 888.281337 s).
- Joint / traditional: 0.986523x (-1.348%).
- End-to-end generated-token rate ratio: 0.988785x.

## Why transfer savings did not become speedup

- Native post-prefill model calls including bootstrap: traditional 488; joint 458.
- Native input positions including disposable pseudo positions: traditional 488; joint 920 (464 extra pseudo positions).
- Joint planning wall time was 2.261733 s; joint-call host wall time was 33.054422 s. These are measured components inside the decode wall time, not additive costs.
- The candidate made fewer outer model calls and hid most transfer waiting, but processed substantially more token positions in its 9-position joint calls. In this native Python runtime, that compute/runtime cost outweighed the transfer gain.

## Measured transfer

- Row-total actual H2D bytes (prefill + decode): traditional 584,558,051,328; joint 427,504,435,200; reduction 26.867%.
- CUDA-event transfer seconds: traditional 23.408150; joint 17.073005; reduction 27.064%.
- CUDA-event exposed stall seconds: traditional 24.367019; joint 3.920401; reduction 83.911%.
- Joint async prefetch: 2688 batches, 330,442,997,760 bytes; deferred ready waits 25633 (0.099350 s).

## Fairness and outputs

- Both policies used exactly 32 CUDA expert slots per routed layer. Full pinned CPU expert bytes were 57,982,058,496; B32 CUDA expert-slot capacity was 14,495,514,624 bytes.
- Traditional dynamically loaded exact per-token natural top-8 into B32. Joint production was resident-only B32 and asynchronously prefetched its next B32.
- GSM8K accuracy: traditional 2/2; joint 2/2.
- Exact-token identity with each policy's frozen reference: traditional 2/2; joint 0/2 (report-only for joint).
- Joint production expert misses: 0.
- V2 smoke q_len1/q_len9 natural-route slot agreement was 81.771%; next-B32 mean overlap was 98.112%. The bridge sampled token matched exactly.

## Boundary

All runtime, H2D, transfer, stall, output, cache, and call-count values above are measured. No transfer or stall value is simulated. Full expert tensors were pinned on CPU and CUDA held exactly B=32 slots per layer. Timed joint rows did not execute the sequential smoke reference.

This Python runtime does not establish full-dataset accuracy, NVMe/NVLink or multi-GPU behavior, CUDA-graph/custom-kernel performance, or production-serving speedup.
