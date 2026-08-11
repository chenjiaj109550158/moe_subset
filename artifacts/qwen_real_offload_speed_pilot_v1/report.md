# Qwen real expert-offload speed pilot

Decision: **STOP_PIVOT_CANDIDATE_IDENTITY_GATE_FAILURE**. This is a two-row engineering result only.

## Measured speed and transfer

- Traditional exact-top-8: 0.509055 post-prefill forwards/s.
- H=8/B=32 pseudo subset: 0.503535 post-prefill forwards/s.
- Candidate / traditional: 0.989157x (-1.084%). No speedup was measured.
- Actual H2D: 584,558,051,328 vs 416,377,995,264 bytes; candidate reduction 28.770%.
- CUDA-event transfer time reduction: 31.386%.
- Candidate production expert misses: 0.

## Closed-loop outputs

- GSM8K accuracy: traditional 2/2; candidate 2/2. This is measured task accuracy.
- Traditional frozen-vanilla exact identity: True (2/2).
- Candidate frozen mask-only exact identity: False (0/2); first divergences were test-44: token 4, test-632: token 2.
- Actual closed-loop rows: 4; identity-materialized rows: 0.

## Physical and information boundary

- Full routed-expert weights were held in pinned CPU BF16 storage; CUDA held 32 expert slots per layer (25%). Every reported transfer was an actual CPU-to-CUDA copy. The candidate used its own closed-loop context and had zero production misses.
- Runtime, outputs, H2D bytes, and CUDA-event transfer/stall are measured. Legacy route/transfer estimates stored in candidate rows are simulated diagnostics only.
- This unoverlapped Python reference runner includes instrumentation overhead. It does not measure NVMe, NVLink, multi-GPU, transfer/compute overlap, a fused serving kernel, or full-dataset accuracy, and it cannot support a production speedup claim.
