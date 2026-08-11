# One-forward current-context continuation v1

Each candidate copies only known current-request token context, then runs one native causal H=8 traversal with fresh native MoE residuals.

| Variant | Route hit | Selected mass | Coverage | Changed anchors | Transfer reduction | Probe s | Plan ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| recent_sequence_causal_uncorrected | 0.700267 | 0.714061 | - | - | 0.4659 | 0.3082 | - |
| longest_suffix_full_continuation | 0.726573 | 0.742827 | 0.6250 | 0.6205 | 0.4977 | 0.3119 | 1.123 |
| sampled_unigram_full_continuation | 0.730357 | 0.747085 | 0.6250 | 0.6205 | 0.5022 | 0.3055 | 0.079 |
| longest_suffix_partial_recent_fill | 0.726573 | 0.742827 | 0.6250 | 0.6205 | 0.4977 | 0.3149 | 1.132 |

## Comparisons to checksum-pinned uncorrected v1

- longest_suffix_full_continuation: hit/mass delta +0.026306/+0.028766; anchor-1 0.928223/0.951405; mean matched suffix 2.000; paired hit CI [+0.018412, +0.034200], mass CI [+0.019853, +0.037679]; signal True.
- sampled_unigram_full_continuation: hit/mass delta +0.030090/+0.033024; anchor-1 0.927734/0.950916; mean matched suffix 1.000; paired hit CI [+0.025065, +0.034587], mass CI [+0.026498, +0.038776]; signal True.
- longest_suffix_partial_recent_fill: hit/mass delta +0.026306/+0.028766; anchor-1 0.928223/0.951405; mean matched suffix 2.000; paired hit CI [+0.018412, +0.034200], mass CI [+0.019853, +0.037679]; signal True.

## Suffix-length strata and worst samples

- longest_suffix_full_continuation: fallback: 12 boundaries at 0.700439/0.712590, suffix_length_1: 10 boundaries at 0.719010/0.736488, suffix_length_2_plus: 10 boundaries at 0.765495/0.785451. Lowest paired sample test-439 at +0.020874/+0.019807.
- sampled_unigram_full_continuation: fallback: 12 boundaries at 0.700901/0.712867, suffix_length_1: 20 boundaries at 0.748031/0.767616. Lowest paired sample test-439 at +0.022746/+0.023784.
- longest_suffix_partial_recent_fill: fallback: 12 boundaries at 0.700439/0.712590, suffix_length_1: 10 boundaries at 0.719010/0.736488, suffix_length_2_plus: 10 boundaries at 0.765495/0.785451. Lowest paired sample test-439 at +0.020874/+0.019807.

Development-only decision: **DEVELOPMENT_ROUTE_SIGNAL**. Selected route signal: sampled_unigram_full_continuation.

Route values are teacher-forced on each policy's own hard-subset state. Content planning and probe cost are measured; transfer is simulated. No held-out route, task accuracy, free generation, exact-token identity, closed-loop runtime, or runtime speedup was measured.
