# One-forward Qwen/GSM8K eight-row accuracy pilot v1

All values below use actual hard closed-loop generation on the same eight frozen rows.
Frozen vanilla is reused rather than regenerated.

| Policy | Correct | Accuracy | Token agreement | Route hit | Selected mass | Runtime s | Probe s | Oracle s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recent_sequence_causal | 7/8 | 0.8750 | 0.042852 | 0.730267 | 0.745076 | 5388.05 | 66.66 | 0.00 |
| sampled_unigram_full_continuation | 7/8 | 0.8750 | 0.050882 | 0.757617 | 0.772934 | 4575.86 | 55.68 | 0.00 |
| future_exact_content_oracle | 6/8 | 0.7500 | 0.116696 | 0.783271 | 0.803592 | 4425.98 | 51.34 | 239.00 |

## Paired deployable comparison

Sampled-unigram versus recent-sequence gains/losses/equal: 1/1/6.

## Scope boundary

Focused decision: **PILOT_NARROW_WITH_ONE_ALLOWED_LOSS**. The maximum conclusion is PILOT_NARROW because N=8. Future-exact uses up to seven full-expert autoregressive lookahead calls plus one pseudo traversal per boundary and is not deployable. Accuracy, token identity, NLL, route coverage, probe cost, and total generation runtime are measured. Transfer reduction is simulated. No production offloading speedup or full-dataset accuracy claim is made.
