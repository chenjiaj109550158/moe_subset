# Penultimate-joint Qwen/GSM8K wave-1 accuracy report

Actual hard closed-loop generation uses the same eight frozen GSM8K rows at 
H=8 and B=32. The online baseline is checksum-pinned and was not rerun.

| Policy | Correct | Token agreement | Route hit | Selected mass |
|---|---:|---:|---:|---:|
| natural_top8_intersection_zero_missing | 8/8 | 0.047599 | 0.713199 | 0.734273 |
| penultimate_unigram_joint_equivalent | 7/8 | 0.031714 | 0.685858 | 0.706211 |

Penultimate versus online paired gains/losses/equal: 0/1/7.

## Scope boundary

Focused decision: **PILOT_NARROW_ONE_ALLOWED_LOSS_AWAITING_USER_CONFIRMATION**. The candidate simulator uses a 
disposable duplicate bridge to preserve the intended state dependency. Its 
latency is not joint-kernel runtime. Offloading, prefetch overlap, fused runtime, 
and production speedup are not measured. The pipeline is stopped at the required 
user-confirmation checkpoint; N=8 cannot establish full-dataset preservation.
