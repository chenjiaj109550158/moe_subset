# Benchmark subset oracle v1 report

Config fingerprint: `b68e18f45d373b31c45d99e8555e25fbf2816d7da8a68946536d6402668767a2`

Execution scope: `benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2` (`a5a908ad0124d2041b589a681e7102d3ea5d3f7df90752075799bb9875871d7c`).

Full accuracy task scope: `gsm8k`.
Unscheduled task artifacts are preserved as provenance and are not aggregated.

Overall decision: **NARROW**

only explicitly NARROW task scopes are authorized

The natural reference is measured v17 generation reused without rerunning. Lossless 
residency transfer/stall metrics are simulated from natural route replay and its 
exact-token identity has actual smoke evidence. Required full actual policies: 
`hard_oracle_commitment`. Their runtime is measured closed-loop generation; no hard 
result is identity-materialized. Preserved unscheduled rows are provenance only.

Mechanism smoke rows: 24; lossless identity passed.

## Paired hard-commitment accuracy

| Model | Task | H | B | Vanilla | Hard | Delta | 95% CI | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt_oss_20b | gsm8k | 1 | 4 | 1242/1319 | 1242/1319 | +0.0000 | [+0.0000, +0.0000] | PASS |

AIME retains integer question counts. Per-task/per-point decisions are in 
`decisions_per_model_task_point.csv`; no cross-task macro average can override them.
