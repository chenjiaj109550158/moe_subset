# One-forward mid-layer shifted self-conditioning v1

Each candidate uses one native causal H=8 traversal, one shifted seven-query LM-head refresh after layer 23, and fresh native MoE residuals.

| Variant | Route hit | Selected mass | Anchor-1 hit | Anchor-1 mass | Probe s |
|---|---:|---:|---:|---:|---:|
| recent_sequence_causal_uncorrected | 0.700267 | 0.714061 | 0.924479 | 0.949430 | 0.3082 |
| midpoint_greedy_shift_uncapped | 0.700521 | 0.714245 | 0.925863 | 0.950540 | 0.3183 |
| midpoint_expected_top8_shift_uncapped | 0.700531 | 0.714320 | 0.925456 | 0.950338 | 0.3067 |
| midpoint_greedy_shift_cap25 | 0.700521 | 0.714245 | 0.925863 | 0.950540 | 0.3027 |

## Comparisons to checksum-pinned uncorrected v1

- midpoint_greedy_shift_uncapped: hit/mass delta +0.000254/+0.000183; anchor-1 delta +0.001383/+0.001111; early/late mass 0.707158/0.721331; signal `False`.
- midpoint_expected_top8_shift_uncapped: hit/mass delta +0.000264/+0.000259; anchor-1 delta +0.000977/+0.000908; early/late mass 0.707158/0.721483; signal `False`.
- midpoint_greedy_shift_cap25: hit/mass delta +0.000254/+0.000183; anchor-1 delta +0.001383/+0.001111; early/late mass 0.707158/0.721331; signal `False`.

Development-only decision: **STOP/PIVOT**. Selected route signal: `None`.

The non-gating strict BF16 norm diagnostic (`torch.allclose`, `rtol=atol=1e-3`) passed 29/96 boundary-policy checks. The implementation applies per-anchor L2 norm restoration and the FP32 native unit test passes; this tighter-than-BF16 post-cast diagnostic was not a frozen gate.

Route values are teacher-forced on each policy's own hard-subset state. Probe and LM-head costs are measured; transfer is simulated. No held-out route, task accuracy, free generation, exact-token identity, closed-loop runtime, or runtime speedup was measured.
