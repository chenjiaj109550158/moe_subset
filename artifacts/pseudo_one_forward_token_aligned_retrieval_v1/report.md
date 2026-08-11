# One-forward token-aligned state retrieval v1

Each candidate uses one native causal H=8 pseudo traversal and retrieves only same-policy prompt or already-realized token states.

| Variant | Route hit | Selected mass | Anchor-1 hit | Anchor-1 mass | Total plan s |
|---|---:|---:|---:|---:|---:|
| recent_sequence_causal_uncorrected | 0.700267 | 0.714061 | 0.924479 | 0.949430 | 0.3082 |
| exact_recent_residual_additive_norm | 0.699565 | 0.712088 | 0.918783 | 0.943416 | 0.3527 |
| exact_recent_residual_replace_norm | 0.694417 | 0.706798 | 0.897624 | 0.923027 | 0.3446 |
| exact_then_embedding_nearest_residual_additive_norm | 0.699025 | 0.712238 | 0.915202 | 0.943171 | 0.3407 |
| exact_recent_router_input_additive_norm | 0.694519 | 0.708124 | 0.892253 | 0.920097 | 0.3354 |
| exact_then_embedding_nearest_router_input_additive_norm | 0.692373 | 0.705243 | 0.882080 | 0.913021 | 0.3401 |

## Comparisons to checksum-pinned uncorrected v1

- exact_recent_residual_additive_norm: hit/mass delta -0.000702/-0.001973; anchor-1 delta -0.005697/-0.006014; signal `False`.
- exact_recent_residual_replace_norm: hit/mass delta -0.005849/-0.007264; anchor-1 delta -0.026855/-0.026402; signal `False`.
- exact_then_embedding_nearest_residual_additive_norm: hit/mass delta -0.001241/-0.001824; anchor-1 delta -0.009277/-0.006258; signal `False`.
- exact_recent_router_input_additive_norm: hit/mass delta -0.005747/-0.005938; anchor-1 delta -0.032227/-0.029333; signal `False`.
- exact_then_embedding_nearest_router_input_additive_norm: hit/mass delta -0.007894/-0.008818; anchor-1 delta -0.042399/-0.036409; signal `False`.

Development-only decision: **STOP/PIVOT**. Selected route signal: `None`.

Route values are teacher-forced on each policy's own hard-subset state. Probe and retrieval costs are measured; transfer is simulated. No held-out route, task accuracy, free generation, exact-token identity, closed-loop runtime, or runtime speedup was measured.
