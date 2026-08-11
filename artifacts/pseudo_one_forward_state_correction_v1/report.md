# One-forward state correction v1

All variants use one native causal eight-token pseudo forward and execute fresh MoE residuals under the previous realized B=32 subset after boundary zero.

| Variant | Route hit | Selected mass | Transfer reduction | Probe s |
|---|---:|---:|---:|---:|
| recent_sequence_causal_hidden_velocity_linear_norm | 0.648031 | 0.661326 | 0.391418 | 0.3253 |
| recent_sequence_causal_residual_velocity_linear_norm | 0.676310 | 0.691583 | 0.434408 | 0.3251 |
| recent_sequence_causal_uncorrected | 0.700267 | 0.714061 | 0.465892 | 0.3082 |

## Comparisons

- recent_sequence_causal_hidden_velocity_linear_norm: route-hit delta -0.052236, selected-mass delta -0.052736, paired 95% intervals [-0.05489095052083329, -0.047922770182291685] and [-0.0602746502906516, -0.045457794141123276]; route signal `False`.
- recent_sequence_causal_residual_velocity_linear_norm: route-hit delta -0.023956, selected-mass delta -0.022478, paired 95% intervals [-0.02849324544270837, -0.019734700520833315] and [-0.02907186346235069, -0.015883915971841245]; route signal `False`.

## Mechanism diagnosis

| Variant | Anchor-1 hit | Anchor-1 mass | Centered-logit RMS vs base | Pseudo top-8 overlap vs base |
|---|---:|---:|---:|---:|
| recent_sequence_causal_hidden_velocity_linear_norm | 0.734782 | 0.762451 | 1.290994 | 0.195079 |
| recent_sequence_causal_residual_velocity_linear_norm | 0.825602 | 0.861247 | 0.957475 | 0.388021 |

The uncorrected anchor-1 reference is 0.924479 route hit and 0.949430 selected mass. Both linear velocity terms perturb this strongest known-token anchor and do not recover the loss at later anchors.

## Integrity

All cache/RNG/shadow/information audits pass: `True`. Validated atomic sample-policy pairs: 12; failed markers: 0; checksum resume: `True`.

Development-only decision: **STOP/PIVOT**.

Route values are teacher-forced on each policy's own hard-subset state. Probe/replay cost is measured and transfer is simulated. No held-out route, task accuracy, free generation, exact-token identity, or runtime speedup was measured.
