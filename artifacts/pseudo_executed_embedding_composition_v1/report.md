# Pseudo-executed embedding composition v1

All candidates use a fresh native MoE residual computed on the pseudo hidden state. Boundary zero executes full native top-8; later boundaries execute only the policy's previous realized B=32 subset.

Development selected **self_greedy_causal** using route/cost metrics only.

## Development

| Variant | Route hit | Selected mass | Transfer reduction | Mean probe s |
|---|---:|---:|---:|---:|
| self_greedy_causal | 0.788767 | 0.805782 | 0.571655 | 2.1108 |
| self_expected_top8_causal | 0.782328 | 0.800275 | 0.560018 | 2.0717 |
| recent_sequence_causal | 0.700267 | 0.714061 | 0.465892 | 0.3100 |
| expected_top8_norm_matched_independent | 0.699310 | 0.712034 | 0.465200 | 2.0986 |
| expected_top8_repeat_independent | 0.697805 | 0.711211 | 0.463399 | 2.0983 |
| expected_top4_repeat_independent | 0.697408 | 0.711131 | 0.462535 | 2.0715 |
| expected_top16_repeat_independent | 0.697500 | 0.710825 | 0.462687 | 2.0915 |
| sampled_repeat_independent | 0.696340 | 0.709550 | 0.459656 | 2.0809 |
| recent_sequence_independent | 0.691060 | 0.704589 | 0.457703 | 2.0658 |
| sampled_then_expected_top8_causal | 0.690572 | 0.704058 | 0.456980 | 0.2993 |
| sampled_repeat_causal | 0.689189 | 0.701974 | 0.455536 | 0.2894 |
| current_repeat_independent | 0.647135 | 0.656730 | 0.401469 | 2.1184 |

Exact-future variants are non-deployable diagnostics and were excluded from selection.

## Particle follow-up

- self_top2_particle_probability_weighted: route hit 0.789358, selected mass 0.808190, mean probe 4.0874 s.
- self_top4_particle_probability_weighted: route hit 0.789388, selected mass 0.807734, mean probe 7.9009 s.

Particle variants are a separately declared follow-up and do not replace the development-selected held-out candidate.

## Held-out route evaluation

- previous_route_commitment: route hit 0.611168, selected mass 0.621438, simulated transfer reduction 0.368125.
- provided_previous_residual_control: route hit 0.631272, selected mass 0.642355, simulated transfer reduction 0.389598.
- sampled_repeat_independent: route hit 0.669112, selected mass 0.682921, simulated transfer reduction 0.445955.
- self_greedy_causal: route hit 0.765119, selected mass 0.784870, simulated transfer reduction 0.555027.

Route-only decision: **ROUTE_SIGNAL**. Oracle-gap recovery was not computed in this composition protocol, so this result cannot authorize task-accuracy generation or a full GO claim.

All route values are teacher-forced measurements on each policy's own hard-subset state. Transfer is simulated. Probe/replay time and temporary memory are measured. No task accuracy, free generation, exact-token identity, or runtime speedup was measured.
