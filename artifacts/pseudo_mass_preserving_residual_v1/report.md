# Mass-preserving previous-subset residual v1

All new candidates are calibration-free and deployable at the boundary: sampled-unigram known-context content, one batched causal H=8 pseudo traversal, and only the previous realized layer-local B32 experts after boundary zero.

| Policy | Route hit | Selected mass | Transfer reduction | Probe s |
|---|---:|---:|---:|---:|
| captured_natural_plus_substitute_missing_mass | 0.728444 | 0.746048 | 0.500773 | 0.5089 |
| natural_top8_intersection_zero_missing | 0.730438 | 0.747387 | 0.504354 | 0.2209 |
| rerouted_top8_scaled_by_captured_natural_mass | 0.728994 | 0.746306 | 0.503540 | 0.2455 |
| sampled_unigram_full_continuation | 0.730357 | 0.747085 | 0.502248 | 0.3055 |

## Comparisons to checksum-pinned sampled-unigram baseline

- captured_natural_plus_substitute_missing_mass: route/mass delta -0.001912/-0.001037; anchor-1 +0.000000/+0.000000; eligible `False`; paired 95% intervals [-0.00406901041666663, 0.0002441406250000555] and [-0.002942120704200718, 0.000642401592972297].
- natural_top8_intersection_zero_missing: route/mass delta +0.000081/+0.000302; anchor-1 +0.000000/+0.000000; eligible `True`; paired 95% intervals [-0.0016886393229166574, 0.0018310546875] and [-0.0012988067778050838, 0.0021002242640024704].
- rerouted_top8_scaled_by_captured_natural_mass: route/mass delta -0.001363/-0.000779; anchor-1 +0.000000/+0.000000; eligible `False`; paired 95% intervals [-0.003743489583333315, 0.0016479492187499722] and [-0.0030108367780997125, 0.0022957188291971575].

## Residual weight semantics

- captured_natural_plus_substitute_missing_mass: execution-weight sum mean 0.999989; captured natural mass mean 0.722162.
- natural_top8_intersection_zero_missing: execution-weight sum mean 0.732427; captured natural mass mean 0.732425.
- rerouted_top8_scaled_by_captured_natural_mass: execution-weight sum mean 0.735197; captured natural mass mean 0.735217.

Development decision: **NARROW**; selected candidate: `natural_top8_intersection_zero_missing`; original development progress gate: `True`.

All candidate cache/RNG/information/weight audits pass: `True`. The baseline, previous-route, and static rows are checksum-pinned prior measured rows and were not rerun.

Route and task-state metrics plus probe cost are measured; transfer is simulated. No held-out route, task accuracy, free generation, exact-token identity, NLL, closed-loop runtime, or runtime speedup was measured.
