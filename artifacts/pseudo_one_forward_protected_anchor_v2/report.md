# One-forward protected-anchor correction v2

All candidates protect the known-token first anchor and use one native causal eight-token pseudo traversal per boundary.

| Variant | Route hit | Selected mass | Anchor-1 hit | Anchor-1 mass | Probe s |
|---|---:|---:|---:|---:|---:|
| recent_sequence_causal_uncorrected | 0.700267 | 0.714061 | 0.924479 | 0.949430 | 0.3082 |
| hidden_velocity_anchor1_zero_horizon_damped | 0.700795 | 0.713902 | 0.924642 | 0.949339 | 0.3443 |
| residual_velocity_anchor1_zero_undamped | 0.698659 | 0.711272 | 0.928304 | 0.952783 | 0.3310 |
| residual_velocity_anchor1_zero_horizon_damped | 0.703389 | 0.716302 | 0.925130 | 0.950148 | 0.3344 |
| residual_velocity_anchor1_zero_horizon_damped_cap25 | 0.701029 | 0.714478 | 0.924805 | 0.949693 | 0.3497 |
| residual_velocity_anchor1_zero_horizon_damped_cap25_anchor1_core | 0.657410 | 0.655205 | 0.912842 | 0.937424 | 0.3513 |

## Comparisons to checksum-pinned uncorrected v1

- hidden_velocity_anchor1_zero_horizon_damped: hit/mass delta +0.000529/-0.000160; anchor-1 delta +0.000163/-0.000091; paired 95% [-0.00152587890625, 0.002583821614583315] and [-0.003218638076024488, 0.002821566590110597]; centered-logit RMS 0.589972; signal `False`.
- residual_velocity_anchor1_zero_undamped: hit/mass delta -0.001607/-0.002789; anchor-1 delta +0.003825/+0.003353; paired 95% [-0.006734212239583315, 0.0030721028645832593] and [-0.008033389669592134, 0.002683451615006577]; centered-logit RMS 0.900979; signal `False`.
- residual_velocity_anchor1_zero_horizon_damped: hit/mass delta +0.003123/+0.002240; anchor-1 delta +0.000651/+0.000718; paired 95% [0.0012613932291666852, 0.00498453776041663] and [0.001701744676105732, 0.0027786128804058174]; centered-logit RMS 0.466586; signal `False`.
- residual_velocity_anchor1_zero_horizon_damped_cap25: hit/mass delta +0.000763/+0.000417; anchor-1 delta +0.000326/+0.000263; paired 95% [-0.0019327799479166297, 0.0037841796874999722] and [-0.001960268228487938, 0.0027943596538393956]; centered-logit RMS 0.197246; signal `False`.
- residual_velocity_anchor1_zero_horizon_damped_cap25_anchor1_core: hit/mass delta -0.042857/-0.058856; anchor-1 delta -0.011637/-0.012005; paired 95% [-0.047607421875, -0.03810628255208337] and [-0.06407789657857121, -0.053634144060630495]; centered-logit RMS 0.273495; signal `False`.

The best protected+damped residual candidate is positive on all four samples, but recovers only 0.003123 route hit and 0.002240 selected mass—far below the frozen +0.02 signal. A same-cache native unit test confirms that coefficient zero leaves anchor-one router logits exact. The separately executed BF16 source and candidate artifacts are not expected to be bitwise identical, and later hard policies realize different contexts.

Development-only decision: **STOP/PIVOT**. Selected route signal: `None`.

Route values are teacher-forced on each policy's own hard-subset state. Probe/replay cost is measured and transfer is simulated. No held-out route, task accuracy, free generation, exact-token identity, or runtime speedup was measured.
