# pseudo_embedding_qwen_gsm8k_residual_window_v1

Focused decision: **STOP/PIVOT**. The best calibration-free residual-window construction did not pass the predeclared development progress gate, so no new held-out route run and no task-accuracy closed-loop generation were executed.

## Scope and evidence boundary

Qwen3-30B-A3B-Instruct-2507, GSM8K v17 saved trajectories, H=8, B=32 (25% of 128 routed experts), native top-k=8. Route metrics below come from teacher-forced saved tokens while each policy's own hard subset changes its hidden state. They are not task accuracy, exact-token identity, or free-generation evidence. Transfer is simulated; probe and replay times are measured.

## Residual and content mechanism smoke

| Method | Route hit | Selected mass | Simulated transfer reduction |
|---|---:|---:|---:|
| `residual__zero` | 0.568848 | 0.570145 | 0.228678 |
| `residual__previous_window_last_repeated` | 0.573079 | 0.569011 | 0.212809 |
| `residual__previous_window_mean_repeated` | 0.625081 | 0.624320 | 0.277100 |
| `residual__previous_window_position_aligned` | 0.668783 | 0.674895 | 0.317546 |

Position-aligned residuals won the frozen residual rule. With that residual bank, sampled-next-token repeated content plus independent anchors was the best deployable content variant. Exact-future-token variants were diagnostic only and excluded from selection.

| Content method | Route hit | Selected mass |
|---|---:|---:|
| `content__sampled_repeat_independent` | 0.668783 | 0.674895 |
| `content__sampled_repeat_causal` | 0.662842 | 0.668941 |
| `content__recent_sequence_independent` | 0.658936 | 0.664874 |
| `content__recent_sequence_causal` | 0.638102 | 0.642685 |
| `content__exact_future_independent` | 0.687337 | 0.697412 |
| `content__exact_future_causal` | 0.681803 | 0.692742 |

## Seven pseudo constructions and controls

| Policy | Hit | Mass | Transfer | Probe s/boundary |
|---|---:|---:|---:|---:|
| `candidate__residual_pseudo_only` | 0.625082 | 0.615756 | 0.402023 | 0.830022 |
| `candidate__sampled_anchor_one_plus_previous_window_route` | 0.634924 | 0.641210 | 0.392834 | 0.826984 |
| `candidate__equal_pseudo_history` | 0.641817 | 0.647567 | 0.406327 | 0.826055 |
| `candidate__sampled_top8_core_plus_history_fill` | 0.629797 | 0.640648 | 0.390399 | 0.832445 |
| `candidate__first_four_anchor_core_plus_history_fill` | 0.639706 | 0.649761 | 0.399547 | 0.829714 |
| `candidate__linear_horizon_decay` | 0.641021 | 0.647330 | 0.404992 | 0.832975 |
| `candidate__inverse_anchor_decay` | 0.641298 | 0.647810 | 0.405151 | 0.824530 |
| `candidate__previous_window_route_only` | 0.623634 | 0.634308 | 0.383814 | 0.824982 |
| `reference__previous_route_commitment` | 0.623634 | 0.634308 | 0.383814 | 0.000000 |
| `reference__hard_oracle_commitment` | 0.949966 | 0.970436 | 0.709207 | 0.000000 |
| `reference__static_frequency` | 0.349518 | 0.336575 | 0.317960 | 0.000000 |

The seven pseudo constructions are the first seven candidate rows above; previous-window-only is an invariant/control and the final three rows are references.

## Frozen development gate

- Route-hit gain over previous: 0.016072 (required 0.05; pass=False).
- Selected-mass gain: 0.015453 (required 0.05; pass=False).
- Oracle-gap recovery: hit 0.049250, mass 0.045973.
- Paired four-sample bootstrap 95% intervals: hit [0.011566162109375, 0.020694556921155183], mass [0.011291668194967353, 0.020387958993581007].
- Static, resident-fraction, simulated-transfer, and audit checks passed; only the two required improvement checks failed.

## Router sensitivity findings

Position alignment matters much more than last/mean repetition. The exact-future content diagnostic adds only about 0.0186 hit and 0.0225 mass over sampled-repeat independent on the two-row smoke, indicating that the real bottleneck is not only token content. On development, gains are front-loaded within the window and in early layers; later anchors decay and the last routed layers can regress.

Largest per-layer selected-minus-previous deltas:

- Layer 0: hit +0.083826, mass +0.081322.
- Layer 1: hit +0.076183, mass +0.077191.
- Layer 2: hit +0.057199, mass +0.066699.
- Layer 3: hit +0.038462, mass +0.040041.
- Layer 5: hit +0.034024, mass +0.033897.
- Layer 6: hit +0.033777, mass +0.032426.

Worst per-layer deltas:

- Layer 47: hit -0.031805, mass -0.036831.
- Layer 46: hit -0.007890, mass -0.010909.
- Layer 42: hit -0.006657, mass -0.006786.
- Layer 43: hit -0.003698, mass -0.003423.

## Audits and terminal action

All production cache identity/data-pointer/version/length, RNG, shadow-cache discard, native Qwen attention/RoPE/router/top-k normalization, hard-mask, and information-boundary audits passed. Previous-window-only exactly matched the independent previous-route reference on all four development samples. No learned or fitted values, default-vector values, labels, correctness, vanilla future trajectory, or future true tokens were used by selectable candidates.

Because the development gate failed, running the predeclared new held-out rows or the 16-row three-policy accuracy pilot would violate the frozen stop rule. Those stages are explicitly recorded as not run.
