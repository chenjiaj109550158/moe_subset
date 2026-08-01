# Calibration-free pseudo-embedding analysis v1

## Purpose and boundary

This protocol analyzes why `pseudo_embedding_qwen_gsm8k_v1` failed its frozen
development gate and constructs the strongest next training-free hypothesis
without changing or relabeling the terminal v1 `STOP/PIVOT`. The analysis uses
only the four existing Qwen/GSM8K development tensors. It does not load a model,
add samples, run held-out routes, inspect task accuracy, or execute actual
closed-loop generation.

Deployable candidates must be calibration-free. They may read the already
sampled next token, the current policy's read-only cache, and that same policy's
realized route history. They may not use learned parameters, fitted coefficients,
offline expert frequencies, route-transition tables, default-vector values,
future true tokens, a vanilla future trajectory, benchmark answers, or
correctness. When no online history exists, candidates fall back to the zero-
contribution pseudo probe rather than a static-frequency artifact.

Diagnostic oracles may use forbidden future information only when explicitly
labeled. Their outputs cannot rank deployable candidates. Any future model-based
mechanism experiment, new trace, or held-out scope requires a separate frozen
amendment before execution.

The exact machine-readable scope is
`configs/analysis/pseudo_embedding_calibration_free_analysis_v1.yaml`.

## Questions fixed before analysis

1. At which anchor does repeated sampled-token pseudo routing stop tracking the
   natural route?
2. How much of natural routing is temporally persistent at lags 1 through 8?
3. Which layers favor token-content pseudo information, online route history, or
   neither?
4. Are errors concentrated at low 8th/9th router margins, or are pseudo hidden
   directions wrong even for stable decisions?
5. How strongly do token content, causal pseudo history, default-vector residuals,
   expected-token averaging, and future RoPE position change router scores?
6. Can a calibration-free, analytically weighted combination of the known next
   token and online route history beat previous-route by the frozen `+0.05`
   route-hit and selected-mass references?

## Frozen first-pass analyses

### A. Horizon and temporal structure

- Report route hit and selected mass independently for anchors 1 through 8.
- Report centered-logit cosine and natural top-8 overlap at temporal lags 1
  through 8.
- Compare the natural temporal change with the change induced only by moving one
  repeated pseudo token through future RoPE positions.

### B. Layer and margin structure

- Report all metrics by layer and by fixed eight-layer blocks.
- Count the layers where each pseudo variant beats previous-route.
- Stratify pseudo-natural score alignment by natural 8th/9th margin quartile.
- Report concrete best and worst layers without selecting a layer-specific
  policy in this analysis.

### C. Component interventions from saved tensors

- `sampled_next` versus `current_token` measures token-content sensitivity.
- `sampled_next_default` versus `sampled_next_zero` measures accumulated
  default-vector residual sensitivity.
- independent versus causal measures repeated pseudo-history sensitivity.
- sampled-token versus expected-top-8 measures expected-embedding sensitivity.
- Each comparison reports normalized centered-logit RMS change, pseudo top-8
  overlap, and final subset Jaccard.

### D. Frozen calibration-free candidates

All candidate utilities use the saved zero-contribution pseudo probabilities.
No default-vector value enters a candidate.

1. `zero_pseudo_all_anchors`: sum the eight position-conditioned zero-pseudo
   distributions.
2. `recent_route_prior_w8`: sum the native selected routing weights from the
   preceding realized eight-token window; at boundary zero use candidate 1.
3. `one_known_plus_seven_history`: represent the one known next token by anchor-1
   zero-pseudo probability and the seven unknown tokens by seven copies of the
   normalized recent-route distribution. At boundary zero use candidate 1.
4. `equal_evidence_future_blend`: use anchor 1 directly; for each unknown anchor,
   average its position-conditioned zero-pseudo distribution and the online
   recent-route prior with equal, non-fitted weight. At boundary zero use
   candidate 1.
5. `native_topk_core_plus_history_reserve`: reserve the native top-8-sized core
   for anchor-1 zero-pseudo experts and fill the other 24 slots from online
   history. At boundary zero use candidate 1.

The factors one, seven, eight, and the 8/24 budget split derive only from frozen
`H=8`, native top-k 8, and `B=32`; none is fitted to result rows.

### E. Stability and cost boundary

- Report each candidate globally, per sample, per anchor, and per layer.
- Report leave-one-sample-out aggregates to expose single-row dependence.
- Reuse the measured zero-pseudo probe cost; online history aggregation receives
  separately measured CPU analysis time but is not called model runtime.
- Transfer remains simulated and no speedup is claimed.

## Result interpretation

This four-row analysis generates hypotheses only. A candidate is interesting
only if it improves both mean route hit and selected mass over the frozen
previous-route baseline by at least `0.05`, retains 25% residency, and has at
least 30% simulated transfer reduction. Passing those references does not revive
v1 or authorize held-out/accuracy execution. It would justify a new versioned
mechanism smoke and disjoint held-out route protocol.

Post-hoc coefficient sweeps may be recorded in a clearly separate diagnostic
appendix, but their maxima cannot select a method. Task accuracy, answer text,
and correctness never enter this analysis.

## Deferred mechanism experiments

The following experiments are planned but explicitly out of the first-pass
tensor-only execution:

1. same-process BF16 natural/pseudo parity;
2. true-future-token independent and causal content oracles;
3. natural token/attention/router-input/expert-residual component patching;
4. router-weight projection and hidden-direction error analysis;
5. recent production token-content and router-input state banks;
6. zero/default shrinkage and layer cutoffs, with default-based variants kept
   separate because they are not calibration-free under this protocol;
7. autoregressive soft-token and multi-particle pseudo rollout;
8. robust/max/entropy-aware subset objectives;
9. a new frozen held-out route set only after a qualifying deployable candidate.

Before any deferred item runs, its exact rows, information regime, variants,
cost cap, and stop rule must be committed. Diagnostic future-token or natural-
state oracles remain ineligible for deployment claims.

## Completion note

The tensor analysis, native content smoke, prompt-route smoke, four-row
development, and separately frozen eight-row held-out route gate are complete.
The held-out candidate improved previous route consistently but failed every
route/selected-mass strong-candidate threshold, so the final result remains
`STOP/PIVOT` and no accuracy ran. The integrated findings, measured costs,
post-hoc horizon diagnostic, and ordered next calibration-free experiments are
recorded in `pseudo_embedding_calibration_free_synthesis_v1.md`.
