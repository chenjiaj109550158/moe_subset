# Calibration-free recent-content mechanism smoke v1

## Frozen question

The tensor-only parent analysis found that the calibration-free
`equal_evidence_future_blend` nearly reached the route progress reference, while
the repeated sampled-token pseudo anchors changed far less across positions than
natural routes. This mechanism smoke asks whether replacing repeated unknown-
future content with a bank of recent, actually realized token contents improves
native-Qwen shadow routing without any calibration artifact.

The scope is fixed before model execution in
`configs/analysis/pseudo_embedding_calibration_free_content_smoke_v1.yaml`. It
reuses the existing mechanism rows `test-786` and `test-394`, runs only 16 saved
v17 tokens per row, and keeps `H=8,B=32`. It does not add samples or authorize
held-out routes or task accuracy.

## Calibration-free deployable variants

All variants use zero routed-expert contribution. A synthetic all-zero tensor is
used only to satisfy the probe shape interface; no saved default-vector value,
count, fingerprint, offline expert prior, transition table, trained parameter,
or fitted coefficient is read by the method.

1. `sampled_repeat_independent_zero` repeats the known sampled next token at all
   eight positions and is the native zero-probe control.
2. `recent_window_independent_zero` places the sampled next token at anchor 1.
   Anchors 2–8 receive the last seven prompt/generated tokens available at the
   boundary, in chronological order. Each anchor independently attends the
   production prefix.
3. `recent_window_causal_zero` uses the same content sequence but lets anchors
   attend earlier shadow anchors.
4. `recent_window_equal_history_blend` uses the independent recent-content
   probabilities. Anchor 1 enters directly; for anchors 2–8, their pseudo
   probability and the normalized preceding realized-window route distribution
   receive equal weight. The one/seven counts and equal evidence weight are fixed
   analytically, not fitted. At the first boundary, where no generated route
   history exists, it uses the recent-content pseudo utility alone.

All deployable content comes from the current policy's already realized prompt,
tokens, routes, and sampled next token. Production cache, hidden state, and RNG
remain read-only.

## Diagnostic content oracles

`true_future_independent_zero` and `true_future_causal_zero` use the saved true
future token IDs as shadow contents. They are explicitly non-deployable
information oracles. Their only purpose is to measure how much headroom exact
future content provides and whether causal shadow-history error remains after
content is corrected. They cannot rank deployable variants or justify accuracy.

## Outputs and stop boundary

The smoke saves sample-atomic tensors and JSON, per-anchor/layer route metrics,
probe cost, cache/RNG audits, aggregate comparisons, provenance, checksums, and a
report. A deployable candidate must improve over the repeated-zero control and
must not use future tokens. This two-row mechanism result is hypothesis evidence
only; even a positive result requires a new frozen four-row development and
disjoint held-out protocol. No task accuracy, closed-loop hard generation, or
speedup claim is permitted here.

## Completed result

Recent realized token contents did not improve the repeated sampled-token
control: independent route hit/mass changed by -0.00513/-0.00138. Causal
zero-residual propagation further changed them by -0.03695/-0.04259. The
non-deployable exact-future-content independent oracle improved the control by
+0.04712/+0.05929, establishing content headroom while also showing that recent
token IDs are not an adequate proxy. Adding online route history to recent
content improved that weak content variant by +0.07983/+0.08774.

All cache/RNG/shadow-lifetime audits passed. The result pivoted the next
mechanism test from recent embeddings to same-request prompt/generated route
history; it did not authorize accuracy.
