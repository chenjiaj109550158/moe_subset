# pseudo_embedding_calibration_free_analysis_v1

Exploratory decision: **STOP/PIVOT**. The terminal v1 decision remains **STOP/PIVOT**.

This is a four-row, tensor-only, calibration-free hypothesis analysis. It is not held-out evidence, task accuracy, actual closed-loop generation, or a speedup claim.

## Calibration-free candidates

Frozen previous-route reference: hit 0.609385, mass 0.629428, simulated transfer reduction 0.361315.

- `zero_pseudo_all_anchors`: hit 0.566499 (-0.042886), mass 0.575402 (-0.054027), simulated transfer reduction 0.345299.
- `recent_route_prior_w8`: hit 0.626004 (+0.016618), mass 0.647512 (+0.018084), simulated transfer reduction 0.380445.
- `one_known_plus_seven_history`: hit 0.648443 (+0.039057), mass 0.670839 (+0.041411), simulated transfer reduction 0.410630.
- `equal_evidence_future_blend`: hit 0.657465 (+0.048079), mass 0.680111 (+0.050683), simulated transfer reduction 0.435678.
- `native_topk_core_plus_history_reserve`: hit 0.649004 (+0.039618), mass 0.675146 (+0.045717), simulated transfer reduction 0.416559.

## Decisive sensitivity observations

- Primary route hit is 0.753743 at anchor 1 and 0.489377 at anchor 8.
- Natural top-8 overlap from anchor 1 to lag 8 is 0.258185; repeated-token pseudo anchor overlap is 0.701761.
- Expected-top-8 changes primary pseudo very little: top-8 overlap 0.967458, subset Jaccard 0.972722.
- Removing default contributions changes pseudo top-8 overlap to 0.516474 and subset Jaccard to 0.519798 relative to primary.
- `equal_evidence_future_blend` paired leave-one-out hit improvement ranges from +0.039345 to +0.054138; mass improvement ranges from +0.041274 to +0.057534.

## Claim boundary

Candidate formulas use only the sampled next-token zero probe and current-policy online realized route history. No learned or fitted value, offline prior, default vector, future token, answer, correctness, or accuracy enters candidate selection. Source replay is open-loop; transfer is simulated; model and GPU runtime were not measured in this tensor-only analysis.
