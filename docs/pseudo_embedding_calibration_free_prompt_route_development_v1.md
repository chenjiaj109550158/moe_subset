# Calibration-free prompt-route development v1

## Frozen scope

The two-row prompt-route smoke passed its predeclared mechanism references and
showed that its two best analytic constructions were effectively tied. This
development protocol therefore evaluates four previously frozen Qwen/GSM8K
development rows: `test-0`, `test-439`, `test-879`, and `test-1318`. Their saved
128-token natural and zero-pseudo tensors are checksum-locked before execution.
The only new model work is same-request native prefill route capture for the
last eight prompt tokens.

No held-out row, new dataset item, answer, correctness, task accuracy, actual
hard generation, default-vector value, learned parameter, fitted coefficient,
offline expert frequency, or route-transition table is used. Exact machine-
readable scope and checksums are in
`configs/analysis/pseudo_embedding_calibration_free_prompt_route_development_v1.yaml`.

## Candidates and selection

All candidates use prompt last-8 route history at boundary zero and the current
policy's preceding generated H=8 route window thereafter. The four candidates
are history-only, one sampled-known anchor plus seven history units, equal
sampled-pseudo/history evidence, and sampled anchor-1 top-8 core plus 24 history
slots. Constants derive only from H=8, native top-k 8, and B=32.

Ranking uses development routing and simulated transfer only. A candidate must
beat the frozen previous-route reference by at least 0.05 in both route hit and
selected mass, not trail static frequency, retain at least 0.30 simulated
transfer reduction, and pass cache/RNG/information audits. Per-sample,
leave-one-out, boundary, anchor, and eight-layer-block results expose instability.

Passing produces only `candidate_for_new_frozen_held_out_route_protocol`.
Held-out route replay would require another committed protocol. It does not
authorize accuracy or revive the focused-v1 terminal `STOP/PIVOT` decision.
