# pseudo_embedding_calibration_free_content_smoke_v1

Mechanism result: **MECHANISM_PROMISING_FOR_NEW_FROZEN_DEVELOPMENT**. The focused v1 terminal decision remains **STOP/PIVOT**.

This is a two-row, 16-token-per-row, native-Qwen teacher-forced route smoke. It is not held-out evidence, task accuracy, hard closed-loop generation, or a speedup claim.

## Deployable calibration-free methods

- `sampled_repeat_independent_zero`: hit 0.524495, mass 0.534683, simulated transfer reduction 0.187500.
- `recent_window_independent_zero`: hit 0.519368, mass 0.533308, simulated transfer reduction 0.206787.
- `recent_window_causal_zero`: hit 0.482422, mass 0.490718, simulated transfer reduction 0.164144.
- `recent_window_equal_history_blend`: hit 0.599202, mass 0.621052, simulated transfer reduction 0.236410.

## Diagnostic content oracles

- `true_future_independent_zero`: hit 0.571615, mass 0.593974. This uses future true token content and is excluded from deployable ranking.
- `true_future_causal_zero`: hit 0.536051, mass 0.559393. This uses future true token content and is excluded from deployable ranking.

## Main comparisons

- Recent independent content versus sampled repeat: hit -0.005127, mass -0.001375.
- True-future independent content headroom versus sampled repeat: hit +0.047119, mass +0.059291.
- Parent-formula replication is best for `one_known_plus_seven_history` at hit 0.620768, mass 0.643289; this diagnostic does not rerank the frozen content-smoke candidates.

## Calibration and claim boundary

Deployable methods use only the sampled next token, current read-only production cache, last seven already-realized prompt/generated token IDs, and the immediately preceding realized route window. No learned/fitted value, offline expert prior, route-transition table, saved default-vector value, answer, correctness, or task accuracy is used. The all-zero expert tensor exists only to satisfy the native probe interface. Transfer is simulated; probe latency and CUDA temporary memory are measured.
