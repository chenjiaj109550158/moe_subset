# pseudo_embedding_calibration_free_prompt_route_smoke_v1

Mechanism decision: **CANDIDATE_FOR_NEW_FROZEN_DEVELOPMENT**. The focused v1 terminal decision remains **STOP/PIVOT**.

This is a two-row native-Qwen prompt-prefill route smoke plus open-loop saved-token scoring. It is not held-out evidence, task accuracy, hard closed-loop generation, or a runtime speedup claim.

## Results

- `sampled_repeat_independent_zero` control: hit 0.524495, mass 0.534683, simulated transfer reduction 0.187500.
- `prompt_or_recent_route_prior`: hit 0.685628 (+0.161133), mass 0.716782 (+0.182098), simulated transfer reduction 0.328776.
- `sampled_known_plus_prompt_or_recent_history`: hit 0.702393 (+0.177897), mass 0.733202 (+0.198519), simulated transfer reduction 0.352132.
- `equal_sampled_pseudo_prompt_or_recent_history`: hit 0.676270 (+0.151774), mass 0.705778 (+0.171095), simulated transfer reduction 0.330322.
- `sampled_top8_core_plus_prompt_or_recent_history_fill`: hit 0.700277 (+0.175781), mass 0.733516 (+0.198832), simulated transfer reduction 0.353027.
- `equal_recent_pseudo_prompt_or_recent_history`: hit 0.673747 (+0.149251), mass 0.703529 (+0.168845), simulated transfer reduction 0.335612.

## Claim boundary

Every deployable score uses only the sampled next token, same-request native prompt routes, current-policy realized generated routes, and saved zero-contribution pseudo probabilities. No training, fitted coefficient, offline expert prior, transition table, default-vector value, future token, answer, correctness, or task accuracy is used. Prompt prefill capture latency is measured capture-inclusive runtime; it is not an isolated overhead measurement. Transfer is simulated.
