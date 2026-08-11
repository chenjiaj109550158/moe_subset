# pseudo_embedding_qwen_gsm8k_v1

Focused decision: **STOP/PIVOT**.

This report is limited to Qwen/GSM8K at `H=8,B=32` (32/128 routed experts per layer, native top-8). It is not a full-dataset GO or a runtime-speedup claim.

## Route evidence

### mechanism_smoke

- `current_token_independent_default_topk`: route hit 0.362142, selected mass 0.349491, simulated transfer reduction -0.137858.
- `expected_top8_independent_default_topk`: route hit 0.631836, selected mass 0.648094, simulated transfer reduction 0.131836.
- `hard_oracle_commitment`: route hit 0.974772, selected mass 0.988284, simulated transfer reduction 0.474772.
- `previous_route_commitment`: route hit 0.214681, selected mass 0.211900, simulated transfer reduction -0.285319.
- `sampled_next_causal_default_topk`: route hit 0.526367, selected mass 0.542333, simulated transfer reduction 0.026367.
- `sampled_next_independent_default_topk`: route hit 0.612956, selected mass 0.627416, simulated transfer reduction 0.112956.
- `sampled_next_independent_zero`: route hit 0.529948, selected mass 0.541039, simulated transfer reduction 0.029948.
- `static_frequency`: route hit 0.214681, selected mass 0.211900, simulated transfer reduction -0.285319.

### development

- `current_token_independent_default_topk`: route hit 0.488626, selected mass 0.485859, simulated transfer reduction 0.252913.
- `expected_top8_independent_default_topk`: route hit 0.533448, selected mass 0.534745, simulated transfer reduction 0.284271.
- `hard_oracle_commitment`: route hit 0.958714, selected mass 0.976395, simulated transfer reduction 0.717288.
- `previous_route_commitment`: route hit 0.609385, selected mass 0.629428, simulated transfer reduction 0.361315.
- `sampled_next_causal_default_topk`: route hit 0.504097, selected mass 0.505684, simulated transfer reduction 0.246557.
- `sampled_next_independent_default_topk`: route hit 0.533015, selected mass 0.534416, simulated transfer reduction 0.283566.
- `sampled_next_independent_zero`: route hit 0.566499, selected mass 0.575402, simulated transfer reduction 0.345299.
- `static_frequency`: route hit 0.339705, selected mass 0.346426, simulated transfer reduction 0.308084.

## Frozen development gate

Progress gate pass: `False`; selected variant: `None`. Ranking used route and measured cost only, never GSM8K correctness or answers.

- `sampled_next_independent_default_topk`: hit delta -0.076370 (paired 95% CI -0.100684 to -0.047501); mass delta -0.095012 (paired 95% CI -0.121571 to -0.062452); oracle-gap recovery -0.218619/-0.273835; failed checks: `route_hit_improvement_pass,selected_mass_improvement_pass,transfer_reduction_pass`.
- `current_token_independent_default_topk`: hit delta -0.120759 (paired 95% CI -0.133667 to -0.108276); mass delta -0.143569 (paired 95% CI -0.158177 to -0.128794); oracle-gap recovery -0.345689/-0.413782; failed checks: `route_hit_improvement_pass,selected_mass_improvement_pass,transfer_reduction_pass`.
- `sampled_next_causal_default_topk`: hit delta -0.105289 (paired 95% CI -0.129643 to -0.082214); mass delta -0.123744 (paired 95% CI -0.151151 to -0.096420); oracle-gap recovery -0.301403/-0.356644; failed checks: `route_hit_improvement_pass,selected_mass_improvement_pass,transfer_reduction_pass`.
- `sampled_next_independent_zero`: hit delta -0.042886 (paired 95% CI -0.075195 to -0.011464); mass delta -0.054027 (paired 95% CI -0.090604 to -0.018500); oracle-gap recovery -0.122768/-0.155711; failed checks: `route_hit_improvement_pass,selected_mass_improvement_pass`.

## Measured probe cost

- `current_token_independent_default_topk`: 0.825793 s/boundary mean, 195166208 temporary CUDA bytes peak, 24576 router calls, 24576 attention queries.
- `expected_top8_independent_default_topk`: 0.835218 s/boundary mean, 195204608 temporary CUDA bytes peak, 24576 router calls, 24576 attention queries.
- `sampled_next_causal_default_topk`: 0.116239 s/boundary mean, 29459456 temporary CUDA bytes peak, 3072 router calls, 24576 attention queries.
- `sampled_next_independent_default_topk`: 0.840174 s/boundary mean, 195166208 temporary CUDA bytes peak, 24576 router calls, 24576 attention queries.
- `sampled_next_independent_zero`: 0.807095 s/boundary mean, 195133440 temporary CUDA bytes peak, 24576 router calls, 24576 attention queries.

## Expected top-8 auxiliary

The protocol-authorized expected next-token embedding was run as an auxiliary ablation. It did not enter the frozen four-variant ranking.

Measured mean probe latency: 0.835218 s/boundary.
Relative to previous route: hit -0.075938, mass -0.094684.

## Actual closed-loop generation

Not run. The frozen development progress gate failed, so held-out route selection and all actual accuracy generation were forbidden by protocol. Measured task-accuracy rows: 0; identity-materialized rows: 0.

## Measurement boundary

Task accuracy, token identity, NLL/perplexity, route divergence, probe cost, and generation runtime are measured where present. Route replay is open-loop. Expert transfer bytes are simulated. Stall is not estimated because focused v1 froze no latency model. The default-vector artifact has 444 unobserved layer/expert pairs stored as zero and is not a complete expert prior.

The four development rows use checksum-verified authoritative v17 route tensors for natural-route scoring. Fresh cross-process BF16 teacher-forced replay matched the strict router tolerance on 0/4 rows; that drift is recorded but is not used as the scoring target. Same-process mechanism smoke is the native cache/RNG/attention/RoPE/router invariant evidence.
