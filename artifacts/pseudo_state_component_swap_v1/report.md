# Pseudo state component swap v1

This is a non-deployable mechanism diagnostic on four frozen Qwen/GSM8K development rows at H=8, B=32. Every variant uses exact saved future tokens and one native causal pseudo traversal; the future component capture is a separate full-expert oracle pass.

| Variant | Route hit | Selected mass | Transfer reduction | Pseudo probe s |
|---|---:|---:|---:|---:|
| exact_future_attention_and_moe_oracle | 0.832560 | 0.840968 | 0.619141 | 0.1959 |
| exact_future_attention_output_oracle | 0.788167 | 0.807961 | 0.567362 | 0.1958 |
| exact_future_hidden_after_layer_15_oracle | 0.808472 | 0.825753 | 0.591502 | 0.1914 |
| exact_future_hidden_after_layer_23_oracle | 0.806498 | 0.824311 | 0.589162 | 0.1890 |
| exact_future_hidden_after_layer_31_oracle | 0.803497 | 0.821541 | 0.585398 | 0.1902 |
| exact_future_hidden_after_layer_39_oracle | 0.802500 | 0.821165 | 0.583832 | 0.1888 |
| exact_future_hidden_after_layer_7_oracle | 0.810140 | 0.826830 | 0.593007 | 0.1906 |
| exact_future_moe_residual_oracle | 0.832560 | 0.840968 | 0.619141 | 0.1979 |
| exact_future_unpatched | 0.799276 | 0.818702 | 0.580048 | 0.1929 |

## Component gains over unpatched

- exact_future_attention_and_moe_oracle: route-hit +0.033285; selected-mass +0.022266; paired 95% intervals [0.023986816406249972, 0.042378743489583315] and [0.014760800193915097, 0.029363802813778872].
- exact_future_attention_output_oracle: route-hit -0.011108; selected-mass -0.010741; paired 95% intervals [-0.014648437500000056, -0.00718180338541663] and [-0.013503321279705632, -0.008230493814447792].
- exact_future_hidden_after_layer_15_oracle: route-hit +0.009196; selected-mass +0.007051; paired 95% intervals [0.00657145182291663, 0.010884602864583315] and [0.004069566662755819, 0.009122022471109659].
- exact_future_hidden_after_layer_23_oracle: route-hit +0.007222; selected-mass +0.005609; paired 95% intervals [0.004872639973958343, 0.009033203125] and [0.0027633870087162005, 0.0076231442728100784].
- exact_future_hidden_after_layer_31_oracle: route-hit +0.004222; selected-mass +0.002839; paired 95% intervals [0.0024210611979166297, 0.006022135416666685] and [0.0011870971126380359, 0.004490533111999762].
- exact_future_hidden_after_layer_39_oracle: route-hit +0.003225; selected-mass +0.002463; paired 95% intervals [0.0019327799479166297, 0.0045166015625000555] and [0.0012811981720384225, 0.0036444805334250874].
- exact_future_hidden_after_layer_7_oracle: route-hit +0.010864; selected-mass +0.008128; paired 95% intervals [0.00773111979166663, 0.01399739583333337] and [0.005096738794784728, 0.011159434553131842].
- exact_future_moe_residual_oracle: route-hit +0.033285; selected-mass +0.022266; paired 95% intervals [0.023986816406249972, 0.042378743489583315] and [0.014760800193915097, 0.029363802813778872].

## Attribution

Frozen classification: **moe_residual_dominant**. Focused decision: **NARROW**. Latest hidden checkpoint recovering the frozen threshold: `None`.

Unpatched mean centered router-logit cosine is 0.962980; mean natural top-8 overlap is 0.831085; mean B32 oracle Jaccard is 0.447699.

Next predeclared calibration-free family: `mass_preserving_previous_subset_residual`. It was not executed here; a separate frozen protocol is required before its first model output.

## Integrity and evidence boundary

All cache/RNG/component/information audits pass: `True`. Validated atomic rows: 36; checksum resume: `True`; failed markers preserved: 0.

Route/state alignment, probe latency, capture latency, and memory are measured. Transfer is simulated. Task accuracy, free generation, exact-token identity, NLL, closed-loop runtime, and runtime speedup were not measured. Component swaps and exact future tokens are diagnostic oracle inputs, not deployable pseudo embeddings.
