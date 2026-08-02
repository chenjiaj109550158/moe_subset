# PseudoRoute-MoE

Research framework for position-conditioned pseudo routing states and
memory-budgeted mixture-of-experts inference.

The repository has completed **M11 at deterministic primary-suite scope** and the
subsequent **multi-model trained-MoE oracle feasibility gate**. It contains a
deterministic, accelerator-aware tiny causal MoE model, typed configuration
loading, route tracing, reproducible run-directory utilities, and the complete
normative specification in `docs/spec/`. The trained gate tested four
floating-point checkpoints plus a separate native-MXFP4 tier and concluded
**STOP/PIVOT** under the pinned v2 protocol. Simulated transfer/stall estimates,
actual model-quality results, and measured adapter memory are labeled separately.

## Completed benchmark subset and focused pseudo pilot

The benchmark-aligned pivot reuses the checksum-valid v17 prompts, tokens, and
evaluators without rerunning vanilla. Its frozen base config is
`configs/benchmark/benchmark_subset_oracle_v1.yaml`. After preserving 808 actual
hard rows from the superseded six-task scope, the explicit 2026-07-31 amendment
`configs/benchmark/benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2.yaml` limits full
accuracy execution to all 1,319 GPT-OSS GSM8K rows at `(H=1,B=4)`. It completed
1,242/1,319 for both vanilla and true hard closed loop, with exact-token
agreement 1.0 and no paired gains or losses. Because `H=1,B=4` equals native
top-k, this is a task-scoped `NARROW` natural-route ceiling, not multi-token
subset evidence or runtime speedup. Existing
HumanEval/MBPP+/previous-route rows and interruption markers remain provenance
and are excluded from the scoped aggregate.

The separately frozen Qwen3-30B-A3B/GSM8K pseudo-embedding pilot at
`(H=8,B=32)` is also complete with **STOP/PIVOT**. Native mechanism smoke passed,
but all four mandatory training-free variants underperformed previous-route on
the four predeclared development traces. The primary sampled-next-token/default-
vector variant reached route hit 0.533015 and selected mass 0.534416 versus
0.609385 and 0.629428 for previous-route. The zero-contribution ablation was the
best mandatory variant at 0.566499/0.575402, still below previous-route. The
frozen development gate therefore prohibited held-out route evaluation and
actual closed-loop accuracy; no task-accuracy or identity-materialized pseudo
rows were produced. See the [focused report](artifacts/pseudo_embedding_qwen_gsm8k_v1/report.md)
and [protocol](docs/pseudo_embedding_qwen_gsm8k_v1_protocol.md).

A separately versioned, strictly calibration-free follow-up then analyzed the
failure without relabeling v1. Native interventions found that repeated sampled
content changes much less than natural routes, recent token IDs do not repair
the hidden direction, and causal rollout with zero MoE residual is harmful.
Same-request prompt/generated route history plus independent sampled-token
pseudo scores passed four-row development, but failed the committed eight-row
held-out gate: route hit/selected mass were 0.658353/0.680584 versus
0.617671/0.637396 for previous route, only +0.040682/+0.043188. Oracle-gap
recovery was 11.7%/12.6%, below 25%. The result is **STOP/PIVOT**; no accuracy or
actual closed-loop rows were run. See the
[calibration-free synthesis](docs/pseudo_embedding_calibration_free_synthesis_v1.md)
and [held-out report](artifacts/pseudo_embedding_calibration_free_prompt_route_held_out_v1/report.md).

The next separately frozen calibration-free experiment tested the user's
window-level residual hypothesis: full-expert prefill supplies the last eight
prompt MoE mixture outputs, then each policy reuses the eight mixture outputs it
actually executed under its preceding hard subset. Position-aligned residuals
were clearly better than zero, last-repeated, or mean-repeated residuals. Seven
analytic pseudo/history constructions were then evaluated on each policy's own
hard-subset state. The best reached route hit/selected mass
0.639706/0.649761 versus 0.623634/0.634308 for previous-route, gains of only
+0.016072/+0.015453 and about 4.9%/4.6% oracle-gap recovery. Both frozen +0.05
requirements failed, so the result is **STOP/PIVOT** and the new held-out and
actual accuracy stages were not run. See the
[residual-window protocol](docs/pseudo_embedding_qwen_gsm8k_residual_window_v1.md)
and [report](artifacts/pseudo_embedding_qwen_gsm8k_residual_window_v1/report.md).

The subsequent composition analysis changed the residual source in the crucial
way: each pseudo forward now executes Qwen's native experts and obtains a fresh
MoE residual from the pseudo hidden state. Boundary zero can execute native
top-8 after full-expert prefill; every later boundary executes only the
preceding realized per-layer B=32 subset. The calibration-free
`self_greedy_causal` rollout was selected on four development rows at
0.788767/0.805782 route hit/selected mass. On eight disjoint held-out route rows
it reached 0.765119/0.784870, versus 0.611168/0.621438 for previous route and
0.669112/0.682921 for sampled-token repeat. All cache/RNG/information audits
passed. Deterministic top-2 and top-4 particle branches added at most
0.000621 route hit over greedy while roughly doubling or quadrupling probe
latency. This is a route-analysis **NARROW**, not a full pilot GO: the protocol
did not compute held-out oracle-gap recovery and authorized no task accuracy,
free generation, exact-token, or runtime-speedup claim. See the
[composition protocol](docs/pseudo_executed_embedding_composition_v1.md) and
[route report](artifacts/pseudo_executed_embedding_composition_v1/report.md).

A final calibration-free development experiment kept that same one-forward
mechanism but tested two state extrapolations. Each boundary still performs one
native causal eight-token pseudo traversal and obtains a fresh MoE residual by
executing the previous realized B=32 subset; the only additions were either a
last-two-token router-input velocity or a last-two-token MoE-output velocity.
Both fixed linear corrections regressed the uncorrected recent-sequence baseline:
hidden velocity scored 0.648031/0.661326 and residual velocity scored
0.676310/0.691583 route hit/selected mass, versus 0.700267/0.714061. Both paired
95% intervals were wholly negative. The result is **STOP/PIVOT** at four-row
development scope, with no held-out route or accuracy execution. See the
[state-correction protocol](docs/pseudo_one_forward_state_correction_v1.md) and
[report](artifacts/pseudo_one_forward_state_correction_v1/report.md).

The predeclared protected-anchor v2 follow-up then set the known sampled-token
anchor correction to zero and tested horizon damping, a 25%-of-state norm cap,
and an anchor-one top-8 subset core. Protection plus residual damping changed the
large v1 regression into a small gain on all four rows: 0.703389/0.716302 versus
0.700267/0.714061, or +0.003123/+0.002240 route hit/selected mass. This remained
far below the frozen +0.02 signal. The norm cap reduced both perturbation and
gain, while replacing the first-four/history selector with only anchor-one plus
later-anchor utility regressed to 0.657410/0.655205. The development result is
therefore **STOP/PIVOT**, with no held-out or accuracy run. See the
[protected-anchor protocol](docs/pseudo_one_forward_protected_anchor_v2.md) and
[report](artifacts/pseudo_one_forward_protected_anchor_v2/report.md).

The next frozen one-forward test replaced temporal extrapolation with current-
request token-aligned state retrieval. It captured native router inputs and
exact executed MoE outputs for prompt and already-realized policy tokens,
retrieved the most recent matching token state for each pseudo anchor, and
optionally used native input-embedding cosine for a missing match. Anchors 2–8
were always exact, but all five residual/router-input mixtures regressed the
uncorrected 0.700267/0.714061 reference. The best selected-mass candidate reached
0.699025/0.712238; exact residual addition reached 0.699565/0.712088. Measured
retrieval cost was tiny and simulated transfer stayed above 45%, but the frozen
+0.02 route signal failed. This is a four-row development **STOP/PIVOT** with no
held-out route or accuracy execution. See the
[token-aligned protocol](docs/pseudo_one_forward_token_aligned_retrieval_v1.md)
and [report](artifacts/pseudo_one_forward_token_aligned_retrieval_v1/report.md).

## Trained-model oracle gate

The checksum-valid final suite is `artifacts/trained_gate/suite_v2_final/` and is
recreated by the versioned `configs/trained/oracle_gate_v2.yaml`. It uses four
independent WikiText-2 validation rows and four GSM8K test rows per tokenizer,
horizons 1/2/4/8/16, absolute and native-top-k-multiple budgets, eight oracle or
cache methods, and 1,000-sample bootstrap confidence intervals. Closed-loop
quality uses complete prompts and 64-token deterministic greedy continuations.

| Checkpoint | Final decision | Decisive result |
|---|---|---|
| OLMoE-1B-7B-0125 | STOP/PIVOT | One WikiText open-loop point passes, but hard quality and 47.97% mean lossless fallback fail |
| Qwen1.5-MoE-A2.7B | STOP/PIVOT | Both-domain B=32/H=16 open-loop points pass, but hard quality and 80.22% fallback fail |
| Mixtral-8x7B-v0.1 | STOP/PIVOT | No point jointly passes the P05 and worst-window open-loop gate; hard quality also fails |
| DeepSeek-V2-Lite-Chat | STOP/PIVOT | Both-domain B=32/H=8 points pass, but hard quality and 70.70% fallback fail |
| gpt-oss-20b (MXFP4) | STOP/PIVOT | Direct native router inspection passes; fused MXFP4 forward bypasses the common trace hook |

The overall result is scoped to these revisions, two datasets, sampled rows,
64-token batch-1 greedy decoding, checkpoint precision, budgets, and two A100s.
It does not authorize predictor training or production runtime optimization. See
[the trained-model plan](docs/trained_model_plan.md) and
[decision log](docs/decisions.md).

## Development

```bash
python -m pip install -e '.[dev,hf]'
pseudoroute inspect-model --config configs/model/tiny_moe.yaml
ruff check .
mypy
pytest
PSEUDOROUTE_RUN_EXTERNAL=1 pytest tests/integration/test_hf_mixtral.py
pseudoroute collect-traces --config configs/model/hf_tiny_mixtral.yaml --output-dir artifacts/traces/hf_tiny_mixtral
pseudoroute oracle-sweep --config configs/experiment/oracle_tiny.yaml --output-dir artifacts/oracle/tiny_m2
pseudoroute closed-loop-eval --config configs/experiment/closed_loop_tiny.yaml --output-dir artifacts/closed_loop/tiny_m3_final
pseudoroute dapq-factorial --config configs/experiment/dapq_factorial_tiny.yaml --output-dir artifacts/factorial/tiny_m4_final
pseudoroute analyze-router --config configs/experiment/analyze_router_tiny.yaml --output-dir artifacts/router/tiny_m5_final
pseudoroute build-default-vectors --config configs/experiment/default_vectors_tiny.yaml --output-dir artifacts/m7_default_vectors
pseudoroute evaluate-probe --config configs/experiment/evaluate_shadow_probe_tiny.yaml --output-dir artifacts/m7_shadow_probe_pre_sample_final_v2
pseudoroute simulate-offload --config configs/experiment/static_adaptive_tiny.yaml --output-dir artifacts/m9_static_adaptive_tiny_m8_final_v2
pseudoroute simulate-offload --config configs/experiment/simulate_offload_tiny.yaml --output-dir artifacts/m8_simulate_offload_tiny_definitive
pseudoroute benchmark-offload --config configs/experiment/benchmark_offload_tiny.yaml --output-dir artifacts/m10_benchmark_offload_tiny_definitive_v2
pseudoroute reproduce --suite primary --config-root configs/paper --output-dir artifacts/reproduction/m11_primary_definitive_v2
pseudoroute aggregate-results --input-root artifacts/reproduction/m11_primary_definitive_v2/runs --output-dir artifacts/reproduction/m11_primary_definitive_v2/paper
pseudoroute trained-suite --config configs/trained/oracle_gate_v2.yaml --output-dir artifacts/trained_gate/suite_v2_final
```

See [docs/reproducibility.md](docs/reproducibility.md) for manifest guarantees, individual benchmark runners, failure retention, and hardware requirements.

The inspection result is labeled with its information regime. M0 performs only
natural inference and does not expose future tokens to an online API. The
example config uses `device: auto`, which selects CUDA when available and falls
back to MPS or CPU; set an explicit device when reproducibility requires it.

Codex agents may download milestone-relevant models and datasets when disk space
is sufficient. They must estimate required space first, retain room for generated
traces/artifacts, pin revisions where possible, and record source fingerprints.
Large downloads are cached outside version control and are never required by the
default unit-test suite.
