# Model Support

| Model / adapter | Natural forward | Analysis capture | M7 shadow probes | M9 static/adaptive | Full M8 simulator | Real offload | M11 primary suite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic tiny MoE (`tiny`) | Yes | Full M4/M5 set | Full reference set | Yes, full M8 custom-plan simulation | Yes, trace-driven simulated | Yes, PyTorch/CUDA batch-1 prototype | Yes, complete |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` (`hf_mixtral`) | Yes | Partial; no post-RoPE query | Unsupported | Unsupported | No | No | No; optional adapter tests only |

## Pinned M1 reference

- Model revision: `ccb12fe2fc142cb752085506c3db22572290e90c`.
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, validation split.
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
- `trust_remote_code` is disabled.
- The model is randomly initialized test data and is used only for structural and exactness testing, not quality claims.
- External tests require the `hf` extra and `PSEUDOROUTE_RUN_EXTERNAL=1`; they skip by default in CI.

M3 constrained execution and M6–M10 integration apply only to the deterministic tiny adapter. Mixtral remains natural-routing and trace-only. M8 outputs are simulated; M10 metrics are measured on real CUDA transfers but are not production-model claims.
