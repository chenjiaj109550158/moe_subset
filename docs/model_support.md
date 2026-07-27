# Model Support

| Model / adapter | Natural + trace parity | Routed/shared semantics | Closed-loop policy support | Trained gate | Runtime status |
|---|---|---|---|---|---|
| Deterministic tiny MoE (`tiny`) | Full deterministic reference | 2 layers × 4 routed, top-2 | Natural/lossless/substitution, full M3–M11 | M11 primary complete | PyTorch/CUDA batch-1 prototype |
| Tiny random Mixtral (`hf_mixtral`) | Native-reference optional test | 2 × 8 routed, top-2 | Natural and common tuple-hook policies | Structural only | No production offload |
| OLMoE-1B-7B-0125 (`hf_olmoe`) | Exact checkpoint logits/routes | 16 × 64 routed, top-8, no shared | 64-token natural/lossless/hard/previous/static | **STOP/PIVOT** | Simulated transfer only |
| Qwen1.5-MoE-A2.7B (`hf_qwen2_moe`) | Exact checkpoint logits/routes | 24 × 60 routed, top-4 + 1 shared/layer | Same common policies | **STOP/PIVOT** | Simulated transfer only |
| Mixtral-8x7B-v0.1 (`hf_mixtral`) | Exact checkpoint logits/routes | 32 × 8 routed, top-2, renormalized | Same; two-A100 full precision | **STOP/PIVOT** | Simulated transfer only |
| DeepSeek-V2-Lite-Chat (`hf_deepseek_v2`) | Exact checkpoint logits/routes with pinned reversible shims | MoE layers 1–26, 64 routed + 2 shared, top-6 | Same common policies | **STOP/PIVOT** | Simulated transfer only |
| gpt-oss-20b (`hf_gpt_oss`) | Direct native router passes; fused forward trace fails | 24 × 32 top-4, native MXFP4, no shared | Not admitted to common trace/closed loop | **STOP/PIVOT**, separate MXFP4 tier | Incompatible trace path |

## Pinned references

- OLMoE: `9b0c1aa87e34a20052389dce1f0cf01da783f654`.
- Qwen: `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`.
- Mixtral: `fc7ac94680e38d7348cfa806e51218e6273104b0`.
- DeepSeek: `85864749cd611b4353ce1decdb286193298f64c7`; `trust_remote_code=True` is allowed only here.
- gpt-oss: `6cee5e81ee83917806bbde320786a8fb61efebee`; full-router mass is invalid for its MXFP4 tier.
- Datasets: WikiText-2 `b08601e04326c79dfdd32d625aee71d232d685c3`; GSM8K `740312add88f781978c0658806c59bc2815b9866`.

Expert IDs are always layer-scoped. Shared experts are always-active metadata and
are never counted as routed residency. The trained suite requires the `hf` extra,
local pinned cache assets, and two A100s for the validated Mixtral placement.
M11's tiny suite remains the only real-transfer runtime; every trained-model
transfer/stall number is simulated, while logits, routes, quality, and memory are
actual checkpoint measurements. No trained model is authorized for predictor
training or production runtime work after the v2 STOP/PIVOT result.
