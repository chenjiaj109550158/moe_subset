# Trained-Model Oracle Preflight and Download Plan

Recorded 2026-07-27 UTC. This plan was written before downloading any trained
model or new dataset.

## Decision

Proceed with `allenai/OLMoE-1B-7B-0125` at immutable revision
`9b0c1aa87e34a20052389dce1f0cf01da783f654`.

This is a trained, English, decoder-only sparse MoE with approximately 7B total
parameters, 1.3B active parameters per token, 16 MoE layers, 64 layer-local
experts, and native top-8 routing. It is language-capable and is the smallest
well-supported trained MoE identified for this environment. The checkpoint and
code are Apache-2.0 licensed. Transformers has a native `OlmoeForCausalLM`
implementation and exposes native router logits, but this repository needs a
new OLMoE implementation of the existing common adapter contract.

The model repository contains 27,680,932,269 bytes in total. Its six
safetensors weight shards contain 27,677,042,464 bytes (about 27.68 GB); the
remainder is tokenizer, index, configuration, model-card, and image metadata.
The pre-download metadata query reported the following immutable weight
SHA-256 values:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00006.safetensors` | 4,993,992,240 | `df5a700fa91fd94e9d1a7ae523c5ae055f5879778a02ea19758edd37089da1ca` |
| `model-00002-of-00006.safetensors` | 4,992,966,080 | `5f299f21e6de71f5334e937fd51a083bab8cffe55682488219e4082d9558fdba` |
| `model-00003-of-00006.safetensors` | 4,992,966,080 | `4491eb552fd917a6777260f75c88e714f3174fbba7dd69d44b97c4124ddcd56b` |
| `model-00004-of-00006.safetensors` | 4,992,966,416 | `7aeea413752cd17a8e7c343aa436594a8158d279817e4993aa5df4b7c7cfdf22` |
| `model-00005-of-00006.safetensors` | 4,992,966,680 | `18c7f749af753359e10c33a9c5deec688b7155f7c01052937806cd351c52cc38` |
| `model-00006-of-00006.safetensors` | 2,711,184,968 | `299009ba118f5e918de33573a316ad3d3b9236651986fb1a14dc894f55269e67` |

## Environment

- Host: Linux 6.8.0-100-generic, x86-64.
- CPU RAM: 235 GiB total, 229 GiB available; no swap.
- GPUs: 2 × NVIDIA A100-SXM4-80GB, each 81,920 MiB total and 81,152 MiB
  free during preflight.
- Driver: 580.126.09.
- Python: 3.14.6.
- PyTorch: 2.13.0+cu130; CUDA runtime 13.0.
- Transformers: 5.14.1; Tokenizers: 0.22.2; Safetensors: 0.8.0.
- Disk: 495 GB total, 413 GB available.
- Project Hugging Face cache: 33 MB, containing only the pinned tiny-random
  Mixtral and WikiText-2 assets. No trained-model weights are cached.
- Existing generated project artifacts: 21 MB.

The default load will use one A100 and the checkpoint's stored dtype. Based on
the published 27.68 GB weight payload, allow up to 35 GB GPU memory for model
weights and up to 15 GB for activations, attention/KV state, router capture, and
allocator reserve. CPU loading/staging allowance is 40 GB. This remains below
one 80 GB GPU and well below available host RAM.

## Dataset and sampling plan

Use two pinned tasks/domains and split only by complete source sample:

1. Language modeling: `Salesforce/wikitext`, `wikitext-2-raw-v1`, validation,
   revision `b08601e04326c79dfdd32d625aee71d232d685c3`, licenses
   CC-BY-SA-3.0 and GFDL. Its required validation parquet is 657,209 bytes and
   already exists in the project cache.
2. Mathematical reasoning: `openai/gsm8k`, `main`, test, revision
   `740312add88f781978c0658806c59bc2815b9866`, MIT license. The required test
   parquet is 419,088 bytes; the complete repository is 5,900,352 bytes.

Sampling, seeds, prompt rendering, context boundaries, tokenizer fingerprint,
and actual row IDs must be saved in the resolved run configuration and source
manifest. The suite will start with a small smoke sample and use natural
teacher-forced text tokens. Final evidence must include multiple independent
documents from each domain and cannot use random token IDs or random routing.

## Storage gate and cache policy

All external assets go under `.cache/pseudoroute/huggingface`; run outputs go
under ignored `artifacts/`. Never commit either.

Conservative storage reservation:

| Item | Planned bytes |
|---|---:|
| Model, tokenizer, and metadata download/cache | 28 GB |
| New GSM8K dataset cache | 10 MB |
| Natural router traces (raw logits plus routes and metadata) | 10 GB |
| Closed-loop and raw tabular results | 5 GB |
| Temporary download/run overhead | 10 GB |
| Mandatory free-space headroom after completion | 100 GB |
| Total required before download | 153.01 GB |

The preflight found 413 GB free, exceeding the requirement by about 260 GB.
Content-addressed cache files must be reused only at the pinned revision.
Downloaded weight shards must match the published SHA-256 values above; run
manifests must also record local file checksums and tokenizer fingerprints.

## Fallback

The architecture-supported fallback is
`mistralai/Mixtral-8x7B-v0.1` at
`fc7ac94680e38d7348cfa806e51218e6273104b0`, Apache-2.0. Its 19
safetensors shards total about 94 GB, and the repository totals about 190 GB
because it also carries an alternate consolidated format. It fits host RAM and
the two-GPU machine only with sharding/offload changes that the current
single-device adapter does not implement. Use it only if OLMoE native-semantic
adapter validation fails for a model-specific reason; do not silently switch
to quantized weights or the random tiny Mixtral.

## Cross-architecture expansion

Authorized 2026-07-27 after the initial OLMoE smoke. Staged additions:

- `Qwen/Qwen1.5-MoE-A2.7B` at `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`: 28,639,598,661 bytes.
- `openai/gpt-oss-20b` at `6cee5e81ee83917806bbde320786a8fb61efebee`: about 13.8 GB excluding duplicate `original/model.safetensors`. The Hub CLI also fetched `metal/model.bin`, making the observed cache about 26 GiB; it is retained pending explicit cleanup approval.
- `mistralai/Mixtral-8x7B-v0.1` at `fc7ac94680e38d7348cfa806e51218e6273104b0`: 93,408,096,711 bytes excluding all alternate `consolidated.*` weights.

Qwen uses its repository-specific license (`other` in Hub metadata); gpt-oss and Mixtral are Apache-2.0. gpt-oss uses native MXFP4 MoE weights, so its results remain separate from floating-point residency results. The planned primary payload was 135.9 GB; the retained gpt-oss Metal duplicate raised observed use by about 13.8 GB. After all downloads, 248 GB remains free, preserving 98 GB beyond the 30 GB trace/result reserve, 20 GB temporary reserve, and 100 GB mandatory headroom. All 3 gpt-oss and 19 Mixtral Transformers shards match Hub-published SHA-256 metadata; all 8 Qwen shards were locally hashed and will be matched to published metadata before its smoke run. Every architecture must pass native route regression and a checksum-valid smoke before the common oracle grid. DeepSeek-MoE-16B is deferred because its reference path requires `trust_remote_code=True` and a non-standard model license.

## Related-work model audit and trusted remote code

Audited 2026-07-27. The 20-model local-routing-consistency study includes OLMoE-1B-7B-0125, Qwen1.5-MoE-A2.7B, Mixtral-8x7B, DeepSeek-V2-Lite, DeepSeekMoE, Qwen3-30B-A3B, LLaMA-MoE-v2, and other families. CommitMoE evaluates Mixtral-8x7B-Instruct, Qwen1.5-MoE-Chat, and DeepSeek-V2-Lite-Chat. The current matrix therefore already covers three architecture families, and its most direct missing related-work control is DeepSeek-V2-Lite-Chat.

Add `deepseek-ai/DeepSeek-V2-Lite-Chat` at immutable revision `85864749cd611b4353ce1decdb286193298f64c7`. The repository totals 31,418,838,089 bytes, including four safetensors shards totaling 31,413,626,576 bytes and pinned custom configuration, modeling, and tokenizer code. Hub metadata labels its license `other`; retain and comply with the included DeepSeek model license. Download code first, inspect it locally, then download weights. Loading may use `trust_remote_code=True` only at this pinned revision. The architecture adds 27 layers, 64 routed experts plus 2 shared experts, and top-6 routed activation. With 248 GB currently free, the completed cache leaves about 216 GB, preserving the 150 GB combined artifact/temporary/mandatory-headroom reserve.

Qwen3-30B-A3B (`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, 61.08 GB) is deferred despite appearing in related work because it is architecturally adjacent to cached Qwen1.5 and would leave little surplus beyond the current reserve. LLaMA-MoE-v2 is a smaller high-consistency follow-up candidate, but is less directly comparable to the closed-loop CommitMoE model set.

## DeepSeek download and smoke outcome

Completed 2026-07-27. All four safetensors shards matched Hub-published SHA-256 metadata. The cache occupies 30 GiB and 218 GB disk remained free. Static review found no subprocess, socket/HTTP, dynamic execution, arbitrary file access, or custom compilation. Direct loading under Transformers 5.14.1 initially failed because the pinned 4.x-era code imports the removed `is_torch_fx_available` helper. A minimal in-memory compatibility shim restoring only that query allowed local-only loading without modifying upstream code. A 10-token natural-forward smoke captured 26 MoE layers, native top-6 gate outputs, finite language-model logits, and 32,646,750,720 peak CUDA bytes on one A100.

This validates fit and basic native execution only. The adapter must encode the compatibility shim explicitly, regression-test router logits/scores/IDs/weights and shared-expert semantics, and retain the initial import failure in its validation report before DeepSeek enters the oracle sweep.

## Go/no-download gate

The environment passes the resource gate for OLMoE. Download is authorized only
after this file is present in Git status. Adapter work must preserve native
OLMoE semantics: bias-free linear router logits, float32 softmax, native top-8,
and the checkpoint's `norm_topk_prob` setting. No shared expert is present in
this architecture. All expert identities remain layer-scoped.

## Final v2 execution and decision

Completed 2026-07-27 UTC. The preflight/download history above is retained; the
final common runner is `pseudoroute trained-suite` with
`configs/trained/oracle_gate_v2.yaml`, and its checksum-valid output is
`artifacts/trained_gate/suite_v2_final/`. The earlier v1 run is protocol
development only: its GSM8K row 0 prompt was truncated at 48 tokens and its
12-token continuation did not reach an answer. V2 kept every model, dataset
revision, row set, horizon, budget, oracle method, bootstrap count, and decision
threshold unchanged, selected complete GSM8K row 500 for closed loop, and fixed
decoding at 64 new tokens before inspecting the final v2 aggregates.

### Adapter and memory validation

| Model | Native semantics preserved | MoE structure | Peak CPU RSS | Peak CUDA allocated |
|---|---|---:|---:|---:|
| OLMoE | float32 softmax, top-8, not renormalized | 16 × 64 routed | 49.95 GiB | 25.83 / 0.00 GiB |
| Qwen1.5-MoE | float32 softmax, top-4, not renormalized, sigmoid-gated shared expert | 24 × 60 routed + 1 shared/layer | 48.80 GiB | 26.75 / 0.00 GiB |
| Mixtral | float32 softmax, top-2, renormalized | 32 × 8 routed | 88.99 GiB | 45.25 / 45.25 GiB |
| DeepSeek-V2-Lite | float32 softmax, top-6, native scaling, 2 always-active shared experts | layers 1–26 × 64 routed | 18.72 GiB | 30.41 / 0.01 GiB |
| gpt-oss | biased logits → top-4 → selected softmax; MXFP4 physical storage | 24 × 32 routed | inspection only | separate failed tier |

For all four floating-point models, base/traced/natural-policy logits are
bit-exact, two-step natural-policy generation matches the untraced path, router
logits/pre-top-k scores/IDs/weights reproduce native outputs, hard subsets emit
only allowed layer-scoped IDs, and shared experts remain outside routed
residency. Mixtral uses unquantized two-A100 placement. DeepSeek loads remote code
only at `85864749cd611b4353ce1decdb286193298f64c7`; the removed
`is_torch_fx_available` query and unbounded-cache `get_usable_length` are restored
only inside reversible contexts. The direct import and legacy-cache failures are
retained under `artifacts/trained_gate/adapter_validation/deepseek/`; cached code
was not edited.

gpt-oss actual inspection records `Mxfp4GptOssExperts`, 13,253,760 physical
bytes per expert, finite direct native routing, and three checkpoint shard
checksums. Its fused MXFP4 forward bypasses the Python router modules, so the
common trace hook captures zero layers and fails explicitly. Full-router-mass
metrics are invalid for this tier. The cached 13,750,886,400-byte
`metal/model.bin` blob was not removed.

### Trace and oracle protocol

Each floating-point tokenizer traces WikiText-2 validation rows 3, 49, 80, 105
and GSM8K test rows 0, 100, 500, 1000 at the pinned revisions above. Each shard
persists row/domain/prompt provenance, token and layer positions, native logits
and pre-top-k scores, selected IDs/weights, routed/shared metadata, model and
tokenizer fingerprints, and checksums. The first two rows per domain calibrate
static frequency; the last two are evaluation samples.

The final tables contain 857,728 evaluation windows across horizons 1/2/4/8/16,
native-top-k multiples and absolute budgets, binary/selected/full-mass oracles,
previous-route/static/on-demand/LRU/LFU baselines, and all layers/domains. Global
summaries contain 232,180 metric rows; the additional 1,705,301 rows stratify by
context-position and router-margin buckets. Both include mean, median, P05, P95,
worst and 95% sample-bootstrap CIs using 1,000 draws. Concrete worst cases retain
the source sample, boundary, layer, natural route, selected subset, and metrics.

The gate additionally requires a budget below all routed experts, mean hit at
least 0.90, selected mass at least 0.95, P05 and worst hit at least 0.80, at least
30% simulated transfer reduction, and at least 0.05 selected-mass improvement
over previous/static baselines. Qualifying final open-loop points are:

- OLMoE: WikiText, H=8, B=32; hit 0.9821, selected mass 0.9918, worst 0.8125, transfer reduction 0.50.
- Qwen: GSM8K and WikiText, H=16, B=32; hit 0.9697/0.9803, selected mass 0.9903/0.9929, worst 0.8125/0.8438, transfer reduction 0.50.
- Mixtral: none after the joint P05 and worst-window requirement.
- DeepSeek: GSM8K and WikiText, H=8, B=32; hit 0.9907/0.9980, selected mass 0.9963/0.9993, worst 0.8542/0.8958, transfer reduction 0.3333.

### Closed-loop quality and terminal gate

Closed loop uses complete WikiText row 3 and GSM8K row 500 prompts, batch-1
deterministic greedy decoding, 64 new tokens, H=4, and B=2× native top-k. It
compares natural, lossless oracle residency, hard natural-trajectory oracle
commitment, previous-route commitment, and static-frequency commitment. Natural
and lossless token sequences/NLL are exactly equal for every checkpoint; this is
correctness, not a quality improvement.

| Model | Hard token agreement | Mean hard relative PPL increase | Mean lossless fallback | Natural GSM answer / truth | Hard GSM answer |
|---|---:|---:|---:|---:|---:|
| OLMoE | 0.2031 | 27,951.39 | 0.4797 | 17 / 16 | 24 |
| Qwen | 0.1953 | 82,580.99 | 0.8022 | 43 / 16 | 43 |
| Mixtral | 0.2656 | 221,159.52 | 0.2882 | 2 / 16 | 4 |
| DeepSeek | 0.0625 | 55,784.02 | 0.7070 | 7 / 16 | 15 |

No hard path meets the maximum 0.05 relative-perplexity-increase gate, no
lossless path meets the maximum 0.10 fallback gate, and no GSM continuation is
answer-correct. Therefore OLMoE, Qwen, Mixtral, DeepSeek, gpt-oss, and the overall
stage are **STOP/PIVOT**. Open-loop predictability does not justify training a
predictor or implementing production offload. Transfer and stall values use a
25 GiB/s plus 10 µs/load simulation and do not include a measured planning
runtime; this uncertainty strengthens rather than relaxes the negative decision.
The result is limited to the exact revisions, tokenizers, sampled rows, hardware,
checkpoint precision, budgets, and decoding regime above.
