# Configuration and Data Schemas

## 1. Configuration principles

- Every experiment is described by a resolved YAML configuration.
- Device selection is explicit. `auto` may resolve to CUDA when available and CPU otherwise, but the resolved run manifest must record the actual device.
- Defaults may be composed, but the resolved file is saved verbatim.
- Unknown keys must be rejected.
- Config validation happens before loading large models.
- A run ID is derived from the resolved config, git commit, model fingerprint, and dataset fingerprint.
- Secrets and access tokens must never be written into resolved configs.

## 2. Top-level experiment configuration

```yaml
schema_version: 1
experiment:
  name: oracle_olmoe_general
  seed: 1234
  output_root: artifacts/runs
  information_regime: oracle
  resume: true
  overwrite: false

model:
  adapter: olmoe
  model_id: MODEL_ID_TO_VERIFY
  revision: null
  dtype: bfloat16
  device: cuda:0
  trust_remote_code: false
  quantization: null
  max_context_length: 4096
  use_flash_attention: false

data:
  dataset_id: DATASET_ID_TO_VERIFY
  split: validation
  text_field: text
  max_samples: 1000
  max_prompt_tokens: 1024
  max_new_tokens: 128
  shuffle: false
  streaming: false

decode:
  method: greedy
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  batch_size: 1

trace:
  level: router_logits
  storage_dtype: float16
  layers: all
  token_stride: 1
  shard_token_limit: 16384
  store_text: false

window:
  horizons: [1, 2, 4, 8, 16]
  boundary_mode: fixed
  global_boundary: true
  discount_gamma: 1.0

budget:
  mode: ratio_to_active
  ratios: [1.0, 1.5, 2.0, 3.0, 4.0]
  static_fraction: 0.0
  include_shared_experts: false

selector:
  name: oracle_routing_mass
  use_full_router_distribution: false
  load_cost_lambda: 0.0

execution:
  routing_policy: masked_substitution
  miss_policy: substitute
  renormalize_weights: true
  emergency_fallback: false

metrics:
  route: true
  quality: true
  performance: false
  bootstrap_samples: 1000

logging:
  level: INFO
  save_per_token: true
  save_per_layer: true
  external_tracker: null
```

## 3. Model adapter configuration

```yaml
adapter: qwen3_moe
model_id: MODEL_ID_TO_VERIFY
revision: null
dtype: bfloat16
device: cuda:0

architecture_overrides:
  moe_layer_indices: null
  num_experts: null
  top_k: null
  router_score_function: auto
  has_shared_expert: auto
  rope_type: auto

capture_points:
  router_input: auto
  post_attention_residual: auto
  pre_rope_query: auto
  post_rope_query: auto

validation:
  logit_atol: 1.0e-5
  logit_rtol: 1.0e-4
  route_exact: true
  generated_tokens_exact: true
```

Overrides exist for debugging only. A run using manual overrides must record them prominently.

## 4. DapQ-style factorial configuration

```yaml
experiment:
  information_regime: offline_teacher_forced

factorial:
  prefix_positions:
    strategy: random_valid
    per_sample: 8
  pseudo_length: 16
  horizons: [1, 2, 4, 8, 16]

  conditions:
    - SC_SP
    - DC_SP
    - SC_DP
    - DC_DP

  different_content:
    variants:
      - random_vocab
      - repeat_current
      - sample_from_prefix
      - fixed_nonsense
    random_seeds: [1, 2, 3, 4]

  different_position:
    variants:
      - reset_zero
      - random_contiguous
      - offset
      - shuffled
    offsets: [-128, -64, -32, 32, 64, 128]

  attention_mask:
    variants:
      - independent
      - causal_pseudo

  capture:
    pre_rope_query: true
    post_rope_query: true
    post_attention_state: true
    router_input: true
    router_logits: true

  context_swap:
    enabled: true
    matching: random_other_sample
```

## 5. Probe configuration

```yaml
probe:
  name: default_vector_shadow_rollout
  information_regime: online_post_sample
  horizons: [1, 4, 8]
  pseudo_content: sampled_next_token
  pseudo_attention: independent
  aggregate:
    target: selected_routing_mass
    gamma: 0.9
    interpolation: interval_weighted

  shadow:
    dense_mlp_mode: exact
    moe_mode: default_vector
    default_vector_definition: moe_residual_contribution
    mixture_mode: pseudo_topk
    track_variance: true

  uncertainty:
    enabled: true
    branches: 4
    selection_score: mean_plus_std
    beta: 0.5

  max_probe_ms: null
```

## 6. Static residency configuration

```yaml
static_residency:
  enabled: true
  fraction: 0.25
  ranking: combined

  combined_weights:
    frequency: 1.0
    routing_mass: 0.0
    quality_sensitivity: 1.0
    downstream_routing: 1.0
    miss_risk: 0.5
    load_cost: 1.0

  calibration_artifact: artifacts/criticality/MODEL/RUN
  per_layer_minimum: 0
```

## 7. Simulator configuration

```yaml
hardware:
  name: simulated_pcie
  hbm_total_bytes: 25769803776
  hbm_reserved_dense_bytes: 12884901888
  hbm_reserved_kv_bytes: 4294967296
  hbm_reserved_workspace_bytes: 2147483648
  cpu_dram_bytes: 137438953472

  transfer:
    fixed_latency_us: 10.0
    h2d_bandwidth_bytes_per_s: 2.4e10
    max_concurrent_transfers: 1
    pinned_memory: true

  compute:
    source: measured_trace
    overlap_rule: separate_stream
    synchronization_penalty_us: 0.0

simulator:
  cache_policy: planned_subset
  double_buffer: false
  record_timeline: true
  timeline_sample_limit: 20
```

## 8. Runtime configuration

```yaml
runtime:
  mode: real_offload
  expert_storage: cpu_pinned
  gpu_slot_count_by_layer: auto_from_budget
  transfer_streams: 1
  synchronous_reference: false
  cuda_graphs: false
  warmup_tokens: 16
  measured_tokens: 128
  repetitions: 5
  record_cuda_events: true
  record_memory_snapshot: true
```

## 9. Trace manifest

```json
{
  "schema_version": 1,
  "trace_id": "sha256-prefix",
  "created_at": "ISO-8601",
  "information_regime": "offline_teacher_forced",
  "model": {
    "model_id": "...",
    "revision": "...",
    "adapter": "...",
    "fingerprint": "..."
  },
  "dataset": {
    "dataset_id": "...",
    "split": "...",
    "fingerprint": "..."
  },
  "trace_level": "router_inputs",
  "storage_dtype": "float16",
  "num_samples": 0,
  "num_tokens": 0,
  "num_moe_layers": 0,
  "num_experts_by_layer": {},
  "top_k_by_layer": {},
  "arrays": {},
  "shards": [],
  "complete": false
}
```

## 10. Recommended trace arrays

Flatten tokens across samples and keep offsets.

### Mandatory

```text
sample_offsets             int64   [num_samples + 1]
token_ids                  int32   [T]
position_ids               int32   [T]
is_prompt                   bool    [T]
router_logits              float16 [T, L_moe, E_max]
router_valid_experts       bool    [L_moe, E_max]
router_topk_ids            int16   [T, L_moe, K_max]
router_topk_weights        float16 [T, L_moe, K_max]
```

For variable expert counts/top-\(k\), pad and use masks.

### Optional

```text
router_inputs              float16 [T, L_moe, H]
post_attention_states      float16 [T, L_moe, H]
pre_rope_queries           float16 [T, L_or_head, ...]
post_rope_queries          float16 [T, L_or_head, ...]
expert_output_norms        float16 [T, L_moe, K_max]
next_token_logits          float16 [T, V]  # usually disabled
```

Storing full vocabulary logits is discouraged. Prefer selected log-probabilities or KL computed online.

## 11. Token metadata table

Recommended Parquet columns:

```text
flat_token_idx: int64
sample_idx: int32
sample_id: string
token_idx_in_sample: int32
absolute_position: int32
token_id: int32
is_prompt: bool
is_decode: bool
task: string
domain: string|null
text_hash: string|null
```

Do not store raw text unless explicitly enabled and licensing/privacy permits it.

## 12. Default-vector artifact

```text
default_vectors/
├── manifest.json
├── means.safetensors
├── variances.safetensors
└── counts.parquet
```

Tensor naming:

```text
layer_{layer_idx}.expert_{expert_idx}.mean
layer_{layer_idx}.expert_{expert_idx}.variance
```

Manifest fields:

- model fingerprint;
- dataset fingerprint;
- exact capture definition;
- normalization/combine stage;
- token count;
- selected-activation count per expert;
- dtype;
- software version.

## 13. Predictor dataset schema

Each row corresponds to a decision boundary, not a random isolated token without grouping.

Metadata:

```text
sample_id
boundary_position
decision_mode
history_length
horizon
layer_idx
split
```

Feature tensors:

```text
current_router_logits
current_router_visible_state
route_history_summary
position_features
next_token_embedding_optional
```

Targets:

```text
future_router_logits
future_router_probs
future_topk_ids
aggregate_count_utility
aggregate_mass_utility
```

Use grouped train/validation/test splits by `sample_id`.

## 14. Probe output artifact

```json
{
  "schema_version": 1,
  "probe_name": "default_vector_shadow_rollout",
  "information_regime": "online_post_sample",
  "sample_id": "...",
  "boundary_position": 128,
  "horizons": [1, 4, 8],
  "layers": {
    "3": {
      "aggregate_utility_path": "...",
      "uncertainty_path": "...",
      "estimated_cost_ms": 0.0
    }
  },
  "metadata": {}
}
```

Large arrays should be stored separately in safetensors or Zarr.

## 15. Subset plan schema

```json
{
  "schema_version": 1,
  "sample_id": "...",
  "boundary_position": 128,
  "decision_mode": "post_sample",
  "information_regime": "online_post_sample",
  "planned_horizon": 8,
  "miss_policy": "substitute",
  "layers": {
    "3": {
      "budget_count": 4,
      "static_experts": [0],
      "dynamic_experts": [2, 5, 7],
      "load_experts": [5, 7],
      "evict_experts": [1, 4],
      "predicted_utility_captured": 0.91,
      "uncertainty": 0.04,
      "estimated_load_bytes": 123456
    }
  },
  "early_termination": {
    "enabled": true,
    "metric": "weighted_out_of_subset_mass",
    "threshold": 0.25
  }
}
```

## 16. Cache-event schema

Parquet columns:

```text
run_id: string
sample_id: string
token_position: int32
window_id: int32
timestamp_us: float64
event: string
layer_idx: int32|null
expert_idx: int32|null
bytes: int64
stream: string
reason: string
resident_bytes_after: int64
```

## 17. Per-token result schema

```text
run_id
sample_id
token_position
token_id_base
token_id_method
tokens_equal
next_token_kl
token_latency_us
probe_latency_us
transfer_bytes
exposed_stall_us
window_id
window_age
early_terminated
natural_out_of_subset_mass
executed_route_diff_count
```

## 18. Per-layer result schema

```text
run_id
sample_id
token_position
layer_idx
natural_topk
executed_topk
topk_recall
routing_mass_coverage
topk_margin
out_of_subset_mass
subset_size
resident_size
loaded_bytes
```

Nested arrays may be encoded as Arrow lists or stored in a separate tensor artifact.

## 19. Aggregate metrics schema

```json
{
  "schema_version": 1,
  "run_id": "...",
  "status": "complete",
  "information_regime": "online_post_sample",
  "quality": {},
  "routing": {},
  "memory": {},
  "transfer": {},
  "latency": {},
  "probe": {},
  "calibration_training_cost": {},
  "counts": {},
  "warnings": []
}
```

## 20. Environment manifest

Record:

- OS and kernel;
- Python;
- PyTorch;
- Transformers;
- CUDA runtime and driver;
- GPU model/count/memory;
- CPU model/core count;
- system RAM;
- PCIe link information where available;
- storage;
- git commit and dirty status;
- package lock hash;
- environment variables affecting kernels/determinism;
- model revision and file hashes.

## 21. Run state files

A run directory contains exactly one terminal marker:

- `DONE.json`;
- `FAILED.json`;
- `CANCELLED.json`.

`DONE.json` includes checksums of primary outputs.

## 22. Schema migration

Every persistent artifact has a schema version. Implement explicit migration functions. Never silently reinterpret old fields.

## 23. Dry-run estimates

Before collection, estimate:

\[
\text{trace bytes}
\approx
T\times L\times E\times \text{dtype bytes}
\]

plus optional hidden state arrays.

Before runtime, estimate:

- dense model bytes;
- KV cache bytes;
- expert cache bytes;
- double buffer;
- probe temporary memory;
- CPU expert storage;
- trace output.

Abort with a clear message when the configured budget is impossible.
