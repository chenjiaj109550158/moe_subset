# System Architecture

## 1. Architectural principles

The codebase must be modular enough to answer scientific questions before committing to a specific inference engine.

The architecture is divided into five layers:

1. **Model adaptation and introspection**
2. **Trace collection and offline analysis**
3. **Deployable probes and subset planning**
4. **Constrained execution and expert residency**
5. **Evaluation, simulation, and reporting**

The implementation must allow each algorithm to run on:

- a deterministic tiny test MoE;
- a Hugging Face-compatible reference model;
- a trace-only simulator;
- eventually a real CPU–GPU offload runtime.

## 2. Proposed repository tree

```text
pseudoroute-moe/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model/
│   │   ├── tiny_moe.yaml
│   │   ├── olmoe.yaml
│   │   ├── qwen3_moe.yaml
│   │   ├── mixtral.yaml
│   │   └── gpt_oss.yaml
│   ├── experiment/
│   │   ├── trace.yaml
│   │   ├── oracle_sweep.yaml
│   │   ├── closed_loop.yaml
│   │   ├── dapq_factorial.yaml
│   │   ├── probe_eval.yaml
│   │   ├── simulator.yaml
│   │   └── runtime.yaml
│   ├── hardware/
│   │   ├── simulated_pcie4.yaml
│   │   └── local_machine.yaml
│   └── benchmark/
├── docs/
│   ├── spec/
│   ├── model_support.md
│   ├── trace_format.md
│   ├── runtime_design.md
│   └── decisions.md
├── scripts/
│   ├── smoke_test.sh
│   ├── run_oracle_grid.sh
│   ├── run_factorial.sh
│   └── run_full_pipeline.sh
├── src/pseudoroute/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── types.py
│   ├── utils/
│   │   ├── determinism.py
│   │   ├── logging.py
│   │   ├── memory.py
│   │   └── timing.py
│   ├── models/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── tiny_moe.py
│   │   ├── introspection.py
│   │   └── adapters/
│   │       ├── generic_hf.py
│   │       ├── olmoe.py
│   │       ├── qwen3_moe.py
│   │       ├── mixtral.py
│   │       └── gpt_oss.py
│   ├── tracing/
│   │   ├── collector.py
│   │   ├── hooks.py
│   │   ├── store.py
│   │   ├── manifest.py
│   │   └── validation.py
│   ├── oracle/
│   │   ├── windows.py
│   │   ├── selectors.py
│   │   ├── locality.py
│   │   └── closed_loop.py
│   ├── analysis/
│   │   ├── dapq_factorial.py
│   │   ├── router_geometry.py
│   │   ├── expert_criticality.py
│   │   ├── margins.py
│   │   └── metrics.py
│   ├── probes/
│   │   ├── base.py
│   │   ├── history.py
│   │   ├── direct.py
│   │   ├── rephased.py
│   │   ├── pseudo_tokens.py
│   │   ├── shadow_rollout.py
│   │   ├── default_vectors.py
│   │   ├── adep_style_rf.py
│   │   └── ensemble.py
│   ├── selection/
│   │   ├── utility.py
│   │   ├── budget.py
│   │   ├── static_residency.py
│   │   └── uncertainty.py
│   ├── execution/
│   │   ├── routing_policy.py
│   │   ├── masks.py
│   │   ├── miss_policy.py
│   │   └── generation.py
│   ├── runtime/
│   │   ├── expert_handle.py
│   │   ├── resident_cache.py
│   │   ├── transfer.py
│   │   ├── simulator.py
│   │   ├── offload_engine.py
│   │   └── hardware_profile.py
│   ├── training/
│   │   ├── datasets.py
│   │   ├── linear_probe.py
│   │   ├── mlp_probe.py
│   │   └── rf_probe.py
│   ├── evaluation/
│   │   ├── route_metrics.py
│   │   ├── quality.py
│   │   ├── performance.py
│   │   ├── benchmarks.py
│   │   └── aggregate.py
│   └── plotting/
│       ├── pareto.py
│       ├── heatmaps.py
│       └── latency.py
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── fixtures/
```

Model adapters beyond the first supported model may initially be stubs with explicit `NotImplementedError`.

## 3. Core identifiers and types

Never represent an expert only by an integer. Expert IDs repeat across layers.

```python
@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_idx: int
    expert_idx: int
```

Recommended types:

```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str
    num_layers: int
    moe_layer_indices: tuple[int, ...]
    num_experts_by_layer: dict[int, int]
    top_k_by_layer: dict[int, int]
    hidden_size: int
    uses_rope: bool
    pre_norm: bool
    expert_bytes: dict[ExpertKey, int]

@dataclass
class DecodeBoundary:
    sample_id: str
    generated_length: int
    absolute_position: int
    decision_mode: Literal["pre_sample", "post_sample"]
    next_token_id: int | None

@dataclass
class DeployableDecodeState:
    boundary: DecodeBoundary
    current_token_id: int
    prefix_length: int
    router_history: "RouterHistoryView"
    current_router_inputs: dict[int, Tensor] | None
    current_router_logits: dict[int, Tensor]
    kv_cache_handle: "ReadOnlyKVCache"
    resident_experts: frozenset[ExpertKey]
    memory_budget_bytes: int
```

Offline data must use a distinct object:

```python
@dataclass
class OfflineFutureTrace:
    sample_id: str
    start_position: int
    token_ids: Tensor
    router_logits: Tensor
    router_probs: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
```

Do not add `future_trace` as an optional field to `DeployableDecodeState`.

## 4. Model adapter interface

Create an abstract `MoEModelAdapter`.

```python
class MoEModelAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> ModelSpec: ...

    @abstractmethod
    def validate_structure(self) -> None: ...

    @abstractmethod
    def iter_moe_layers(self) -> Iterable["MoELayerHandle"]: ...

    @abstractmethod
    def run_base_forward(
        self,
        input_ids: Tensor,
        *,
        position_ids: Tensor | None = None,
        kv_cache: object | None = None,
        use_cache: bool = False,
        trace_request: "TraceRequest | None" = None,
    ) -> "ForwardResult": ...

    @abstractmethod
    def route_from_state(
        self,
        layer_idx: int,
        router_input: Tensor,
    ) -> "RouteResult": ...

    @abstractmethod
    def forward_with_policy(
        self,
        input_ids: Tensor,
        policy: "RoutingPolicy",
        *,
        kv_cache: object | None = None,
        use_cache: bool = False,
    ) -> "ForwardResult": ...

    @abstractmethod
    def clone_kv_cache_for_shadow(self, kv_cache: object) -> object: ...

    @abstractmethod
    def build_shadow_probe_components(
        self,
        layer_idx: int,
    ) -> "ShadowLayerComponents": ...

    @abstractmethod
    def get_expert_handle(self, key: ExpertKey) -> "ExpertHandle": ...
```

### 4.1 `MoELayerHandle`

Must expose:

- layer index;
- router module;
- expert modules;
- top-\(k\);
- normalization before router;
- post-attention residual location;
- expert-combine semantics;
- shared experts, if any;
- whether router logits are normalized or biased;
- any capacity or group-routing rules.

### 4.2 Adapter validation

`inspect-model` must fail early when the architecture is unsupported. Validate:

- exact count of MoE layers and experts;
- router weight shape;
- top-\(k\);
- shared expert handling;
- expert parameter byte size;
- position encoding type;
- KV-cache format;
- ability to capture router input without modifying output;
- ability to apply a mask before top-\(k\).

## 5. Tiny MoE model

Implement a tiny reference model inside the repository.

Required capabilities:

- decoder-only causal attention;
- configurable RoPE on/off;
- pre-norm block;
- top-\(k\) router;
- configurable number of experts;
- deterministic expert MLPs;
- optional shared expert;
- trace hooks;
- mask-constrained routing;
- synthetic expert byte sizes;
- optional artificial H2D latency.

Use this model for unit tests and algorithm debugging. It should be small enough to run on CPU.

## 6. Trace collection architecture

### 6.1 Trace modes

```python
class TraceLevel(Enum):
    ROUTES_ONLY = "routes_only"
    ROUTER_LOGITS = "router_logits"
    ROUTER_INPUTS = "router_inputs"
    POST_ATTENTION = "post_attention"
    ATTENTION_QKV = "attention_qkv"
    EXPERT_OUTPUTS = "expert_outputs"
    FULL_ANALYSIS = "full_analysis"
```

Mandatory fields:

- sample ID;
- token ID;
- absolute position;
- prompt/decode flag;
- layer index;
- router logits or probabilities;
- top-\(k\) IDs;
- top-\(k\) weights.

Optional fields must be independently selectable.

### 6.2 Hook behavior

Hooks must:

- be side-effect free;
- detach tensors immediately;
- move tensors to CPU asynchronously when safe;
- cast to configured storage dtype;
- batch writes;
- avoid holding full sequence tensors in GPU memory;
- record original tensor shape and dtype;
- validate that hook insertion does not change model logits.

### 6.3 Trace store

Use a chunked store with a manifest. A recommended implementation is Zarr for dense arrays plus JSON metadata. If Zarr support is problematic, use sharded safetensors plus an index.

The store must support:

- sequential append;
- random access by sample and token range;
- reading routes without loading hidden states;
- schema versioning;
- checksums;
- resumable collection;
- shard-level atomic commits;
- a `validate-trace` function.

## 7. Router history

An online probe should receive a bounded view, not the entire raw trace.

```python
class RouterHistoryView(Protocol):
    def topk(self, layer_idx: int, lookback: int) -> Tensor: ...
    def weights(self, layer_idx: int, lookback: int) -> Tensor: ...
    def recency(self, key: ExpertKey) -> int | None: ...
    def rolling_frequency(self, layer_idx: int, window: int) -> Tensor: ...
    def coactivation(self, layer_idx: int, window: int) -> Tensor: ...
```

The maximum history length is configured and logged.

## 8. Probe interface

All deployable future-routing methods implement:

```python
class FutureRoutingProbe(ABC):
    @property
    def information_regime(self) -> str: ...

    @property
    def requires_training(self) -> bool: ...

    @abstractmethod
    def predict(
        self,
        state: DeployableDecodeState,
        horizons: Sequence[int],
    ) -> "ProbeOutput": ...
```

`ProbeOutput`:

```python
@dataclass
class ProbeOutput:
    horizons: tuple[int, ...]
    per_horizon_logits: dict[int, Tensor] | None  # [H, E_l]
    per_horizon_probs: dict[int, Tensor] | None
    aggregate_utility: dict[int, Tensor]          # [E_l]
    uncertainty: dict[int, Tensor] | None
    estimated_cost_ms: float
    metadata: dict[str, Any]
```

The dictionary key is layer index.

Required probe implementations:

1. `CurrentRouteProbe`
2. `PreviousWindowFrequencyProbe`
3. `MarkovTransitionProbe`
4. `ADEPTStyleRFProbe`
5. `DirectLinearProbe`
6. `DirectMLPProbe`
7. `FuturePositionRephasedProbe`
8. `PseudoTokenProbe`
9. `DefaultVectorShadowRolloutProbe`
10. `EnsembleProbe`

Oracle selectors must not implement this interface; they belong in `oracle/`.

## 9. Subset selector interface

```python
class ExpertSubsetSelector(ABC):
    @abstractmethod
    def select(
        self,
        probe: ProbeOutput,
        *,
        resident: frozenset[ExpertKey],
        budgets: dict[int, int] | dict[int, int],
        static_set: frozenset[ExpertKey],
        load_costs: dict[ExpertKey, float],
    ) -> "SubsetPlan": ...
```

`SubsetPlan` must contain:

- decision boundary;
- planned maximum horizon;
- per-layer subset;
- static and dynamic members;
- planned load and eviction deltas;
- predicted utility captured;
- uncertainty;
- estimated probe and transfer costs;
- selected miss policy;
- early-termination thresholds;
- information regime.

Validate that every layer obeys its budget.

## 10. Routing policy architecture

Use a runtime policy object rather than globally patching router modules.

```python
class RoutingPolicy(Protocol):
    def choose(
        self,
        layer_idx: int,
        natural_logits: Tensor,
        natural_topk: Tensor,
        token_context: "TokenRoutingContext",
    ) -> "ExecutedRoute": ...
```

Implement:

- `NaturalRoutingPolicy`
- `MaskedSubstitutionPolicy`
- `MaskedTruncationPolicy`
- `LosslessFallbackPolicy`
- `EarlyTerminationPolicy`
- `RecordingPolicy`

The base model path must use `NaturalRoutingPolicy`.

## 11. Production and shadow KV caches

The system must distinguish:

- **production KV cache**, used by real autoregressive generation;
- **shadow probe cache**, used by pseudo tokens or rephased queries.

Rules:

- A shadow probe may read production keys/values.
- It must not append to, evict from, reorder, quantize, or mutate the production cache.
- Shadow positions must not advance the production cache position.
- A probe must return without changing the random number generator state used by generation, unless explicitly configured.
- Add regression tests comparing generation with and without a no-op shadow probe.

Where cloning a full cache is too expensive, implement a read-only adapter plus temporary pseudo K/V buffers.

## 12. Default-vector store

Default vectors are model- and calibration-specific.

```python
@dataclass
class DefaultVectorManifest:
    model_fingerprint: str
    dataset_fingerprint: str
    layer_idx: int
    expert_idx: int
    count: int
    vector_dtype: str
    vector_shape: tuple[int, ...]
    definition: Literal[
        "expert_output",
        "weighted_expert_output",
        "moe_residual_contribution",
    ]
```

Support online running means during calibration without storing every expert output.

## 13. Router-geometry module

For every MoE layer, compute and store:

- router row norms;
- router bias;
- singular values;
- effective rank;
- pairwise row cosine;
- empirical router-input mean/covariance or low-rank sketch;
- covariance-aware logit variance;
- top-\(k\) margin distributions;
- expert pair boundary frequency;
- layerwise route entropy;
- route transition matrices.

For large hidden dimensions, use randomized SVD and streaming covariance sketches.

## 14. Expert criticality module

Implement interventions on sampled token/layer positions:

- remove one selected expert;
- substitute with the next resident candidate;
- zero its contribution;
- replace output with its default vector;
- perturb gate weight;
- replace with nearest expert by output similarity.

Measure:

- immediate MoE output error;
- next-token logit KL;
- downstream router KL over configured layers;
- downstream top-\(k\) flips;
- final task quality where feasible.

Interventions must be batched or sampled; exhaustive analysis may be intractable.

## 15. Offload simulator

The simulator consumes:

- route or executed-route trace;
- subset plans;
- expert sizes;
- HBM cache capacity;
- bandwidth and fixed transfer latency;
- compute intervals;
- prefetch issue times;
- CUDA stream overlap rules;
- eviction policy.

It outputs an event timeline:

```python
@dataclass
class CacheEvent:
    timestamp_us: float
    event: Literal[
        "probe_start", "probe_end",
        "load_start", "load_end",
        "evict", "expert_compute_start",
        "expert_compute_end", "stall_start", "stall_end",
        "window_terminate",
    ]
    expert: ExpertKey | None
    layer_idx: int | None
    bytes: int
    stream: str
    reason: str
```

Simulator invariants:

- no expert is used before load completion unless CPU execution is selected;
- resident bytes never exceed budget;
- transfer streams obey configured concurrency;
- overlapping transfers do not exceed aggregate bandwidth;
- probe cost is placed at the correct decision time;
- lossless fallback loads every missing natural expert.

## 16. Real offload engine

The first real engine targets single-GPU, batch-1 decode.

### 16.1 Expert storage

Represent each expert using an `ExpertHandle` with:

- CPU parameter tensors;
- pinned-memory status;
- optional quantized representation;
- GPU slot assignment;
- CUDA event for transfer completion;
- byte size;
- state: `CPU`, `LOADING`, `RESIDENT`, `EVICTING`.

### 16.2 Transfers

Use:

- pinned CPU memory where possible;
- a dedicated CUDA transfer stream;
- CUDA events for dependency;
- asynchronous non-blocking copies;
- preallocated GPU expert slots;
- no repeated allocator churn in the decode loop.

### 16.3 Slot model

Prefer fixed GPU slots over moving full PyTorch modules repeatedly. The adapter should map a logical expert to a slot and execute the slot's current tensors.

Initially support equal-size experts. Add variable-size packing only after the fixed-slot engine works.

### 16.4 Correctness mode

Provide a slow reference mode that synchronizes every transfer. Compare outputs against the asynchronous engine before using performance results.

### 16.5 Memory accounting

Measure with:

- parameter byte totals;
- allocated and reserved CUDA memory;
- peak memory snapshots;
- KV-cache size;
- temporary workspaces;
- pinned CPU memory.

Do not report only `torch.cuda.max_memory_allocated()` as total HBM usage without explaining excluded/reserved memory.

## 17. Evaluation architecture

Every run emits:

```text
run_dir/
├── resolved_config.yaml
├── environment.json
├── model_manifest.json
├── hardware.json
├── metrics.json
├── per_token.parquet
├── per_layer.parquet
├── cache_events.parquet
├── stdout.log
├── stderr.log
└── DONE
```

If a run fails, write `FAILED.json` with exception and partial progress.

## 18. CLI behavior

All commands must:

- validate config before loading the model;
- print an estimated storage and memory requirement in `--dry-run`;
- support resume where meaningful;
- never overwrite a completed run unless `--force`;
- save a fully resolved config;
- log the information regime.

Example:

```bash
pseudoroute collect-traces \
  --config configs/experiment/trace.yaml \
  model=olmoe \
  trace.level=router_logits \
  output_dir=artifacts/traces/olmoe_wikitext
```

## 19. Dependency and tooling guidance

Use a modern Python packaging workflow with a lock file. Suggested dependencies:

- PyTorch;
- Transformers;
- Accelerate;
- Datasets;
- Safetensors;
- Zarr and/or PyArrow;
- NumPy, SciPy, scikit-learn;
- Pydantic or structured dataclasses;
- Hydra/OmegaConf or a simple YAML composition layer;
- pytest;
- mypy or pyright;
- ruff;
- matplotlib;
- optional experiment tracking integration.

Keep the core runnable without a hosted tracking service.

## 20. Model support strategy

### Tier 0 — Tiny model

Full support and exhaustive tests.

### Tier 1 — Small open MoE

First real adapter and all analysis features.

### Tier 2 — Medium/large MoE

Route tracing, oracle, probes, and simulator.

### Tier 3 — Real offloading

Only architectures whose expert tensors can be safely mapped into fixed GPU slots.

Maintain a support matrix listing each capability per model.

## 21. Security and robustness

- Use `trust_remote_code=False` by default.
- Require explicit opt-in for remote model code.
- Never execute arbitrary dataset code by default.
- Validate output paths.
- Avoid unsafe pickle for persistent artifacts.
- Use safetensors, JSON, YAML, Parquet, or Zarr.
- Sanitize model IDs when creating directories.
