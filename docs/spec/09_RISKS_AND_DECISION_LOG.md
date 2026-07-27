# Risks, Blind Spots, and Decision Log

## 1. Fixed initial decisions

These decisions define the first implementation. They may change only through an explicit entry in `docs/decisions.md`.

1. The core problem is ordinary autoregressive decode, not speculative decoding.
2. The first target is batch size 1.
3. Every MoE layer has its own expert subset.
4. All MoE layers initially share a global window boundary.
5. The primary method is training-free.
6. Learned RF/linear/MLP methods are baselines/extensions.
7. The initial selector uses a fixed per-layer expert count.
8. The initial hard policy is masked substitution with explicit renormalization.
9. Lossless fallback is always available as a baseline.
10. Real offloading follows oracle, closed-loop, and simulator validation.
11. DapQ-style position dominance is a hypothesis, not an axiom.
12. Offline future traces and online decode states use incompatible APIs.

## 2. Risk: Oracle hard commitment is not viable

### Failure mode

Even the best future-aware subset causes large closed-loop quality loss at practical budgets.

### Detection

- oracle Pareto has no useful region;
- early token divergence;
- reasoning/code accuracy collapses;
- low route mass is not correlated with low quality impact.

### Mitigation/pivot

- shorten the window;
- enlarge subset;
- use early termination;
- use lossless fallback for high-risk layers;
- keep critical experts static;
- apply hard commitment only to tolerant layers;
- move from hard constraint to prefetch hint;
- investigate similar-expert replacement;
- post-train router/controller only as a later direction.

## 3. Risk: Position does not dominate MoE routing

### Why plausible

DapQ's finding is tied to RoPE-transformed attention queries. Routers consume post-attention hidden states and may depend more on token identity, context, or prior expert outputs.

### Detection

- DC/SP does not outperform SC/DP at router input/logits;
- benefits disappear after attention;
- direct history/linear baselines dominate rephased probes.

### Pivot

Reframe from `position-conditioned` to `future pseudo routing states` using:

- context summaries;
- route history;
- sampled next token;
- router-visible dynamics;
- default-vector shadow propagation;
- hybrid position + content/context.

A negative factorial result is still a useful analysis contribution.

## 4. Risk: Pseudo probe costs more than it saves

### Failure mode

Periodic pseudo attention or shadow rollout adds enough GPU compute to offset transfer savings.

### Detection

\[
T_{\mathrm{probe}} \ge T_{\mathrm{avoided\ exposed\ stalls}}.
\]

### Mitigation

- sparse horizon anchors;
- only probe sensitive/uncertain layers;
- use direct low-rank predictor;
- execute probe on CPU only if actually beneficial;
- reuse intermediate states;
- lower-frequency planning;
- early exit from shadow rollout;
- use probe to update only dynamic layers;
- optimize after profiling.

## 5. Risk: Closed-loop distribution shift

### Failure mode

A wrong expert changes hidden states, next tokens, and all subsequent routes. A predictor trained on base-model traces becomes miscalibrated.

### Detection

- open-loop prediction remains high while closed-loop misses rise;
- quality errors compound with window age;
- predictor confidence is wrong after the first divergence.

### Mitigation

- re-plan frequently;
- adaptive early termination;
- train/evaluate on constrained trajectories;
- use robust subset coverage rather than exact route;
- uncertainty-aware selection;
- static critical experts;
- conservative lossless policy for low-margin states.

## 6. Risk: Window boundary stall

### Failure mode

Window-internal transfers disappear, but a large synchronous load occurs at every boundary.

### Detection

- average bytes/token falls but p95/p99 TPOT rises;
- no memory for double buffering;
- large subset turnover.

### Mitigation

- add load-cost/hysteresis term;
- prefer subset overlap;
- prefetch next delta while current window executes;
- stagger layer loads;
- adaptive window length;
- reserve a staging slot;
- minimize delta rather than absolute subset score.

## 7. Risk: HBM budget accounting is incomplete

### Common omissions

- KV cache;
- attention/dense weights;
- workspace;
- CUDA allocator reserve;
- pseudo-probe buffers;
- next-window staging;
- quantization scales;
- shared experts;
- temporary dequantization buffers.

### Mitigation

Use explicit accounting and measured memory snapshots. Abort impossible configs before running.

## 8. Risk: Expert ID aliasing across layers

Expert 5 in different layers is unrelated.

### Mitigation

Always use `ExpertKey(layer_idx, expert_idx)`. Never keep a global set of bare expert integers.

## 9. Risk: Confusing router weights and expert weights

Router row norms do not directly measure expert FFN quality importance.

### Mitigation

Keep separate analyses for:

- router geometry;
- activation frequency;
- expert output contribution;
- causal quality sensitivity;
- downstream routing influence.

Do not select permanent experts solely by \(\|w_{\ell,e}^R\|\).

## 10. Risk: Top-\(k\) recall overstates quality

Missing a low-weight expert and missing a dominant expert count equally in set recall.

### Mitigation

Report:

- routing-mass coverage;
- output reconstruction error;
- next-token KL;
- task quality;
- margin and certainty;
- closed-loop results.

## 11. Risk: Full future hidden-state traces are too large

### Mitigation

- router logits first;
- optional hidden capture;
- layer/token sampling;
- float16;
- chunking;
- dry-run estimate;
- low-rank online projection;
- compute metrics streaming when possible.

## 12. Risk: Pseudo KV cache mutates production state

### Failure mode

Generation positions shift or pseudo tokens contaminate actual context.

### Mitigation

Read-only cache wrapper, temporary pseudo buffers, regression tests, and RNG isolation.

## 13. Risk: Model-specific router semantics

MoE models differ in:

- softmax versus sigmoid;
- group routing;
- shared experts;
- routing bias;
- normalization;
- expert capacity;
- aux-loss-related behavior;
- fused kernels;
- quantized experts.

### Mitigation

Exact adapter-level route implementation and validation. Do not use a generic `softmax + topk` when the model differs.

## 14. Risk: Masking produces invalid cardinality

A subset smaller than natural top-\(k\) cannot support substitution.

### Mitigation

Validate:

\[
B_\ell \ge k_\ell
\]

for substitution. Otherwise require truncation or another explicit policy.

## 15. Risk: Static experts destroy adaptability

Hot experts from calibration data may be poor under domain shift.

### Mitigation

- sweep static fraction;
- evaluate held-out domains;
- reserve dynamic capacity;
- include miss risk and criticality;
- allow domain-conditioned static sets only as a separate method.

## 16. Risk: Batch union explosion

For concurrent requests:

\[
\left|\bigcup_q S_{\ell}^{(q)}\right|
\]

may approach all experts.

### Mitigation

Batch-1 first. Later compare:

- per-request subsets;
- shared batch subset;
- clustering requests by predicted expert demand;
- synchronized boundaries;
- CPU execution for outliers.

Do not imply batch-1 latency gains automatically transfer to high-throughput serving.

## 17. Risk: Learned baseline leakage

Random token splits allow nearly identical positions from the same prompt in train and test.

### Mitigation

Group splits by sample/document and report split fingerprints.

## 18. Risk: Sampling changes due to probe RNG

Pseudo branches may consume RNG and alter the base generated sequence independently of routing.

### Mitigation

Use separate generators and save/restore RNG state.

## 19. Risk: Timing methodology is misleading

### Common errors

- timing without CUDA synchronization;
- including model load in one method only;
- excluding probe time;
- reporting simulator TPOT as measured;
- ignoring cold/warm distinction;
- measuring bytes but not exposed stall.

### Mitigation

Use CUDA events, host wall time, warmups, repeated runs, and per-token traces. Label every timing source.

## 20. Risk: Predictor chosen after looking at test tasks

### Mitigation

Use validation data for architecture/hyperparameters. Preserve a final test suite. Report exploratory versus confirmatory runs.

## 21. Risk: Citation or reproduction mismatch

Some related systems may not release complete code or all hyperparameters.

### Mitigation

Use names such as `ADEPTStyleRFProbe` or `CommitStyleNoFallback` unless exact reproduction is verified. Document differences.

## 22. Decision template

Every changed decision should be added to `docs/decisions.md`:

```markdown
## D-YYYYMMDD-NNN — Short title

**Status:** proposed | accepted | superseded  
**Context:**  
**Decision:**  
**Alternatives considered:**  
**Consequences:**  
**Experiments affected:**  
**Migration required:**  
```

## 23. Recommended pivot tree

```text
Oracle feasible?
├── no
│   ├── adaptive/fallback/layer-selective method feasible?
│   │   ├── yes -> continue with safer execution semantics
│   │   └── no  -> stop hard-commitment runtime work; publish/record analysis
│
└── yes
    ├── position signal helps routers?
    │   ├── yes -> develop position-conditioned probes
    │   └── no  -> develop context/history/router-visible probes
    │
    └── deployable probe beats simple baselines?
        ├── no -> prefer simple predictor/cache policy
        └── yes
            └── real net latency benefit?
                ├── no -> reduce probe cost or target slower interconnect
                └── yes -> full system evaluation
```
