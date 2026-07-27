# Experiment Plan

## 1. Experimental philosophy

The project must answer three questions in order:

1. **Can a fixed multi-token subset work even with oracle future information?**
2. **Can a deployable pseudo-routing method predict a useful subset?**
3. **Does the resulting system improve real end-to-end performance after probe overhead?**

Do not reverse this order. A high route-prediction score is not sufficient, and a simulator speedup is not an end-to-end result.

## 2. Model tiers

### Tier 0 — Deterministic tiny MoE

Use for:

- unit tests;
- exact oracle validation;
- cache simulator validation;
- controlled position/content experiments;
- synthetic failure cases.

Required configurations:

- RoPE and no-RoPE;
- high and low temporal routing consistency;
- high and low top-\(k\) margins;
- one deliberately routing-critical expert;
- equal and variable expert sizes.

### Tier 1 — Small open MoE

Use one model that can run natural inference and route tracing on available hardware. The adapter must support:

- exact routing capture;
- hidden/router-input capture;
- mask-constrained execution;
- teacher-forced analysis.

A candidate is an OLMoE-scale model, but the model ID must be selected and verified in the actual environment rather than hard-coded into the scientific claims.

### Tier 2 — Representative larger models

Select models that differ in:

- number of experts;
- top-\(k\);
- hidden size;
- shared experts;
- router design;
- local routing consistency;
- architecture family.

Candidate families include OLMoE, Qwen MoE, Mixtral, and gpt-oss. Support depends on hardware and adapter feasibility.

### Tier 3 — Real offload target

Choose one architecture whose experts can be mapped into fixed GPU slots and whose full expert weights exceed the intended expert-residency budget.

## 3. Task and data categories

Use multiple task types because routing locality and sensitivity may be domain-dependent.

### 3.1 Language modeling

Purpose:

- token-level likelihood;
- KL/perplexity;
- long continuous traces;
- controlled teacher forcing.

Potential corpora:

- WikiText-style validation text;
- PG19-style long documents;
- a held-out general web/text corpus with clear licensing.

### 3.2 Mathematical reasoning

Purpose:

- detect cascade errors and exact-answer degradation.

Potential benchmarks:

- GSM8K;
- MATH-500 or another manageable MATH subset.

### 3.3 Code generation

Purpose:

- test abrupt structural transitions and exact execution correctness.

Potential benchmarks:

- HumanEval;
- MBPP.

### 3.4 Long-context retrieval or QA

Purpose:

- study context length and position effects.

Use a small, reproducible subset first. Avoid conflating KV-cache compression with expert offloading.

### 3.5 Domain shift

At least two domains must be held out from any predictor/calibration training to evaluate generalization.

## 4. Dataset splitting

For learned probes:

- split by full prompt/document;
- never split random token positions from the same sequence across train/test;
- keep benchmark test labels untouched;
- use a calibration set distinct from predictor training when possible;
- store dataset fingerprints.

For training-free methods, calibration data for default vectors and static experts must still be disjoint from final test data.

## 5. Phase 0 — Base-model and adapter validation

### Experiments

1. Natural forward without hooks.
2. Natural forward with route-only hooks.
3. Natural forward with full selected trace hooks.
4. Natural generation through the adapter.
5. Natural generation through `NaturalRoutingPolicy`.

### Metrics

- max/mean absolute next-token-logit error;
- top-1 token agreement;
- router top-\(k\) exact match;
- router-weight error;
- generated-token exact match;
- hook memory overhead.

### Acceptance

- greedy outputs exactly match or differ only within documented numerical tolerance;
- router top-\(k\) IDs exactly match;
- no production KV mutation by analysis hooks.

## 6. Phase 1 — Routing characterization

Collect natural traces and report:

- expert activation frequency by layer;
- route entropy;
- top-\(k\) margin;
- expert union size versus horizon;
- previous-token route reuse;
- transition matrices;
- co-activation;
- layerwise SCH/count and mass curves;
- route locality by task and context position;
- cross-token and cross-layer correlation.

### Required plots

- expert-frequency heatmap;
- union-size versus horizon;
- SCH versus cache ratio and segment length;
- route entropy by layer;
- top-\(k\) margin distribution;
- transition-matrix sparsity;
- prompt-position buckets.

## 7. Phase 2 — Oracle feasibility

### Grid

At minimum:

\[
t\in\{1,2,4,8,16\}
\]

and:

\[
\rho=B/k\in\{1,1.5,2,3,4\}
\]

rounded to valid integer budgets.

Selectors:

- binary activation count;
- selected routing mass;
- full router probability mass;
- contribution-weighted;
- reconstruction greedy on a small subset;
- cost-aware oracle.

Execution policies:

- substitution;
- truncation, preserved weights;
- truncation, renormalized weights;
- lossless fallback;
- early termination;
- base model.

### Open-loop metrics

- top-\(k\) hit rate;
- routing-mass coverage;
- union coverage;
- layerwise miss rate;
- predicted transfer bytes;
- subset overlap between windows.

### Closed-loop metrics

- next-token KL;
- perplexity;
- exact-match/task score;
- output-token divergence point;
- output length;
- repetition and degeneration indicators;
- constrained-route miss rate on the altered trajectory;
- H2D bytes/token;
- boundary count.

### Required output

A quality-versus-transfer Pareto plot for every model/task and a summary table of feasible \((t,\rho)\) regions.

### Gate

If no oracle method yields a useful region, stop building expensive pseudo probes for that model and document the negative result.

## 8. Phase 3 — DapQ-style factorial routing analysis

### Core factorial conditions

For each prefix boundary, horizon, layer, and pseudo-content construction:

- SC/SP;
- DC/SP;
- SC/DP;
- DC/DP.

Additional controls:

- context swap;
- relative-position-preserving global shift;
- future-position offset sweep;
- independent versus causal pseudo mask;
- random-token seeds;
- pre-RoPE query;
- post-RoPE query;
- post-attention state;
- router input;
- router logits;
- top-\(k\) route.

### Horizons

\[
r\in\{1,2,4,8,16\}
\]

plus dense horizons for a smaller sample.

### Questions

- Does position dominate content before or only after RoPE?
- Does any post-RoPE advantage survive through attention and into the router input?
- Which layers preserve the signal?
- Does correct context dominate both?
- How quickly does performance decay with horizon?
- Is the result task/model dependent?
- Is aggregate subset utility more stable than exact route?

### Required plots

- per-layer position-dominance heatmap;
- metric versus horizon;
- position-offset sensitivity;
- pre-/post-RoPE comparison;
- router-input versus router-logit comparison;
- task/model facet plots;
- bootstrap confidence intervals.

### Pivot rule

If DC/SP does not outperform SC/DP at router level, do not force a position-centric narrative. Use the result to design context/history-conditioned probes.

## 9. Phase 4 — Router-visible representation

Compare inputs/targets:

- full router input;
- exact router logits;
- centered logits;
- top singular coordinates of router row space;
- current route probabilities;
- hidden-state PCA coordinates.

Predict future:

- per-horizon logits;
- per-horizon probabilities;
- aggregate window utility.

Models:

- persistence baseline;
- ridge regression;
- low-rank linear map;
- MLP;
- horizon-conditioned shared predictor;
- separate predictor per horizon.

Metrics:

- prediction quality;
- model size;
- trace storage;
- training cost;
- online latency.

## 10. Phase 5 — Default vectors and pseudo routing probes

### 10.1 Default-vector calibration

Sweep:

- number of calibration tokens;
- mean definition;
- dtype;
- per-domain versus global;
- variance storage;
- robust mean/clipping.

Evaluate same-token next-layer prediction first to validate against the motivating method.

### 10.2 Probe variants

1. Current route.
2. Rolling history.
3. Markov.
4. Direct linear.
5. Direct MLP.
6. ADEPT-style RF.
7. Future-position rephased query.
8. Independent pseudo tokens.
9. Causal pseudo sequence.
10. Default-vector shadow rollout.
11. Multi-branch uncertainty ensemble.
12. Oracle.

### 10.3 Equal-cost comparison

Compare methods under:

- equal number of probe attention queries;
- equal measured probe latency;
- equal predictor parameter budget;
- equal calibration data;
- equal HBM subset budget.

### 10.4 Metrics

- per-horizon top-\(k\) recall;
- aggregate top-\(B\) expert recall;
- routing-mass coverage;
- subset regret relative to oracle;
- calibration error;
- probe latency;
- temporary GPU memory;
- window H2D bytes;
- closed-loop quality.

## 11. Phase 6 — Static expert residency

Rank experts by:

- activation frequency;
- routing mass;
- router-weight row norm;
- covariance-aware logit variance;
- quality sensitivity;
- downstream routing influence;
- predictor miss risk;
- combined score;
- random.

Sweep static budget fraction:

\[
f_{\mathrm{static}}\in\{0,0.25,0.5,0.75,1.0\}
\]

of the expert cache.

Report:

- dynamic subset flexibility;
- load bytes;
- miss rate;
- quality;
- sensitivity to domain shift.

## 12. Phase 7 — Adaptive termination

Sweep termination signals:

- out-of-subset mass;
- sensitive-layer maximum;
- route-disagreement count;
- low margin plus missing expert;
- cumulative risk.

Sweep thresholds to produce:

- average realized window length;
- early termination rate;
- H2D bytes;
- quality;
- p95/p99 boundary stall.

Compare fixed \(t\) and adaptive \(t_{\max}\).

## 13. Phase 8 — Trace-driven simulator

### Baselines

- all experts resident, hypothetical upper bound;
- on-demand no cache;
- LRU;
- LFU/static hot experts;
- previous-token reuse;
- history predictor + fallback;
- one-step hard commitment;
- multi-step oracle;
- multi-step deployable probes.

### Hardware profiles

At minimum:

- configurable fixed transfer latency;
- PCIe-like bandwidth;
- faster interconnect profile;
- CPU-execution profile;
- no-overlap and perfect-overlap bounds.

### Sensitivity

Sweep:

- HBM expert budget;
- bandwidth;
- latency;
- compute time;
- probe cost;
- window size;
- batch size in trace-only experiments;
- double-buffer availability.

### Required outputs

- mean TPOT;
- p50/p95/p99 TPOT;
- exposed transfer stall;
- H2D bytes/token;
- cache hit rate;
- wasted prefetch bytes;
- boundary-stall distribution;
- resident occupancy;
- event timeline examples.

## 14. Phase 9 — Real offload prototype

### Correctness sequence

1. Synchronous CPU-to-GPU expert swapping.
2. Fixed GPU slots.
3. Asynchronous transfer stream.
4. Transfer/compute overlap.
5. Subset preloading at window boundary.
6. Lossless fallback baseline.
7. Hard commitment.
8. Pseudo probe integration.

### Measurements

Use CUDA events and host timers. Record:

- probe GPU time;
- transfer start/end;
- synchronization;
- layer compute;
- expert compute;
- token latency;
- memory allocated/reserved;
- CPU utilization;
- pinned memory;
- transfer throughput.

### Warm-up and repetition

- separate cold start;
- warm up kernels and allocator;
- report median across repeated runs;
- preserve per-token traces for tail analysis;
- avoid mixing model load time with decode TPOT.

## 15. Phase 10 — Batch and concurrency stress tests

Only after batch-1 works.

Study:

- static batch;
- several independent requests;
- union of per-request subsets;
- shared subset selection;
- synchronized versus asynchronous boundaries;
- cache fragmentation;
- throughput versus latency.

A negative result at larger batch sizes is expected and should be reported.

## 16. Primary metrics

### 16.1 Quality

- perplexity or average negative log-likelihood;
- next-token KL/JSD;
- task accuracy/exact match/pass rate;
- output-token agreement;
- semantic judge only as a secondary metric;
- repetition and degeneration;
- length.

### 16.2 Routing

- exact top-\(k\) match;
- recall@\(k\);
- Jaccard;
- routing-mass coverage;
- expert rank correlation;
- top-\(k\) margin;
- oracle regret;
- out-of-subset mass;
- subset turnover.

### 16.3 Memory/system

- peak HBM;
- expert-resident HBM;
- KV HBM;
- workspace HBM;
- pinned DRAM;
- H2D bytes/token;
- wasted bytes;
- cache hit rate;
- exposed stall;
- probe overhead;
- mean and tail TPOT;
- throughput.

### 16.4 Training/calibration cost

For every learned or calibrated method:

- number of prompts/tokens;
- target-model trace cost;
- trainable parameters;
- optimizer/epochs;
- hardware;
- wall-clock;
- peak memory;
- artifact size.

## 17. Mandatory ablations

- no future position;
- wrong future position;
- no pseudo semantic content;
- current-token content;
- sampled-next-token content;
- independent versus causal pseudo tokens;
- no default vector;
- global mean versus expert default;
- full hidden versus router-visible state;
- exact horizons versus sparse anchors;
- uniform versus discounted aggregation;
- no load-cost term;
- no static experts;
- fixed versus adaptive window;
- fallback versus no fallback;
- substitution versus truncation;
- per-layer versus global budget;
- base route versus constrained route.

## 18. Statistical reporting

- use paired examples across methods;
- report mean, median, standard error or confidence intervals;
- bootstrap quality and routing metrics;
- report all seeds;
- avoid significance claims from token-level pseudo-replication;
- treat prompts/documents as the sampling unit where appropriate;
- show per-model and per-task results before macro-averaging.

## 19. Minimum paper-ready tables

### Table A — Model and hardware

Architecture, total/active parameters, layers, experts, top-\(k\), expert bytes, HBM, DRAM, interconnect.

### Table B — Oracle feasibility

Model/task, \(t\), budget, mass coverage, quality delta, H2D reduction.

### Table C — Factorial analysis

Layer groups/horizons, SC/SP, DC/SP, SC/DP, DC/DP, position-dominance score.

### Table D — Probe comparison

Information regime, training/calibration, probe latency, route/subset metrics, closed-loop quality.

### Table E — End-to-end runtime

Baseline/method, peak HBM, bytes/token, TPOT mean/p95/p99, quality.

### Table F — Static residency

Ranking method, static fraction, miss rate, transfer, quality.

## 20. Minimum paper-ready figures

1. System overview.
2. Oracle quality–transfer Pareto.
3. Position/content per-layer heatmap.
4. Prediction decay versus horizon.
5. Router margin versus prediction error.
6. Static/dynamic expert map.
7. TPOT breakdown.
8. HBM/transfer sensitivity.
9. Batch-size degradation.
10. Failure-case traces.

## 21. Failure-case analysis

Manually inspect examples with:

- first token divergence;
- high out-of-subset mass;
- low router margin;
- abrupt topic/syntax transition;
- rare expert activation;
- reasoning or code failure;
- repeated early termination;
- high predictor confidence but large quality damage.

Save natural and executed routes, subset plans, and token text with privacy/licensing precautions.

## 22. Reporting negative results

Required negative-result categories:

- models with insufficient local routing consistency;
- layers where position signal disappears;
- tasks where oracle hard commitment fails;
- probes whose overhead exceeds savings;
- static experts that hurt domain adaptation;
- batch sizes where subset union destroys savings.

Do not remove these from aggregate results.
