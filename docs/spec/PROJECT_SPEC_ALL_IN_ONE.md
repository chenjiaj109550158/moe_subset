# PseudoRoute-MoE — Complete Project Specification
> This file concatenates the modular specifications. When possible, use the individual files because they are easier to update and reference.
## Document order
1. `README.md`
2. `README_zh-TW.md`
3. `00_CODEX_START_HERE.md`
4. `01_RESEARCH_SPEC.md`
5. `02_RELATED_WORK_AND_NOVELTY.md`
6. `03_SYSTEM_ARCHITECTURE.md`
7. `04_ALGORITHMS.md`
8. `05_EXPERIMENT_PLAN.md`
9. `06_IMPLEMENTATION_ROADMAP.md`
10. `07_CONFIG_AND_DATA_SCHEMAS.md`
11. `08_CODEX_TASK_PROMPTS.md`
12. `09_RISKS_AND_DECISION_LOG.md`
13. `10_REPRODUCIBILITY_CHECKLIST.md`


---

<!-- BEGIN FILE: README.md -->

# PseudoRoute-MoE Project Specification

**Working title:** *Position-Conditioned Pseudo Routing States for Memory-Budgeted MoE Offloading*

**Repository codename:** `pseudoroute-moe`

**Specification status:** research and implementation blueprint, verified against the cited primary sources on 2026-07-27.

This directory is a complete, from-scratch specification for building a research codebase that studies and implements lossy expert offloading for large Mixture-of-Experts (MoE) language models under a strict HBM/VRAM budget.

The central idea is to decide, at a decode-window boundary, which subset of experts should remain available for the next several decode steps. Instead of predicting an exact future route one token at a time, the system estimates **window-level future expert utility** using **pseudo routing states** conditioned on future positions, current context, router geometry, and routing history. The selected expert subset can then be hard-committed for the window, optionally with adaptive early termination.

## The core research question

Given the prefix available at decode step \(i\), a future horizon \(t\), a per-layer expert budget \(B_\ell\), and the current GPU-resident expert set \(C_{\ell,i}\), construct a deployable estimate

\[
\widehat u_{\ell,e}^{(i,t)}
\approx
\sum_{r=1}^{t}
\mathbb E\!\left[
p_{\ell,i+r,e}
\mid
x_{\le i}, \mathcal H_i
\right]
\]

and select a subset

\[
S_{\ell,i}
=
\arg\max_{|S|\le B_\ell}
\left[
\sum_{e\in S}\widehat u_{\ell,e}^{(i,t)}
-
\lambda \operatorname{LoadCost}(S\setminus C_{\ell,i})
\right].
\]

During the following decode window, the model may be constrained to route only within \(S_{\ell,i}\), converting unpredictable per-token expert misses into fewer, scheduled window-boundary transfers.

## Read order for Codex

1. [`00_CODEX_START_HERE.md`](00_CODEX_START_HERE.md)
2. [`01_RESEARCH_SPEC.md`](01_RESEARCH_SPEC.md)
3. [`02_RELATED_WORK_AND_NOVELTY.md`](02_RELATED_WORK_AND_NOVELTY.md)
4. [`03_SYSTEM_ARCHITECTURE.md`](03_SYSTEM_ARCHITECTURE.md)
5. [`04_ALGORITHMS.md`](04_ALGORITHMS.md)
6. [`05_EXPERIMENT_PLAN.md`](05_EXPERIMENT_PLAN.md)
7. [`06_IMPLEMENTATION_ROADMAP.md`](06_IMPLEMENTATION_ROADMAP.md)
8. [`07_CONFIG_AND_DATA_SCHEMAS.md`](07_CONFIG_AND_DATA_SCHEMAS.md)
9. [`08_CODEX_TASK_PROMPTS.md`](08_CODEX_TASK_PROMPTS.md)
10. [`09_RISKS_AND_DECISION_LOG.md`](09_RISKS_AND_DECISION_LOG.md)
11. [`10_REPRODUCIBILITY_CHECKLIST.md`](10_REPRODUCIBILITY_CHECKLIST.md)

A single concatenated copy is also provided as [`PROJECT_SPEC_ALL_IN_ONE.md`](PROJECT_SPEC_ALL_IN_ONE.md).

## Non-negotiable separation

The project has two distinct paths and must never mix them silently:

### Offline analysis path

May use teacher forcing and ground-truth future tokens to answer scientific questions:

- Is a multi-token fixed subset feasible even with an oracle?
- Does correct future position help predict future router outputs?
- Is position more important than content for routing?
- Which layers and experts exhibit local routing consistency?
- Which experts are hot, quality-critical, routing-critical, or hard to predict?

This path is allowed to inspect future traces, but all outputs must be labeled `oracle` or `offline_analysis`.

### Online deployable path

May use only information available at the chosen decision time:

- the prefix \(x_{\le i}\);
- current and past hidden/router states;
- current KV cache;
- current resident expert set;
- optionally the already sampled next token, but only in the explicitly named `post_sample` decision mode.

It must not use future token IDs, future hidden states, future router outputs, or data derived from the current test sample's future continuation.

## Initial implementation scope

The first end-to-end version should target:

- batch size 1;
- autoregressive decode, not speculative decoding;
- standard top-\(k\) MoE layers;
- a global decode-window boundary, with a separate subset for each MoE layer;
- a training-free core method;
- optional learned RF/linear/MLP baselines;
- a trace-driven offload simulator before a real CPU-to-GPU offload engine;
- a tiny in-repository MoE model for unit tests;
- one small open MoE model for functional tests;
- larger models only after the analysis pipeline is correct.

## What counts as success

The project is successful only if it reports a **quality–memory–transfer–latency Pareto frontier**, not merely expert prediction accuracy. Required outcomes include:

- closed-loop generation quality under constrained routing;
- HBM/VRAM usage;
- host-to-device bytes per generated token;
- exposed transfer stall after overlap;
- mean and tail time per output token;
- pseudo-probe overhead;
- top-\(k\) and routing-mass coverage;
- comparisons against lossless prefetch, one-step hard commitment, history-based caching, and oracle subsets.

## Important scientific caution

The DapQ result that position dominates content applies directly to RoPE-transformed attention queries. An MoE router usually consumes a post-attention normalized hidden state and does not directly consume a position ID. Therefore, **position dominance for MoE routing is a hypothesis to test, not an assumption to encode into the implementation**.

<!-- END FILE: README.md -->


---

<!-- BEGIN FILE: README_zh-TW.md -->

# PseudoRoute-MoE 專案規格說明

這套 Markdown 文件是把前面完整討論整理成一個可直接交給 Codex 的「從零建構專案」規格。

## 專案目標

在 HBM/VRAM 無法容納大型 MoE 全部 experts 的情況下，把 expert weights 放在 CPU DRAM，並在 decode 過程中以 window 為單位決定：

\[
i+1,\ldots,i+t
\]

這段期間，每一個 MoE layer 可以使用的 expert subset。Window 內盡量不重新從 DRAM 載入 expert，以減少每-token 不規則的 PCIe 傳輸與同步等待。

研究核心不是單純預測下一個 expert，而是建立一種 **future-position-conditioned pseudo routing state**，估計未來一段 window 中每個 expert 的累積 utility，接著在 HBM budget 與載入成本下做 subset selection。

## Codex 應從哪裡開始

先把整個資料夾交給 Codex，並要求它從 [`00_CODEX_START_HERE.md`](00_CODEX_START_HERE.md) 開始讀。若 Codex 一次只能吃一份文件，可使用 [`PROJECT_SPEC_ALL_IN_ONE.md`](PROJECT_SPEC_ALL_IN_ONE.md)。

## 這套規格已經替你固定的初始決策

- 先做 batch size 1。
- 先做普通 autoregressive decoding，不把 speculative decoding 當核心依賴。
- 每層有自己的 subset，但初版所有 layers 共用 window boundary。
- 先做 training-free 版本，再加 RF／linear／MLP predictor baseline。
- 先建立 route trace、oracle、closed-loop constrained generation，再做真正 offloading runtime。
- DapQ 式 position hypothesis 必須透過 content × position factorial experiment 驗證。
- 必須區分 offline oracle 與 online deployable predictor，禁止 future leakage。
- 必須比較：
  - lossless predictor + fallback；
  - one-step no-fallback；
  - multi-step persistent subset；
  - oracle subset。
- 最終目標是 quality、VRAM、H2D bytes、TPOT 的 Pareto frontier，而不是只報 prediction accuracy。

## 最重要的 go/no-go 判斷

1. **Oracle feasibility**：即使知道未來 routing，固定小 subset 是否仍能保持 closed-loop generation 品質？
2. **Position hypothesis**：正確 future position 是否真的比正確 semantic content 更能恢復未來 router output？
3. **System benefit**：pseudo routing probe 的計算成本是否小於它省下的 exposed expert-loading stall？

若任一項不成立，文件中已經定義對應 pivot：
- 從 position-centric 改成 context／routing-history-centric；
- 從 hard commitment 改成 adaptive termination 或 emergency fallback；
- 從真實 probe 改成直接 low-rank/linear predictor；
- 從 per-request subset 改成 static + dynamic residency。

<!-- END FILE: README_zh-TW.md -->


---

<!-- BEGIN FILE: 00_CODEX_START_HERE.md -->

# Codex Start Here

## Role

You are the primary implementation agent for a new research repository named `pseudoroute-moe`.

Start from an empty repository. Read every specification document in this directory before implementing model-specific optimizations. The documents are normative: when implementation choices conflict, follow the priority order below.

1. Correctness and no future leakage.
2. Reproducibility and testability.
3. Exact baseline model semantics.
4. Scientific usefulness.
5. Runtime performance.
6. Breadth of model support.

Do not silently weaken a requirement. Record unavoidable deviations in `STATUS.md` and `docs/DECISIONS.md`.

## Mission

Build an end-to-end research framework for studying and implementing position-conditioned pseudo routing states and horizon-level expert commitment for memory-budgeted MoE inference.

The framework must support all of the following:

1. Instrument a Hugging Face-style MoE causal language model and collect decode-time router traces.
2. Compute oracle segment/window expert subsets from true future routing.
3. Run closed-loop constrained generation using fixed expert subsets.
4. Reproduce a DapQ-style content-versus-position factorial analysis for future MoE routing.
5. Analyze router geometry, top-\(k\) margins, temporal locality, and expert criticality.
6. Implement training-free and learned future-routing baselines.
7. Implement future-position pseudo routing probes.
8. Aggregate predicted future routes into a cost-aware per-layer subset.
9. Simulate HBM expert caching and CPU-to-GPU transfers.
10. Implement a real single-GPU expert offload prototype after the simulator and oracle gates pass.
11. Produce paper-ready metrics, tables, and plots with fully recorded configs.

## First action

Create the following repository skeleton without implementing model-specific logic yet:

```text
pseudoroute-moe/
├── README.md
├── pyproject.toml
├── LICENSE
├── .gitignore
├── configs/
├── docs/
├── scripts/
├── src/pseudoroute/
├── tests/
├── artifacts/.gitkeep
├── STATUS.md
└── CHANGELOG.md
```

Then copy these specification documents into `docs/spec/` or retain them at repository root.

Create `STATUS.md` with:

- current milestone;
- completed tasks;
- failing tests;
- known limitations;
- next exact tasks;
- decisions that need human review.

## Engineering rules

### 1. Keep offline and online APIs separate

Use explicit types or namespaces:

```python
OfflineFutureTrace
DeployableDecodeState
```

An online probe must not accept an `OfflineFutureTrace`. Do not rely only on comments. Make leakage difficult at the type and API level.

Every result file must contain:

```yaml
information_regime: oracle | offline_teacher_forced | online_pre_sample | online_post_sample
```

### 2. Preserve original model behavior

Before adding masks or offloading, verify that the adapter reproduces the unmodified model:

- same next-token logits within configured tolerance;
- same router top-\(k\) indices;
- same router weights;
- deterministic output under fixed seed and greedy decoding.

A model adapter is not accepted until these tests pass.

### 3. Build a tiny MoE test model

Do not make unit tests download a large model. Implement a deterministic tiny causal MoE model with:

- configurable layers, experts, hidden size, and top-\(k\);
- a router linear layer;
- expert MLPs;
- explicit router logits and expert outputs;
- deterministic generation;
- optional synthetic transfer latency.

Use it to test every algorithmic contract.

### 4. Simulator before real offloading

The first performance result must come from a trace-driven simulator. A simulator result must be labeled as simulated and must never be presented as an end-to-end speedup.

Only implement the real offload engine after:

- oracle closed-loop feasibility is demonstrated;
- cache behavior is validated against hand-computed examples;
- transfer accounting tests pass;
- probe overhead is measured.

### 5. No hidden fallback

Execution semantics must be an explicit enum:

```python
class MissPolicy(Enum):
    LOSSLESS_FALLBACK = "lossless_fallback"
    SUBSTITUTE = "substitute"
    TRUNCATE = "truncate"
    EARLY_TERMINATE_WINDOW = "early_terminate_window"
    CPU_EXECUTE = "cpu_execute"
```

Never change policy automatically without logging it.

### 6. Avoid model-name conditionals in core logic

Use a model-adapter registry. Architecture-specific code belongs under `src/pseudoroute/models/adapters/`.

### 7. Make expensive traces optional

Router logits and top-\(k\) traces are mandatory. Full router inputs, post-attention residuals, attention queries, and expert outputs are opt-in because they are large.

### 8. Configuration drives every experiment

No paper experiment may depend on hard-coded constants. All runs must save a resolved config, environment manifest, git commit, random seeds, and hardware profile.

### 9. Test numerical and semantic invariants

Required tests include:

- subset size never exceeds the budget;
- selected experts belong to the correct layer;
- no cross-layer expert-ID aliasing;
- online probes never access future arrays;
- H2D bytes equal the sum of loaded expert bytes;
- resident-set capacity is never exceeded;
- a lossless fallback run matches the base model;
- an oracle subset computed from a window is optimal for the specified additive score;
- top-\(k\) stability bound is correct on synthetic logits;
- pseudo shadow cache cannot mutate the production KV cache.

### 10. Optimize only after profiling

Avoid Triton/CUDA extensions in the initial milestones. Establish correctness with PyTorch. Add optimized kernels only when a profiler shows a material bottleneck.

### 11. Download models and datasets when needed

Codex agents may download milestone-relevant models and datasets when local disk space permits. Before downloading, estimate the download and expanded-cache sizes, check available space, and leave enough headroom for traces and experiment artifacts. Reuse a configurable cache, pin model and dataset revisions where supported, and record source identifiers, revisions, cache location, and fingerprints in run manifests. Do not commit downloaded weights, raw datasets, credentials, or access tokens to Git. Large external-model tests must remain optional/skippable in CI.

## Initial assumptions

These are defaults, not universal truths:

- standard decoder-only, pre-norm, RoPE-based MoE;
- top-\(k\) routing;
- one expert subset per MoE layer;
- a global window boundary for all MoE layers;
- batch size 1;
- expert weights live in CPU memory when not resident;
- dense attention and non-expert weights remain on GPU if memory permits;
- the core online method uses no speculative draft model;
- the primary method is training-free;
- predictor training is allowed only for baselines or later extensions.

All assumptions must be configurable or isolated behind an interface.

## Milestone order

Implement in this order:

1. **M0 — Repository and tiny model**
2. **M1 — Model introspection and exact trace collection**
3. **M2 — Oracle window analysis**
4. **M3 — Closed-loop constrained routing**
5. **M4 — DapQ-style routing factorial analysis**
6. **M5 — Router geometry and expert criticality**
7. **M6 — Deployable probe baselines**
8. **M7 — Position-conditioned pseudo routing probes**
9. **M8 — Cost-aware subset selection and cache simulator**
10. **M9 — Static + dynamic residency**
11. **M10 — Real offload prototype**
12. **M11 — Full benchmark and paper artifacts**

Do not skip M2 or M3. A predictor is not useful if the oracle hard-subset method is intrinsically too damaging.

## Required command surface

The final repository must expose a single CLI entry point named `pseudoroute` with at least:

```bash
pseudoroute inspect-model
pseudoroute collect-traces
pseudoroute oracle-sweep
pseudoroute closed-loop-eval
pseudoroute dapq-factorial
pseudoroute analyze-router
pseudoroute build-default-vectors
pseudoroute evaluate-probe
pseudoroute train-predictor
pseudoroute simulate-offload
pseudoroute benchmark-offload
pseudoroute aggregate-results
```

Each command must support `--config`, `--output-dir`, and `--dry-run`.

## Definition of done

The repository is complete when:

- all required commands exist and are documented;
- unit and integration tests pass;
- at least one real MoE model adapter works;
- oracle and closed-loop sweeps run end to end;
- the DapQ-style routing analysis produces per-layer/per-horizon results;
- at least four deployable baselines and two pseudo-routing probes are implemented;
- the simulator reports exact memory and transfer accounting;
- the real offload prototype runs on one GPU with CPU-resident experts;
- all metrics are aggregated into reproducible tables;
- no online result uses future information;
- limitations and negative findings are reported rather than hidden.

## Codex behavior during implementation

At the beginning of each milestone:

1. Read the relevant sections of all specifications.
2. Add a file-level checklist to `STATUS.md`.
3. Implement the smallest correct vertical slice.
4. Add tests before optimization.
5. Run formatting, type checking, and tests.
6. Update `STATUS.md` with exact results and remaining limitations.

When a paper detail is ambiguous or unavailable, implement the documented *style of baseline*, label it clearly, and do not claim exact reproduction.

<!-- END FILE: 00_CODEX_START_HERE.md -->


---

<!-- BEGIN FILE: 01_RESEARCH_SPEC.md -->

# Research Specification

## 1. Problem statement

Large MoE language models activate only a small number of experts per token, but all expert weights may still exceed available HBM/VRAM. A common deployment stores expert weights in CPU DRAM and moves selected experts to GPU memory on demand. Decode-time expert churn can expose PCIe or interconnect latency on the critical path.

This project studies a deliberately lossy alternative:

1. At a decode-window boundary \(i\), estimate expert demand for future steps \(i+1,\ldots,i+t\).
2. Select a bounded expert subset \(S_{\ell,i}\) for each MoE layer \(\ell\).
3. Load the subset once and keep it resident during the window.
4. Restrict, approximate, or terminate routing when the natural route requires experts outside the subset.
5. Re-plan at the next boundary or earlier when a confidence/coverage signal deteriorates.

The goal is to reduce transfer frequency and tail latency while retaining acceptable model quality.

## 2. Formal model

Let:

- \(L\): number of MoE layers;
- \(N_\ell\): number of experts in layer \(\ell\);
- \(k_\ell\): experts naturally activated per token;
- \(p_{\ell,j,e}\): natural router probability for expert \(e\) at token \(j\);
- \(E_{\ell,j}\): natural top-\(k_\ell\) expert set;
- \(C_{\ell,i}\): experts resident at the decision boundary;
- \(B_\ell\): resident expert budget for layer \(\ell\);
- \(t\): planned window length;
- \(S_{\ell,i}\): committed expert subset for the window.

The natural route is

\[
E_{\ell,j}
=
\operatorname{TopK}_{e\in\mathcal E_\ell}
p_{\ell,j,e}.
\]

A hard-subset route is

\[
\widetilde E_{\ell,j}
=
\operatorname{TopK}_{e\in S_{\ell,i}}
p_{\ell,j,e},
\qquad
j=i+1,\ldots,i+t.
\]

The HBM constraint is

\[
M_{\mathrm{dense}}
+
M_{\mathrm{KV}}
+
M_{\mathrm{workspace}}
+
\sum_{\ell=1}^{L}\sum_{e\in S_{\ell,i}}
\operatorname{Bytes}_{\ell,e}
\le
M_{\mathrm{HBM}}.
\]

If double buffering is enabled, include next-window staging memory explicitly.

## 3. Decision-time regimes

The implementation must distinguish two online regimes.

### 3.1 `online_pre_sample`

The subset decision begins before the next token \(x_{i+1}\) is sampled. Available information includes the current prefix, current hidden/router states, and current KV cache. This regime offers the largest opportunity to overlap expert loading with remaining computation, but lacks the identity of the next token.

### 3.2 `online_post_sample`

The model has completed step \(i\), produced logits, and sampled \(x_{i+1}\), but has not executed the transformer layers for \(x_{i+1}\). The next token embedding is available and may seed a pseudo rollout. Future tokens \(x_{i+2:}\) remain unavailable.

Results from these regimes must never be merged without a label.

## 4. Primary proposed abstraction

The method should predict **window-level expert utility**, not necessarily an exact sequence of future top-\(k\) routes.

A general utility target is

\[
u_{\ell,e}^{(i,t)}
=
\sum_{r=1}^{t}
\gamma^{r-1}
q_{\ell,i+r,e},
\]

where \(q\) may be:

- router probability;
- selected routing weight;
- binary activation;
- routing weight multiplied by an estimated expert-output norm;
- routing weight multiplied by a layer/expert sensitivity score.

A deployable pseudo-routing probe estimates

\[
\widehat u_{\ell,e}^{(i,t)}
=
\Psi_{\ell,e}
\left(
\mathcal H_i,
p_i,
t,
\text{cache state}
\right).
\]

The selector solves

\[
S_{\ell,i}
=
\arg\max_{|S|\le B_\ell}
\left[
\sum_{e\in S}\widehat u_{\ell,e}^{(i,t)}
-
\lambda
\sum_{e\in S\setminus C_{\ell,i}}
\operatorname{LoadCost}_{\ell,e}
+
\eta
\sum_{e\in S\cap S_\ell^{\mathrm{static}}}
R_{\ell,e}
\right].
\]

When all expert sizes are equal and there are no pairwise terms, this reduces to top-\(B_\ell\) by adjusted score.

## 5. Pseudo routing state hypothesis

DapQ shows that future positional information can be more important than semantic content for approximating future **RoPE-transformed attention queries**. This project tests whether a related construction helps approximate future **MoE router inputs or router outputs**.

For a standard pre-norm MoE layer:

\[
a_{\ell,j}
=
h_{\ell,j}
+
\operatorname{Attention}_\ell
\left(
\operatorname{Norm}(h_{\ell,j}),
KV_{\ell,\le j},
p_j
\right),
\]

\[
s_{\ell,j}
=
\operatorname{Norm}(a_{\ell,j}),
\]

\[
z_{\ell,j}
=
W_\ell^R s_{\ell,j}+b_\ell.
\]

The router usually does not consume position IDs directly. Position affects routing indirectly through attention and residual-state evolution. Therefore the research target is a pseudo state

\[
\widetilde s_{\ell,i+r}
=
\Phi_{\ell,r}
\left(
x_{\le i},
KV_{\le i},
\mathcal H_i,
p_{i+r}
\right)
\]

whose router projection

\[
\widetilde z_{\ell,i+r}
=
W_\ell^R\widetilde s_{\ell,i+r}+b_\ell
\]

matches the future route well enough for window-level subset selection.

## 6. Research questions

### RQ1 — Oracle feasibility

Under a given \(B_\ell\) and \(t\), does an oracle fixed subset preserve closed-loop generation quality while reducing expert transfers?

### RQ2 — Position versus content

For future MoE routing, which signal contributes more:

- future position;
- future token content;
- prefix/context;
- current routing history;
- preceding expert-output trajectory?

### RQ3 — Router-visible state

Can future routing be predicted by approximating only the subspace visible to the router rather than the full hidden state?

### RQ4 — Window utility versus exact route

Is aggregate window-level utility easier and more useful to predict than exact per-token top-\(k\) expert IDs?

### RQ5 — Static residency

Do some experts warrant permanent residency because they are frequent, quality-critical, downstream-routing-critical, unpredictable, or expensive to miss?

### RQ6 — System value

After including pseudo-probe cost, loading overlap, cache capacity, and boundary stalls, does the method improve end-to-end TPOT and tail latency?

### RQ7 — Generality

How do results vary across model architecture, layer depth, expert count, top-\(k\), task, context length, and batch size?

## 7. Testable hypotheses

### H1 — Oracle local consistency

For at least some models and tasks, a subset with size near a small multiple of \(k_\ell\) can cover a useful multi-token window with modest closed-loop quality loss.

### H2 — Position-conditioned routing signal

Correct future positions improve future router-logit or top-\(k\) prediction over a matched-content, wrong-position condition for a nontrivial set of layers/horizons.

This is explicitly falsifiable. A negative result is valid.

### H3 — Layer heterogeneity

Early, middle, and late layers respond differently to position, content, context, and route history. A single global pseudo-state construction is unlikely to be optimal.

### H4 — Router-visible compression

A low-dimensional representation aligned with router weights predicts future routing as well as or better than full-hidden-state similarity metrics at lower storage and compute cost.

### H5 — Aggregate utility robustness

Predicting

\[
\sum_r p_{\ell,i+r,e}
\]

is more stable than predicting every future top-\(k\) set independently.

### H6 — Static plus dynamic split

A permanent set selected using frequency, miss risk, quality sensitivity, and downstream-routing influence improves the Pareto frontier over purely dynamic selection.

### H7 — Adaptive horizon

A receding-horizon policy with early termination is more robust than a fixed \(t\) under topic, sentence, code-block, or reasoning-phase transitions.

## 8. Main experimental variants

### 8.1 Lossless predictive prefetch

The predictor controls only loading. The natural router remains authoritative. Missing experts are loaded on demand.

Purpose: ADEPT-style upper-quality baseline.

### 8.2 One-step hard commitment

A prediction is made every step or layer, and the predicted experts are executed without fallback.

Purpose: CommitMoE/Speculating-Experts-style execution baseline.

### 8.3 Multi-step hard commitment

One per-layer subset persists for \(t\) decode steps.

Purpose: primary method.

### 8.4 Multi-step adaptive commitment

The subset persists until \(t_{\max}\), but terminates early when confidence or out-of-subset mass crosses a threshold.

Purpose: robust primary extension.

### 8.5 Oracle commitment

The subset is selected using true future routing.

Purpose: feasibility upper bound, never deployable.

## 9. Miss semantics

The project must implement and compare:

### Substitution

Select the top-\(k\) experts among resident experts and renormalize their weights.

### Truncation

Execute only naturally selected experts that are resident. Define whether weights are preserved or renormalized.

### Lossless fallback

Load the missing natural expert, preserving base-model semantics.

### CPU execution

Execute the missing expert on CPU without loading it into GPU cache.

### Early window termination

Abort the current subset, load/re-plan, and continue.

Each policy must be evaluated independently. Do not combine them without an explicit hybrid policy configuration.

## 10. Static expert categories

Do not equate router-weight norm with expert importance. Track separate quantities:

1. **Hotness**
   \[
   F_{\ell,e}=P(e\text{ is selected})
   \]

2. **Quality sensitivity**
   \[
   Q_{\ell,e}
   =
   \mathbb E
   \left[
   D_{\mathrm{KL}}
   \left(
   P_{\mathrm{base}}
   \parallel
   P_{\text{remove/substitute }e}
   \right)
   \right]
   \]

3. **Downstream routing influence**
   \[
   D_{\ell,e}^{\mathrm{route}}
   =
   \mathbb E
   \sum_{d=1}^{H}
   \alpha_d
   D_{\mathrm{KL}}
   \left(
   p_{\ell+d}^{\mathrm{base}}
   \parallel
   p_{\ell+d}^{(-e)}
   \right)
   \]

4. **Prediction miss risk**
   \[
   U_{\ell,e}
   =
   P(\text{predictor misses }e\mid e\text{ is needed})
   \]

5. **Transfer cost**
   \[
   C_{\ell,e}^{\mathrm{load}}
   \]

A static residency score may combine them, but every term must also be reported separately.

## 11. Primary optimization objective

The paper-level objective is not classification accuracy. It is a constrained multi-objective tradeoff:

\[
\min
\quad
\Delta Q
+
\lambda_1 D_{\mathrm{H2D}}
+
\lambda_2 T_{\mathrm{stall}}
+
\lambda_3 T_{\mathrm{probe}}
+
\lambda_4 M_{\mathrm{peak}}
\]

subject to:

\[
M_{\mathrm{peak}}\le M_{\mathrm{HBM}}.
\]

Required reporting should show a Pareto frontier rather than a single hand-picked operating point.

## 12. Go/no-go gates

### Gate A — Oracle feasibility

Proceed to complex prediction only if at least one practical \((B,t)\) region shows meaningful transfer reduction and acceptable closed-loop quality.

Default exploratory threshold, configurable per task:

- at least 30% fewer expert-load bytes/token than on-demand;
- no more than 5% relative perplexity increase or 2 percentage-point task accuracy loss;
- no catastrophic repetition or output collapse.

These are research defaults, not universal claims.

### Gate B — Position hypothesis

Continue to position-centric method development only if future-position conditioning improves router-level or subset-level metrics over equal-cost non-position baselines for meaningful layers/horizons.

If not, retain the factorial analysis as a negative result and pivot to context/history-conditioned pseudo routing.

### Gate C — Runtime economics

A deployable method must satisfy

\[
T_{\mathrm{probe}}
+
T_{\mathrm{exposed\ boundary\ load}}
<
T_{\mathrm{avoided\ demand\ stalls}}
\]

on real hardware, not only in the simulator.

## 13. Non-goals for the first version

- distributed expert parallelism;
- multi-node serving;
- high-concurrency continuous batching;
- training a new MoE from scratch;
- claiming universal position dominance;
- exact reproduction of every related system;
- custom CUDA kernels before the PyTorch implementation is validated;
- using speculative decoding as a required component;
- hiding quality loss behind only perplexity or only route recall.

## 14. Expected scientific contributions, conditional on results

Only claim contributions that experiments support. Candidate contributions are:

1. A systematic position/content/context/history decomposition of future MoE routing.
2. A router-visible pseudo-state analysis inspired by DapQ but adapted to MoE routing geometry.
3. A training-free horizon-level expert-utility estimator.
4. A memory- and load-cost-aware hard commitment policy.
5. A static/dynamic residency framework based on causal routing and quality sensitivity.
6. An end-to-end offloading prototype and analysis of when lossy window commitment helps or fails.

Do not claim that multi-token persistent expert masks, expert budgeting, fallback-free expert execution, or predictive expert prefetching are new by themselves.

<!-- END FILE: 01_RESEARCH_SPEC.md -->


---

<!-- BEGIN FILE: 02_RELATED_WORK_AND_NOVELTY.md -->

# Related Work and Novelty Boundaries

## 1. Purpose

This document prevents the project from accidentally re-claiming existing ideas and maps each related method to a baseline or design component.

The project combines ideas from:

- future-position pseudo queries for KV-cache importance;
- persistent expert masks;
- expert budgeting over a multi-token group;
- fallback-free predicted expert execution;
- hidden-state-based expert speculation;
- oracle segment caching;
- predictive expert prefetching.

The combination may be novel only if the actual implementation and experiments establish a distinct deployable method. The documents do not assert novelty in advance.

## 2. Comparison table

| Work | How experts/objects are selected | Training | Execution semantics | Relationship to this project |
|---|---|---:|---|---|
| **DapQ** | Appends synthetic pseudo tokens with future position IDs; uses their RoPE-aware attention queries to score prompt KV importance | No gradient training | KV eviction, not expert routing | Inspiration for future-position-conditioned pseudo routing; its position-dominance result must be re-tested for routers |
| **Temporally Extended MoE** | Per-layer learned controller decides whether to keep the current mask; when switching, a learned selection head samples a new fixed-size expert mask | Yes; controller plus lightweight model adaptation | Router is constrained to active mask across tokens | Closest conceptual prior for temporally persistent expert subsets |
| **MoE-Spec** | Aggregates router probabilities across a speculative draft tree and takes top-\(B\) experts per layer | Training-free | Verification loads only the budgeted subset; uses truncation or substitution | Closest prior for group-level aggregate router-mass budgeting |
| **CommitMoE** | Lightweight Commit Router predicts experts and unconditionally executes predicted experts | Predictor training | No fallback; predicted experts replace natural activation experts | Baseline for prediction-as-commitment |
| **Speculating Experts** | Builds a quasi-hidden state from the post-attention residual and expert-conditioned default vector, then applies the next-layer router; optional learned estimator | Main method calibration-only; optional estimator training | Can execute speculated experts to avoid refetch | Baseline and building block for cheap shadow-state propagation |
| **Not All Models Suit Expert Offloading** | Uses true future segment routing to choose the most frequently activated experts under a fixed segment cache size | No training; offline trace analysis | Oracle metric, not deployable | Oracle local-consistency and cache-hit upper bound |
| **ADEPT** | Domain-aware preloading plus a routing-history/locality predictor for upcoming experts | Calibration plus Random Forest training | Predicted experts are prefetch hints; natural router remains authoritative | Lossless predictive-prefetch baseline |

## 3. DapQ

**Title:** *Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries*  
**Primary source:** https://arxiv.org/abs/2603.11564

DapQ appends a synthetic pseudo context of length \(N\) to the prompt and assigns those tokens the positions of the first \(N\) future decode steps:

\[
[L_p,L_p+1,\ldots,L_p+N-1].
\]

The resulting pseudo attention queries score original prompt keys for KV-cache eviction. Its central observation is that correct future positions can preserve query similarity and attention importance even when pseudo-token content is incorrect.

### What is transferable

- A future-position-conditioned synthetic observation window.
- Content-versus-position factorial interventions.
- Independent evaluation of pre- and post-RoPE representations.
- The idea that exact future token generation may not be necessary to estimate future resource utility.

### What is not directly transferable

- The MoE router usually receives a post-attention hidden state, not a RoPE-transformed query.
- Position affects routing through attention and residual evolution rather than a direct router input.
- DapQ is performed during prefill, whereas this project may invoke probes periodically during decode.
- DapQ scores old KV entries; this project predicts future expert utility and may alter model execution.

### Required baseline/ablation

Reproduce the SC/SP, DC/SP, SC/DP, and DC/DP design for router outputs. Do not assume the same ordering as DapQ.

## 4. Temporally Extended Mixture-of-Experts Models

**Primary source:** https://arxiv.org/abs/2604.20156

Each MoE layer has an active fixed-size expert mask \(\omega_t^{(\ell)}\). A learned controller decides whether to retain the mask or terminate it. When a switch occurs, a selection head initialized from router weights produces candidate logits and samples a new expert mask. Experts outside the active mask are assigned \(-\infty\) before top-\(k\) routing.

### Overlap

- Expert subsets persist across multiple tokens.
- Router choices are constrained by the resident subset.
- The method explicitly trades model capability against switching cost.

### Differences to target design

- The target core is training-free.
- The target plans a future horizon from pseudo routing states.
- The target objective includes measured HBM capacity and transfer cost.
- The initial target uses a shared global window boundary rather than independent per-layer termination.
- The target includes a complete trace simulator and real CPU–GPU offload prototype.
- Adaptive termination is an extension, not the only operating mode.

### Novelty claim to avoid

Do not claim the first multi-token persistent expert subset or the first expert mask with switching cost.

## 5. MoE-Spec

**Primary source:** https://arxiv.org/abs/2602.16052

Given a speculative draft tree, MoE-Spec aggregates target-router probabilities over all nodes:

\[
s_{\ell,e}
=
\sum_{j\in\text{draft tree}}
p_{\ell,j,e}
\]

and chooses top-\(B\) experts per layer. It imposes a strict verification-time expert budget and handles unavailable natural routes using truncation or substitution.

### Overlap

- Aggregate routing mass over a multi-token group.
- Fixed per-layer expert budget.
- Lossy execution outside the selected subset.
- Training-free subset selection.

### Difference

MoE-Spec already has draft-tree token representations. The target project predicts a future autoregressive window when most future tokens and hidden states are unavailable. The target does not require speculative decoding.

### Baseline mapping

The oracle future-window top-\(B\) router-mass selector should match the MoE-Spec-style aggregate score, but operate on a linear autoregressive window.

## 6. CommitMoE

**Primary source:** https://ojs.aaai.org/index.php/AAAI/article/view/39454

CommitMoE introduces a Commit Router and eliminates fallback by unconditionally adopting predicted expert selections. The predicted experts are executed instead of loading the natural router's missed experts.

### Overlap

- Prediction has execution authority.
- Prediction errors alter model execution.
- Avoids serial demand-loading after an incorrect prefetch.

### Difference

The target primary method commits a subset for multiple future decode steps and predicts window utility. CommitMoE focuses on lightweight expert prediction and fallback-free execution, primarily over shorter layer/step relationships.

### Baseline mapping

Implement:

- one-step learned predictor;
- one-step no-fallback execution;
- router-certainty stratification;
- optional output-weight adjustment only after the basic baseline is correct.

Do not label an approximate implementation as exact CommitMoE unless every reported detail is reproduced.

## 7. Speculating Experts Accelerates Inference for MoE

**Primary source:** https://arxiv.org/abs/2603.19289

The method computes an offline expert-specific default vector \(d_{\ell,e}\), representing average expert activation contribution. For a token, selected expert default vectors are gate-weighted:

\[
d_\ell
=
\sum_{e\in E_\ell}
g_{\ell,e}d_{\ell,e}.
\]

A quasi-hidden state approximates the next-layer router input:

\[
q_\ell
=
\operatorname{Norm}_{\ell+1}(r_\ell+d_\ell),
\]

where \(r_\ell\) is the post-attention residual. The next-layer router is applied to \(q_\ell\). An optional shallow estimator improves high-drift layers.

### Overlap

- Approximate a future router input rather than loading experts reactively.
- Use expert-conditioned average vectors to propagate a cheap state.
- Execute speculated experts without necessarily falling back.

### Difference

The paper primarily predicts the next layer for the same token. The target extends cheap state propagation across future positions and aggregates utility over multiple decode steps.

### Baseline/building block

Implement default-vector collection and same-token next-layer prediction before using the vectors in a future-token shadow rollout.

## 8. Not All Models Suit Expert Offloading

**Primary source:** https://arxiv.org/abs/2505.16056

The paper defines local routing consistency metrics. Segment Cache Best Hit Rate (SCH) chooses, for every future segment, the cache-limited set of experts with highest true activation counts. It is an oracle upper bound for any segment-static cache.

### Overlap

- Fixed subset across a token segment.
- Direct evaluation of whether a model is structurally suitable for segment-level caching.
- Per-model and per-layer heterogeneity.

### Difference

It is an analysis framework, not an online predictor and not a constrained-generation method.

### Baseline mapping

Implement:

- activation-count SCH;
- routing-mass-weighted SCH extension;
- per-layer and whole-model curves over segment length and cache ratio;
- oracle closed-loop generation, which goes beyond the trace-only metric.

## 9. ADEPT

**Title:** *Two-Stage Expert Offloading for Domain-Aware MoE Inference*  
**Primary source:** https://ieeexplore.ieee.org/document/11397596/

ADEPT combines prompt-domain-aware preloading with a locality/routing-history predictor for decode-time expert prefetching. Its predicted expert set is a cache/loading hint; the natural router remains authoritative and prediction misses may require demand loading.

### Overlap

- Future expert prediction.
- CPU–GPU expert offloading.
- Routing-history features.
- Cache-aware loading.

### Difference

The target's hard subset can constrain routing and persist across multiple tokens. ADEPT aims to hide transfer without changing model semantics.

### Baseline mapping

Implement an `ADEPTStyleRFProbe`, clearly labeled as style-compatible unless exact code and features are reproduced. Candidate features:

- current/previous same-layer top-\(k\) expert IDs;
- expert recency;
- rolling activation frequency;
- co-activation mask or counts;
- layer index;
- optional prompt-domain embedding/cluster.

Compare both:

- RF prediction + lossless fallback;
- the same RF output used as a hard multi-step commitment.

This isolates execution semantics from predictor quality.

## 10. Required comparison taxonomy

Use the following taxonomy in code, plots, and writing.

### Prediction as hint

The prediction affects prefetch/cache state but not final expert execution.

Examples: ADEPT-style, lossless Speculating-Experts variant.

### Prediction as one-step commitment

Predicted experts are directly executed for one target layer or token.

Examples: CommitMoE-style, fallback-free Speculating Experts.

### Temporally persistent commitment

A subset remains authoritative for multiple decode steps.

Examples: Temporally Extended MoE and this project's main execution setting.

### Group-level expert budgeting

Experts are ranked by aggregate utility over multiple token candidates or positions.

Example: MoE-Spec and this project's oracle/window utility selector.

### Oracle consistency analysis

True future routing is used only to measure feasibility.

Example: SCH and this project's oracle sweep.

## 11. What the project may claim only after evidence

Potentially defensible distinctions include:

- applying DapQ-style future-position probes to MoE router outputs;
- analyzing position/content/context/routing-history contributions to future routing;
- router-visible future-state approximation;
- training-free future-window expert utility prediction in ordinary autoregressive decode;
- measured HBM-aware hard commitment with a real offloading runtime;
- static residency based on downstream routing causality and miss risk.

Each statement must be narrowed to the exact models, tasks, and information regime evaluated.

## 12. Claims that must not appear

Do not claim:

- the first expert predictor;
- the first MoE offloading system;
- the first fallback-free expert prediction;
- the first multi-token fixed expert subset;
- the first training-free expert budget;
- the first analysis of local routing consistency;
- that position is universally more important than semantics for routing;
- that simulated transfer savings equal real speedup.

<!-- END FILE: 02_RELATED_WORK_AND_NOVELTY.md -->


---

<!-- BEGIN FILE: 03_SYSTEM_ARCHITECTURE.md -->

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

<!-- END FILE: 03_SYSTEM_ARCHITECTURE.md -->


---

<!-- BEGIN FILE: 04_ALGORITHMS.md -->

# Algorithms

## 1. Notation

For MoE layer \(\ell\), token position \(j\), and expert \(e\):

- \(h_{\ell,j}\): layer input or residual state, architecture-dependent;
- \(s_{\ell,j}\): exact router input;
- \(W_\ell^R,b_\ell^R\): router parameters;
- \(z_{\ell,j}=W_\ell^R s_{\ell,j}+b_\ell^R\): router logits;
- \(p_{\ell,j}=\operatorname{softmax}(z_{\ell,j})\): router distribution before model-specific top-\(k\) normalization;
- \(E_{\ell,j}\): natural top-\(k\) expert IDs;
- \(g_{\ell,j,e}\): executed gate weight;
- \(S_{\ell,i}\): expert subset committed at boundary \(i\);
- \(B_\ell\): subset budget;
- \(t\): maximum horizon;
- \(r\in\{1,\ldots,t\}\): relative horizon.

Model-specific adapters must document deviations such as sigmoid routing, group routing, routing bias, shared experts, or top-\(k\) renormalization.

## 2. Exact baseline route extraction

For every adapter, define one canonical `RouteResult`:

```python
@dataclass
class RouteResult:
    raw_logits: Tensor
    pre_topk_scores: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
```

The adapter, not generic code, is responsible for reproducing the model's exact routing rule.

## 3. Oracle window subset

### 3.1 Binary-count oracle

For window \(w=(i+1,\ldots,i+t)\):

\[
c_{\ell,e}
=
\sum_{r=1}^{t}
\mathbf 1[e\in E_{\ell,i+r}].
\]

Select:

\[
S_{\ell,i}^{\mathrm{count}}
=
\operatorname{TopB}_{e} c_{\ell,e}.
\]

This matches the intuition behind segment cache hit rate.

### 3.2 Routing-mass oracle

\[
m_{\ell,e}
=
\sum_{r=1}^{t}
\gamma^{r-1}
\widehat g_{\ell,i+r,e},
\]

where \(\widehat g\) is zero for unselected experts or uses the full router probability, depending on the configured target.

Select:

\[
S_{\ell,i}^{\mathrm{mass}}
=
\operatorname{TopB}_{e} m_{\ell,e}.
\]

This is the primary oracle for comparison with predicted utility.

### 3.3 Contribution-weighted oracle

When expert-output norms are available:

\[
a_{\ell,e}
=
\sum_{r=1}^{t}
\gamma^{r-1}
g_{\ell,i+r,e}
\left\|
E_{\ell,e}(s_{\ell,i+r})
\right\|_2.
\]

This is more expensive and used only for analysis.

### 3.4 Reconstruction oracle

Greedily choose experts that minimize MoE output reconstruction error across the window. This is non-additive and should be labeled separately.

### 3.5 Cost-aware oracle

When the current resident set matters:

\[
\operatorname{score}_{\ell,e}
=
m_{\ell,e}
-
\lambda
\mathbf 1[e\notin C_{\ell,i}]
C_{\ell,e}^{\mathrm{load}}.
\]

For variable-size experts, solve a 0/1 knapsack or use a documented greedy approximation.

## 4. Oracle pseudocode

```text
for each sample:
    obtain teacher-forced future router trace
    for each boundary i:
        for each horizon t:
            for each layer l:
                score each expert over positions i+1 ... i+t
                choose budgeted subset S[l]
            record open-loop coverage metrics
            optionally run closed-loop generation from boundary i
                using S for the next t generated steps
```

Closed-loop oracle must be interpreted carefully: its subset is selected from the base-model future trajectory, but constrained execution may diverge. It is an optimistic planner, not an omniscient oracle over the altered trajectory.

Optionally implement an iterative oracle that re-plans from the constrained trajectory at every boundary.

## 5. Local-routing-consistency metrics

### 5.1 Segment hit rate

\[
\operatorname{HitRate}
=
\frac{
\sum_{\ell,r}
|E_{\ell,i+r}\cap S_{\ell,i}|
}{
\sum_{\ell,r}
|E_{\ell,i+r}|
}.
\]

### 5.2 Routing-mass coverage

\[
\operatorname{MassCoverage}
=
\frac{
\sum_{\ell,r}
\sum_{e\in S_{\ell,i}}
q_{\ell,i+r,e}
}{
\sum_{\ell,r}
\sum_e
q_{\ell,i+r,e}
}.
\]

Specify whether \(q\) is full router probability or selected gate mass.

### 5.3 Expert-union size

\[
U_\ell(t)
=
\left|
\bigcup_{r=1}^{t}E_{\ell,i+r}
\right|.
\]

Report its distribution by layer and task.

### 5.4 SCH-style curve

For budget ratio \(\rho\):

\[
B_\ell=\lceil \rho k_\ell\rceil.
\]

Plot best hit rate over segment length and \(\rho\).

## 6. DapQ-style factorial analysis for routing

### 6.1 Offline sample construction

Given a real sequence and a prefix ending at position \(i\), define a pseudo future span of length \(N\).

Four core conditions:

1. **SC/SP** — same future token content, same/correct future positions.
2. **DC/SP** — different content, same/correct future positions.
3. **SC/DP** — same content, different positions.
4. **DC/DP** — different content, different positions.

Content variants:

- random vocabulary tokens;
- repeated current token;
- tokens sampled from the prefix;
- prefix suffix;
- a fixed nonsensical sequence;
- a learned/null token only in a separate trained extension.

Position variants:

- correct future positions;
- reset positions starting at zero;
- current-position repetition if architecture permits;
- a random contiguous position span;
- shuffled future positions;
- fixed offset \(\Delta\);
- same horizon but globally shifted prefix and pseudo positions.

### 6.2 Context and route-history interventions

Add:

- **context swap:** same pseudo tokens and positions, different prefix;
- **route-history swap:** replace earlier expert contributions with default vectors or alternate experts while keeping token/position controls;
- **attention-only probe:** isolate the state immediately after attention;
- **router-only projection:** compare only \(W_\ell^R s\).

### 6.3 Ground truth

Ground-truth future routing must be computed using teacher forcing on the original sequence with natural routing.

### 6.4 Metrics

Per layer and horizon:

- full-state cosine;
- router-visible-state cosine;
- router-logit cosine;
- logit MSE;
- probability KL and Jensen–Shannon divergence;
- top-\(k\) recall and Jaccard;
- expert rank correlation;
- top-\(B\) window utility recall;
- routing-mass coverage;
- top-\(k\) margin-normalized error.

Define position dominance for metric \(M\), where higher is better:

\[
D_{\ell,r}^{\mathrm{pos}}
=
M_{\ell,r}(\mathrm{DC/SP})
-
M_{\ell,r}(\mathrm{SC/DP}).
\]

For divergence metrics, reverse the sign or use a normalized improvement.

### 6.5 Statistical model

Fit a mixed-effects or bootstrap-based decomposition:

\[
M
=
\beta_0
+
\beta_{\mathrm{position}}
+
\beta_{\mathrm{content}}
+
\beta_{\mathrm{context}}
+
\beta_{\mathrm{history}}
+
\text{interactions}
+
\epsilon.
\]

At minimum, report paired bootstrap confidence intervals over examples.

### 6.6 Leakage rule

This entire factorial pipeline is offline analysis. SC conditions use true future tokens and can never be reported as online methods.

## 7. Router-visible subspace analysis

The router is linear in its input:

\[
z=W^R s+b.
\]

Only components of \(s\) in the row space of \(W^R\) affect logits.

### 7.1 Projection

Using the thin SVD

\[
W^R=U\Sigma V^\top,
\]

define a rank-\(r\) projection

\[
P_r=V_rV_r^\top.
\]

The router-visible representation is

\[
c=P_r s.
\]

Because \(N_\ell\ll d\) for many models, the exact row-space rank is at most \(N_\ell\). For prediction, storing logits or \(V_r^\top s\) may be cheaper than storing \(c\).

### 7.2 Metrics

For each \(r\):

- explained router-logit variance;
- future-logit prediction performance;
- storage bytes;
- linear predictor cost;
- top-\(k\) recall.

### 7.3 Practical formulation

A direct compact target is:

\[
y_{\ell,j}=W_\ell^R s_{\ell,j},
\]

possibly centered or whitened. Predict future \(y\) directly instead of reconstructing \(s\).

## 8. Router geometry

### 8.1 Covariance-aware logit variance

Given empirical covariance \(\Sigma_\ell\):

\[
\operatorname{Var}(z_{\ell,e})
=
(w_{\ell,e}^R)^\top
\Sigma_\ell
w_{\ell,e}^R.
\]

### 8.2 Pairwise boundary

Experts \(e\) and \(f\) exchange order when:

\[
(w_{\ell,e}^R-w_{\ell,f}^R)^\top s
+
(b_{\ell,e}-b_{\ell,f})
=0.
\]

Estimate how often active candidates lie near this boundary.

### 8.3 Top-\(k\) margin

\[
m_{\ell,j}
=
z_{\ell,j}^{(k)}
-
z_{\ell,j}^{(k+1)}.
\]

If

\[
\|\widetilde z-z\|_\infty
<
m_{\ell,j}/2,
\]

the top-\(k\) set is guaranteed unchanged. Report the fraction of predictions satisfying this sufficient condition.

### 8.4 Entropy/certainty

Record route entropy, top-1 gap, top-\(k\) margin, and cumulative top-\(B\) mass. Use these for confidence calibration and early termination.

## 9. Deployable baseline probes

### 9.1 Current route reuse

\[
\widehat u_{\ell,e}
=
p_{\ell,i,e}.
\]

Optionally multiply by the horizon length.

### 9.2 Rolling frequency

\[
\widehat u_{\ell,e}
=
\sum_{q=0}^{H-1}
\alpha^q
\mathbf 1[e\in E_{\ell,i-q}].
\]

### 9.3 Rolling routing mass

\[
\widehat u_{\ell,e}
=
\sum_{q=0}^{H-1}
\alpha^q
g_{\ell,i-q,e}.
\]

### 9.4 Markov transition

Estimate:

\[
P(E_{\ell,j+1}\mid E_{\ell,j})
\]

or expert-level transition probabilities from training/calibration traces. Roll the transition matrix forward \(r\) steps and aggregate.

### 9.5 ADEPT-style Random Forest

Features may include:

- previous same-layer expert IDs;
- recency per candidate expert;
- rolling frequency;
- co-activation;
- layer ID;
- prompt-domain cluster;
- router certainty.

Train candidate-wise binary probabilities or multiclass/multilabel outputs. Report data collection and training cost.

### 9.6 Direct linear/MLP predictor

Inputs:

- current router logits;
- compact router-visible coordinates;
- route history summary;
- position/horizon features;
- optional next-token embedding in `post_sample` mode.

Targets:

- per-horizon future router logits;
- per-horizon router probabilities;
- aggregate window utility.

A linear/ridge model is mandatory as a strong low-cost baseline.

## 10. Future-position rephased probe

This method tests the cheapest DapQ analog.

For current token content state \(u_{\ell,i}\), compute the attention query before RoPE:

\[
q_{\ell,i}=W_\ell^Q u_{\ell,i}.
\]

For horizon \(r\), apply the future RoPE phase:

\[
\widetilde q_{\ell,r}
=
R(p_i+r)q_{\ell,i}.
\]

Attend to the existing prefix K/V:

\[
\widetilde o_{\ell,r}
=
\operatorname{Attention}
\left(
\widetilde q_{\ell,r},
K_{\ell,\le i},
V_{\ell,\le i}
\right).
\]

Construct a pseudo post-attention state using the model-specific residual path:

\[
\widetilde a_{\ell,r}
=
\operatorname{ResidualCompose}
\left(
h_{\ell,i},
\widetilde o_{\ell,r}
\right).
\]

Then:

\[
\widetilde s_{\ell,r}
=
\operatorname{RouterNorm}_\ell(\widetilde a_{\ell,r}),
\]

\[
\widetilde z_{\ell,r}
=
W_\ell^R\widetilde s_{\ell,r}+b_\ell.
\]

### Limitations

- Reuses current semantic query content.
- Does not model unknown intervening future tokens.
- Does not naturally propagate across layers.
- Architecture-specific RoPE details must come from the adapter.
- May be invalid for non-RoPE or nonstandard attention.

### Required ablations

- no rephasing/current position;
- wrong position offset;
- correct future position;
- random content query with future position;
- direct current-router-logit baseline.

## 11. Pseudo-token probe

### 11.1 Pseudo content choices

For a horizon anchor \(r\), initialize pseudo embedding \(\widetilde x_{0,r}\) using:

- current token;
- repeated sampled next token (`post_sample` only);
- prefix-sampled token;
- random token;
- top-\(M\) sampled branches;
- expected embedding over top-\(M\) next-token probabilities;
- fixed pseudo token.

Every choice must have an explicit information regime.

### 11.2 Independent probes

Each anchor attends to the real prefix but not to other pseudo tokens. This avoids pseudo-error accumulation and allows parallelism.

### 11.3 Causal pseudo sequence

Pseudo token \(r\) attends to the prefix and pseudo tokens \(1,\ldots,r-1\). This approximates autoregressive progression but may accumulate errors.

Compare both.

## 12. Default-vector shadow rollout

This is the main proposed training-free deep probe.

### 12.1 Offline calibration

For expert \(e\) at layer \(\ell\), collect a default residual contribution:

\[
d_{\ell,e}
=
\mathbb E[
\Delta h_{\ell,e}
\mid e\text{ selected}
].
\]

Support definitions:

- raw expert output;
- gate-weighted expert output;
- expert contribution after model-specific combine/projection.

Store count and variance in addition to the mean.

### 12.2 Shadow layer update

For pseudo anchor \(r\):

1. Run the layer's attention path using future position.
2. Compute pseudo router logits and probabilities.
3. Approximate the MoE contribution:

\[
\widetilde d_{\ell,r}
=
\sum_{e\in \mathcal C_{\ell,r}}
\widetilde g_{\ell,r,e}
d_{\ell,e},
\]

where \(\mathcal C\) may be pseudo top-\(k\), top-\(B\), or all experts.

4. Update the pseudo residual:

\[
\widetilde h_{\ell+1,r}
=
\widetilde a_{\ell,r}
+
\widetilde d_{\ell,r}.
\]

5. Continue to the next layer without evaluating full expert MLPs.

### 12.3 Variants

- zero expert contribution;
- global layer mean;
- selected-expert default vector;
- mixture of all expert defaults;
- low-rank learned correction;
- uncertainty propagation from per-expert default variance.

### 12.4 Output

At each MoE layer, save \(\widetilde p_{\ell,r,e}\). Aggregate over anchors/horizons.

## 13. Horizon anchors

Do not assume one pseudo token per future decode step is economical.

Use an anchor set:

\[
\mathcal A\subseteq\{1,\ldots,t\},
\]

such as:

\[
\{1,t\},\quad
\{1,\lceil t/2\rceil,t\},\quad
\{1,2,4,8\}.
\]

Interpolate or aggregate between anchors. Compare prediction gain against probe FLOPs and latency.

## 14. Utility aggregation

Given predicted probabilities:

\[
\widehat u_{\ell,e}
=
\sum_{r\in\mathcal A}
w_r
\widetilde p_{\ell,r,e}.
\]

Possible weights:

- uniform;
- exponential discount \(w_r=\gamma^{r-1}\);
- interval quadrature weights for sparse anchors;
- learned nonnegative weights;
- load-cost-aware weights.

### 14.1 Top-\(k\)-only utility

\[
\widehat u_{\ell,e}
=
\sum_r
w_r
\mathbf 1[e\in\operatorname{TopK}(\widetilde p_{\ell,r})]
\widetilde g_{\ell,r,e}.
\]

### 14.2 Risk-aware utility

For multiple pseudo-content branches \(b\):

\[
\mu_{\ell,e}
=
\mathbb E_b[\widehat u_{\ell,e}^{(b)}],
\]

\[
\sigma_{\ell,e}
=
\sqrt{\operatorname{Var}_b[\widehat u_{\ell,e}^{(b)}]}.
\]

A conservative score may be:

\[
\operatorname{score}_{\ell,e}
=
\mu_{\ell,e}
+
\beta \sigma_{\ell,e}.
\]

Interpretation: uncertain experts may be included to reduce miss risk. Also test \(\mu-\beta\sigma\) for a precision-oriented policy.

## 15. Static plus dynamic residency

Let \(S_\ell^{\mathrm{static}}\) be permanently resident and \(S_{\ell,i}^{\mathrm{dynamic}}\) be window-specific.

\[
S_{\ell,i}
=
S_\ell^{\mathrm{static}}
\cup
S_{\ell,i}^{\mathrm{dynamic}}.
\]

A candidate static score:

\[
R_{\ell,e}
=
\frac{
F_{\ell,e}
\left(
c_0
+
\lambda_Q Q_{\ell,e}
+
\lambda_D D_{\ell,e}^{\mathrm{route}}
\right)
\left(
1+\lambda_U U_{\ell,e}
\right)
}{
\operatorname{Bytes}_{\ell,e}
}.
\]

Do not tune all weights on the test set. Report single-factor rankings and combined rankings.

## 16. Cost-aware subset selection

### 16.1 Equal-size experts

Adjusted score:

\[
a_{\ell,e}
=
\widehat u_{\ell,e}
-
\lambda_L
\mathbf 1[e\notin C_{\ell,i}]
C_{\ell,e}^{\mathrm{load}}
-
\lambda_E
\mathbf 1[e\in C_{\ell,i}\setminus S]
C_{\ell,e}^{\mathrm{evict}}.
\]

Take top-\(B_\ell\), respecting mandatory static experts.

### 16.2 Variable-size experts

Solve:

\[
\max_{x_e\in\{0,1\}}
\sum_e x_e a_{\ell,e}
\]

subject to:

\[
\sum_e x_e\operatorname{Bytes}_{\ell,e}
\le M_\ell.
\]

Provide exact dynamic programming for small cases and a greedy value-per-byte baseline.

### 16.3 Global cross-layer budget

Later extension:

\[
\max \sum_{\ell,e}x_{\ell,e}a_{\ell,e}
\]

subject to one global HBM budget. This allows allocating more experts to sensitive or unpredictable layers.

The initial implementation may use fixed per-layer budgets.

## 17. Constrained routing policies

### 17.1 Substitution

\[
\widetilde E_{\ell,j}
=
\operatorname{TopK}_{e\in S_{\ell,i}}
z_{\ell,j,e}.
\]

Renormalize according to model semantics.

### 17.2 Truncation

\[
\widetilde E_{\ell,j}
=
E_{\ell,j}\cap S_{\ell,i}.
\]

Support:

- preserve original weights;
- renormalize surviving weights;
- scale output to preserve expected norm.

### 17.3 Similar-expert substitution

Choose resident replacement using:

- router rank;
- router-row similarity;
- expert weight similarity;
- default-vector similarity;
- output similarity on calibration data.

This is an extension, not required for the first vertical slice.

## 18. Adaptive early termination

Even under a hard subset, the natural router logits are cheap to compute because the router is small and resident. Define out-of-subset mass:

\[
q_{\ell,j}^{\mathrm{out}}
=
1-
\sum_{e\in S_{\ell,i}}
p_{\ell,j,e}.
\]

Aggregate:

\[
Q_j^{\mathrm{out}}
=
\sum_\ell \alpha_\ell q_{\ell,j}^{\mathrm{out}}.
\]

Possible termination rules:

- \(Q_j^{\mathrm{out}}>\tau\);
- any sensitive layer exceeds \(\tau_\ell\);
- predicted-vs-natural route disagreement exceeds a threshold;
- top-\(k\) margin is low and a missing expert is near the boundary;
- cumulative out-of-subset mass exceeds a budget.

Log every termination reason.

## 19. Closed-loop generation

At a boundary:

```text
1. Construct DeployableDecodeState.
2. Run the selected future-routing probe.
3. Aggregate future expert utility.
4. Select a per-layer subset under HBM budget.
5. Schedule required expert loads.
6. For each token in the window:
   a. run normal attention and router;
   b. apply configured routing/miss policy;
   c. execute resident experts;
   d. collect natural and executed routes;
   e. evaluate early termination;
   f. generate/sample next token.
7. Re-plan at the next boundary.
```

The generated trajectory is authoritative for subsequent windows. Do not reuse the base-model future trace after divergence.

## 20. Transfer accounting

For window \(w\):

\[
D_w
=
\sum_{\ell}
\sum_{e\in S_{\ell,w}\setminus C_{\ell,w}}
\operatorname{Bytes}_{\ell,e}.
\]

Average bytes/token:

\[
\overline D
=
\frac{\sum_w D_w}{\sum_w t_w}.
\]

Exposed transfer time is not simply bytes/bandwidth. Use the simulator or CUDA events to compute transfer portions not hidden by other work.

## 21. Probe economics

For each method report:

- probe latency;
- probe FLOPs if available;
- temporary memory;
- number of attention queries;
- number of router evaluations;
- CPU/GPU synchronization;
- experts whose loads were avoided;
- exposed stall avoided.

A useful net benefit metric:

\[
\Delta T_{\mathrm{net}}
=
T_{\mathrm{baseline}}
-
\left(
T_{\mathrm{probe}}
+
T_{\mathrm{commit}}
\right).
\]

## 22. Pseudocode for the primary probe and planner

```text
INPUT:
    deployable state at boundary i
    horizon anchors A
    per-layer budgets B
    current resident set C
    static set S_static

shadow_states = initialize_pseudo_states(state, A)

for layer in transformer_layers:
    shadow_states = run_shadow_attention(
        layer,
        shadow_states,
        production_kv_read_only=True,
        future_positions=i+A,
    )

    if layer is MoE:
        pseudo_logits = exact_router(layer, shadow_states)
        pseudo_probs = router_normalize(pseudo_logits)
        record(pseudo_probs)

        default_contribution = mix_default_vectors(
            layer,
            pseudo_probs,
            mode=config.default_vector_mode,
        )
        shadow_states = add_residual(
            shadow_states,
            default_contribution,
        )
    else:
        shadow_states = run_dense_mlp(layer, shadow_states)
        # Optionally approximate dense MLP later; first version may execute it.

for each MoE layer:
    utility = aggregate_over_anchors(pseudo_probs, weights)
    utility = combine_uncertainty_and_static_scores(utility)
    subset = cost_aware_select(
        utility,
        budget=B[layer],
        resident=C[layer],
        mandatory=S_static[layer],
    )

return SubsetPlan
```

The implementation must profile whether running dense MLPs for shadow states is affordable. Provide a configuration to skip, approximate, or execute them.

## 23. Predictor calibration and evaluation

Any learned predictor must split data by prompt/document, not random token positions, to avoid leakage. Prefer train/validation/test datasets or non-overlapping sample IDs.

Calibrate predicted probabilities using validation data if uncertainty is used for selection.

Report:

- training examples/tokens;
- target model trace-collection cost;
- predictor parameters;
- training hardware and duration;
- inference latency;
- performance at equal probe-cost budgets.

## 24. Determinism

For greedy decoding, reruns must reproduce tokens and routes. For sampled decoding:

- record RNG state and seed;
- isolate shadow-probe RNG;
- avoid pseudo sampling changing production RNG unless intended;
- use paired seeds for baseline/method comparison.

<!-- END FILE: 04_ALGORITHMS.md -->


---

<!-- BEGIN FILE: 05_EXPERIMENT_PLAN.md -->

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

<!-- END FILE: 05_EXPERIMENT_PLAN.md -->


---

<!-- BEGIN FILE: 06_IMPLEMENTATION_ROADMAP.md -->

# Implementation Roadmap

## Overview

The roadmap is organized as vertical milestones. Each milestone must leave the repository runnable and tested.

Do not start the optimized offload engine before the oracle and closed-loop milestones are complete.

## M0 — Repository bootstrap and tiny MoE

### Deliverables

- package skeleton;
- typed configuration loading;
- logging and run-directory utilities;
- deterministic tiny MoE model;
- natural generation;
- router trace capture;
- basic test suite;
- CLI skeleton;
- continuous integration configuration.

### Required tests

- deterministic forward/generation;
- exact router top-\(k\);
- expert IDs are layer-scoped;
- device-agnostic tests pass on CPU and the configured accelerator;
- a CUDA smoke test runs when CUDA is available;
- config round-trip;
- run directory is reproducible.

### Acceptance

```bash
pseudoroute inspect-model --config configs/model/tiny_moe.yaml
pytest
```

must pass in a clean environment.

## M1 — Generic model introspection and first real adapter

### Deliverables

- `MoEModelAdapter` interface;
- adapter registry;
- first real model adapter;
- model manifest export;
- route-only and router-logit hooks;
- trace storage and validation;
- exact-output regression test on a short prompt.

### Required tests

- hooks do not change next-token logits;
- captured route equals model's native route;
- trace resume does not duplicate positions;
- trace schema validation catches corruption;
- unsupported architecture fails with actionable message.

### Acceptance

Collect and validate a small real-model trace.

## M2 — Oracle window analysis

### Deliverables

- window iterator;
- count and mass oracle selectors;
- SCH-style metrics;
- expert union analysis;
- open-loop sweep CLI;
- plots and result aggregation.

### Required tests

- synthetic oracle top-\(B\) matches brute force for additive scores;
- segment boundaries are correct;
- horizon does not cross sample boundaries;
- budgets are obeyed;
- metrics match hand calculations.

### Acceptance

Produce a complete oracle sweep on the tiny model and a small real trace.

## M3 — Closed-loop constrained generation

### Deliverables

- routing-policy abstraction;
- mask injection;
- substitution;
- truncation variants;
- lossless fallback reference;
- fixed-window planner using oracle subsets;
- generated-route tracing;
- divergence and quality metrics.

### Required tests

- natural policy exactly matches base model;
- lossless fallback exactly matches base model;
- mask never activates unavailable experts;
- substitution preserves top-\(k\) cardinality when budget permits;
- truncation semantics match config;
- constrained generation re-plans from its own trajectory.

### Acceptance

Generate oracle quality–transfer curves. Decide whether the project passes Gate A.

## M4 — DapQ-style factorial analysis

### Deliverables

- teacher-forced pseudo-sequence constructor;
- content and position intervention library;
- pre-/post-RoPE query capture where supported;
- router-input and router-logit comparison;
- per-layer/per-horizon metrics;
- bootstrap confidence intervals;
- heatmap plots.

### Required tests

- SC/SP reproduces ground truth within tolerance;
- DC/SP changes content but preserves intended positions;
- SC/DP preserves token IDs but changes positions;
- pseudo inputs cannot enter online APIs;
- no sample crosses document boundary.

### Acceptance

Produce the full 2×2 analysis on the tiny model and a smaller real-model sample. Decide whether the project passes Gate B.

## M5 — Router geometry and expert criticality

### Deliverables

- router SVD/effective-rank analysis;
- row-space coordinates;
- covariance sketch;
- top-\(k\) margin analysis;
- expert pair boundary analysis;
- sampled expert interventions;
- quality and downstream-routing influence metrics.

### Required tests

- exact row-space projection reproduces logits;
- low-rank reconstruction error is correct;
- margin stability theorem verified on synthetic cases;
- intervention restores original state after completion.

### Acceptance

Produce per-layer router geometry and criticality artifacts.

## M6 — Deployable baseline probes

### Deliverables

- current-route reuse;
- rolling frequency/mass;
- Markov transition;
- direct ridge/linear predictor;
- direct MLP predictor;
- ADEPT-style RF;
- predictor dataset creation;
- leakage-safe split;
- online latency benchmark.

### Required tests

- online probes accept only `DeployableDecodeState`;
- train/test prompt IDs do not overlap;
- predictor serialization is safe;
- probabilities and shapes are valid;
- inference is deterministic.

### Acceptance

A common evaluation command compares every baseline against the same oracle target.

## M7 — Position-conditioned pseudo routing probes

### Deliverables

- future-position rephased probe;
- pseudo-token independent probe;
- pseudo-token causal probe;
- production KV read-only wrapper;
- default-vector collector;
- default-vector shadow rollout;
- sparse horizon anchors;
- uncertainty ensemble;
- probe cost accounting.

### Required tests

- shadow cache does not mutate production cache;
- future positions are correct;
- pre-sample mode cannot use next token;
- post-sample mode may use exactly one sampled next token;
- pseudo RNG does not alter generation RNG;
- default-vector counts/means match brute force on tiny model.

### Acceptance

Compare probes under equal measured cost and produce subset-regret curves.

## M8 — Cost-aware subset selection and simulator

### Deliverables

- adjusted top-\(B\) selector;
- variable-size knapsack;
- global budget extension;
- cache state machine;
- event-driven transfer simulator;
- hardware profiles;
- LRU/LFU/on-demand baselines;
- overlap model;
- timeline export.

### Required tests

- no capacity violation;
- byte accounting exact;
- no use before load completion;
- bandwidth sharing correct;
- perfect/no-overlap bounds;
- synthetic timeline matches expected TPOT.

### Acceptance

Produce simulator Pareto plots, clearly labeled simulated.

## M9 — Static plus dynamic residency and adaptive termination

### Deliverables

- static ranking methods;
- expert criticality integration;
- static/dynamic budget split;
- out-of-subset mass monitoring;
- early termination;
- adaptive horizon;
- sensitivity sweeps.

### Required tests

- static experts are never evicted;
- dynamic budget respects remaining capacity;
- early termination logs reason;
- re-planning occurs at the correct boundary;
- no infinite re-plan loop.

### Acceptance

Show whether static/dynamic and adaptive horizon improve the oracle or deployable Pareto frontier.

## M10 — Real single-GPU offload engine

### Deliverables

- CPU expert handles;
- pinned-memory setup;
- fixed GPU slots;
- synchronous reference swapping;
- asynchronous transfer stream;
- CUDA event synchronization;
- resident-cache integration;
- lossless and hard-commit execution;
- per-token timeline.

### Required tests

- synchronous offload matches base model in lossless mode;
- asynchronous result matches synchronous reference;
- slot replacement cannot race with compute;
- CUDA event dependencies are correct;
- measured resident bytes obey budget;
- graceful skip of CUDA-specific tests when CUDA is unavailable in CI.

### Acceptance

Run one real model whose full experts exceed the configured GPU expert budget.

## M11 — Full evaluation and artifact generation

### Deliverables

- benchmark runners;
- end-to-end experiment scripts;
- result aggregator;
- paper tables;
- paper plots;
- failure-case reports;
- reproducibility manifest;
- model support matrix;
- final documentation.

### Required tests

- aggregator rejects incompatible schemas;
- duplicated runs are detected;
- incomplete runs are excluded and reported;
- plots can be regenerated from saved tabular data;
- every table cell links to run IDs.

### Acceptance

A clean command sequence regenerates the primary results.

## Cross-cutting task: documentation

At every milestone update:

- root README;
- `STATUS.md`;
- model support matrix;
- CLI help;
- configuration examples;
- known limitations;
- changelog.

## Cross-cutting task: profiling

Profile in stages:

1. Python overhead;
2. model forward;
3. trace capture;
4. pseudo attention;
5. default-vector mixing;
6. subset selection;
7. transfer;
8. synchronization;
9. expert compute.

Do not optimize a stage before measuring its share of TPOT.

## Cross-cutting task: artifact size control

Full hidden traces can become enormous. Implement:

- trace-level switches;
- token/layer sampling;
- float16 storage;
- compression where lossless enough;
- per-layer capture;
- shard quotas;
- dry-run size estimates;
- deletion-safe manifests.

## Suggested first vertical slice

The first scientifically meaningful slice is:

1. tiny model;
2. natural route trace;
3. oracle mass subset;
4. substitution-constrained generation;
5. H2D byte simulation;
6. quality–transfer plot.

The second slice is:

1. first real model;
2. SC/SP, DC/SP, SC/DP, DC/DP;
3. current-route baseline;
4. rephased future-position probe;
5. per-layer router-logit heatmap.

The third slice is:

1. default vectors;
2. shadow rollout;
3. cost-aware subset;
4. adaptive termination;
5. simulator.

## Optimization backlog, not initial requirements

Only consider after M10 correctness:

- fused expert slot kernels;
- Triton expert execution;
- quantized CPU storage;
- decompression while transferring;
- multi-stream expert loads;
- CUDA graphs;
- pinned-memory pool;
- direct-storage/SSD tier;
- multi-GPU;
- continuous batching;
- speculative decoding integration.

<!-- END FILE: 06_IMPLEMENTATION_ROADMAP.md -->


---

<!-- BEGIN FILE: 07_CONFIG_AND_DATA_SCHEMAS.md -->

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

<!-- END FILE: 07_CONFIG_AND_DATA_SCHEMAS.md -->


---

<!-- BEGIN FILE: 08_CODEX_TASK_PROMPTS.md -->

# Incremental Codex Task Prompts

These prompts are intended to be issued one at a time to Codex after providing the full specification directory. They keep implementation aligned with the research plan.

## Prompt 0 — Read and plan

```text
Read every Markdown file in docs/spec, beginning with 00_CODEX_START_HERE.md. Do not implement model-specific optimizations yet.

Create:
1. STATUS.md with milestone M0 and a file-level checklist.
2. docs/decisions.md containing the fixed initial assumptions and any implementation decisions you must make.
3. A proposed repository tree matching the specification.
4. A dependency plan that avoids unnecessary packages.
5. A model/dataset download and cache plan that estimates disk use before downloading.

Then bootstrap the package, formatting/type-check/test configuration, CLI skeleton, and deterministic utilities. Run the test suite and update STATUS.md with exact results.
```

## Prompt 1 — Tiny MoE vertical slice

```text
Implement milestone M0.

Build a deterministic, device-configurable tiny decoder-only MoE model with optional RoPE, pre-norm attention, a linear top-k router, configurable experts, and deterministic generation. Use an explicit device setting; the default example should select CUDA when available and fall back to CPU for portability and CI. Implement route capture and a ModelSpec.

Add unit tests for natural routing, layer-scoped ExpertKey, deterministic generation, and exact trace values. Add `pseudoroute inspect-model` and a tiny model config.

External model or dataset downloads are permitted when useful and disk space is sufficient, but M0 acceptance must not depend on a large download. Estimate disk use first and record anything downloaded. Run all checks and update STATUS.md.
```

## Prompt 2 — Adapter and trace store

```text
Implement milestone M1 using the model adapter and trace-store contracts in 03_SYSTEM_ARCHITECTURE.md and 07_CONFIG_AND_DATA_SCHEMAS.md.

First make the tiny model use the same adapter interface. Then implement one real Hugging Face MoE adapter, downloading a verified model and a suitable evaluation dataset if they are not already cached and disk space is sufficient. Pin and record their identifiers and revisions. Validate structure before inference. Add route-only and router-logit trace collection, chunked storage, manifest, resume, and validation.

Add regression tests that hooks and NaturalRoutingPolicy do not change model outputs. External-model tests must be optional/skippable in CI. Update model support documentation and STATUS.md.
```

## Prompt 3 — Oracle analysis

```text
Implement milestone M2.

Add window iteration, binary-count oracle, selected-routing-mass oracle, full-router-mass oracle, expert union metrics, SCH-style curves, and cost-aware top-B selection. Implement `pseudoroute oracle-sweep`.

Use brute-force tests on tiny cases to prove additive oracle selection is correct. Prevent windows from crossing sample boundaries. Produce tabular outputs and basic plots. Run a tiny-model end-to-end sweep and update STATUS.md.
```

## Prompt 4 — Closed-loop constrained generation

```text
Implement milestone M3.

Create RoutingPolicy, NaturalRoutingPolicy, MaskedSubstitutionPolicy, MaskedTruncationPolicy, and LosslessFallbackPolicy. Add fixed-window oracle planning and closed-loop generation.

Natural and lossless fallback modes must reproduce the base model. Constrained runs must continue from their own generated trajectory. Record natural and executed routes, divergence points, out-of-subset mass, transfer-byte estimates, and quality metrics.

Add `pseudoroute closed-loop-eval`, tests, and an oracle quality-versus-transfer plot. State whether Gate A passes on the tiny model; do not generalize to real models yet.
```

## Prompt 5 — DapQ-style factorial analysis

```text
Implement milestone M4 exactly as specified.

Build an offline-only pseudo-sequence constructor with SC/SP, DC/SP, SC/DP, and DC/DP conditions. Capture pre-RoPE query, post-RoPE query, post-attention state, router input, router logits, and top-k where the adapter supports them.

Add context-swap and position-offset controls. Enforce a type boundary so these offline future tokens cannot be passed to online probes.

Implement per-layer/per-horizon metrics, paired bootstrap confidence intervals, and `pseudoroute dapq-factorial`. Validate SC/SP against teacher-forced ground truth on the tiny model.
```

## Prompt 6 — Router geometry and criticality

```text
Implement milestone M5.

Add exact/randomized SVD of router weights, router-visible coordinates, empirical covariance sketches, covariance-aware logit variance, pairwise boundary analysis, route entropy, and top-k margins.

Prove in tests that exact row-space projection reproduces router logits and that the margin stability condition is correct.

Add sampled expert interventions measuring immediate output error, next-token KL, and downstream route changes. Implement `pseudoroute analyze-router`.
```

## Prompt 7 — Baseline probes

```text
Implement milestone M6.

Add deployable CurrentRoute, RollingFrequency, RollingMass, MarkovTransition, DirectLinear/Ridge, DirectMLP, and ADEPTStyleRF probes. All online probes must accept only DeployableDecodeState.

Build grouped predictor datasets split by prompt/document, serialize models safely, record training/calibration cost, and benchmark online latency. Implement `pseudoroute train-predictor` and `pseudoroute evaluate-probe`.

Compare every probe against the same oracle window-utility target and include equal-cost reporting.
```

## Prompt 8 — Position-conditioned probes

```text
Implement milestone M7.

Start with FuturePositionRephasedProbe and its no-rephase/wrong-position ablations. Then implement independent and causal pseudo-token probes with a read-only production KV-cache interface.

Implement streaming default-vector calibration and validate same-token next-layer prediction. Extend it into DefaultVectorShadowRolloutProbe over future-position anchors. Support pre_sample and post_sample regimes without leakage. Isolate probe RNG from generation RNG.

Measure probe latency and temporary memory. Add all required tests and update STATUS.md with limitations by model adapter.
```

## Prompt 9 — Selector and simulator

```text
Implement milestone M8.

Add window utility aggregation, uncertainty-aware scores, adjusted top-B selection, variable-size knapsack, subset plans, and exact load/eviction deltas.

Build an event-driven expert-cache simulator with on-demand, LRU, LFU, lossless predictor, one-step commitment, and multi-step commitment baselines. Enforce capacity, bandwidth, latency, and load-completion invariants.

Implement `pseudoroute simulate-offload`, timeline export, and sensitivity sweeps. Clearly label outputs as simulated.
```

## Prompt 10 — Static residency and adaptive horizon

```text
Implement milestone M9.

Add static expert rankings by frequency, covariance-aware variance, quality sensitivity, downstream routing influence, miss risk, and combined score. Split the HBM budget into static and dynamic portions.

Add out-of-subset-mass monitoring and adaptive early termination. Log every termination and re-plan event. Compare fixed and adaptive horizons in closed-loop generation and simulation.
```

## Prompt 11 — Real offload engine

```text
Implement milestone M10 only after all prior correctness tests pass.

Build a single-GPU, batch-1 expert offload prototype with CPU-resident expert tensors, optional pinned memory, fixed preallocated GPU slots, a synchronous reference path, an asynchronous transfer stream, and CUDA event dependencies.

First prove lossless synchronous output equivalence, then asynchronous equivalence. Integrate subset plans and measure real H2D bytes, exposed stall, TPOT, and peak memory. Do not add Triton or custom CUDA until profiling shows the need.

Implement `pseudoroute benchmark-offload`.
```

## Prompt 12 — Full evaluation

```text
Implement milestone M11.

Add benchmark runners, complete run manifests, aggregation, paper-ready tables/plots, failure-case reports, and a reproducibility command. Reject incompatible/incomplete runs and retain negative results.

Create scripts that regenerate the primary oracle, factorial, probe, simulator, and real-runtime results from configs. Update README, model support matrix, STATUS.md, and CHANGELOG.md.
```

## Prompt for code review after any milestone

```text
Review the current repository against all specification documents. Focus on:
1. future-information leakage;
2. mismatch from native router semantics;
3. production KV mutation by pseudo probes;
4. incorrect expert byte/capacity accounting;
5. open-loop metrics being mislabeled as closed-loop;
6. simulator results being presented as real speedups;
7. train/test prompt leakage;
8. missing tests or undocumented assumptions.

Fix confirmed issues, add regression tests, and update STATUS.md. Do not broaden scope during this review.
```

## Prompt for experiment audit

```text
Audit every completed run under artifacts/runs.

For each run, verify:
- resolved config and information regime;
- model/dataset fingerprints;
- git commit and environment;
- terminal marker;
- no future leakage;
- correct routing policy;
- HBM capacity compliance;
- quality and transfer metrics;
- whether timing is simulated or measured.

Produce an audit table and quarantine invalid runs without deleting them.
```

<!-- END FILE: 08_CODEX_TASK_PROMPTS.md -->


---

<!-- BEGIN FILE: 09_RISKS_AND_DECISION_LOG.md -->

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

<!-- END FILE: 09_RISKS_AND_DECISION_LOG.md -->


---

<!-- BEGIN FILE: 10_REPRODUCIBILITY_CHECKLIST.md -->

# Reproducibility and Reporting Checklist

## 1. Before running any experiment

- [ ] Repository commit is recorded.
- [ ] Dirty working tree status is recorded.
- [ ] Resolved configuration is saved.
- [ ] Model ID, revision, and fingerprint are saved.
- [ ] Dataset ID, split, revision (when supported), and fingerprint are saved.
- [ ] Download source and cache location are recorded.
- [ ] Available disk space was checked against download, expanded-cache, trace, and artifact estimates.
- [ ] Information regime is explicit.
- [ ] Random seeds are explicit.
- [ ] Hardware profile is saved.
- [ ] Software environment is saved.
- [ ] Expected trace/output size is printed.
- [ ] Expected HBM/DRAM usage is printed.
- [ ] Output directory does not contain a completed incompatible run.
- [ ] Required calibration/predictor artifacts match the model fingerprint.

## 2. Base-model validation

- [ ] Natural adapter logits match the native model.
- [ ] Natural router IDs and weights match.
- [ ] Greedy generated tokens match.
- [ ] Trace hooks do not change outputs.
- [ ] No-op probe does not change outputs.
- [ ] Shadow probe does not mutate production KV cache.
- [ ] Probe RNG does not change generation RNG.

## 3. Trace integrity

- [ ] No window crosses sample boundaries.
- [ ] Positions are monotonic within a sample.
- [ ] Prompt and decode tokens are labeled.
- [ ] Layer count matches model manifest.
- [ ] Expert IDs are valid for each layer.
- [ ] Top-\(k\) count matches the adapter.
- [ ] Router weights follow model normalization semantics.
- [ ] Shards have checksums.
- [ ] Trace completion marker exists.
- [ ] Resume did not duplicate tokens.

## 4. Oracle experiments

- [ ] Oracle is labeled and never called deployable.
- [ ] Future information is limited to offline analysis.
- [ ] Count and mass target definitions are explicit.
- [ ] Budget rounding is documented.
- [ ] Open-loop and closed-loop results are separate.
- [ ] Constrained generation continues from its own trajectory.
- [ ] Base, lossless fallback, substitution, and truncation are distinguished.
- [ ] Expert-transfer calculation uses layer-scoped byte sizes.
- [ ] Quality and system metrics are both reported.

## 5. DapQ-style factorial experiments

- [ ] SC/SP uses the intended true future content and positions.
- [ ] DC/SP preserves positions exactly.
- [ ] SC/DP preserves content exactly.
- [ ] DC/DP changes both.
- [ ] Random content seeds are recorded.
- [ ] Position-offset policy is recorded.
- [ ] Pre-RoPE and post-RoPE metrics are not conflated.
- [ ] Router-input and router-logit metrics are included.
- [ ] Context-swap results are included.
- [ ] Paired examples are used.
- [ ] Confidence intervals use prompt/document sampling units.
- [ ] No factorial future data enters online probe evaluation.

## 6. Learned predictor experiments

- [ ] Train/validation/test split is grouped by prompt/document.
- [ ] Calibration data is separate from test.
- [ ] Target model trace-collection cost is recorded.
- [ ] Predictor parameter count is recorded.
- [ ] Optimizer and stopping rule are recorded.
- [ ] Training hardware and wall-clock are recorded.
- [ ] Predictor serialization format is safe.
- [ ] Online inference latency is measured.
- [ ] Equal-cost baseline comparison is included.
- [ ] Test data was not used to tune thresholds.

## 7. Pseudo routing probes

- [ ] Probe information regime is explicit.
- [ ] `pre_sample` does not use the next token.
- [ ] `post_sample` uses at most the already sampled next token.
- [ ] Horizon anchor set is recorded.
- [ ] Pseudo content is recorded.
- [ ] Independent/causal pseudo mask is recorded.
- [ ] Future positions are verified.
- [ ] Default-vector definition and calibration fingerprint are recorded.
- [ ] Dense/experts shadow approximation modes are recorded.
- [ ] Probe latency and temporary memory are measured.
- [ ] Production KV and RNG are unchanged.
- [ ] Utility aggregation and uncertainty formula are explicit.

## 8. Closed-loop evaluation

- [ ] Natural and executed routes are both recorded.
- [ ] Miss policy is explicit.
- [ ] Weight renormalization semantics are explicit.
- [ ] Early termination reason is logged.
- [ ] Realized window length is reported.
- [ ] First token divergence is reported.
- [ ] Next-token KL/perplexity or task metric is reported.
- [ ] Degeneration/repetition is checked.
- [ ] H2D bytes and exposed stall are reported.
- [ ] Failed generations are not silently dropped.

## 9. Simulator

- [ ] Results are labeled simulated.
- [ ] Hardware profile is saved.
- [ ] Fixed latency and bandwidth are explicit.
- [ ] Compute timing source is explicit.
- [ ] Overlap model is explicit.
- [ ] Cache capacity is never exceeded.
- [ ] Experts are not used before load completion.
- [ ] Transfer concurrency obeys the profile.
- [ ] H2D bytes are exact.
- [ ] Probe cost is included.
- [ ] No-overlap and perfect-overlap bounds are shown.
- [ ] Timeline examples are saved.

## 10. Real runtime

- [ ] Synchronous reference is correct.
- [ ] Asynchronous result matches synchronous reference.
- [ ] Warmup is separate.
- [ ] Model load time is separate.
- [ ] CUDA synchronization/timing method is documented.
- [ ] Probe time is included.
- [ ] Transfer time and exposed stall are distinct.
- [ ] Peak allocated and reserved memory are recorded.
- [ ] KV/workspace/static/dynamic expert memory are itemized.
- [ ] CPU pinned memory is recorded.
- [ ] Per-token latency is retained.
- [ ] Mean, p50, p95, and p99 TPOT are reported.
- [ ] Repetitions and variance are reported.

## 11. Static residency

- [ ] Static ranking calibration set is recorded.
- [ ] Every ranking component is available separately.
- [ ] Static budget fraction is explicit.
- [ ] Static experts are never accidentally evicted.
- [ ] Domain-shift results are reported.
- [ ] Combined-score weights are validation-selected.
- [ ] Router-weight norm is not called expert quality importance.

## 12. Result aggregation

- [ ] Only completed runs are aggregated.
- [ ] Invalid/quarantined runs are listed.
- [ ] Duplicate config/run IDs are detected.
- [ ] Schema versions are compatible.
- [ ] Macro and micro averaging are distinguished.
- [ ] Per-model/task values are retained.
- [ ] Paired confidence intervals are used where applicable.
- [ ] Table cells can be traced to run IDs.
- [ ] Plot data is saved in tabular form.
- [ ] Negative results are included.

## 13. Claims checklist

Before writing a claim:

- [ ] Is it supported by closed-loop results?
- [ ] Is the information regime deployable?
- [ ] Is the result measured or simulated?
- [ ] Is the comparison equal-budget/equal-cost?
- [ ] Does it hold across more than one seed?
- [ ] Is it restricted to evaluated models/tasks?
- [ ] Is there a simpler baseline that performs similarly?
- [ ] Is the claimed concept already present in related work?
- [ ] Are quality changes reported beside speed/memory gains?
- [ ] Are failure conditions stated?

## 14. Artifact release checklist

- [ ] Source code has a license.
- [ ] Model/dataset licenses permit the released artifacts.
- [ ] No private prompt text is included.
- [ ] No access tokens are present.
- [ ] Large artifacts have checksums and manifests.
- [ ] Exact commands are documented.
- [ ] Minimal smoke-test data is included or generated.
- [ ] External downloads are pinned by revision where possible.
- [ ] Paper figures/tables regenerate from public scripts.
- [ ] Known unsupported models and failure modes are documented.

## 15. Final reproducibility command

The project should eventually provide a command similar to:

```bash
pseudoroute reproduce \
  --suite primary \
  --config-root configs/paper \
  --output-dir artifacts/reproduction
```

This command may orchestrate existing scripts rather than reimplementing experiment logic.

<!-- END FILE: 10_REPRODUCIBILITY_CHECKLIST.md -->
