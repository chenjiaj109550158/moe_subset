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
