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
