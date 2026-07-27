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
