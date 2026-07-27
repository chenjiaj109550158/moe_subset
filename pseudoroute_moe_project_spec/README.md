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
