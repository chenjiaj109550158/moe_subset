# Calibration-free pseudo-embedding synthesis v1

## Scope and claim boundary

This document records the analysis plan, completed experiments, and strongest
remaining hypothesis after the terminal `pseudo_embedding_qwen_gsm8k_v1`
`STOP/PIVOT`. All results are limited to pinned Qwen3-30B-A3B-Instruct-2507,
GSM8K v17 trajectories, BF16, `H=8`, `B=32`, 128 routed experts, native top-8,
and 25% residency. They do not relabel v1, report task accuracy, establish hard
closed-loop quality, or claim runtime speedup.

A deployable construction is calibration-free here only when it uses the
current request's sampled token, read-only production cache, and online prompt
or generated-token state. Learned or fitted values, offline expert priors,
default-vector values, route-transition tables, future true tokens, answer
labels, and correctness are forbidden. Diagnostic future-token interventions
are kept separate and cannot select a deployable method.

## Executed analysis ledger

1. The checksum-frozen four-row tensor analysis measured horizon decay,
   temporal persistence, layer and router-margin sensitivity, component
   interventions, five analytic candidate utilities, leave-one-out stability,
   and cost. Its config fingerprint is
   `c09d70901bacafabdbacda398390574f7c24f67f634909d39b2084be3e04f898`.
2. The two-row native-Qwen content smoke compared sampled-token repetition,
   recent realized token contents, independent and causal shadow anchors, and
   true-future-token diagnostic oracles. It also measured cache/RNG parity,
   probe latency, temporary memory, router calls, and attention queries.
3. The two-row native prompt-route smoke replaced the boundary-zero static
   prior with same-request prompt routes and tested four analytic history/pseudo
   compositions.
4. The four-row development run captured prompt routes and selected exactly one
   candidate using route and simulated-transfer metrics only.
5. The separately frozen eight-row held-out route protocol evaluated only that
   candidate and the declared references. It completed 8/8 atomic samples and
   stopped before accuracy because its strong-candidate gate failed.
6. After the failed gate, a clearly post-hoc tensor diagnostic compared analytic
   horizon-damping schedules. These observations may generate a future
   hypothesis but are not held-out validation and cannot rescue the decision.

Exact protocols and reports are in the neighboring
`pseudo_embedding_calibration_free_*_v1.md` documents and their artifact roots.

## What the native router is sensitive to

| Intervention | Native observation | Interpretation |
|---|---:|---|
| sampled token -> current token | normalized centered-logit RMS 0.749; pseudo top-8 overlap 0.383 | token/hidden direction strongly controls the router |
| default residual -> zero residual | RMS 0.558; top-8 overlap 0.516 | residual approximation strongly moves routes, but the saved defaults move them in an unhelpful direction |
| independent -> causal zero rollout | RMS 0.287 in the four-row tensor analysis; native content smoke route/mass delta -0.037/-0.043 | chaining a shadow with missing MoE residual compounds state error |
| sampled token -> one-shot expected top-8 embedding | RMS 0.032; top-8 overlap 0.967 | averaging the same boundary distribution supplies almost no new route diversity |
| sampled repeat -> exact future contents | native route/mass delta +0.047/+0.059 | future token content contains real headroom, but this is a non-deployable oracle |
| sampled repeat -> recent realized contents | route/mass delta -0.005/-0.001 | a recent token-ID bank is not a substitute for future contextual state |

The repeated-token pseudo trajectory also changes much too slowly. Natural
router top-8 overlap is 0.392 at lag 2 and about 0.258 at lag 8. Moving the same
pseudo content through future RoPE positions leaves overlap at 0.832 and 0.702
respectively. Correct positions are necessary but cannot create the missing
content and residual-state evolution.

Router-margin quartiles do not explain the error: pseudo/natural centered-logit
alignment is nearly flat and sometimes worse in the highest-margin quartile.
The primary failure is a wrong hidden direction, not merely unstable 8th/9th
ties. Layer behavior is heterogeneous; held-out gains over previous route fall
from roughly 0.062 route hit in layers 0-7 to 0.016 in layers 40-47.

## Held-out result

The frozen candidate utility for each layer was

`p(anchor 1) + 0.5 * sum(p(anchors 2..8)) + 3.5 * route_history`,

followed by deterministic top-32 selection. Boundary zero used the same
request's final eight prompt routes; later boundaries used the policy's previous
eight realized generated-token routes. It used independent native attention,
correct future RoPE, the exact native router, and zero expert contribution.

On eight disjoint rows and 393,216 routed slots it achieved route hit 0.658353,
selected mass 0.680584, and simulated transfer reduction 0.439423. Previous
route achieved 0.617671/0.637396, so gains were +0.040682/+0.043188. The paired
sample-bootstrap 95% intervals were [0.036001, 0.046422] and
[0.038634, 0.049149]: consistently positive, but below both frozen +0.05 gates.
Only 11.7%/12.6% of the oracle-minus-previous gaps were recovered, versus the
required 25%; absolute 0.6967/0.7162 references also failed. Transfer and every
cache/RNG/information audit passed. The decision is therefore `STOP/PIVOT`, and
task accuracy and actual hard closed-loop rows remain zero.

The gain is not all pseudo-embedding gain. Replacing the boundary-zero static
fallback with same-request prompt route history accounts for about
+0.02164/+0.02367 over previous route. The frozen pseudo/history mixture adds
about +0.01904/+0.01952 beyond that prompt-history baseline. Its advantage is
front-loaded: versus previous route, route-hit gain is about +0.139 at anchor 1,
+0.071 at anchor 2, +0.034 at anchor 3, and generally near +0.006 to +0.024 for
anchors 4-8.

## Strongest remaining composition hypothesis

The most defensible current construction keeps:

- same-request prompt route history at the first boundary and current-policy
  realized route history thereafter;
- the known sampled token, native attention/RoPE/router, independent shadow
  anchors, zero offline priors, and deterministic ascending-ID tie-breaks;
- strong pseudo evidence only near the boundary, with unknown later anchors
  represented mainly by online route history.

A post-hoc diagnostic utility `p(anchor 1) + p(anchor 2) + p(anchor 3) +
5 * route_history` was best among the recorded analytic horizon schedules at
0.664205 route hit and 0.687282 selected mass. It still missed every strong
held-out reference: gains over previous route were only about
+0.04653/+0.04989 and oracle-gap recovery remained far below 25%. The anchor-3
cutoff was observed after held-out results, so this formula is a hypothesis, not
a calibration-free validation result. It must be frozen and tested on a new
disjoint scope before any promotion.

## Next calibration-free experiments, in order

1. **Online MoE-residual bank.** Capture the current policy's actual per-layer
   MoE residuals for the last one and last eight realized tokens. Compare
   last-residual reuse, unweighted online mean, and route-weight-matched reuse in
   a copy-on-write shadow. This directly targets the causal-zero failure without
   an offline default-vector artifact.
2. **Autoregressive content rollout.** Anchor 1 uses the known sampled token;
   later anchors use tokens predicted by the shadow LM head, not the same
   boundary distribution repeated eight times. Start with deterministic greedy,
   then a fixed architecture-derived beam or particle count. Production RNG
   remains untouched.
3. **Predeclared horizon damping.** Compare only formulas derived from `H`,
   native top-k, and `B`: first `B/top_k=4` anchors plus history, linear remaining
   horizon weights, inverse-anchor weights, and history-only. Do not select the
   post-hoc anchor-3 cutoff without a new disjoint evaluation.
4. **Online state rather than token-ID banks.** Reuse recent post-attention
   router-input states or their current-request mean, with no fitted projection.
   Token-ID reuse already failed and should not be repeated.
5. **Agreement-aware rank composition.** Use deterministic rank union/core rules
   based on current pseudo/history agreement, avoiding fitted thresholds or
   weights. Report churn and transfer together with route coverage.
6. **Resident-expert shadow iteration.** Evaluate exact residuals only for
   already resident experts, reroute once, and stop at a fixed single iteration.
   This is calibration-free but must report circularity, latency, and temporary
   memory explicitly.
7. **Fixed multi-particle rollout.** If greedy autoregression shows route
   headroom, aggregate a small deterministic top-token particle set. Cost caps
   must be frozen because the current independent probe already uses 384 native
   attention and router calls per boundary.

Do not repeat one-shot expected embeddings, recent token-ID content banks,
zero-residual causal rollout, or default-vector propagation as primary
candidates: they already supplied negative mechanism evidence. Any new rows or
sample scope require explicit authorization and a committed protocol before
model execution.

## Cost and provenance boundary

The held-out run made 128 pseudo calls. Each call issued 384 attention queries
and 384 router calls; measured mean latency was 0.796 seconds, total probe time
101.92 seconds, mean temporary CUDA allocation 110.3 MB, and maximum 178.6 MB.
The synthetic zero interface occupied 25.2 MB persistently but supplied no
expert values. Eight 128-token replay samples averaged 44.33 seconds; candidate
CPU postprocessing totaled 24.79 seconds. These are measured analysis costs,
not hard-generation latency or speedup. Transfer reduction is simulated.

The authoritative held-out artifact manifest SHA-256 is
`3f1b509d7d012e8266ee6a28e1863efb44f00a946b29646a484c859a5ecd08db`.
Execution used clean revision `d835b1243178fdf181b62ff6dea5c8c4ed2bdc94`,
one worker on physical GPU 1, offline local caches, and no downloads.
