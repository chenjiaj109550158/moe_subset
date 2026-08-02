# Previous-window residual pseudo-embedding experiment v1

## Frozen question

This calibration-free experiment tests the user's window-level hypothesis at
Qwen/GSM8K `H=8,B=32`: the eight exact MoE mixture outputs realized in the
previous window may be a better approximation for the next eight pseudo
positions than forcing every MoE contribution to zero. A step in this protocol
means one eight-token output window, not one token.

At boundary zero, full-expert prefill supplies the exact native MoE mixture
outputs for the final eight prompt tokens. At every later boundary, the bank is
the exact mixture output actually executed under the policy's fixed hard subset
for its preceding output window. Position `a` in the preceding bank is mapped to
pseudo anchor `a` in the primary variant. The subset is planned once per layer
and remains fixed for the entire following eight-token window.

The complete scope, variants, gates, token caps, and exact sample IDs are frozen
before any new model output in
`configs/analysis/pseudo_embedding_qwen_gsm8k_residual_window_v1.yaml` and its
adjacent JSON sample manifest. This experiment does not train, fit, calibrate,
or read default-vector values.

## Seven staged experiments

1. Validate exact per-token/per-layer capture of the Qwen weighted routed-expert
   mixture output, including prompt order and the distinction between natural
   routes and policy-executed hard routes.
2. On the two fixed mechanism rows compare zero contribution, position-aligned
   previous-window residuals, the previous window's last residual repeated, and
   the previous-window mean repeated. Select by frozen route/cost/audit ordering.
3. With the selected residual source compare repeated sampled-token and recent-
   token contents under independent and causal shadow attention. Exact-future
   token versions are non-deployable diagnostics and cannot select a candidate.
4. Measure where the signal works or fails by pseudo anchor, layer, eight-layer
   block, router-margin tertile, context length, residual norm, and residual-to-
   current-hidden cosine. Save concrete worst boundaries.
5. Compare the frozen analytic top-32 selection formulas that use pseudo utility,
   previous-window pre-mask natural route history, or both. No coefficient sweep
   or accuracy-based selection is allowed.
6. On `test-0`, `test-439`, `test-879`, and `test-1318`, teacher-force the saved
   v17 tokens while propagating each policy's own hard-subset hidden/cache state,
   executed MoE residuals, and pre-mask natural route history. This is route
   evaluation, not measured task accuracy or vanilla open-loop replay.
7. Only after the development gate passes, run the selected method on the newly
   frozen disjoint route set. Only after its strong gate passes may the original
   frozen 16-row closed-loop accuracy partition be run for oracle, previous, and
   pseudo policies.

## Capture and shadow semantics

For token `t`, layer `l`, the stored residual is the weighted output returned by
Qwen's native routed experts before the decoder layer adds it to the post-
attention residual stream. It is not the full hidden state and not a router
probability vector. A hard forward first records natural pre-mask logits, masks
experts outside the fixed layer subset, applies native top-8 selection and
normalization, executes those experts, and stores that executed mixture output.

Each pseudo layer uses the native attention, correct future position/RoPE, and
exact native router. The chosen vector from the previous-window bank is then
added along the normal residual path before the next layer. Independent anchors
use separate disposable cache forks; causal anchors use one disposable shadow
sequence. Production KV objects, sequence lengths, tensor identities/data
pointers/version counters, hidden state, and RNG must remain unchanged. Shadow
caches are discarded at the boundary.

## Information and selection boundaries

Deployable variants can use only the known sampled next token, the current
policy's read-only production state, and that same policy's already realized
previous-window residuals and routes. Future true tokens, a vanilla trajectory,
answers, correctness, learned parameters, fitted weights, offline expert priors,
route transition tables, and default-vector values are forbidden. Exact-future
contents are labelled information-oracle diagnostics and excluded from all
candidate rankings.

For an exact definition of the analytic combinations, let `p_a` be the full
pre-top-k probability vector at pseudo anchor `a`, `P=sum(p_1..p_8)` (total mass
8), and `R` be the previous window's selected native weights scattered into 128
experts and summed (also total mass 8). The methods are fixed as follows:

- pseudo-only and history-only select `top32(P)` and `top32(R)`;
- sampled-anchor-plus-history selects `top32(8*p_1 + R)`;
- equal evidence selects `top32(P + R)`;
- sampled core locks `top8(p_1)` and fills 24 positions by `R`;
- first-four-anchor core locks the union of each of anchors 1–4's native top-8,
  ranks an oversized union by `sum(p_1..p_4)`, caps it at 32, then fills any
  remaining positions by `R`;
- linear decay selects `top32(8*sum((9-a)*p_a)/36 + R)`;
- inverse decay selects `top32(8*sum(p_a/a)/sum(1/a) + R)`.

Every ranking breaks equal scores by ascending layer-scoped expert ID. Core-fill
methods keep the core, exclude it from fill ranking, and return exactly 32
experts. The mass factors are analytic normalization, not fitted coefficients.

The two smoke rows are mechanism evidence. Development selects one frozen
analytic candidate without task accuracy. The old eight-row held-out set is now
contaminated by previous method selection and is regression-only. The new
held-out rows are SHA-256 ranked from the 1,289 rows not present anywhere in the
original sample manifest, using only row index and sample ID.

## Gates and claims

Development must improve the policy-matched previous-route baseline by at least
0.05 absolute in both mean route hit and mean selected mass, remain no worse
than static frequency, preserve 25% residency, retain at least 30% simulated
transfer reduction, and pass all information/cache/RNG/capture and measured-cost
audits. Held-out additionally requires at least 25% oracle-minus-previous gap
recovery and reference means of at least 0.6967 route hit and 0.7162 selected
mass.

Failure is `STOP/PIVOT`; no held-out accuracy is then run. If held-out route
passes, closed-loop uses the original 16 frozen IDs and original v17 prompt,
decoding, parser, and token caps. Their frozen vanilla result is 16/16, so the
predeclared pilot threshold is at least 15/16 for a policy, with paired 10,000-
resample percentile intervals. After the first eight rows, a projected total
above 24 hours requires an ETA report and user approval before continuing.

Teacher-forced route metrics and simulated transfer/stall are not task accuracy
or runtime speedup. Measured task accuracy, exact-token agreement, NLL, and
generation runtime exist only if true closed-loop generation is reached. Even a
positive result is at most `NARROW` for this Qwen/GSM8K `H=8,B=32` pilot.
