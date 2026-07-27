# Speculating Experts accuracy comparison protocol

This protocol was frozen before local accuracy aggregates were inspected.

The target paper is arXiv:2603.19289 and the public implementation reference is
YALIS commit `b1970f7881129d92448e2f83b0702fea48644b92` on its
`offload_prefetch` branch. The benchmark task definitions are ported from
lm-evaluation-harness commit `d800e04dcb1ce96791d8b2926cf0cc7703d58457`;
code-task semantics are cross-checked against EvalPlus commit
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.

## Protocol revision history

The formal suite is v8, fingerprint
`528d5ba8c8b8e66a882a19488aaacd62dc972b9d6ff95bfdbf1270c9f377fec3`.
An interrupted v1 HumanEval smoke revealed that replacing `socket.socket` before
a candidate's benign `import doctest` caused Python's `ssl` module to fail while
loading. The generated functions were correct, but the harness reported false
negatives. V2 preloads `doctest` and `ssl` before applying the same network guard,
adds a regression test, and starts all formal task results from row zero. V1 rows
are protocol-development artifacts and are excluded from final checksums and
claims. A second interrupted pre-AIME audit found that v2's unboxed AIME fallback
selected the last numeric token, while the pinned harness compares the complete
unboxed response. V3 matches that behavior and adds a regression test. No AIME
or aggregate accuracy was generated before freezing v3. A final interrupted
HumanEval audit found that GPT-OSS can close the prompt-provided fence immediately
and emit a complete replacement function in a later `python` block. V4 accepts a
complete fenced rewrite containing the required entry point, while retaining the
continuation rule otherwise. The triggering generated program passes its pinned
tests under v4; v3 rows remain excluded from claims. An interrupted MBPP+ audit
then found that a correct non-empty continuation before the prompt-provided
closing fence was displaced by a later explanatory fence. V5 always prefers a
non-empty prefix continuation and looks for a complete replacement only when the
continuation is empty. A regression covers both shapes; all pre-v5 rows remain
protocol-development artifacts. V6 raises GPT-OSS GSM8K and StrategyQA output
limits to 4096 after high-effort smokes repeatedly exhausted the shorter limit.
V7 fixes GSM8K final-number selection so that a number explicitly following
`Final Answer` takes precedence over later explanatory numbers, and compares
numeric answers by decimal value. It also version-controls which old scores may
be reused and which generations must be deterministically rescored. The
triggering old rows and an importer-race duplicate remain in separate diagnostic
roots and are excluded from claims. V8 freezes GPT-OSS at
`reasoning_effort=medium`, with 4096-token limits for GSM8K, StrategyQA, AIME24,
and AIME25. High-effort AIME repeatedly failed to terminate even at 32768 tokens;
the full medium smoke matched the paper discrete AIME outcomes while remaining
finite. Qwen generation settings did not change. Formal v8 shards reuse only
generation-compatible rows, retaining their provenance and applying the v7
scorer where required.

## Disclosure boundary

The paper and public branch do not publish the Table 1 benchmark driver,
prompt/chat rendering, random seeds, decoding settings, default-vector
calibration corpus, generated samples, or default-vector artifacts. The public
branch contains a Qwen YALIS implementation but no GPT-OSS prefetch
implementation. Therefore local results are a documented reconstruction, not
an exact execution of unpublished author artifacts.

## Fixed comparison

- Checkpoints, dataset revisions, sample counts, decoding limits, and reported
  paper values are fixed in
  `configs/benchmark/speculating_experts_accuracy_v8.yaml`.
- Decoding is batch one and greedy (`do_sample=false`) for deterministic
  paired comparisons. Qwen uses its non-thinking Instruct chat template.
  GPT-OSS uses its pinned Harmony template with `reasoning_effort=medium`.
  GSM8K, StrategyQA, AIME24, and AIME25 use a 4096-token cap so that the
  Harmony analysis channel has a meaningful but finite allowance; only the
  final channel is scored. Raw task stop strings remain enabled for Qwen, but
  are disabled for GPT-OSS because Harmony analysis can quote
  `Q:`/`Question:` before its final channel; GPT-OSS terminates by EOS or the
  recorded token cap.
- `vanilla` executes native selected expert IDs and weights.
- `router_pf` reconstructs YALIS layer-ahead default-vector routing: layer zero
  is native; each later layer executes IDs and routing weights predicted at the
  previous layer from the previous layer's executed route and calibrated mean
  expert outputs.
- `oracle_pf` is the same native-top-k residency budget with perfect
  layer-ahead knowledge of the vanilla trajectory's native IDs and weights.
  Its quality output must equal vanilla; it is an optimistic selection-error
  upper bound, not an implementable predictor. After exact real-forward parity
  is validated on both architectures and cross-task smokes, full oracle rows are
  identity-materialized from vanilla outputs; they are marked derived and carry
  no fabricated runtime measurement.
- Mean unweighted selected-expert outputs are calibrated on the listed,
  disjoint WikiText-2 validation rows. Experts with zero natural selections retain the explicitly configured zero vector; the manifest records their counts. This matches the public code's vector
  definition, but the corpus differs from the unpublished author corpus.
- The Qwen BF16 and GPT-OSS native-MXFP4 tiers remain separate.

## Vanilla alignment gate

For each model and task, local vanilla is aligned only when its absolute
accuracy difference from the paper is no greater than the larger of:

1. two paper-reported standard errors; and
2. one question divided by that task's sample count.

All six tasks must pass for a model-level aligned claim. A failed alignment is
retained as a protocol mismatch; prompts or decoding are not tuned after seeing
results. Router-PF versus-paper deltas are described as paper-comparable only
after this vanilla gate passes.

## Accuracy and provenance

The suite records every rendered prompt, generated token IDs/text, parsed
answer or code, per-sample correctness, policy route agreement, completion or
failure envelope, resolved config, checkpoint/dataset revisions, software and
hardware, and resumable progress. HumanEval uses the standard tests; MBPP+
uses the pinned augmented tests. GSM8K uses flexible final-number extraction,
AIME prefers the final boxed/final numeric answer, and StrategyQA accepts only
an explicit yes/no or true/false answer. Aggregate uncertainty uses exact
binomial sample counts and reports standard error plus a 95% Wilson interval.
Oracle-versus-Router-PF deltas are paired by sample and include discordant counts,
a paired standard error, and a 95% interval. Router-PF summaries also aggregate
exact-top-k agreement and selected-expert hit rate by routed token rather than by
prompt. A root completion envelope is written only after every configured sample
exists for all three policies and records byte sizes and SHA-256 checksums.
