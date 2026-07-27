# Speculating Experts accuracy comparison protocol

This protocol was frozen before local accuracy aggregates were inspected.

The target paper is arXiv:2603.19289 and the public implementation reference is
YALIS commit `b1970f7881129d92448e2f83b0702fea48644b92` on its
`offload_prefetch` branch. The benchmark task definitions are ported from
lm-evaluation-harness commit `d800e04dcb1ce96791d8b2926cf0cc7703d58457`;
code-task semantics are cross-checked against EvalPlus commit
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.

## Protocol revision history

The formal suite is v2, fingerprint
`b06a195cd93b75ce30d3f5af8261056ecf54937b4e47e217230678bd3a0e970e`.
An interrupted v1 HumanEval smoke revealed that replacing `socket.socket` before
a candidate's benign `import doctest` caused Python's `ssl` module to fail while
loading. The generated functions were correct, but the harness reported false
negatives. V2 preloads `doctest` and `ssl` before applying the same network guard,
adds a regression test, and starts all formal task results from row zero. V1 rows
are protocol-development artifacts and are excluded from final checksums and
claims. No aggregate accuracy was inspected before freezing v2.

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
  `configs/benchmark/speculating_experts_accuracy_v2.yaml`.
- Decoding is batch one and greedy (`do_sample=false`) for deterministic
  paired comparisons. Qwen uses its non-thinking Instruct chat template.
  GPT-OSS uses its pinned Harmony template with `reasoning_effort=high`. Its
  StrategyQA cap is 512 rather than the raw harness's 128 so that Harmony's
  analysis channel cannot consume the entire allowance; only the final channel
  is scored. Raw task stop strings remain enabled for Qwen, but are disabled
  for GPT-OSS because Harmony analysis can quote `Q:`/`Question:` before its
  final channel; GPT-OSS terminates by EOS or the recorded token cap.
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
