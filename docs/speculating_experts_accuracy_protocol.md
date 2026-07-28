# Speculating Experts accuracy comparison protocol

This protocol was frozen before local accuracy aggregates were inspected.

The target paper is arXiv:2603.19289 and the public implementation reference is
YALIS commit `b1970f7881129d92448e2f83b0702fea48644b92` on its
`offload_prefetch` branch. The benchmark task definitions are ported from
lm-evaluation-harness commit `d800e04dcb1ce96791d8b2926cf0cc7703d58457`;
code-task semantics are cross-checked against EvalPlus commit
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.

## Protocol revision history

The first full-run reconstruction is v8, fingerprint
`528d5ba8c8b8e66a882a19488aaacd62dc972b9d6ff95bfdbf1270c9f377fec3`.
A targeted GPT code-alignment candidate is v9, fingerprint
`8cf30430b17b2e82a23adb573872d2fb440c0bf31a22eb2d2e5d5264f6974d1d`.
A finite-code-cap candidate is v10, fingerprint
`0d4fbb451a55f8fab39dabb8ab55c4e798f24f1489f8c94a10d717aced612fdc`.
A scorer-only correction is v11, fingerprint
`4c2b60dbb4cc6cc2aa1ebd6b3f8bad936b6ca6ee423166229fc12aed4cc5576d`.
A prompt-helper scorer correction is v12, fingerprint
`beaf89e1a035fc093358433ce3a01758f03919f291e3af00542805e552dc29fc`.
A GSM8K final-answer scorer correction is v13, fingerprint
`1a11f813076b50c4ef8b47d635da578f5ab2237c68f9964ceb08e67bbbb6ade1`.
An MBPP+ fenced-code scorer correction is v14, fingerprint
`e85cf403a078ad416ded894fda93ada30c498e28bd734a752c12c8879f191fe2`.
A finite GPT HumanEval cap-extension candidate is v15, fingerprint
`dac2637dcce7c6c9cbbff44204a011969c66017b552194090206254457cb1c01`.
A checkpoint-native sampling candidate is v16, fingerprint
`46f764f29c0ff97b9300c3d5f80dc0d60c37f9bef0678d3914d3c2c8b67a3698`.
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
the full medium smoke matched the paper-reported discrete AIME outcomes while
remaining finite. Qwen generation settings did not change. Formal v8 shards reuse only
generation-compatible rows, retaining their provenance and applying the v7
scorer where required.

V9 was declared after completed v8 GPT HumanEval and in-progress MBPP+
diagnostics showed that GPT code quality missed the paper gate. The official
GPT-OSS model card requires Harmony formatting and demonstrates a user message
followed by model-generated reasoning and final channels. V8 continued an
assistant `final` message containing a code fence, structurally preventing that
reasoning channel. V9 changes only GPT HumanEval and MBPP+: it sends the
identical user prompt without assistant prefill and scores the standalone final
code fence. Qwen and every QA/math generation remain byte-compatible with v8.
Before any v9 generation, the adoption rule is fixed: v9 replaces v8 for final
GPT code claims only if both complete vanilla code tasks pass the unchanged
two-standard-error alignment gate and code-extraction audit. Otherwise both
protocol failures are retained; results are not selected task by task.

V9 was stopped after its first completed rows exposed a structural cap failure:
one GPT HumanEval response reached exactly 1024 tokens after entering the final
channel but before closing its code fence. This was audited before a v9
aggregate existed. V10 changes only GPT HumanEval and MBPP+ maximum generation
from 1024 to the same finite 4096-token allowance already used for its four QA
and math tasks. The v9 rows and interruption envelopes remain diagnostics. The
same adoption rule applies to complete v10 HumanEval and MBPP+; no individual
task may be selected from another protocol.

V11 was declared after a completed v10 HumanEval shard audit found three
demonstrable harness false negatives: standalone programs were appended to the
HumanEval stub, generated `from __future__` statements followed the safety
preamble, and candidate-only `if __name__ == "__main__"` blocks executed during
scoring. V11 changes no model, prompt, decoding, dataset, or generated token. It
compiles a complete fenced replacement as a separate non-main source unit and
deterministically rescores both code tasks; all other v10 scores are reusable.
The raw v10 rows remain protocol-history evidence.

V12 was declared after the complete v11 HumanEval audit found one remaining
demonstrable harness false negative: the standalone target function called a
helper defined before the target stub in the HumanEval prompt, but v11 executed
only the replacement block. V12 executes that prompt prelude and the candidate
as separate non-main source units before the official tests. It changes no
generation input or token and deterministically rescores both code tasks; the
other scores remain reusable.

V13 was declared during the in-progress GSM8K vanilla audit after raw GPT-OSS
final responses exposed false negatives in v7's first-number-after-`Final
Answer` rule. Verbose correct responses could repeat problem values before the
conclusion, express a thousands separator as `\boxed{8{,}000}`, or give an
answer in one unit followed by a conversion. V13 prioritizes normalized boxed
answers, explicit answer lead-ins, and emphasized conclusions, with regression
cases taken from the audited raw responses. It changes no model, prompt,
decoding, dataset, or generated token. GSM8K is deterministically rescored;
all generation-compatible non-GSM scores retain their prior scorer result.

V14 was declared after the complete GPT-OSS MBPP+ failure audit found that the
standalone-code extractor selected the first Markdown fence even when it was an
unlabelled derivation or pseudocode block and a later `python` fence defined the
canonical MBPP entry point. V14 selects a fenced definition of a function from
the pinned canonical solution, then a Python-labelled fence, before falling
back to the first fence. Seven previously failed raw generations pass the same
pinned augmented tests under this correction. It changes no generation input or
token and deterministically rescores MBPP+ only.

V15 was declared after the complete v12/v14 HumanEval scorer audit retained 13
GPT-OSS failures, five of which ended exactly at the 4096-token cap without a
complete target-function fence. The official checkpoint supports substantially
longer outputs, while the paper does not disclose its code-task cap. V15 changes
only GPT-OSS HumanEval `max_new_tokens` from 4096 to a finite 16384 and reruns all
164 rows; no capped-row-only selection is permitted. The adoption rule remains
the predeclared vanilla alignment gate. Every other model/task generation is
byte-compatible with v14.

V16 was declared after complete Qwen AIME24 v10 generation scored 18/30 versus
the paper's 24/30, and a raw-output audit showed that the mismatch was not an
answer-extraction error. The pinned lm-eval task requests greedy decoding, but
the Table 1 benchmark driver and decoding settings are not public. In contrast,
both pinned checkpoint `generation_config.json` files publish
`do_sample=true`; Qwen additionally publishes `temperature=0.7`, `top_p=0.8`,
and `top_k=20`. The pinned public YALIS engine also samples by default with
temperature 1.0 and top-p 1.0. V16 changes only `decode.do_sample` from false
to true and otherwise inherits each pinned checkpoint's generation config.
It is evaluated as a complete protocol candidate rather than selecting
individual rows or tasks. The fixed adoption rule is unchanged: both models
must pass all six vanilla alignment gates before oracle materialization or any
paper-comparable Router-PF claim.
The formal v16 run assigns Qwen to GPU1 and GPT-OSS to the identical-model GPU0
so that the two complete model suites can execute concurrently after all v15
workers exit; device assignment does not mix rows within a model/task.

## Disclosure boundary

The paper and public branch do not publish the Table 1 benchmark driver,
prompt/chat rendering, random seeds, decoding settings, default-vector
calibration corpus, generated samples, or default-vector artifacts. The public
branch contains a Qwen YALIS implementation but no GPT-OSS prefetch
implementation. Therefore local results are a documented reconstruction, not
an exact execution of unpublished author artifacts.

## Fixed comparison

- Checkpoints, dataset revisions, sample counts, decoding limits, and reported
  paper values are fixed in the versioned v15 and v16 suite configs.
- Decoding is batch one. V15 is greedy (`do_sample=false`); v16 uses seeded,
  checkpoint-native sampling (`do_sample=true`) and inherits the pinned
  checkpoint's temperature/top-p/top-k. Both remain exactly replayable because
  the runner resets the frozen seed before every sample. Qwen uses its
  non-thinking Instruct chat template.
  GPT-OSS uses its pinned Harmony template with `reasoning_effort=medium`.
  All six GPT tasks use a 4096-token cap so that the
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
