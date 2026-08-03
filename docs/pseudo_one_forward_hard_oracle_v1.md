# `pseudo_one_forward_hard_oracle_v1` protocol

This is a post-hoc, separately versioned execution amendment requested after the
completion of `pseudo_one_forward_accuracy_pilot_v1`. It measures the true
closed-loop hard routing-oracle ceiling on exactly the same eight frozen
Qwen/GSM8K rows. It neither changes nor retroactively selects the parent pilot's
samples, pseudo variants, gates, or decision.

## Frozen scope

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bfloat16.
- Dataset: the eight `closed_loop_wave_1` rows in the already committed
  `configs/benchmark/pseudo_one_forward_accuracy_pilot_v1_samples.json`, in its
  original order. No replacement or result-conditioned selection is allowed.
- Decode: the exact saved v17 prompts, greedy decoding, v17 stop strings and
  GSM8K parser, seed `20260727`, and the original 512-token cap.
- Operating point: `H=8`, `B=32`, 128 routed experts per layer, native top-8,
  and 25% resident fraction.
- Output root: `artifacts/pseudo_one_forward_hard_oracle_v1`.

The parent resolved sample manifest is reused verbatim rather than copied. Its
path and SHA-256, the parent config and artifact manifest, and the base subset
oracle config and canonical fingerprint are pinned in the amendment config.

## Policy semantics

At every eight-token boundary, the runner performs a natural full-expert greedy
lookahead from the hard policy's own current context. For each routed layer, it
sums the selected natural router weights across the future window and selects
the deterministic top 32 experts. Ties use ascending layer-scoped expert ID.

Actual generation then masks all routed-expert logits outside that layer's
subset before native top-k selection and normalization. The next boundary is
planned from the resulting hard policy context, not from a frozen vanilla
trajectory. Lookahead uses copy-on-write cache state or exact cache/RNG rewind.
Benchmark answers, correctness, and v17 future tokens are forbidden inputs.

This is a nondeployable routing-information oracle ceiling. It is not
`future_exact_content_oracle`, is not identity materialization, and cannot be
described as a one-extra-forward deployable pseudo-embedding policy.

## Frozen reporting rule

The primary result is measured closed-loop GSM8K successes out of eight and
accuracy. It is paired descriptively with the frozen same-row vanilla reference
(8/8), with gains/losses/ties and a 10,000-resample paired bootstrap interval
using seed `20260803`. Route preservation, exact-token agreement, NLL,
perplexity, runtime, and peak memory are secondary diagnostics.

Eight rows can support only a scoped `PILOT_NARROW` statement. Accuracy cannot
select a policy or sample, cannot revise the parent pilot's decision, and cannot
support a full-dataset GO or production-speedup claim. Transfer reduction is a
simulation; task accuracy and runtime are measured.
