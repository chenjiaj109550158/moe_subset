# Reproducibility

The primary deterministic suite is regenerated with:

```bash
pseudoroute reproduce --suite primary --config-root configs/paper \
  --output-dir artifacts/reproduction/primary
```

Use `--dry-run` to list the ordered commands without creating artifacts. The real-runtime task requires CUDA; all other primary tasks use the portable tiny configuration. No external download is required.

Every reproduced run contains `resolved_config.json`, `metrics.json`, captured stdout/stderr, terminal `DONE` or `FAILED.json`, and `run_manifest.json`. Manifest schema 1 records the command, result kind, information regime, config digest, stable run ID, completion state, retained negative findings, and SHA-256/size for every artifact.

`pseudoroute aggregate-results --input-root RUNS --output-dir PAPER` validates all checksums, rejects incompatible schemas and duplicate run IDs, excludes and reports incomplete runs, and never discards negative findings. Paper tables link each value to its source run. Plots are generated only from saved tabular data.

Individual runners are available as `scripts/regenerate_oracle.sh`, `regenerate_factorial.sh`, `regenerate_probe.sh`, `regenerate_simulator.sh`, and `regenerate_runtime.sh`.

## Trained oracle gate v2

Install the trained dependencies and run only from the pinned local cache:

```bash
python -m pip install -e '.[dev,hf]'
CUDA_VISIBLE_DEVICES=0,1 \
HF_HOME=.cache/pseudoroute/huggingface \
HF_DATASETS_CACHE=.cache/pseudoroute/datasets \
pseudoroute trained-suite \
  --config configs/trained/oracle_gate_v2.yaml \
  --output-dir artifacts/trained_gate/suite_v2_final
```

`--dry-run` validates and lists the models/datasets without loading checkpoints.
The runner is config-aware and resumable: a `DONE` model is reused only when its
resolved suite/model config matches; completed trace shards are checksum-validated
and source/quality metadata are retained. Floating-point model failure stops the
suite. The separate MXFP4 tier retains `FAILED.json` and allows the primary gate
to finish. Every model and the suite root has a completion/failure envelope with
SHA-256 and size for every artifact.

Final environment: Linux 6.8.0-100-generic x86-64; Python 3.14.6; PyTorch
2.13.0+cu130; CUDA runtime 13.0; driver 580.126.09; Transformers 5.14.1;
Tokenizers 0.22.2; Safetensors 0.8.0; Datasets 4.8.5; Accelerate 1.14.0;
Kernels 0.15.2/Kernels-data 0.16.0; NumPy 2.3.5; two NVIDIA
A100-SXM4-80GB GPUs; 235 GiB host RAM and no swap. The `hf` extra declares
Accelerate, Kernels and psutil because validated two-GPU placement, native MXFP4
loading, and per-model RSS sampling require them.

For each floating-point model, `traces/{wikitext,gsm8k}/manifest.json` is schema
2 and includes the pinned model/dataset revision, model/tokenizer fingerprint,
seed, prompt/source provenance, non-contiguous layer IDs, routed/shared metadata,
native pre-top-k scores, and shard checksums. `checkpoint_files.json` records Hub
content-addressed weight checksums. `inspection.json` records the direct native
router call before trace validation.

Saved result tables are:

- `oracle_windows.csv`: source/boundary/layer/method-level metrics and concrete routes/subsets;
- `oracle_aggregates.csv`: global mean/median/P05/P95/worst and 1,000-bootstrap CIs;
- `oracle_stratified_aggregates.csv`: the same metrics by context-position and router-margin buckets;
- `oracle_worst_cases.json`: 200 concrete worst windows;
- `closed_loop.csv` and `closed_loop_worst_cases.json`: actual token/NLL/answer outcomes and simulated transfer/stall fields;
- `operating_points.csv`, `decision.json`, and `oracle_gate.svg`: gate inputs, terminal decisions, and a plot generated only from the saved table.

The final root envelope contains 114 artifacts and was independently rehashed
with zero mismatches. The four model envelopes contain 26 artifacts each; the
gpt-oss failure envelope contains five. The direct DeepSeek 5.14 import failure
and DynamicCache compatibility failure remain under
`artifacts/trained_gate/adapter_validation/deepseek/`. The shim edits only
in-memory module/class attributes inside restoring contexts and never edits the
Hub module cache. `gpt-oss-20b/metal/model.bin` remains a cache symlink to the
13,750,886,400-byte retained blob.

Timing fields named `simulated_*` use 25 GiB/s plus 10 µs/load and are not
measured end-to-end runtime. Adapter elapsed time, CPU RSS, CUDA allocation,
model logits/routes, NLL/perplexity, token agreement, fallback, and GSM answer
extraction come from actual checkpoint execution. Lossless fallback equality is
a correctness result and must never be described as a quality improvement.

## Benchmark subset oracle: GPT-OSS/GSM8K hard scope

The immutable scientific config remains
`configs/benchmark/benchmark_subset_oracle_v1.yaml`. The separately versioned
execution amendment is
`configs/benchmark/benchmark_subset_oracle_v1_gpt_gsm8k_hard_v2.yaml`, with
scope fingerprint
`a5a908ad0124d2041b589a681e7102d3ea5d3f7df90752075799bb9875871d7c`.
It schedules only actual GPT-OSS GSM8K hard commitment at the preselected
`(H=1,B=4)` point and requires all 1,319 frozen v17 rows.

Run or resume offline from the existing model and dataset cache:

```bash
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/complete_subset_oracle_v1.py
```

Each successful policy/sample row is written atomically and is reused only when
its complete state and base-config fingerprint match. `.FAILED.json` markers are
retained but do not suppress regeneration of a missing success row. Per-GPU
queues preserve the four frozen logical shards while allowing each GPU to start
its next shard independently; at most one worker occupies a physical GPU.
Unscheduled HumanEval/MBPP+/AIME/StrategyQA and previous-route artifacts remain
under the artifact root as provenance, but the scoped summary, paired CI,
decision, and audit include only GSM8K. A passing scoped gate is reported as
`NARROW`, not as an all-task `GO`.
