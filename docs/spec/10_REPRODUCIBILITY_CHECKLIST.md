# Reproducibility and Reporting Checklist

## 1. Before running any experiment

- [ ] Repository commit is recorded.
- [ ] Dirty working tree status is recorded.
- [ ] Resolved configuration is saved.
- [ ] Model ID, revision, and fingerprint are saved.
- [ ] Dataset ID, split, revision (when supported), and fingerprint are saved.
- [ ] Download source and cache location are recorded.
- [ ] Available disk space was checked against download, expanded-cache, trace, and artifact estimates.
- [ ] Information regime is explicit.
- [ ] Random seeds are explicit.
- [ ] Hardware profile is saved.
- [ ] Software environment is saved.
- [ ] Expected trace/output size is printed.
- [ ] Expected HBM/DRAM usage is printed.
- [ ] Output directory does not contain a completed incompatible run.
- [ ] Required calibration/predictor artifacts match the model fingerprint.

## 2. Base-model validation

- [ ] Natural adapter logits match the native model.
- [ ] Natural router IDs and weights match.
- [ ] Greedy generated tokens match.
- [ ] Trace hooks do not change outputs.
- [ ] No-op probe does not change outputs.
- [ ] Shadow probe does not mutate production KV cache.
- [ ] Probe RNG does not change generation RNG.

## 3. Trace integrity

- [ ] No window crosses sample boundaries.
- [ ] Positions are monotonic within a sample.
- [ ] Prompt and decode tokens are labeled.
- [ ] Layer count matches model manifest.
- [ ] Expert IDs are valid for each layer.
- [ ] Top-\(k\) count matches the adapter.
- [ ] Router weights follow model normalization semantics.
- [ ] Shards have checksums.
- [ ] Trace completion marker exists.
- [ ] Resume did not duplicate tokens.

## 4. Oracle experiments

- [ ] Oracle is labeled and never called deployable.
- [ ] Future information is limited to offline analysis.
- [ ] Count and mass target definitions are explicit.
- [ ] Budget rounding is documented.
- [ ] Open-loop and closed-loop results are separate.
- [ ] Constrained generation continues from its own trajectory.
- [ ] Base, lossless fallback, substitution, and truncation are distinguished.
- [ ] Expert-transfer calculation uses layer-scoped byte sizes.
- [ ] Quality and system metrics are both reported.

## 5. DapQ-style factorial experiments

- [ ] SC/SP uses the intended true future content and positions.
- [ ] DC/SP preserves positions exactly.
- [ ] SC/DP preserves content exactly.
- [ ] DC/DP changes both.
- [ ] Random content seeds are recorded.
- [ ] Position-offset policy is recorded.
- [ ] Pre-RoPE and post-RoPE metrics are not conflated.
- [ ] Router-input and router-logit metrics are included.
- [ ] Context-swap results are included.
- [ ] Paired examples are used.
- [ ] Confidence intervals use prompt/document sampling units.
- [ ] No factorial future data enters online probe evaluation.

## 6. Learned predictor experiments

- [ ] Train/validation/test split is grouped by prompt/document.
- [ ] Calibration data is separate from test.
- [ ] Target model trace-collection cost is recorded.
- [ ] Predictor parameter count is recorded.
- [ ] Optimizer and stopping rule are recorded.
- [ ] Training hardware and wall-clock are recorded.
- [ ] Predictor serialization format is safe.
- [ ] Online inference latency is measured.
- [ ] Equal-cost baseline comparison is included.
- [ ] Test data was not used to tune thresholds.

## 7. Pseudo routing probes

- [ ] Probe information regime is explicit.
- [ ] `pre_sample` does not use the next token.
- [ ] `post_sample` uses at most the already sampled next token.
- [ ] Horizon anchor set is recorded.
- [ ] Pseudo content is recorded.
- [ ] Independent/causal pseudo mask is recorded.
- [ ] Future positions are verified.
- [ ] Default-vector definition and calibration fingerprint are recorded.
- [ ] Dense/experts shadow approximation modes are recorded.
- [ ] Probe latency and temporary memory are measured.
- [ ] Production KV and RNG are unchanged.
- [ ] Utility aggregation and uncertainty formula are explicit.

## 8. Closed-loop evaluation

- [ ] Natural and executed routes are both recorded.
- [ ] Miss policy is explicit.
- [ ] Weight renormalization semantics are explicit.
- [ ] Early termination reason is logged.
- [ ] Realized window length is reported.
- [ ] First token divergence is reported.
- [ ] Next-token KL/perplexity or task metric is reported.
- [ ] Degeneration/repetition is checked.
- [ ] H2D bytes and exposed stall are reported.
- [ ] Failed generations are not silently dropped.

## 9. Simulator

- [ ] Results are labeled simulated.
- [ ] Hardware profile is saved.
- [ ] Fixed latency and bandwidth are explicit.
- [ ] Compute timing source is explicit.
- [ ] Overlap model is explicit.
- [ ] Cache capacity is never exceeded.
- [ ] Experts are not used before load completion.
- [ ] Transfer concurrency obeys the profile.
- [ ] H2D bytes are exact.
- [ ] Probe cost is included.
- [ ] No-overlap and perfect-overlap bounds are shown.
- [ ] Timeline examples are saved.

## 10. Real runtime

- [ ] Synchronous reference is correct.
- [ ] Asynchronous result matches synchronous reference.
- [ ] Warmup is separate.
- [ ] Model load time is separate.
- [ ] CUDA synchronization/timing method is documented.
- [ ] Probe time is included.
- [ ] Transfer time and exposed stall are distinct.
- [ ] Peak allocated and reserved memory are recorded.
- [ ] KV/workspace/static/dynamic expert memory are itemized.
- [ ] CPU pinned memory is recorded.
- [ ] Per-token latency is retained.
- [ ] Mean, p50, p95, and p99 TPOT are reported.
- [ ] Repetitions and variance are reported.

## 11. Static residency

- [ ] Static ranking calibration set is recorded.
- [ ] Every ranking component is available separately.
- [ ] Static budget fraction is explicit.
- [ ] Static experts are never accidentally evicted.
- [ ] Domain-shift results are reported.
- [ ] Combined-score weights are validation-selected.
- [ ] Router-weight norm is not called expert quality importance.

## 12. Result aggregation

- [ ] Only completed runs are aggregated.
- [ ] Invalid/quarantined runs are listed.
- [ ] Duplicate config/run IDs are detected.
- [ ] Schema versions are compatible.
- [ ] Macro and micro averaging are distinguished.
- [ ] Per-model/task values are retained.
- [ ] Paired confidence intervals are used where applicable.
- [ ] Table cells can be traced to run IDs.
- [ ] Plot data is saved in tabular form.
- [ ] Negative results are included.

## 13. Claims checklist

Before writing a claim:

- [ ] Is it supported by closed-loop results?
- [ ] Is the information regime deployable?
- [ ] Is the result measured or simulated?
- [ ] Is the comparison equal-budget/equal-cost?
- [ ] Does it hold across more than one seed?
- [ ] Is it restricted to evaluated models/tasks?
- [ ] Is there a simpler baseline that performs similarly?
- [ ] Is the claimed concept already present in related work?
- [ ] Are quality changes reported beside speed/memory gains?
- [ ] Are failure conditions stated?

## 14. Artifact release checklist

- [ ] Source code has a license.
- [ ] Model/dataset licenses permit the released artifacts.
- [ ] No private prompt text is included.
- [ ] No access tokens are present.
- [ ] Large artifacts have checksums and manifests.
- [ ] Exact commands are documented.
- [ ] Minimal smoke-test data is included or generated.
- [ ] External downloads are pinned by revision where possible.
- [ ] Paper figures/tables regenerate from public scripts.
- [ ] Known unsupported models and failure modes are documented.

## 15. Final reproducibility command

The project should eventually provide a command similar to:

```bash
pseudoroute reproduce \
  --suite primary \
  --config-root configs/paper \
  --output-dir artifacts/reproduction
```

This command may orchestrate existing scripts rather than reimplementing experiment logic.
