# Numerical protocol revision 2 — calibrated before method development/confirmation

The original 20260922T031933Z P1 run failed the full-prefix fixed-route bound: official fused versus retained Python reference NRMSE 0.017962689 (original limit 0.01). It was stopped. No task-quality or official performance rows had run. Original source, protocol, result and report are retained under that run's `attempts/attempt1/`.

The C08 exception in the user-supplied restart specification explicitly permits a newly justified correctness protocol when unchanged vendor/native controls also fail the initial threshold, before held-out or performance-winner confirmation. This revision invokes that exception; it does not claim the original full-prefix gate passed.

On the same real checkpoint and first development prefix, fixed routes, unchanged BF16 matrices and no TF32:

- Transformers 5.17's own native Qwen3MoeExperts versus Python reference full-prefix NRMSE: 0.017653453.
- Official vLLM fused versus native full-prefix NRMSE: 0.021618595.
- Official fused with all 128 experts resident versus the same fused backend streaming m32 prefill groups: 0.016609029. This reference-only diagnostic used a 76,230,693,376-byte CUDA allocation peak; it is not part of the common m32 performance envelope.
- Fused repeat versus fused: bitwise equal, NRMSE 0.
- Fully synchronized copies versus event-ordered copies: bitwise equal, NRMSE 0.
- All 48 layers at identical actual hidden inputs pass the original single-layer bound against FP32 arithmetic on the same BF16 values: worst NRMSE 0.007889564. Native and Python controls have essentially the same worst single-layer error.
- Disabling BF16 reduced-precision reduction did not repair the global discrepancy (0.017766459). Composing the official grouped GEMMs while matching Python activation/weighting order also did not repair it (0.018414742); that variant was not selected. No custom kernel search or dependency change followed.

Installed source inspection shows different BF16 rounding at activation, down-projection weighting and expert summation; streaming prefill adds partial-group output rounding. The empirical native controls demonstrate that the 48-layer cross-implementation 1% bound is below this arithmetic variation. They do not establish an error bound for arbitrary prefixes or a task-quality guarantee.

Revision 2 is frozen before any new policy outputs or confirmation results:

1. Single-layer, synthetic and all 48 real-activation cases: unchanged finite, NRMSE <= 0.01, cosine >= 0.999; zero reference requires exact zero.
2. Full-prefix different BF16 implementations: finite, NRMSE <= 0.03, cosine >= 0.999. The 3% bound is an explicit engineering calibration with margin over the 2.162% native/vendor control; it is not a theorem or an original-protocol success. Three predeclared development prefixes (first three seeded IDs) are checked under this frozen bound, including two not used for calibration.
3. Full-prefix same-backend synchronous versus event-ordered offloading: require bitwise equality in addition to numerical metrics, for each of those three prefixes.
4. Routing, KV, causality, slot lifetime, RNG, task-quality non-inferiority, speed, static-value and pseudo/history gates are unchanged. All required regression/resident/model checks are rerun; no old official model rows are reused (there were none).

Production keeps the official vLLM 0.11.0 fused-experts API and BF16 checkpoint. Runtime VALIDATED, if later obtained, is conditional on this transparently revised numerical protocol. Evidence scripts, full metrics and logs are in the run's `tests/diagnose_numerics*.py`, `tests/diagnose_lifecycle.py`, `numerical_diagnosis2.json` and `numerical_diagnosis3.json`.
