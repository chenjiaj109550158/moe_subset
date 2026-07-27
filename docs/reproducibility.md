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
