# Model and Dataset Download/Cache Plan

## M1 recorded download

M1 downloaded the pinned 12.65 MB tiny-random Mixtral repository and WikiText-2 data after recording `artifacts/downloads/m1_estimate.json`. The project-local cache uses 33 MB. Identifiers, revisions, paths, and row counts are in `artifacts/downloads/m1_assets.json`. Default CI remains self-contained because external tests are opt-in.

## Pre-download gate for later milestones

Before any external model or dataset download:

1. Pin and record the source identifier, revision, dataset split, and expected file set.
2. Obtain upstream file sizes where available; otherwise perform a metadata-only/dry-run query.
3. Estimate download bytes, expanded cache bytes, conversion/shard overhead, planned trace bytes, and artifact bytes.
4. Check free space at the configured cache and artifact roots. Require the full estimate plus a documented headroom margin; abort before downloading if insufficient.
5. Print and save the estimate and selected paths in the run manifest.

A conservative planning equation is:

```text
required_free = download + expanded_cache + trace_estimate + artifact_estimate + headroom
```

## Cache policy

- Use a configurable project cache root, with Hugging Face cache variables derived from it when those dependencies arrive.
- Reuse content-addressed cached files and never duplicate weights per run.
- Keep model weights and raw datasets outside Git; `.gitignore` covers the project-local cache.
- Store identifiers, revisions, cache location, fingerprints, and licenses in run manifests.
- Never persist credentials or access tokens in configs, logs, or manifests.
- Large external-model tests remain optional and skip cleanly in CI.

## Cleanup and safety

Cache cleanup must target explicit, validated cache subdirectories and should prefer recoverable deletion. Completed run artifacts and manifests are not removed as a side effect. Any material cache removal must be reported.
