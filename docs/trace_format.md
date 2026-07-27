# Trace Format

M1 uses sample-sharded safetensors plus an atomic versioned `manifest.json`. Persistent Python pickle is prohibited.

## Required shard tensors

- `token_ids`: `int64 [T]`
- `position_ids`: `int64 [T]`
- `is_prompt`: `bool [T]`
- `router_topk_ids`: `int64 [T, L, K]`
- `router_topk_weights`: `float32 [T, L, K]`
- `router_logits`: `float32 [T, L, E]` for `router_logits` traces only

`routes_only` shards omit full router logits. Expert IDs are interpreted with the layer axis and never as global integers.

## Integrity and resume

Each committed shard has a SHA-256 digest and contiguous global token offsets. Shards are written to a temporary path and atomically renamed before the manifest is committed. Resume skips committed sample IDs and rejects incompatible manifests. Validation checks schema, completion, checksums, required arrays, offsets, token counts, top-k cardinality, expert ranges, and monotonic positions.

## Information regime

M1 trace collection is `offline_teacher_forced`. Its artifacts are never accepted as deployable online state.
