# Mass-preserving pseudo residual held-out v1

The corrected static-first previous-route reference retained the development gate: `True`.

| Stage/policy | Route hit | Selected mass | Transfer reduction | Probe s |
|---|---:|---:|---:|---:|
| development/natural_top8_intersection_zero_missing | 0.730438 | 0.747387 | 0.504354 | 0.2209 |
| development/previous_route_commitment | 0.589345 | 0.593025 | 0.323385 | 0.0000 |
| held-out/hard_oracle_commitment | 0.948199 | 0.969558 | 0.709001 | 0.0000 |
| held-out/natural_top8_intersection_zero_missing | 0.699121 | 0.717628 | 0.484724 | 0.1755 |
| held-out/previous_route_commitment | 0.594117 | 0.602651 | 0.344931 | 0.0000 |

Held-out decision: **CANDIDATE_FOR_SEPARATELY_FROZEN_CLOSED_LOOP_PILOT**. Oracle-gap recovery route/mass: 0.2965525745080548/0.3133702038981879.

All route metrics are measured teacher-forced on each policy's own hard-subset state. Probe/replay cost is measured; transfer is simulated. No task accuracy, free generation, exact-token identity, NLL/perplexity, closed-loop runtime, or runtime speedup was measured.
